"""Mixture of Experts routing mechanisms for PyTorch."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Optional, Tuple, Union, NamedTuple
from dataclasses import dataclass


@dataclass
class RouterIndices:
    """Dispatch indices and combine weights for scatter/gather-based routing."""

    dispatch_indices: (
        torch.Tensor
    )  # [num_groups, tokens_per_group, num_selected_experts, 2]
    combine_weights: (
        torch.Tensor
    )  # [num_groups, tokens_per_group, num_selected_experts]
    auxiliary_loss: float
    router_z_loss: float = 0.0


@dataclass
class RouterMask:
    """Dispatch and combine arrays for expert routing with masked matmuls."""

    dispatch_mask: (
        torch.Tensor
    )  # [num_groups, tokens_per_group, num_experts, expert_capacity]
    combine_array: (
        torch.Tensor
    )  # [num_groups, tokens_per_group, num_experts, expert_capacity]
    auxiliary_loss: float
    router_z_loss: float = 0.0


class RouterWeights(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        use_bias: bool = True,
        dtype: torch.dtype = torch.float32,
        kernel_init: str = "normal",
        kernel_init_scale: float = 2e-2,
    ):
        super().__init__()
        self.use_bias = use_bias
        self.dtype = dtype
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts

        # Initialize router weights with correct shape
        self.weight = nn.Parameter(torch.empty(num_experts, hidden_dim))
        if kernel_init == "normal":
            nn.init.normal_(self.weight, std=kernel_init_scale)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(num_experts))
        else:
            self.register_parameter("bias", None)

    def forward(self, token_inputs: torch.Tensor) -> torch.Tensor:
        """Applies router weights to compute logits.

        Args:
            token_inputs: [num_groups, tokens_per_group, hidden_dim]

        Returns:
            Router logits: [num_groups, tokens_per_group, num_experts]
        """
        # Get input shape
        num_groups, tokens_per_group, hidden_dim = token_inputs.shape

        # Reshape input to 2D for linear layer
        flat_inputs = token_inputs.view(
            -1, hidden_dim
        )  # [num_groups * tokens_per_group, hidden_dim]

        # Apply linear transformation
        router_logits = F.linear(
            flat_inputs, self.weight, self.bias
        )  # [num_groups * tokens_per_group, num_experts]

        # Reshape back to 3D
        router_logits = router_logits.view(
            num_groups, tokens_per_group, self.num_experts
        )

        return router_logits


class TokensChooseScatterRouter(nn.Module):
    """Scatter router using tokens choose top-k experts assignment."""

    def __init__(
        self,
        hidden_dim: int,
        num_experts: int,
        num_selected_experts: int = 1,
        jitter_noise: float = 0.0,
        batch_prioritized_routing: bool = True,
        dtype: torch.dtype = torch.float32,
        ignore_padding_tokens: bool = False,
    ):
        super().__init__()
        self.num_selected_experts = num_selected_experts
        self.jitter_noise = jitter_noise
        self.batch_prioritized_routing = batch_prioritized_routing
        self.dtype = dtype
        self.ignore_padding_tokens = ignore_padding_tokens

        self.router_weights = RouterWeights(
            hidden_dim=hidden_dim, num_experts=num_experts, dtype=dtype
        )

    def _compute_router_probabilities(
        self, token_inputs: torch.Tensor, apply_jitter: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes router probabilities from input tokens."""

        if apply_jitter and self.jitter_noise > 0:
            noise = torch.empty_like(token_inputs).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
            token_inputs = token_inputs * noise

        # Compute router logits and probabilities
        router_logits = self.router_weights(token_inputs)
        router_probs = F.softmax(router_logits, dim=-1)

        return router_probs, router_logits

    def _compute_load_balancing_loss(
        self, router_probs: torch.Tensor, expert_indices: torch.Tensor
    ) -> float:
        """Computes load balancing loss to encourage uniform expert utilization.

        Args:
            router_probs: [num_groups, tokens_per_group, num_experts]
            expert_indices: [num_groups, tokens_per_group, num_selected_experts]
        """
        # Calculate expert usage
        num_experts = router_probs.size(-1)
        expert_mask = F.one_hot(expert_indices, num_experts).float()
        expert_mask = expert_mask.sum(dim=2)  # Sum over selected experts
        tokens_per_expert = expert_mask.sum(dim=[0, 1])

        # Compute load balancing loss
        total_tokens = tokens_per_expert.sum()
        target_tokens_per_expert = total_tokens / num_experts
        load_balancing_loss = torch.mean(
            (tokens_per_expert - target_tokens_per_expert) ** 2
        )
        return float(load_balancing_loss)

    def _compute_router_z_loss(self, router_logits: torch.Tensor) -> float:
        """Computes router z-loss to encourage small logit values.

        Args:
            router_logits: [num_groups, tokens_per_group, num_experts]
        """
        return float(torch.mean(router_logits**2))

    def forward(
        self,
        token_inputs: torch.Tensor,
        expert_capacity: int,
        apply_jitter: bool = True,
    ) -> RouterIndices:
        """Main routing computation.

        Args:
            token_inputs: Input tokens of shape [num_groups, tokens_per_group, hidden_dim]
            expert_capacity: Maximum number of tokens per expert
            apply_jitter: Whether to apply jitter noise to inputs

        Returns:
            RouterIndices containing dispatch indices and combine weights
        """
        # Get router probabilities
        router_probs, router_logits = self._compute_router_probabilities(
            token_inputs, apply_jitter
        )

        num_groups, tokens_per_group, num_experts = router_probs.shape

        if self.ignore_padding_tokens:
            # Mask padding tokens
            padding_mask = (torch.abs(token_inputs).sum(dim=-1) > 0).float()
            router_probs = router_probs * padding_mask.unsqueeze(-1)

        # Get top-k experts per token
        combine_weights, expert_indices = torch.topk(
            router_probs, k=self.num_selected_experts, dim=-1
        )

        # Compute auxiliary losses
        auxiliary_loss = self._compute_load_balancing_loss(router_probs, expert_indices)
        router_z_loss = self._compute_router_z_loss(router_logits)

        # Create dispatch indices tensor
        batch_size = num_groups * tokens_per_group
        batch_indices = torch.arange(batch_size, device=token_inputs.device)
        batch_indices = batch_indices.repeat_interleave(self.num_selected_experts)

        # Reshape expert indices for scatter
        expert_indices_flat = expert_indices.reshape(-1)

        # Create dispatch indices
        dispatch_indices = torch.stack([batch_indices, expert_indices_flat], dim=1)
        dispatch_indices = dispatch_indices.reshape(
            num_groups, tokens_per_group, self.num_selected_experts, 2
        )

        return RouterIndices(
            dispatch_indices=dispatch_indices,
            combine_weights=combine_weights,
            auxiliary_loss=auxiliary_loss,
            router_z_loss=router_z_loss,
        )


if __name__ == "__main__":
    # Test configuration
    hidden_dim = 64
    num_experts = 4
    num_selected_experts = 2
    num_groups = 2
    tokens_per_group = 8
    expert_capacity = 4

    # Create router
    router = TokensChooseScatterRouter(
        hidden_dim=hidden_dim,
        num_experts=num_experts,
        num_selected_experts=num_selected_experts,
        jitter_noise=0.1,
    )

    # Create sample input - shape: [num_groups, tokens_per_group, hidden_dim]
    token_inputs = torch.randn(num_groups, tokens_per_group, hidden_dim)

    # Get router outputs
    router_outputs = router(token_inputs, expert_capacity)

    print("Router outputs:")
    print(f"Dispatch indices shape: {router_outputs.dispatch_indices.shape}")
    print(f"Combine weights shape: {router_outputs.combine_weights.shape}")
    print(f"Auxiliary loss: {router_outputs.auxiliary_loss:.4f}")
    print(f"Router z-loss: {router_outputs.router_z_loss:.4f}")

    # Verify shapes
    expected_dispatch_shape = (num_groups, tokens_per_group, num_selected_experts, 2)
    expected_combine_shape = (num_groups, tokens_per_group, num_selected_experts)

    assert router_outputs.dispatch_indices.shape == expected_dispatch_shape
    assert router_outputs.combine_weights.shape == expected_combine_shape
    print("\nAll shape checks passed!")
