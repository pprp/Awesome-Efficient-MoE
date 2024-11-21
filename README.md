# Awesome-Efficient-MoE

<div align='center'>
  <img src=https://cdn.rawgit.com/sindresorhus/awesome/d7305f38d29fed78fa85652e3a63e154dd8e8829/media/badge.svg >
  <img src=https://img.shields.io/github/stars/pprp/Awesome-Efficient-MoE.svg?style=social >
  <img src=https://img.shields.io/github/watchers/pprp/Awesome-Efficient-MoE.svg?style=social >
  <img src=https://img.shields.io/badge/Release-v0.1-brightgreen.svg >
  <img src=https://img.shields.io/badge/License-GPLv3.0-turquoise.svg >
 </div>

This repository collects research papers and resources about Mixture-of-Experts (MoE) models and their efficient variants. MoE is a machine learning technique that divides a complex task among multiple "expert" neural networks, each specializing in handling different aspects of the input space, coordinated by a gating network that decides which expert(s) to use for each input.

MoE models have gained significant attention in recent years due to their:

- Scalability: Ability to scale model capacity without proportionally increasing computation
- Efficiency: Selective activation of only relevant experts for each input
- Specialization: Different experts can learn to handle different types of inputs
- Adaptability: Dynamic routing of inputs to the most appropriate experts

This collection focuses particularly on methods to make MoE models more efficient through various techniques like pruning, quantization, decomposition and acceleration, making them more practical for real-world applications.

## Table of Contents

- [Sparse Mixture-of-Experts](#sparse-moe)
- [MoE Compression](#moe-compression)
  - [MoE Pruning](#moe-pruning)
  - [MoE Quantization](#moe-quantization)
  - [MoE Decomposition](#moe-decomposition)
  - [MoE Acceleration](#moe-acceleration)
- [MoE Survey](#moe-survey)
- [MoE Resources](#moe-resources)
- [MoE FAQ](#moe-faq)
- [Contributing](#contributing)

### Sparse Mixture-of-Experts

- Adaptive Mixtures of Local Experts

  - ![alt text](./assets/image_1.png)
  - URL: https://watermark.silverchair.com/neco.1991.3.1.79.pdf
  - Author: Robert A. Jacobs, Michael I. Jordan, Stevven J. Nowlan, Geoffrey E. Hinton
  - Pub: Neural Computation 1991
  - Summary: This paper introduces a supervised learning method for modular networks composed of multiple expert networks. Each network specializes in a subset of the task, controlled by a gating network. It bridges modular multilayer networks and competitive learning models. The methodology ensures task-specific specialization, reducing interference and improving generalization. A vowel recognition task demonstrates the system's efficacy, showing faster learning and robust performance compared to traditional backpropagation networks.
  - 摘要: 本文提出了一种用于模块化网络的新型监督学习方法，该网络由多个专家网络组成，每个网络专注于任务的一部分，由一个门控网络进行控制。这种方法将模块化多层网络与竞争学习模型相结合，通过减少干扰和提高泛化能力实现任务特定的专业化，与传统的反向传播网络相比，该系统学习更快，性能更加稳健。

### MoE Compression

#### MoE Pruning

- MoE-I2: Compressing Mixture of Experts Models through Inter-Expert Pruning and Intra-Expert Low-Rank Decomposition
  ![alt text](./assets/MoEI2.png)
  - Authors: Cheng Yang, Yang Sui, Jinqi Xiao, Lingyi Huang, Yu Gong, Yuanlin Duan, Wenqi Jia, Miao Yin, Yu Cheng, Bo Yuan
  - Link: https://arxiv.org/abs/2411.01016
  - Code: https://github.com/xiaochengsky/MoEI-2
  - Summary: The emergence of Mixture of Experts (MoE) LLMs has significantly advanced the development of language models. Compared to traditional LLMs, MoE LLMs outperform traditional LLMs by achieving higher performance with considerably fewer activated parameters. Despite this efficiency, their enormous parameter size still leads to high deployment costs. In this paper, we introduce a two-stage compression method tailored for MoE to reduce the model size and decrease the computational cost. First, in the inter-expert pruning stage, we analyze the importance of each layer and propose the Layer-wise Genetic Search and Block-wise KT-Reception Field with the non-uniform pruning ratio to prune the individual expert. Second, in the intra-expert decomposition stage, we apply the low-rank decomposition to further compress the parameters within the remaining experts. Extensive experiments on Qwen1.5-MoE-A2.7B, DeepSeek-V2-Lite, and Mixtral-8×7B demonstrate that our proposed methods can both reduce the model size and enhance inference efficiency while maintaining performance in various zero-shot tasks.

#### MoE Quantization

- QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models
  ![alt text](./assets/image_2.png)
  - Authors: Elias Frantar, Dan Alistarh
  - Link: https://arxiv.org/pdf/2310.16795
  - Code: github.com/IST-DASLab/qmoe
  - Summary: This paper introduces QMoE, a framework for compressing and efficiently inferencing massive Mixture-of-Experts (MoE) models to less than 1 bit per parameter. QMoE addresses the memory challenges of trillion-parameter models like SwitchTransformer-c2048, achieving 10-20x compression (e.g., compressing the 1.6 trillion parameter model to under 160GB) with minimal accuracy loss and runtime overhead (under 5%). This is accomplished through a scalable compression algorithm, a custom compression format, and bespoke GPU decoding kernels for fast inference. The framework enables running trillion-parameter models on affordable commodity hardware.
  - 摘要：本文介绍了 QMoE，这是一个用于压缩和高效推理大型混合专家（MoE）模型的框架，其压缩率低于每参数 1 比特。QMoE 解决了像 SwitchTransformer-c2048 这样万亿参数模型的内存挑战，实现了 10-20 倍的压缩（例如，将 1.6 万亿参数模型压缩到 160GB 以下），同时精度损失和运行时间开销最小（低于 5%）。这是通过可扩展的压缩算法、自定义压缩格式和用于快速推理的定制 GPU 解码内核来实现的。该框架能够在价格合理的商品硬件上运行万亿参数模型。

#### MoE Decomposition

#### MoE Acceleration

### MoE Survey

### MoE Resources

- [Mixture of Experts (MoE) Explained](https://huggingface.co/blog/moe)

### MoE FAQ

1. Q: What is a Mixture of Experts (MoE) model?
   A: A MoE model is a machine learning architecture that combines multiple "expert" neural networks, each specializing in different aspects of a task, with a gating network that routes inputs to the most appropriate experts.

2. Q: What are the main advantages of MoE models?
   A: Key advantages include:

- **Pretrained faster than dense model**
- Have **faster inference** compared to a model with the same number of parameters
- Increased model **capacity** without proportional compute costs
- Better task **specialization** through expert specialization
- **Faster training** and **better inference** under high-load scenarios

3. Q: What are the challenges in implementing MoE models?
   A: Common challenges include:

   - Require high VRAM as all experts are loaded in memory
   - Complex routing mechanisms and load balancing
   - Training instability
   - Communication overhead in distributed settings

4. Q: How do MoE models compare to dense transformers?
   A: MoE models can achieve better performance with similar or lower computational costs, but require more memory and have more complex training dynamics.

5. Q: Why MoE can be pretrained faster than dense model?
   A: From my experience in NAS, MoE can be treated as a way to decouple the weights. The more weights shared, the faster the model can be pretrained. Additionally, decoupling the weights can increase the model capacity.

6. Q: How to choose the number of experts?
   A: TBD

7. Q: Why MoE is suitable for high-load scenarios? How does it work when inference?
   A: TBD

### Contributing

We welcome contributions to this repository! If you have any resources, papers, or insights related to Mixture-of-Experts (MoE) models and their efficient variants, please consider contributing to this repository.

To contribute, please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and ensure they are well-documented.
4. Submit a pull request with your changes.
