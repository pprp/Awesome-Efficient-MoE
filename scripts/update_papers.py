#!/usr/bin/env python3
"""
ArXiv Paper Update Script for MoE Repository
This script searches for new papers related to Mixture of Experts and updates the README.md
"""

import re
import os
import sys
import json
import time
import logging
import base64
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import requests
import feedparser
from dateutil.parser import parse as parse_date
from PIL import Image
import io

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("Loaded environment variables from .env file")
except ImportError:
    print("python-dotenv not installed, using system environment variables only")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ArXivTracker:
    """Handles tracking of processed arXiv papers to avoid duplicates"""
    
    def __init__(self, tracking_file: str = "scripts/processed_papers.json"):
        self.tracking_file = tracking_file
        self.processed_ids: Set[str] = set()
        self.load_processed_ids()
        
    def load_processed_ids(self):
        """Load previously processed arXiv IDs from JSON file"""
        try:
            if os.path.exists(self.tracking_file):
                with open(self.tracking_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.processed_ids = set(data.get('processed_arxiv_ids', []))
                    logger.info(f"Loaded {len(self.processed_ids)} previously processed arXiv IDs")
            else:
                logger.info("No tracking file found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading tracking file: {e}")
            
    def save_processed_ids(self):
        """Save processed arXiv IDs to JSON file"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.tracking_file), exist_ok=True)
            
            # Prepare data structure
            data = {
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_papers": len(self.processed_ids),
                "processed_arxiv_ids": sorted(list(self.processed_ids))
            }
            
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Updated tracking file with {len(self.processed_ids)} arXiv IDs")
            
        except Exception as e:
            logger.error(f"Error saving tracking file: {e}")
            
    def is_processed(self, arxiv_id: str) -> bool:
        """Check if an arXiv paper has already been processed"""
        return arxiv_id in self.processed_ids
        
    def add_processed(self, arxiv_id: str):
        """Add an arXiv ID to the processed list"""
        self.processed_ids.add(arxiv_id)
        
    def filter_new_papers(self, papers: List[Dict]) -> List[Dict]:
        """Filter out papers that have already been processed"""
        new_papers = []
        
        for paper in papers:
            if not self.is_processed(paper['id']):
                new_papers.append(paper)
            else:
                logger.info(f"Skipping already processed paper: {paper['title']}")
                
        logger.info(f"Filtered {len(papers)} papers -> {len(new_papers)} new papers")
        return new_papers

class ImageExtractor:
    """Handles extracting key images from PDF papers using MinerU API"""
    
    def __init__(self):
        self.api_key = os.getenv('MINERU_API_KEY')
        if not self.api_key:
            logger.warning("MINERU_API_KEY not found, image extraction will be disabled")
        self.base_url = "https://mineru.net"
        
    def extract_key_image(self, paper: Dict) -> Optional[str]:
        """Extract the most important image from a paper and save to assets folder"""
        # TODO: 图片提取功能暂时禁用，等待 MinerU API 集成
        logger.info(f"Image extraction disabled for paper: {paper['title']}")
        logger.info("Reason: MinerU API cannot return images reliably")
        logger.info("To enable image extraction in the future:")
        logger.info("1. Set MINERU_API_KEY environment variable")
        logger.info("2. Implement proper MinerU v4 API workflow")
        logger.info("3. Uncomment the implementation below")
        
        return None
        
        # COMMENTED OUT: Original image extraction implementation
        # TODO: Re-enable when MinerU API integration is ready
        """
        if not self.api_key:
            logger.warning("No MinerU API key, skipping image extraction")
            return None
            
        try:
            # Get PDF URL from paper
            pdf_url = paper.get('pdf_url')
            if not pdf_url:
                logger.warning(f"No PDF URL for paper: {paper['title']}")
                return None
                
            logger.info(f"Extracting images from: {pdf_url}")
            
            # Download PDF content
            pdf_content = self._download_pdf(pdf_url)
            if not pdf_content:
                return None
                
            # Upload PDF to MinerU and get analysis
            images = self._analyze_pdf_with_mineru(pdf_content, paper['title'])
            if not images:
                logger.warning(f"No images extracted from paper: {paper['title']}")
                return None
                
            # Select the most important image (using AI evaluation)
            key_image = self._select_key_image(images, paper['title'])
            if not key_image:
                return None
                
            # Save image to assets folder
            image_filename = self._save_image_to_assets(key_image, paper['id'])
            return image_filename
            
        except Exception as e:
            logger.error(f"Error extracting image for paper {paper['title']}: {e}")
            return None
        """
            
    def _download_pdf(self, pdf_url: str) -> Optional[bytes]:
        """Download PDF content from URL with simple approach"""
        import time
        
        max_retries = 3
        retry_delay = 2
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading PDF (attempt {attempt + 1}/{max_retries}): {pdf_url}")
                
                response = requests.get(pdf_url, timeout=180)
                response.raise_for_status()
                
                logger.info(f"Successfully downloaded {len(response.content)} bytes")
                return response.content
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Download timeout on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"All {max_retries} download attempts failed due to timeout")
                    
            except Exception as e:
                logger.error(f"Download error on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    logger.info(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"All {max_retries} download attempts failed")
                    
        return None
            
    def _analyze_pdf_with_mineru(self, pdf_content: bytes, paper_title: str) -> List[Dict]:
        """Send PDF to MinerU API for analysis and image extraction using v4 API"""
        try:
            logger.warning("MinerU v4 API requires publicly accessible PDF URLs. Skipping image extraction for now.")
            logger.info("To use MinerU v4 API, you need to:")
            logger.info("1. Upload PDF to a public URL first")
            logger.info("2. Use the new task-based API workflow")
            logger.info("3. Poll for results using task_id")
            
            # TODO: Implement v4 API workflow:
            # 1. Create task: POST /api/v4/extract/task with {"url": pdf_url, "is_ocr": true}
            # 2. Get task_id from response
            # 3. Poll: GET /api/v4/extract/task/{task_id} until state="done"
            # 4. Download and extract images from full_zip_url
            
            return []
                
        except Exception as e:
            logger.error(f"Error analyzing PDF with MinerU: {e}")
            return []
            
    def _select_key_image(self, images: List[Dict], paper_title: str) -> Optional[Dict]:
        """Select the most important image from extracted images using GPT-4o-mini"""
        if not images:
            return None
            
        # If only one image, return it
        if len(images) == 1:
            logger.info("Only one image found, selecting it")
            return images[0]
            
        # Use GPT-4o-mini to intelligently score and select the best image
        try:
            best_image = self._ai_select_image(images, paper_title)
            if best_image:
                return best_image
        except Exception as e:
            logger.warning(f"AI image selection failed: {e}, falling back to heuristic")
            
        # Fallback to heuristic method
        return self._heuristic_select_image(images)
        
    def _ai_select_image(self, images: List[Dict], paper_title: str) -> Optional[Dict]:
        """Use GPT-4o-mini to select the best image"""
        api_key = os.getenv('OPENAI_API_KEY')
        base_url = os.getenv('OPENAI_BASE_URL')
        
        if not api_key:
            logger.warning("No OpenAI API key for image selection")
            return None
            
        try:
            import openai
            
            # Initialize client
            if base_url:
                client = openai.OpenAI(api_key=api_key, base_url=base_url)
            else:
                client = openai.OpenAI(api_key=api_key)
            
            # Prepare image descriptions for AI evaluation
            image_descriptions = []
            for i, img in enumerate(images):
                desc = {
                    'index': i,
                    'width': img.get('width', 0),
                    'height': img.get('height', 0),
                    'context': img.get('context', ''),
                    'caption': img.get('caption', ''),
                    'page': img.get('page', 0)
                }
                image_descriptions.append(desc)
            
            prompt = f"""
            You are evaluating images from a research paper titled: "{paper_title}"
            
            Here are the extracted images with their metadata:
            {json.dumps(image_descriptions, indent=2)}
            
            Please evaluate each image and select the ONE most important image that best represents the core contribution of this research paper. Consider:
            
            1. **Research Relevance**: Does the image show the main architecture, algorithm, or key concept?
            2. **Information Density**: Does it contain important technical details, diagrams, or results?
            3. **Visual Quality**: Is it clear, well-sized, and informative?
            4. **Context Clues**: Does the caption/context suggest it's a key figure?
            
            For MoE (Mixture of Experts) papers, prioritize:
            - Architecture diagrams showing expert routing
            - Performance comparison charts
            - Model structure illustrations
            - Algorithm flowcharts
            
            Respond with ONLY the index number (0, 1, 2, etc.) of the best image, followed by a brief explanation.
            Format: "INDEX: X - REASON: brief explanation"
            """
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in computer science research and can identify the most important figures in academic papers."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            result = response.choices[0].message.content.strip()
            logger.info(f"AI image selection result: {result}")
            
            # Parse the response to extract index
            import re
            match = re.search(r'INDEX:\s*(\d+)', result)
            if match:
                selected_index = int(match.group(1))
                if 0 <= selected_index < len(images):
                    logger.info(f"AI selected image {selected_index} from {len(images)} candidates")
                    return images[selected_index]
                    
            logger.warning(f"Invalid AI response format: {result}")
            return None
            
        except Exception as e:
            logger.error(f"Error in AI image selection: {e}")
            return None
            
    def _heuristic_select_image(self, images: List[Dict]) -> Optional[Dict]:
        """Fallback heuristic method to select image"""
        key_image = None
        max_score = 0
        
        for image in images:
            score = 0
            
            # Prefer images with larger dimensions
            width = image.get('width', 0)
            height = image.get('height', 0)
            if width > 300 and height > 200:  # Reasonable figure size
                score += width * height / 10000  # Normalize score
                
            # Prefer images with certain keywords in context
            context = image.get('context', '').lower()
            if any(keyword in context for keyword in ['figure', 'diagram', 'architecture', 'model', 'algorithm']):
                score += 100
                
            # Prefer images that are not too small
            if width > 200 and height > 150:
                score += 50
                
            if score > max_score:
                max_score = score
                key_image = image
                
        # Fallback to first image if no good candidate found
        if not key_image and images:
            key_image = images[0]
            
        logger.info(f"Heuristic selected image with score {max_score}")
        return key_image
        
    def _save_image_to_assets(self, image_data: Dict, paper_id: str) -> Optional[str]:
        """Save extracted image to assets folder"""
        try:
            # Get image data (assuming it's base64 encoded)
            image_base64 = image_data.get('image_base64') or image_data.get('data')
            if not image_base64:
                logger.error("No image data found in image object")
                return None
                
            # Decode base64 image
            image_bytes = base64.b64decode(image_base64)
            
            # Open image with PIL to validate and potentially convert
            image = Image.open(io.BytesIO(image_bytes))
            
            # Ensure assets directory exists
            assets_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'assets')
            os.makedirs(assets_dir, exist_ok=True)
            
            # Find next available image number
            existing_images = [f for f in os.listdir(assets_dir) if f.startswith('image_') and f.endswith('.png')]
            if existing_images:
                # Extract numbers and find the highest
                numbers = []
                for img in existing_images:
                    match = re.search(r'image_(\d+)\.png', img)
                    if match:
                        numbers.append(int(match.group(1)))
                next_num = max(numbers) + 1 if numbers else 1
            else:
                next_num = 1
                
            filename = f"image_{next_num}.png"
            filepath = os.path.join(assets_dir, filename)
            
            # Save as PNG for consistency
            if image.mode in ('RGBA', 'LA'):
                # Convert to RGB for PNG if has alpha channel issues
                image = image.convert('RGB')
            image.save(filepath, 'PNG', optimize=True)
            
            logger.info(f"Saved image as {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error saving image: {e}")
            return None

class ArXivPaperFetcher:
    """Handles fetching papers from arXiv API"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def __init__(self):
        self.search_terms = [
            "mixture of experts",
            "MoE",
            "sparse mixture",
            "expert routing", 
            "gating network",
            "mixture-of-experts",
            "switch transformer",
            "sparse expert",
            "expert specialization"
        ]
        
    def build_query(self, days_back: int = 1) -> str:
        """Build arXiv API query string - focused on MoE and related methods"""
        
        # Core MoE terms that should be in most searches
        moe_terms = [
            'all:"mixture of experts"',
            'all:"MoE"',
            'all:"mixture-of-experts"',
            'all:"sparse mixture"',
            'all:"expert routing"'
        ]
        
        # Related AI/ML terms for context
        context_terms = [
            'all:"transformer"',
            'all:"language model"',
            'all:"neural network"',
            'all:"deep learning"'
        ]
        
        # Create combinations ensuring MoE + context
        query_parts = []
        for moe_term in moe_terms:
            for context_term in context_terms:
                query_parts.append(f'({moe_term} AND {context_term})')
        
        # Add specific MoE method combinations
        specific_combinations = [
            '(all:"switch transformer")',
            '(all:"expert specialization" AND all:"neural")',
            '(all:"gating network" AND all:"mixture")',
            '(all:"sparse expert" AND all:"transformer")',
            '(all:"expert routing" AND all:"language model")',
            '(all:"mixture of experts" AND all:"compression")',
            '(all:"MoE" AND all:"efficiency")',
            '(all:"mixture-of-experts" AND all:"scaling")'
        ]
        
        query_parts.extend(specific_combinations)
        
        # Join with OR to cast a wider net, but each part ensures relevance
        query = ' OR '.join(query_parts)
        
        logger.info(f"Built MoE-focused query with {len(query_parts)} combinations")
        logger.info(f"Query: {query[:400]}...")  # Log first 400 chars for debugging
        return query
        
    def is_relevant_paper(self, paper: Dict) -> bool:
        """Check if paper is actually relevant to MoE research"""
        title_lower = paper['title'].lower()
        abstract_lower = paper['summary'].lower()
        
        # Must have MoE-related terms (REQUIRED - this is the key filter)
        moe_keywords = [
            'mixture of experts', 'mixture-of-experts', 'moe', 'expert routing',
            'gating network', 'switch transformer', 'sparse mixture', 'expert specialization',
            'sparse expert', 'expert selection', 'multi-expert', 'expert fusion',
            'conditional computation', 'expert networks', 'routing mechanism'
        ]
        
        # Must have AI/ML context terms 
        context_keywords = [
            'transformer', 'language model', 'neural network', 'deep learning',
            'machine learning', 'artificial intelligence', 'nlp', 'neural',
            'bert', 'gpt', 'llama', 't5', 'model', 'network', 'architecture'
        ]
        
        # MoE requirement is STRICT - must have MoE terms
        has_moe = any(keyword in title_lower or keyword in abstract_lower for keyword in moe_keywords)
        has_context = any(keyword in title_lower or keyword in abstract_lower for keyword in context_keywords)
        
        # Exclude papers that are clearly not about computational MoE
        exclude_keywords = [
            'mechanical engineering', 'civil engineering', 'hardware design', 'circuit design',
            'materials science', 'chemistry', 'biology', 'medical diagnosis', 'drug discovery',
            'physics simulation', 'weather prediction', 'climate model', 'fluid dynamics'
        ]
        
        has_exclude = any(keyword in title_lower or keyword in abstract_lower for keyword in exclude_keywords)
        
        # BOTH MoE and context terms are required, no exclusions allowed
        is_relevant = has_moe and has_context and not has_exclude
        
        if not is_relevant:
            if not has_moe:
                logger.info(f"Filtered out - NO MoE terms: {paper['title']}")
            elif not has_context:
                logger.info(f"Filtered out - NO context terms: {paper['title']}")
            else:
                logger.info(f"Filtered out - excluded domain: {paper['title']}")
            
        return is_relevant
        
    def fetch_recent_papers(self, days_back: int = 1) -> List[Dict]:
        query = self.build_query(days_back)
        
        params = {
            'search_query': query,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending',
            'max_results': 200  # Get many results to find recent ones
        }
        
        try:
            logger.info(f"Making arXiv API request with {days_back} days lookback...")
            response = requests.get(self.BASE_URL, params=params, timeout=30)
            response.raise_for_status()
            
            logger.info(f"ArXiv API response status: {response.status_code}")
            logger.info(f"Response content length: {len(response.content)} bytes")
            
            # Parse the Atom feed
            feed = feedparser.parse(response.content)
            
            logger.info(f"Feed parsed. Total entries found: {len(feed.entries)}")
            
            if len(feed.entries) == 0:
                logger.warning("No entries found in arXiv response - may indicate query issues")
                return []
            
            papers = []
            cutoff_date = datetime.now() - timedelta(days=days_back)
            logger.info(f"Looking for papers newer than: {cutoff_date}")
            
            date_filtered_count = 0
            relevance_filtered_count = 0
            
            for i, entry in enumerate(feed.entries):
                if i < 3:  # Log first few entries for debugging
                    logger.info(f"Entry {i+1}: {entry.title[:100]}...")
                    logger.info(f"  Published: {entry.published}")
                
                # Parse submission date
                published_date = parse_date(entry.published)
                logger.info(f"  Parsed date: {published_date}")
                
                # Check date filter
                if published_date.replace(tzinfo=None) > cutoff_date:
                    date_filtered_count += 1
                    
                    paper = {
                        'id': entry.id.split('/')[-1],  # Extract arXiv ID
                        'title': entry.title.replace('\n', ' ').strip(),
                        'authors': [author.name for author in entry.authors],
                        'summary': entry.summary.replace('\n', ' ').strip(),
                        'published': published_date,
                        'link': entry.id,
                        'pdf_url': entry.id.replace('/abs/', '/pdf/') + '.pdf'
                    }
                    
                    # Apply relevance filter
                    if self.is_relevant_paper(paper):
                        relevance_filtered_count += 1
                        papers.append(paper)
                        logger.info(f"Added relevant paper: {paper['title'][:100]}...")
                    
            logger.info(f"Date filtering: {date_filtered_count} papers within {days_back} days")
            logger.info(f"Relevance filtering: {relevance_filtered_count} relevant papers")
            logger.info(f"Final result: {len(papers)} papers to process")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

class PaperSummarizer:
    """Handles paper summarization using LLM APIs"""
    
    def __init__(self):
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.openai_base_url = os.getenv('OPENAI_BASE_URL')
        self.anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
        
    def summarize_paper(self, paper: Dict) -> tuple[str, str]:
        """Summarize a paper using available LLM APIs, returns (english_summary, chinese_summary)"""
        
        english_prompt = f"""
        Please provide a concise summary of this research paper about Mixture of Experts (MoE) in 2-3 sentences. 
        Focus on the key contributions, methods, and results. End with relevant hashtags.
        
        Title: {paper['title']}
        Authors: {', '.join(paper['authors'])}
        Abstract: {paper['summary'][:1500]}...
        
        Format your response as a single paragraph summary followed by hashtags like #MixtureOfExperts #MoE #Efficiency
        """
        
        chinese_prompt = f"""
        请用中文为这篇关于混合专家模型 (MoE) 的研究论文提供2-3句话的简洁摘要。
        重点关注关键贡献、方法和结果。以相关标签结尾。
        
        标题: {paper['title']}
        作者: {', '.join(paper['authors'])}
        摘要: {paper['summary'][:1500]}...
        
        请用中文回答，格式为一个段落的摘要，然后是中文标签如 #混合专家 #MoE #效率
        """
        
        # Get English summary
        english_summary = self._get_summary_from_apis(english_prompt)
        if not english_summary:
            english_summary = self._generate_basic_summary(paper, "en")
            
        # Get Chinese summary
        chinese_summary = self._get_summary_from_apis(chinese_prompt)
        if not chinese_summary:
            chinese_summary = self._generate_basic_summary(paper, "zh")
            
        return english_summary, chinese_summary
        
    def _get_summary_from_apis(self, prompt: str) -> Optional[str]:
        """Get summary from available APIs"""
        
        # Try OpenAI first
        if self.openai_api_key:
            try:
                summary = self._summarize_with_openai(prompt)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"OpenAI API failed: {e}")
                
        # Try Anthropic as fallback
        if self.anthropic_api_key:
            try:
                summary = self._summarize_with_anthropic(prompt)
                if summary:
                    return summary
            except Exception as e:
                logger.warning(f"Anthropic API failed: {e}")
                
        return None
        
    def _summarize_with_openai(self, prompt: str) -> Optional[str]:
        """Summarize using OpenAI API"""
        try:
            import openai
            
            # Initialize client with custom base URL if provided
            if self.openai_base_url:
                client = openai.OpenAI(
                    api_key=self.openai_api_key,
                    base_url=self.openai_base_url
                )
            else:
                client = openai.OpenAI(api_key=self.openai_api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert in machine learning and Mixture of Experts (MoE) research."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.4
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            return None
            
    def _summarize_with_anthropic(self, prompt: str) -> Optional[str]:
        """Summarize using Anthropic API"""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.anthropic_api_key)
            
            response = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=200,
                temperature=0.3,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text.strip()
            
        except Exception as e:
            logger.error(f"Anthropic API error: {e}")
            return None
            
    def _generate_basic_summary(self, paper: Dict, language: str) -> str:
        """Generate a basic summary when APIs are unavailable"""
        if language == "zh":
            return f"这篇论文研究了{paper['title'].lower()}。{paper['summary'][:150]}... <br/>#混合专家 #MoE"
        else:
            return f"This paper presents research on {paper['title'].lower()}. {paper['summary'][:150]}... <br/>#MixtureOfExperts #MoE"

class ReadmeUpdater:
    """Handles updating the README.md file"""
    
    def __init__(self, readme_path: str = "README.md"):
        self.readme_path = readme_path
        
    def load_readme(self) -> str:
        """Load the current README.md content"""
        try:
            with open(self.readme_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading README: {e}")
            return ""
            
    def save_readme(self, content: str) -> bool:
        """Save the updated README.md content"""
        try:
            with open(self.readme_path, 'w', encoding='utf-8') as f:
                f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error saving README: {e}")
            return False
            
    def format_paper_entry(self, paper: Dict, english_summary: str, chinese_summary: str, image_filename: Optional[str] = None) -> str:
        """Format a paper entry for the README list format"""
        # Format authors - limit to first 3 for space
        authors = paper['authors'][:3]
        if len(paper['authors']) > 3:
            authors_str = ', '.join(authors) + ', et al.'
        else:
            authors_str = ', '.join(authors)
            
        # Format date
        date_str = paper['published'].strftime('%Y-%m-%d')
        
        # Create the list entry in the new format with optional image
        entry_parts = [f"- {paper['title']}"]
        
        # Add image if available
        if image_filename:
            entry_parts.extend([
                "",  # Empty line for spacing
                "  <div align=\"center\">",
                f"    <img src=\"./assets/{image_filename}\" width=\"80%\">",
                "  </div>",
                ""   # Empty line after image
            ])
        
        # Add paper details
        entry_parts.extend([
            f"  - Authors: {authors_str}",
            f"  - Link: {paper['link'].replace('/abs/', '/pdf/')}",
            f"  - Code: Not available",
            f"  - Summary: {english_summary}",
            f"  - 摘要: {chinese_summary}"
        ])
        
        return '\n'.join(entry_parts)
        
    def update_papers_list(self, content: str, new_papers: List[Dict], summaries: List[tuple], image_filenames: List[Optional[str]]) -> str:
        """Update the papers list with new entries, adding them to the end"""
        if not new_papers:
            logger.info("No new papers to add")
            return content
            
        # Find the last paper entry in the README to add after it
        # Look for the last occurrence of a paper entry pattern
        lines = content.split('\n')
        last_paper_end = -1
        
        # Find the last line that belongs to a paper entry (摘要: line)
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith('- 摘要:') or lines[i].strip().startswith('    - 摘要:'):
                last_paper_end = i
                break
        
        if last_paper_end == -1:
            # If no papers found, try to find after taxonomy table
            taxonomy_end = content.find('| <img src=https://img.shields.io/badge/benchmark-purple.svg > |')
            if taxonomy_end == -1:
                logger.error("Could not find insertion point in README")
                return content
            insert_pos = content.find('\n', taxonomy_end)
        else:
            # Insert after the last paper
            insert_pos = content.find('\n', sum(len(line) + 1 for line in lines[:last_paper_end + 1]) - 1)
            
        logger.info(f"Adding {len(new_papers)} new papers to end of README")
            
        # Create new entries
        new_entries = []
        for i, (paper, (english_summary, chinese_summary)) in enumerate(zip(new_papers, summaries)):
            image_filename = image_filenames[i] if i < len(image_filenames) else None
            entry = self.format_paper_entry(paper, english_summary, chinese_summary, image_filename)
            new_entries.append(entry)
            
        # Add spacing and new entries at the end
        new_content = (
            content[:insert_pos + 1] + 
            '\n\n' + 
            '\n\n'.join(new_entries) +
            content[insert_pos + 1:]
        )
        
        return new_content

def main():
    """Main execution function"""
    logger.info("Starting daily arXiv paper update")
    
    # Initialize components
    tracker = ArXivTracker()
    fetcher = ArXivPaperFetcher()
    summarizer = PaperSummarizer()
    updater = ReadmeUpdater()
    image_extractor = ImageExtractor()
    
    # Test mode - just fetch and show papers without API requirement
    test_mode = len(sys.argv) > 1 and sys.argv[1] == "--test"
    
    if test_mode:
        logger.info("Running in test mode - will fetch papers but not update README")
    else:
        # Check if we have API keys
        if not (summarizer.openai_api_key or summarizer.anthropic_api_key):
            logger.error("No LLM API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY")
            logger.info("You can run in test mode with: python update_papers.py --test")
            sys.exit(1)
        
    # Fetch recent papers
    logger.info("Fetching recent papers from arXiv...")
    papers = fetcher.fetch_recent_papers(days_back=30)  # Increase to 30 days for better results
    
    if not papers:
        logger.info("No new papers found from arXiv - exiting without changes")
        sys.exit(0)
        
    # Filter out papers we've already processed
    new_papers = tracker.filter_new_papers(papers)
    
    if not new_papers:
        logger.info("No new unprocessed papers found - exiting without changes")
        sys.exit(0)
        
    logger.info(f"Found {len(new_papers)} new papers")
    
    # SINGLE PAPER PER RUN: Process only one paper to ensure one PR per paper
    papers_to_process = new_papers[:1]  # Only take the first paper
    
    if len(new_papers) > 1:
        logger.info(f"Found {len(new_papers)} papers, but processing only 1 paper per run")
        logger.info(f"Remaining {len(new_papers) - 1} papers will be processed in subsequent runs")
    
    if test_mode:
        logger.info("Test mode - showing found papers:")
        for i, paper in enumerate(papers_to_process):  # Show limited papers
            logger.info(f"Paper {i+1}: {paper['title']}")
            logger.info(f"  Authors: {', '.join(paper['authors'][:3])}")
            logger.info(f"  Date: {paper['published']}")
            logger.info(f"  Abstract: {paper['summary'][:200]}...")
            logger.info("---")
        logger.info(f"Found {len(new_papers)} total papers, showing first {len(papers_to_process)}. Exiting test mode.")
        sys.exit(0)
    
    # Process single paper to ensure one PR per paper
    total_added = 0
    
    # Process the single paper
    paper = papers_to_process[0]  # Only one paper
    logger.info(f"Processing paper: {paper['title']}")
    
    # Extract image first (currently disabled)
    logger.info(f"Attempting image extraction for paper: {paper['title']}")
    image_filename = image_extractor.extract_key_image(paper)
    
    if image_filename:
        logger.info(f"Successfully extracted image: {image_filename}")
    else:
        logger.info(f"No image extracted for paper: {paper['title']} (image extraction disabled)")
    
    # Then summarize the paper
    logger.info(f"Summarizing paper: {paper['title']}")
    english_summary, chinese_summary = summarizer.summarize_paper(paper)
    
    # Prepare data for README update
    summaries = [(english_summary, chinese_summary)]
    image_filenames = [image_filename]
    
    # Load current README once
    readme_content = updater.load_readme()
    if not readme_content:
        logger.error("Failed to load README.md")
        sys.exit(1)
        
    # Update with the single paper
    updated_content = updater.update_papers_list(readme_content, papers_to_process, summaries, image_filenames)
    
    # Check if content actually changed
    if len(updated_content) != len(readme_content):
        # Save the updated README
        if updater.save_readme(updated_content):
            logger.info(f"Successfully added paper: {paper['title']}")
            total_added = 1
            
            # Mark paper as processed
            tracker.add_processed(paper['id'])
            
            # Create descriptive commit message for single paper
            paper_title_short = paper['title'][:50] + "..." if len(paper['title']) > 50 else paper['title']
            commit_message = f'Add MoE paper: {paper_title_short}'
            
            # Create commit with both README and any new images
            os.system(f'git add README.md assets/')
            os.system(f'git commit -m "{commit_message}"')
            
        else:
            logger.error("Failed to save README.md")
    else:
        logger.info("No changes to README, no papers added")
    
    # Save the updated tracking file
    if total_added > 0:
        tracker.save_processed_ids()
        logger.info(f"Successfully processed 1 paper: {paper['title']}")
        
        if len(new_papers) > 1:
            logger.info(f"Note: {len(new_papers) - 1} papers remain for next runs")
            logger.info("These will be automatically processed when the script runs again")
        
        logger.info("Git commit created - ready for PR")
    else:
        logger.info("No new papers were added")
        
        # Even if no papers were added, update the tracking file to mark attempted papers as processed
        # This prevents infinite retry of papers that fail to be added
        tracker.save_processed_ids()
        
    logger.info("Paper update completed successfully")

if __name__ == "__main__":
    main()