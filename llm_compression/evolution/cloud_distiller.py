
"""
AI-OS Cloud Distiller.

Enables weak hardware nodes to learn from cloud AI APIs and
internet knowledge sources, distilling responses into local
LoRA skill cards.

The core principle: Cloud APIs are TEACHERS, not permanent
dependencies. We learn from them once, then operate independently.

Supported knowledge sources:
- Cloud APIs: OpenAI, Anthropic, Google (via unified interface)
- Web Search: Extract knowledge from search results
- Static: User-provided QA datasets

Flow:
    1. Generate diverse QA pairs from a topic using cloud API
    2. Encode all QA pairs using local ArrowEngine
    3. Compute attention-weighted embedding deltas
    4. Distill deltas into a LoRA adapter via SVD
    5. Validate the adapter quality
    6. Save as .lora.arrow card
"""

import json
import time
import hashlib
import logging
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

import numpy as np
import torch

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# Cloud Provider Abstraction
# ──────────────────────────────────────────────

class CloudProvider(ABC):
    """Abstract interface for cloud AI providers."""
    
    @abstractmethod
    def query(self, prompt: str, system: str = "") -> str:
        """Send a query and return the response text."""
        ...
    
    @abstractmethod
    def name(self) -> str:
        """Provider identifier."""
        ...


class OpenAIProvider(CloudProvider):
    """OpenAI-compatible API provider (works with OpenAI, local servers, etc.)."""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        
    def query(self, prompt: str, system: str = "") -> str:
        """Query the OpenAI-compatible API."""
        import urllib.request
        import urllib.error
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 2048,
        }).encode("utf-8")
        
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            logger.error(f"Cloud API error: {e.code} {e.reason}")
            raise
        except Exception as e:
            logger.error(f"Cloud query failed: {e}")
            raise
    
    def name(self) -> str:
        return f"openai:{self.model}"


class MockCloudProvider(CloudProvider):
    """Mock provider for testing without actual API access."""
    
    def __init__(self, responses: Optional[Dict[str, str]] = None):
        self.responses = responses or {}
        self.call_log: List[Dict] = []
        
    def query(self, prompt: str, system: str = "") -> str:
        self.call_log.append({"prompt": prompt, "system": system})
        
        # Check for exact match first
        if prompt in self.responses:
            return self.responses[prompt]
        
        # Check for substring match
        for key, val in self.responses.items():
            if key in prompt:
                return val
        
        # Default: echo a structured response
        return f"This is a detailed explanation about: {prompt[:100]}"
    
    def name(self) -> str:
        return "mock"


# ──────────────────────────────────────────────
# QA Pair Generation
# ──────────────────────────────────────────────

@dataclass
class QAPair:
    """A question-answer pair used for distillation."""
    question: str
    answer: str
    topic: str
    source: str  # Which provider generated this
    embedding_q: Optional[np.ndarray] = None
    embedding_a: Optional[np.ndarray] = None


class CloudDistiller:
    """
    Distills cloud AI knowledge into local LoRA skill cards.
    
    This is the core of Tier 2 self-evolution:
    1. Uses cloud API as a "teacher" to generate diverse QA pairs
    2. Encodes QA pairs using the local model
    3. Computes embedding shift patterns (what the teacher knows that we don't)
    4. Synthesizes a LoRA adapter that captures these patterns
    
    The key insight: We don't need to fine-tune the full model.
    We just need to capture the DIRECTION in embedding space that
    the teacher's knowledge points to, and create a LoRA that
    nudges our local model in that direction.
    
    Usage:
        provider = OpenAIProvider(api_key="sk-...")
        distiller = CloudDistiller(engine=arrow_engine)
        
        card = distiller.distill_topic(
            topic="quantum mechanics",
            provider=provider,
            num_pairs=20
        )
        # → quantum_mechanics.lora.arrow saved locally
    """
    
    def __init__(
        self,
        engine=None,
        output_dir: str = "./lora_skills",
        rank: int = 8,
        alpha: float = 16.0,
    ):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.alpha = alpha
        
        # Cache for generated QA pairs
        self.qa_cache: List[QAPair] = []
    
    def generate_qa_pairs(
        self,
        topic: str,
        provider: CloudProvider,
        num_pairs: int = 10,
    ) -> List[QAPair]:
        """
        Generate diverse QA pairs about a topic using the cloud provider.
        
        Uses a structured prompt to get the teacher to produce
        varied questions covering different aspects of the topic.
        
        Args:
            topic: The topic to learn about.
            provider: Cloud AI provider to use as teacher.
            num_pairs: Number of QA pairs to generate.
            
        Returns:
            List of QAPair objects.
        """
        system_prompt = (
            "You are an expert educator. Generate diverse question-answer pairs "
            "for effective learning. Each pair should cover a different aspect. "
            "Response MUST be a valid JSON array of objects with 'q' and 'a' keys. "
            "Keep answers concise but informative (2-4 sentences)."
        )
        
        user_prompt = (
            f"Generate {num_pairs} diverse question-answer pairs about: {topic}\n\n"
            f"Cover: fundamentals, applications, common mistakes, advanced concepts, "
            f"and real-world examples.\n\n"
            f"Format: [{{'q': '...', 'a': '...'}}, ...]"
        )
        
        try:
            response = provider.query(user_prompt, system=system_prompt)
            pairs = self._parse_qa_response(response, topic, provider.name())
            
            logger.info(
                f"Generated {len(pairs)} QA pairs for '{topic}' "
                f"from {provider.name()}"
            )
            
            self.qa_cache.extend(pairs)
            return pairs
            
        except Exception as e:
            logger.error(f"QA generation failed for '{topic}': {e}")
            return []
    
    def distill_topic(
        self,
        topic: str,
        provider: CloudProvider,
        num_pairs: int = 10,
        skill_name: Optional[str] = None,
    ) -> Optional['LoRACard']:
        """
        Full distillation pipeline: topic → QA pairs → LoRA card.
        
        Args:
            topic: Topic to learn.
            provider: Cloud provider for QA generation.
            num_pairs: Number of QA pairs.
            skill_name: Custom name for the skill card.
            
        Returns:
            LoRACard if successful, None otherwise.
        """
        from llm_compression.inference.lora_format import LoRACard, LoRAFormat
        from llm_compression.evolution.lora_extractor import LoRAExtractor
        
        start = time.time()
        
        if not skill_name:
            # Sanitize topic into a valid skill name
            skill_name = topic.lower().replace(" ", "_").replace("-", "_")
            skill_name = "".join(c for c in skill_name if c.isalnum() or c == "_")
            skill_name = f"{skill_name}_v1"
        
        logger.info(f"Starting cloud distillation: '{topic}' → '{skill_name}'")
        
        # 1. Generate QA pairs from the teacher
        pairs = self.generate_qa_pairs(topic, provider, num_pairs)
        if not pairs:
            logger.error("No QA pairs generated. Aborting distillation.")
            return None
        
        # 2. Encode all pairs using local engine
        if not self.engine:
            logger.error("No engine available for encoding.")
            return None
            
        pairs = self._encode_pairs(pairs)
        
        # 3. Compute embedding shift matrix
        # The shift represents "what direction in embedding space
        # does this new knowledge point to?"
        shift_matrix = self._compute_shift_matrix(pairs)
        
        if shift_matrix is None:
            logger.error("Failed to compute shift matrix.")
            return None
        
        # 4. Decompose shift into LoRA via SVD
        extractor = LoRAExtractor(rank=self.rank, alpha=self.alpha, min_explained_variance=0.01)
        extraction = extractor.extract_single(
            torch.from_numpy(shift_matrix),
            layer_name="cloud_distilled"
        )
        
        # 5. Build LoRA card
        card = LoRACard(
            name=skill_name,
            rank=extraction.rank,
            alpha=self.alpha,
            target_modules=["query", "value"],  # Default targets
            weights_A={"cloud_distilled": extraction.weights_A},
            weights_B={"cloud_distilled": extraction.weights_B},
            metadata={
                "description": f"Cloud-distilled knowledge about {topic}",
                "source": provider.name(),
                "num_qa_pairs": str(len(pairs)),
                "extraction_method": "cloud_distillation",
                "explained_variance": str(extraction.explained_variance),
                "topic": topic,
            }
        )
        
        # 6. Save
        output_path = self.output_dir / f"{skill_name}.lora.arrow"
        LoRAFormat.save(card, str(output_path))
        
        duration = time.time() - start
        logger.info(
            f"Cloud distillation complete: '{skill_name}' in {duration:.1f}s. "
            f"QA pairs: {len(pairs)}, Variance retained: {extraction.explained_variance:.3f}, "
            f"Saved to: {output_path}"
        )
        
        return card
    
    def distill_from_existing_qa(
        self,
        qa_pairs: List[Dict[str, str]],
        topic: str,
        skill_name: Optional[str] = None,
    ) -> Optional['LoRACard']:
        """
        Distill from user-provided QA pairs (no cloud API needed).
        
        Args:
            qa_pairs: List of dicts with 'question' and 'answer' keys.
            topic: Topic description.
            skill_name: Custom skill name.
            
        Returns:
            LoRACard if successful.
        """
        from llm_compression.inference.lora_format import LoRACard, LoRAFormat
        from llm_compression.evolution.lora_extractor import LoRAExtractor
        
        start = time.time()
        
        if not skill_name:
            skill_name = topic.lower().replace(" ", "_")
            skill_name = "".join(c for c in skill_name if c.isalnum() or c == "_")
            skill_name = f"{skill_name}_v1"
        
        # Convert to QAPair objects
        pairs = [
            QAPair(
                question=qa.get("question", qa.get("q", "")),
                answer=qa.get("answer", qa.get("a", "")),
                topic=topic,
                source="user_provided",
            )
            for qa in qa_pairs
        ]
        
        if not pairs:
            return None
        
        # Encode
        pairs = self._encode_pairs(pairs)
        
        # Compute shift
        shift_matrix = self._compute_shift_matrix(pairs)
        if shift_matrix is None:
            return None
        
        # Extract LoRA
        extractor = LoRAExtractor(rank=self.rank, alpha=self.alpha, min_explained_variance=0.01)
        extraction = extractor.extract_single(
            torch.from_numpy(shift_matrix),
            layer_name="qa_distilled"
        )
        
        card = LoRACard(
            name=skill_name,
            rank=extraction.rank,
            alpha=self.alpha,
            target_modules=["query", "value"],
            weights_A={"qa_distilled": extraction.weights_A},
            weights_B={"qa_distilled": extraction.weights_B},
            metadata={
                "description": f"Distilled knowledge about {topic}",
                "source": "user_provided",
                "num_qa_pairs": str(len(pairs)),
                "extraction_method": "qa_distillation",
                "explained_variance": str(extraction.explained_variance),
                "topic": topic,
            }
        )
        
        output_path = self.output_dir / f"{skill_name}.lora.arrow"
        LoRAFormat.save(card, str(output_path))
        
        duration = time.time() - start
        logger.info(f"QA distillation complete: '{skill_name}' in {duration:.1f}s")
        
        return card
    
    # ──────────────────────────────────────────
    # Internal Methods
    # ──────────────────────────────────────────
    
    def _encode_pairs(self, pairs: List[QAPair]) -> List[QAPair]:
        """Encode all QA pairs using the local ArrowEngine."""
        questions = [p.question for p in pairs]
        answers = [p.answer for p in pairs]
        
        # Batch encode
        q_embeddings = self.engine.encode(questions, normalize=True)
        a_embeddings = self.engine.encode(answers, normalize=True)
        
        for i, pair in enumerate(pairs):
            pair.embedding_q = q_embeddings[i]
            pair.embedding_a = a_embeddings[i]
        
        return pairs
    
    def _compute_shift_matrix(self, pairs: List[QAPair]) -> Optional[np.ndarray]:
        """
        Compute the embedding shift matrix from QA pairs.
        
        The shift represents the "direction of knowledge":
        For each QA pair, the shift is (answer_embedding - question_embedding).
        This captures "given this question, what direction does the answer point?"
        
        The outer product of shifts creates a matrix that represents
        the aggregate knowledge pattern, which can be decomposed via SVD
        into a LoRA adapter.
        
        Returns:
            Shift matrix of shape (embed_dim, embed_dim).
        """
        valid_pairs = [
            p for p in pairs
            if p.embedding_q is not None and p.embedding_a is not None
        ]
        
        if not valid_pairs:
            return None
        
        embed_dim = len(valid_pairs[0].embedding_q)
        
        # Compute shift vectors: answer - question
        shifts = []
        for pair in valid_pairs:
            shift = pair.embedding_a - pair.embedding_q
            shifts.append(shift)
        
        shifts = np.array(shifts)  # (N, embed_dim)
        
        # Build shift matrix via outer product accumulation:
        # M = sum_i (shift_i ⊗ shift_i^T) / N
        # This is equivalent to: M = shifts^T @ shifts / N
        # Shape: (embed_dim, embed_dim)
        shift_matrix = shifts.T @ shifts / len(shifts)
        
        logger.debug(
            f"Computed shift matrix: {shift_matrix.shape}, "
            f"norm={np.linalg.norm(shift_matrix):.4f}, "
            f"from {len(shifts)} QA pairs"
        )
        
        return shift_matrix.astype(np.float32)
    
    def _parse_qa_response(
        self, 
        response: str, 
        topic: str, 
        source: str
    ) -> List[QAPair]:
        """Parse JSON QA pairs from a cloud API response."""
        # Try to extract JSON from the response
        try:
            # Find JSON array in the response
            start_idx = response.find("[")
            end_idx = response.rfind("]") + 1
            
            if start_idx == -1 or end_idx == 0:
                logger.warning("No JSON array found in response.")
                return []
            
            json_str = response[start_idx:end_idx]
            raw_pairs = json.loads(json_str)
            
            pairs = []
            for item in raw_pairs:
                q = item.get("q", item.get("question", ""))
                a = item.get("a", item.get("answer", ""))
                if q and a:
                    pairs.append(QAPair(
                        question=q,
                        answer=a,
                        topic=topic,
                        source=source,
                    ))
            
            return pairs
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse QA JSON: {e}")
            # Fallback: treat entire response as a single QA pair
            return [QAPair(
                question=f"Explain {topic}",
                answer=response,
                topic=topic,
                source=source,
            )]
