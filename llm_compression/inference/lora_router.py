
"""
AI-OS LoRA Router.

Automatically selects the most relevant LoRA adapters based on user intent.
Uses semantic similarity between user query and LoRA metadata.
"""

import numpy as np
from typing import List, Dict, Callable
import logging

from .lora_format import LoRACard
from .lora_manager import LoRAManager

logger = logging.getLogger(__name__)

class LoRARouter:
    def __init__(self, manager: LoRAManager, embedder_func: Callable[[str], np.ndarray]):
        self.manager = manager
        self.embedder = embedder_func
        self.index: Dict[str, np.ndarray] = {} # buffer for LoRA description embeddings
        
    def register_card(self, card: LoRACard):
        """Register a card and pre-compute its semantic vector."""
        description = card.metadata.get("description", card.name)
        # Enhance description with tags/name
        text = f"{card.name}: {description}"
        vector = self.embedder(text)
        self.index[card.name] = vector
        logger.info(f"Registered LoRA {card.name} to Router index.")

    def register_virtual_candidate(self, name: str, description: str):
        """Register a remote/virtual candidate without loading weights."""
        text = f"{name}: {description}"
        vector = self.embedder(text)
        self.index[name] = vector
        logger.info(f"Registered virtual LoRA {name} to Router index.")

    def select(self, query: str, threshold: float = 0.6, top_k: int = 1) -> List[str]:
        """Select top-k relevant LoRAs for the query."""
        if not self.index:
            return []
            
        query_vec = self.embedder(query)
        
        scores = []
        for name, doc_vec in self.index.items():
            # Cosine similarity
            # Assuming vectors are normalized? If not, normalize.
            # Let's assume input vectors are not necessarily normalized.
            
            norm_q = np.linalg.norm(query_vec)
            norm_d = np.linalg.norm(doc_vec)
            
            if norm_q == 0 or norm_d == 0:
                sim = 0.0
            else:
                sim = np.dot(query_vec, doc_vec) / (norm_q * norm_d)
                
            scores.append((name, sim))
            
        # Sort
        scores.sort(key=lambda x: x[1], reverse=True)
        
        selected = []
        for name, score in scores[:top_k]:
            if score >= threshold:
                selected.append(name)
                logger.debug(f"LoRA Router selected {name} (score={score:.4f})")
            else:
                logger.debug(f"LoRA Router rejected {name} (score={score:.4f} < {threshold})")
                
        return selected
