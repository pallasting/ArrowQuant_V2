
"""
AI-OS Skill Distiller.

The central orchestrator for AI-OS's self-evolution capability.
Coordinates the three-tier knowledge acquisition strategy:

- Tier 1: Swarm Query (Federation) — already implemented in Phase 8
- Tier 2: Cloud/Web Distillation — external knowledge acquisition
- Tier 3: Weight Map Extraction — surgical LoRA extraction from large models

When the LoRARouter reports low confidence (cognitive dissonance),
the SkillDistiller activates the appropriate tier based on available
resources and acquires the missing capability.
"""

import time
import logging
import threading
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

from llm_compression.inference.lora_format import LoRACard, LoRAFormat
from llm_compression.evolution.weight_probe import WeightMapProbe, ActivationHeatMap
from llm_compression.evolution.lora_extractor import LoRAExtractor

logger = logging.getLogger(__name__)


class NodeTier(Enum):
    """Hardware capability tier."""
    LEAF = "leaf"       # Low compute: CPU only, <8GB RAM
    HUB = "hub"         # Mid compute: GPU 8-16GB
    SUPER = "super"     # High compute: GPU 24GB+


@dataclass
class EvolutionEvent:
    """Record of a single evolution event."""
    timestamp: float
    trigger_query: str
    tier_used: str
    skill_name: str
    success: bool
    duration_seconds: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class SkillDistiller:
    """
    Central orchestrator for AI-OS self-evolution.
    
    This class coordinates the entire learning loop:
    1. Detects cognitive dissonance (router confidence < threshold)
    2. Determines the best tier for knowledge acquisition
    3. Acquires knowledge and distills it into a LoRA card
    4. Validates the new skill
    5. Publishes to the swarm
    
    Usage:
        distiller = SkillDistiller(
            engine=arrow_engine,
            node_tier=NodeTier.HUB,
            cloud_api_key="sk-..."
        )
        
        # Automatic: triggers when router confidence is low
        distiller.enable_auto_evolution(confidence_threshold=0.3)
        
        # Manual: extract skill from a specific topic
        card = distiller.learn_topic("quantum mechanics")
    """
    
    def __init__(
        self,
        engine=None,
        node_tier: NodeTier = NodeTier.LEAF,
        lora_output_dir: str = "./lora_skills",
        cloud_providers: Optional[Dict[str, Any]] = None,
        rank: int = 8,
        alpha: float = 16.0,
    ):
        self.engine = engine
        self.node_tier = node_tier
        self.output_dir = Path(lora_output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.cloud_providers = cloud_providers or {}
        
        # Core components
        self.probe = WeightMapProbe(top_k=10)
        self.extractor = LoRAExtractor(rank=rank, alpha=alpha)
        
        # QA Log for future training
        self.qa_log: List[Dict] = []
        
        # Evolution history
        self.history: List[EvolutionEvent] = []
        
        # Auto-evolution
        self._auto_enabled = False
        self._confidence_threshold = 0.3
        
    def log_qa(self, query: str, response: str, source: str = "user"):
        """Log a QA pair for future LoRA training."""
        self.qa_log.append({
            "query": query,
            "response": response,
            "source": source,
            "timestamp": time.time()
        })
        
        # Periodic save
        if len(self.qa_log) % 100 == 0:
            self._save_qa_log()
            
    def extract_skill_from_weights(
        self,
        name: str,
        weights: Dict[str, torch.Tensor],
        test_queries: List[str],
        description: str = "",
        reference_weights: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Optional[LoRACard]:
        """
        Tier 3: Extract a LoRA skill from model weights.
        
        Uses WeightMapProbe to identify relevant layers,
        then LoRAExtractor to create a compact adapter.
        
        Args:
            name: Name for the new skill.
            weights: Model weights to extract from.
            test_queries: Representative queries for the target domain.
            description: Human-readable description.
            reference_weights: Optional base weights for delta extraction.
            
        Returns:
            LoRACard if successful, None otherwise.
        """
        start = time.time()
        logger.info(f"Extracting skill '{name}' from weights...")
        
        try:
            if reference_weights:
                # Delta extraction: compare base vs fine-tuned
                card = self.extractor.extract_delta(
                    base_weights=reference_weights,
                    finetuned_weights=weights,
                    name=name,
                    description=description
                )
            else:
                # Direct extraction: probe activations then extract
                heat_map = self.probe.analyze_from_weights(weights)
                card = self.extractor.extract_from_heat_map(
                    weights=weights,
                    heat_map=heat_map,
                    name=name,
                    description=description
                )
            
            # Save
            output_path = self.output_dir / f"{name}.lora.arrow"
            LoRAFormat.save(card, str(output_path))
            
            duration = time.time() - start
            self.history.append(EvolutionEvent(
                timestamp=time.time(),
                trigger_query="; ".join(test_queries[:3]),
                tier_used="weight_extraction",
                skill_name=name,
                success=True,
                duration_seconds=duration,
                metadata={
                    "num_layers": len(card.weights_A),
                    "avg_variance": card.metadata.get("avg_explained_variance", 0),
                }
            ))
            
            logger.info(
                f"Skill '{name}' extracted in {duration:.1f}s: "
                f"{len(card.weights_A)} layers, saved to {output_path}"
            )
            return card
            
        except Exception as e:
            logger.error(f"Extraction failed for '{name}': {e}")
            self.history.append(EvolutionEvent(
                timestamp=time.time(),
                trigger_query="; ".join(test_queries[:3]),
                tier_used="weight_extraction",
                skill_name=name,
                success=False,
                duration_seconds=time.time() - start,
                metadata={"error": str(e)}
            ))
            return None
    
    def extract_skill_from_engine(
        self,
        name: str,
        test_queries: List[str],
        description: str = "",
    ) -> Optional[LoRACard]:
        """
        Extract a LoRA skill from the currently loaded ArrowEngine model.
        
        Uses the engine's inference_core for activation probing
        and the engine's weights for SVD extraction.
        
        Args:
            name: Name for the new skill.
            test_queries: Representative queries for the target domain.
            description: Human-readable description.
            
        Returns:
            LoRACard if successful, None otherwise.
        """
        if not self.engine:
            logger.error("No engine available for extraction.")
            return None
            
        start = time.time()
        logger.info(f"Extracting skill '{name}' from live engine...")
        
        try:
            # 1. Probe activations using actual inference
            heat_map = self.probe.analyze(
                inference_core=self.engine.inference_core,
                test_inputs=test_queries,
                tokenizer=self.engine.tokenizer
            )
            
            # 2. Extract from hot zones using model state_dict
            model_weights = self.engine.inference_core.state_dict()
            card = self.extractor.extract_from_heat_map(
                weights=model_weights,
                heat_map=heat_map,
                name=name,
                description=description
            )
            
            # 3. Save
            output_path = self.output_dir / f"{name}.lora.arrow"
            LoRAFormat.save(card, str(output_path))
            
            duration = time.time() - start
            self.history.append(EvolutionEvent(
                timestamp=time.time(),
                trigger_query="; ".join(test_queries[:3]),
                tier_used="engine_extraction",
                skill_name=name,
                success=True,
                duration_seconds=duration,
                metadata={
                    "num_layers": len(card.weights_A),
                    "hot_zones": [hz[0] for hz in heat_map.hot_zones[:5]],
                }
            ))
            
            logger.info(
                f"Skill '{name}' extracted from live engine in {duration:.1f}s"
            )
            return card
            
        except Exception as e:
            logger.error(f"Engine extraction failed: {e}")
            return None
    
    def get_evolution_summary(self) -> Dict:
        """Get summary of all evolution events."""
        if not self.history:
            return {"total_events": 0}
            
        successes = [e for e in self.history if e.success]
        failures = [e for e in self.history if not e.success]
        
        return {
            "total_events": len(self.history),
            "successes": len(successes),
            "failures": len(failures),
            "success_rate": len(successes) / len(self.history),
            "total_skills_created": len(successes),
            "tiers_used": list(set(e.tier_used for e in self.history)),
            "avg_duration": np.mean([e.duration_seconds for e in self.history]),
            "qa_log_size": len(self.qa_log),
        }
    
    def _save_qa_log(self):
        """Persist QA log to disk."""
        import json
        log_path = self.output_dir / "qa_log.jsonl"
        with open(log_path, 'a') as f:
            for entry in self.qa_log[-100:]:  # Save last 100
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        logger.debug(f"Saved QA log ({len(self.qa_log)} entries)")
