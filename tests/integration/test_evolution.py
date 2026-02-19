
"""
Test: AI-OS Self-Evolution Pipeline.

Verifies the complete extraction flow:
1. WeightMapProbe identifies hot zones in the model.
2. LoRAExtractor decomposes those zones via SVD.
3. SkillDistiller orchestrates the full pipeline.
4. The resulting LoRA card is valid and can be loaded.
"""

import unittest
import shutil
import numpy as np
import torch
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestEvolution")

from llm_compression.evolution.weight_probe import WeightMapProbe, ActivationHeatMap
from llm_compression.evolution.lora_extractor import LoRAExtractor
from llm_compression.inference.lora_format import LoRACard, LoRAFormat


class TestWeightMapProbe(unittest.TestCase):
    """Test weight probing without full model."""
    
    def test_analyze_from_weights(self):
        """Probe weights by magnitude analysis."""
        # Simulate model weights: some layers have much larger norms
        weights = {
            "encoder.layer.0.attention.self.query.weight": torch.randn(384, 384) * 0.1,
            "encoder.layer.0.attention.self.key.weight": torch.randn(384, 384) * 0.1,
            "encoder.layer.0.attention.self.value.weight": torch.randn(384, 384) * 0.1,
            "encoder.layer.0.intermediate.dense.weight": torch.randn(1536, 384) * 0.1,
            # Layer 3 is "specialized" — much larger weights
            "encoder.layer.3.attention.self.query.weight": torch.randn(384, 384) * 5.0,
            "encoder.layer.3.attention.self.key.weight": torch.randn(384, 384) * 3.0,
            "encoder.layer.3.attention.self.value.weight": torch.randn(384, 384) * 4.0,
            "encoder.layer.3.intermediate.dense.weight": torch.randn(1536, 384) * 2.0,
        }
        
        probe = WeightMapProbe(top_k=3)
        heat_map = probe.analyze_from_weights(weights)
        
        # Verify heat map structure
        self.assertIsInstance(heat_map, ActivationHeatMap)
        self.assertEqual(len(heat_map.hot_zones), 3)
        self.assertGreater(heat_map.total_energy, 0)
        self.assertGreater(heat_map.concentration_ratio, 0)
        
        # The hot zones should be from layer 3 (higher norms)
        hot_names = [name for name, _ in heat_map.hot_zones]
        self.assertTrue(any("layer.3" in name for name in hot_names))
        
        logger.info(f"Hot zones: {heat_map.hot_zones}")
        logger.info(f"Concentration: {heat_map.concentration_ratio:.1%}")
        
    def test_delta_analysis(self):
        """Probe weight deltas between base and fine-tuned."""
        base = {
            "layer.0.weight": torch.randn(384, 384),
            "layer.1.weight": torch.randn(384, 384),
        }
        finetuned = {
            "layer.0.weight": base["layer.0.weight"] + torch.randn(384, 384) * 0.01,  # Small change
            "layer.1.weight": base["layer.1.weight"] + torch.randn(384, 384) * 2.0,   # Large change
        }
        
        probe = WeightMapProbe(top_k=2)
        heat_map = probe.analyze_from_weights(finetuned, reference_weights=base)
        
        # Layer 1 should be hotter (larger delta)
        self.assertEqual(heat_map.hot_zones[0][0], "layer.1.weight")
        logger.info(f"Delta hot zones: {heat_map.hot_zones}")


class TestLoRAExtractor(unittest.TestCase):
    """Test SVD-based LoRA extraction."""
    
    def test_single_extraction(self):
        """Extract LoRA from a single weight matrix."""
        # Create a weight matrix with clear low-rank structure
        # W = A_true @ B_true + noise
        d_out, d_in, true_rank = 384, 384, 4
        A_true = torch.randn(d_out, true_rank)
        B_true = torch.randn(true_rank, d_in)
        W = A_true @ B_true + torch.randn(d_out, d_in) * 0.01  # Small noise
        
        extractor = LoRAExtractor(rank=4)
        result = extractor.extract_single(W, "test_layer")
        
        # Should capture most of the variance
        self.assertGreater(result.explained_variance, 0.95)
        self.assertEqual(result.rank, 4)
        self.assertEqual(result.weights_A.shape[0], 4)  # (rank, d_in)
        self.assertEqual(result.weights_B.shape[1], 4)  # (d_out, rank)
        
        # Verify reconstruction quality
        W_reconstructed = torch.from_numpy(result.weights_B) @ torch.from_numpy(result.weights_A)
        error = torch.norm(W - W_reconstructed) / torch.norm(W)
        self.assertLess(error.item(), 0.1)  # <10% error
        
        logger.info(
            f"Extraction: variance={result.explained_variance:.4f}, "
            f"reconstruction_error={error.item():.4f}"
        )
    
    def test_heat_map_extraction(self):
        """Extract LoRA from hot zones."""
        # Create weights with clear low-rank structure (like real pre-trained models)
        # W = A @ B + small noise → rank-8 captures most variance
        def make_structured(d_out, d_in, rank=8):
            A = torch.randn(d_out, rank)
            B = torch.randn(rank, d_in)
            return A @ B + torch.randn(d_out, d_in) * 0.01
        
        weights = {
            "encoder.layer.0.attention.self.query.weight": make_structured(384, 384) * 0.1,
            "encoder.layer.0.attention.self.key.weight": make_structured(384, 384) * 0.1,
            "encoder.layer.3.attention.self.query.weight": make_structured(384, 384) * 5.0,
            "encoder.layer.3.intermediate.dense.weight": make_structured(1536, 384) * 3.0,
        }
        
        # Heat map identifying layer 3 as hot
        heat_map = ActivationHeatMap(
            layer_activations={},
            hot_zones=[
                ("encoder.layer.3.attention.self.query", 100.0),
                ("encoder.layer.3.intermediate.dense", 80.0),
            ],
            total_energy=200.0,
            concentration_ratio=0.9,
        )
        
        extractor = LoRAExtractor(rank=8, min_explained_variance=0.3)
        card = extractor.extract_from_heat_map(
            weights=weights,
            heat_map=heat_map,
            name="math_expert_test",
            description="Test extraction from hot zones."
        )
        
        self.assertEqual(card.name, "math_expert_test")
        self.assertGreater(len(card.weights_A), 0)
        self.assertEqual(card.rank, 8)
        
        logger.info(f"Extracted card: {card.name}, layers={len(card.weights_A)}")
        
    def test_delta_extraction(self):
        """Extract LoRA from weight delta."""
        base_weights = {
            "layer.0.weight": torch.randn(384, 384),
            "layer.1.weight": torch.randn(384, 384),
            "layer.0.bias": torch.randn(384),  # 1D, should be skipped
        }
        
        # Simulate fine-tuning: add a low-rank perturbation to layer 1
        A_delta = torch.randn(384, 4)
        B_delta = torch.randn(4, 384)
        finetuned_weights = {
            "layer.0.weight": base_weights["layer.0.weight"] + torch.randn(384, 384) * 0.001,
            "layer.1.weight": base_weights["layer.1.weight"] + A_delta @ B_delta,
            "layer.0.bias": base_weights["layer.0.bias"],
        }
        
        extractor = LoRAExtractor(rank=4, min_explained_variance=0.3)
        card = extractor.extract_delta(
            base_weights=base_weights,
            finetuned_weights=finetuned_weights,
            name="finetune_delta_test",
            description="Delta extraction test."
        )
        
        # Layer 1 should be extracted (large delta), layer 0 might be skipped (tiny delta)
        self.assertGreater(len(card.weights_A), 0)
        self.assertIn("layer.1.weight", card.weights_A)
        
        logger.info(f"Delta extracted: {len(card.weights_A)} layers")
    
    def test_save_and_load_extracted(self):
        """Verify extracted LoRA can be saved and loaded."""
        ws = Path("test_extraction_ws")
        ws.mkdir(exist_ok=True)
        
        try:
            weights = {
                "encoder.layer.0.attention.self.query.weight": torch.randn(384, 384) * 3.0,
            }
            
            heat_map = ActivationHeatMap(
                hot_zones=[("encoder.layer.0.attention.self.query", 50.0)],
                total_energy=50.0,
                concentration_ratio=1.0,
            )
            
            extractor = LoRAExtractor(rank=4, min_explained_variance=0.01)
            card = extractor.extract_from_heat_map(
                weights=weights,
                heat_map=heat_map,
                name="roundtrip_test",
                description="Save/load roundtrip test."
            )
            
            # Save
            save_path = ws / "roundtrip_test.lora.arrow"
            LoRAFormat.save(card, str(save_path))
            self.assertTrue(save_path.exists())
            
            # Load
            loaded_card = LoRAFormat.load(str(save_path))
            self.assertEqual(loaded_card.name, "roundtrip_test")
            self.assertEqual(loaded_card.rank, 4)
            self.assertGreater(len(loaded_card.weights_A), 0)
            
            logger.info("Save/Load roundtrip: SUCCESS")
        finally:
            shutil.rmtree(ws, ignore_errors=True)


class TestSkillDistillerIntegration(unittest.TestCase):
    """Test the full distillation pipeline with live ArrowEngine."""
    
    def test_extract_from_live_engine(self):
        """Extract a skill from the real MiniLM model."""
        ws = Path("test_distiller_ws")
        ws.mkdir(exist_ok=True)
        
        try:
            from llm_compression.inference.arrow_engine import ArrowEngine
            from llm_compression.evolution.skill_distiller import SkillDistiller, NodeTier
            
            # Load real model
            model_path = "M:/Documents/ai-os-memory/models/minilm"
            engine = ArrowEngine(model_path)
            
            # Create distiller with low threshold for MiniLM (small model = distributed weights)
            distiller = SkillDistiller(
                engine=engine,
                node_tier=NodeTier.HUB,
                lora_output_dir=str(ws),
                rank=4
            )
            # MiniLM is a small, well-distributed model — lower the extraction threshold
            distiller.extractor.min_explained_variance = 0.01
            
            # Extract skill using actual inference probing
            card = distiller.extract_skill_from_engine(
                name="nlp_general_v1",
                test_queries=[
                    "Natural language processing and text understanding",
                    "Sentiment analysis of customer reviews",
                    "Named entity recognition in news articles",
                ],
                description="General NLP capabilities extracted from MiniLM."
            )
            
            # Verify
            self.assertIsNotNone(card)
            self.assertEqual(card.name, "nlp_general_v1")
            self.assertGreater(len(card.weights_A), 0)
            
            # Verify file was saved
            saved_path = ws / "nlp_general_v1.lora.arrow"
            self.assertTrue(saved_path.exists())
            
            # Verify can be loaded
            loaded = LoRAFormat.load(str(saved_path))
            self.assertEqual(loaded.name, "nlp_general_v1")
            
            # Check evolution summary
            summary = distiller.get_evolution_summary()
            self.assertEqual(summary["total_events"], 1)
            self.assertEqual(summary["successes"], 1)
            
            logger.info(f"LIVE ENGINE EXTRACTION SUCCESS!")
            logger.info(f"  Card: {card.name}")
            logger.info(f"  Layers: {len(card.weights_A)}")
            logger.info(f"  Rank: {card.rank}")
            logger.info(f"  Metadata: {card.metadata}")
            
        finally:
            shutil.rmtree(ws, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
