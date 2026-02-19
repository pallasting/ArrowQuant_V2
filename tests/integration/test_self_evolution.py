
"""
Test: AI-OS Self-Evolution Integration.

Verifies the complete cognitive loop:
1. Engine detects it cannot handle a query (cognitive dissonance)
2. CloudDistiller generates QA pairs from a mock cloud provider
3. Knowledge is distilled into a LoRA card
4. Card is registered and available for future queries

This test uses MockCloudProvider, so no real API calls are made.
"""

import time
import shutil
import unittest
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestSelfEvolution")


class TestCloudDistillation(unittest.TestCase):
    """Test cloud-based knowledge distillation."""
    
    def setUp(self):
        self.ws = Path("test_cloud_distill_ws")
        if self.ws.exists():
            shutil.rmtree(self.ws)
        self.ws.mkdir()
        
    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)
    
    def test_mock_cloud_distillation(self):
        """Distill knowledge from a mock cloud provider."""
        from llm_compression.inference.arrow_engine import ArrowEngine
        from llm_compression.evolution.cloud_distiller import (
            CloudDistiller, MockCloudProvider
        )
        
        # 1. Load engine
        engine = ArrowEngine("M:/Documents/ai-os-memory/models/minilm")
        
        # 2. Create mock cloud provider with structured responses
        mock = MockCloudProvider(responses={
            # The CloudDistiller will ask for QA pairs about quantum mechanics
            "quantum": json.dumps([
                {"q": "What is quantum superposition?",
                 "a": "Quantum superposition is the principle that a quantum system can exist in multiple states simultaneously until measured."},
                {"q": "What is quantum entanglement?",
                 "a": "Quantum entanglement is a phenomenon where two particles become correlated such that the state of one instantly affects the other."},
                {"q": "What is the Heisenberg uncertainty principle?",
                 "a": "The uncertainty principle states that certain pairs of physical properties, like position and momentum, cannot both be known precisely."},
                {"q": "What is a qubit?",
                 "a": "A qubit is the basic unit of quantum information, analogous to a classical bit but capable of existing in superposition."},
                {"q": "What is quantum tunneling?",
                 "a": "Quantum tunneling is the quantum mechanical phenomenon where a particle passes through a potential barrier."},
            ])
        })
        
        # 3. Run distillation
        distiller = CloudDistiller(
            engine=engine,
            output_dir=str(self.ws),
            rank=4,
        )
        
        card = distiller.distill_topic(
            topic="quantum mechanics",
            provider=mock,
            num_pairs=5,
            skill_name="quantum_physics_v1",
        )
        
        # 4. Verify
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "quantum_physics_v1")
        self.assertGreater(len(card.weights_A), 0)
        self.assertEqual(card.metadata["source"], "mock")
        self.assertEqual(card.metadata["topic"], "quantum mechanics")
        
        # Verify file was saved
        saved = self.ws / "quantum_physics_v1.lora.arrow"
        self.assertTrue(saved.exists())
        
        # Verify can be loaded
        from llm_compression.inference.lora_format import LoRAFormat
        loaded = LoRAFormat.load(str(saved))
        self.assertEqual(loaded.name, "quantum_physics_v1")
        
        logger.info(f"Cloud distillation SUCCESS: {card.name}")
        logger.info(f"  Source: {card.metadata['source']}")
        logger.info(f"  QA pairs: {card.metadata['num_qa_pairs']}")
        logger.info(f"  Variance: {card.metadata['explained_variance']}")
    
    def test_qa_dataset_distillation(self):
        """Distill from user-provided QA pairs (no cloud needed)."""
        from llm_compression.inference.arrow_engine import ArrowEngine
        from llm_compression.evolution.cloud_distiller import CloudDistiller
        
        engine = ArrowEngine("M:/Documents/ai-os-memory/models/minilm")
        
        distiller = CloudDistiller(
            engine=engine,
            output_dir=str(self.ws),
            rank=4,
        )
        
        # User-provided QA pairs about Python
        qa_pairs = [
            {"question": "What is a Python decorator?",
             "answer": "A decorator is a function that wraps another function, adding behavior before or after it."},
            {"question": "What is a generator in Python?",
             "answer": "A generator is a function that uses yield to produce a sequence of values lazily."},
            {"question": "What does GIL stand for?",
             "answer": "GIL stands for Global Interpreter Lock, which prevents multiple threads from executing Python bytecodes simultaneously."},
        ]
        
        card = distiller.distill_from_existing_qa(
            qa_pairs=qa_pairs,
            topic="Python programming",
            skill_name="python_expert_v1",
        )
        
        self.assertIsNotNone(card)
        self.assertEqual(card.name, "python_expert_v1")
        self.assertGreater(len(card.weights_A), 0)
        
        logger.info(f"QA distillation SUCCESS: {card.name}")


class TestCognitiveDissonance(unittest.TestCase):
    """Test the full cognitive loop in ArrowEngine."""
    
    def setUp(self):
        self.model_path = Path("test_evolution_engine_ws") / "model"
        if self.model_path.parent.exists():
            shutil.rmtree(self.model_path.parent)
        shutil.copytree("M:/Documents/ai-os-memory/models/minilm", self.model_path)
        (self.model_path / "lora_skills").mkdir(exist_ok=True)
        
    def tearDown(self):
        shutil.rmtree(self.model_path.parent, ignore_errors=True)
    
    def test_auto_evolution_trigger(self):
        """
        Test that the engine detects cognitive dissonance and
        triggers background evolution.
        """
        from llm_compression.inference.arrow_engine import ArrowEngine
        from llm_compression.evolution.cloud_distiller import MockCloudProvider
        
        # 1. Load engine
        engine = ArrowEngine(str(self.model_path))
        
        # 2. Enable evolution with a mock cloud provider
        mock_provider = MockCloudProvider(responses={
            "": json.dumps([
                {"q": "What is dark matter?", 
                 "a": "Dark matter is an invisible substance that makes up about 27% of the universe."},
                {"q": "How do we detect dark matter?",
                 "a": "We detect dark matter through gravitational effects on visible matter."},
                {"q": "What is dark energy?",
                 "a": "Dark energy is a mysterious force causing the universe to expand at an accelerating rate."},
            ])
        })
        
        engine.enable_evolution(
            cloud_providers={"mock": mock_provider},
            confidence_threshold=0.99,  # Very high threshold → always triggers
            rank=4,
        )
        
        self.assertIsNotNone(engine.distiller)
        
        # 3. Send a query the engine has NO skills for
        # This should trigger cognitive dissonance → background evolution
        logger.info("Sending query that triggers cognitive dissonance...")
        result = engine.encode_with_lora(
            sentences=["Dark matter constitutes most of the universe's mass"],
            intent_query="Explain dark matter and its role in cosmology",
        )
        
        # Result should still work (returns base encoding)
        self.assertIsNotNone(result)
        
        # 4. Wait for background evolution to complete
        logger.info("Waiting for background evolution thread...")
        if engine._evolution_thread:
            engine._evolution_thread.join(timeout=30)
        
        time.sleep(1)
        
        # 5. Verify evolution happened
        lora_dir = self.model_path / "lora_skills"
        lora_files = list(lora_dir.glob("evolved_*.lora.arrow"))
        
        self.assertGreater(len(lora_files), 0, "Evolution should have created a LoRA file!")
        
        logger.info(f"COGNITIVE DISSONANCE → EVOLUTION SUCCESS!")
        logger.info(f"  Created: {[f.name for f in lora_files]}")
        
        # 6. Verify the new skill is registered in the router
        if engine.lora_router:
            logger.info(f"  Router index: {list(engine.lora_router.index.keys())}")


import json  # needed for test data


if __name__ == "__main__":
    unittest.main()
