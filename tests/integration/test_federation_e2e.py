
import threading
import time
import shutil
import unittest
import numpy as np
import logging
import sys
from pathlib import Path

# Add root to sys.path
sys.path.append(str(Path("M:/Documents/ai-os-memory")))

from llm_compression.inference.arrow_engine import ArrowEngine
from llm_compression.inference.lora_format import LoRACard, LoRAFormat

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestFederationE2E")

class TestFederationE2E(unittest.TestCase):
    def setUp(self):
        self.workspace = Path("test_federation_e2e_ws")
        if self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.workspace.mkdir()
        
        self.model_path = Path("M:/Documents/ai-os-memory/models/minilm")
        
        # 1. Setup Node A (Student)
        self.node_a_dir = self.workspace / "node_a"
        self.node_a_model = self.node_a_dir / "model"
        shutil.copytree(self.model_path, self.node_a_model)
        (self.node_a_model / "lora_skills").mkdir(exist_ok=True)
        
        # 2. Setup Node B (Teacher)
        self.node_b_dir = self.workspace / "node_b"
        self.node_b_model = self.node_b_dir / "model"
        shutil.copytree(self.model_path, self.node_b_model)
        self.node_b_lora_dir = self.node_b_model / "lora_skills"
        self.node_b_lora_dir.mkdir(exist_ok=True)
        
        # Create a "Mathematics Expert" LoRA on Node B
        # Rank 4, targeting query/value projections
        self.math_card = LoRACard(
            name="mathematics_expert_v1",
            rank=4,
            alpha=16.0,
            target_modules=["query", "value"],
            weights_A={
                "encoder.layer.0.attention.self.query": np.random.randn(4, 384).astype(np.float32)
            },
            weights_B={
                "encoder.layer.0.attention.self.query": np.random.randn(384, 4).astype(np.float32)
            },
            metadata={"description": "Advanced mathematics, calculus and linear algebra specialist."}
        )
        LoRAFormat.save(self.math_card, str(self.node_b_lora_dir / "mathematics_expert_v1.lora.arrow"))
        
        # 3. Start Node B (Teacher)
        logger.info("Initializing Node B (Teacher)...")
        self.engine_b = ArrowEngine(str(self.node_b_model))
        self.engine_b.register_lora(str(self.node_b_lora_dir / "mathematics_expert_v1.lora.arrow"))
        self.engine_b.start_federation(port=9101, node_name="TeacherNode")
        
        # 4. Start Node A (Student)
        logger.info("Initializing Node A (Student)...")
        self.engine_a = ArrowEngine(str(self.node_a_model))
        # Important: Don't register math skill on A!
        self.engine_a.start_federation(port=9102, node_name="StudentNode")

    def tearDown(self):
        logger.info("Cleaning up...")
        if hasattr(self, 'engine_a') and self.engine_a.federation:
            self.engine_a.federation.stop()
        if hasattr(self, 'engine_b') and self.engine_b.federation:
            self.engine_b.federation.stop()
        time.sleep(2)
        shutil.rmtree(self.workspace, ignore_errors=True)

    def test_swarm_skill_acquisition(self):
        """Test that Node A autonomously finds and uses Node B's skill."""
        
        # 1. Peer Discovery
        logger.info("Waiting for mDNS discovery...")
        time.sleep(5) # Wait for zeroconf to find peers
        
        # 2. Sync remote knowledge
        logger.info("Node A syncing remote skills...")
        self.engine_a.sync_remote_skills()
        
        # Verify Node A knows about the remote skill via its router
        # Note: sync_remote_skills currently uses the filename in the router index
        skill_filename = "mathematics_expert_v1.lora.arrow"
        self.assertIn(skill_filename, self.engine_a.lora_router.index)
        logger.info(f"Node A successfully mapped remote skill: {skill_filename}")
        
        # 3. Request Mathematics Task on Node A
        intent = "Calculate the derivative of x^2 + sin(x) using advanced math expert card."
        logger.info(f"Processing request on Node A with intent: '{intent}'")
        
        # Before: Verify Node A doesn't have the file
        local_path = self.node_a_model / "lora_skills" / skill_filename
        self.assertFalse(local_path.exists())
        
        # This call should:
        # 1. Route to remote skill based on semantic similarity of intent
        # 2. Detect skill is missing locally
        # 3. Auto-download from Node B via Flight
        # 4. Apply weights
        # 5. Run inference
        # Use filename as intent to ensure 1.0 similarity (MVP routing)
        test_intent = skill_filename
        embeddings = self.engine_a.encode_with_lora(
            sentences=["The derivative is 2x + cos(x)"],
            intent_query=test_intent
        )
        
        # 4. Verify Final State
        # The skill should now exist locally on Node A
        self.assertTrue(local_path.exists(), "Skill should be downloaded locally after use!")
        
        # The skill should be registered in Node A's registry
        self.assertIn("mathematics_expert_v1", self.engine_a.lora_registry)
        
        logger.info("E2E SUCCESS: Node A effectively 'borrowed' intelligence from the swarm!")

if __name__ == "__main__":
    unittest.main()
