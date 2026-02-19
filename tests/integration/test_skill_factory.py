
"""
Test: AI-OS Skill Factory.

Verifies the automated production pipeline:
1. Task queue management
2. Background worker execution
3. LoRA training job completion

This simulates a 'Nightly Build' where the system learns while idle.
"""

import time
import json
import shutil
import unittest
import logging
from pathlib import Path

# Configure logging to see Factory output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("TestSkillFactory")


class TestSkillFactory(unittest.TestCase):
    """Test the automated skill production system."""
    
    def setUp(self):
        self.ws = Path("test_skill_factory_ws")
        if self.ws.exists():
            shutil.rmtree(self.ws)
        self.ws.mkdir()
        
        # Create a dummy dataset
        self.dataset_path = self.ws / "medical_qa.jsonl"
        with open(self.dataset_path, 'w') as f:
            data = [
                {"q": "What is hypertension?", "a": "High blood pressure."},
                {"q": "Symptoms of flu?", "a": "Fever, cough, fatigue."},
                {"q": "Treatment for headache?", "a": "Rest, hydration, NSAIDs."},
            ]
            for item in data:
                f.write(json.dumps(item) + "\n")
                
    def tearDown(self):
        shutil.rmtree(self.ws, ignore_errors=True)
        
    def test_factory_pipeline(self):
        """Test the full task lifecycle: Pending -> Running -> Completed."""
        from llm_compression.inference.arrow_engine import ArrowEngine
        from llm_compression.evolution.skill_factory import SkillFactory
        
        # 1. Load Engine (Mock or Real)
        # We use real engine for integration test
        engine = ArrowEngine("M:/Documents/ai-os-memory/models/minilm")
        
        # 2. Initialize Factory
        factory = SkillFactory(
            engine=engine,
            workspace_dir=str(self.ws / "factory_data"),
        )
        
        # 3. Add a Training Task
        task_id = factory.add_task(
            name="medical_basics_v1",
            task_type="train_dataset",
            priority=10,
            dataset_path=str(self.dataset_path),
            epochs=1  # Fast test
        )
        
        logger.info(f"Task added: {task_id}")
        
        # Verify task is pending
        self.assertIn(task_id, factory.tasks)
        self.assertEqual(factory.tasks[task_id].status, "pending")
        
        # 4. Start Worker
        factory.start_worker()
        
        # 5. Wait for completion
        # Max wait 10s
        for _ in range(20):
            if factory.tasks[task_id].status in ("completed", "failed"):
                break
            time.sleep(0.5)
            
        # 6. Verify Success
        task = factory.tasks[task_id]
        logger.info(f"Task final status: {task.status}")
        
        self.assertEqual(
            task.status, 
            "completed", 
            f"Task failed with error: {task.error}"
        )
        self.assertIsNotNone(task.completed_at)
        
        # 7. Verify Artifacts
        product_path = self.ws / "factory_data" / "products" / "medical_basics_v1.lora.arrow"
        self.assertTrue(product_path.exists())
        
        # 8. Load the product to ensure valid LoRA
        from llm_compression.inference.lora_format import LoRAFormat
        card = LoRAFormat.load(str(product_path))
        self.assertEqual(card.name, "medical_basics_v1")
        self.assertEqual(card.metadata["source"], "LoRATrainer")
        
        # 9. Stop Worker
        factory.stop_worker()
        
    def test_persistence(self):
        """Test that tasks are saved and loaded correctly."""
        from llm_compression.inference.arrow_engine import ArrowEngine
        from llm_compression.evolution.skill_factory import SkillFactory
        
        engine = ArrowEngine("M:/Documents/ai-os-memory/models/minilm")
        workspace = self.ws / "persistence_test"
        
        # Session 1: Create task
        f1 = SkillFactory(engine, str(workspace))
        tid = f1.add_task("persisted_task", "distill_cloud", topic="rust_lang")
        
        # Session 2: Reload
        f2 = SkillFactory(engine, str(workspace))
        self.assertIn(tid, f2.tasks)
        self.assertEqual(f2.tasks[tid].name, "persisted_task")


if __name__ == "__main__":
    unittest.main()
