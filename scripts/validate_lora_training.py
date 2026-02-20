
import os
import sys
import torch
import torch.nn as nn
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from llm_compression.inference.inference_core import InferenceCore
from llm_compression.evolution.lora_trainer import LoRATrainer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MockTokenizer:
    def encode(self, texts):
        # Return dummy tokens
        return {
            "input_ids": [[101, 102, 103, 104, 105, 102]],
            "attention_mask": [[1, 1, 1, 1, 1, 1]]
        }

class MockEngine:
    def __init__(self):
        self.device = "cpu"
        self.tokenizer = MockTokenizer()

        # Create a tiny InferenceCore
        config = {
            "hidden_size": 32,
            "num_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 64,
            "vocab_size": 1000
        }

        # Initialize with random weights
        weights = {}
        # (WeightLoader would usually provide these, here we let nn.Module init randoms if not loaded)
        # But InferenceCore expects weights in init.
        # Actually InferenceCore init doesn't enforce weight keys if we don't call load_weights explicitly
        # inside init? Let's check InferenceCore code.
        # It calls super().__init__() then sets config.
        # It doesn't seem to force load.
        # Let's just create it.

        self.inference_core = InferenceCore(weights={}, config=config, device=self.device)

        # Manually initialize weights since we passed empty dict
        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.inference_core.apply(init_weights)

def test_training_loop():
    logger.info("=== Starting LoRA Training Verification ===")

    # 1. Setup Mock Engine
    engine = MockEngine()

    # 2. Initialize Trainer
    output_dir = Path("./validation_results/lora_test")
    trainer = LoRATrainer(
        engine=engine,
        output_dir=str(output_dir),
        rank=4,
        alpha=8.0,
        learning_rate=1e-3
    )

    # 3. Verify Injection
    lora_count = 0
    for name, module in engine.inference_core.named_modules():
        if "LoRALinear" in str(type(module)):
            lora_count += 1

    logger.info(f"Injected {lora_count} LoRA layers.")
    if lora_count == 0:
        logger.error("❌ LoRA injection failed!")
        return False

    # 4. Run Training
    qa_data = [
        {"q": "What is AI?", "a": "Artificial Intelligence."},
        {"q": "Who are you?", "a": "I am AI-OS."}
    ]

    logger.info("Running training loop...")
    card = trainer.train_qa(qa_data, "test_skill", epochs=5)

    if not card:
        logger.error("❌ Training returned None.")
        return False

    # 5. Verify Output
    arrow_path = output_dir / "test_skill.lora.arrow"
    if arrow_path.exists():
        size = arrow_path.stat().st_size
        logger.info(f"✅ Generated .lora.arrow file ({size} bytes)")
    else:
        logger.error("❌ Output file not found.")
        return False

    logger.info("✅ Verification Passed!")
    return True

if __name__ == "__main__":
    test_training_loop()
