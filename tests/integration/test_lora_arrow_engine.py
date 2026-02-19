
import unittest
import torch
import numpy as np
import tempfile
import json
import shutil
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from unittest.mock import MagicMock, patch

from llm_compression.inference.arrow_engine import ArrowEngine
from llm_compression.inference.lora_format import LoRACard, LoRAFormat

class TestLoRAIntegration(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        self.model_path = self.test_dir / "dummy_model"
        self.model_path.mkdir()
        
        # 1. Create Dummy Weights (Parquet)
        self.hidden_size = 32
        
        # Keys matching InferenceCore expectation for BERT
        weights = {
            "embeddings.word_embeddings.weight": np.random.randn(100, self.hidden_size).astype(np.float32),
            "embeddings.position_embeddings.weight": np.random.randn(512, self.hidden_size).astype(np.float32),
            "embeddings.token_type_embeddings.weight": np.random.randn(2, self.hidden_size).astype(np.float32),
            "embeddings.LayerNorm.weight": np.ones(self.hidden_size, dtype=np.float32),
            "embeddings.LayerNorm.bias": np.zeros(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.attention.self.query.weight": np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32),
            "encoder.layer.0.attention.self.query.bias": np.zeros(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.attention.self.key.weight": np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32),
            "encoder.layer.0.attention.self.key.bias": np.zeros(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.attention.self.value.weight": np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32),
            "encoder.layer.0.attention.self.value.bias": np.zeros(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.attention.output.dense.weight": np.random.randn(self.hidden_size, self.hidden_size).astype(np.float32),
            "encoder.layer.0.attention.output.dense.bias": np.zeros(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.attention.output.LayerNorm.weight": np.ones(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.attention.output.LayerNorm.bias": np.zeros(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.intermediate.dense.weight": np.random.randn(self.hidden_size*4, self.hidden_size).astype(np.float32),
            "encoder.layer.0.intermediate.dense.bias": np.zeros(self.hidden_size*4, dtype=np.float32),
            "encoder.layer.0.output.dense.weight": np.random.randn(self.hidden_size, self.hidden_size*4).astype(np.float32),
            "encoder.layer.0.output.dense.bias": np.zeros(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.output.LayerNorm.weight": np.ones(self.hidden_size, dtype=np.float32),
            "encoder.layer.0.output.LayerNorm.bias": np.zeros(self.hidden_size, dtype=np.float32),
        }
        
        names = []
        blobs = []
        shapes = []
        for k, v in weights.items():
            names.append(k)
            blobs.append(v.tobytes())
            shapes.append(list(v.shape))
            
        table = pa.Table.from_pydict({
            "layer_name": names,
            "data": blobs,
            "shape": shapes,
            "dtype": ["float32"] * len(names)
        })
        pq.write_table(table, self.model_path / "weights.parquet")
        
        # 2. Metadata
        meta = {
            "model_info": {
                "hidden_size": self.hidden_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "vocab_size": 100,
                "max_position_embeddings": 512
            }
        }
        with open(self.model_path / "metadata.json", 'w') as f:
            json.dump(meta, f)
            
        with open(self.model_path / "tokenizer.json", 'w') as f:
            f.write("{}")
            
    def tearDown(self):
        try:
            shutil.rmtree(self.test_dir)
        except:
            pass
        
    def test_end_to_end_lora(self):
        """Test ArrowEngine LoRA workflow with mocks."""
        
        # Patch FastTokenizer to avoid loading real files
        with patch('llm_compression.inference.arrow_engine.FastTokenizer') as MockTokenizer:
            mock_inst = MagicMock()
            # Return valid input_ids and attention_mask (batch=1, seq=10)
            mock_inst.encode.return_value = {
                'input_ids': np.ones((1, 10), dtype=np.int64),
                'attention_mask': np.ones((1, 10), dtype=np.int64)
            }
            mock_inst.max_length = 128
            MockTokenizer.return_value = mock_inst
            
            # --- Init Engine ---
            engine = ArrowEngine(str(self.model_path), device="cpu")
            import logging
            logging.getLogger().setLevel(logging.INFO)
            
            # Check InferenceCore init
            self.assertIsNotNone(engine.inference_core)
            self.assertIsNotNone(engine.inference_core.lora_manager)
            
            # --- Prepare LoRA ---
            lora_path = self.test_dir / "test_skill.lora.arrow"
            
            card = LoRACard(
                name="test_skill",
                rank=4,
                alpha=16.0,
                target_modules=["query"], # target suffix matches "attention.self.query"
                weights_A={"attention.self.query": np.random.randn(4, 32).astype(np.float32)}, 
                weights_B={"attention.self.query": np.random.randn(32, 4).astype(np.float32)},
                metadata={"description": "a specific skill"} 
            )
            
            LoRAFormat.save(card, str(lora_path))
            
            # --- Register ---
            engine.register_lora(str(lora_path))
            
            # Verify Router has it
            self.assertTrue("test_skill" in engine.lora_registry)
            
            # --- Encode with Intent ---
            # Mock Router.select to return our LoRA regardless of query vector
            
            with patch.object(engine.lora_router, 'select', return_value=["test_skill"]) as mock_select:
                # Spy on apply_card to confirm injection happened
                with patch.object(engine.inference_core.lora_manager, 'apply_card', wraps=engine.inference_core.lora_manager.apply_card) as mock_apply:
                    
                    emb = engine.encode_with_lora("hello world", intent_query="activate skill")
                    
                    # Verify flow
                    mock_select.assert_called()
                    mock_apply.assert_called() 
                    
                    # Output shape check
                    self.assertEqual(emb.shape, (1, 32))
                    
            # Verify Unload (should be automatic in finally block)
            # Check active cards in manager is empty
            self.assertEqual(len(engine.inference_core.lora_manager.active_cards), 0)

if __name__ == "__main__":
    unittest.main()
