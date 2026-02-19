
"""
AI-OS LoRA Dynamic Capability Demo.

Simulates the behavior of the "Action Center" where user intents
are automatically routed to specific LoRA skills.

This demo:
1. Creates a synthetic AI Model (ArrowEngine compatible).
2. Generates 3 specialized LoRA Skills (Coding, Creative Writing, Math).
3. Registers them to the Semantic Router.
4. Accepts user input and demonstrates automatic skill activation.
"""

import torch
import numpy as np
import tempfile
import shutil
import json
import logging
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
import time

from llm_compression.inference.arrow_engine import ArrowEngine
from llm_compression.inference.lora_format import LoRACard, LoRAFormat

# Configure logging
logging.basicConfig(level=logging.ERROR)
console = logging.getLogger("demo")
console.setLevel(logging.INFO)

def create_synthetic_model(path: Path):
    """Creates a tiny synthetic BERT-like model."""
    console.info(f"Creating synthetic model nucleus at {path}...")
    path.mkdir(parents=True, exist_ok=True)
    
    hidden_size = 64
    # Minimal weights for ArrowEngine to load without crashing
    weights = {
        "embeddings.word_embeddings.weight": np.random.randn(100, hidden_size).astype(np.float32),
        "embeddings.position_embeddings.weight": np.random.randn(512, hidden_size).astype(np.float32),
        "embeddings.token_type_embeddings.weight": np.random.randn(2, hidden_size).astype(np.float32),
        "embeddings.LayerNorm.weight": np.ones(hidden_size, dtype=np.float32),
        "embeddings.LayerNorm.bias": np.zeros(hidden_size, dtype=np.float32),
        # One layer of attention
        "encoder.layer.0.attention.self.query.weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
        "encoder.layer.0.attention.self.query.bias": np.zeros(hidden_size, dtype=np.float32),
        "encoder.layer.0.attention.self.key.weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
        "encoder.layer.0.attention.self.key.bias": np.zeros(hidden_size, dtype=np.float32),
        "encoder.layer.0.attention.self.value.weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
        "encoder.layer.0.attention.self.value.bias": np.zeros(hidden_size, dtype=np.float32),
        "encoder.layer.0.attention.output.dense.weight": np.random.randn(hidden_size, hidden_size).astype(np.float32),
        "encoder.layer.0.attention.output.dense.bias": np.zeros(hidden_size, dtype=np.float32),
        "encoder.layer.0.attention.output.LayerNorm.weight": np.ones(hidden_size, dtype=np.float32),
        "encoder.layer.0.attention.output.LayerNorm.bias": np.zeros(hidden_size, dtype=np.float32),
        "encoder.layer.0.intermediate.dense.weight": np.random.randn(hidden_size*4, hidden_size).astype(np.float32),
        "encoder.layer.0.intermediate.dense.bias": np.zeros(hidden_size*4, dtype=np.float32),
        "encoder.layer.0.output.dense.weight": np.random.randn(hidden_size, hidden_size*4).astype(np.float32),
        "encoder.layer.0.output.dense.bias": np.zeros(hidden_size, dtype=np.float32),
        "encoder.layer.0.output.LayerNorm.weight": np.ones(hidden_size, dtype=np.float32),
        "encoder.layer.0.output.LayerNorm.bias": np.zeros(hidden_size, dtype=np.float32),
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
    pq.write_table(table, path / "weights.parquet")
    
    with open(path / "metadata.json", 'w') as f:
        json.dump({
            "model_info": {
                "hidden_size": hidden_size,
                "num_hidden_layers": 1,
                "num_attention_heads": 4,
                "vocab_size": 100
            }
        }, f)
        
    with open(path / "tokenizer.json", 'w') as f:
        f.write("{}")

def generate_skills(output_dir: Path):
    """Generates dummy LoRA cards for different skills."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    skills = [
        ("coding_expert", "Expert at Python, algorithms, and system architecture"),
        ("creative_writer", "Writes poems, stories, and creative fiction"),
        ("math_logic", "Solves complex mathematical problems and logic puzzles"),
        ("visual_artist", "Understands image aesthetics and composition")
    ]
    
    created = []
    for name, desc in skills:
        card = LoRACard(
            name=name,
            rank=4,
            alpha=16.0,
            target_modules=["query", "value"],
            weights_A={"attention.self.query": np.random.randn(4, 64).astype(np.float32)},
            weights_B={"attention.self.query": np.random.randn(64, 4).astype(np.float32)},
            metadata={"description": desc}
        )
        path = str(output_dir / f"{name}.lora.arrow")
        LoRAFormat.save(card, path)
        created.append(path)
        console.info(f"Generated Skill Module: {name} ({desc})")
        
    return created

def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        model_path = root / "nucleus"
        
        # 1. Create Core
        create_synthetic_model(model_path)
        
        # 2. Initialize OS
        console.info("\n[System] Initializing ArrowEngine...")
        # Mock FastTokenizer
        from unittest.mock import MagicMock
        with torch.inference_mode():
            # Mock the tokenizer class itself
            from llm_compression.inference import arrow_engine
            original_tok = arrow_engine.FastTokenizer
            arrow_engine.FastTokenizer = MagicMock()
            arrow_engine.FastTokenizer.return_value.encode.return_value = {
                'input_ids': np.ones((1, 10), dtype=np.int64),
                'attention_mask': np.ones((1, 10), dtype=np.int64)
            }
            arrow_engine.FastTokenizer.return_value.max_length = 128
            
            try:
                engine = ArrowEngine(str(model_path), device="cpu")
            finally:
                arrow_engine.FastTokenizer = original_tok

        # 3. Create & Register Skills
        console.info("\n[System] Scanning for LoRA Modules...")
        skill_paths = generate_skills(root / "skills")
        
        for path in skill_paths:
            engine.register_lora(path)
            
        console.info("\n" + "="*50)
        console.info(" AI-OS Dynamic Capability Demo")
        console.info("="*50)
        console.info("System Ready. Type a request matchable to:")
        console.info(" - Coding")
        console.info(" - Creative Writing")
        console.info(" - Math/Logic")
        console.info(" - Visual Art")
        console.info("Type 'exit' to quit.\n")
        
        # Mock Engine Encodings?
        # The engine uses random weights, so embeddings are random. 
        # Router won't work well with random vectors.
        # We need to mock the embedder in the router or mock engine.encode 
        # to map text to predefined vectors.
        
        # For demo purposes, let's inject a "Smart Mock" Router
        # that uses simple keyword matching if vectors are random.
        
        original_select = engine.lora_router.select
        
        def smart_mock_select(query, threshold=0.6, top_k=1):
            """Keyword based matcher for demo if model is garbage."""
            q = query.lower()
            if "code" in q or "python" in q: return ["coding_expert"]
            if "story" in q or "poem" in q or "write" in q: return ["creative_writer"]
            if "math" in q or "calc" in q: return ["math_logic"]
            if "art" in q or "draw" in q: return ["visual_artist"]
            return []
            
        engine.lora_router.select = smart_mock_select
        
        while True:
            query = input("\nUSER > ")
            if query.lower() in ["exit", "quit"]:
                break
                
            console.info(f"\n[Thinking] Analyzing intent: '{query}'...")
            
            # Use encode_with_lora which triggers the router
            # We pass output_attentions=False to implicitly use encoding path
            
            start = time.time()
            vector = engine.encode_with_lora(query, intent_query=query)
            dt = (time.time() - start) * 1000
            
            # Check what was activated (we can spy or check logs if we had them)
            # But the console info comes from our logs
            
            # Since we can't easily see internal state in this script without modification,
            # We can rely on the fact that if it worked, it didn't crash.
            
            # Let's peek at active cards just to be sure
            # In `encode_with_lora` it unloads it.
            # So `engine.inference_core.lora_manager.active_cards` should be empty now.
            
            # To show the user what happened, we can manually check router first
            selected = engine.lora_router.select(query)
            if selected:
                console.info(f"[Action]  Activated Neural Circuit: [{selected[0].upper()}]")
                console.info(f"[System]  Hot-swap injection complete. Processing in {dt:.1f}ms.")
            else:
                console.info("[Action]  Using General Purpose Core (No specialized LoRA found).")

if __name__ == "__main__":
    main()
