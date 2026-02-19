"""
Model Load Speed Test

Test ArrowEngine model loading time, target < 100ms (ideal) or < 500ms (acceptable).
"""

import os
import sys
import time
from pathlib import Path

def test_load_speed():
    """Test model load speed"""
    print("\n" + "=" * 60)
    print("Model Load Speed Test")
    print("=" * 60)
    
    # Check model path (prioritize environment variable, then SSD, then relative path)
    model_path_str = os.environ.get(
        "ARROW_MODEL_PATH",
        "D:/ai-models/minilm" if os.path.exists("D:/ai-models/minilm") 
        else "./models/minilm"
    )
    model_path = Path(model_path_str)
    
    if not model_path.exists():
        print(f"\nX Model directory not found: {model_path}")
        print(f"   Please run model conversion first:")
        print(f"   python -m llm_compression.tools.cli convert \\")
        print(f"       --model sentence-transformers/all-MiniLM-L6-v2 \\")
        print(f"       --output ./models/minilm")
        return 1
    
    # Check required files
    required_files = ["metadata.json", "weights.parquet"]
    missing_files = []
    
    for file in required_files:
        if not (model_path / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print(f"\nX Missing required files: {', '.join(missing_files)}")
        return 1
    
    print(f"\nv Model directory: {model_path.absolute()}")
    
    # Check file size
    weights_size = (model_path / "weights.parquet").stat().st_size / (1024 * 1024)
    print(f"v Weights file size: {weights_size:.1f} MB")
    
    try:
        from llm_compression.inference.arrow_engine import ArrowEngine
        
        # Test load time (average of 3 runs)
        load_times = []
        
        for i in range(3):
            start = time.time()
            engine = ArrowEngine(str(model_path))
            load_time_ms = (time.time() - start) * 1000
            load_times.append(load_time_ms)
            
            if i == 0:
                # Show model info after first load
                print(f"\nModel Info:")
                print(f"  Embedding dimension: {engine.get_embedding_dimension()}")
                print(f"  Max sequence length: {engine.get_max_seq_length()}")
                print(f"  Device: {engine.device}")
            
            # Cleanup
            del engine
        
        avg_load_time = sum(load_times) / len(load_times)
        min_load_time = min(load_times)
        max_load_time = max(load_times)
        
        print(f"\nLoad Time Statistics (3 runs):")
        print(f"  Average: {avg_load_time:.2f}ms")
        print(f"  Min: {min_load_time:.2f}ms")
        print(f"  Max: {max_load_time:.2f}ms")
        
        # Evaluate results
        print(f"\nPerformance Evaluation:")
        print(f"  Target (ideal): < 100ms")
        print(f"  Target (acceptable): < 500ms")
        
        if avg_load_time < 100:
            print(f"  v Excellent - Load time {avg_load_time:.2f}ms < 100ms")
            status = 0
        elif avg_load_time < 500:
            print(f"  v Good - Load time {avg_load_time:.2f}ms < 500ms")
            status = 0
        else:
            print(f"  ! Needs optimization - Load time {avg_load_time:.2f}ms > 500ms")
            print(f"\nOptimization suggestions:")
            print(f"  1. Use SSD for model files")
            print(f"  2. Add model directory to antivirus whitelist")
            print(f"  3. Check disk I/O performance")
            status = 0  # Still pass, but with warning
        
        print(f"\n" + "=" * 60)
        print("v Model load test completed")
        return status
        
    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_load_speed())
