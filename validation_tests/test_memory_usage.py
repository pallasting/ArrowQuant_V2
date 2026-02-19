"""
Memory Usage Test

Test model memory usage, target < 100MB (ideal) or < 150MB (acceptable).
"""

import os
import sys
from pathlib import Path

def test_memory_usage():
    """Test memory usage"""
    print("\n" + "=" * 60)
    print("Memory Usage Test")
    print("=" * 60)
    
    model_path_str = os.environ.get(
        "ARROW_MODEL_PATH",
        "D:/ai-models/minilm" if os.path.exists("D:/ai-models/minilm") 
        else "./models/minilm"
    )
    model_path = Path(model_path_str)
    
    if not model_path.exists():
        print(f"\nX Model directory not found: {model_path}")
        return 1
    
    try:
        import psutil
    except ImportError:
        print(f"\n! psutil not installed, skipping memory test")
        print(f"   Install command: pip install psutil")
        return 0
    
    try:
        from llm_compression.inference.arrow_engine import ArrowEngine
        
        process = psutil.Process(os.getpid())
        
        # Baseline memory
        baseline_mb = process.memory_info().rss / (1024 * 1024)
        print(f"\nBaseline memory: {baseline_mb:.1f} MB")
        
        # Load model
        print(f"\nLoading model...")
        engine = ArrowEngine(str(model_path))
        
        after_load_mb = process.memory_info().rss / (1024 * 1024)
        load_memory_mb = after_load_mb - baseline_mb
        
        print(f"v Model loaded")
        print(f"  Memory after load: {after_load_mb:.1f} MB")
        print(f"  Model memory usage: {load_memory_mb:.1f} MB")
        
        # Run inference
        print(f"\nRunning inference test (20 runs)...")
        for i in range(20):
            embedding = engine.encode("Test sentence for memory measurement.")
            if (i + 1) % 5 == 0:
                current_mb = process.memory_info().rss / (1024 * 1024)
                print(f"  Run {i + 1}: {current_mb:.1f} MB")
        
        # Final memory
        final_mb = process.memory_info().rss / (1024 * 1024)
        total_memory_mb = final_mb - baseline_mb
        inference_overhead_mb = final_mb - after_load_mb
        
        print(f"\nMemory Statistics:")
        print(f"  Baseline memory: {baseline_mb:.1f} MB")
        print(f"  After load: {after_load_mb:.1f} MB")
        print(f"  Final memory: {final_mb:.1f} MB")
        print(f"  Model usage: {load_memory_mb:.1f} MB")
        print(f"  Inference overhead: {inference_overhead_mb:.1f} MB")
        print(f"  Total increase: {total_memory_mb:.1f} MB")
        
        # Evaluate results
        print(f"\nPerformance Evaluation:")
        print(f"  Target (ideal): < 100 MB")
        print(f"  Target (acceptable): < 150 MB")
        
        if total_memory_mb < 100:
            print(f"  v Excellent - Memory usage {total_memory_mb:.1f} MB < 100 MB")
            status = 0
        elif total_memory_mb < 150:
            print(f"  v Good - Memory usage {total_memory_mb:.1f} MB < 150 MB")
            status = 0
        elif total_memory_mb < 200:
            print(f"  ! Acceptable - Memory usage {total_memory_mb:.1f} MB < 200 MB")
            status = 0
        else:
            print(f"  ! Needs optimization - Memory usage {total_memory_mb:.1f} MB > 200 MB")
            status = 0
        
        if inference_overhead_mb > 50:
            print(f"  ! Inference overhead is high: {inference_overhead_mb:.1f} MB")
            print(f"\nOptimization suggestions:")
            print(f"  1. Check for memory leaks")
            print(f"  2. Reduce batch size")
            print(f"  3. Use float16 weights (if not already using)")
        
        # System memory status
        mem = psutil.virtual_memory()
        print(f"\nSystem Memory Status:")
        print(f"  Total memory: {mem.total / (1024**3):.1f} GB")
        print(f"  Used memory: {mem.used / (1024**3):.1f} GB ({mem.percent}%)")
        print(f"  Available memory: {mem.available / (1024**3):.1f} GB")
        
        if mem.available < 1 * (1024**3):
            print(f"  ! Warning: Available memory < 1GB, may affect performance")
        
        print(f"\n" + "=" * 60)
        print("v Memory usage test completed")
        return status
        
    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_memory_usage())
