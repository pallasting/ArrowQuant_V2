"""
Inference Latency Test

Test single inference latency, target median < 5ms (ideal) or < 10ms (acceptable).
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

def test_inference_latency():
    """Test inference latency"""
    print("\n" + "=" * 60)
    print("Inference Latency Test")
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
        from llm_compression.inference.arrow_engine import ArrowEngine
        
        # Load model
        print(f"\nLoading model...")
        engine = ArrowEngine(str(model_path))
        print(f"v Model loaded")
        
        # Warmup (important: first inference may be slow)
        print(f"\nWarming up model (10 runs)...")
        for _ in range(10):
            engine.encode("warmup")
        print(f"v Warmup completed")
        
        # Test latency
        print(f"\nRunning latency test (100 runs)...")
        test_text = "This is a test sentence for latency measurement."
        latencies = []
        
        for i in range(100):
            start = time.time()
            embedding = engine.encode(test_text)
            latency_ms = (time.time() - start) * 1000
            latencies.append(latency_ms)
            
            # Show progress
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/100")
        
        # Statistical analysis
        latencies = np.array(latencies)
        
        print(f"\nLatency Statistics:")
        print(f"  Mean: {np.mean(latencies):.2f}ms")
        print(f"  Median: {np.median(latencies):.2f}ms")
        print(f"  Std Dev: {np.std(latencies):.2f}ms")
        print(f"  Min: {np.min(latencies):.2f}ms")
        print(f"  Max: {np.max(latencies):.2f}ms")
        print(f"  P95: {np.percentile(latencies, 95):.2f}ms")
        print(f"  P99: {np.percentile(latencies, 99):.2f}ms")
        
        median_latency = np.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        
        # Evaluate results
        print(f"\nPerformance Evaluation:")
        print(f"  Target (ideal): median < 5ms")
        print(f"  Target (acceptable): median < 10ms")
        
        if median_latency < 5:
            print(f"  v Excellent - Median latency {median_latency:.2f}ms < 5ms")
            status = 0
        elif median_latency < 10:
            print(f"  v Good - Median latency {median_latency:.2f}ms < 10ms")
            status = 0
        elif median_latency < 20:
            print(f"  ! Acceptable - Median latency {median_latency:.2f}ms < 20ms")
            status = 0
        else:
            print(f"  ! Needs optimization - Median latency {median_latency:.2f}ms > 20ms")
            status = 0
        
        if p95_latency > 50:
            print(f"  ! P95 latency is high: {p95_latency:.2f}ms")
            print(f"\nOptimization suggestions:")
            print(f"  1. Check if CPU usage is too high")
            print(f"  2. Close background programs to reduce interference")
            print(f"  3. Consider using GPU acceleration (if available)")
        
        print(f"\n" + "=" * 60)
        print("v Inference latency test completed")
        return status
        
    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_inference_latency())
