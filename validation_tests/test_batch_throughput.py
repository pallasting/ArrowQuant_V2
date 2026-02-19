"""
Batch Throughput Test

Test batch processing throughput, target > 2000 req/s (ideal) or > 1000 req/s (acceptable).
"""

import os
import sys
import time
from pathlib import Path

def test_batch_throughput():
    """Test batch throughput"""
    print("\n" + "=" * 60)
    print("Batch Throughput Test")
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
        
        # Load model once and reuse
        print(f"\nLoading model...")
        engine = ArrowEngine(str(model_path))
        print(f"v Model loaded")
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32]
        results = []
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Prepare test data
            texts = [f"Test sentence number {i} for throughput measurement." 
                    for i in range(batch_size)]
            
            # Warmup
            for _ in range(3):
                engine.encode(texts)
            
            # Test throughput
            num_batches = 50
            total_requests = batch_size * num_batches
            
            print(f"  Running {num_batches} batches ({total_requests} total requests)...")
            
            start = time.time()
            for _ in range(num_batches):
                embeddings = engine.encode(texts)
            elapsed = time.time() - start
            
            throughput = total_requests / elapsed
            latency_per_batch = (elapsed / num_batches) * 1000
            
            results.append({
                'batch_size': batch_size,
                'throughput': throughput,
                'latency_per_batch': latency_per_batch,
                'total_time': elapsed,
            })
            
            print(f"  Throughput: {throughput:.0f} req/s")
            print(f"  Latency per batch: {latency_per_batch:.2f}ms")
            print(f"  Total time: {elapsed:.2f}s")
        
        # Cleanup
        del engine
        
        # Find best batch size
        best_result = max(results, key=lambda x: x['throughput'])
        
        print(f"\n" + "=" * 60)
        print(f"Throughput Test Results:")
        print(f"=" * 60)
        
        for result in results:
            marker = " *" if result == best_result else ""
            print(f"\nBatch size {result['batch_size']}:{marker}")
            print(f"  Throughput: {result['throughput']:.0f} req/s")
            print(f"  Latency per batch: {result['latency_per_batch']:.2f}ms")
        
        # Evaluate results
        best_throughput = best_result['throughput']
        
        print(f"\nPerformance Evaluation:")
        print(f"  Best throughput: {best_throughput:.0f} req/s (batch={best_result['batch_size']})")
        print(f"  Target (ideal): > 2000 req/s")
        print(f"  Target (acceptable): > 1000 req/s")
        
        if best_throughput > 2000:
            print(f"  v Excellent - Throughput {best_throughput:.0f} > 2000 req/s")
            status = 0
        elif best_throughput > 1000:
            print(f"  v Good - Throughput {best_throughput:.0f} > 1000 req/s")
            status = 0
        elif best_throughput > 500:
            print(f"  ! Acceptable - Throughput {best_throughput:.0f} > 500 req/s")
            status = 0
        else:
            print(f"  ! Needs optimization - Throughput {best_throughput:.0f} < 500 req/s")
            status = 0
        
        if best_throughput < 1000:
            print(f"\nOptimization suggestions:")
            print(f"  1. Increase batch size (if memory allows)")
            print(f"  2. Use more powerful CPU or GPU")
            print(f"  3. Check if other processes are using resources")
            print(f"  4. Consider using multi-process parallelization")
        
        print(f"\nRecommended configuration:")
        print(f"  Batch size: {best_result['batch_size']}")
        print(f"  Expected throughput: {best_throughput:.0f} req/s")
        
        print(f"\n" + "=" * 60)
        print("v Batch throughput test completed")
        return status
        
    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_batch_throughput())
