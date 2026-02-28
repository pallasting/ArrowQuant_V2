#!/usr/bin/env python3
"""
Existing API Performance Analysis

Since the new methods (quantize_batch, quantize_arrow) are not yet exported,
this benchmark analyzes the existing quantize() method to understand:
1. Current performance baseline
2. PyO3 boundary crossing overhead
3. Data transfer characteristics

This provides context for the zero-copy optimization review.
"""

import time
import numpy as np
import tracemalloc
import psutil
from typing import Dict, List
import json

try:
    from arrow_quant_v2 import ArrowQuantV2
    ARROW_QUANT_AVAILABLE = True
except ImportError:
    print("ERROR: arrow_quant_v2 not available.")
    ARROW_QUANT_AVAILABLE = False
    exit(1)


class ExistingAPIAnalysis:
    """Analyze existing API performance."""
    
    def __init__(self):
        self.quantizer = ArrowQuantV2(mode="diffusion")
        self.process = psutil.Process()
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_single_layer(
        self,
        num_elements: int,
        iterations: int = 10
    ) -> Dict:
        """Benchmark single layer quantization."""
        np.random.seed(42)
        
        # Generate test data
        weights = np.random.randn(num_elements).astype(np.float32)
        layer_size_mb = num_elements * 4 / 1024 / 1024
        
        times = []
        memory_deltas = []
        
        for _ in range(iterations):
            tracemalloc.start()
            mem_before = self.get_memory_mb()
            
            # Convert to list (required by current API)
            start_convert = time.perf_counter()
            weights_list = weights.tolist()
            convert_time = (time.perf_counter() - start_convert) * 1000
            
            # Quantize
            start_quant = time.perf_counter()
            result = self.quantizer.quantize(
                {"layer.0.weight": weights_list},
                bit_width=4
            )
            quant_time = (time.perf_counter() - start_quant) * 1000
            
            mem_after = self.get_memory_mb()
            tracemalloc.stop()
            
            total_time = convert_time + quant_time
            times.append({
                "convert": convert_time,
                "quantize": quant_time,
                "total": total_time
            })
            memory_deltas.append(mem_after - mem_before)
        
        # Calculate statistics
        convert_times = [t["convert"] for t in times]
        quant_times = [t["quantize"] for t in times]
        total_times = [t["total"] for t in times]
        
        return {
            "layer_size_mb": layer_size_mb,
            "convert_time_ms": {
                "median": float(np.median(convert_times)),
                "mean": float(np.mean(convert_times)),
                "std": float(np.std(convert_times)),
            },
            "quantize_time_ms": {
                "median": float(np.median(quant_times)),
                "mean": float(np.mean(quant_times)),
                "std": float(np.std(quant_times)),
            },
            "total_time_ms": {
                "median": float(np.median(total_times)),
                "mean": float(np.mean(total_times)),
                "std": float(np.std(total_times)),
            },
            "memory_delta_mb": float(np.median(memory_deltas)),
        }
    
    def benchmark_multiple_layers_sequential(
        self,
        num_elements: int,
        num_layers: int,
        iterations: int = 5
    ) -> Dict:
        """Benchmark multiple layers with sequential calls."""
        np.random.seed(42)
        
        # Generate test data
        weights_dict = {
            f"layer.{i}.weight": np.random.randn(num_elements).astype(np.float32)
            for i in range(num_layers)
        }
        total_size_mb = num_elements * 4 * num_layers / 1024 / 1024
        
        times = []
        memory_deltas = []
        
        for _ in range(iterations):
            tracemalloc.start()
            mem_before = self.get_memory_mb()
            
            start = time.perf_counter()
            
            # Sequential calls (one per layer)
            for layer_name, weights in weights_dict.items():
                weights_list = weights.tolist()
                result = self.quantizer.quantize(
                    {layer_name: weights_list},
                    bit_width=4
                )
            
            elapsed = (time.perf_counter() - start) * 1000
            
            mem_after = self.get_memory_mb()
            tracemalloc.stop()
            
            times.append(elapsed)
            memory_deltas.append(mem_after - mem_before)
        
        return {
            "num_layers": num_layers,
            "total_size_mb": total_size_mb,
            "total_time_ms": {
                "median": float(np.median(times)),
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
            },
            "per_layer_time_ms": float(np.median(times) / num_layers),
            "memory_delta_mb": float(np.median(memory_deltas)),
        }
    
    def run_analysis(self):
        """Run complete analysis."""
        print("\n" + "="*80)
        print("EXISTING API PERFORMANCE ANALYSIS")
        print("="*80)
        print("\nAnalyzing current quantize() method to establish baseline")
        print("for zero-copy optimization comparison.")
        
        results = {}
        
        # Test 1: Single layer, different sizes
        print("\n" + "─"*80)
        print("Test 1: Single Layer Performance (Different Sizes)")
        print("─"*80)
        
        sizes = [
            (1_000_000, "4MB"),
            (10_000_000, "40MB"),
            (100_000_000, "400MB"),
        ]
        
        single_layer_results = []
        for num_elements, size_label in sizes:
            print(f"\n  Testing {size_label} ({num_elements:,} elements)...")
            result = self.benchmark_single_layer(num_elements, iterations=5)
            result["size_label"] = size_label
            single_layer_results.append(result)
            
            print(f"    Convert time: {result['convert_time_ms']['median']:8.2f} ms")
            print(f"    Quantize time: {result['quantize_time_ms']['median']:8.2f} ms")
            print(f"    Total time: {result['total_time_ms']['median']:8.2f} ms")
            print(f"    Memory delta: {result['memory_delta_mb']:8.2f} MB")
        
        results["single_layer"] = single_layer_results
        
        # Test 2: Multiple layers, sequential calls
        print("\n" + "─"*80)
        print("Test 2: Multiple Layers (Sequential Calls)")
        print("─"*80)
        
        layer_counts = [10, 50, 100]
        multi_layer_results = []
        
        for num_layers in layer_counts:
            print(f"\n  Testing {num_layers} layers (4MB each)...")
            result = self.benchmark_multiple_layers_sequential(
                num_elements=1_000_000,
                num_layers=num_layers,
                iterations=3
            )
            multi_layer_results.append(result)
            
            print(f"    Total time: {result['total_time_ms']['median']:8.2f} ms")
            print(f"    Per-layer time: {result['per_layer_time_ms']:8.2f} ms")
            print(f"    Memory delta: {result['memory_delta_mb']:8.2f} MB")
        
        results["multi_layer"] = multi_layer_results
        
        # Generate summary
        self.generate_summary(results)
        
        # Save results
        output_file = "existing_api_analysis_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n  📊 Results saved to: {output_file}")
    
    def generate_summary(self, results: Dict):
        """Generate summary report."""
        print("\n" + "="*80)
        print("SUMMARY & OPTIMIZATION OPPORTUNITIES")
        print("="*80)
        
        # Analyze data conversion overhead
        print("\n1. Data Conversion Overhead (numpy → list):")
        print("   " + "─"*76)
        
        for result in results["single_layer"]:
            size = result["size_label"]
            convert_pct = (result["convert_time_ms"]["median"] / 
                          result["total_time_ms"]["median"] * 100)
            print(f"   {size:<10} Convert: {result['convert_time_ms']['median']:6.2f} ms "
                  f"({convert_pct:5.1f}% of total)")
        
        print("\n   💡 Opportunity: Zero-copy APIs can eliminate this overhead")
        
        # Analyze boundary crossing overhead
        print("\n2. Boundary Crossing Overhead (Multiple Layers):")
        print("   " + "─"*76)
        
        for result in results["multi_layer"]:
            layers = result["num_layers"]
            per_layer = result["per_layer_time_ms"]
            print(f"   {layers:3d} layers: {per_layer:6.2f} ms per layer")
        
        # Calculate overhead
        if len(results["multi_layer"]) >= 2:
            overhead_10 = results["multi_layer"][0]["per_layer_time_ms"]
            overhead_100 = results["multi_layer"][-1]["per_layer_time_ms"]
            
            print(f"\n   Per-layer overhead: {overhead_10:.2f} ms (10 layers) → "
                  f"{overhead_100:.2f} ms (100 layers)")
            print(f"   💡 Opportunity: Batch API can reduce to ~0.02 ms per layer (100x)")
        
        # Memory analysis
        print("\n3. Memory Overhead:")
        print("   " + "─"*76)
        
        for result in results["single_layer"]:
            size = result["size_label"]
            size_mb = result["layer_size_mb"]
            memory_mb = result["memory_delta_mb"]
            ratio = memory_mb / size_mb if size_mb > 0 else 0
            print(f"   {size:<10} Data: {size_mb:6.1f} MB, "
                  f"Memory: {memory_mb:6.1f} MB ({ratio:.2f}x)")
        
        print("\n   💡 Opportunity: Zero-copy can reduce memory overhead")
        
        # Expected improvements
        print("\n4. Expected Improvements with Zero-Copy Optimization:")
        print("   " + "─"*76)
        
        # Find 4MB result
        mb4_result = next((r for r in results["single_layer"] 
                          if r["size_label"] == "4MB"), None)
        
        if mb4_result:
            current_time = mb4_result["total_time_ms"]["median"]
            convert_time = mb4_result["convert_time_ms"]["median"]
            
            # Estimate Arrow IPC time (target: 5ms)
            arrow_target = 5.0
            speedup = current_time / arrow_target
            
            print(f"   Current (4MB): {current_time:.2f} ms")
            print(f"   Arrow IPC target: {arrow_target:.2f} ms")
            print(f"   Expected speedup: {speedup:.1f}x")
            print(f"   Conversion overhead eliminated: {convert_time:.2f} ms")
        
        # Batch API improvements
        if results["multi_layer"]:
            result_100 = results["multi_layer"][-1]
            current_per_layer = result_100["per_layer_time_ms"]
            batch_target = 0.02  # Target: 2ms / 100 layers
            
            print(f"\n   Current per-layer (100 layers): {current_per_layer:.2f} ms")
            print(f"   Batch API target: {batch_target:.2f} ms")
            print(f"   Expected improvement: {current_per_layer / batch_target:.0f}x")
        
        print("\n" + "="*80)


def main():
    """Run analysis."""
    if not ARROW_QUANT_AVAILABLE:
        return
    
    analysis = ExistingAPIAnalysis()
    analysis.run_analysis()
    
    print("\n✅ Analysis complete!")
    print("\nNext steps:")
    print("  1. Build extension with new methods: maturin develop --release")
    print("  2. Run comprehensive comparison: python bench_comprehensive_comparison.py")
    print("  3. Update documentation with findings")


if __name__ == "__main__":
    main()
