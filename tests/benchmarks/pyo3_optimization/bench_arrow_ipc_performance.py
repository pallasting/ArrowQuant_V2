#!/usr/bin/env python3
"""
Arrow IPC Performance Benchmark

Measures Arrow IPC data transfer and quantization performance to validate:
- 30x speedup target (150ms → 5ms for 4MB tensors)
- Zero-copy data transfer
- Comparison against batch API and legacy approaches

Requirements validated: 3.3, 4.3
"""

import time
import numpy as np
import pyarrow as pa
from typing import Dict, List, Tuple
import json

try:
    from arrow_quant_v2 import ArrowQuantV2
    ARROW_QUANT_AVAILABLE = True
except ImportError:
    print("ERROR: arrow_quant_v2 not available. Run 'maturin develop --release' first.")
    ARROW_QUANT_AVAILABLE = False
    exit(1)


class ArrowIPCBenchmark:
    """Benchmark suite for Arrow IPC performance."""
    
    def __init__(self):
        self.quantizer = ArrowQuantV2(mode="diffusion")
        self.results = {}
    
    def generate_test_data(self, num_elements: int, num_layers: int = 1) -> Dict:
        """Generate test data for benchmarking."""
        np.random.seed(42)  # Reproducible results
        
        layer_names = [f"layer.{i}.weight" for i in range(num_layers)]
        weights_list = [
            np.random.randn(num_elements).astype(np.float32).tolist()
            for _ in range(num_layers)
        ]
        shapes = [[num_elements] for _ in range(num_layers)]
        
        return {
            "layer_name": layer_names,
            "weights": weights_list,
            "shape": shapes,
        }
    
    def benchmark_arrow_ipc_transfer(
        self, 
        num_elements: int, 
        iterations: int = 10
    ) -> Tuple[float, float, float]:
        """
        Benchmark Arrow IPC data transfer time.
        
        Returns:
            (transfer_time_ms, quantization_time_ms, total_time_ms)
        """
        # Generate test data
        data = self.generate_test_data(num_elements, num_layers=1)
        
        transfer_times = []
        quantization_times = []
        total_times = []
        
        for _ in range(iterations):
            # Measure table creation time (data transfer to Arrow)
            start_transfer = time.perf_counter()
            table = pa.Table.from_pydict(data)
            transfer_time = (time.perf_counter() - start_transfer) * 1000
            
            # Measure quantization time
            start_quant = time.perf_counter()
            result_table = self.quantizer.quantize_arrow(table, bit_width=4)
            quant_time = (time.perf_counter() - start_quant) * 1000
            
            # Total time
            total_time = transfer_time + quant_time
            
            transfer_times.append(transfer_time)
            quantization_times.append(quant_time)
            total_times.append(total_time)
        
        # Return median times (more robust than mean)
        return (
            np.median(transfer_times),
            np.median(quantization_times),
            np.median(total_times)
        )
    
    def benchmark_batch_api(
        self, 
        num_elements: int, 
        iterations: int = 10
    ) -> Tuple[float, float, float]:
        """
        Benchmark batch API for comparison.
        
        Returns:
            (transfer_time_ms, quantization_time_ms, total_time_ms)
        """
        # Generate test data as numpy arrays
        np.random.seed(42)
        weights_dict = {
            "layer.0.weight": np.random.randn(num_elements).astype(np.float32)
        }
        
        transfer_times = []
        quantization_times = []
        total_times = []
        
        for _ in range(iterations):
            # For batch API, transfer time is minimal (just dict creation)
            start_transfer = time.perf_counter()
            # Dict already created, so transfer time is negligible
            transfer_time = (time.perf_counter() - start_transfer) * 1000
            
            # Measure quantization time (includes PyO3 boundary crossing)
            start_quant = time.perf_counter()
            result = self.quantizer.quantize_batch(weights_dict, bit_width=4)
            quant_time = (time.perf_counter() - start_quant) * 1000
            
            total_time = transfer_time + quant_time
            
            transfer_times.append(transfer_time)
            quantization_times.append(quant_time)
            total_times.append(total_time)
        
        return (
            np.median(transfer_times),
            np.median(quantization_times),
            np.median(total_times)
        )
    
    def benchmark_multi_layer_arrow_ipc(
        self,
        num_elements: int,
        num_layers: int,
        iterations: int = 10
    ) -> Tuple[float, float, float]:
        """
        Benchmark Arrow IPC with multiple layers.
        
        Returns:
            (transfer_time_ms, quantization_time_ms, total_time_ms)
        """
        data = self.generate_test_data(num_elements, num_layers=num_layers)
        
        transfer_times = []
        quantization_times = []
        total_times = []
        
        for _ in range(iterations):
            start_transfer = time.perf_counter()
            table = pa.Table.from_pydict(data)
            transfer_time = (time.perf_counter() - start_transfer) * 1000
            
            start_quant = time.perf_counter()
            result_table = self.quantizer.quantize_arrow(table, bit_width=4)
            quant_time = (time.perf_counter() - start_quant) * 1000
            
            total_time = transfer_time + quant_time
            
            transfer_times.append(transfer_time)
            quantization_times.append(quant_time)
            total_times.append(total_time)
        
        return (
            np.median(transfer_times),
            np.median(quantization_times),
            np.median(total_times)
        )
    
    def run_tensor_size_benchmarks(self):
        """Run benchmarks for different tensor sizes (4MB, 40MB, 400MB)."""
        print("\n" + "="*80)
        print("ARROW IPC PERFORMANCE BENCHMARK - Tensor Size Comparison")
        print("="*80)
        
        # Test sizes: 4MB, 40MB, 400MB (float32 = 4 bytes)
        test_sizes = [
            (1_000_000, "4MB"),      # 1M elements = 4MB
            (10_000_000, "40MB"),    # 10M elements = 40MB
            (100_000_000, "400MB"),  # 100M elements = 400MB
        ]
        
        results = []
        
        for num_elements, size_label in test_sizes:
            print(f"\n--- Testing {size_label} tensor ({num_elements:,} elements) ---")
            
            # Benchmark Arrow IPC
            print("  Running Arrow IPC benchmark...")
            arrow_transfer, arrow_quant, arrow_total = self.benchmark_arrow_ipc_transfer(
                num_elements, iterations=5
            )
            
            # Benchmark Batch API for comparison
            print("  Running Batch API benchmark...")
            batch_transfer, batch_quant, batch_total = self.benchmark_batch_api(
                num_elements, iterations=5
            )
            
            # Calculate speedup
            speedup = batch_total / arrow_total if arrow_total > 0 else 0
            
            result = {
                "size": size_label,
                "elements": num_elements,
                "arrow_ipc": {
                    "transfer_ms": round(arrow_transfer, 2),
                    "quantization_ms": round(arrow_quant, 2),
                    "total_ms": round(arrow_total, 2),
                },
                "batch_api": {
                    "transfer_ms": round(batch_transfer, 2),
                    "quantization_ms": round(batch_quant, 2),
                    "total_ms": round(batch_total, 2),
                },
                "speedup": round(speedup, 2),
            }
            results.append(result)
            
            # Print results
            print(f"\n  Arrow IPC:")
            print(f"    Transfer:      {arrow_transfer:8.2f} ms")
            print(f"    Quantization:  {arrow_quant:8.2f} ms")
            print(f"    Total:         {arrow_total:8.2f} ms")
            
            print(f"\n  Batch API:")
            print(f"    Transfer:      {batch_transfer:8.2f} ms")
            print(f"    Quantization:  {batch_quant:8.2f} ms")
            print(f"    Total:         {batch_total:8.2f} ms")
            
            print(f"\n  Speedup: {speedup:.2f}x")
            
            # Check if target met for 4MB
            if size_label == "4MB":
                target_met = arrow_total <= 5.0
                print(f"\n  Target (≤5ms for 4MB): {'✅ PASS' if target_met else '❌ FAIL'}")
        
        self.results["tensor_size_benchmarks"] = results
        return results
    
    def run_multi_layer_benchmarks(self):
        """Run benchmarks for different numbers of layers."""
        print("\n" + "="*80)
        print("ARROW IPC PERFORMANCE BENCHMARK - Multi-Layer Comparison")
        print("="*80)
        
        # Test with 4MB tensors, varying number of layers
        num_elements = 1_000_000  # 4MB per layer
        layer_counts = [1, 10, 50, 100]
        
        results = []
        
        for num_layers in layer_counts:
            print(f"\n--- Testing {num_layers} layers (4MB each) ---")
            
            arrow_transfer, arrow_quant, arrow_total = self.benchmark_multi_layer_arrow_ipc(
                num_elements, num_layers, iterations=5
            )
            
            # Calculate per-layer time
            per_layer_ms = arrow_total / num_layers
            
            result = {
                "num_layers": num_layers,
                "total_size_mb": num_layers * 4,
                "transfer_ms": round(arrow_transfer, 2),
                "quantization_ms": round(arrow_quant, 2),
                "total_ms": round(arrow_total, 2),
                "per_layer_ms": round(per_layer_ms, 2),
            }
            results.append(result)
            
            print(f"  Transfer:      {arrow_transfer:8.2f} ms")
            print(f"  Quantization:  {arrow_quant:8.2f} ms")
            print(f"  Total:         {arrow_total:8.2f} ms")
            print(f"  Per layer:     {per_layer_ms:8.2f} ms")
        
        self.results["multi_layer_benchmarks"] = results
        return results
    
    def run_recordbatch_vs_table_benchmark(self):
        """Compare RecordBatch vs Table API performance."""
        print("\n" + "="*80)
        print("ARROW IPC PERFORMANCE BENCHMARK - RecordBatch vs Table")
        print("="*80)
        
        num_elements = 1_000_000  # 4MB
        iterations = 10
        
        # Generate test data
        data = self.generate_test_data(num_elements, num_layers=1)
        
        # Benchmark Table API
        print("\n--- Table API ---")
        table_times = []
        for _ in range(iterations):
            table = pa.Table.from_pydict(data)
            start = time.perf_counter()
            result = self.quantizer.quantize_arrow(table, bit_width=4)
            elapsed = (time.perf_counter() - start) * 1000
            table_times.append(elapsed)
        
        table_median = np.median(table_times)
        print(f"  Median time: {table_median:.2f} ms")
        
        # Benchmark RecordBatch API
        print("\n--- RecordBatch API ---")
        batch_times = []
        for _ in range(iterations):
            batch = pa.RecordBatch.from_pydict(data)
            start = time.perf_counter()
            result = self.quantizer.quantize_arrow_batch(batch, bit_width=4)
            elapsed = (time.perf_counter() - start) * 1000
            batch_times.append(elapsed)
        
        batch_median = np.median(batch_times)
        print(f"  Median time: {batch_median:.2f} ms")
        
        # Compare
        diff_pct = ((table_median - batch_median) / batch_median) * 100
        print(f"\n  Difference: {abs(diff_pct):.1f}% ({'Table faster' if diff_pct < 0 else 'RecordBatch faster'})")
        
        self.results["recordbatch_vs_table"] = {
            "table_median_ms": round(table_median, 2),
            "recordbatch_median_ms": round(batch_median, 2),
            "difference_pct": round(diff_pct, 2),
        }
    
    def generate_report(self):
        """Generate final benchmark report."""
        print("\n" + "="*80)
        print("BENCHMARK SUMMARY")
        print("="*80)
        
        # Tensor size summary
        if "tensor_size_benchmarks" in self.results:
            print("\n--- Tensor Size Performance ---")
            print(f"{'Size':<10} {'Arrow IPC':<15} {'Batch API':<15} {'Speedup':<10} {'Target':<10}")
            print("-" * 70)
            
            for result in self.results["tensor_size_benchmarks"]:
                size = result["size"]
                arrow_total = result["arrow_ipc"]["total_ms"]
                batch_total = result["batch_api"]["total_ms"]
                speedup = result["speedup"]
                
                # Check target for 4MB
                target_status = ""
                if size == "4MB":
                    target_status = "✅ PASS" if arrow_total <= 5.0 else "❌ FAIL"
                
                print(f"{size:<10} {arrow_total:>8.2f} ms    {batch_total:>8.2f} ms    {speedup:>5.2f}x    {target_status}")
        
        # Multi-layer summary
        if "multi_layer_benchmarks" in self.results:
            print("\n--- Multi-Layer Performance ---")
            print(f"{'Layers':<10} {'Total Size':<15} {'Total Time':<15} {'Per Layer':<15}")
            print("-" * 60)
            
            for result in self.results["multi_layer_benchmarks"]:
                layers = result["num_layers"]
                size_mb = result["total_size_mb"]
                total_ms = result["total_ms"]
                per_layer = result["per_layer_ms"]
                
                print(f"{layers:<10} {size_mb:>8} MB      {total_ms:>8.2f} ms    {per_layer:>8.2f} ms")
        
        # Key findings
        print("\n--- Key Findings ---")
        
        if "tensor_size_benchmarks" in self.results:
            # Find 4MB result
            mb4_result = next(
                (r for r in self.results["tensor_size_benchmarks"] if r["size"] == "4MB"),
                None
            )
            
            if mb4_result:
                arrow_time = mb4_result["arrow_ipc"]["total_ms"]
                speedup = mb4_result["speedup"]
                
                print(f"  • 4MB tensor transfer time: {arrow_time:.2f} ms")
                print(f"  • Target (≤5ms): {'✅ ACHIEVED' if arrow_time <= 5.0 else '❌ NOT MET'}")
                print(f"  • Speedup vs Batch API: {speedup:.2f}x")
                
                # Calculate speedup vs baseline (150ms)
                baseline_speedup = 150.0 / arrow_time
                print(f"  • Speedup vs Baseline (150ms): {baseline_speedup:.1f}x")
                print(f"  • Target (30x speedup): {'✅ ACHIEVED' if baseline_speedup >= 30 else '❌ NOT MET'}")
        
        # Save results to JSON
        output_file = "arrow_ipc_benchmark_results.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n  Results saved to: {output_file}")
        
        print("\n" + "="*80)


def main():
    """Run all benchmarks."""
    if not ARROW_QUANT_AVAILABLE:
        return
    
    print("Arrow IPC Performance Benchmark")
    print("================================")
    print("\nThis benchmark validates:")
    print("  • 30x speedup target (150ms → 5ms for 4MB)")
    print("  • Zero-copy data transfer performance")
    print("  • Comparison with batch API")
    print("\nRequirements validated: 3.3, 4.3")
    
    benchmark = ArrowIPCBenchmark()
    
    # Run all benchmark suites
    benchmark.run_tensor_size_benchmarks()
    benchmark.run_multi_layer_benchmarks()
    benchmark.run_recordbatch_vs_table_benchmark()
    
    # Generate final report
    benchmark.generate_report()


if __name__ == "__main__":
    main()
