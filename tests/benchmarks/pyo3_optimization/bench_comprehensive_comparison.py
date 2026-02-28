#!/usr/bin/env python3
"""
Comprehensive Performance Comparison Benchmark

This benchmark specifically addresses the data copy overhead identified in
ZERO_COPY_IMPLEMENTATION_REVIEW.md and provides detailed comparison between:
- Batch API (with data copy for parallel processing)
- Arrow IPC (true zero-copy)

Objectives:
1. Quantify the actual cost of data copying in Batch API
2. Compare Batch API vs Arrow IPC performance across different scenarios
3. Provide data-driven recommendations for API selection

Priority 2 Task from ZERO_COPY_IMPLEMENTATION_REVIEW.md
"""

import time
import numpy as np
import pyarrow as pa
import tracemalloc
import psutil
from typing import Dict, List, Tuple
import json
from dataclasses import dataclass, asdict

try:
    from arrow_quant_v2 import ArrowQuantV2
    ARROW_QUANT_AVAILABLE = True
except ImportError:
    print("ERROR: arrow_quant_v2 not available. Run 'maturin develop --release' first.")
    ARROW_QUANT_AVAILABLE = False
    exit(1)


@dataclass
class BenchmarkResult:
    """Structured benchmark result."""
    api_name: str
    scenario: str
    num_layers: int
    layer_size_mb: float
    total_size_mb: float
    
    # Timing metrics (milliseconds)
    median_time_ms: float
    mean_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    
    # Memory metrics (MB)
    memory_before_mb: float
    memory_after_mb: float
    memory_delta_mb: float
    peak_memory_mb: float
    
    # Derived metrics
    throughput_mb_per_sec: float
    per_layer_time_ms: float


class ComprehensiveBenchmark:
    """Comprehensive performance comparison benchmark."""
    
    def __init__(self):
        self.quantizer = ArrowQuantV2(mode="diffusion")
        self.results: List[BenchmarkResult] = []
        self.process = psutil.Process()
    
    def get_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def generate_numpy_weights(self, num_elements: int, num_layers: int) -> Dict[str, np.ndarray]:
        """Generate numpy weight dictionary for Batch API."""
        np.random.seed(42)
        return {
            f"layer.{i}.weight": np.random.randn(num_elements).astype(np.float32)
            for i in range(num_layers)
        }
    
    def generate_arrow_table(self, num_elements: int, num_layers: int) -> pa.Table:
        """Generate Arrow table for Arrow IPC API."""
        np.random.seed(42)
        
        layer_names = [f"layer.{i}.weight" for i in range(num_layers)]
        weights_list = [
            np.random.randn(num_elements).astype(np.float32).tolist()
            for _ in range(num_layers)
        ]
        shapes = [[num_elements] for _ in range(num_layers)]
        
        return pa.Table.from_pydict({
            "layer_name": layer_names,
            "weights": weights_list,
            "shape": shapes,
        })
    
    def benchmark_batch_api(
        self,
        num_elements: int,
        num_layers: int,
        iterations: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark Batch API with detailed metrics.
        
        This API copies data (to_vec()) to enable parallel processing.
        """
        # Generate test data
        weights = self.generate_numpy_weights(num_elements, num_layers)
        layer_size_mb = num_elements * 4 / 1024 / 1024  # float32 = 4 bytes
        total_size_mb = layer_size_mb * num_layers
        
        times = []
        memory_deltas = []
        peak_memories = []
        
        for _ in range(iterations):
            # Start memory tracking
            tracemalloc.start()
            mem_before = self.get_memory_mb()
            
            # Run quantization
            start = time.perf_counter()
            result = self.quantizer.quantize_batch(weights, bit_width=4)
            elapsed = (time.perf_counter() - start) * 1000
            
            # Measure memory
            mem_after = self.get_memory_mb()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times.append(elapsed)
            memory_deltas.append(mem_after - mem_before)
            peak_memories.append(peak / 1024 / 1024)
        
        # Calculate statistics
        times_array = np.array(times)
        median_time = np.median(times_array)
        throughput = total_size_mb / (median_time / 1000) if median_time > 0 else 0
        
        return BenchmarkResult(
            api_name="Batch API",
            scenario=f"{num_layers} layers × {layer_size_mb:.1f}MB",
            num_layers=num_layers,
            layer_size_mb=layer_size_mb,
            total_size_mb=total_size_mb,
            median_time_ms=float(np.median(times_array)),
            mean_time_ms=float(np.mean(times_array)),
            std_time_ms=float(np.std(times_array)),
            min_time_ms=float(np.min(times_array)),
            max_time_ms=float(np.max(times_array)),
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=float(np.median(memory_deltas)),
            peak_memory_mb=float(np.median(peak_memories)),
            throughput_mb_per_sec=throughput,
            per_layer_time_ms=float(np.median(times_array) / num_layers),
        )
    
    def benchmark_arrow_ipc(
        self,
        num_elements: int,
        num_layers: int,
        iterations: int = 10
    ) -> BenchmarkResult:
        """
        Benchmark Arrow IPC API with detailed metrics.
        
        This API uses true zero-copy via Arrow C Data Interface.
        """
        # Generate test data
        table = self.generate_arrow_table(num_elements, num_layers)
        layer_size_mb = num_elements * 4 / 1024 / 1024
        total_size_mb = layer_size_mb * num_layers
        
        times = []
        memory_deltas = []
        peak_memories = []
        
        for _ in range(iterations):
            # Start memory tracking
            tracemalloc.start()
            mem_before = self.get_memory_mb()
            
            # Run quantization
            start = time.perf_counter()
            result = self.quantizer.quantize_arrow(table, bit_width=4)
            elapsed = (time.perf_counter() - start) * 1000
            
            # Measure memory
            mem_after = self.get_memory_mb()
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times.append(elapsed)
            memory_deltas.append(mem_after - mem_before)
            peak_memories.append(peak / 1024 / 1024)
        
        # Calculate statistics
        times_array = np.array(times)
        median_time = np.median(times_array)
        throughput = total_size_mb / (median_time / 1000) if median_time > 0 else 0
        
        return BenchmarkResult(
            api_name="Arrow IPC",
            scenario=f"{num_layers} layers × {layer_size_mb:.1f}MB",
            num_layers=num_layers,
            layer_size_mb=layer_size_mb,
            total_size_mb=total_size_mb,
            median_time_ms=float(np.median(times_array)),
            mean_time_ms=float(np.mean(times_array)),
            std_time_ms=float(np.std(times_array)),
            min_time_ms=float(np.min(times_array)),
            max_time_ms=float(np.max(times_array)),
            memory_before_mb=mem_before,
            memory_after_mb=mem_after,
            memory_delta_mb=float(np.median(memory_deltas)),
            peak_memory_mb=float(np.median(peak_memories)),
            throughput_mb_per_sec=throughput,
            per_layer_time_ms=float(np.median(times_array) / num_layers),
        )
    
    def run_scenario_comparison(
        self,
        num_elements: int,
        num_layers: int,
        scenario_name: str
    ):
        """Run comparison for a specific scenario."""
        print(f"\n{'='*80}")
        print(f"Scenario: {scenario_name}")
        print(f"  Layers: {num_layers}")
        print(f"  Elements per layer: {num_elements:,}")
        print(f"  Size per layer: {num_elements * 4 / 1024 / 1024:.1f} MB")
        print(f"  Total size: {num_elements * 4 * num_layers / 1024 / 1024:.1f} MB")
        print(f"{'='*80}")
        
        # Benchmark Batch API
        print("\n[1/2] Benchmarking Batch API...")
        batch_result = self.benchmark_batch_api(num_elements, num_layers, iterations=10)
        self.results.append(batch_result)
        
        print(f"  ✓ Median time: {batch_result.median_time_ms:.2f} ms")
        print(f"  ✓ Memory delta: {batch_result.memory_delta_mb:.2f} MB")
        print(f"  ✓ Throughput: {batch_result.throughput_mb_per_sec:.1f} MB/s")
        
        # Benchmark Arrow IPC
        print("\n[2/2] Benchmarking Arrow IPC...")
        arrow_result = self.benchmark_arrow_ipc(num_elements, num_layers, iterations=10)
        self.results.append(arrow_result)
        
        print(f"  ✓ Median time: {arrow_result.median_time_ms:.2f} ms")
        print(f"  ✓ Memory delta: {arrow_result.memory_delta_mb:.2f} MB")
        print(f"  ✓ Throughput: {arrow_result.throughput_mb_per_sec:.1f} MB/s")
        
        # Compare results
        print(f"\n{'─'*80}")
        print("Comparison:")
        print(f"{'─'*80}")
        
        time_diff = batch_result.median_time_ms - arrow_result.median_time_ms
        time_ratio = batch_result.median_time_ms / arrow_result.median_time_ms if arrow_result.median_time_ms > 0 else 0
        
        memory_diff = batch_result.memory_delta_mb - arrow_result.memory_delta_mb
        memory_ratio = batch_result.memory_delta_mb / arrow_result.memory_delta_mb if arrow_result.memory_delta_mb > 0 else 0
        
        print(f"  Time difference: {time_diff:+.2f} ms ({time_ratio:.2f}x)")
        print(f"  Memory difference: {memory_diff:+.2f} MB ({memory_ratio:.2f}x)")
        
        # Estimate data copy overhead
        expected_copy_size = num_elements * 4 * num_layers / 1024 / 1024
        print(f"\n  Expected data copy size: {expected_copy_size:.1f} MB")
        print(f"  Actual memory overhead: {memory_diff:.1f} MB")
        print(f"  Copy overhead ratio: {memory_diff / expected_copy_size:.2f}x")
        
        # Winner
        if arrow_result.median_time_ms < batch_result.median_time_ms:
            winner = "Arrow IPC"
            advantage = time_ratio
        else:
            winner = "Batch API"
            advantage = 1 / time_ratio if time_ratio > 0 else 0
        
        print(f"\n  🏆 Winner: {winner} ({advantage:.2f}x faster)")
    
    def run_all_scenarios(self):
        """Run all benchmark scenarios."""
        print("\n" + "="*80)
        print("COMPREHENSIVE PERFORMANCE COMPARISON")
        print("="*80)
        print("\nObjective: Quantify data copy overhead and compare Batch API vs Arrow IPC")
        print("\nScenarios:")
        print("  1. Small model (10 layers × 4MB)")
        print("  2. Medium model (50 layers × 4MB)")
        print("  3. Large model (100 layers × 4MB)")
        print("  4. Single large layer (1 layer × 400MB)")
        print("  5. Memory-constrained (100 layers × 1MB)")
        
        # Scenario 1: Small model
        self.run_scenario_comparison(
            num_elements=1_000_000,  # 4MB
            num_layers=10,
            scenario_name="Small Model (10 layers × 4MB)"
        )
        
        # Scenario 2: Medium model
        self.run_scenario_comparison(
            num_elements=1_000_000,  # 4MB
            num_layers=50,
            scenario_name="Medium Model (50 layers × 4MB)"
        )
        
        # Scenario 3: Large model
        self.run_scenario_comparison(
            num_elements=1_000_000,  # 4MB
            num_layers=100,
            scenario_name="Large Model (100 layers × 4MB)"
        )
        
        # Scenario 4: Single large layer
        self.run_scenario_comparison(
            num_elements=100_000_000,  # 400MB
            num_layers=1,
            scenario_name="Single Large Layer (1 layer × 400MB)"
        )
        
        # Scenario 5: Memory-constrained
        self.run_scenario_comparison(
            num_elements=250_000,  # 1MB
            num_layers=100,
            scenario_name="Memory-Constrained (100 layers × 1MB)"
        )
    
    def generate_summary_report(self):
        """Generate comprehensive summary report."""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        # Group results by scenario
        scenarios = {}
        for result in self.results:
            if result.scenario not in scenarios:
                scenarios[result.scenario] = {}
            scenarios[result.scenario][result.api_name] = result
        
        # Print comparison table
        print("\n" + "─"*80)
        print("Performance Comparison Table")
        print("─"*80)
        print(f"{'Scenario':<35} {'Batch API':<15} {'Arrow IPC':<15} {'Speedup':<10}")
        print("─"*80)
        
        for scenario_name, apis in scenarios.items():
            batch = apis.get("Batch API")
            arrow = apis.get("Arrow IPC")
            
            if batch and arrow:
                speedup = batch.median_time_ms / arrow.median_time_ms if arrow.median_time_ms > 0 else 0
                speedup_str = f"{speedup:.2f}x"
                
                print(f"{scenario_name:<35} {batch.median_time_ms:>8.2f} ms   {arrow.median_time_ms:>8.2f} ms   {speedup_str:>8}")
        
        # Memory overhead analysis
        print("\n" + "─"*80)
        print("Memory Overhead Analysis")
        print("─"*80)
        print(f"{'Scenario':<35} {'Batch API':<15} {'Arrow IPC':<15} {'Overhead':<10}")
        print("─"*80)
        
        for scenario_name, apis in scenarios.items():
            batch = apis.get("Batch API")
            arrow = apis.get("Arrow IPC")
            
            if batch and arrow:
                overhead = batch.memory_delta_mb - arrow.memory_delta_mb
                overhead_str = f"+{overhead:.1f} MB" if overhead > 0 else f"{overhead:.1f} MB"
                
                print(f"{scenario_name:<35} {batch.memory_delta_mb:>8.1f} MB   {arrow.memory_delta_mb:>8.1f} MB   {overhead_str:>8}")
        
        # Key findings
        print("\n" + "─"*80)
        print("Key Findings")
        print("─"*80)
        
        # Calculate average speedup
        speedups = []
        memory_overheads = []
        
        for scenario_name, apis in scenarios.items():
            batch = apis.get("Batch API")
            arrow = apis.get("Arrow IPC")
            
            if batch and arrow:
                speedup = batch.median_time_ms / arrow.median_time_ms if arrow.median_time_ms > 0 else 0
                speedups.append(speedup)
                
                overhead = batch.memory_delta_mb - arrow.memory_delta_mb
                memory_overheads.append(overhead)
        
        avg_speedup = np.mean(speedups)
        avg_memory_overhead = np.mean(memory_overheads)
        
        print(f"\n1. Performance:")
        print(f"   • Average Arrow IPC speedup: {avg_speedup:.2f}x")
        print(f"   • Range: {min(speedups):.2f}x - {max(speedups):.2f}x")
        
        print(f"\n2. Memory Overhead:")
        print(f"   • Average Batch API overhead: {avg_memory_overhead:.1f} MB")
        print(f"   • This is due to data copying (to_vec()) for parallel processing")
        
        print(f"\n3. Data Copy Cost:")
        # Find large model scenario
        large_model = scenarios.get("Large Model (100 layers × 4MB)")
        if large_model:
            batch = large_model.get("Batch API")
            arrow = large_model.get("Arrow IPC")
            if batch and arrow:
                expected_copy = batch.total_size_mb
                actual_overhead = batch.memory_delta_mb - arrow.memory_delta_mb
                copy_ratio = actual_overhead / expected_copy if expected_copy > 0 else 0
                
                print(f"   • Expected copy size (100 layers × 4MB): {expected_copy:.1f} MB")
                print(f"   • Actual memory overhead: {actual_overhead:.1f} MB")
                print(f"   • Copy efficiency: {copy_ratio:.2f}x (1.0 = perfect)")
        
        print(f"\n4. Recommendations:")
        print(f"   • Use Arrow IPC for:")
        print(f"     - Large models (>50 layers)")
        print(f"     - Memory-constrained environments")
        print(f"     - Maximum performance requirements")
        print(f"   • Use Batch API for:")
        print(f"     - Small models (<10 layers)")
        print(f"     - Convenience over performance")
        print(f"     - When parallel processing benefit > copy cost")
        
        # Save detailed results
        output_file = "comprehensive_benchmark_results.json"
        results_dict = {
            "summary": {
                "average_speedup": float(avg_speedup),
                "average_memory_overhead_mb": float(avg_memory_overhead),
                "speedup_range": [float(min(speedups)), float(max(speedups))],
            },
            "detailed_results": [asdict(r) for r in self.results],
        }
        
        with open(output_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n  📊 Detailed results saved to: {output_file}")
        
        print("\n" + "="*80)


def main():
    """Run comprehensive benchmark."""
    if not ARROW_QUANT_AVAILABLE:
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE PERFORMANCE COMPARISON BENCHMARK")
    print("="*80)
    print("\nThis benchmark addresses Priority 2 tasks from")
    print("ZERO_COPY_IMPLEMENTATION_REVIEW.md:")
    print("  1. Run performance benchmarks")
    print("  2. Quantify data copy overhead")
    print("  3. Compare Batch API vs Arrow IPC")
    print("  4. Update documentation with findings")
    
    benchmark = ComprehensiveBenchmark()
    
    # Run all scenarios
    benchmark.run_all_scenarios()
    
    # Generate summary report
    benchmark.generate_summary_report()
    
    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()
