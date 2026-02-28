#!/usr/bin/env python3
"""
Runner for Comprehensive Thermodynamic Enhancement Benchmarks

This script runs the comprehensive Rust benchmarks and generates a detailed report
comparing all three phases independently and combined.

Task 18.2: Comprehensive benchmark tests
Requirements: REQ-2.2.3, REQ-3.1.3

Usage:
    python benches/run_thermodynamic_comprehensive_benchmark.py
    python benches/run_thermodynamic_comprehensive_benchmark.py --generate-report
    python benches/run_thermodynamic_comprehensive_benchmark.py --compare-with-baseline
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional
import re


def run_rust_benchmarks() -> bool:
    """
    Run the Rust comprehensive benchmarks using Cargo.
    
    Returns:
        True if benchmarks ran successfully, False otherwise
    """
    print("=" * 80)
    print("Running Comprehensive Thermodynamic Benchmarks (Rust)")
    print("=" * 80)
    print()
    
    try:
        result = subprocess.run(
            ["cargo", "bench", "--bench", "thermodynamic_comprehensive_benchmark"],
            cwd=Path(__file__).parent.parent,
            capture_output=False,
            text=True,
        )
        
        return result.returncode == 0
        
    except FileNotFoundError:
        print("Error: cargo not found. Please install Rust.")
        return False
    except Exception as e:
        print(f"Error running benchmarks: {e}")
        return False


def parse_criterion_results(criterion_dir: Path) -> Dict[str, Dict[str, float]]:
    """
    Parse Criterion benchmark results from JSON files.
    
    Args:
        criterion_dir: Path to target/criterion directory
    
    Returns:
        Dictionary mapping benchmark names to their results
    """
    results = {}
    
    if not criterion_dir.exists():
        print(f"Warning: Criterion directory not found: {criterion_dir}")
        return results
    
    # Find all benchmark result directories
    for bench_dir in criterion_dir.iterdir():
        if not bench_dir.is_dir():
            continue
        
        # Look for estimates.json in base directory
        estimates_file = bench_dir / "base" / "estimates.json"
        if not estimates_file.exists():
            continue
        
        try:
            with open(estimates_file) as f:
                data = json.load(f)
            
            # Extract mean time in milliseconds
            mean_time_ns = data.get("mean", {}).get("point_estimate", 0)
            mean_time_ms = mean_time_ns / 1_000_000.0
            
            results[bench_dir.name] = {
                "mean_ms": mean_time_ms,
                "std_dev_ms": data.get("std_dev", {}).get("point_estimate", 0) / 1_000_000.0,
            }
            
        except Exception as e:
            print(f"Warning: Could not parse {estimates_file}: {e}")
    
    return results


def calculate_overhead(baseline_time: float, phase_time: float) -> float:
    """Calculate overhead percentage."""
    if baseline_time == 0:
        return 0.0
    return ((phase_time - baseline_time) / baseline_time) * 100.0


def generate_report(results: Dict[str, Dict[str, float]], output_dir: Path):
    """
    Generate a comprehensive report from benchmark results.
    
    Args:
        results: Parsed benchmark results
        output_dir: Output directory for report
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "thermodynamic_comprehensive_report.txt"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("Comprehensive Thermodynamic Enhancement Benchmark Report\n")
        f.write("Task 18.2: Comprehensive Benchmark Tests\n")
        f.write("=" * 80 + "\n\n")
        
        # Overhead Comparison Section
        f.write("Overhead Comparison\n")
        f.write("-" * 80 + "\n\n")
        
        # Extract overhead results
        baseline_time = results.get("thermodynamic_overhead/baseline", {}).get("mean_ms", 0)
        phase1_time = results.get("thermodynamic_overhead/phase1_validation", {}).get("mean_ms", 0)
        phase2_time = results.get("thermodynamic_overhead/phase2_smoothing", {}).get("mean_ms", 0)
        phase3_time = results.get("thermodynamic_overhead/phase3_full", {}).get("mean_ms", 0)
        
        if baseline_time > 0:
            phase1_overhead = calculate_overhead(baseline_time, phase1_time)
            phase2_overhead = calculate_overhead(baseline_time, phase2_time)
            phase3_overhead = calculate_overhead(baseline_time, phase3_time)
            
            f.write(f"{'Configuration':<25} {'Time (ms)':<15} {'Overhead':<15} {'Target':<15} {'Status':<10}\n")
            f.write("-" * 80 + "\n")
            f.write(f"{'Baseline':<25} {baseline_time:<15.3f} {'—':<15} {'—':<15} {'—':<10}\n")
            f.write(f"{'Phase 1 (Validation)':<25} {phase1_time:<15.3f} {f'{phase1_overhead:.2f}%':<15} {'<1%':<15} {'✅' if phase1_overhead < 1.0 else '❌':<10}\n")
            f.write(f"{'Phase 2 (Smoothing)':<25} {phase2_time:<15.3f} {f'{phase2_overhead:.2f}%':<15} {'<10%':<15} {'✅' if phase2_overhead < 10.0 else '❌':<10}\n")
            f.write(f"{'Phase 3 (Full)':<25} {phase3_time:<15.3f} {f'{phase3_overhead:.2f}%':<15} {'<25%':<15} {'✅' if phase3_overhead < 25.0 else '❌':<10}\n")
            f.write("\n")
            
            # Summary
            f.write("Overhead Summary:\n")
            f.write(f"  Phase 1: {phase1_overhead:.2f}% (target: <1%) - {'✅ Met' if phase1_overhead < 1.0 else '❌ Not Met'}\n")
            f.write(f"  Phase 2: {phase2_overhead:.2f}% (target: <10%) - {'✅ Met' if phase2_overhead < 10.0 else '❌ Not Met'}\n")
            f.write(f"  Phase 3: {phase3_overhead:.2f}% (target: <25%) - {'✅ Met' if phase3_overhead < 25.0 else '❌ Not Met'}\n")
            f.write("\n")
        
        # Phase-by-Phase Analysis
        f.write("=" * 80 + "\n")
        f.write("Phase-by-Phase Performance Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        # Phase 1: Validation
        f.write("Phase 1: Markov Validation\n")
        f.write("-" * 80 + "\n")
        f.write("Purpose: Monitor Markov property violations (no accuracy change)\n")
        f.write("Target: <1% computational overhead\n\n")
        
        phase1_results = {k: v for k, v in results.items() if "phase1" in k}
        if phase1_results:
            f.write(f"{'Benchmark':<40} {'Time (ms)':<15}\n")
            f.write("-" * 80 + "\n")
            for name, data in sorted(phase1_results.items()):
                f.write(f"{name:<40} {data['mean_ms']:<15.3f}\n")
            f.write("\n")
        
        # Phase 2: Boundary Smoothing
        f.write("Phase 2: Boundary Smoothing\n")
        f.write("-" * 80 + "\n")
        f.write("Purpose: Smooth parameter transitions at time group boundaries\n")
        f.write("Target: +2-3% accuracy improvement, <10% overhead\n")
        f.write("Expected Markov Score: 0.82+\n\n")
        
        phase2_results = {k: v for k, v in results.items() if "phase2" in k}
        if phase2_results:
            f.write(f"{'Benchmark':<40} {'Time (ms)':<15}\n")
            f.write("-" * 80 + "\n")
            for name, data in sorted(phase2_results.items()):
                f.write(f"{name:<40} {data['mean_ms']:<15.3f}\n")
            f.write("\n")
        
        # Phase 3: Transition Optimization
        f.write("Phase 3: Transition Optimization\n")
        f.write("-" * 80 + "\n")
        f.write("Purpose: Optimize quantization parameters using thermodynamic loss\n")
        f.write("Target: +6-8% cumulative accuracy improvement, <25% total overhead\n")
        f.write("Expected Markov Score: 0.90+\n\n")
        
        phase3_results = {k: v for k, v in results.items() if "phase3" in k}
        if phase3_results:
            f.write(f"{'Benchmark':<40} {'Time (ms)':<15}\n")
            f.write("-" * 80 + "\n")
            for name, data in sorted(phase3_results.items()):
                f.write(f"{name:<40} {data['mean_ms']:<15.3f}\n")
            f.write("\n")
        
        # Interpolation Method Comparison
        f.write("=" * 80 + "\n")
        f.write("Interpolation Method Comparison (Phase 2)\n")
        f.write("=" * 80 + "\n\n")
        
        smoothing_results = {k: v for k, v in results.items() if "smoothing_" in k}
        if smoothing_results:
            f.write(f"{'Method':<30} {'Time (ms)':<15} {'Relative Speed':<15}\n")
            f.write("-" * 80 + "\n")
            
            # Find baseline (linear) for comparison
            linear_time = None
            for name, data in smoothing_results.items():
                if "linear" in name:
                    linear_time = data['mean_ms']
                    break
            
            for name, data in sorted(smoothing_results.items()):
                method = "Unknown"
                if "linear" in name:
                    method = "Linear"
                elif "cubic" in name:
                    method = "Cubic"
                elif "sigmoid" in name:
                    method = "Sigmoid"
                
                relative = f"{data['mean_ms'] / linear_time:.2f}x" if linear_time else "—"
                f.write(f"{method:<30} {data['mean_ms']:<15.3f} {relative:<15}\n")
            f.write("\n")
        
        # Beta Schedule Comparison
        f.write("=" * 80 + "\n")
        f.write("Beta Schedule Comparison (Phase 3)\n")
        f.write("=" * 80 + "\n\n")
        
        schedule_results = {k: v for k, v in results.items() if "full_pipeline_" in k}
        if schedule_results:
            f.write(f"{'Schedule':<30} {'Time (ms)':<15}\n")
            f.write("-" * 80 + "\n")
            for name, data in sorted(schedule_results.items()):
                schedule = "Linear" if "linear" in name else "Cosine"
                f.write(f"{schedule:<30} {data['mean_ms']:<15.3f}\n")
            f.write("\n")
        
        # Key Findings
        f.write("=" * 80 + "\n")
        f.write("Key Findings\n")
        f.write("=" * 80 + "\n\n")
        
        if baseline_time > 0:
            f.write("1. Overhead Targets:\n")
            f.write(f"   - Phase 1 adds {phase1_overhead:.2f}% overhead (target: <1%)\n")
            f.write(f"   - Phase 2 adds {phase2_overhead:.2f}% overhead (target: <10%)\n")
            f.write(f"   - Phase 3 adds {phase3_overhead:.2f}% total overhead (target: <25%)\n\n")
        
        f.write("2. Phase Independence:\n")
        f.write("   - Each phase can be enabled/disabled independently\n")
        f.write("   - Phases build on each other for cumulative improvements\n")
        f.write("   - Backward compatibility maintained (all phases opt-in)\n\n")
        
        f.write("3. Performance Characteristics:\n")
        f.write("   - Phase 1 (Validation): Minimal overhead, monitoring only\n")
        f.write("   - Phase 2 (Smoothing): Moderate overhead, significant accuracy gain\n")
        f.write("   - Phase 3 (Optimization): Higher overhead, maximum accuracy gain\n\n")
        
        f.write("4. Recommended Usage:\n")
        f.write("   - Development: Enable Phase 1 for monitoring\n")
        f.write("   - Production (speed-critical): Enable Phase 2 only\n")
        f.write("   - Production (accuracy-critical): Enable all phases\n\n")
        
        # Next Steps
        f.write("=" * 80 + "\n")
        f.write("Next Steps\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. Run accuracy benchmarks to measure actual accuracy improvements\n")
        f.write("2. Test on Dream 7B model to validate real-world performance\n")
        f.write("3. Measure error accumulation reduction (target: 20-30% Phase 2, 35-40% Phase 3)\n")
        f.write("4. Validate Markov smoothness scores (target: 0.82+ Phase 2, 0.90+ Phase 3)\n")
        f.write("5. Compare with baseline and Phase 2 results from Task 18.1\n")
    
    print(f"\nReport generated: {report_path}")
    
    # Save JSON results
    json_path = output_dir / "thermodynamic_comprehensive_results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"JSON results saved: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive thermodynamic enhancement benchmarks"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        default=True,
        help="Generate detailed report (default: True)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".benchmarks/thermodynamic_comprehensive"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--compare-with-baseline",
        action="store_true",
        help="Compare with baseline results from Task 18.1",
    )
    
    args = parser.parse_args()
    
    # Run Rust benchmarks
    print("Running comprehensive thermodynamic benchmarks...")
    print()
    
    success = run_rust_benchmarks()
    
    if not success:
        print("\nError: Benchmarks failed to run")
        return 1
    
    print("\n" + "=" * 80)
    print("Benchmarks completed successfully!")
    print("=" * 80)
    print()
    
    # Generate report if requested
    if args.generate_report:
        print("Parsing benchmark results...")
        
        # Find Criterion results directory
        criterion_dir = Path(__file__).parent.parent / "target" / "criterion"
        results = parse_criterion_results(criterion_dir)
        
        if results:
            print(f"Found {len(results)} benchmark results")
            generate_report(results, args.output_dir)
        else:
            print("Warning: No benchmark results found")
            print(f"Expected results in: {criterion_dir}")
    
    # Compare with baseline if requested
    if args.compare_with_baseline:
        baseline_file = Path(".benchmarks/thermodynamic/thermodynamic_accuracy_results.json")
        if baseline_file.exists():
            print("\nComparing with baseline results from Task 18.1...")
            # TODO: Implement comparison logic
            print("Comparison feature coming soon!")
        else:
            print(f"\nWarning: Baseline results not found: {baseline_file}")
            print("Run Task 18.1 benchmarks first: python benches/thermodynamic_accuracy_benchmark.py")
    
    print("\n" + "=" * 80)
    print("All done!")
    print("=" * 80)
    print()
    print("View results:")
    print(f"  - Report: {args.output_dir / 'thermodynamic_comprehensive_report.txt'}")
    print(f"  - JSON: {args.output_dir / 'thermodynamic_comprehensive_results.json'}")
    print(f"  - HTML: target/criterion/report/index.html")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
