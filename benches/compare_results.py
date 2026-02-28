#!/usr/bin/env python3
"""
Compare benchmark results between two runs.

Supports both Criterion benchmarks and memory benchmarks.

Usage:
    # Compare Criterion benchmarks
    python benches/compare_results.py --current target/criterion/ --baseline target/criterion/ --threshold 10

    # Compare memory benchmarks
    python benches/compare_results.py baseline.json current.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any


def load_criterion_results(criterion_dir: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load Criterion benchmark results from directory.
    
    Args:
        criterion_dir: Path to Criterion output directory
        
    Returns:
        Dictionary mapping benchmark names to their results
    """
    results = {}
    
    # Find all benchmark directories
    if not criterion_dir.exists():
        return results
    
    for bench_dir in criterion_dir.iterdir():
        if not bench_dir.is_dir():
            continue
        
        # Look for estimates.json or new/estimates.json
        estimates_paths = [
            bench_dir / "base" / "estimates.json",
            bench_dir / "new" / "estimates.json",
            bench_dir / "estimates.json",
        ]
        
        for estimates_path in estimates_paths:
            if estimates_path.exists():
                try:
                    with open(estimates_path) as f:
                        data = json.load(f)
                        # Extract mean time in nanoseconds
                        if "mean" in data:
                            mean_ns = data["mean"]["point_estimate"]
                            results[bench_dir.name] = {
                                "mean_ns": mean_ns,
                                "mean_ms": mean_ns / 1_000_000,
                                "std_error": data["mean"].get("standard_error", 0),
                            }
                            break
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"Warning: Failed to parse {estimates_path}: {e}", file=sys.stderr)
    
    return results


def load_memory_results(path: Path) -> List[Dict]:
    """Load memory benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def compare_criterion_benchmarks(
    baseline: Dict[str, Dict[str, Any]],
    current: Dict[str, Dict[str, Any]],
    threshold: float = 10.0,
) -> Dict[str, Any]:
    """
    Compare Criterion benchmark results.
    
    Args:
        baseline: Baseline benchmark results
        current: Current benchmark results
        threshold: Regression threshold percentage
        
    Returns:
        Dictionary with comparison results and regression status
    """
    comparison = {}
    regressions = []
    improvements = []
    
    all_benchmarks = sorted(set(baseline.keys()) | set(current.keys()))
    
    for bench_name in all_benchmarks:
        baseline_result = baseline.get(bench_name)
        current_result = current.get(bench_name)
        
        if not baseline_result:
            comparison[bench_name] = {
                "status": "new",
                "current_ms": current_result["mean_ms"] if current_result else None,
            }
            continue
        
        if not current_result:
            comparison[bench_name] = {
                "status": "removed",
                "baseline_ms": baseline_result["mean_ms"],
            }
            continue
        
        # Calculate change
        baseline_ms = baseline_result["mean_ms"]
        current_ms = current_result["mean_ms"]
        diff_ms = current_ms - baseline_ms
        change_pct = (diff_ms / baseline_ms * 100) if baseline_ms > 0 else 0
        
        status = "ok"
        if change_pct > threshold:
            status = "regression"
            regressions.append((bench_name, change_pct))
        elif change_pct < -threshold:
            status = "improvement"
            improvements.append((bench_name, abs(change_pct)))
        
        comparison[bench_name] = {
            "status": status,
            "baseline_ms": baseline_ms,
            "current_ms": current_ms,
            "diff_ms": diff_ms,
            "change_percent": change_pct,
            "regression_percent": change_pct if change_pct > 0 else 0,
        }
    
    return {
        "benchmarks": comparison,
        "regressions": regressions,
        "improvements": improvements,
        "threshold": threshold,
    }


def print_criterion_comparison(results: Dict[str, Any]):
    """Print Criterion benchmark comparison in human-readable format."""
    print("=" * 80)
    print("Criterion Benchmark Comparison")
    print("=" * 80)
    print()
    
    benchmarks = results["benchmarks"]
    threshold = results["threshold"]
    
    for bench_name in sorted(benchmarks.keys()):
        data = benchmarks[bench_name]
        status = data["status"]
        
        print(f"\n{bench_name}")
        print("-" * 80)
        
        if status == "new":
            print(f"  ✨ NEW benchmark")
            print(f"  Current: {data['current_ms']:.3f} ms")
        elif status == "removed":
            print(f"  ⚠️  REMOVED benchmark")
            print(f"  Baseline: {data['baseline_ms']:.3f} ms")
        else:
            baseline_ms = data["baseline_ms"]
            current_ms = data["current_ms"]
            diff_ms = data["diff_ms"]
            change_pct = data["change_percent"]
            
            print(f"  Baseline: {baseline_ms:.3f} ms")
            print(f"  Current:  {current_ms:.3f} ms")
            print(f"  Change:   {diff_ms:+.3f} ms ({change_pct:+.2f}%)")
            
            if status == "regression":
                print(f"  ❌ REGRESSION: Performance degraded by {change_pct:.2f}%")
            elif status == "improvement":
                print(f"  ✅ IMPROVEMENT: Performance improved by {abs(change_pct):.2f}%")
            else:
                print(f"  ✓ OK: Change within threshold (±{threshold}%)")
    
    print()
    print("=" * 80)
    print("\nSummary:")
    print(f"  Total benchmarks: {len(benchmarks)}")
    print(f"  Regressions (>{threshold}%): {len(results['regressions'])}")
    print(f"  Improvements (>{threshold}%): {len(results['improvements'])}")
    
    if results["regressions"]:
        print(f"\n❌ Performance regressions detected:")
        for name, pct in results["regressions"]:
            print(f"  - {name}: {pct:.2f}% slower")
    
    if results["improvements"]:
        print(f"\n✅ Performance improvements:")
        for name, pct in results["improvements"]:
            print(f"  - {name}: {pct:.2f}% faster")
    
    print()


def generate_markdown_report(results: Dict[str, Any]) -> str:
    """Generate markdown report for benchmark comparison."""
    lines = ["# Benchmark Comparison Report", ""]
    
    benchmarks = results["benchmarks"]
    threshold = results["threshold"]
    
    # Summary
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Total benchmarks**: {len(benchmarks)}")
    lines.append(f"- **Regressions (>{threshold}%)**: {len(results['regressions'])}")
    lines.append(f"- **Improvements (>{threshold}%)**: {len(results['improvements'])}")
    lines.append("")
    
    # Regressions
    if results["regressions"]:
        lines.append("## ❌ Performance Regressions")
        lines.append("")
        lines.append("| Benchmark | Change | Status |")
        lines.append("|-----------|--------|--------|")
        for name, pct in sorted(results["regressions"], key=lambda x: x[1], reverse=True):
            lines.append(f"| `{name}` | +{pct:.2f}% | ❌ Slower |")
        lines.append("")
    
    # Improvements
    if results["improvements"]:
        lines.append("## ✅ Performance Improvements")
        lines.append("")
        lines.append("| Benchmark | Change | Status |")
        lines.append("|-----------|--------|--------|")
        for name, pct in sorted(results["improvements"], key=lambda x: x[1], reverse=True):
            lines.append(f"| `{name}` | -{pct:.2f}% | ✅ Faster |")
        lines.append("")
    
    # Detailed results
    lines.append("## Detailed Results")
    lines.append("")
    lines.append("| Benchmark | Baseline | Current | Change | Status |")
    lines.append("|-----------|----------|---------|--------|--------|")
    
    for bench_name in sorted(benchmarks.keys()):
        data = benchmarks[bench_name]
        status = data["status"]
        
        if status == "new":
            lines.append(f"| `{bench_name}` | - | {data['current_ms']:.3f} ms | NEW | ✨ |")
        elif status == "removed":
            lines.append(f"| `{bench_name}` | {data['baseline_ms']:.3f} ms | - | REMOVED | ⚠️ |")
        else:
            baseline_ms = data["baseline_ms"]
            current_ms = data["current_ms"]
            change_pct = data["change_percent"]
            
            if status == "regression":
                emoji = "❌"
            elif status == "improvement":
                emoji = "✅"
            else:
                emoji = "✓"
            
            lines.append(
                f"| `{bench_name}` | {baseline_ms:.3f} ms | {current_ms:.3f} ms | "
                f"{change_pct:+.2f}% | {emoji} |"
            )
    
    lines.append("")
    return "\n".join(lines)


def compare_memory_results(baseline: List[Dict], current: List[Dict]):
    """Compare two sets of benchmark results."""
    print("=" * 80)
    print("Memory Benchmark Comparison")
    print("=" * 80)
    print()
    
    # Group by model_size, bit_width, implementation
    baseline_map = {
        (r["model_size"], r["bit_width"], r["implementation"]): r
        for r in baseline
    }
    current_map = {
        (r["model_size"], r["bit_width"], r["implementation"]): r
        for r in current
    }
    
    # Find all keys
    all_keys = sorted(set(baseline_map.keys()) | set(current_map.keys()))
    
    for key in all_keys:
        model_size, bit_width, impl = key
        
        baseline_result = baseline_map.get(key)
        current_result = current_map.get(key)
        
        print(f"\n{model_size} INT{bit_width} ({impl.upper()})")
        print("-" * 80)
        
        if not baseline_result:
            print("  ⚠️  No baseline result")
            if current_result:
                print(f"  Current: {current_result['quantization_memory_mb']:.2f} MB")
            continue
        
        if not current_result:
            print("  ⚠️  No current result")
            print(f"  Baseline: {baseline_result['quantization_memory_mb']:.2f} MB")
            continue
        
        # Compare memory
        baseline_mem = baseline_result["quantization_memory_mb"]
        current_mem = current_result["quantization_memory_mb"]
        mem_diff = current_mem - baseline_mem
        mem_pct = (mem_diff / baseline_mem * 100) if baseline_mem > 0 else 0
        
        print(f"  Memory:")
        print(f"    Baseline: {baseline_mem:.2f} MB")
        print(f"    Current:  {current_mem:.2f} MB")
        print(f"    Change:   {mem_diff:+.2f} MB ({mem_pct:+.1f}%)")
        
        if mem_pct > 10:
            print(f"    ❌ Memory increased by >10%")
        elif mem_pct < -10:
            print(f"    ✅ Memory decreased by >10%")
        else:
            print(f"    ✓ Memory change within acceptable range")
        
        # Compare time
        baseline_time = baseline_result["quantization_time_s"]
        current_time = current_result["quantization_time_s"]
        time_diff = current_time - baseline_time
        time_pct = (time_diff / baseline_time * 100) if baseline_time > 0 else 0
        
        print(f"  Time:")
        print(f"    Baseline: {baseline_time:.2f}s")
        print(f"    Current:  {current_time:.2f}s")
        print(f"    Change:   {time_diff:+.2f}s ({time_pct:+.1f}%)")
        
        if time_pct > 10:
            print(f"    ❌ Time increased by >10%")
        elif time_pct < -10:
            print(f"    ✅ Time decreased by >10%")
        else:
            print(f"    ✓ Time change within acceptable range")
    
    print()
    print("=" * 80)
    
    # Summary
    print("\nSummary:")
    
    # Check Rust vs Python ratio
    rust_python_pairs = []
    for model_size in set(r["model_size"] for r in current):
        for bit_width in set(r["bit_width"] for r in current):
            rust = current_map.get((model_size, bit_width, "rust"))
            python = current_map.get((model_size, bit_width, "python"))
            
            if rust and python:
                ratio = rust["quantization_memory_mb"] / python["quantization_memory_mb"]
                rust_python_pairs.append((model_size, bit_width, ratio))
    
    if rust_python_pairs:
        print("\nRust vs Python Memory Ratios:")
        for model_size, bit_width, ratio in rust_python_pairs:
            status = "✅" if ratio < 0.50 else "❌"
            print(f"  {status} {model_size} INT{bit_width}: {ratio:.2%} (target: <50%)")
    
    print()


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results")
    parser.add_argument(
        "--current",
        type=Path,
        help="Current results (Criterion directory or JSON file)",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline results (Criterion directory or JSON file)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=10.0,
        help="Regression threshold percentage (default: 10%%)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output file for comparison results (JSON or markdown)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "markdown"],
        default="json",
        help="Output format (default: json)",
    )
    
    # Legacy positional arguments for memory benchmarks
    parser.add_argument(
        "baseline_legacy",
        nargs="?",
        type=Path,
        help="Baseline results JSON file (legacy)",
    )
    parser.add_argument(
        "current_legacy",
        nargs="?",
        type=Path,
        help="Current results JSON file (legacy)",
    )
    
    args = parser.parse_args()
    
    # Handle legacy positional arguments
    if args.baseline_legacy and args.current_legacy:
        baseline_path = args.baseline_legacy
        current_path = args.current_legacy
        
        if not baseline_path.exists():
            print(f"Error: Baseline file not found: {baseline_path}")
            sys.exit(1)
        
        if not current_path.exists():
            print(f"Error: Current file not found: {current_path}")
            sys.exit(1)
        
        baseline = load_memory_results(baseline_path)
        current = load_memory_results(current_path)
        compare_memory_results(baseline, current)
        return
    
    # Handle new --current/--baseline arguments
    if not args.current or not args.baseline:
        parser.error("Either provide --current and --baseline, or use legacy positional arguments")
    
    # Determine if we're comparing Criterion or memory benchmarks
    if args.current.is_dir() and args.baseline.is_dir():
        # Criterion benchmarks
        print("Loading Criterion benchmark results...")
        baseline_results = load_criterion_results(args.baseline)
        current_results = load_criterion_results(args.current)
        
        if not baseline_results and not current_results:
            print("Warning: No Criterion benchmark results found")
            sys.exit(0)
        
        print(f"Found {len(baseline_results)} baseline benchmarks")
        print(f"Found {len(current_results)} current benchmarks")
        print()
        
        comparison = compare_criterion_benchmarks(
            baseline_results,
            current_results,
            threshold=args.threshold,
        )
        
        # Print to console
        print_criterion_comparison(comparison)
        
        # Save output if requested
        if args.output:
            if args.format == "json":
                with open(args.output, "w") as f:
                    json.dump(comparison["benchmarks"], f, indent=2)
                print(f"Saved comparison results to {args.output}")
            elif args.format == "markdown":
                report = generate_markdown_report(comparison)
                with open(args.output, "w") as f:
                    f.write(report)
                print(f"Saved markdown report to {args.output}")
        
        # Exit with error if regressions detected
        if comparison["regressions"]:
            sys.exit(1)
    
    elif args.current.is_file() and args.baseline.is_file():
        # Memory benchmarks
        baseline = load_memory_results(args.baseline)
        current = load_memory_results(args.current)
        compare_memory_results(baseline, current)
    
    else:
        print("Error: Both current and baseline must be either directories (Criterion) or files (memory)")
        sys.exit(1)


if __name__ == "__main__":
    main()
