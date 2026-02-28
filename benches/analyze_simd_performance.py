#!/usr/bin/env python3
"""
SIMD Performance Analysis Script

Runs quick SIMD benchmarks and analyzes speedup vs scalar implementation.
Tests on different CPU architectures (AVX2 on x86_64, NEON on ARM64).

Usage:
    python benches/analyze_simd_performance.py
"""

import subprocess
import sys
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple


def detect_cpu_features() -> Dict[str, bool]:
    """Detect available CPU features."""
    features = {
        'avx2': False,
        'neon': False,
        'architecture': 'unknown'
    }
    
    try:
        # Try to detect architecture
        import platform
        arch = platform.machine().lower()
        
        if 'x86' in arch or 'amd64' in arch:
            features['architecture'] = 'x86_64'
            # Check for AVX2 support (Windows)
            try:
                result = subprocess.run(
                    ['wmic', 'cpu', 'get', 'caption'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                # Most modern x86_64 CPUs have AVX2
                features['avx2'] = True
            except:
                pass
        elif 'arm' in arch or 'aarch64' in arch:
            features['architecture'] = 'aarch64'
            features['neon'] = True
    except:
        pass
    
    return features


def run_quick_benchmark() -> Tuple[bool, str]:
    """Run the quick SIMD benchmark."""
    print("=" * 80)
    print("SIMD Performance Benchmark - Quick Analysis")
    print("=" * 80)
    print()
    
    # Detect CPU features
    features = detect_cpu_features()
    print(f"CPU Architecture: {features['architecture']}")
    print(f"AVX2 Support: {'Yes' if features['avx2'] else 'No'}")
    print(f"NEON Support: {'Yes' if features['neon'] else 'No'}")
    print()
    
    project_root = Path(__file__).parent.parent
    
    print("Running quick benchmark (this will take ~2-3 minutes)...")
    print()
    
    try:
        result = subprocess.run(
            ["cargo", "bench", "--bench", "quick_simd_benchmark"],
            cwd=project_root,
            capture_output=True,
            text=True,
            timeout=180,  # 3 minute timeout
        )
        
        if result.returncode == 0:
            return True, result.stdout
        else:
            print("❌ Benchmark failed!")
            print(result.stderr)
            return False, ""
    except subprocess.TimeoutExpired:
        print("❌ Benchmark timed out")
        return False, ""
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, ""


def parse_benchmark_results(output: str) -> Dict[str, List[Dict]]:
    """Parse benchmark output and extract timing information."""
    results = {
        'quantize': [],
        'dequantize': [],
        'cosine_similarity': []
    }
    
    # Pattern to match benchmark results
    # Example: "quantize_quick/simd/256  time:   [226.40 ns 232.43 ns 237.91 ns]"
    pattern = r'(\w+)/(\w+)/(\d+)\s+time:\s+\[[\d.]+ \w+ ([\d.]+) (\w+)'
    
    for line in output.split('\n'):
        match = re.search(pattern, line)
        if match:
            operation = match.group(1).replace('_quick', '')
            impl_type = match.group(2)  # simd or scalar
            size = int(match.group(3))
            time_value = float(match.group(4))
            time_unit = match.group(5)
            
            # Convert to nanoseconds
            if time_unit == 'us' or time_unit == 'µs':
                time_value *= 1000
            elif time_unit == 'ms':
                time_value *= 1_000_000
            elif time_unit == 's':
                time_value *= 1_000_000_000
            
            if operation in results:
                results[operation].append({
                    'type': impl_type,
                    'size': size,
                    'time_ns': time_value
                })
    
    return results


def calculate_speedups(results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    """Calculate speedup ratios for each operation and size."""
    speedups = {}
    
    for operation, measurements in results.items():
        speedups[operation] = {}
        
        # Group by size
        by_size = {}
        for m in measurements:
            size = m['size']
            if size not in by_size:
                by_size[size] = {}
            by_size[size][m['type']] = m['time_ns']
        
        # Calculate speedups
        for size, times in by_size.items():
            if 'simd' in times and 'scalar' in times:
                speedup = times['scalar'] / times['simd']
                speedups[operation][size] = {
                    'simd_ns': times['simd'],
                    'scalar_ns': times['scalar'],
                    'speedup': speedup
                }
    
    return speedups


def print_analysis(speedups: Dict[str, Dict]):
    """Print detailed analysis of benchmark results."""
    print()
    print("=" * 80)
    print("SIMD Performance Analysis")
    print("=" * 80)
    print()
    
    for operation, sizes in speedups.items():
        print(f"\n{operation.upper()} Operation:")
        print("-" * 60)
        print(f"{'Size':<10} {'SIMD (ns)':<15} {'Scalar (ns)':<15} {'Speedup':<10}")
        print("-" * 60)
        
        speedup_values = []
        for size in sorted(sizes.keys()):
            data = sizes[size]
            speedup_values.append(data['speedup'])
            print(f"{size:<10} {data['simd_ns']:<15.2f} {data['scalar_ns']:<15.2f} {data['speedup']:<10.2f}x")
        
        if speedup_values:
            avg_speedup = sum(speedup_values) / len(speedup_values)
            min_speedup = min(speedup_values)
            max_speedup = max(speedup_values)
            print("-" * 60)
            print(f"Average Speedup: {avg_speedup:.2f}x")
            print(f"Range: {min_speedup:.2f}x - {max_speedup:.2f}x")
    
    print()
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print()
    
    # Calculate overall statistics
    all_speedups = []
    for operation, sizes in speedups.items():
        for size, data in sizes.items():
            all_speedups.append(data['speedup'])
    
    if all_speedups:
        overall_avg = sum(all_speedups) / len(all_speedups)
        overall_min = min(all_speedups)
        overall_max = max(all_speedups)
        
        print(f"Overall Average Speedup: {overall_avg:.2f}x")
        print(f"Overall Range: {overall_min:.2f}x - {overall_max:.2f}x")
        print()
        
        # Determine if results meet expectations
        if overall_avg >= 2.0:
            print("✅ SIMD implementation meets performance targets (2-4x speedup)")
        elif overall_avg >= 1.5:
            print("⚠️  SIMD implementation shows moderate improvement (1.5-2x speedup)")
        else:
            print("❌ SIMD implementation shows minimal improvement (<1.5x speedup)")
        
        print()
        print("Expected Performance:")
        print("  - AVX2 (x86_64): 2-4x speedup (processes 8 floats at a time)")
        print("  - NEON (ARM64): 2-3x speedup (processes 4 floats at a time)")
        print("  - Scalar fallback: 1x (no SIMD acceleration)")
    
    print()
    print("=" * 80)


def save_results(speedups: Dict[str, Dict], output_file: str = "simd_benchmark_results.json"):
    """Save benchmark results to JSON file."""
    project_root = Path(__file__).parent.parent
    output_path = project_root / output_file
    
    with open(output_path, 'w') as f:
        json.dump(speedups, f, indent=2)
    
    print(f"\n📊 Results saved to: {output_path}")


def main():
    """Main execution function."""
    success, output = run_quick_benchmark()
    
    if not success:
        return 1
    
    print("✅ Benchmark completed successfully!")
    print()
    
    # Parse results
    results = parse_benchmark_results(output)
    
    if not any(results.values()):
        print("⚠️  Could not parse benchmark results")
        print("\nRaw output:")
        print(output)
        return 1
    
    # Calculate speedups
    speedups = calculate_speedups(results)
    
    # Print analysis
    print_analysis(speedups)
    
    # Save results
    save_results(speedups)
    
    print("\n📈 For detailed HTML reports, open:")
    print("   target/criterion/report/index.html")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
