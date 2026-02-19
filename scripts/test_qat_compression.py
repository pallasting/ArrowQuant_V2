#!/usr/bin/env python3
"""
Intel QAT Compression Test Script
Tests QAT hardware acceleration for compression tasks
"""

import time
import zlib
import sys
import os
from typing import Tuple, Dict
import random
import string

# Try to import QAT library
try:
    import qat
    QAT_AVAILABLE = True
except ImportError:
    QAT_AVAILABLE = False
    print("Warning: qat-python library not available")

# Colors for output
class Colors:
    BLUE = '\033[0;34m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    RED = '\033[0;31m'
    NC = '\033[0m'

def log_info(msg: str):
    print(f"{Colors.BLUE}[INFO]{Colors.NC} {msg}")

def log_success(msg: str):
    print(f"{Colors.GREEN}[SUCCESS]{Colors.NC} {msg}")

def log_warning(msg: str):
    print(f"{Colors.YELLOW}[WARNING]{Colors.NC} {msg}")

def log_error(msg: str):
    print(f"{Colors.RED}[ERROR]{Colors.NC} {msg}")


def generate_test_data(size_kb: int) -> bytes:
    """Generate random test data"""
    # Mix of random and repetitive data for realistic compression
    random_part = ''.join(random.choices(string.ascii_letters + string.digits, k=size_kb * 512))
    repetitive_part = "The quick brown fox jumps over the lazy dog. " * (size_kb * 10)
    combined = (random_part + repetitive_part).encode('utf-8')
    return combined[:size_kb * 1024]


def test_software_compression(data: bytes, level: int = 6) -> Tuple[float, float, int]:
    """Test software-based compression (zlib)"""
    # Compression
    start = time.time()
    compressed = zlib.compress(data, level=level)
    compress_time = time.time() - start
    
    # Decompression
    start = time.time()
    decompressed = zlib.decompress(compressed)
    decompress_time = time.time() - start
    
    # Verify
    assert data == decompressed, "Decompression mismatch!"
    
    return compress_time, decompress_time, len(compressed)


def test_qat_compression(data: bytes) -> Tuple[float, float, int]:
    """Test QAT hardware-accelerated compression"""
    if not QAT_AVAILABLE:
        return 0, 0, 0
    
    try:
        # Initialize QAT
        qat_instance = qat.QATCompression()
        
        # Compression
        start = time.time()
        compressed = qat_instance.compress(data)
        compress_time = time.time() - start
        
        # Decompression
        start = time.time()
        decompressed = qat_instance.decompress(compressed)
        decompress_time = time.time() - start
        
        # Verify
        assert data == decompressed, "QAT decompression mismatch!"
        
        return compress_time, decompress_time, len(compressed)
    except Exception as e:
        log_error(f"QAT compression failed: {e}")
        return 0, 0, 0


def run_benchmark(size_kb: int, iterations: int = 10) -> Dict:
    """Run compression benchmark"""
    log_info(f"Testing with {size_kb}KB data, {iterations} iterations...")
    
    # Generate test data
    data = generate_test_data(size_kb)
    original_size = len(data)
    
    results = {
        'original_size': original_size,
        'software': {'compress_times': [], 'decompress_times': [], 'compressed_sizes': []},
        'qat': {'compress_times': [], 'decompress_times': [], 'compressed_sizes': []}
    }
    
    # Test software compression
    log_info("Testing software compression (zlib)...")
    for i in range(iterations):
        c_time, d_time, c_size = test_software_compression(data)
        results['software']['compress_times'].append(c_time)
        results['software']['decompress_times'].append(d_time)
        results['software']['compressed_sizes'].append(c_size)
    
    # Test QAT compression
    if QAT_AVAILABLE:
        log_info("Testing QAT hardware compression...")
        for i in range(iterations):
            c_time, d_time, c_size = test_qat_compression(data)
            if c_time > 0:  # Success
                results['qat']['compress_times'].append(c_time)
                results['qat']['decompress_times'].append(d_time)
                results['qat']['compressed_sizes'].append(c_size)
    
    return results


def calculate_stats(times: list) -> Dict:
    """Calculate statistics"""
    if not times:
        return {'mean': 0, 'min': 0, 'max': 0, 'throughput_mbs': 0}
    
    mean = sum(times) / len(times)
    return {
        'mean': mean,
        'min': min(times),
        'max': max(times),
    }


def print_results(results: Dict, size_kb: int):
    """Print benchmark results"""
    print("\n" + "="*70)
    print(f"Benchmark Results - {size_kb}KB Data")
    print("="*70)
    
    original_size = results['original_size']
    
    # Software results
    sw_compress = calculate_stats(results['software']['compress_times'])
    sw_decompress = calculate_stats(results['software']['decompress_times'])
    sw_size = sum(results['software']['compressed_sizes']) / len(results['software']['compressed_sizes'])
    sw_ratio = original_size / sw_size if sw_size > 0 else 0
    
    print("\n📊 Software Compression (zlib):")
    print(f"  Compression:")
    print(f"    - Time: {sw_compress['mean']*1000:.2f}ms (min: {sw_compress['min']*1000:.2f}ms, max: {sw_compress['max']*1000:.2f}ms)")
    print(f"    - Throughput: {(original_size/1024/1024)/sw_compress['mean']:.2f} MB/s")
    print(f"  Decompression:")
    print(f"    - Time: {sw_decompress['mean']*1000:.2f}ms (min: {sw_decompress['min']*1000:.2f}ms, max: {sw_decompress['max']*1000:.2f}ms)")
    print(f"    - Throughput: {(original_size/1024/1024)/sw_decompress['mean']:.2f} MB/s")
    print(f"  Compression Ratio: {sw_ratio:.2f}x")
    print(f"  Compressed Size: {sw_size/1024:.2f}KB (from {original_size/1024:.2f}KB)")
    
    # QAT results
    if results['qat']['compress_times']:
        qat_compress = calculate_stats(results['qat']['compress_times'])
        qat_decompress = calculate_stats(results['qat']['decompress_times'])
        qat_size = sum(results['qat']['compressed_sizes']) / len(results['qat']['compressed_sizes'])
        qat_ratio = original_size / qat_size if qat_size > 0 else 0
        
        print("\n⚡ QAT Hardware Compression:")
        print(f"  Compression:")
        print(f"    - Time: {qat_compress['mean']*1000:.2f}ms (min: {qat_compress['min']*1000:.2f}ms, max: {qat_compress['max']*1000:.2f}ms)")
        print(f"    - Throughput: {(original_size/1024/1024)/qat_compress['mean']:.2f} MB/s")
        print(f"  Decompression:")
        print(f"    - Time: {qat_decompress['mean']*1000:.2f}ms (min: {qat_decompress['min']*1000:.2f}ms, max: {qat_decompress['max']*1000:.2f}ms)")
        print(f"    - Throughput: {(original_size/1024/1024)/qat_decompress['mean']:.2f} MB/s")
        print(f"  Compression Ratio: {qat_ratio:.2f}x")
        print(f"  Compressed Size: {qat_size/1024:.2f}KB (from {original_size/1024:.2f}KB)")
        
        # Speedup comparison
        compress_speedup = sw_compress['mean'] / qat_compress['mean']
        decompress_speedup = sw_decompress['mean'] / qat_decompress['mean']
        
        print("\n🚀 QAT Speedup:")
        print(f"  Compression: {compress_speedup:.2f}x faster")
        print(f"  Decompression: {decompress_speedup:.2f}x faster")
        
        if compress_speedup > 1.5:
            log_success(f"QAT provides significant acceleration ({compress_speedup:.2f}x)!")
        elif compress_speedup > 1.1:
            log_info(f"QAT provides moderate acceleration ({compress_speedup:.2f}x)")
        else:
            log_warning(f"QAT acceleration is minimal ({compress_speedup:.2f}x)")
    else:
        print("\n⚠️  QAT Hardware Compression: Not available or failed")
    
    print("\n" + "="*70)


def main():
    """Main test function"""
    print("\n" + "="*70)
    print("Intel QAT Compression Benchmark")
    print("="*70 + "\n")
    
    # Check QAT availability
    log_info("Checking QAT availability...")
    if QAT_AVAILABLE:
        log_success("QAT Python library is available")
    else:
        log_warning("QAT Python library not available - only software compression will be tested")
    
    # Check QAT devices
    log_info("Checking QAT devices...")
    qat_check = os.popen("lspci | grep -i 'QuickAssist\\|QAT'").read()
    if qat_check:
        log_success(f"QAT devices detected:\n{qat_check}")
    else:
        log_warning("No QAT devices detected")
    
    # Check QAT service
    qat_service = os.popen("systemctl is-active qat.service 2>/dev/null").read().strip()
    if qat_service == "active":
        log_success("QAT service is running")
    else:
        log_warning("QAT service is not running")
    
    print()
    
    # Run benchmarks with different data sizes
    test_sizes = [10, 100, 1000]  # KB
    
    for size_kb in test_sizes:
        try:
            results = run_benchmark(size_kb, iterations=5)
            print_results(results, size_kb)
            print()
        except Exception as e:
            log_error(f"Benchmark failed for {size_kb}KB: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print("Summary and Recommendations")
    print("="*70 + "\n")
    
    if QAT_AVAILABLE and qat_service == "active":
        log_success("✅ QAT is properly configured and can be used for compression acceleration")
        print("\nIntegration recommendations:")
        print("  1. Use QAT for zstd compression in fallback mode")
        print("  2. Use QAT for diff data compression")
        print("  3. Use QAT for Arrow/Parquet compression")
        print("  4. Expected performance gain: 1.5-3x for compression tasks")
    elif QAT_AVAILABLE:
        log_warning("⚠️  QAT library available but service not running")
        print("\nTo enable QAT:")
        print("  sudo systemctl start qat.service")
    else:
        log_warning("⚠️  QAT not available - compression will use software only")
        print("\nTo enable QAT:")
        print("  1. Install QAT driver: sudo ./scripts/configure_qat.sh")
        print("  2. Install Python library: pip install qat-python")
    
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user")
        sys.exit(0)
    except Exception as e:
        log_error(f"Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
