#!/usr/bin/env python3
"""
Multimodal Encoder Performance Benchmark

Measures performance metrics for vision and audio encoders:
- Model loading time
- Single encoding latency
- Batch throughput
- Memory usage
"""

import sys
import time
import gc
from pathlib import Path
from typing import Dict, List, Any
import json

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger
from llm_compression.multimodal.vision_encoder import VisionEncoder
from llm_compression.multimodal.audio_encoder import AudioEncoder


class PerformanceBenchmark:
    """Performance benchmark suite for multimodal encoders."""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results = {}
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def benchmark_loading_time(
        self,
        encoder_class,
        model_path: str,
        encoder_type: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Benchmark model loading time.
        
        Args:
            encoder_class: Encoder class to instantiate
            model_path: Path to model directory
            encoder_type: Type of encoder ('vision' or 'audio')
            **kwargs: Additional arguments for encoder
            
        Returns:
            Dictionary with loading metrics
        """
        logger.info(f"Benchmarking {encoder_type} encoder loading time...")
        
        # Clear memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        mem_before = self.get_memory_usage()
        
        # Measure loading time
        start_time = time.perf_counter()
        encoder = encoder_class(model_path, **kwargs)
        load_time = (time.perf_counter() - start_time) * 1000  # Convert to ms
        
        mem_after = self.get_memory_usage()
        mem_used = mem_after - mem_before
        
        return {
            "load_time_ms": load_time,
            "memory_mb": mem_used,
            "encoder": encoder
        }
    
    def benchmark_single_encoding(
        self,
        encoder,
        test_data: Any,
        encoder_type: str,
        warmup_runs: int = 3,
        test_runs: int = 10
    ) -> Dict[str, float]:
        """
        Benchmark single encoding latency.
        
        Args:
            encoder: Encoder instance
            test_data: Test data (image or audio)
            encoder_type: Type of encoder
            warmup_runs: Number of warmup runs
            test_runs: Number of test runs
            
        Returns:
            Dictionary with latency metrics
        """
        logger.info(f"Benchmarking {encoder_type} single encoding latency...")
        
        # Warmup
        for _ in range(warmup_runs):
            _ = encoder.encode(test_data)
        
        # Benchmark
        latencies = []
        for _ in range(test_runs):
            start_time = time.perf_counter()
            _ = encoder.encode(test_data)
            latency = (time.perf_counter() - start_time) * 1000  # ms
            latencies.append(latency)
        
        return {
            "mean_latency_ms": np.mean(latencies),
            "std_latency_ms": np.std(latencies),
            "min_latency_ms": np.min(latencies),
            "max_latency_ms": np.max(latencies),
            "p50_latency_ms": np.percentile(latencies, 50),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
        }
    
    def benchmark_batch_throughput(
        self,
        encoder,
        test_data_batch: List[Any],
        encoder_type: str,
        duration_seconds: float = 5.0
    ) -> Dict[str, float]:
        """
        Benchmark batch encoding throughput.
        
        Args:
            encoder: Encoder instance
            test_data_batch: Batch of test data
            encoder_type: Type of encoder
            duration_seconds: Duration to run benchmark
            
        Returns:
            Dictionary with throughput metrics
        """
        logger.info(f"Benchmarking {encoder_type} batch throughput...")
        
        batch_size = len(test_data_batch)
        total_encoded = 0
        start_time = time.perf_counter()
        
        while (time.perf_counter() - start_time) < duration_seconds:
            for data in test_data_batch:
                _ = encoder.encode(data)
                total_encoded += 1
        
        elapsed = time.perf_counter() - start_time
        throughput = total_encoded / elapsed
        
        return {
            "throughput_per_sec": throughput,
            "total_encoded": total_encoded,
            "duration_sec": elapsed,
            "batch_size": batch_size
        }
    
    def benchmark_vision_encoder(
        self,
        model_path: str,
        image_size: int = 224
    ) -> Dict[str, Any]:
        """Benchmark vision encoder performance."""
        logger.info("="*70)
        logger.info("Vision Encoder Benchmark")
        logger.info("="*70)
        
        results = {}
        
        # 1. Loading time
        load_metrics = self.benchmark_loading_time(
            VisionEncoder,
            model_path,
            "vision"
        )
        results["loading"] = {
            "time_ms": load_metrics["load_time_ms"],
            "memory_mb": load_metrics["memory_mb"]
        }
        encoder = load_metrics["encoder"]
        
        # Generate test image
        test_image = np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
        
        # 2. Single encoding latency
        latency_metrics = self.benchmark_single_encoding(
            encoder,
            test_image,
            "vision"
        )
        results["latency"] = latency_metrics
        
        # 3. Batch throughput
        test_batch = [
            np.random.randint(0, 256, (image_size, image_size, 3), dtype=np.uint8)
            for _ in range(10)
        ]
        throughput_metrics = self.benchmark_batch_throughput(
            encoder,
            test_batch,
            "vision"
        )
        results["throughput"] = throughput_metrics
        
        # 4. Peak memory usage
        results["peak_memory_mb"] = self.get_memory_usage()
        
        return results
    
    def benchmark_audio_encoder(
        self,
        model_path: str,
        sample_rate: int = 16000,
        duration: float = 3.0
    ) -> Dict[str, Any]:
        """Benchmark audio encoder performance."""
        logger.info("="*70)
        logger.info("Audio Encoder Benchmark")
        logger.info("="*70)
        
        results = {}
        
        # Load config from metadata
        import json
        metadata_path = Path(model_path) / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            config_dict = metadata.get("config", {})
            
            from llm_compression.multimodal.audio_encoder import AudioConfig
            config = AudioConfig(
                n_mels=config_dict.get("n_mels", 80),
                hidden_size=config_dict.get("hidden_size", 384),
                num_layers=config_dict.get("num_layers", 4),
                num_attention_heads=config_dict.get("num_attention_heads", 6),
                intermediate_size=config_dict.get("intermediate_size", 1536),
                layer_norm_eps=config_dict.get("layer_norm_eps", 1e-5),
                max_positions=config_dict.get("max_positions", 1500),
            )
        else:
            config = None
        
        # 1. Loading time
        load_metrics = self.benchmark_loading_time(
            AudioEncoder,
            model_path,
            "audio",
            config=config
        )
        results["loading"] = {
            "time_ms": load_metrics["load_time_ms"],
            "memory_mb": load_metrics["memory_mb"]
        }
        encoder = load_metrics["encoder"]
        
        # Generate test audio
        num_samples = int(sample_rate * duration)
        test_audio = np.random.randn(num_samples).astype(np.float32)
        
        # 2. Single encoding latency
        latency_metrics = self.benchmark_single_encoding(
            encoder,
            test_audio,
            "audio"
        )
        results["latency"] = latency_metrics
        
        # 3. Batch throughput
        test_batch = [
            np.random.randn(num_samples).astype(np.float32)
            for _ in range(10)
        ]
        throughput_metrics = self.benchmark_batch_throughput(
            encoder,
            test_batch,
            "audio"
        )
        results["throughput"] = throughput_metrics
        
        # 4. Peak memory usage
        results["peak_memory_mb"] = self.get_memory_usage()
        
        return results


def print_results(results: Dict[str, Any], encoder_type: str) -> None:
    """Print benchmark results in a formatted way."""
    print(f"\n{'='*70}")
    print(f"  {encoder_type.upper()} ENCODER BENCHMARK RESULTS")
    print(f"{'='*70}\n")
    
    # Loading metrics
    print("--- Loading Performance ---")
    print(f"Load Time:       {results['loading']['time_ms']:.2f} ms")
    print(f"Memory Used:     {results['loading']['memory_mb']:.2f} MB")
    print()
    
    # Latency metrics
    print("--- Single Encoding Latency ---")
    lat = results['latency']
    print(f"Mean:            {lat['mean_latency_ms']:.2f} ms")
    print(f"Std Dev:         {lat['std_latency_ms']:.2f} ms")
    print(f"Min:             {lat['min_latency_ms']:.2f} ms")
    print(f"Max:             {lat['max_latency_ms']:.2f} ms")
    print(f"P50 (Median):    {lat['p50_latency_ms']:.2f} ms")
    print(f"P95:             {lat['p95_latency_ms']:.2f} ms")
    print(f"P99:             {lat['p99_latency_ms']:.2f} ms")
    print()
    
    # Throughput metrics
    print("--- Batch Throughput ---")
    thr = results['throughput']
    print(f"Throughput:      {thr['throughput_per_sec']:.2f} items/sec")
    print(f"Total Encoded:   {thr['total_encoded']}")
    print(f"Duration:        {thr['duration_sec']:.2f} sec")
    print(f"Batch Size:      {thr['batch_size']}")
    print()
    
    # Memory metrics
    print("--- Memory Usage ---")
    print(f"Peak Memory:     {results['peak_memory_mb']:.2f} MB")
    print()


def check_targets(results: Dict[str, Any], encoder_type: str) -> bool:
    """
    Check if results meet performance targets.
    
    Targets:
    - Vision: <500ms load, <100ms encode, 150+ img/s, <1GB memory
    - Audio: <500ms load, <200ms encode, 50+ audio/s, <500MB memory
    """
    if encoder_type == "vision":
        targets = {
            "load_time_ms": 500,
            "encode_latency_ms": 100,
            "throughput_per_sec": 150,
            "memory_mb": 1024
        }
    else:  # audio
        targets = {
            "load_time_ms": 500,
            "encode_latency_ms": 200,
            "throughput_per_sec": 50,
            "memory_mb": 512
        }
    
    passed = True
    print(f"{'='*70}")
    print(f"  TARGET VALIDATION - {encoder_type.upper()}")
    print(f"{'='*70}\n")
    
    # Check load time
    load_time = results['loading']['time_ms']
    load_pass = load_time < targets['load_time_ms']
    print(f"Load Time:       {load_time:.2f} ms < {targets['load_time_ms']} ms")
    print(f"                 {'✅ PASS' if load_pass else '❌ FAIL'}\n")
    passed = passed and load_pass
    
    # Check encoding latency
    encode_latency = results['latency']['mean_latency_ms']
    latency_pass = encode_latency < targets['encode_latency_ms']
    print(f"Encode Latency:  {encode_latency:.2f} ms < {targets['encode_latency_ms']} ms")
    print(f"                 {'✅ PASS' if latency_pass else '❌ FAIL'}\n")
    passed = passed and latency_pass
    
    # Check throughput
    throughput = results['throughput']['throughput_per_sec']
    throughput_pass = throughput >= targets['throughput_per_sec']
    print(f"Throughput:      {throughput:.2f} items/s >= {targets['throughput_per_sec']} items/s")
    print(f"                 {'✅ PASS' if throughput_pass else '❌ FAIL'}\n")
    passed = passed and throughput_pass
    
    # Check memory
    memory = results['peak_memory_mb']
    memory_pass = memory < targets['memory_mb']
    print(f"Peak Memory:     {memory:.2f} MB < {targets['memory_mb']} MB")
    print(f"                 {'✅ PASS' if memory_pass else '❌ FAIL'}\n")
    passed = passed and memory_pass
    
    print(f"{'='*70}")
    if passed:
        print("✅ ALL TARGETS MET")
    else:
        print("❌ SOME TARGETS NOT MET")
    print(f"{'='*70}\n")
    
    return passed


def main():
    """Main benchmark script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Benchmark multimodal encoder performance"
    )
    parser.add_argument(
        "--vision-model",
        type=str,
        default="D:/ai-models/clip-vit-b32",
        help="Path to vision model directory"
    )
    parser.add_argument(
        "--audio-model",
        type=str,
        default="D:/ai-models/whisper-tiny",
        help="Path to audio model directory"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON file for results (optional)"
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip vision encoder benchmark"
    )
    parser.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio encoder benchmark"
    )
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    all_results = {}
    all_passed = True
    
    # Benchmark vision encoder
    if not args.skip_vision:
        try:
            vision_results = benchmark.benchmark_vision_encoder(args.vision_model)
            all_results["vision"] = vision_results
            print_results(vision_results, "vision")
            vision_passed = check_targets(vision_results, "vision")
            all_passed = all_passed and vision_passed
        except Exception as e:
            logger.error(f"Vision encoder benchmark failed: {e}")
            all_passed = False
    
    # Benchmark audio encoder
    if not args.skip_audio:
        try:
            audio_results = benchmark.benchmark_audio_encoder(args.audio_model)
            all_results["audio"] = audio_results
            print_results(audio_results, "audio")
            audio_passed = check_targets(audio_results, "audio")
            all_passed = all_passed and audio_passed
        except Exception as e:
            logger.error(f"Audio encoder benchmark failed: {e}")
            all_passed = False
    
    # Save results to JSON if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"Results saved to {args.output}")
    
    # Exit with appropriate code
    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
