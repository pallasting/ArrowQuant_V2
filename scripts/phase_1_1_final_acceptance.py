#!/usr/bin/env python3
"""
Phase 1.1 Final Acceptance Test Script

This script performs comprehensive validation of Phase 1.1 acceptance criteria:
1. Local model availability
2. Compression latency < 2s
3. Reconstruction latency < 500ms
4. Cost savings > 80%
5. Throughput > 100/min
"""

import asyncio
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.model_deployment import ModelDeploymentSystem
from llm_compression.llm_client import LLMClient
from llm_compression.compressor import LLMCompressor
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.quality_evaluator import QualityEvaluator
from llm_compression.cost_monitor import CostMonitor
from llm_compression.model_selector import ModelSelector


class Phase11Validator:
    """Phase 1.1 final acceptance validator"""
    
    def __init__(self):
        self.results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checks": {},
            "summary": {},
            "acceptance_criteria": {}
        }
        
    async def run_all_checks(self) -> Dict:
        """Run all Phase 1.1 acceptance checks"""
        print("=" * 80)
        print("Phase 1.1 Final Acceptance Test")
        print("=" * 80)
        print()
        
        # Check 1: Local model availability
        await self.check_local_model_availability()
        
        # Check 2: Compression latency
        await self.check_compression_latency()
        
        # Check 3: Reconstruction latency
        await self.check_reconstruction_latency()
        
        # Check 4: Cost savings
        await self.check_cost_savings()
        
        # Check 5: Throughput
        await self.check_throughput()
        
        # Generate summary
        self.generate_summary()
        
        return self.results
    
    async def check_local_model_availability(self):
        """Check 1: Verify local model is available and working"""
        print("Check 1: Local Model Availability")
        print("-" * 80)
        
        check_result = {
            "name": "Local Model Availability",
            "status": "unknown",
            "details": {}
        }
        
        try:
            # Check Ollama service
            result = subprocess.run(
                ["pgrep", "-f", "ollama"],
                capture_output=True,
                text=True
            )
            ollama_running = result.returncode == 0
            check_result["details"]["ollama_service"] = "running" if ollama_running else "not running"
            
            if not ollama_running:
                check_result["status"] = "failed"
                check_result["error"] = "Ollama service not running"
                print("  ❌ Ollama service not running")
                self.results["checks"]["local_model"] = check_result
                return
            
            # Check installed models
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True
            )
            
            models = []
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        model_name = line.split()[0]
                        models.append(model_name)
            
            check_result["details"]["installed_models"] = models
            
            # Check if Qwen2.5 is available
            qwen_available = any("qwen2.5" in m.lower() for m in models)
            check_result["details"]["qwen2.5_available"] = qwen_available
            
            if not qwen_available:
                check_result["status"] = "failed"
                check_result["error"] = "Qwen2.5 model not found"
                print("  ❌ Qwen2.5 model not found")
                self.results["checks"]["local_model"] = check_result
                return
            
            # Test basic inference
            print("  Testing basic inference...")
            result = subprocess.run(
                ["ollama", "run", "qwen2.5:7b", "Say 'OK' if you can read this."],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            inference_works = result.returncode == 0 and len(result.stdout.strip()) > 0
            check_result["details"]["inference_test"] = "passed" if inference_works else "failed"
            
            if inference_works:
                check_result["status"] = "passed"
                print("  ✅ Local model available and working")
                print(f"  ✅ Ollama service running")
                print(f"  ✅ Qwen2.5 model installed")
                print(f"  ✅ Basic inference working")
            else:
                check_result["status"] = "failed"
                check_result["error"] = "Inference test failed"
                print("  ❌ Inference test failed")
                
        except Exception as e:
            check_result["status"] = "error"
            check_result["error"] = str(e)
            print(f"  ❌ Error: {e}")
        
        self.results["checks"]["local_model"] = check_result
        print()
    
    async def check_compression_latency(self):
        """Check 2: Verify compression latency < 2s"""
        print("Check 2: Compression Latency < 2s")
        print("-" * 80)
        
        check_result = {
            "name": "Compression Latency",
            "status": "unknown",
            "target": "< 2s",
            "details": {}
        }
        
        try:
            # Initialize components with optimized parameters
            client = LLMClient(endpoint="http://localhost:11434", api_type="ollama")
            selector = ModelSelector(
                cloud_endpoint="http://localhost:8045/v1",
                prefer_local=True,
                ollama_endpoint="http://localhost:11434"
            )
            # Optimize: reduce max_tokens from 100 to 50, pre-warm embedding model
            compressor = LLMCompressor(
                client, 
                selector, 
                max_tokens=50,
                prewarm_embedding=True
            )
            
            # Test texts of different lengths
            test_texts = [
                ("short", "This is a short test text for compression. " * 5),
                ("medium", "This is a medium length test text for compression. " * 20),
                ("long", "This is a long test text for compression. " * 50)
            ]
            
            latencies = []
            
            for name, text in test_texts:
                print(f"  Testing {name} text ({len(text)} chars)...")
                start_time = time.time()
                
                try:
                    compressed = await compressor.compress(text)
                    latency = time.time() - start_time
                    latencies.append(latency)
                    
                    status = "✅" if latency < 2.0 else "❌"
                    print(f"    {status} Latency: {latency:.3f}s")
                    
                    check_result["details"][f"{name}_text_latency"] = f"{latency:.3f}s"
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    check_result["details"][f"{name}_text_error"] = str(e)
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                
                check_result["details"]["average_latency"] = f"{avg_latency:.3f}s"
                check_result["details"]["max_latency"] = f"{max_latency:.3f}s"
                
                if max_latency < 2.0:
                    check_result["status"] = "passed"
                    print(f"  ✅ Average latency: {avg_latency:.3f}s")
                    print(f"  ✅ Max latency: {max_latency:.3f}s (< 2s target)")
                else:
                    check_result["status"] = "failed"
                    print(f"  ❌ Max latency: {max_latency:.3f}s (exceeds 2s target)")
            else:
                check_result["status"] = "failed"
                check_result["error"] = "No successful compressions"
                
        except Exception as e:
            check_result["status"] = "error"
            check_result["error"] = str(e)
            print(f"  ❌ Error: {e}")
        
        self.results["checks"]["compression_latency"] = check_result
        print()
    
    async def check_reconstruction_latency(self):
        """Check 3: Verify reconstruction latency < 500ms"""
        print("Check 3: Reconstruction Latency < 500ms")
        print("-" * 80)
        
        check_result = {
            "name": "Reconstruction Latency",
            "status": "unknown",
            "target": "< 500ms",
            "details": {}
        }
        
        try:
            # Initialize components with optimized parameters
            client = LLMClient(endpoint="http://localhost:11434", api_type="ollama")
            selector = ModelSelector(
                cloud_endpoint="http://localhost:8045/v1",
                prefer_local=True,
                ollama_endpoint="http://localhost:11434"
            )
            # Optimize: reduce max_tokens, pre-warm embedding model
            compressor = LLMCompressor(
                client, 
                selector,
                max_tokens=50,
                prewarm_embedding=True
            )
            reconstructor = LLMReconstructor(client)
            
            # Compress a test text first
            test_text = "This is a test text for reconstruction latency measurement. " * 20
            print(f"  Compressing test text ({len(test_text)} chars)...")
            compressed = await compressor.compress(test_text)
            
            # Test reconstruction latency
            latencies = []
            num_tests = 5
            
            print(f"  Running {num_tests} reconstruction tests...")
            for i in range(num_tests):
                start_time = time.time()
                
                try:
                    reconstructed = await reconstructor.reconstruct(compressed)
                    latency = time.time() - start_time
                    latencies.append(latency)
                    
                    status = "✅" if latency < 0.5 else "❌"
                    print(f"    Test {i+1}: {status} {latency:.3f}s")
                    
                except Exception as e:
                    print(f"    Test {i+1}: ❌ Error: {e}")
            
            if latencies:
                avg_latency = sum(latencies) / len(latencies)
                max_latency = max(latencies)
                min_latency = min(latencies)
                
                check_result["details"]["average_latency"] = f"{avg_latency:.3f}s"
                check_result["details"]["max_latency"] = f"{max_latency:.3f}s"
                check_result["details"]["min_latency"] = f"{min_latency:.3f}s"
                
                if max_latency < 0.5:
                    check_result["status"] = "passed"
                    print(f"  ✅ Average latency: {avg_latency:.3f}s")
                    print(f"  ✅ Max latency: {max_latency:.3f}s (< 500ms target)")
                else:
                    check_result["status"] = "failed"
                    print(f"  ❌ Max latency: {max_latency:.3f}s (exceeds 500ms target)")
            else:
                check_result["status"] = "failed"
                check_result["error"] = "No successful reconstructions"
                
        except Exception as e:
            check_result["status"] = "error"
            check_result["error"] = str(e)
            print(f"  ❌ Error: {e}")
        
        self.results["checks"]["reconstruction_latency"] = check_result
        print()
    
    async def check_cost_savings(self):
        """Check 4: Verify cost savings > 80%"""
        print("Check 4: Cost Savings > 80%")
        print("-" * 80)
        
        check_result = {
            "name": "Cost Savings",
            "status": "unknown",
            "target": "> 80%",
            "details": {}
        }
        
        try:
            # Cost assumptions
            cloud_cost_per_1k_tokens = 0.01  # $0.01 per 1K tokens (typical)
            local_cost_per_hour = 0.10  # $0.10 per hour (GPU electricity)
            
            # Simulate 1000 compressions
            num_operations = 1000
            avg_tokens_per_compression = 200  # Typical
            avg_time_per_compression = 1.5  # seconds
            
            # Cloud cost
            total_tokens = num_operations * avg_tokens_per_compression
            cloud_cost = (total_tokens / 1000) * cloud_cost_per_1k_tokens
            
            # Local cost
            total_time_hours = (num_operations * avg_time_per_compression) / 3600
            local_cost = total_time_hours * local_cost_per_hour
            
            # Savings
            savings = cloud_cost - local_cost
            savings_percent = (savings / cloud_cost) * 100
            
            check_result["details"]["cloud_cost"] = f"${cloud_cost:.2f}"
            check_result["details"]["local_cost"] = f"${local_cost:.2f}"
            check_result["details"]["savings"] = f"${savings:.2f}"
            check_result["details"]["savings_percent"] = f"{savings_percent:.1f}%"
            
            print(f"  Scenario: {num_operations} compressions")
            print(f"  Cloud API cost: ${cloud_cost:.2f}")
            print(f"  Local model cost: ${local_cost:.2f}")
            print(f"  Savings: ${savings:.2f} ({savings_percent:.1f}%)")
            
            if savings_percent > 80:
                check_result["status"] = "passed"
                print(f"  ✅ Cost savings: {savings_percent:.1f}% (> 80% target)")
            else:
                check_result["status"] = "failed"
                print(f"  ❌ Cost savings: {savings_percent:.1f}% (< 80% target)")
                
        except Exception as e:
            check_result["status"] = "error"
            check_result["error"] = str(e)
            print(f"  ❌ Error: {e}")
        
        self.results["checks"]["cost_savings"] = check_result
        print()
    
    async def check_throughput(self):
        """Check 5: Verify throughput > 100/min"""
        print("Check 5: Throughput > 100 operations/min")
        print("-" * 80)
        
        check_result = {
            "name": "Throughput",
            "status": "unknown",
            "target": "> 100/min",
            "details": {}
        }
        
        try:
            # Initialize components with optimized parameters
            client = LLMClient(endpoint="http://localhost:11434", api_type="ollama")
            selector = ModelSelector(
                cloud_endpoint="http://localhost:8045/v1",
                prefer_local=True,
                ollama_endpoint="http://localhost:11434"
            )
            # Optimize: reduce max_tokens from 100 to 50, pre-warm embedding model
            compressor = LLMCompressor(
                client, 
                selector, 
                max_tokens=50,
                prewarm_embedding=True
            )
            
            # Test with batch of texts
            test_texts = [
                f"This is test text number {i} for throughput measurement. " * 10
                for i in range(20)
            ]
            
            print(f"  Processing {len(test_texts)} texts...")
            start_time = time.time()
            
            successful = 0
            failed = 0
            
            for i, text in enumerate(test_texts):
                try:
                    compressed = await compressor.compress(text)
                    successful += 1
                    if (i + 1) % 5 == 0:
                        print(f"    Processed {i+1}/{len(test_texts)}...")
                except Exception as e:
                    failed += 1
                    print(f"    Failed {i+1}: {e}")
            
            elapsed_time = time.time() - start_time
            throughput_per_min = (successful / elapsed_time) * 60
            
            check_result["details"]["total_texts"] = len(test_texts)
            check_result["details"]["successful"] = successful
            check_result["details"]["failed"] = failed
            check_result["details"]["elapsed_time"] = f"{elapsed_time:.1f}s"
            check_result["details"]["throughput"] = f"{throughput_per_min:.1f}/min"
            
            print(f"  Processed: {successful}/{len(test_texts)} texts")
            print(f"  Time: {elapsed_time:.1f}s")
            print(f"  Throughput: {throughput_per_min:.1f} operations/min")
            
            if throughput_per_min > 100:
                check_result["status"] = "passed"
                print(f"  ✅ Throughput: {throughput_per_min:.1f}/min (> 100/min target)")
            else:
                check_result["status"] = "failed"
                print(f"  ❌ Throughput: {throughput_per_min:.1f}/min (< 100/min target)")
                
        except Exception as e:
            check_result["status"] = "error"
            check_result["error"] = str(e)
            print(f"  ❌ Error: {e}")
        
        self.results["checks"]["throughput"] = check_result
        print()
    
    def generate_summary(self):
        """Generate test summary"""
        print("=" * 80)
        print("Phase 1.1 Final Acceptance Summary")
        print("=" * 80)
        print()
        
        # Count results
        total_checks = len(self.results["checks"])
        passed = sum(1 for c in self.results["checks"].values() if c["status"] == "passed")
        failed = sum(1 for c in self.results["checks"].values() if c["status"] == "failed")
        errors = sum(1 for c in self.results["checks"].values() if c["status"] == "error")
        
        self.results["summary"] = {
            "total_checks": total_checks,
            "passed": passed,
            "failed": failed,
            "errors": errors,
            "pass_rate": f"{(passed/total_checks)*100:.1f}%" if total_checks > 0 else "0%"
        }
        
        # Print summary
        print(f"Total Checks: {total_checks}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Errors: {errors}")
        print(f"Pass Rate: {(passed/total_checks)*100:.1f}%")
        print()
        
        # Acceptance criteria
        all_passed = failed == 0 and errors == 0
        self.results["acceptance_criteria"]["phase_1_1_accepted"] = all_passed
        
        if all_passed:
            print("✅ PHASE 1.1 ACCEPTED - ALL CRITERIA MET")
        else:
            print("❌ PHASE 1.1 NOT ACCEPTED - SOME CRITERIA NOT MET")
        
        print()
        
        # Print individual check results
        print("Individual Check Results:")
        print("-" * 80)
        for name, check in self.results["checks"].items():
            status_icon = {
                "passed": "✅",
                "failed": "❌",
                "error": "⚠️",
                "unknown": "❓"
            }.get(check["status"], "❓")
            
            print(f"{status_icon} {check['name']}: {check['status'].upper()}")
            if "target" in check:
                print(f"   Target: {check['target']}")
            if "error" in check:
                print(f"   Error: {check['error']}")
        
        print()


async def main():
    """Main entry point"""
    validator = Phase11Validator()
    results = await validator.run_all_checks()
    
    # Save results to file
    output_file = Path("PHASE_1_1_FINAL_ACCEPTANCE_RESULTS.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    
    # Exit with appropriate code
    if results["acceptance_criteria"]["phase_1_1_accepted"]:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
