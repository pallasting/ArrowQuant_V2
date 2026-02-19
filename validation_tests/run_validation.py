"""
ArrowEngine Hardware Environment Deployment Validation - Main Script

Run all validation tests and generate report.
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

# Test suite definition
TESTS = [
    {
        "name": "Environment Check",
        "script": "test_environment.py",
        "required": True,
        "timeout": 30,
    },
    {
        "name": "Model Load Speed",
        "script": "test_load_speed.py",
        "required": True,
        "timeout": 60,
    },
    {
        "name": "Inference Latency",
        "script": "test_inference_latency.py",
        "required": True,
        "timeout": 120,
    },
    {
        "name": "Batch Throughput",
        "script": "test_batch_throughput.py",
        "required": True,
        "timeout": 180,
    },
    {
        "name": "Memory Usage",
        "script": "test_memory_usage.py",
        "required": False,  # psutil may not be installed
        "timeout": 60,
    },
    {
        "name": "Precision Validation",
        "script": "test_precision_validation.py",
        "required": False,  # sentence-transformers may not be installed
        "timeout": 120,
    },
    {
        "name": "EmbeddingProvider Interface",
        "script": "test_embedding_provider.py",
        "required": True,
        "timeout": 60,
    },
    {
        "name": "ArrowStorage Integration",
        "script": "test_arrow_storage_integration.py",
        "required": True,
        "timeout": 60,
    },
]


def print_header():
    """Print header"""
    print("=" * 70)
    print(" " * 10 + "ArrowEngine Hardware Environment Deployment Validation")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test count: {len(TESTS)}")
    print(f"Required tests: {sum(1 for t in TESTS if t['required'])}")
    print(f"Optional tests: {sum(1 for t in TESTS if not t['required'])}")
    print()


def run_test(test_info):
    """Run a single test"""
    name = test_info["name"]
    script = test_info["script"]
    required = test_info["required"]
    timeout = test_info["timeout"]
    
    print("=" * 70)
    print(f"Test: {name}")
    print(f"Script: {script}")
    print(f"Required: {'Yes' if required else 'No'}")
    print("=" * 70)
    
    script_path = Path(__file__).parent / script
    
    if not script_path.exists():
        print(f"X Test script does not exist: {script_path}")
        return {
            "name": name,
            "status": "error",
            "message": "Script not found",
            "required": required,
        }
    
    try:
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path(__file__).parent.parent,  # Run in project root
        )
        
        elapsed = time.time() - start_time
        
        # Print output
        if result.stdout:
            print(result.stdout)
        
        if result.stderr and result.returncode != 0:
            print("\nError output:")
            print(result.stderr)
        
        # Determine result
        if result.returncode == 0:
            print(f"\nv {name} - Passed (Time: {elapsed:.1f}s)")
            return {
                "name": name,
                "status": "passed",
                "elapsed": elapsed,
                "required": required,
            }
        else:
            status_msg = "Failed" if required else "Skipped"
            print(f"\n{'X' if required else '!'} {name} - {status_msg} (Time: {elapsed:.1f}s)")
            return {
                "name": name,
                "status": "failed" if required else "skipped",
                "elapsed": elapsed,
                "required": required,
                "message": result.stderr[:200] if result.stderr else "Unknown error",
            }
    
    except subprocess.TimeoutExpired:
        print(f"\n@ {name} - Timeout (>{timeout}s)")
        return {
            "name": name,
            "status": "timeout",
            "required": required,
            "message": f"Timeout (>{timeout}s)",
        }
    
    except Exception as e:
        print(f"\nX {name} - Error: {e}")
        return {
            "name": name,
            "status": "error",
            "required": required,
            "message": str(e),
        }


def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 70)
    print(" " * 25 + "Test Summary")
    print("=" * 70)
    
    passed = sum(1 for r in results if r["status"] == "passed")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"] in ["error", "timeout"])
    
    required_passed = sum(1 for r in results if r["status"] == "passed" and r["required"])
    required_total = sum(1 for r in results if r["required"])
    
    print(f"\nTotal tests: {len(results)}")
    print(f"  v Passed: {passed}")
    print(f"  X Failed: {failed}")
    print(f"  ! Skipped: {skipped}")
    print(f"  @ Errors: {errors}")
    
    print(f"\nRequired tests: {required_passed}/{required_total} passed")
    
    success_rate = (passed / len(results)) * 100
    print(f"Success rate: {success_rate:.1f}%")
    
    # Detailed results
    print(f"\nDetailed results:")
    for r in results:
        status_icon = {
            "passed": "v",
            "failed": "X",
            "skipped": "!",
            "error": "@",
            "timeout": "@",
        }.get(r["status"], "?")
        
        elapsed_str = f" ({r['elapsed']:.1f}s)" if "elapsed" in r else ""
        required_str = " [Required]" if r["required"] else " [Optional]"
        
        print(f"  {status_icon} {r['name']}{required_str}{elapsed_str}")
        
        if r["status"] in ["failed", "error", "timeout"] and "message" in r:
            print(f"     Reason: {r['message'][:100]}")
    
    # Final judgment
    print(f"\n" + "=" * 70)
    
    if required_passed == required_total:
        print("v All required tests passed!")
        print("\nArrowEngine is ready to run in the current hardware environment.")
        print("Next step: Proceed with AI-OS Memory system integration or Multimodal Encoder extension.")
        return 0
    else:
        print(f"X {required_total - required_passed} required test(s) failed")
        print("\nPlease check failed tests and resolve issues before re-running.")
        print("Reference: ARROWENGINE_DEPLOYMENT_VALIDATION_PLAN.md troubleshooting section")
        return 1


def save_report(results):
    """Save test report"""
    report_path = Path(__file__).parent.parent / "VALIDATION_REPORT.md"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# ArrowEngine Hardware Environment Validation Report\n\n")
        f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        f.write("## Test Results\n\n")
        
        for r in results:
            status_icon = {
                "passed": "v",
                "failed": "X",
                "skipped": "!",
                "error": "@",
                "timeout": "@",
            }.get(r["status"], "?")
            
            f.write(f"### {status_icon} {r['name']}\n\n")
            f.write(f"- **Status**: {r['status']}\n")
            f.write(f"- **Required**: {'Yes' if r['required'] else 'No'}\n")
            
            if "elapsed" in r:
                f.write(f"- **Time**: {r['elapsed']:.1f}s\n")
            
            if "message" in r:
                f.write(f"- **Info**: {r['message']}\n")
            
            f.write("\n")
        
        # Statistics
        passed = sum(1 for r in results if r["status"] == "passed")
        f.write("---\n\n")
        f.write("## Statistics\n\n")
        f.write(f"- Total tests: {len(results)}\n")
        f.write(f"- Passed: {passed}\n")
        f.write(f"- Success rate: {(passed/len(results))*100:.1f}%\n")
    
    print(f"\nReport saved: {report_path}")


def main():
    """Main function"""
    print_header()
    
    results = []
    
    for test_info in TESTS:
        result = run_test(test_info)
        results.append(result)
        
        # If required test fails, continue with remaining tests
        if result["status"] in ["failed", "error"] and result["required"]:
            print(f"\n! Required test failed: {result['name']}")
            print("Continuing with remaining tests...")
        
        print()  # Empty line separator
    
    # Print summary
    exit_code = print_summary(results)
    
    # Save report
    save_report(results)
    
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
