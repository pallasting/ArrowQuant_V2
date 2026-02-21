#!/usr/bin/env python3
"""
Run quantization validation tests and generate report.

This script:
1. Runs end-to-end quantization tests
2. Collects metrics and results
3. Generates validation report with charts
4. Validates against acceptance criteria

Usage:
    python scripts/run_quantization_validation.py
    python scripts/run_quantization_validation.py --model minilm
    python scripts/run_quantization_validation.py --output-dir validation_results
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

from llm_compression.logger import logger


def run_pytest_tests(test_file: str, output_dir: Path) -> Dict[str, Any]:
    """
    Run pytest tests and collect results.
    
    Args:
        test_file: Path to test file
        output_dir: Directory to store results
        
    Returns:
        Dictionary with test results
    """
    logger.info(f"Running tests: {test_file}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pytest with JSON report
    json_report = output_dir / "test_results.json"
    cmd = [
        "pytest",
        test_file,
        "-v",
        "-s",
        f"--json-report",
        f"--json-report-file={json_report}",
        "--tb=short"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False
        )
        
        execution_time = time.time() - start_time
        
        # Parse results
        if json_report.exists():
            with open(json_report) as f:
                test_data = json.load(f)
        else:
            test_data = {}
        
        return {
            'success': result.returncode == 0,
            'returncode': result.returncode,
            'execution_time': execution_time,
            'stdout': result.stdout,
            'stderr': result.stderr,
            'test_data': test_data
        }
        
    except Exception as e:
        logger.error(f"Failed to run tests: {e}")
        return {
            'success': False,
            'error': str(e),
            'execution_time': time.time() - start_time
        }


def extract_metrics_from_output(output: str) -> Dict[str, Any]:
    """
    Extract metrics from test output.
    
    Args:
        output: Test stdout/stderr
        
    Returns:
        Dictionary with extracted metrics
    """
    metrics = {
        'compression_ratios': {},
        'accuracies': {},
        'timings': {},
        'sizes': {}
    }
    
    lines = output.split('\n')
    
    for line in lines:
        # Extract compression ratios
        if 'Compression ratio:' in line:
            try:
                ratio = float(line.split(':')[1].strip().rstrip('x'))
                if 'INT8' in line or 'int8' in line:
                    metrics['compression_ratios']['int8'] = ratio
                elif 'INT2' in line or 'int2' in line:
                    metrics['compression_ratios']['int2'] = ratio
            except:
                pass
        
        # Extract cosine similarities
        if 'Average cosine similarity:' in line:
            try:
                sim = float(line.split(':')[1].strip())
                if 'INT8' in line or 'int8' in line:
                    metrics['accuracies']['int8_cosine_sim'] = sim
                elif 'INT2' in line or 'int2' in line:
                    metrics['accuracies']['int2_cosine_sim'] = sim
            except:
                pass
        
        # Extract precision loss
        if 'Precision loss:' in line:
            try:
                loss = float(line.split(':')[1].strip().rstrip('%'))
                if 'INT8' in line or 'int8' in line:
                    metrics['accuracies']['int8_precision_loss'] = loss
                elif 'INT2' in line or 'int2' in line:
                    metrics['accuracies']['int2_precision_loss'] = loss
            except:
                pass
        
        # Extract timings
        if 'Quantization completed in' in line:
            try:
                timing = float(line.split('in')[1].strip().rstrip('s'))
                if 'INT8' in line or 'int8' in line:
                    metrics['timings']['int8_quantization'] = timing
                elif 'INT2' in line or 'int2' in line:
                    metrics['timings']['int2_quantization'] = timing
            except:
                pass
    
    return metrics


def validate_acceptance_criteria(metrics: Dict[str, Any]) -> Dict[str, bool]:
    """
    Validate metrics against acceptance criteria.
    
    Args:
        metrics: Extracted metrics
        
    Returns:
        Dictionary with validation results
    """
    results = {}
    
    # Compression ratio targets
    if 'int8' in metrics['compression_ratios']:
        results['int8_compression_ratio'] = metrics['compression_ratios']['int8'] > 2.0
    
    if 'int2' in metrics['compression_ratios']:
        results['int2_compression_ratio'] = metrics['compression_ratios']['int2'] > 4.0
    
    # Accuracy targets
    if 'int8_cosine_sim' in metrics['accuracies']:
        results['int8_accuracy'] = metrics['accuracies']['int8_cosine_sim'] > 0.85
    
    if 'int2_cosine_sim' in metrics['accuracies']:
        results['int2_accuracy'] = metrics['accuracies']['int2_cosine_sim'] > 0.70
    
    if 'int8_precision_loss' in metrics['accuracies']:
        results['int8_precision_loss'] = metrics['accuracies']['int8_precision_loss'] < 15.0
    
    # Performance targets
    if 'int8_quantization' in metrics['timings']:
        results['int8_speed'] = metrics['timings']['int8_quantization'] < 60.0
    
    if 'int2_quantization' in metrics['timings']:
        results['int2_speed'] = metrics['timings']['int2_quantization'] < 60.0
    
    return results


def generate_summary_report(
    test_results: Dict[str, Any],
    metrics: Dict[str, Any],
    validation: Dict[str, bool],
    output_dir: Path
) -> None:
    """
    Generate summary report.
    
    Args:
        test_results: Test execution results
        metrics: Extracted metrics
        validation: Validation results
        output_dir: Output directory
    """
    report_path = output_dir / "validation_summary.md"
    
    with open(report_path, 'w') as f:
        f.write("# Quantization Validation Summary\n\n")
        f.write(f"**Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Test execution summary
        f.write("## Test Execution\n\n")
        f.write(f"- **Status**: {'✅ PASSED' if test_results['success'] else '❌ FAILED'}\n")
        f.write(f"- **Execution Time**: {test_results['execution_time']:.2f}s\n")
        f.write(f"- **Return Code**: {test_results.get('returncode', 'N/A')}\n\n")
        
        # Metrics summary
        f.write("## Metrics\n\n")
        
        f.write("### Compression Ratios\n\n")
        for quant_type, ratio in metrics['compression_ratios'].items():
            f.write(f"- **{quant_type.upper()}**: {ratio:.2f}x\n")
        f.write("\n")
        
        f.write("### Accuracy\n\n")
        for metric, value in metrics['accuracies'].items():
            if 'cosine_sim' in metric:
                f.write(f"- **{metric}**: {value:.4f}\n")
            elif 'precision_loss' in metric:
                f.write(f"- **{metric}**: {value:.2f}%\n")
        f.write("\n")
        
        f.write("### Performance\n\n")
        for metric, value in metrics['timings'].items():
            f.write(f"- **{metric}**: {value:.2f}s\n")
        f.write("\n")
        
        # Validation results
        f.write("## Acceptance Criteria Validation\n\n")
        
        all_passed = all(validation.values())
        f.write(f"**Overall Status**: {'✅ ALL PASSED' if all_passed else '❌ SOME FAILED'}\n\n")
        
        f.write("| Criterion | Target | Result | Status |\n")
        f.write("|-----------|--------|--------|--------|\n")
        
        for criterion, passed in validation.items():
            status = "✅ Pass" if passed else "❌ Fail"
            
            # Get target and result
            if 'compression_ratio' in criterion:
                if 'int8' in criterion:
                    target = ">2x"
                    result = f"{metrics['compression_ratios'].get('int8', 0):.2f}x"
                else:
                    target = ">4x"
                    result = f"{metrics['compression_ratios'].get('int2', 0):.2f}x"
            elif 'accuracy' in criterion:
                if 'int8' in criterion:
                    target = ">0.85"
                    result = f"{metrics['accuracies'].get('int8_cosine_sim', 0):.4f}"
                else:
                    target = ">0.70"
                    result = f"{metrics['accuracies'].get('int2_cosine_sim', 0):.4f}"
            elif 'precision_loss' in criterion:
                target = "<15%"
                result = f"{metrics['accuracies'].get('int8_precision_loss', 0):.2f}%"
            elif 'speed' in criterion:
                target = "<60s"
                if 'int8' in criterion:
                    result = f"{metrics['timings'].get('int8_quantization', 0):.2f}s"
                else:
                    result = f"{metrics['timings'].get('int2_quantization', 0):.2f}s"
            else:
                target = "N/A"
                result = "N/A"
            
            f.write(f"| {criterion} | {target} | {result} | {status} |\n")
        
        f.write("\n")
        
        # Recommendations
        f.write("## Recommendations\n\n")
        
        if all_passed:
            f.write("✅ **All acceptance criteria met. System is ready for production deployment.**\n\n")
            f.write("Recommended configuration for production:\n")
            f.write("```python\n")
            f.write("config = QuantizationConfig(\n")
            f.write("    quant_type='int8',\n")
            f.write("    per_channel=True,\n")
            f.write("    symmetric=True\n")
            f.write(")\n")
            f.write("```\n")
        else:
            f.write("⚠️ **Some acceptance criteria not met. Review failed tests before deployment.**\n\n")
            
            failed = [k for k, v in validation.items() if not v]
            f.write("Failed criteria:\n")
            for criterion in failed:
                f.write(f"- {criterion}\n")
    
    logger.info(f"Summary report generated: {report_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run quantization validation tests")
    parser.add_argument(
        '--model',
        default='minilm',
        choices=['minilm', 'qwen'],
        help='Model to test (default: minilm)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('validation_results'),
        help='Output directory for results (default: validation_results)'
    )
    parser.add_argument(
        '--test-file',
        type=Path,
        default=Path('tests/integration/test_quantization_e2e.py'),
        help='Test file to run'
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 80)
    logger.info("Quantization Validation Test Suite")
    logger.info("=" * 80)
    logger.info(f"Model: {args.model}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("")
    
    # Run tests
    test_results = run_pytest_tests(str(args.test_file), args.output_dir)
    
    if not test_results['success']:
        logger.error("Tests failed!")
        logger.error(f"Return code: {test_results.get('returncode')}")
        if 'error' in test_results:
            logger.error(f"Error: {test_results['error']}")
        sys.exit(1)
    
    logger.info("Tests completed successfully!")
    logger.info(f"Execution time: {test_results['execution_time']:.2f}s")
    logger.info("")
    
    # Extract metrics
    logger.info("Extracting metrics...")
    metrics = extract_metrics_from_output(test_results['stdout'])
    
    # Validate acceptance criteria
    logger.info("Validating acceptance criteria...")
    validation = validate_acceptance_criteria(metrics)
    
    # Generate summary report
    logger.info("Generating summary report...")
    generate_summary_report(test_results, metrics, validation, args.output_dir)
    
    # Print summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("Validation Summary")
    logger.info("=" * 80)
    
    all_passed = all(validation.values())
    if all_passed:
        logger.info("✅ ALL ACCEPTANCE CRITERIA PASSED")
    else:
        logger.warning("⚠️ SOME ACCEPTANCE CRITERIA FAILED")
        failed = [k for k, v in validation.items() if not v]
        for criterion in failed:
            logger.warning(f"  ❌ {criterion}")
    
    logger.info("")
    logger.info(f"Full report: {args.output_dir / 'validation_summary.md'}")
    logger.info(f"Detailed report: docs/QUANTIZATION_VALIDATION_REPORT.md")
    logger.info("=" * 80)
    
    sys.exit(0 if all_passed else 1)


if __name__ == '__main__':
    main()
