#!/usr/bin/env python3
"""
ArrowQuant V2 Validation Script

Validates quantized models by comparing them with original models and generating
comprehensive validation reports. Supports custom accuracy thresholds and provides
per-layer accuracy analysis.

Usage:
    # Basic validation
    python scripts/validate_quantization.py \\
        --original dream-7b/ \\
        --quantized dream-7b-int2/

    # Custom thresholds
    python scripts/validate_quantization.py \\
        --original dream-7b/ \\
        --quantized dream-7b-int2/ \\
        --min-accuracy 0.75 \\
        --bit-width 2

    # Generate detailed report
    python scripts/validate_quantization.py \\
        --original dream-7b/ \\
        --quantized dream-7b-int2/ \\
        --report validation_report.json \\
        --html-report validation_report.html

Author: ArrowQuant V2 Team
License: MIT
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("validation.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationConfig:
    """Configuration for validation."""

    original_path: str
    quantized_path: str
    min_accuracy: Optional[float] = None
    bit_width: Optional[int] = None
    report_path: Optional[str] = None
    html_report_path: Optional[str] = None
    per_layer_threshold: Optional[float] = None
    fail_on_threshold: bool = False


@dataclass
class ValidationResult:
    """Stores validation results."""

    passed: bool
    cosine_similarity: float
    compression_ratio: float
    model_size_mb: float
    per_layer_accuracy: Dict[str, float]
    statistics: Dict[str, float]
    threshold: float
    bit_width: Optional[int]
    validation_time_s: float
    warnings: List[str]
    errors: List[str]


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ArrowQuant V2 Validation Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic validation
  %(prog)s --original dream-7b/ --quantized dream-7b-int2/

  # INT2 validation with standard threshold (0.70)
  %(prog)s --original dream-7b/ --quantized dream-7b-int2/ --bit-width 2

  # Custom threshold
  %(prog)s --original dream-7b/ --quantized dream-7b-int2/ --min-accuracy 0.75

  # Generate reports
  %(prog)s --original dream-7b/ --quantized dream-7b-int2/ \\
           --report validation.json --html-report validation.html

  # Fail on threshold violation
  %(prog)s --original dream-7b/ --quantized dream-7b-int2/ \\
           --min-accuracy 0.70 --fail-on-threshold

For more information, see: docs/DEPLOYMENT.md
        """,
    )

    # Required arguments
    parser.add_argument(
        "--original",
        type=str,
        required=True,
        help="Path to original model directory",
    )
    parser.add_argument(
        "--quantized",
        type=str,
        required=True,
        help="Path to quantized model directory",
    )

    # Threshold arguments
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="Minimum cosine similarity threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        help="Bit width for standard threshold (INT2: 0.70, INT4: 0.90, INT8: 0.95)",
    )
    parser.add_argument(
        "--per-layer-threshold",
        type=float,
        help="Minimum threshold for individual layers (default: same as --min-accuracy)",
    )

    # Report arguments
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save JSON validation report",
    )
    parser.add_argument(
        "--html-report",
        type=str,
        help="Path to save HTML validation report",
    )

    # Behavior arguments
    parser.add_argument(
        "--fail-on-threshold",
        action="store_true",
        help="Exit with error code if validation fails threshold",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages",
    )

    return parser.parse_args()


def get_standard_threshold(bit_width: int) -> float:
    """Get standard accuracy threshold for bit width."""
    thresholds = {
        2: 0.70,  # INT2
        4: 0.90,  # INT4
        8: 0.95,  # INT8
    }
    return thresholds.get(bit_width, 0.85)


def validate_model(config: ValidationConfig) -> ValidationResult:
    """Validate quantized model against original."""
    try:
        # Import here to provide better error messages
        try:
            from arrow_quant_v2 import ArrowQuantV2
        except ImportError as e:
            logger.error(
                "Failed to import arrow_quant_v2. "
                "Please install with: pip install -e ."
            )
            raise

        # Validate paths
        original_path = Path(config.original_path)
        quantized_path = Path(config.quantized_path)

        if not original_path.exists():
            raise FileNotFoundError(f"Original model not found: {config.original_path}")

        if not quantized_path.exists():
            raise FileNotFoundError(
                f"Quantized model not found: {config.quantized_path}"
            )

        # Determine threshold
        if config.min_accuracy is not None:
            threshold = config.min_accuracy
        elif config.bit_width is not None:
            threshold = get_standard_threshold(config.bit_width)
        else:
            threshold = 0.85  # Default threshold

        per_layer_threshold = config.per_layer_threshold or threshold

        logger.info(f"Validating quantized model...")
        logger.info(f"  Original: {config.original_path}")
        logger.info(f"  Quantized: {config.quantized_path}")
        logger.info(f"  Threshold: {threshold:.2f}")
        logger.info(f"  Per-layer threshold: {per_layer_threshold:.2f}")

        # Create quantizer and validate
        quantizer = ArrowQuantV2(mode="diffusion")

        start_time = time.time()
        result = quantizer.validate_quality(
            original_path=str(original_path),
            quantized_path=str(quantized_path),
        )
        validation_time = time.time() - start_time

        # Extract results
        cosine_similarity = result["cosine_similarity"]
        per_layer_accuracy = result["per_layer_accuracy"]
        compression_ratio = result["compression_ratio"]
        model_size_mb = result["model_size_mb"]

        # Compute statistics
        accuracies = list(per_layer_accuracy.values())
        statistics = {
            "min": min(accuracies) if accuracies else 0.0,
            "max": max(accuracies) if accuracies else 0.0,
            "mean": sum(accuracies) / len(accuracies) if accuracies else 0.0,
            "std": (
                (
                    sum((x - sum(accuracies) / len(accuracies)) ** 2 for x in accuracies)
                    / len(accuracies)
                )
                ** 0.5
                if accuracies
                else 0.0
            ),
        }

        # Check for warnings
        warnings = []
        errors = []

        # Check overall threshold
        passed = cosine_similarity >= threshold
        if not passed:
            warnings.append(
                f"Overall accuracy {cosine_similarity:.4f} below threshold {threshold:.4f}"
            )

        # Check per-layer thresholds
        layers_below_threshold = [
            (name, acc)
            for name, acc in per_layer_accuracy.items()
            if acc < per_layer_threshold
        ]

        if layers_below_threshold:
            warnings.append(
                f"{len(layers_below_threshold)} layer(s) below per-layer threshold {per_layer_threshold:.4f}"
            )
            for name, acc in layers_below_threshold[:5]:  # Show first 5
                warnings.append(f"  - {name}: {acc:.4f}")
            if len(layers_below_threshold) > 5:
                warnings.append(
                    f"  ... and {len(layers_below_threshold) - 5} more layers"
                )

        # Log results
        logger.info(f"\nValidation Results:")
        logger.info(f"  Overall accuracy: {cosine_similarity:.4f}")
        logger.info(f"  Threshold: {threshold:.4f}")
        logger.info(f"  Passed: {passed}")
        logger.info(f"  Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"  Model size: {model_size_mb:.2f} MB")
        logger.info(f"  Validation time: {validation_time:.2f}s")
        logger.info(f"\nPer-Layer Statistics:")
        logger.info(f"  Min: {statistics['min']:.4f}")
        logger.info(f"  Max: {statistics['max']:.4f}")
        logger.info(f"  Mean: {statistics['mean']:.4f}")
        logger.info(f"  Std: {statistics['std']:.4f}")
        logger.info(f"  Total layers: {len(per_layer_accuracy)}")

        if warnings:
            logger.warning(f"\nWarnings:")
            for warning in warnings:
                logger.warning(f"  {warning}")

        return ValidationResult(
            passed=passed,
            cosine_similarity=cosine_similarity,
            compression_ratio=compression_ratio,
            model_size_mb=model_size_mb,
            per_layer_accuracy=per_layer_accuracy,
            statistics=statistics,
            threshold=threshold,
            bit_width=config.bit_width,
            validation_time_s=validation_time,
            warnings=warnings,
            errors=errors,
        )

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        logger.exception("Full traceback:")
        return ValidationResult(
            passed=False,
            cosine_similarity=0.0,
            compression_ratio=0.0,
            model_size_mb=0.0,
            per_layer_accuracy={},
            statistics={},
            threshold=threshold if "threshold" in locals() else 0.85,
            bit_width=config.bit_width,
            validation_time_s=0.0,
            warnings=[],
            errors=[str(e)],
        )


def generate_json_report(
    config: ValidationConfig, result: ValidationResult, output_path: str
):
    """Generate JSON validation report."""
    report = {
        "validation": {
            "original_path": config.original_path,
            "quantized_path": config.quantized_path,
            "threshold": result.threshold,
            "bit_width": result.bit_width,
            "validation_time_s": result.validation_time_s,
        },
        "results": {
            "passed": result.passed,
            "cosine_similarity": result.cosine_similarity,
            "compression_ratio": result.compression_ratio,
            "model_size_mb": result.model_size_mb,
        },
        "statistics": result.statistics,
        "per_layer_accuracy": result.per_layer_accuracy,
        "warnings": result.warnings,
        "errors": result.errors,
    }

    try:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"JSON report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON report: {e}")


def generate_html_report(
    config: ValidationConfig, result: ValidationResult, output_path: str
):
    """Generate HTML validation report."""
    # Sort layers by accuracy (worst first)
    sorted_layers = sorted(
        result.per_layer_accuracy.items(), key=lambda x: x[1]
    )

    # Generate HTML
    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>ArrowQuant V2 Validation Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #4CAF50;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 30px;
        }}
        .status {{
            font-size: 24px;
            font-weight: bold;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .status.passed {{
            background-color: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }}
        .status.failed {{
            background-color: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric {{
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 5px;
            border-left: 4px solid #4CAF50;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .accuracy-bar {{
            height: 20px;
            background-color: #e0e0e0;
            border-radius: 10px;
            overflow: hidden;
        }}
        .accuracy-fill {{
            height: 100%;
            background-color: #4CAF50;
            transition: width 0.3s;
        }}
        .accuracy-fill.low {{
            background-color: #f44336;
        }}
        .accuracy-fill.medium {{
            background-color: #ff9800;
        }}
        .warning {{
            background-color: #fff3cd;
            border: 1px solid #ffc107;
            color: #856404;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
        .error {{
            background-color: #f8d7da;
            border: 1px solid #f5c6cb;
            color: #721c24;
            padding: 15px;
            border-radius: 5px;
            margin: 10px 0;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>ArrowQuant V2 Validation Report</h1>
        
        <div class="status {'passed' if result.passed else 'failed'}">
            {'✓ VALIDATION PASSED' if result.passed else '✗ VALIDATION FAILED'}
        </div>
        
        <h2>Configuration</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Value</th>
            </tr>
            <tr>
                <td>Original Model</td>
                <td>{config.original_path}</td>
            </tr>
            <tr>
                <td>Quantized Model</td>
                <td>{config.quantized_path}</td>
            </tr>
            <tr>
                <td>Threshold</td>
                <td>{result.threshold:.4f}</td>
            </tr>
            <tr>
                <td>Bit Width</td>
                <td>{f'INT{result.bit_width}' if result.bit_width else 'N/A'}</td>
            </tr>
            <tr>
                <td>Validation Time</td>
                <td>{result.validation_time_s:.2f}s</td>
            </tr>
        </table>
        
        <h2>Overall Metrics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Cosine Similarity</div>
                <div class="metric-value">{result.cosine_similarity:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Compression Ratio</div>
                <div class="metric-value">{result.compression_ratio:.2f}x</div>
            </div>
            <div class="metric">
                <div class="metric-label">Model Size</div>
                <div class="metric-value">{result.model_size_mb:.1f} MB</div>
            </div>
            <div class="metric">
                <div class="metric-label">Total Layers</div>
                <div class="metric-value">{len(result.per_layer_accuracy)}</div>
            </div>
        </div>
        
        <h2>Per-Layer Statistics</h2>
        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Minimum</div>
                <div class="metric-value">{result.statistics['min']:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Maximum</div>
                <div class="metric-value">{result.statistics['max']:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Mean</div>
                <div class="metric-value">{result.statistics['mean']:.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Std Dev</div>
                <div class="metric-value">{result.statistics['std']:.4f}</div>
            </div>
        </div>
"""

    # Add warnings
    if result.warnings:
        html += """
        <h2>Warnings</h2>
"""
        for warning in result.warnings:
            html += f'        <div class="warning">{warning}</div>\n'

    # Add errors
    if result.errors:
        html += """
        <h2>Errors</h2>
"""
        for error in result.errors:
            html += f'        <div class="error">{error}</div>\n'

    # Add per-layer accuracy table
    html += """
        <h2>Per-Layer Accuracy</h2>
        <table>
            <tr>
                <th>Layer Name</th>
                <th>Cosine Similarity</th>
                <th>Accuracy Bar</th>
            </tr>
"""

    for layer_name, accuracy in sorted_layers:
        bar_class = "low" if accuracy < 0.7 else "medium" if accuracy < 0.9 else ""
        html += f"""            <tr>
                <td>{layer_name}</td>
                <td>{accuracy:.4f}</td>
                <td>
                    <div class="accuracy-bar">
                        <div class="accuracy-fill {bar_class}" style="width: {accuracy*100}%"></div>
                    </div>
                </td>
            </tr>
"""

    html += """        </table>
    </div>
</body>
</html>
"""

    try:
        with open(output_path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save HTML report: {e}")


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)

    # Create configuration
    config = ValidationConfig(
        original_path=args.original,
        quantized_path=args.quantized,
        min_accuracy=args.min_accuracy,
        bit_width=args.bit_width,
        report_path=args.report,
        html_report_path=args.html_report,
        per_layer_threshold=args.per_layer_threshold,
        fail_on_threshold=args.fail_on_threshold,
    )

    # Validate model
    logger.info("Starting validation...")
    result = validate_model(config)

    # Generate reports
    if args.report:
        generate_json_report(config, result, args.report)

    if args.html_report:
        generate_html_report(config, result, args.html_report)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Status: {'PASSED' if result.passed else 'FAILED'}")
    logger.info(f"Overall accuracy: {result.cosine_similarity:.4f}")
    logger.info(f"Threshold: {result.threshold:.4f}")
    logger.info(f"Compression: {result.compression_ratio:.2f}x")
    logger.info(f"Model size: {result.model_size_mb:.2f} MB")
    logger.info("=" * 80)

    # Exit with appropriate code
    if args.fail_on_threshold and not result.passed:
        logger.error("Validation failed threshold check")
        sys.exit(1)
    elif result.errors:
        logger.error("Validation encountered errors")
        sys.exit(1)
    else:
        logger.info("Validation completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
