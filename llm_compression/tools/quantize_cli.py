#!/usr/bin/env python3
"""
CLI tool for quantizing model weights to INT8/INT2 format.

This tool provides a command-line interface to the ArrowQuantizer class,
allowing easy quantization of model weights stored in Arrow/Parquet format
with automatic precision validation.

Usage:
    python -m llm_compression.tools.quantize_cli quantize \\
        --input ./models/minilm/weights.parquet \\
        --output ./models/minilm/weights_int8.parquet \\
        --quant-type int8 \\
        --validate

Example:
    # Quantize to INT8 with validation
    python -m llm_compression.tools.quantize_cli quantize \\
        --input ./models/minilm/weights.parquet \\
        --output ./models/minilm/weights_int8.parquet \\
        --quant-type int8 \\
        --per-channel \\
        --validate

    # Quantize to INT2 with mixed precision
    python -m llm_compression.tools.quantize_cli quantize \\
        --input ./models/qwen/weights.parquet \\
        --output ./models/qwen/weights_int2.parquet \\
        --quant-type int2 \\
        --mixed-precision lm_head embed \\
        --validate

    # Load configuration from YAML
    python -m llm_compression.tools.quantize_cli quantize \\
        --config quantization_config.yaml
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, List

import yaml

from llm_compression.logger import logger
from llm_compression.inference.arrow_quantizer import (
    ArrowQuantizer,
    QuantizationConfig,
)
from llm_compression.inference.precision_validator import (
    PrecisionValidator,
    ValidationResult,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Quantize model weights to INT8/INT2 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Quantize command
    quantize_parser = subparsers.add_parser(
        "quantize",
        help="Quantize model weights",
    )

    # Configuration file
    quantize_parser.add_argument(
        "--config",
        "-c",
        type=str,
        help="Path to YAML configuration file (overrides other arguments)",
    )

    # Input/Output
    quantize_parser.add_argument(
        "--input",
        "-i",
        type=str,
        help="Input Parquet file (Schema V1)",
    )

    quantize_parser.add_argument(
        "--output",
        "-o",
        type=str,
        help="Output Parquet file (Schema V2)",
    )

    # Quantization parameters
    quantize_parser.add_argument(
        "--quant-type",
        "-q",
        choices=["int8", "int2", "fp16"],
        default="int8",
        help="Quantization type (default: int8)",
    )

    quantize_parser.add_argument(
        "--calibration-method",
        choices=["ptq", "gptq"],
        default="ptq",
        help="Calibration method (default: ptq)",
    )

    quantize_parser.add_argument(
        "--calibration-data-path",
        type=str,
        help="Path to pre-computed layer-wise calibration data (.pt file)",
    )

    quantize_parser.add_argument(
        "--per-channel",
        action="store_true",
        default=True,
        help="Use per-channel quantization (default: True)",
    )

    quantize_parser.add_argument(
        "--per-tensor",
        action="store_true",
        help="Use per-tensor quantization (overrides --per-channel)",
    )

    quantize_parser.add_argument(
        "--symmetric",
        action="store_true",
        default=True,
        help="Use symmetric quantization (default: True)",
    )

    quantize_parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="Use asymmetric quantization (overrides --symmetric)",
    )

    quantize_parser.add_argument(
        "--mixed-precision",
        nargs="+",
        type=str,
        help="Layer name patterns to skip quantization (keep FP16)",
    )

    # Validation parameters
    quantize_parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Run precision validation after quantization",
    )

    quantize_parser.add_argument(
        "--test-texts",
        nargs="+",
        type=str,
        help="Test texts for validation (default: built-in test set)",
    )

    quantize_parser.add_argument(
        "--cosine-threshold",
        type=float,
        default=0.95,
        help="Minimum cosine similarity threshold (default: 0.95)",
    )

    quantize_parser.add_argument(
        "--ppl-threshold",
        type=float,
        default=0.15,
        help="Maximum PPL increase threshold (default: 0.15 = 15%%)",
    )

    quantize_parser.add_argument(
        "--validate-ppl",
        action="store_true",
        help="Validate perplexity (for language models)",
    )

    # Output options
    quantize_parser.add_argument(
        "--report",
        "-r",
        type=str,
        help="Path to save validation report (JSON)",
    )

    quantize_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    quantize_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def load_config_from_yaml(config_path: str) -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary

    Example YAML:
        quantization:
          input: ./models/minilm/weights.parquet
          output: ./models/minilm/weights_int8.parquet
          quant_type: int8
          calibration_method: ptq
          per_channel: true
          symmetric: true
          mixed_precision_layers:
            - lm_head
            - embed

        validation:
          enabled: true
          cosine_threshold: 0.95
          ppl_threshold: 0.15
          validate_ppl: false
          test_texts:
            - "Hello world"
            - "Test sentence"
    """
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded configuration from {config_path}")
    return config


def merge_config(yaml_config: dict, args: argparse.Namespace) -> dict:
    """
    Merge YAML configuration with command-line arguments.

    Command-line arguments take precedence over YAML configuration.

    Args:
        yaml_config: Configuration from YAML file
        args: Command-line arguments

    Returns:
        Merged configuration dictionary
    """
    merged = yaml_config.copy()

    # Merge quantization parameters
    quant_config = merged.get("quantization", {})
    if args.input:
        quant_config["input"] = args.input
    if args.output:
        quant_config["output"] = args.output
    if args.quant_type:
        quant_config["quant_type"] = args.quant_type
    if args.calibration_method:
        quant_config["calibration_method"] = args.calibration_method
    if args.per_tensor:
        quant_config["per_channel"] = False
    if args.asymmetric:
        quant_config["symmetric"] = False
    if args.mixed_precision:
        quant_config["mixed_precision_layers"] = args.mixed_precision

    merged["quantization"] = quant_config

    # Merge validation parameters
    val_config = merged.get("validation", {})
    if args.validate:
        val_config["enabled"] = True
    if args.test_texts:
        val_config["test_texts"] = args.test_texts
    if args.cosine_threshold:
        val_config["cosine_threshold"] = args.cosine_threshold
    if args.ppl_threshold:
        val_config["ppl_threshold"] = args.ppl_threshold
    if args.validate_ppl:
        val_config["validate_ppl"] = True

    merged["validation"] = val_config

    return merged


def get_default_test_texts() -> List[str]:
    """
    Get default test texts for validation.

    Returns:
        List of test texts
    """
    return [
        "Hello world",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning is a subset of artificial intelligence",
        "Natural language processing enables computers to understand human language",
        "Deep learning models have revolutionized computer vision",
        "Transformers are the foundation of modern language models",
        "Quantization reduces model size while preserving accuracy",
        "Zero-copy loading improves inference performance",
        "Semantic search finds relevant information based on meaning",
        "Embeddings capture semantic relationships between words",
    ]


def handle_quantize(args: argparse.Namespace) -> int:
    """
    Handle the quantize command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    start_time = time.time()

    import logging

    # Set logging level
    if args.json:
        # Suppress all logs to keep stdout clean for JSON output
        logger.setLevel(logging.CRITICAL)
    elif args.verbose:
        logger.setLevel("DEBUG")

    # Load configuration
    if args.config:
        yaml_config = load_config_from_yaml(args.config)
        config = merge_config(yaml_config, args)
    else:
        config = {
            "quantization": {
                "input": args.input,
                "output": args.output,
                "quant_type": args.quant_type,
                "calibration_method": args.calibration_method,
                "per_channel": not args.per_tensor,
                "symmetric": not args.asymmetric,
                "mixed_precision_layers": args.mixed_precision,
            },
            "validation": {
                "enabled": args.validate,
                "test_texts": args.test_texts,
                "cosine_threshold": args.cosine_threshold,
                "ppl_threshold": args.ppl_threshold,
                "validate_ppl": args.validate_ppl,
            },
        }

    # Validate required parameters
    quant_config = config["quantization"]
    if not quant_config.get("input"):
        logger.error("Input file is required (--input or config file)")
        return 1
    if not quant_config.get("output"):
        logger.error("Output file is required (--output or config file)")
        return 1

    input_path = quant_config["input"]
    output_path = quant_config["output"]

    logger.info("=" * 60)
    logger.info("QUANTIZATION PIPELINE")
    logger.info("=" * 60)
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")
    logger.info(f"Quantization type: {quant_config['quant_type']}")
    logger.info(f"Calibration method: {quant_config['calibration_method']}")
    logger.info(f"Per-channel: {quant_config['per_channel']}")
    logger.info(f"Symmetric: {quant_config['symmetric']}")
    if quant_config.get("mixed_precision_layers"):
        logger.info(f"Mixed precision layers: {quant_config['mixed_precision_layers']}")
    logger.info("=" * 60)

    try:
        # Create quantization configuration
        quantization_config = QuantizationConfig(
            quant_type=quant_config["quant_type"],
            calibration_method=quant_config["calibration_method"],
            per_channel=quant_config["per_channel"],
            symmetric=quant_config["symmetric"],
            mixed_precision_layers=quant_config.get("mixed_precision_layers"),
        )

        # Create quantizer
        quantizer = ArrowQuantizer(quantization_config)

        # Load calibration data if GPTQ is used
        calibration_data = None
        if quantization_config.calibration_method == "gptq":
            if hasattr(args, "calibration_data_path") and args.calibration_data_path:
                logger.info(f"Loading calibration data from {args.calibration_data_path}...")
                import torch
                calibration_data = torch.load(args.calibration_data_path)
                logger.info(f"Successfully loaded calibration data for {len(calibration_data)} layers.")
            else:
                logger.warning("GPTQ requested but no --calibration-data-path provided. Will fallback to PTQ. Please pre-compute calibration data using generate_calibration_data.py.")

        # Run quantization
        logger.info("\n🔄 Starting quantization...")
        quantizer.quantize_model(
            input_parquet=input_path,
            output_parquet=output_path,
            calibration_data=calibration_data,
            show_progress=not args.json,  # Show progress unless JSON output
        )

        quantization_time = time.time() - start_time
        logger.info(f"✅ Quantization complete in {quantization_time:.2f}s")

        # Calculate compression metrics
        input_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
        output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
        compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0
        memory_savings = (1 - output_size_mb / input_size_mb) * 100 if input_size_mb > 0 else 0

        logger.info("\n📊 Compression Metrics:")
        logger.info(f"  • Input size: {input_size_mb:.2f} MB")
        logger.info(f"  • Output size: {output_size_mb:.2f} MB")
        logger.info(f"  • Compression ratio: {compression_ratio:.2f}x")
        logger.info(f"  • Memory savings: {memory_savings:.1f}%")

        # Run validation if requested
        validation_result = None
        val_config = config.get("validation", {})
        if val_config.get("enabled", False):
            logger.info("\n🔍 Running precision validation...")

            # Get test texts
            test_texts = val_config.get("test_texts")
            if not test_texts:
                test_texts = get_default_test_texts()
                logger.info(f"Using {len(test_texts)} default test texts")

            # Create validator
            validator = PrecisionValidator(
                cosine_threshold=val_config.get("cosine_threshold", 0.95),
                ppl_threshold=val_config.get("ppl_threshold", 0.15),
                validate_ppl=val_config.get("validate_ppl", False),
            )

            # Run validation
            validation_result = validator.validate(
                original_model_path=input_path,
                quantized_model_path=output_path,
                test_texts=test_texts,
            )

            # Save validation report if requested
            if args.report:
                report_path = args.report
                validator.generate_report(
                    validation_result,
                    output_path=report_path,
                    format="json",
                )
                logger.info(f"📄 Validation report saved to {report_path}")

        # Output results
        total_time = time.time() - start_time

        if args.json:
            result = {
                "success": True,
                "input": input_path,
                "output": output_path,
                "quantization": {
                    "quant_type": quant_config["quant_type"],
                    "calibration_method": quant_config["calibration_method"],
                    "per_channel": quant_config["per_channel"],
                    "symmetric": quant_config["symmetric"],
                    "time_sec": quantization_time,
                },
                "compression": {
                    "input_size_mb": input_size_mb,
                    "output_size_mb": output_size_mb,
                    "compression_ratio": compression_ratio,
                    "memory_savings_percent": memory_savings,
                },
                "validation": validation_result.to_dict() if validation_result else None,
                "total_time_sec": total_time,
            }
            print(json.dumps(result, indent=2))
        else:
            print_result(
                input_path=input_path,
                output_path=output_path,
                quant_config=quant_config,
                input_size_mb=input_size_mb,
                output_size_mb=output_size_mb,
                compression_ratio=compression_ratio,
                memory_savings=memory_savings,
                quantization_time=quantization_time,
                validation_result=validation_result,
                total_time=total_time,
            )

        # Return exit code based on validation result
        if validation_result:
            return 0 if validation_result.passed else 1
        return 0

    except Exception as e:
        logger.error(f"Quantization failed: {e}", exc_info=True)
        if args.json:
            error_result = {
                "success": False,
                "input": input_path,
                "output": output_path,
                "error": str(e),
            }
            print(json.dumps(error_result, indent=2))
        else:
            print(f"\n❌ Quantization failed: {e}", file=sys.stderr)
        return 1


def print_result(
    input_path: str,
    output_path: str,
    quant_config: dict,
    input_size_mb: float,
    output_size_mb: float,
    compression_ratio: float,
    memory_savings: float,
    quantization_time: float,
    validation_result: Optional[ValidationResult],
    total_time: float,
) -> None:
    """
    Print quantization result in human-readable format.

    Args:
        input_path: Input file path
        output_path: Output file path
        quant_config: Quantization configuration
        input_size_mb: Input file size in MB
        output_size_mb: Output file size in MB
        compression_ratio: Compression ratio
        memory_savings: Memory savings percentage
        quantization_time: Quantization time in seconds
        validation_result: Validation result (optional)
        total_time: Total time in seconds
    """
    print("\n" + "=" * 60)
    print("✅ QUANTIZATION SUCCESSFUL")
    print("=" * 60)
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"\n⚙️  Configuration:")
    print(f"  • Quantization type: {quant_config['quant_type']}")
    print(f"  • Calibration method: {quant_config['calibration_method']}")
    print(f"  • Per-channel: {quant_config['per_channel']}")
    print(f"  • Symmetric: {quant_config['symmetric']}")
    if quant_config.get("mixed_precision_layers"):
        print(f"  • Mixed precision layers: {', '.join(quant_config['mixed_precision_layers'])}")

    print(f"\n📊 Compression Metrics:")
    print(f"  • Input size: {input_size_mb:.2f} MB")
    print(f"  • Output size: {output_size_mb:.2f} MB")
    print(f"  • Compression ratio: {compression_ratio:.2f}x")
    print(f"  • Memory savings: {memory_savings:.1f}%")

    print(f"\n⏱️  Performance:")
    print(f"  • Quantization time: {quantization_time:.2f}s")
    print(f"  • Total time: {total_time:.2f}s")

    if validation_result:
        print(f"\n🔍 Validation:")
        if validation_result.passed:
            print(f"  ✅ Status: PASSED")
        else:
            print(f"  ❌ Status: FAILED")
        print(f"  • Cosine similarity: {validation_result.cosine_similarity:.4f}")
        print(f"    (min: {validation_result.min_cosine_similarity:.4f}, max: {validation_result.max_cosine_similarity:.4f})")
        if validation_result.ppl_increase is not None:
            print(f"  • PPL increase: {validation_result.ppl_increase:.2%}")
        print(f"  • Test samples: {validation_result.num_samples}")
        print(f"  • Validation time: {validation_result.validation_time_ms:.2f}ms")
        if validation_result.error_message:
            print(f"  • Error: {validation_result.error_message}")

    print("\n" + "=" * 60 + "\n")


def main() -> int:
    """
    Main entry point for CLI.

    Returns:
        Exit code
    """
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == "quantize":
        return handle_quantize(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
