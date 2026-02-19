#!/usr/bin/env python3
"""
CLI tool for converting embedding models to Arrow/Parquet format.

This tool provides a command-line interface to the ModelConverter class,
allowing easy conversion of HuggingFace and sentence-transformers models
to optimized Arrow/Parquet format with zero-copy loading.

Usage:
    python -m llm_compression.tools.cli convert \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --output ./models/optimized/minilm \\
        --float16 \\
        --compression lz4

Example:
    # Convert with all optimizations
    python -m llm_compression.tools.cli convert \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --output ./models/minilm \\
        --float16 \\
        --validate

    # Convert without float16 optimization
    python -m llm_compression.tools.cli convert \\
        --model bert-base-uncased \\
        --output ./models/bert \\
        --no-float16 \\
        --model-type transformers
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from llm_compression.logger import logger
from llm_compression.tools.model_converter import (
    ConversionConfig,
    ConversionResult,
    ModelConverter,
)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="Convert embedding models to Arrow/Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Convert command
    convert_parser = subparsers.add_parser(
        "convert",
        help="Convert a model to Arrow/Parquet format",
    )

    # Required arguments
    convert_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="HuggingFace model name or local path (e.g., sentence-transformers/all-MiniLM-L6-v2)",
    )

    convert_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for converted model",
    )

    # Optional arguments
    convert_parser.add_argument(
        "--model-type",
        "-t",
        choices=["sentence-transformers", "transformers"],
        default="sentence-transformers",
        help="Type of model to convert (default: sentence-transformers)",
    )

    convert_parser.add_argument(
        "--compression",
        "-c",
        choices=["lz4", "snappy", "gzip", "zstd", "none"],
        default="lz4",
        help="Compression algorithm for Parquet (default: lz4)",
    )

    convert_parser.add_argument(
        "--float16",
        action="store_true",
        default=False,
        help="Enable float16 optimization (reduces size by ~50%%)",
    )

    convert_parser.add_argument(
        "--no-float16",
        action="store_true",
        default=False,
        help="Disable float16 optimization (keeps float32)",
    )

    convert_parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        default=False,
        help="Skip tokenizer export",
    )

    convert_parser.add_argument(
        "--validate",
        action="store_true",
        default=False,
        help="Validate conversion output (slower but safer)",
    )

    convert_parser.add_argument(
        "--no-validate",
        action="store_true",
        default=False,
        help="Skip validation (faster but less safe)",
    )

    convert_parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    convert_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser


def handle_convert(args: argparse.Namespace) -> int:
    """
    Handle the convert command.

    Args:
        args: Parsed command-line arguments

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    # Set logging level
    if args.verbose:
        logger.setLevel("DEBUG")

    # Resolve float16 flag (--float16 takes precedence over --no-float16)
    use_float16 = args.float16 or not args.no_float16
    if args.float16 and args.no_float16:
        logger.warning("Both --float16 and --no-float16 specified. Using --float16.")
        use_float16 = True

    # Resolve validation flag
    validate_output = args.validate or not args.no_validate
    if args.validate and args.no_validate:
        logger.warning("Both --validate and --no-validate specified. Using --validate.")
        validate_output = True

    # Create configuration
    config = ConversionConfig(
        compression=args.compression if args.compression != "none" else None,
        use_float16=use_float16,
        extract_tokenizer=not args.no_tokenizer,
        validate_output=validate_output,
    )

    logger.info(f"Converting model: {args.model}")
    logger.info(f"Output directory: {args.output}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Configuration: {config}")

    # Create converter and run conversion
    converter = ModelConverter(config)

    try:
        result = converter.convert(
            model_name_or_path=args.model,
            output_dir=args.output,
            model_type=args.model_type,
        )

        # Output results
        if args.json:
            print(result.to_json())
        else:
            print_result(result)

        return 0 if result.success else 1

    except Exception as e:
        logger.error(f"Conversion failed with exception: {e}", exc_info=True)
        if args.json:
            error_result = ConversionResult(
                success=False,
                model_name=args.model,
                output_dir=Path(args.output),
                error_message=str(e),
            )
            print(error_result.to_json())
        else:
            print(f"\n❌ Conversion failed: {e}", file=sys.stderr)
        return 1


def print_result(result: ConversionResult) -> None:
    """
    Print conversion result in human-readable format.

    Args:
        result: Conversion result to print
    """
    if result.success:
        print("\n" + "=" * 60)
        print("✅ CONVERSION SUCCESSFUL")
        print("=" * 60)
        print(f"\nModel: {result.model_name}")
        print(f"Output: {result.output_dir}")
        print(f"\n📊 Statistics:")
        print(f"  • Total parameters: {result.total_parameters:,}")
        print(f"  • File size: {result.file_size_mb:.2f} MB")
        print(f"  • Compression ratio: {result.compression_ratio:.2f}x")
        print(f"  • Conversion time: {result.conversion_time_sec:.2f}s")

        if result.validation_passed:
            print(f"\n✓ Validation: PASSED")
        elif result.metadata.get("conversion_config", {}).get("validate_output"):
            print(f"\n⚠ Validation: FAILED")

        print(f"\n📁 Output files:")
        if result.parquet_path:
            print(f"  • Weights: {result.parquet_path}")
        if result.tokenizer_path:
            print(f"  • Tokenizer: {result.tokenizer_path}")
        if result.metadata_path:
            print(f"  • Metadata: {result.metadata_path}")

        print("\n" + "=" * 60 + "\n")
    else:
        print("\n" + "=" * 60)
        print("❌ CONVERSION FAILED")
        print("=" * 60)
        print(f"\nModel: {result.model_name}")
        print(f"Output: {result.output_dir}")
        print(f"\nError: {result.error_message}")
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

    if args.command == "convert":
        return handle_convert(args)

    return 1


if __name__ == "__main__":
    sys.exit(main())
