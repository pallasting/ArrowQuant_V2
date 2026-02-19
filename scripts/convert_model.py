#!/usr/bin/env python3
"""
Unified Model Converter - Convert any supported model to Arrow/Parquet format

This script provides a unified interface for converting HuggingFace models to
optimized Arrow/Parquet format for zero-copy loading with ArrowEngine.

Supported Model Types:
- BERT and sentence-transformers models (text embeddings)
- CLIP vision models (image embeddings)
- Whisper audio models (audio embeddings)

Features:
- Automatic model type detection
- Float16 optimization for reduced memory footprint
- Zstandard compression with configurable levels
- Comprehensive validation and metadata generation

Usage:
    python scripts/convert_model.py --model <name> --output <dir> [options]

Examples:
    # Auto-detect model type (recommended)
    python scripts/convert_model.py \\
        --model openai/clip-vit-base-patch32 \\
        --output D:/ai-models/clip-vit-b32
    
    # Explicit model type
    python scripts/convert_model.py \\
        --model openai/whisper-base \\
        --output D:/ai-models/whisper-base \\
        --type whisper
    
    # Custom compression settings
    python scripts/convert_model.py \\
        --model sentence-transformers/all-MiniLM-L6-v2 \\
        --output D:/ai-models/minilm \\
        --compression zstd \\
        --compression-level 9
    
    # Skip validation for faster conversion
    python scripts/convert_model.py \\
        --model openai/clip-vit-base-patch32 \\
        --output D:/ai-models/clip-vit-b32 \\
        --no-validate

Author: ArrowEngine Team
Version: 0.2.0
"""

import argparse
import sys
from pathlib import Path

from llm_compression.tools import ModelConverter, ConversionConfig
from llm_compression.logger import logger


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace models to Arrow/Parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect model type
  python scripts/convert_model.py --model openai/clip-vit-base-patch32 --output models/clip

  # Explicit model type
  python scripts/convert_model.py --model openai/whisper-base --output models/whisper --type whisper

  # High compression
  python scripts/convert_model.py --model bert-base-uncased --output models/bert --compression-level 9

For more information, see: docs/QUICKSTART_MULTIMODAL.md
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="HuggingFace model name or local path (e.g., 'openai/clip-vit-base-patch32')"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for converted model"
    )
    
    # Optional arguments
    parser.add_argument(
        "--type",
        type=str,
        default="auto",
        choices=["auto", "bert", "clip", "whisper", "sentence-transformers", "transformers"],
        help="Model type (default: auto-detect)"
    )
    parser.add_argument(
        "--float16",
        action="store_true",
        default=True,
        help="Convert weights to float16 (default: True)"
    )
    parser.add_argument(
        "--no-float16",
        action="store_false",
        dest="float16",
        help="Keep weights in float32"
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip validation (faster but less safe)"
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Skip tokenizer export (for vision/audio models)"
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zstd",
        choices=["zstd", "lz4", "snappy", "gzip"],
        help="Compression algorithm (default: zstd)"
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        default=3,
        help="Compression level for zstd (1-22, default: 3)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("  ArrowEngine Model Converter")
    print("  Convert HuggingFace models to optimized Arrow/Parquet format")
    print("=" * 70)
    print()


def print_result_summary(result):
    """Print conversion result summary."""
    print()
    print("=" * 70)
    print("  Conversion Summary")
    print("=" * 70)
    print(f"  Model:           {result.model_name}")
    print(f"  Output:          {result.output_dir}")
    print(f"  Parameters:      {result.total_parameters:,}")
    print(f"  File size:       {result.file_size_mb:.2f} MB")
    print(f"  Compression:     {result.compression_ratio:.2f}x")
    print(f"  Time:            {result.conversion_time_sec:.2f} seconds")
    print(f"  Validation:      {'PASSED' if result.validation_passed else 'SKIPPED'}")
    print("=" * 70)
    print()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Print banner
    print_banner()
    
    # Create conversion config
    config = ConversionConfig(
        compression=args.compression,
        compression_level=args.compression_level,
        use_float16=args.float16,
        extract_tokenizer=not args.no_tokenizer,
        validate_output=not args.no_validate
    )
    
    # Log configuration
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Output: {args.output}")
    logger.info(f"  Type: {args.type}")
    logger.info(f"  Float16: {config.use_float16}")
    logger.info(f"  Compression: {config.compression} (level {config.compression_level})")
    logger.info(f"  Validate: {config.validate_output}")
    logger.info(f"  Extract tokenizer: {config.extract_tokenizer}")
    print()
    
    # Create converter
    converter = ModelConverter(config)
    
    # Convert model
    logger.info(f"Starting conversion of {args.model}...")
    print()
    
    try:
        result = converter.convert(
            model_name_or_path=args.model,
            output_dir=args.output,
            model_type=args.type
        )
        
        # Check result
        if result.success:
            print_result_summary(result)
            logger.info("✅ SUCCESS: Model converted successfully")
            
            # Print next steps
            print("Next steps:")
            print(f"  1. Load the model: ArrowEngine.from_pretrained('{args.output}')")
            print(f"  2. See examples: examples/multimodal_complete_examples.py")
            print(f"  3. Read docs: docs/QUICKSTART_MULTIMODAL.md")
            print()
            
            return 0
        else:
            logger.error(f"❌ FAILED: Conversion failed")
            logger.error(f"Error: {result.error_message}")
            print()
            print("Troubleshooting:")
            print("  1. Check that the model name is correct")
            print("  2. Ensure you have internet connection (for downloading)")
            print("  3. Try with --no-validate flag")
            print("  4. Check logs above for detailed error messages")
            print()
            return 1
            
    except KeyboardInterrupt:
        logger.warning("Conversion interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
