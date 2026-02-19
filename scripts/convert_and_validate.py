"""
Convert model and run validation tests for ArrowEngine.

This script:
1. Converts sentence-transformers model to ArrowEngine format
2. Runs precision validation tests
3. Runs performance benchmarks

Usage:
    python scripts/convert_and_validate.py
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger
from llm_compression.tools.model_converter import ModelConverter, ConversionConfig


def convert_model():
    """Convert sentence-transformers model to ArrowEngine format."""
    logger.info("=" * 60)
    logger.info("Step 1: Converting Model")
    logger.info("=" * 60)
    
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = "./models/minilm"
    
    # Check if already converted
    if Path(output_dir).exists():
        logger.info(f"Model already exists at {output_dir}, skipping conversion")
        return True
    
    try:
        # Create converter with float16 optimization
        config = ConversionConfig(
            use_float16=True,
            validate_output=True,
            compression="lz4",
            extract_tokenizer=True
        )
        converter = ModelConverter(config)
        
        # Convert model
        logger.info(f"Converting {model_name} to {output_dir}...")
        result = converter.convert(
            model_name_or_path=model_name,
            output_dir=output_dir,
            model_type="sentence-transformers"
        )
        
        if result.success:
            logger.info(f"✅ Conversion successful!")
            logger.info(f"   Output: {result.output_dir}")
            logger.info(f"   Original size: {result.original_size_mb:.1f} MB")
            logger.info(f"   Compressed size: {result.compressed_size_mb:.1f} MB")
            logger.info(f"   Compression ratio: {result.compression_ratio:.2f}x")
            return True
        else:
            logger.error(f"❌ Conversion failed: {result.error_message}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Conversion failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_precision_tests():
    """Run precision validation tests."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Running Precision Validation")
    logger.info("=" * 60)
    
    import subprocess
    
    # Ensure package is installed in development mode
    logger.info("Ensuring package is installed...")
    try:
        subprocess.run(
            ["pip", "install", "-e", "."],
            capture_output=True,
            text=True,
            check=True
        )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Package installation warning: {e}")
    
    try:
        result = subprocess.run(
            ["pytest", "tests/integration/inference/test_e2e_precision.py", "-v", "-s"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Precision tests passed!")
            return True
        else:
            logger.error("❌ Precision tests failed!")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run precision tests: {e}")
        return False


def run_performance_benchmark():
    """Run performance benchmarks."""
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Running Performance Benchmark")
    logger.info("=" * 60)
    
    try:
        import subprocess
        
        result = subprocess.run(
            ["python", "benchmarks/arrowengine_benchmark.py"],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ Performance benchmarks passed!")
            return True
        else:
            logger.warning("⚠️  Some performance targets not met")
            return False
            
    except Exception as e:
        logger.error(f"❌ Failed to run benchmarks: {e}")
        return False


def main():
    """Main execution flow."""
    logger.info("ArrowEngine Conversion and Validation")
    logger.info("=" * 60)
    
    # Step 1: Convert model
    if not convert_model():
        logger.error("Model conversion failed, aborting")
        sys.exit(1)
    
    # Step 2: Run precision tests
    if not run_precision_tests():
        logger.error("Precision validation failed, aborting")
        sys.exit(1)
    
    # Step 3: Run performance benchmarks
    run_performance_benchmark()  # Don't abort on performance issues
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ Validation Complete!")
    logger.info("=" * 60)
    logger.info("\nNext steps:")
    logger.info("  1. Review test results above")
    logger.info("  2. If all tests passed, proceed to Phase 3 (Semantic Indexing)")
    logger.info("  3. Run: Check tasks.md for next tasks")


if __name__ == "__main__":
    main()
