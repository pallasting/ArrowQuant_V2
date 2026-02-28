#!/usr/bin/env python3
"""
ArrowQuant V2 Offline Quantization Script

Production-ready script for quantizing diffusion models with comprehensive
CLI support, progress reporting, error handling, and batch processing.

Usage:
    # Single model quantization
    python scripts/quantize_diffusion.py \\
        --model dream-7b/ \\
        --output dream-7b-int2/ \\
        --profile edge \\
        --bit-width 2

    # Batch quantization
    python scripts/quantize_diffusion.py \\
        --batch models_to_quantize.json \\
        --output-dir quantized_models/

    # Custom configuration
    python scripts/quantize_diffusion.py \\
        --model dream-7b/ \\
        --output dream-7b-custom/ \\
        --config custom_config.yaml \\
        --num-time-groups 15 \\
        --group-size 64

Author: ArrowQuant V2 Team
License: MIT
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("quantization.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class QuantizationJob:
    """Represents a single quantization job."""

    model_path: str
    output_path: str
    bit_width: Optional[int] = None
    profile: Optional[str] = None
    config_path: Optional[str] = None
    modality: Optional[str] = None
    num_time_groups: Optional[int] = None
    group_size: Optional[int] = None
    enable_time_aware: Optional[bool] = None
    enable_spatial: Optional[bool] = None
    min_accuracy: Optional[float] = None
    calibration_samples: Optional[int] = None
    fail_fast: Optional[bool] = None


@dataclass
class QuantizationResult:
    """Stores quantization results for reporting."""

    job: QuantizationJob
    success: bool
    quantized_path: Optional[str] = None
    compression_ratio: Optional[float] = None
    cosine_similarity: Optional[float] = None
    model_size_mb: Optional[float] = None
    modality: Optional[str] = None
    bit_width: Optional[int] = None
    quantization_time_s: Optional[float] = None
    error_message: Optional[str] = None


class ProgressReporter:
    """Handles progress reporting for quantization jobs."""

    def __init__(self, job_name: str, verbose: bool = True):
        self.job_name = job_name
        self.verbose = verbose
        self.start_time = time.time()

    def __call__(self, message: str, progress: float):
        """Progress callback for Python API."""
        if self.verbose:
            elapsed = time.time() - self.start_time
            logger.info(
                f"[{self.job_name}] {message} ({progress*100:.1f}% complete, {elapsed:.1f}s elapsed)"
            )


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ArrowQuant V2 Offline Quantization Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Edge deployment (INT2, <35MB)
  %(prog)s --model dream-7b/ --output dream-7b-edge/ --profile edge

  # Local deployment (INT4, balanced)
  %(prog)s --model dream-7b/ --output dream-7b-local/ --profile local

  # Cloud deployment (INT8, high accuracy)
  %(prog)s --model dream-7b/ --output dream-7b-cloud/ --profile cloud

  # Custom configuration
  %(prog)s --model dream-7b/ --output dream-7b-custom/ \\
           --bit-width 4 --num-time-groups 15 --group-size 64

  # Batch quantization
  %(prog)s --batch models.json --output-dir quantized/

For more information, see: docs/QUICKSTART.md
        """,
    )

    # Input/output arguments
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--model",
        type=str,
        help="Path to input model directory",
    )
    input_group.add_argument(
        "--batch",
        type=str,
        help="Path to JSON file with batch quantization jobs",
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Path to output quantized model directory (required for --model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch quantization (required for --batch)",
    )

    # Configuration arguments
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--profile",
        type=str,
        choices=["edge", "local", "cloud"],
        help="Deployment profile (edge/local/cloud)",
    )

    # Quantization parameters
    parser.add_argument(
        "--bit-width",
        type=int,
        choices=[2, 4, 8],
        help="Target bit width (2, 4, or 8)",
    )
    parser.add_argument(
        "--modality",
        type=str,
        choices=["text", "code", "image", "audio"],
        help="Model modality (auto-detected if not specified)",
    )
    parser.add_argument(
        "--num-time-groups",
        type=int,
        help="Number of time groups for time-aware quantization",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        choices=[32, 64, 128, 256],
        help="Group size for per-group quantization",
    )
    parser.add_argument(
        "--enable-time-aware",
        action="store_true",
        help="Enable time-aware quantization",
    )
    parser.add_argument(
        "--disable-time-aware",
        action="store_true",
        help="Disable time-aware quantization",
    )
    parser.add_argument(
        "--enable-spatial",
        action="store_true",
        help="Enable spatial quantization",
    )
    parser.add_argument(
        "--disable-spatial",
        action="store_true",
        help="Disable spatial quantization",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="Minimum cosine similarity threshold (0.0-1.0)",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        help="Number of calibration samples",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Disable automatic fallback to higher bit widths",
    )

    # Output control
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
    parser.add_argument(
        "--report",
        type=str,
        help="Path to save JSON summary report",
    )

    return parser.parse_args()


def load_batch_jobs(batch_file: str) -> List[QuantizationJob]:
    """Load batch quantization jobs from JSON file."""
    try:
        with open(batch_file, "r") as f:
            data = json.load(f)

        jobs = []
        for job_data in data.get("jobs", []):
            job = QuantizationJob(**job_data)
            jobs.append(job)

        logger.info(f"Loaded {len(jobs)} quantization jobs from {batch_file}")
        return jobs

    except Exception as e:
        logger.error(f"Failed to load batch file {batch_file}: {e}")
        sys.exit(1)


def create_config(job: QuantizationJob, args: argparse.Namespace) -> Dict[str, Any]:
    """Create configuration dictionary from job and CLI arguments."""
    config = {}

    # Priority: job-specific > CLI args > defaults
    if job.bit_width is not None:
        config["bit_width"] = job.bit_width
    elif args.bit_width is not None:
        config["bit_width"] = args.bit_width

    if job.modality is not None:
        config["modality"] = job.modality
    elif args.modality is not None:
        config["modality"] = args.modality

    if job.num_time_groups is not None:
        config["num_time_groups"] = job.num_time_groups
    elif args.num_time_groups is not None:
        config["num_time_groups"] = args.num_time_groups

    if job.group_size is not None:
        config["group_size"] = job.group_size
    elif args.group_size is not None:
        config["group_size"] = args.group_size

    # Handle boolean flags
    if job.enable_time_aware is not None:
        config["enable_time_aware"] = job.enable_time_aware
    elif args.enable_time_aware:
        config["enable_time_aware"] = True
    elif args.disable_time_aware:
        config["enable_time_aware"] = False

    if job.enable_spatial is not None:
        config["enable_spatial"] = job.enable_spatial
    elif args.enable_spatial:
        config["enable_spatial"] = True
    elif args.disable_spatial:
        config["enable_spatial"] = False

    if job.min_accuracy is not None:
        config["min_accuracy"] = job.min_accuracy
    elif args.min_accuracy is not None:
        config["min_accuracy"] = args.min_accuracy

    if job.calibration_samples is not None:
        config["calibration_samples"] = job.calibration_samples
    elif args.calibration_samples is not None:
        config["calibration_samples"] = args.calibration_samples

    if job.fail_fast is not None:
        config["fail_fast"] = job.fail_fast
    elif args.fail_fast:
        config["fail_fast"] = True

    # Deployment profile
    profile = job.profile or args.profile
    if profile:
        config["deployment_profile"] = profile

    return config


def quantize_single_model(
    job: QuantizationJob, args: argparse.Namespace
) -> QuantizationResult:
    """Quantize a single model."""
    try:
        # Import here to provide better error messages if not installed
        try:
            from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
        except ImportError as e:
            logger.error(
                "Failed to import arrow_quant_v2. "
                "Please install with: pip install -e ."
            )
            return QuantizationResult(
                job=job,
                success=False,
                error_message=f"Import error: {e}",
            )

        # Validate paths
        model_path = Path(job.model_path)
        if not model_path.exists():
            error_msg = f"Model path does not exist: {job.model_path}"
            logger.error(error_msg)
            return QuantizationResult(
                job=job,
                success=False,
                error_message=error_msg,
            )

        output_path = Path(job.output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create configuration
        config_dict = create_config(job, args)

        # Load from YAML if specified
        if job.config_path or args.config:
            config_file = job.config_path or args.config
            logger.info(f"Loading configuration from {config_file}")
            config = DiffusionQuantConfig.from_yaml(config_file)
            # Override with CLI/job-specific parameters
            for key, value in config_dict.items():
                setattr(config, key, value)
        elif config_dict.get("deployment_profile"):
            # Use profile
            profile = config_dict.pop("deployment_profile")
            logger.info(f"Using {profile} deployment profile")
            config = DiffusionQuantConfig.from_profile(profile)
            # Override with CLI/job-specific parameters
            for key, value in config_dict.items():
                setattr(config, key, value)
        else:
            # Create from parameters
            config = DiffusionQuantConfig(**config_dict)

        # Apply environment variable overrides
        config.apply_env_overrides()

        # Validate configuration
        try:
            config.validate()
        except Exception as e:
            error_msg = f"Configuration validation failed: {e}"
            logger.error(error_msg)
            return QuantizationResult(
                job=job,
                success=False,
                error_message=error_msg,
            )

        # Log configuration
        logger.info(f"Quantization configuration:")
        logger.info(f"  Model: {job.model_path}")
        logger.info(f"  Output: {job.output_path}")
        logger.info(f"  Bit width: {config.bit_width}")
        logger.info(f"  Profile: {config.deployment_profile}")
        logger.info(f"  Time groups: {config.num_time_groups}")
        logger.info(f"  Group size: {config.group_size}")
        logger.info(f"  Time-aware: {config.enable_time_aware}")
        logger.info(f"  Spatial: {config.enable_spatial}")
        logger.info(f"  Min accuracy: {config.min_accuracy}")
        logger.info(f"  Calibration samples: {config.calibration_samples}")

        # Create quantizer
        quantizer = ArrowQuantV2(mode="diffusion")

        # Create progress reporter
        job_name = Path(job.model_path).name
        progress_reporter = ProgressReporter(
            job_name=job_name,
            verbose=args.verbose and not args.quiet,
        )

        # Execute quantization
        logger.info(f"Starting quantization of {job.model_path}...")
        start_time = time.time()

        result = quantizer.quantize_diffusion_model(
            model_path=str(model_path),
            output_path=str(output_path),
            config=config,
            progress_callback=progress_reporter if not args.quiet else None,
        )

        elapsed_time = time.time() - start_time

        # Log results
        logger.info(f"Quantization completed successfully!")
        logger.info(f"  Output: {result['quantized_path']}")
        logger.info(f"  Size: {result['model_size_mb']:.2f} MB")
        logger.info(f"  Compression: {result['compression_ratio']:.2f}x")
        logger.info(f"  Accuracy: {result['cosine_similarity']:.4f}")
        logger.info(f"  Modality: {result['modality']}")
        logger.info(f"  Bit width: {result['bit_width']}")
        logger.info(f"  Time: {elapsed_time:.2f}s")

        return QuantizationResult(
            job=job,
            success=True,
            quantized_path=result["quantized_path"],
            compression_ratio=result["compression_ratio"],
            cosine_similarity=result["cosine_similarity"],
            model_size_mb=result["model_size_mb"],
            modality=result["modality"],
            bit_width=result["bit_width"],
            quantization_time_s=elapsed_time,
        )

    except Exception as e:
        error_msg = f"Quantization failed: {e}"
        logger.error(error_msg)
        logger.exception("Full traceback:")
        return QuantizationResult(
            job=job,
            success=False,
            error_message=error_msg,
        )


def generate_summary_report(
    results: List[QuantizationResult], output_path: Optional[str] = None
):
    """Generate and save summary report."""
    # Calculate statistics
    total_jobs = len(results)
    successful_jobs = sum(1 for r in results if r.success)
    failed_jobs = total_jobs - successful_jobs

    total_size_mb = sum(r.model_size_mb or 0 for r in results if r.success)
    avg_compression = (
        sum(r.compression_ratio or 0 for r in results if r.success) / successful_jobs
        if successful_jobs > 0
        else 0
    )
    avg_accuracy = (
        sum(r.cosine_similarity or 0 for r in results if r.success) / successful_jobs
        if successful_jobs > 0
        else 0
    )
    total_time_s = sum(r.quantization_time_s or 0 for r in results if r.success)

    # Create report
    report = {
        "summary": {
            "total_jobs": total_jobs,
            "successful": successful_jobs,
            "failed": failed_jobs,
            "success_rate": successful_jobs / total_jobs if total_jobs > 0 else 0,
            "total_size_mb": total_size_mb,
            "avg_compression_ratio": avg_compression,
            "avg_cosine_similarity": avg_accuracy,
            "total_time_s": total_time_s,
        },
        "results": [
            {
                "model_path": r.job.model_path,
                "output_path": r.job.output_path,
                "success": r.success,
                "quantized_path": r.quantized_path,
                "compression_ratio": r.compression_ratio,
                "cosine_similarity": r.cosine_similarity,
                "model_size_mb": r.model_size_mb,
                "modality": r.modality,
                "bit_width": r.bit_width,
                "quantization_time_s": r.quantization_time_s,
                "error_message": r.error_message,
            }
            for r in results
        ],
    }

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("QUANTIZATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Total jobs: {total_jobs}")
    logger.info(f"Successful: {successful_jobs}")
    logger.info(f"Failed: {failed_jobs}")
    logger.info(f"Success rate: {report['summary']['success_rate']*100:.1f}%")
    if successful_jobs > 0:
        logger.info(f"Total size: {total_size_mb:.2f} MB")
        logger.info(f"Avg compression: {avg_compression:.2f}x")
        logger.info(f"Avg accuracy: {avg_accuracy:.4f}")
        logger.info(f"Total time: {total_time_s:.2f}s")
    logger.info("=" * 80)

    # Save to file if requested
    if output_path:
        try:
            with open(output_path, "w") as f:
                json.dump(report, f, indent=2)
            logger.info(f"Summary report saved to {output_path}")
        except Exception as e:
            logger.error(f"Failed to save report to {output_path}: {e}")

    return report


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)

    # Validate arguments
    if args.model and not args.output:
        logger.error("--output is required when using --model")
        sys.exit(1)

    if args.batch and not args.output_dir:
        logger.error("--output-dir is required when using --batch")
        sys.exit(1)

    # Create jobs
    if args.model:
        # Single model quantization
        job = QuantizationJob(
            model_path=args.model,
            output_path=args.output,
            profile=args.profile,
            config_path=args.config,
        )
        jobs = [job]
    else:
        # Batch quantization
        jobs = load_batch_jobs(args.batch)
        # Update output paths to use output_dir
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            if not Path(job.output_path).is_absolute():
                job.output_path = str(output_dir / job.output_path)

    # Execute quantization jobs
    logger.info(f"Starting quantization of {len(jobs)} model(s)...")
    results = []

    for i, job in enumerate(jobs, 1):
        logger.info(f"\n{'='*80}")
        logger.info(f"Job {i}/{len(jobs)}: {job.model_path}")
        logger.info(f"{'='*80}")

        result = quantize_single_model(job, args)
        results.append(result)

        if not result.success:
            logger.warning(f"Job {i} failed: {result.error_message}")

    # Generate summary report
    report_path = args.report or "quantization_report.json"
    generate_summary_report(results, report_path)

    # Exit with appropriate code
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        logger.warning(f"{failed_count} job(s) failed")
        sys.exit(1)
    else:
        logger.info("All jobs completed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
