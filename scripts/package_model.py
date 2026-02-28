#!/usr/bin/env python3
"""
ArrowQuant V2 Model Packaging Script

Packages quantized models with metadata, generates deployment manifests, validates
deployment requirements, and creates deployment-ready archives.

Usage:
    # Basic packaging
    python scripts/package_model.py \\
        --model dream-7b-int2/ \\
        --output dream-7b-int2-package.tar.gz

    # With deployment validation
    python scripts/package_model.py \\
        --model dream-7b-int2/ \\
        --output dream-7b-int2-package.tar.gz \\
        --target edge \\
        --max-size 35

    # Generate manifest only
    python scripts/package_model.py \\
        --model dream-7b-int2/ \\
        --manifest-only \\
        --output manifest.json

Author: ArrowQuant V2 Team
License: MIT
"""

import argparse
import hashlib
import json
import logging
import os
import sys
import tarfile
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
        logging.FileHandler("packaging.log"),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class DeploymentTarget:
    """Deployment target specifications."""

    name: str
    max_size_mb: float
    min_accuracy: float
    recommended_bit_width: int
    hardware_requirements: Dict[str, str]


# Deployment target presets
DEPLOYMENT_TARGETS = {
    "edge": DeploymentTarget(
        name="Edge Device",
        max_size_mb=35.0,
        min_accuracy=0.65,
        recommended_bit_width=2,
        hardware_requirements={
            "ram": "2-4GB",
            "cpu": "ARM64 or x86_64",
            "storage": "50MB minimum",
            "os": "Linux, Android, iOS",
        },
    ),
    "local": DeploymentTarget(
        name="Local Workstation",
        max_size_mb=200.0,
        min_accuracy=0.85,
        recommended_bit_width=4,
        hardware_requirements={
            "ram": "8GB+",
            "cpu": "x86_64 with AVX2",
            "storage": "500MB minimum",
            "os": "Linux, macOS, Windows",
        },
    ),
    "cloud": DeploymentTarget(
        name="Cloud Server",
        max_size_mb=2000.0,
        min_accuracy=0.95,
        recommended_bit_width=8,
        hardware_requirements={
            "ram": "32GB+",
            "cpu": "Multi-core x86_64",
            "gpu": "Optional (CUDA/ROCm)",
            "storage": "5GB minimum",
            "os": "Linux",
        },
    ),
}


@dataclass
class PackagingConfig:
    """Configuration for model packaging."""

    model_path: str
    output_path: str
    target: Optional[str] = None
    max_size_mb: Optional[float] = None
    min_accuracy: Optional[float] = None
    manifest_only: bool = False
    include_validation: bool = True
    compression_level: int = 9


@dataclass
class PackagingResult:
    """Results from packaging operation."""

    success: bool
    package_path: Optional[str] = None
    manifest_path: Optional[str] = None
    package_size_mb: Optional[float] = None
    model_size_mb: Optional[float] = None
    validation_passed: bool = False
    warnings: List[str] = None
    errors: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []
        if self.errors is None:
            self.errors = []


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="ArrowQuant V2 Model Packaging Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Package for edge deployment
  %(prog)s --model dream-7b-int2/ --output dream-7b-edge.tar.gz --target edge

  # Package for local deployment
  %(prog)s --model dream-7b-int4/ --output dream-7b-local.tar.gz --target local

  # Package for cloud deployment
  %(prog)s --model dream-7b-int8/ --output dream-7b-cloud.tar.gz --target cloud

  # Custom size limit
  %(prog)s --model dream-7b-int2/ --output package.tar.gz --max-size 50

  # Generate manifest only
  %(prog)s --model dream-7b-int2/ --manifest-only --output manifest.json

For more information, see: docs/DEPLOYMENT.md
        """,
    )

    # Required arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to quantized model directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output package file (.tar.gz) or manifest (.json)",
    )

    # Deployment target
    parser.add_argument(
        "--target",
        type=str,
        choices=["edge", "local", "cloud"],
        help="Deployment target (edge/local/cloud)",
    )

    # Validation arguments
    parser.add_argument(
        "--max-size",
        type=float,
        help="Maximum model size in MB",
    )
    parser.add_argument(
        "--min-accuracy",
        type=float,
        help="Minimum required accuracy (cosine similarity)",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip deployment requirement validation",
    )

    # Packaging options
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Generate manifest only (no archive)",
    )
    parser.add_argument(
        "--compression-level",
        type=int,
        choices=range(0, 10),
        default=9,
        help="Compression level (0-9, default: 9)",
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

    return parser.parse_args()


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_directory_size(path: Path) -> int:
    """Get total size of directory in bytes."""
    total_size = 0
    for entry in path.rglob("*"):
        if entry.is_file():
            total_size += entry.stat().st_size
    return total_size


def extract_model_metadata(model_path: Path) -> Dict[str, Any]:
    """Extract metadata from quantized model."""
    metadata = {
        "model_name": model_path.name,
        "model_path": str(model_path),
        "files": [],
        "total_size_bytes": 0,
        "total_size_mb": 0.0,
    }

    # List all files
    for file_path in sorted(model_path.rglob("*")):
        if file_path.is_file():
            file_size = file_path.stat().st_size
            rel_path = file_path.relative_to(model_path)

            file_info = {
                "path": str(rel_path),
                "size_bytes": file_size,
                "size_mb": file_size / (1024 * 1024),
                "sha256": compute_file_hash(file_path),
            }
            metadata["files"].append(file_info)
            metadata["total_size_bytes"] += file_size

    metadata["total_size_mb"] = metadata["total_size_bytes"] / (1024 * 1024)
    metadata["file_count"] = len(metadata["files"])

    # Try to read quantization metadata
    metadata_file = model_path / "metadata.json"
    if metadata_file.exists():
        try:
            with open(metadata_file, "r") as f:
                quant_metadata = json.load(f)
                metadata["quantization"] = quant_metadata
        except Exception as e:
            logger.warning(f"Failed to read metadata.json: {e}")

    return metadata


def validate_deployment_requirements(
    model_metadata: Dict[str, Any],
    target: DeploymentTarget,
    max_size_mb: Optional[float] = None,
    min_accuracy: Optional[float] = None,
) -> tuple[bool, List[str]]:
    """Validate model meets deployment requirements."""
    warnings = []
    passed = True

    # Check size requirement
    size_limit = max_size_mb if max_size_mb is not None else target.max_size_mb
    model_size = model_metadata["total_size_mb"]

    if model_size > size_limit:
        warnings.append(
            f"Model size {model_size:.2f}MB exceeds {target.name} limit of {size_limit:.2f}MB"
        )
        passed = False
    else:
        logger.info(
            f"✓ Size check passed: {model_size:.2f}MB <= {size_limit:.2f}MB"
        )

    # Check accuracy requirement (if available in metadata)
    accuracy_limit = (
        min_accuracy if min_accuracy is not None else target.min_accuracy
    )
    if "quantization" in model_metadata:
        quant_meta = model_metadata["quantization"]
        if "cosine_similarity" in quant_meta:
            accuracy = quant_meta["cosine_similarity"]
            if accuracy < accuracy_limit:
                warnings.append(
                    f"Model accuracy {accuracy:.4f} below {target.name} requirement of {accuracy_limit:.4f}"
                )
                passed = False
            else:
                logger.info(
                    f"✓ Accuracy check passed: {accuracy:.4f} >= {accuracy_limit:.4f}"
                )

    # Check bit width recommendation
    if "quantization" in model_metadata:
        quant_meta = model_metadata["quantization"]
        if "bit_width" in quant_meta:
            bit_width = quant_meta["bit_width"]
            if bit_width != target.recommended_bit_width:
                warnings.append(
                    f"Model uses INT{bit_width}, but {target.name} recommends INT{target.recommended_bit_width}"
                )
            else:
                logger.info(
                    f"✓ Bit width check passed: INT{bit_width}"
                )

    return passed, warnings


def generate_deployment_manifest(
    model_metadata: Dict[str, Any],
    target: Optional[DeploymentTarget] = None,
    validation_passed: bool = False,
) -> Dict[str, Any]:
    """Generate deployment manifest."""
    manifest = {
        "version": "1.0",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model": {
            "name": model_metadata["model_name"],
            "size_mb": model_metadata["total_size_mb"],
            "file_count": model_metadata["file_count"],
            "files": model_metadata["files"],
        },
    }

    # Add quantization info if available
    if "quantization" in model_metadata:
        manifest["quantization"] = model_metadata["quantization"]

    # Add deployment target info
    if target:
        manifest["deployment"] = {
            "target": target.name,
            "validation_passed": validation_passed,
            "requirements": {
                "max_size_mb": target.max_size_mb,
                "min_accuracy": target.min_accuracy,
                "recommended_bit_width": target.recommended_bit_width,
            },
            "hardware": target.hardware_requirements,
        }

    # Add installation instructions
    manifest["installation"] = {
        "extract": "tar -xzf <package_file>",
        "verify": "sha256sum -c checksums.txt",
        "usage": "See README.md for usage instructions",
    }

    return manifest


def create_package_archive(
    model_path: Path,
    output_path: Path,
    manifest: Dict[str, Any],
    compression_level: int = 9,
) -> float:
    """Create compressed package archive."""
    logger.info(f"Creating package archive: {output_path}")

    # Create temporary manifest file
    manifest_path = model_path / "deployment_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Create README
    readme_path = model_path / "README.md"
    if not readme_path.exists():
        with open(readme_path, "w") as f:
            f.write(f"""# {manifest['model']['name']}

## Model Information

- **Size**: {manifest['model']['size_mb']:.2f} MB
- **Files**: {manifest['model']['file_count']}
- **Generated**: {manifest['generated_at']}

## Quantization Details

""")
            if "quantization" in manifest:
                quant = manifest["quantization"]
                f.write(f"- **Bit Width**: INT{quant.get('bit_width', 'N/A')}\n")
                f.write(f"- **Accuracy**: {quant.get('cosine_similarity', 'N/A'):.4f}\n")
                f.write(f"- **Compression**: {quant.get('compression_ratio', 'N/A'):.2f}x\n")

            if "deployment" in manifest:
                deploy = manifest["deployment"]
                f.write(f"\n## Deployment Target\n\n")
                f.write(f"- **Target**: {deploy['target']}\n")
                f.write(f"- **Validation**: {'Passed' if deploy['validation_passed'] else 'Failed'}\n")
                f.write(f"\n## Hardware Requirements\n\n")
                for key, value in deploy["hardware"].items():
                    f.write(f"- **{key.upper()}**: {value}\n")

            f.write(f"\n## Installation\n\n")
            f.write(f"```bash\n")
            f.write(f"# Extract package\n")
            f.write(f"{manifest['installation']['extract']}\n\n")
            f.write(f"# Verify integrity\n")
            f.write(f"{manifest['installation']['verify']}\n")
            f.write(f"```\n")

    # Create checksums file
    checksums_path = model_path / "checksums.txt"
    with open(checksums_path, "w") as f:
        for file_info in manifest["model"]["files"]:
            f.write(f"{file_info['sha256']}  {file_info['path']}\n")

    # Create tar.gz archive
    with tarfile.open(output_path, f"w:gz", compresslevel=compression_level) as tar:
        # Add all model files
        for file_info in manifest["model"]["files"]:
            file_path = model_path / file_info["path"]
            arcname = f"{model_path.name}/{file_info['path']}"
            tar.add(file_path, arcname=arcname)

        # Add manifest
        tar.add(manifest_path, arcname=f"{model_path.name}/deployment_manifest.json")

        # Add README
        tar.add(readme_path, arcname=f"{model_path.name}/README.md")

        # Add checksums
        tar.add(checksums_path, arcname=f"{model_path.name}/checksums.txt")

    # Clean up temporary files
    manifest_path.unlink()
    if not (model_path / "README.md").exists():
        readme_path.unlink()
    checksums_path.unlink()

    # Get package size
    package_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Package created: {package_size_mb:.2f} MB")

    return package_size_mb


def package_model(config: PackagingConfig) -> PackagingResult:
    """Package quantized model for deployment."""
    try:
        # Validate model path
        model_path = Path(config.model_path)
        if not model_path.exists():
            return PackagingResult(
                success=False,
                errors=[f"Model path not found: {config.model_path}"],
            )

        if not model_path.is_dir():
            return PackagingResult(
                success=False,
                errors=[f"Model path is not a directory: {config.model_path}"],
            )

        logger.info(f"Packaging model: {config.model_path}")

        # Extract model metadata
        logger.info("Extracting model metadata...")
        model_metadata = extract_model_metadata(model_path)
        logger.info(
            f"Model size: {model_metadata['total_size_mb']:.2f} MB ({model_metadata['file_count']} files)"
        )

        # Validate deployment requirements
        validation_passed = True
        warnings = []

        if config.target and not config.manifest_only:
            target = DEPLOYMENT_TARGETS[config.target]
            logger.info(f"Validating for {target.name} deployment...")

            validation_passed, warnings = validate_deployment_requirements(
                model_metadata,
                target,
                config.max_size_mb,
                config.min_accuracy,
            )

            if not validation_passed:
                logger.warning("Deployment validation failed:")
                for warning in warnings:
                    logger.warning(f"  - {warning}")
        else:
            target = None

        # Generate manifest
        logger.info("Generating deployment manifest...")
        manifest = generate_deployment_manifest(
            model_metadata, target, validation_passed
        )

        # Save manifest
        output_path = Path(config.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if config.manifest_only:
            # Save manifest only
            with open(output_path, "w") as f:
                json.dump(manifest, f, indent=2)
            logger.info(f"Manifest saved to {output_path}")

            return PackagingResult(
                success=True,
                manifest_path=str(output_path),
                model_size_mb=model_metadata["total_size_mb"],
                validation_passed=validation_passed,
                warnings=warnings,
            )
        else:
            # Create package archive
            package_size_mb = create_package_archive(
                model_path,
                output_path,
                manifest,
                config.compression_level,
            )

            logger.info(f"Package created successfully: {output_path}")

            return PackagingResult(
                success=True,
                package_path=str(output_path),
                package_size_mb=package_size_mb,
                model_size_mb=model_metadata["total_size_mb"],
                validation_passed=validation_passed,
                warnings=warnings,
            )

    except Exception as e:
        logger.error(f"Packaging failed: {e}")
        logger.exception("Full traceback:")
        return PackagingResult(
            success=False,
            errors=[str(e)],
        )


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    elif args.quiet:
        logger.setLevel(logging.WARNING)

    # Create configuration
    config = PackagingConfig(
        model_path=args.model,
        output_path=args.output,
        target=args.target,
        max_size_mb=args.max_size,
        min_accuracy=args.min_accuracy,
        manifest_only=args.manifest_only,
        include_validation=not args.skip_validation,
        compression_level=args.compression_level,
    )

    # Package model
    logger.info("Starting model packaging...")
    result = package_model(config)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("PACKAGING SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Status: {'SUCCESS' if result.success else 'FAILED'}")

    if result.success:
        if result.package_path:
            logger.info(f"Package: {result.package_path}")
            logger.info(f"Package size: {result.package_size_mb:.2f} MB")
        if result.manifest_path:
            logger.info(f"Manifest: {result.manifest_path}")
        logger.info(f"Model size: {result.model_size_mb:.2f} MB")
        logger.info(f"Validation: {'PASSED' if result.validation_passed else 'FAILED'}")

        if result.warnings:
            logger.warning(f"\nWarnings:")
            for warning in result.warnings:
                logger.warning(f"  - {warning}")

    if result.errors:
        logger.error(f"\nErrors:")
        for error in result.errors:
            logger.error(f"  - {error}")

    logger.info("=" * 80)

    # Exit with appropriate code
    if not result.success:
        sys.exit(1)
    elif result.warnings and not result.validation_passed:
        logger.warning("Packaging completed with validation warnings")
        sys.exit(0)  # Don't fail on warnings
    else:
        logger.info("Packaging completed successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
