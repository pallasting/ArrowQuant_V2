"""
Unit tests for the offline quantization script.

Tests the CLI argument parsing, configuration creation, and batch job loading.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import the script module
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from quantize_diffusion import (
    QuantizationJob,
    QuantizationResult,
    create_config,
    load_batch_jobs,
    parse_arguments,
)


class TestQuantizationJob:
    """Test QuantizationJob dataclass."""

    def test_create_job(self):
        """Test creating a quantization job."""
        job = QuantizationJob(
            model_path="models/dream-7b/",
            output_path="output/dream-7b-int2/",
            bit_width=2,
            profile="edge",
        )

        assert job.model_path == "models/dream-7b/"
        assert job.output_path == "output/dream-7b-int2/"
        assert job.bit_width == 2
        assert job.profile == "edge"


class TestBatchJobLoading:
    """Test batch job loading from JSON."""

    def test_load_batch_jobs(self):
        """Test loading batch jobs from JSON file."""
        # Create temporary batch file
        batch_data = {
            "jobs": [
                {
                    "model_path": "models/dream-7b/",
                    "output_path": "dream-7b-edge/",
                    "profile": "edge",
                    "bit_width": 2,
                },
                {
                    "model_path": "models/dream-7b/",
                    "output_path": "dream-7b-local/",
                    "profile": "local",
                    "bit_width": 4,
                },
            ]
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            batch_file = f.name

        try:
            jobs = load_batch_jobs(batch_file)

            assert len(jobs) == 2
            assert jobs[0].model_path == "models/dream-7b/"
            assert jobs[0].output_path == "dream-7b-edge/"
            assert jobs[0].profile == "edge"
            assert jobs[0].bit_width == 2

            assert jobs[1].model_path == "models/dream-7b/"
            assert jobs[1].output_path == "dream-7b-local/"
            assert jobs[1].profile == "local"
            assert jobs[1].bit_width == 4
        finally:
            Path(batch_file).unlink()

    def test_load_empty_batch(self):
        """Test loading empty batch file."""
        batch_data = {"jobs": []}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(batch_data, f)
            batch_file = f.name

        try:
            jobs = load_batch_jobs(batch_file)
            assert len(jobs) == 0
        finally:
            Path(batch_file).unlink()


class TestConfigCreation:
    """Test configuration creation from jobs and CLI args."""

    def test_create_config_from_job(self):
        """Test creating config from job parameters."""
        job = QuantizationJob(
            model_path="models/dream-7b/",
            output_path="output/",
            bit_width=2,
            num_time_groups=5,
            group_size=256,
            enable_time_aware=True,
            enable_spatial=False,
            min_accuracy=0.70,
        )

        # Mock args with no overrides
        args = MagicMock()
        args.bit_width = None
        args.num_time_groups = None
        args.group_size = None
        args.enable_time_aware = False
        args.disable_time_aware = False
        args.enable_spatial = False
        args.disable_spatial = False
        args.min_accuracy = None
        args.calibration_samples = None
        args.fail_fast = False
        args.profile = None
        args.modality = None

        config = create_config(job, args)

        assert config["bit_width"] == 2
        assert config["num_time_groups"] == 5
        assert config["group_size"] == 256
        assert config["enable_time_aware"] is True
        assert config["enable_spatial"] is False
        assert config["min_accuracy"] == 0.70

    def test_create_config_cli_override(self):
        """Test job parameters take priority over CLI args."""
        job = QuantizationJob(
            model_path="models/dream-7b/",
            output_path="output/",
            bit_width=2,
            num_time_groups=5,
        )

        # Mock args with different values
        args = MagicMock()
        args.bit_width = 4  # Job param takes priority
        args.num_time_groups = 10  # Job param takes priority
        args.group_size = None
        args.enable_time_aware = False
        args.disable_time_aware = False
        args.enable_spatial = False
        args.disable_spatial = False
        args.min_accuracy = None
        args.calibration_samples = None
        args.fail_fast = False
        args.profile = None
        args.modality = None

        config = create_config(job, args)

        # Job params should take priority (for batch processing)
        assert config["bit_width"] == 2
        assert config["num_time_groups"] == 5

    def test_create_config_cli_fallback(self):
        """Test CLI args are used when job params are None."""
        job = QuantizationJob(
            model_path="models/dream-7b/",
            output_path="output/",
            # No bit_width or num_time_groups specified
        )

        # Mock args with values
        args = MagicMock()
        args.bit_width = 4
        args.num_time_groups = 10
        args.group_size = 128
        args.enable_time_aware = False
        args.disable_time_aware = False
        args.enable_spatial = False
        args.disable_spatial = False
        args.min_accuracy = None
        args.calibration_samples = None
        args.fail_fast = False
        args.profile = None
        args.modality = None

        config = create_config(job, args)

        # CLI args should be used when job params are None
        assert config["bit_width"] == 4
        assert config["num_time_groups"] == 10
        assert config["group_size"] == 128

    def test_create_config_with_profile(self):
        """Test creating config with deployment profile."""
        job = QuantizationJob(
            model_path="models/dream-7b/",
            output_path="output/",
            profile="edge",
        )

        args = MagicMock()
        args.bit_width = None
        args.num_time_groups = None
        args.group_size = None
        args.enable_time_aware = False
        args.disable_time_aware = False
        args.enable_spatial = False
        args.disable_spatial = False
        args.min_accuracy = None
        args.calibration_samples = None
        args.fail_fast = False
        args.profile = None
        args.modality = None

        config = create_config(job, args)

        assert config["deployment_profile"] == "edge"


class TestArgumentParsing:
    """Test CLI argument parsing."""

    def test_parse_single_model_args(self):
        """Test parsing single model arguments."""
        with patch(
            "sys.argv",
            [
                "quantize_diffusion.py",
                "--model",
                "dream-7b/",
                "--output",
                "dream-7b-int2/",
                "--profile",
                "edge",
            ],
        ):
            args = parse_arguments()

            assert args.model == "dream-7b/"
            assert args.output == "dream-7b-int2/"
            assert args.profile == "edge"
            assert args.batch is None

    def test_parse_batch_args(self):
        """Test parsing batch arguments."""
        with patch(
            "sys.argv",
            [
                "quantize_diffusion.py",
                "--batch",
                "batch.json",
                "--output-dir",
                "quantized/",
            ],
        ):
            args = parse_arguments()

            assert args.batch == "batch.json"
            assert args.output_dir == "quantized/"
            assert args.model is None

    def test_parse_custom_params(self):
        """Test parsing custom quantization parameters."""
        with patch(
            "sys.argv",
            [
                "quantize_diffusion.py",
                "--model",
                "dream-7b/",
                "--output",
                "output/",
                "--bit-width",
                "4",
                "--num-time-groups",
                "15",
                "--group-size",
                "64",
                "--min-accuracy",
                "0.80",
                "--calibration-samples",
                "256",
            ],
        ):
            args = parse_arguments()

            assert args.bit_width == 4
            assert args.num_time_groups == 15
            assert args.group_size == 64
            assert args.min_accuracy == 0.80
            assert args.calibration_samples == 256

    def test_parse_boolean_flags(self):
        """Test parsing boolean flags."""
        with patch(
            "sys.argv",
            [
                "quantize_diffusion.py",
                "--model",
                "dream-7b/",
                "--output",
                "output/",
                "--enable-time-aware",
                "--disable-spatial",
                "--fail-fast",
                "--verbose",
            ],
        ):
            args = parse_arguments()

            assert args.enable_time_aware is True
            assert args.disable_spatial is True
            assert args.fail_fast is True
            assert args.verbose is True


class TestQuantizationResult:
    """Test QuantizationResult dataclass."""

    def test_successful_result(self):
        """Test creating a successful result."""
        job = QuantizationJob(
            model_path="models/dream-7b/",
            output_path="output/dream-7b-int2/",
        )

        result = QuantizationResult(
            job=job,
            success=True,
            quantized_path="output/dream-7b-int2/",
            compression_ratio=16.0,
            cosine_similarity=0.72,
            model_size_mb=34.5,
            modality="text",
            bit_width=2,
            quantization_time_s=300.0,
        )

        assert result.success is True
        assert result.compression_ratio == 16.0
        assert result.cosine_similarity == 0.72
        assert result.model_size_mb == 34.5
        assert result.error_message is None

    def test_failed_result(self):
        """Test creating a failed result."""
        job = QuantizationJob(
            model_path="models/nonexistent/",
            output_path="output/",
        )

        result = QuantizationResult(
            job=job,
            success=False,
            error_message="Model path does not exist",
        )

        assert result.success is False
        assert result.error_message == "Model path does not exist"
        assert result.quantized_path is None
        assert result.compression_ratio is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
