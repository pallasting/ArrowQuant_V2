"""
Unit tests for PrecisionValidator.

Tests validation of quantized model precision using cosine similarity
and perplexity metrics.

Feature: memory-optimization
Requirements: 8.2, 8.6, 12.1
"""

import pytest
import tempfile
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from llm_compression.inference.precision_validator import (
    PrecisionValidator,
    ValidationResult
)
from llm_compression.inference.quantization_schema import (
    WEIGHT_SCHEMA_V1,
    create_v1_row
)


class TestValidationResult:
    """Test ValidationResult dataclass."""
    
    def test_validation_result_creation(self):
        """Test creating ValidationResult."""
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            ppl_increase=None,
            error_message=None,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        assert result.passed is True
        assert result.cosine_similarity == 0.97
        assert result.min_cosine_similarity == 0.95
        assert result.max_cosine_similarity == 0.99
        assert result.ppl_increase is None
        assert result.error_message is None
        assert result.num_samples == 10
        assert result.validation_time_ms == 123.45
    
    def test_validation_result_to_dict(self):
        """Test converting ValidationResult to dictionary."""
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['passed'] is True
        assert result_dict['cosine_similarity'] == 0.97
        assert result_dict['min_cosine_similarity'] == 0.95
        assert result_dict['max_cosine_similarity'] == 0.99
        assert result_dict['num_samples'] == 10
        assert result_dict['validation_time_ms'] == 123.45
    
    def test_validation_result_to_json(self):
        """Test converting ValidationResult to JSON."""
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        json_str = result.to_json()
        
        assert isinstance(json_str, str)
        assert '"passed": true' in json_str
        assert '"cosine_similarity": 0.97' in json_str
    
    def test_validation_result_str(self):
        """Test ValidationResult string representation."""
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        result_str = str(result)
        
        assert "PASSED" in result_str
        assert "0.9700" in result_str
        assert "10" in result_str
    
    def test_validation_result_with_ppl(self):
        """Test ValidationResult with PPL metrics."""
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            ppl_increase=0.12,
            original_ppl=10.5,
            quantized_ppl=11.76,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        assert result.ppl_increase == 0.12
        assert result.original_ppl == 10.5
        assert result.quantized_ppl == 11.76
        
        result_str = str(result)
        assert "PPL Increase" in result_str
    
    def test_validation_result_failed(self):
        """Test failed ValidationResult."""
        result = ValidationResult(
            passed=False,
            cosine_similarity=0.92,
            min_cosine_similarity=0.90,
            max_cosine_similarity=0.94,
            error_message="Cosine similarity below threshold",
            num_samples=10,
            validation_time_ms=123.45
        )
        
        assert result.passed is False
        assert result.error_message == "Cosine similarity below threshold"
        
        result_str = str(result)
        assert "FAILED" in result_str
        assert "Error:" in result_str


class TestPrecisionValidator:
    """Test PrecisionValidator class."""
    
    def test_validator_initialization(self):
        """Test PrecisionValidator initialization."""
        validator = PrecisionValidator(
            cosine_threshold=0.95,
            ppl_threshold=0.15,
            validate_ppl=False
        )
        
        assert validator.cosine_threshold == 0.95
        assert validator.ppl_threshold == 0.15
        assert validator.validate_ppl is False
    
    def test_validator_default_thresholds(self):
        """Test PrecisionValidator with default thresholds."""
        validator = PrecisionValidator()
        
        assert validator.cosine_threshold == 0.95
        assert validator.ppl_threshold == 0.15
        assert validator.validate_ppl is False
    
    def test_compute_cosine_similarities(self):
        """Test cosine similarity computation."""
        validator = PrecisionValidator()
        
        # Create test embeddings
        embeddings1 = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        
        embeddings2 = np.array([
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 0.1, 0.9]
        ], dtype=np.float32)
        
        similarities = validator._compute_cosine_similarities(embeddings1, embeddings2)
        
        assert similarities.shape == (3,)
        assert np.all(similarities > 0.8)
        assert np.all(similarities <= 1.0)
    
    def test_compute_cosine_similarities_identical(self):
        """Test cosine similarity with identical embeddings."""
        validator = PrecisionValidator()
        
        embeddings = np.random.randn(10, 128).astype(np.float32)
        
        similarities = validator._compute_cosine_similarities(embeddings, embeddings)
        
        # Identical embeddings should have similarity ~1.0
        assert np.allclose(similarities, 1.0, atol=1e-5)
    
    def test_compute_cosine_similarities_orthogonal(self):
        """Test cosine similarity with orthogonal embeddings."""
        validator = PrecisionValidator()
        
        embeddings1 = np.array([
            [1.0, 0.0],
            [0.0, 1.0]
        ], dtype=np.float32)
        
        embeddings2 = np.array([
            [0.0, 1.0],
            [1.0, 0.0]
        ], dtype=np.float32)
        
        similarities = validator._compute_cosine_similarities(embeddings1, embeddings2)
        
        # Orthogonal vectors should have similarity ~0.0
        assert np.allclose(similarities, 0.0, atol=1e-5)
    
    def test_validate_missing_original_model(self):
        """Test validation with missing original model."""
        validator = PrecisionValidator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            quantized_path = Path(tmpdir) / "quantized.parquet"
            
            # Create dummy quantized model
            rows = [
                create_v1_row(
                    layer_name="test.weight",
                    shape=[64, 128],
                    dtype="torch.float32",
                    data=np.random.randn(64, 128).astype(np.float32).tobytes(),
                    num_params=64 * 128
                )
            ]
            table = pa.Table.from_pylist(rows, schema=WEIGHT_SCHEMA_V1)
            pq.write_table(table, quantized_path)
            
            # Validate with non-existent original model
            result = validator.validate(
                original_model_path=str(Path(tmpdir) / "nonexistent.parquet"),
                quantized_model_path=str(quantized_path),
                test_texts=["test"]
            )
            
            assert result.passed is False
            assert "not found" in result.error_message.lower()
    
    def test_validate_missing_quantized_model(self):
        """Test validation with missing quantized model."""
        validator = PrecisionValidator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "original.parquet"
            
            # Create dummy original model
            rows = [
                create_v1_row(
                    layer_name="test.weight",
                    shape=[64, 128],
                    dtype="torch.float32",
                    data=np.random.randn(64, 128).astype(np.float32).tobytes(),
                    num_params=64 * 128
                )
            ]
            table = pa.Table.from_pylist(rows, schema=WEIGHT_SCHEMA_V1)
            pq.write_table(table, original_path)
            
            # Validate with non-existent quantized model
            result = validator.validate(
                original_model_path=str(original_path),
                quantized_model_path=str(Path(tmpdir) / "nonexistent.parquet"),
                test_texts=["test"]
            )
            
            assert result.passed is False
            assert "not found" in result.error_message.lower()
    
    def test_validate_empty_test_texts(self):
        """Test validation with empty test texts."""
        validator = PrecisionValidator()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            original_path = Path(tmpdir) / "original.parquet"
            quantized_path = Path(tmpdir) / "quantized.parquet"
            
            # Create dummy models
            rows = [
                create_v1_row(
                    layer_name="test.weight",
                    shape=[64, 128],
                    dtype="torch.float32",
                    data=np.random.randn(64, 128).astype(np.float32).tobytes(),
                    num_params=64 * 128
                )
            ]
            table = pa.Table.from_pylist(rows, schema=WEIGHT_SCHEMA_V1)
            pq.write_table(table, original_path)
            pq.write_table(table, quantized_path)
            
            # Validate with empty test texts
            result = validator.validate(
                original_model_path=str(original_path),
                quantized_model_path=str(quantized_path),
                test_texts=[]
            )
            
            assert result.passed is False
            assert "no test texts" in result.error_message.lower()
    
    def test_generate_report_json(self):
        """Test generating JSON validation report."""
        validator = PrecisionValidator()
        
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            
            report = validator.generate_report(
                result,
                output_path=str(report_path),
                format='json'
            )
            
            assert report_path.exists()
            assert '"passed": true' in report
            assert '"cosine_similarity": 0.97' in report
    
    def test_generate_report_text(self):
        """Test generating text validation report."""
        validator = PrecisionValidator()
        
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = Path(tmpdir) / "report.txt"
            
            report = validator.generate_report(
                result,
                output_path=str(report_path),
                format='text'
            )
            
            assert report_path.exists()
            assert "PASSED" in report
            assert "0.9700" in report
    
    def test_generate_report_invalid_format(self):
        """Test generating report with invalid format."""
        validator = PrecisionValidator()
        
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        with pytest.raises(ValueError, match="Unsupported format"):
            validator.generate_report(result, format='xml')
    
    def test_generate_report_without_output_path(self):
        """Test generating report without saving to file."""
        validator = PrecisionValidator()
        
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        report = validator.generate_report(result, format='json')
        
        assert isinstance(report, str)
        assert '"passed": true' in report
    
    def test_validation_result_timestamp(self):
        """Test that ValidationResult includes timestamp."""
        result = ValidationResult(
            passed=True,
            cosine_similarity=0.97,
            min_cosine_similarity=0.95,
            max_cosine_similarity=0.99,
            num_samples=10,
            validation_time_ms=123.45
        )
        
        assert result.timestamp is not None
        assert isinstance(result.timestamp, str)
        # Should be ISO format
        assert 'T' in result.timestamp or '-' in result.timestamp


class TestPrecisionValidatorThresholds:
    """Test PrecisionValidator threshold checking."""
    
    def test_cosine_threshold_pass(self):
        """Test validation passes when cosine similarity meets threshold."""
        validator = PrecisionValidator(cosine_threshold=0.95)
        
        # Mock embeddings with high similarity
        embeddings1 = np.random.randn(10, 128).astype(np.float32)
        embeddings2 = embeddings1 + np.random.randn(10, 128).astype(np.float32) * 0.01
        
        similarities = validator._compute_cosine_similarities(embeddings1, embeddings2)
        avg_similarity = np.mean(similarities)
        
        # Should be high similarity
        assert avg_similarity > 0.95
    
    def test_cosine_threshold_fail(self):
        """Test validation fails when cosine similarity below threshold."""
        validator = PrecisionValidator(cosine_threshold=0.95)
        
        # Mock embeddings with low similarity
        embeddings1 = np.random.randn(10, 128).astype(np.float32)
        embeddings2 = np.random.randn(10, 128).astype(np.float32)
        
        similarities = validator._compute_cosine_similarities(embeddings1, embeddings2)
        avg_similarity = np.mean(similarities)
        
        # Random embeddings should have low similarity
        assert avg_similarity < 0.5


class TestPrecisionValidatorEdgeCases:
    """Test PrecisionValidator edge cases."""
    
    def test_single_sample_validation(self):
        """Test validation with single test sample."""
        validator = PrecisionValidator()
        
        embeddings1 = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
        embeddings2 = np.array([[0.9, 0.1, 0.0]], dtype=np.float32)
        
        similarities = validator._compute_cosine_similarities(embeddings1, embeddings2)
        
        assert similarities.shape == (1,)
        assert similarities[0] > 0.8
    
    def test_zero_embeddings(self):
        """Test validation with zero embeddings."""
        validator = PrecisionValidator()
        
        embeddings1 = np.zeros((5, 128), dtype=np.float32)
        embeddings2 = np.zeros((5, 128), dtype=np.float32)
        
        # Should handle zero embeddings gracefully (avoid division by zero)
        similarities = validator._compute_cosine_similarities(embeddings1, embeddings2)
        
        assert similarities.shape == (5,)
        # Zero embeddings should result in 0 or NaN, but not crash
        assert not np.any(np.isinf(similarities))
    
    def test_large_batch_validation(self):
        """Test validation with large batch of samples."""
        validator = PrecisionValidator()
        
        # Large batch
        embeddings1 = np.random.randn(1000, 128).astype(np.float32)
        embeddings2 = embeddings1 + np.random.randn(1000, 128).astype(np.float32) * 0.01
        
        similarities = validator._compute_cosine_similarities(embeddings1, embeddings2)
        
        assert similarities.shape == (1000,)
        assert np.all(similarities > 0.9)
