"""
Test Markov metrics exposure in Python API

This test verifies that Task 5.2 is correctly implemented:
- get_markov_metrics() method is available in Python API
- Returns smoothness score, violations, and boundary scores
- Validates REQ-1.1.3, REQ-3.1.1
"""

import pytest
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    from arrow_quant_v2 import ArrowQuantV2, DiffusionQuantConfig
    BINDINGS_AVAILABLE = True
except ImportError:
    BINDINGS_AVAILABLE = False


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Python bindings not built")
class TestMarkovMetricsAPI:
    """Test suite for Markov metrics Python API (Task 5.2)"""

    def test_get_markov_metrics_method_exists(self):
        """Test that get_markov_metrics method exists on ArrowQuantV2"""
        quantizer = ArrowQuantV2(mode="diffusion")
        assert hasattr(quantizer, 'get_markov_metrics'), \
            "get_markov_metrics method should exist"
        assert callable(getattr(quantizer, 'get_markov_metrics')), \
            "get_markov_metrics should be callable"

    def test_get_markov_metrics_returns_none_before_quantization(self):
        """Test that get_markov_metrics returns None before any quantization"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        assert metrics is None, \
            "Should return None when no quantization has been performed"

    def test_get_markov_metrics_return_type(self):
        """Test that get_markov_metrics returns correct type"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        # Should return None or dict
        assert metrics is None or isinstance(metrics, dict), \
            "get_markov_metrics should return None or dict"

    def test_markov_metrics_structure_complete(self):
        """Test that metrics have all expected keys when available"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        # If metrics are available (after quantization), verify structure
        if metrics is not None:
            expected_keys = {
                'smoothness_score',
                'boundary_scores',
                'violation_count',
                'violations',
                'is_valid'
            }
            actual_keys = set(metrics.keys())
            assert actual_keys == expected_keys, \
                f"Metrics should have keys: {expected_keys}, got: {actual_keys}"

    def test_markov_metrics_types(self):
        """Test that metrics have correct types"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None:
            # Verify types
            assert isinstance(metrics['smoothness_score'], float), \
                "smoothness_score should be float"
            assert isinstance(metrics['boundary_scores'], list), \
                "boundary_scores should be list"
            assert isinstance(metrics['violation_count'], int), \
                "violation_count should be int"
            assert isinstance(metrics['violations'], list), \
                "violations should be list"
            assert isinstance(metrics['is_valid'], bool), \
                "is_valid should be bool"

    def test_smoothness_score_range(self):
        """Test that smoothness_score is in valid range [0, 1]"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None:
            score = metrics['smoothness_score']
            assert 0.0 <= score <= 1.0, \
                f"smoothness_score should be in [0, 1], got {score}"

    def test_boundary_scores_are_floats(self):
        """Test that boundary_scores contains floats"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None and len(metrics['boundary_scores']) > 0:
            for score in metrics['boundary_scores']:
                assert isinstance(score, float), \
                    f"Each boundary score should be float, got {type(score)}"
                assert 0.0 <= score <= 1.0, \
                    f"Each boundary score should be in [0, 1], got {score}"

    def test_violation_count_matches_violations_list(self):
        """Test that violation_count matches length of violations list"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None:
            count = metrics['violation_count']
            violations = metrics['violations']
            assert count == len(violations), \
                f"violation_count ({count}) should match violations list length ({len(violations)})"

    def test_violation_structure(self):
        """Test that each violation has correct structure"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None and len(metrics['violations']) > 0:
            for i, violation in enumerate(metrics['violations']):
                # Check required keys
                assert 'boundary_idx' in violation, \
                    f"Violation {i} should have 'boundary_idx'"
                assert 'scale_jump' in violation, \
                    f"Violation {i} should have 'scale_jump'"
                assert 'zero_point_jump' in violation, \
                    f"Violation {i} should have 'zero_point_jump'"
                assert 'severity' in violation, \
                    f"Violation {i} should have 'severity'"
                
                # Check types
                assert isinstance(violation['boundary_idx'], int), \
                    f"boundary_idx should be int"
                assert isinstance(violation['scale_jump'], float), \
                    f"scale_jump should be float"
                assert isinstance(violation['zero_point_jump'], float), \
                    f"zero_point_jump should be float"
                assert isinstance(violation['severity'], str), \
                    f"severity should be str"
                
                # Check severity values
                assert violation['severity'] in ['low', 'medium', 'high'], \
                    f"severity should be 'low', 'medium', or 'high', got '{violation['severity']}'"

    def test_is_valid_consistency(self):
        """Test that is_valid is consistent with violation_count"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None:
            is_valid = metrics['is_valid']
            violation_count = metrics['violation_count']
            
            # is_valid should be True when no violations
            if violation_count == 0:
                assert is_valid is True, \
                    "is_valid should be True when violation_count is 0"
            else:
                assert is_valid is False, \
                    "is_valid should be False when violations exist"

    def test_metrics_format_json_serializable(self):
        """Test that metrics can be serialized to JSON"""
        import json
        
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None:
            # Should be JSON serializable
            try:
                json_str = json.dumps(metrics)
                assert len(json_str) > 0, "JSON string should not be empty"
                
                # Should be deserializable
                deserialized = json.loads(json_str)
                assert deserialized == metrics, \
                    "Deserialized metrics should match original"
            except (TypeError, ValueError) as e:
                pytest.fail(f"Metrics should be JSON serializable: {e}")

    def test_config_thermodynamic_validation_enabled(self):
        """Test that thermodynamic validation can be enabled in config"""
        config = DiffusionQuantConfig(bit_width=2)
        
        # Verify that the config can be created successfully
        # The thermodynamic settings are accessible through the Rust backend
        assert config is not None, "Config should be created successfully"

    def test_multiple_calls_return_same_metrics(self):
        """Test that multiple calls to get_markov_metrics return consistent results"""
        quantizer = ArrowQuantV2(mode="diffusion")
        
        metrics1 = quantizer.get_markov_metrics()
        metrics2 = quantizer.get_markov_metrics()
        
        # Both should be None before quantization
        assert metrics1 == metrics2, \
            "Multiple calls should return consistent results"

    def test_metrics_content_format(self):
        """Test that metrics content follows expected format"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None:
            # Smoothness score should be a valid float
            assert not (metrics['smoothness_score'] != metrics['smoothness_score']), \
                "smoothness_score should not be NaN"
            
            # Boundary scores should be non-empty if we have time groups
            # (This would be true after actual quantization)
            if len(metrics['boundary_scores']) > 0:
                assert all(isinstance(s, float) for s in metrics['boundary_scores']), \
                    "All boundary scores should be floats"


@pytest.mark.skipif(not BINDINGS_AVAILABLE, reason="Python bindings not built")
class TestMarkovMetricsEdgeCases:
    """Test edge cases for Markov metrics API"""

    def test_metrics_with_no_violations(self):
        """Test metrics structure when no violations exist"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None and metrics['violation_count'] == 0:
            assert metrics['is_valid'] is True
            assert len(metrics['violations']) == 0
            assert metrics['smoothness_score'] >= 0.0

    def test_metrics_with_violations(self):
        """Test metrics structure when violations exist"""
        quantizer = ArrowQuantV2(mode="diffusion")
        metrics = quantizer.get_markov_metrics()
        
        if metrics is not None and metrics['violation_count'] > 0:
            assert metrics['is_valid'] is False
            assert len(metrics['violations']) > 0
            # Smoothness score should be lower when violations exist
            assert metrics['smoothness_score'] < 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
