"""
Test suite for daily API cost validation

Validates that the daily API cost target of < $1 is achieved
for typical usage (1000 memories/day).
"""

import pytest
import sys
import os

# Add scripts directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from cost_analysis import (
    calculate_semantic_indexing_cost,
    calculate_vector_embedding_cost,
    calculate_arrow_compression_cost,
    calculate_total_daily_cost,
    MODELS
)


class TestCostValidation:
    """Test suite for cost validation"""
    
    def test_semantic_indexing_cost_with_batching(self):
        """Test semantic indexing cost with batch API"""
        result = calculate_semantic_indexing_cost(
            memories_per_day=1000,
            batch_size=8,
            use_batching=True
        )
        
        # Verify batch API reduces calls
        assert result.calls_per_day == 125  # 1000 / 8
        
        # Verify cost is reasonable
        assert result.total_daily_cost < 0.1  # Should be much less than $0.10
        assert result.total_daily_cost > 0.0  # Should have some cost
    
    def test_semantic_indexing_cost_without_batching(self):
        """Test semantic indexing cost without batch API"""
        result = calculate_semantic_indexing_cost(
            memories_per_day=1000,
            batch_size=8,
            use_batching=False
        )
        
        # Verify individual calls
        assert result.calls_per_day == 1000
        
        # Verify cost is higher than batched
        batched = calculate_semantic_indexing_cost(1000, 8, True)
        assert result.total_daily_cost > batched.total_daily_cost
    
    def test_vector_embedding_cost_is_zero(self):
        """Test that vector embeddings have zero API cost (local model)"""
        result = calculate_vector_embedding_cost(memories_per_day=1000)
        
        assert result.total_daily_cost == 0.0
        assert result.cost_per_call == 0.0
        assert result.operation == "Vector Embeddings (Local)"
    
    def test_arrow_compression_cost_is_zero(self):
        """Test that Arrow compression has zero API cost (local)"""
        result = calculate_arrow_compression_cost(memories_per_day=1000)
        
        assert result.total_daily_cost == 0.0
        assert result.cost_per_call == 0.0
        assert result.operation == "Arrow Compression (Local)"
    
    def test_total_daily_cost_meets_target(self):
        """Test that total daily cost meets < $1 target"""
        result = calculate_total_daily_cost(
            memories_per_day=1000,
            use_batching=True
        )
        
        # Primary validation: Cost must be under $1
        assert result['total_daily_cost'] < 1.0, \
            f"Daily cost ${result['total_daily_cost']:.4f} exceeds target of $1.00"
        
        # Verify target is met
        assert result['meets_target'] is True
        
        # Verify we have significant headroom (at least 50% under budget)
        assert result['total_daily_cost'] < 0.5, \
            f"Daily cost ${result['total_daily_cost']:.4f} should have more headroom"
    
    def test_cost_per_memory_is_reasonable(self):
        """Test that cost per memory is reasonable"""
        result = calculate_total_daily_cost(
            memories_per_day=1000,
            use_batching=True
        )
        
        # Cost per memory should be less than $0.001 (0.1 cents)
        assert result['cost_per_memory'] < 0.001
        
        # Cost per memory should be positive
        assert result['cost_per_memory'] > 0.0
    
    def test_high_volume_scenario(self):
        """Test cost at high volume (5000 memories/day)"""
        result = calculate_total_daily_cost(
            memories_per_day=5000,
            use_batching=True
        )
        
        # Even at 5x volume, should still be under $1
        assert result['total_daily_cost'] < 1.0
        assert result['meets_target'] is True
    
    def test_low_volume_scenario(self):
        """Test cost at low volume (100 memories/day)"""
        result = calculate_total_daily_cost(
            memories_per_day=100,
            use_batching=True
        )
        
        # Low volume should be well under $1
        assert result['total_daily_cost'] < 0.1
        assert result['meets_target'] is True
    
    def test_batch_api_provides_cost_savings(self):
        """Test that batch API provides significant cost savings"""
        with_batch = calculate_total_daily_cost(1000, use_batching=True)
        without_batch = calculate_total_daily_cost(1000, use_batching=False)
        
        # Batch API should reduce cost
        assert with_batch['total_daily_cost'] < without_batch['total_daily_cost']
        
        # Calculate savings percentage
        savings = (without_batch['total_daily_cost'] - with_batch['total_daily_cost']) / without_batch['total_daily_cost']
        
        # Should have at least 10% savings from batching
        assert savings > 0.1
    
    def test_model_pricing_is_defined(self):
        """Test that model pricing is properly defined"""
        # Verify Gemini Flash is available (cheapest model)
        assert "gemini-flash" in MODELS
        
        gemini = MODELS["gemini-flash"]
        
        # Verify pricing is reasonable
        assert gemini.input_cost_per_1k > 0.0
        assert gemini.output_cost_per_1k > 0.0
        
        # Gemini Flash should be cheaper than GPT-3.5
        gpt = MODELS["gpt-3.5-turbo"]
        assert gemini.input_cost_per_1k < gpt.input_cost_per_1k
        assert gemini.output_cost_per_1k < gpt.output_cost_per_1k
    
    def test_cost_breakdown_completeness(self):
        """Test that cost breakdown includes all operations"""
        result = calculate_total_daily_cost(1000, use_batching=True)
        
        # Should have 3 operations
        assert len(result['breakdown']) == 3
        
        # Verify all operations are present
        operations = [c.operation for c in result['breakdown']]
        assert "Semantic Indexing (Batch)" in operations
        assert "Vector Embeddings (Local)" in operations
        assert "Arrow Compression (Local)" in operations
    
    def test_cost_calculation_consistency(self):
        """Test that cost calculations are consistent"""
        result = calculate_total_daily_cost(1000, use_batching=True)
        
        # Sum of breakdown should equal total
        breakdown_sum = sum(c.total_daily_cost for c in result['breakdown'])
        
        # Allow for floating point precision
        assert abs(breakdown_sum - result['total_daily_cost']) < 0.0001
    
    def test_optimization_impact(self):
        """Test that optimizations provide significant cost reduction"""
        # Baseline: GPT-3.5 without batching
        baseline_model = MODELS["gpt-3.5-turbo"]
        memories = 1000
        avg_tokens = 200
        prompt_overhead = 150
        output_tokens = 100
        
        baseline_input_cost = ((avg_tokens + prompt_overhead) / 1000) * baseline_model.input_cost_per_1k
        baseline_output_cost = (output_tokens / 1000) * baseline_model.output_cost_per_1k
        baseline_cost_per_call = baseline_input_cost + baseline_output_cost
        baseline_daily_cost = baseline_cost_per_call * memories
        
        # Optimized
        optimized = calculate_total_daily_cost(memories, use_batching=True)
        optimized_cost = optimized['total_daily_cost']
        
        # Calculate reduction
        cost_reduction = baseline_daily_cost - optimized_cost
        reduction_percentage = (cost_reduction / baseline_daily_cost) * 100
        
        # Should have at least 50% cost reduction
        assert reduction_percentage > 50.0, \
            f"Cost reduction {reduction_percentage:.1f}% is less than expected 50%"
        
        # Verify absolute cost reduction
        assert cost_reduction > 0.1, \
            f"Absolute cost reduction ${cost_reduction:.4f} is too small"


class TestCostValidationEdgeCases:
    """Test edge cases and boundary conditions"""
    
    def test_zero_memories(self):
        """Test cost calculation with zero memories"""
        result = calculate_total_daily_cost(
            memories_per_day=0,
            use_batching=True
        )
        
        # Should have zero cost
        assert result['total_daily_cost'] == 0.0
    
    def test_single_memory(self):
        """Test cost calculation with single memory"""
        result = calculate_total_daily_cost(
            memories_per_day=1,
            use_batching=True
        )
        
        # Should still be under $1
        assert result['total_daily_cost'] < 1.0
        assert result['meets_target'] is True
    
    def test_very_high_volume(self):
        """Test cost at very high volume (100,000 memories/day)"""
        result = calculate_total_daily_cost(
            memories_per_day=100000,
            use_batching=True
        )
        
        # Cost should scale linearly
        # At 100x volume, cost should be roughly 100x
        baseline = calculate_total_daily_cost(1000, use_batching=True)
        expected_cost = baseline['total_daily_cost'] * 100
        
        # Allow 10% variance
        assert abs(result['total_daily_cost'] - expected_cost) / expected_cost < 0.1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
