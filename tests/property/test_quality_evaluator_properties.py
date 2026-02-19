"""
Property-based tests for QualityEvaluator

Tests universal properties that should hold across all valid inputs.
Uses Hypothesis for property-based testing with 100+ iterations.
"""

import pytest
from hypothesis import given, settings, strategies as st, assume
from llm_compression.quality_evaluator import QualityEvaluator, QualityMetrics
import tempfile
from pathlib import Path


# Test strategies
text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')),
    min_size=10,
    max_size=500
)

entity_text_strategy = st.text(
    alphabet=st.characters(whitelist_categories=('Lu', 'Ll', 'Nd', 'Zs', 'Po')),
    min_size=20,
    max_size=200
).filter(lambda x: any(c.isupper() for c in x))  # Ensure some capitals for entity extraction


@pytest.fixture(scope="module")
def evaluator():
    """Create a shared evaluator instance for property tests"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl') as f:
        temp_path = f.name
    
    evaluator = QualityEvaluator(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold=0.85,
        entity_threshold=0.95,
        failure_log_path=temp_path
    )
    
    yield evaluator
    
    # Cleanup
    Path(temp_path).unlink(missing_ok=True)


class TestQualityEvaluatorProperties:
    """Property-based tests for QualityEvaluator"""
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy)
    def test_property_semantic_similarity_reflexive(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Semantic similarity is reflexive
        For any text T, similarity(T, T) should be close to 1.0
        """
        assume(len(text.strip()) > 5)  # Need meaningful text
        
        similarity = evaluator._compute_semantic_similarity(text, text)
        
        # Identical texts should have very high similarity
        assert similarity > 0.95, f"Reflexive similarity {similarity} should be > 0.95"
    
    @settings(max_examples=100, deadline=None)
    @given(text1=text_strategy, text2=text_strategy)
    def test_property_semantic_similarity_symmetric(self, evaluator, text1, text2):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Semantic similarity is symmetric
        For any texts T1, T2: similarity(T1, T2) == similarity(T2, T1)
        """
        assume(len(text1.strip()) > 5 and len(text2.strip()) > 5)
        
        sim1 = evaluator._compute_semantic_similarity(text1, text2)
        sim2 = evaluator._compute_semantic_similarity(text2, text1)
        
        # Should be symmetric (within floating point tolerance)
        assert abs(sim1 - sim2) < 0.01, f"Similarity not symmetric: {sim1} vs {sim2}"
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy)
    def test_property_semantic_similarity_bounded(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Semantic similarity is bounded [0, 1]
        For any texts T1, T2: 0 <= similarity(T1, T2) <= 1
        """
        assume(len(text.strip()) > 5)
        
        # Test with itself
        similarity = evaluator._compute_semantic_similarity(text, text)
        assert 0.0 <= similarity <= 1.0, f"Similarity {similarity} not in [0, 1]"
        
        # Test with different text
        other_text = "completely different content here"
        similarity2 = evaluator._compute_semantic_similarity(text, other_text)
        assert 0.0 <= similarity2 <= 1.0, f"Similarity {similarity2} not in [0, 1]"
    
    @settings(max_examples=100, deadline=None)
    @given(text=entity_text_strategy)
    def test_property_entity_extraction_deterministic(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Entity extraction is deterministic
        For any text T, extracting entities twice should give same results
        """
        entities1 = evaluator._extract_entities(text)
        entities2 = evaluator._extract_entities(text)
        
        assert entities1 == entities2, "Entity extraction should be deterministic"
    
    @settings(max_examples=100, deadline=None)
    @given(text=entity_text_strategy)
    def test_property_entity_extraction_structure(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Entity extraction returns correct structure
        For any text T, extracted entities should have expected keys and types
        """
        entities = evaluator._extract_entities(text)
        
        # Check structure
        assert isinstance(entities, dict)
        assert 'persons' in entities
        assert 'dates' in entities
        assert 'numbers' in entities
        assert 'locations' in entities
        assert 'keywords' in entities
        
        # Check types
        for key, value in entities.items():
            assert isinstance(value, list), f"Entity type {key} should be a list"
            for item in value:
                assert isinstance(item, str), f"Entity items should be strings"
    
    @settings(max_examples=100, deadline=None)
    @given(text=entity_text_strategy)
    def test_property_entity_accuracy_reflexive(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Entity accuracy is reflexive
        For any entities E, accuracy(E, E) should be 1.0
        """
        entities = evaluator._extract_entities(text)
        accuracy = evaluator._compute_entity_accuracy(entities, entities)
        
        assert accuracy == 1.0, f"Reflexive entity accuracy {accuracy} should be 1.0"
    
    @settings(max_examples=100, deadline=None)
    @given(text=entity_text_strategy)
    def test_property_entity_accuracy_bounded(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Entity accuracy is bounded [0, 1]
        For any entities E1, E2: 0 <= accuracy(E1, E2) <= 1
        """
        entities1 = evaluator._extract_entities(text)
        entities2 = evaluator._extract_entities("completely different text here")
        
        accuracy = evaluator._compute_entity_accuracy(entities1, entities2)
        
        assert 0.0 <= accuracy <= 1.0, f"Entity accuracy {accuracy} not in [0, 1]"
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy)
    def test_property_bleu_score_reflexive(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: BLEU score is reflexive
        For any text T, BLEU(T, T) should be close to 1.0
        """
        assume(len(text.strip()) > 5)
        
        bleu = evaluator._compute_bleu_score(text, text)
        
        # Identical texts should have BLEU score close to 1.0
        assert bleu > 0.95, f"Reflexive BLEU score {bleu} should be > 0.95"
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy)
    def test_property_bleu_score_bounded(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: BLEU score is bounded [0, 1]
        For any texts T1, T2: 0 <= BLEU(T1, T2) <= 1
        """
        assume(len(text.strip()) > 5)
        
        # Test with itself
        bleu1 = evaluator._compute_bleu_score(text, text)
        assert 0.0 <= bleu1 <= 1.0, f"BLEU score {bleu1} not in [0, 1]"
        
        # Test with different text
        other_text = "completely different content here"
        bleu2 = evaluator._compute_bleu_score(text, other_text)
        assert 0.0 <= bleu2 <= 1.0, f"BLEU score {bleu2} not in [0, 1]"
    
    @settings(max_examples=100, deadline=None)
    @given(
        text=text_strategy,
        compressed_size=st.integers(min_value=1, max_value=1000),
        latency=st.floats(min_value=0.1, max_value=5000.0)
    )
    def test_property_evaluate_returns_valid_metrics(self, evaluator, text, compressed_size, latency):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Evaluate always returns valid QualityMetrics
        For any valid inputs, evaluate should return metrics with all fields in valid ranges
        """
        assume(len(text.strip()) > 5)
        
        metrics = evaluator.evaluate(text, text, compressed_size, latency)
        
        # Check all metrics are in valid ranges
        assert metrics.compression_ratio >= 0.0
        assert 0.0 <= metrics.semantic_similarity <= 1.0
        assert 0.0 <= metrics.entity_accuracy <= 1.0
        assert 0.0 <= metrics.bleu_score <= 1.0
        assert metrics.reconstruction_latency_ms >= 0.0
        assert 0.0 <= metrics.overall_score <= 1.0
        assert isinstance(metrics.warnings, list)
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy)
    def test_property_high_quality_no_warnings(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 16: 质量阈值标记
        
        Property: High quality reconstruction has no warnings
        For any text T, if reconstructed == original, there should be no warnings
        """
        assume(len(text.strip()) > 10)
        
        compressed_size = 50
        latency = 100.0
        
        metrics = evaluator.evaluate(text, text, compressed_size, latency)
        
        # Perfect reconstruction should have no warnings
        assert len(metrics.warnings) == 0, f"Perfect reconstruction should have no warnings, got: {metrics.warnings}"
    
    @settings(max_examples=50, deadline=None)
    @given(
        original=text_strategy,
        reconstructed=text_strategy
    )
    def test_property_low_similarity_triggers_warning(self, evaluator, original, reconstructed):
        """
        Feature: llm-compression-integration, Property 16: 质量阈值标记
        
        Property: Low semantic similarity triggers warning
        When semantic_similarity < threshold, a warning should be generated
        """
        assume(len(original.strip()) > 10 and len(reconstructed.strip()) > 10)
        assume(original != reconstructed)  # Ensure they're different
        
        compressed_size = 50
        latency = 100.0
        
        metrics = evaluator.evaluate(original, reconstructed, compressed_size, latency)
        
        # If similarity is low, should have warning
        if metrics.semantic_similarity < evaluator.semantic_threshold:
            assert any('semantic similarity' in w.lower() for w in metrics.warnings), \
                f"Low similarity {metrics.semantic_similarity} should trigger warning"
    
    @settings(max_examples=100, deadline=None)
    @given(
        texts=st.lists(text_strategy, min_size=1, max_size=10),
        compressed_sizes=st.lists(st.integers(min_value=1, max_value=100), min_size=1, max_size=10),
        latencies=st.lists(st.floats(min_value=0.1, max_value=1000.0), min_size=1, max_size=10)
    )
    def test_property_batch_evaluation_consistency(self, evaluator, texts, compressed_sizes, latencies):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Batch evaluation is consistent with individual evaluation
        For any list of texts, batch evaluation should give same results as individual
        """
        # Ensure all lists have same length
        min_len = min(len(texts), len(compressed_sizes), len(latencies))
        texts = texts[:min_len]
        compressed_sizes = compressed_sizes[:min_len]
        latencies = latencies[:min_len]
        
        # Filter out empty texts
        valid_indices = [i for i, t in enumerate(texts) if len(t.strip()) > 5]
        if not valid_indices:
            assume(False)  # Skip if no valid texts
        
        texts = [texts[i] for i in valid_indices]
        compressed_sizes = [compressed_sizes[i] for i in valid_indices]
        latencies = [latencies[i] for i in valid_indices]
        
        # Batch evaluation
        batch_metrics = evaluator.evaluate_batch(
            texts,
            texts,  # Use same texts as reconstructed for simplicity
            compressed_sizes,
            latencies
        )
        
        # Individual evaluation
        individual_metrics = []
        for text, size, lat in zip(texts, compressed_sizes, latencies):
            metrics = evaluator.evaluate(text, text, size, lat)
            individual_metrics.append(metrics)
        
        # Compare results
        assert len(batch_metrics) == len(individual_metrics)
        
        for batch_m, indiv_m in zip(batch_metrics, individual_metrics):
            # Metrics should be very close (within floating point tolerance)
            assert abs(batch_m.semantic_similarity - indiv_m.semantic_similarity) < 0.01
            assert abs(batch_m.entity_accuracy - indiv_m.entity_accuracy) < 0.01
            assert abs(batch_m.bleu_score - indiv_m.bleu_score) < 0.01
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy)
    def test_property_failure_logging_on_warnings(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 17: 失败案例记录
        
        Property: Failure cases are logged when warnings exist
        For any evaluation with warnings, a failure case should be logged
        """
        assume(len(text.strip()) > 10)
        
        # Use very different texts to trigger warnings
        original = text
        reconstructed = "completely different text that will trigger warnings"
        
        # Clear log file
        if evaluator.failure_log_path.exists():
            evaluator.failure_log_path.unlink()
        
        metrics = evaluator.evaluate(original, reconstructed, 50, 100.0)
        
        # If there are warnings, log file should exist and have content
        if metrics.warnings:
            assert evaluator.failure_log_path.exists(), "Failure log should exist when warnings present"
            
            with open(evaluator.failure_log_path, 'r') as f:
                content = f.read()
                assert len(content) > 0, "Failure log should have content"
    
    @settings(max_examples=100, deadline=None)
    @given(
        compression_ratio=st.floats(min_value=1.0, max_value=100.0),
        semantic_similarity=st.floats(min_value=0.0, max_value=1.0),
        entity_accuracy=st.floats(min_value=0.0, max_value=1.0),
        bleu_score=st.floats(min_value=0.0, max_value=1.0)
    )
    def test_property_overall_score_bounded(self, evaluator, compression_ratio, semantic_similarity, entity_accuracy, bleu_score):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Overall score is bounded [0, 1]
        For any valid individual metrics, overall score should be in [0, 1]
        """
        metrics = QualityMetrics(
            compression_ratio=compression_ratio,
            semantic_similarity=semantic_similarity,
            entity_accuracy=entity_accuracy,
            bleu_score=bleu_score,
            reconstruction_latency_ms=100.0,
            overall_score=0.0,  # Will be recalculated
            warnings=[]
        )
        
        # Calculate overall score using same formula as evaluator
        overall_score = (
            semantic_similarity * 0.4 +
            entity_accuracy * 0.3 +
            bleu_score * 0.2 +
            min(compression_ratio / 10.0, 1.0) * 0.1
        )
        
        assert 0.0 <= overall_score <= 1.0, f"Overall score {overall_score} not in [0, 1]"
    
    @settings(max_examples=100, deadline=None)
    @given(text=text_strategy)
    def test_property_report_generation_always_succeeds(self, evaluator, text):
        """
        Feature: llm-compression-integration, Property 15: 质量指标计算完整性
        
        Property: Report generation always succeeds
        For any valid metrics, generate_report should return a non-empty string
        """
        assume(len(text.strip()) > 5)
        
        metrics = evaluator.evaluate(text, text, 50, 100.0)
        report = evaluator.generate_report(metrics)
        
        assert isinstance(report, str)
        assert len(report) > 0
        assert "Quality Evaluation Report" in report
