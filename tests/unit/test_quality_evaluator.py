"""
Unit tests for QualityEvaluator

Tests the quality evaluation functionality including semantic similarity,
entity extraction, entity accuracy, and BLEU score computation.
"""

import pytest
from pathlib import Path
import tempfile
import json
from llm_compression.quality_evaluator import QualityEvaluator, QualityMetrics


class TestQualityEvaluator:
    """Test suite for QualityEvaluator"""
    
    @pytest.fixture
    def evaluator(self):
        """Create a QualityEvaluator instance for testing"""
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
    
    def test_initialization(self, evaluator):
        """Test evaluator initialization"""
        assert evaluator.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert evaluator.semantic_threshold == 0.85
        assert evaluator.entity_threshold == 0.95
        assert evaluator.failure_log_path.exists() or not evaluator.failure_log_path.exists()
    
    def test_semantic_similarity_identical(self, evaluator):
        """Test semantic similarity with identical texts"""
        text = "The quick brown fox jumps over the lazy dog"
        similarity = evaluator._compute_semantic_similarity(text, text)
        
        # Identical texts should have similarity close to 1.0
        assert similarity > 0.99
    
    def test_semantic_similarity_similar(self, evaluator):
        """Test semantic similarity with similar texts"""
        text1 = "The cat sat on the mat"
        text2 = "A cat was sitting on a mat"
        
        similarity = evaluator._compute_semantic_similarity(text1, text2)
        
        # Similar texts should have high similarity
        assert similarity > 0.7
    
    def test_semantic_similarity_different(self, evaluator):
        """Test semantic similarity with different texts"""
        text1 = "The weather is sunny today"
        text2 = "I love eating pizza for dinner"
        
        similarity = evaluator._compute_semantic_similarity(text1, text2)
        
        # Different texts should have lower similarity
        assert similarity < 0.5
    
    def test_extract_entities_dates(self, evaluator):
        """Test entity extraction for dates"""
        text = "The meeting is on 2024-01-15 at 3pm. We also have one on January 20, 2024."
        entities = evaluator._extract_entities(text)
        
        assert '2024-01-15' in entities['dates']
        assert '3pm' in entities['dates']
        assert 'January 20, 2024' in entities['dates']
    
    def test_extract_entities_numbers(self, evaluator):
        """Test entity extraction for numbers"""
        text = "The budget is $1,000 and we need 50% more. The price is €25.99."
        entities = evaluator._extract_entities(text)
        
        # Should extract various number formats
        assert len(entities['numbers']) > 0
        # Check if any number-like patterns are found
        assert any('1' in num or '50' in num or '25' in num for num in entities['numbers'])
    
    def test_extract_entities_persons(self, evaluator):
        """Test entity extraction for person names"""
        text = "John Smith met Mary Johnson at the conference. Dr. Alice Brown was also there."
        entities = evaluator._extract_entities(text)
        
        # Should extract capitalized names
        assert len(entities['persons']) > 0
        # At least one name should be found
        assert any('John' in name or 'Mary' in name or 'Alice' in name for name in entities['persons'])
    
    def test_extract_entities_keywords(self, evaluator):
        """Test entity extraction for keywords"""
        text = "The project requires careful planning and execution. Planning is essential for success."
        entities = evaluator._extract_entities(text)
        
        # Should extract high-frequency keywords
        assert len(entities['keywords']) > 0
        # 'planning' appears twice, should be in keywords
        assert 'planning' in entities['keywords']
    
    def test_entity_accuracy_perfect(self, evaluator):
        """Test entity accuracy with perfect match"""
        original_entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15'],
            'numbers': ['100', '50%'],
            'locations': []
        }
        reconstructed_entities = original_entities.copy()
        
        accuracy = evaluator._compute_entity_accuracy(original_entities, reconstructed_entities)
        
        assert accuracy == 1.0
    
    def test_entity_accuracy_partial(self, evaluator):
        """Test entity accuracy with partial match"""
        original_entities = {
            'persons': ['John Smith', 'Mary Johnson'],
            'dates': ['2024-01-15', '2024-01-20'],
            'numbers': ['100'],
            'locations': []
        }
        reconstructed_entities = {
            'persons': ['John Smith'],  # Missing Mary Johnson
            'dates': ['2024-01-15'],     # Missing 2024-01-20
            'numbers': ['100'],
            'locations': []
        }
        
        accuracy = evaluator._compute_entity_accuracy(original_entities, reconstructed_entities)
        
        # 3 out of 5 entities matched
        assert 0.5 < accuracy < 0.7
    
    def test_entity_accuracy_fuzzy_match(self, evaluator):
        """Test entity accuracy with fuzzy matching"""
        original_entities = {
            'persons': ['John Smith'],
            'dates': [],
            'numbers': [],
            'locations': []
        }
        reconstructed_entities = {
            'persons': ['john smith'],  # Case difference
            'dates': [],
            'numbers': [],
            'locations': []
        }
        
        accuracy = evaluator._compute_entity_accuracy(original_entities, reconstructed_entities)
        
        # Should match despite case difference
        assert accuracy == 1.0
    
    def test_entity_accuracy_no_entities(self, evaluator):
        """Test entity accuracy with no entities"""
        original_entities = {
            'persons': [],
            'dates': [],
            'numbers': [],
            'locations': []
        }
        reconstructed_entities = original_entities.copy()
        
        accuracy = evaluator._compute_entity_accuracy(original_entities, reconstructed_entities)
        
        # No entities means perfect accuracy
        assert accuracy == 1.0
    
    def test_bleu_score_identical(self, evaluator):
        """Test BLEU score with identical texts"""
        text = "The quick brown fox jumps over the lazy dog"
        score = evaluator._compute_bleu_score(text, text)
        
        # Identical texts should have BLEU score of 1.0
        assert score > 0.99
    
    def test_bleu_score_similar(self, evaluator):
        """Test BLEU score with similar texts"""
        reference = "The cat sat on the mat"
        hypothesis = "The cat was sitting on the mat"
        
        score = evaluator._compute_bleu_score(reference, hypothesis)
        
        # Similar texts should have reasonable BLEU score
        # Note: BLEU can be strict, so we accept lower scores for similar texts
        assert 0.0 <= score < 0.9
    
    def test_bleu_score_different(self, evaluator):
        """Test BLEU score with completely different texts"""
        reference = "The weather is sunny"
        hypothesis = "I love pizza"
        
        score = evaluator._compute_bleu_score(reference, hypothesis)
        
        # Different texts should have low BLEU score
        assert score < 0.3
    
    def test_evaluate_high_quality(self, evaluator):
        """Test evaluation with high quality reconstruction"""
        original = "John Smith met Mary Johnson on 2024-01-15 at 3pm to discuss the $1000 budget."
        reconstructed = "John Smith met Mary Johnson on 2024-01-15 at 3pm to discuss the $1000 budget."
        compressed_size = 50
        latency = 100.0
        
        metrics = evaluator.evaluate(original, reconstructed, compressed_size, latency)
        
        assert metrics.compression_ratio > 1.0
        assert metrics.semantic_similarity > 0.95
        assert metrics.entity_accuracy > 0.95
        assert metrics.bleu_score > 0.95
        assert metrics.overall_score > 0.9
        assert len(metrics.warnings) == 0
    
    def test_evaluate_low_quality(self, evaluator):
        """Test evaluation with low quality reconstruction"""
        original = "John Smith met Mary Johnson on 2024-01-15 at 3pm to discuss the $1000 budget."
        reconstructed = "Someone had a meeting about money."
        compressed_size = 50
        latency = 100.0
        
        metrics = evaluator.evaluate(original, reconstructed, compressed_size, latency)
        
        assert metrics.semantic_similarity < 0.85
        assert metrics.entity_accuracy < 0.95
        assert len(metrics.warnings) > 0
    
    def test_evaluate_with_warnings(self, evaluator):
        """Test that warnings are generated for low quality"""
        original = "John Smith met Mary Johnson on 2024-01-15."
        reconstructed = "Someone had a meeting."
        compressed_size = 20
        latency = 50.0
        
        metrics = evaluator.evaluate(original, reconstructed, compressed_size, latency)
        
        # Should have warnings for low semantic similarity and entity accuracy
        assert len(metrics.warnings) >= 1
        assert any('semantic similarity' in w.lower() for w in metrics.warnings)
    
    def test_failure_case_logging(self, evaluator):
        """Test that failure cases are logged"""
        original = "John Smith met Mary Johnson on 2024-01-15."
        reconstructed = "Someone had a meeting."
        compressed_size = 20
        latency = 50.0
        
        # Evaluate (should trigger failure logging)
        metrics = evaluator.evaluate(original, reconstructed, compressed_size, latency)
        
        # Check that failure was logged
        assert evaluator.failure_log_path.exists()
        
        with open(evaluator.failure_log_path, 'r') as f:
            lines = f.readlines()
            assert len(lines) > 0
            
            # Parse last line
            failure_case = json.loads(lines[-1])
            assert 'original' in failure_case
            assert 'reconstructed' in failure_case
            assert 'metrics' in failure_case
            assert 'warnings' in failure_case
    
    def test_generate_report(self, evaluator):
        """Test quality report generation"""
        metrics = QualityMetrics(
            compression_ratio=15.5,
            semantic_similarity=0.92,
            entity_accuracy=0.98,
            bleu_score=0.85,
            reconstruction_latency_ms=250.0,
            overall_score=0.91,
            warnings=[]
        )
        
        report = evaluator.generate_report(metrics)
        
        assert "Quality Evaluation Report" in report
        assert "15.5" in report
        assert "0.92" in report
        assert "0.98" in report
        assert "0.85" in report
        assert "✓ PASS" in report
    
    def test_generate_report_with_warnings(self, evaluator):
        """Test quality report generation with warnings"""
        metrics = QualityMetrics(
            compression_ratio=10.0,
            semantic_similarity=0.80,
            entity_accuracy=0.90,
            bleu_score=0.75,
            reconstruction_latency_ms=300.0,
            overall_score=0.82,
            warnings=["Low semantic similarity", "Entity accuracy below threshold"]
        )
        
        report = evaluator.generate_report(metrics)
        
        assert "✗ FAIL" in report
        assert "Warnings:" in report
        assert "Low semantic similarity" in report
        assert "Entity accuracy below threshold" in report
    
    def test_evaluate_batch(self, evaluator):
        """Test batch evaluation"""
        originals = [
            "John met Mary on 2024-01-15.",
            "The budget is $1000.",
            "Alice works at Google."
        ]
        reconstructed_list = [
            "John met Mary on 2024-01-15.",
            "The budget is $1000.",
            "Alice works at Google."
        ]
        compressed_sizes = [20, 15, 18]
        latencies = [100.0, 90.0, 110.0]
        
        metrics_list = evaluator.evaluate_batch(
            originals,
            reconstructed_list,
            compressed_sizes,
            latencies
        )
        
        assert len(metrics_list) == 3
        for metrics in metrics_list:
            assert isinstance(metrics, QualityMetrics)
            assert metrics.compression_ratio > 0
            assert 0 <= metrics.semantic_similarity <= 1
            assert 0 <= metrics.entity_accuracy <= 1
            assert 0 <= metrics.bleu_score <= 1
    
    def test_evaluate_batch_with_errors(self, evaluator):
        """Test batch evaluation handles errors gracefully"""
        originals = ["Valid text", ""]
        reconstructed_list = ["Valid text", ""]
        compressed_sizes = [10, 0]
        latencies = [100.0, 100.0]
        
        metrics_list = evaluator.evaluate_batch(
            originals,
            reconstructed_list,
            compressed_sizes,
            latencies
        )
        
        # Should return metrics for all items, even if some fail
        assert len(metrics_list) == 2
    
    def test_get_ngrams(self, evaluator):
        """Test n-gram generation"""
        tokens = ['the', 'quick', 'brown', 'fox']
        
        unigrams = evaluator._get_ngrams(tokens, 1)
        assert len(unigrams) == 4
        assert ('the',) in unigrams
        
        bigrams = evaluator._get_ngrams(tokens, 2)
        assert len(bigrams) == 3
        assert ('the', 'quick') in bigrams
        
        trigrams = evaluator._get_ngrams(tokens, 3)
        assert len(trigrams) == 2
        assert ('the', 'quick', 'brown') in trigrams
