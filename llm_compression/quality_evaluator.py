"""
Quality Evaluator for LLM Compression System

This module implements quality evaluation for compressed memories,
calculating metrics like semantic similarity, entity accuracy, BLEU score,
and compression ratio.

Requirements: 7.1-7.7
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import time
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class QualityMetrics:
    """Quality metrics for compression evaluation
    
    Attributes:
        compression_ratio: Compression ratio (original_size / compressed_size)
        semantic_similarity: Semantic similarity score (0-1)
        entity_accuracy: Entity accuracy score (0-1)
        bleu_score: BLEU score (0-1)
        reconstruction_latency_ms: Reconstruction latency in milliseconds
        overall_score: Overall quality score (0-1)
        warnings: List of warning messages
    """
    compression_ratio: float
    semantic_similarity: float
    entity_accuracy: float
    bleu_score: float
    reconstruction_latency_ms: float
    overall_score: float
    warnings: List[str]


class QualityEvaluator:
    """Quality evaluator for compression and reconstruction
    
    This class evaluates the quality of compressed memories by calculating
    various metrics including semantic similarity, entity accuracy, and BLEU score.
    
    Requirements: 7.1-7.7
    """
    
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        semantic_threshold: float = 0.85,
        entity_threshold: float = 0.95,
        failure_log_path: Optional[str] = None
    ):
        """Initialize quality evaluator
        
        Args:
            embedding_model: Sentence transformer model for semantic similarity
                           Default: paraphrase-multilingual-MiniLM-L12-v2 (multilingual, 50+ languages)
                           Alternative: all-MiniLM-L6-v2 (English only, faster)
            semantic_threshold: Threshold for low quality warning (default: 0.85)
            entity_threshold: Threshold for critical information loss (default: 0.95)
            failure_log_path: Path to log failure cases (default: ./quality_failures.jsonl)
        
        Requirements: 7.1, 7.3, 7.4, 7.7
        """
        self.embedding_model_name = embedding_model
        self.semantic_threshold = semantic_threshold
        self.entity_threshold = entity_threshold
        
        # Set up failure log path
        if failure_log_path is None:
            failure_log_path = "./quality_failures.jsonl"
        self.failure_log_path = Path(failure_log_path)
        
        # Lazy load embedding model (only when needed)
        self._embedding_model = None
        self._tokenizer = None
        
        logger.info(
            f"QualityEvaluator initialized with model={embedding_model}, "
            f"semantic_threshold={semantic_threshold}, "
            f"entity_threshold={entity_threshold}"
        )
    
    @property
    def embedding_model(self):
        """Lazy load embedding model"""
        if self._embedding_model is None:
            try:
                import os
                # 使用 HF 镜像（国内访问快）
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
                
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self.embedding_model_name}")
                logger.info("Model: paraphrase-multilingual-MiniLM-L12-v2 (50+ languages, CPU mode)")
                
                self._embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    device='cpu'  # AMD ROCm 不支持，使用 CPU
                )
                logger.info("Embedding model loaded on CPU")
            except ImportError:
                logger.error("sentence-transformers not installed. Install with: pip install sentence-transformers")
                raise
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
                raise
        return self._embedding_model
    
    def evaluate(
        self,
        original: str,
        reconstructed: str,
        compressed_size: int,
        reconstruction_latency_ms: float,
        original_entities: Optional[Dict[str, List[str]]] = None
    ) -> QualityMetrics:
        """Evaluate compression quality
        
        Args:
            original: Original text
            reconstructed: Reconstructed text
            compressed_size: Size of compressed data in bytes
            reconstruction_latency_ms: Reconstruction latency in milliseconds
            original_entities: Pre-extracted entities from original text (optional)
        
        Returns:
            QualityMetrics: Quality metrics
        
        Requirements: 7.1, 7.2
        """
        logger.debug(f"Evaluating quality for text of length {len(original)}")
        
        warnings = []
        
        # Calculate compression ratio
        original_size = len(original.encode('utf-8'))
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0.0
        
        # Calculate semantic similarity
        semantic_similarity = self._compute_semantic_similarity(original, reconstructed)
        
        # Calculate entity accuracy
        if original_entities is None:
            original_entities = self._extract_entities(original)
        reconstructed_entities = self._extract_entities(reconstructed)
        entity_accuracy = self._compute_entity_accuracy(original_entities, reconstructed_entities)
        
        # Calculate BLEU score
        bleu_score = self._compute_bleu_score(original, reconstructed)
        
        # Check thresholds and add warnings
        if semantic_similarity < self.semantic_threshold:
            warning = f"Low semantic similarity: {semantic_similarity:.3f} < {self.semantic_threshold}"
            warnings.append(warning)
            logger.warning(warning)
        
        if entity_accuracy < self.entity_threshold:
            warning = f"Critical information loss: entity accuracy {entity_accuracy:.3f} < {self.entity_threshold}"
            warnings.append(warning)
            logger.warning(warning)
        
        # Calculate overall score (weighted average)
        overall_score = (
            semantic_similarity * 0.4 +
            entity_accuracy * 0.3 +
            bleu_score * 0.2 +
            min(compression_ratio / 10.0, 1.0) * 0.1  # Normalize compression ratio
        )
        
        metrics = QualityMetrics(
            compression_ratio=compression_ratio,
            semantic_similarity=semantic_similarity,
            entity_accuracy=entity_accuracy,
            bleu_score=bleu_score,
            reconstruction_latency_ms=reconstruction_latency_ms,
            overall_score=overall_score,
            warnings=warnings
        )
        
        # Log failure cases
        if warnings:
            self._log_failure_case(original, reconstructed, metrics)
        
        logger.debug(
            f"Quality evaluation complete: "
            f"compression_ratio={compression_ratio:.2f}x, "
            f"semantic_similarity={semantic_similarity:.3f}, "
            f"entity_accuracy={entity_accuracy:.3f}, "
            f"bleu_score={bleu_score:.3f}, "
            f"overall_score={overall_score:.3f}"
        )
        
        return metrics
    
    def _compute_semantic_similarity(self, text1: str, text2: str) -> float:
        """Compute semantic similarity using embedding cosine similarity
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            float: Cosine similarity (0-1)
        
        Requirements: 7.1
        """
        try:
            # Generate embeddings for both texts
            embeddings = self.embedding_model.encode([text1, text2])
            
            # Compute cosine similarity
            import numpy as np
            from numpy.linalg import norm
            
            emb1, emb2 = embeddings[0], embeddings[1]
            
            # Cosine similarity: dot(a, b) / (norm(a) * norm(b))
            similarity = np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
            
            # Ensure result is in [0, 1] range
            # Cosine similarity is in [-1, 1], but for text it's usually [0, 1]
            similarity = max(0.0, min(1.0, float(similarity)))
            
            return similarity
        except Exception as e:
            logger.error(f"Failed to compute semantic similarity: {e}")
            # Return 0 on error to be conservative
            return 0.0
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract entities from text
        
        Uses regex patterns to extract:
        - persons: Capitalized names
        - dates: ISO dates and natural language dates
        - numbers: Integers, decimals, percentages
        - locations: Common location patterns
        - keywords: High-frequency meaningful words
        
        Args:
            text: Input text
        
        Returns:
            Dict mapping entity types to lists of entities
        
        Requirements: 7.1
        """
        import re
        from collections import Counter
        
        entities = {
            'persons': [],
            'dates': [],
            'numbers': [],
            'locations': [],
            'keywords': []
        }
        
        try:
            # Extract dates
            # ISO format: 2024-01-15
            iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', text)
            entities['dates'].extend(iso_dates)
            
            # Natural language dates: January 15, 2024 or Jan 15, 2024
            natural_dates = re.findall(
                r'(?:January|February|March|April|May|June|July|August|'
                r'September|October|November|December|'
                r'Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
                r'\s+\d{1,2},?\s+\d{4}',
                text
            )
            entities['dates'].extend(natural_dates)
            
            # Time patterns: 3pm, 15:30, 3:30pm
            times = re.findall(r'\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)', text)
            entities['dates'].extend(times)
            
            # Extract numbers (integers, decimals, percentages, currency)
            numbers = re.findall(r'[$€£¥]?\d+(?:,\d{3})*(?:\.\d+)?%?', text)
            entities['numbers'].extend(numbers)
            
            # Extract person names (capitalized words, 2-4 words)
            # This is a simple heuristic - real NER would be better
            potential_names = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}\b', text)
            entities['persons'].extend(potential_names)
            
            # Extract keywords (meaningful words, 4+ characters, high frequency)
            words = re.findall(r'\b[a-z]{4,}\b', text.lower())
            if words:
                word_freq = Counter(words)
                # Get top 5 most frequent words
                top_keywords = [word for word, _ in word_freq.most_common(5)]
                entities['keywords'] = top_keywords
            
            # Remove duplicates while preserving order
            for key in entities:
                entities[key] = list(dict.fromkeys(entities[key]))
            
        except Exception as e:
            logger.error(f"Failed to extract entities: {e}")
        
        return entities
    
    def _compute_entity_accuracy(
        self,
        original_entities: Dict[str, List[str]],
        reconstructed_entities: Dict[str, List[str]]
    ) -> float:
        """Compute entity accuracy
        
        Calculates the percentage of original entities that are present
        in the reconstructed text. Uses fuzzy matching for robustness.
        
        Args:
            original_entities: Entities from original text
            reconstructed_entities: Entities from reconstructed text
        
        Returns:
            float: Entity accuracy (0-1)
        
        Requirements: 7.1
        """
        try:
            total_entities = 0
            matched_entities = 0
            
            # Check each entity type
            for entity_type in ['persons', 'dates', 'numbers', 'locations']:
                original_list = original_entities.get(entity_type, [])
                reconstructed_list = reconstructed_entities.get(entity_type, [])
                
                if not original_list:
                    continue
                
                total_entities += len(original_list)
                
                # Check how many original entities are in reconstructed
                for orig_entity in original_list:
                    # Exact match
                    if orig_entity in reconstructed_list:
                        matched_entities += 1
                    else:
                        # Fuzzy match (case-insensitive, partial match)
                        orig_lower = orig_entity.lower()
                        for recon_entity in reconstructed_list:
                            recon_lower = recon_entity.lower()
                            # Check if one contains the other
                            if orig_lower in recon_lower or recon_lower in orig_lower:
                                matched_entities += 1
                                break
            
            # Calculate accuracy
            if total_entities == 0:
                # No entities to check, consider it perfect
                return 1.0
            
            accuracy = matched_entities / total_entities
            return accuracy
            
        except Exception as e:
            logger.error(f"Failed to compute entity accuracy: {e}")
            return 0.0
    
    def _compute_bleu_score(self, reference: str, hypothesis: str) -> float:
        """Compute BLEU score
        
        Implements a simplified BLEU score calculation using n-gram precision.
        Uses unigrams, bigrams, trigrams, and 4-grams.
        
        Args:
            reference: Reference text (original)
            hypothesis: Hypothesis text (reconstructed)
        
        Returns:
            float: BLEU score (0-1)
        
        Requirements: 7.1
        """
        try:
            # Try using nltk if available
            try:
                from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
                from nltk.tokenize import word_tokenize
                
                # Tokenize
                reference_tokens = word_tokenize(reference.lower())
                hypothesis_tokens = word_tokenize(hypothesis.lower())
                
                # Calculate BLEU with smoothing
                smoothing = SmoothingFunction().method1
                score = sentence_bleu(
                    [reference_tokens],
                    hypothesis_tokens,
                    smoothing_function=smoothing
                )
                
                return float(score)
                
            except ImportError:
                # Fallback to custom implementation
                logger.debug("nltk not available, using custom BLEU implementation")
                return self._compute_bleu_custom(reference, hypothesis)
                
        except Exception as e:
            logger.error(f"Failed to compute BLEU score: {e}")
            return 0.0
    
    def _compute_bleu_custom(self, reference: str, hypothesis: str) -> float:
        """Custom BLEU score implementation
        
        Simplified BLEU using n-gram precision (n=1,2,3,4) with geometric mean.
        
        Args:
            reference: Reference text
            hypothesis: Hypothesis text
        
        Returns:
            float: BLEU score (0-1)
        """
        from collections import Counter
        import math
        
        # Tokenize (simple whitespace split)
        ref_tokens = reference.lower().split()
        hyp_tokens = hypothesis.lower().split()
        
        if not hyp_tokens or not ref_tokens:
            return 0.0
        
        # For identical texts, return 1.0 immediately
        if ref_tokens == hyp_tokens:
            return 1.0
        
        # Calculate n-gram precisions for n=1,2,3,4
        precisions = []
        max_n = min(4, len(ref_tokens), len(hyp_tokens))  # Don't use n-grams longer than text
        
        for n in range(1, max_n + 1):
            ref_ngrams = Counter(self._get_ngrams(ref_tokens, n))
            hyp_ngrams = Counter(self._get_ngrams(hyp_tokens, n))
            
            if not hyp_ngrams:
                # If no n-grams of this size, skip this n
                continue
            
            # Count matches (clipped counts)
            matches = sum((ref_ngrams & hyp_ngrams).values())
            total = sum(hyp_ngrams.values())
            
            precision = matches / total if total > 0 else 0.0
            precisions.append(precision)
        
        if not precisions:
            return 0.0
        
        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            # If any precision is 0, BLEU is 0 (standard BLEU behavior)
            # But for very short texts, use arithmetic mean as fallback
            if len(ref_tokens) < 4:
                non_zero = [p for p in precisions if p > 0]
                geo_mean = sum(non_zero) / len(precisions) if non_zero else 0.0
            else:
                return 0.0
        
        # Brevity penalty
        ref_len = len(ref_tokens)
        hyp_len = len(hyp_tokens)
        
        if hyp_len >= ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - ref_len / hyp_len) if hyp_len > 0 else 0.0
        
        bleu = bp * geo_mean
        return min(1.0, bleu)  # Ensure it doesn't exceed 1.0
    
    def _get_ngrams(self, tokens: List[str], n: int) -> List[tuple]:
        """Get n-grams from token list
        
        Args:
            tokens: List of tokens
            n: N-gram size
        
        Returns:
            List of n-gram tuples
        """
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngrams.append(tuple(tokens[i:i+n]))
        return ngrams
    
    def _log_failure_case(
        self,
        original: str,
        reconstructed: str,
        metrics: QualityMetrics
    ):
        """Log failure case for optimization
        
        Args:
            original: Original text
            reconstructed: Reconstructed text
            metrics: Quality metrics
        
        Requirements: 7.7
        """
        try:
            failure_case = {
                "timestamp": time.time(),
                "original": original,
                "reconstructed": reconstructed,
                "metrics": {
                    "compression_ratio": metrics.compression_ratio,
                    "semantic_similarity": metrics.semantic_similarity,
                    "entity_accuracy": metrics.entity_accuracy,
                    "bleu_score": metrics.bleu_score,
                    "overall_score": metrics.overall_score,
                },
                "warnings": metrics.warnings
            }
            
            # Append to JSONL file
            with open(self.failure_log_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(failure_case, ensure_ascii=False) + '\n')
            
            logger.debug(f"Logged failure case to {self.failure_log_path}")
        except Exception as e:
            logger.error(f"Failed to log failure case: {e}")
    
    def generate_report(self, metrics: QualityMetrics) -> str:
        """Generate quality report
        
        Args:
            metrics: Quality metrics
        
        Returns:
            str: Formatted quality report
        
        Requirements: 7.2
        """
        report = f"""
Quality Evaluation Report
========================

Compression Metrics:
  - Compression Ratio: {metrics.compression_ratio:.2f}x
  - Reconstruction Latency: {metrics.reconstruction_latency_ms:.2f}ms

Quality Metrics:
  - Semantic Similarity: {metrics.semantic_similarity:.3f}
  - Entity Accuracy: {metrics.entity_accuracy:.3f}
  - BLEU Score: {metrics.bleu_score:.3f}
  - Overall Score: {metrics.overall_score:.3f}

Status: {"✓ PASS" if not metrics.warnings else "✗ FAIL"}
"""
        
        if metrics.warnings:
            report += "\nWarnings:\n"
            for warning in metrics.warnings:
                report += f"  - {warning}\n"
        
        return report.strip()
    
    def evaluate_batch(
        self,
        originals: List[str],
        reconstructed_list: List[str],
        compressed_sizes: List[int],
        reconstruction_latencies: List[float]
    ) -> List[QualityMetrics]:
        """Evaluate batch of compressions
        
        Args:
            originals: List of original texts
            reconstructed_list: List of reconstructed texts
            compressed_sizes: List of compressed sizes
            reconstruction_latencies: List of reconstruction latencies
        
        Returns:
            List[QualityMetrics]: List of quality metrics
        
        Requirements: 7.5
        """
        logger.info(f"Evaluating batch of {len(originals)} compressions")
        
        metrics_list = []
        for i, (original, reconstructed, size, latency) in enumerate(
            zip(originals, reconstructed_list, compressed_sizes, reconstruction_latencies)
        ):
            try:
                metrics = self.evaluate(original, reconstructed, size, latency)
                metrics_list.append(metrics)
            except Exception as e:
                logger.error(f"Failed to evaluate item {i}: {e}")
                # Create a failed metrics object
                metrics_list.append(QualityMetrics(
                    compression_ratio=0.0,
                    semantic_similarity=0.0,
                    entity_accuracy=0.0,
                    bleu_score=0.0,
                    reconstruction_latency_ms=latency,
                    overall_score=0.0,
                    warnings=[f"Evaluation failed: {str(e)}"]
                ))
        
        logger.info(f"Batch evaluation complete: {len(metrics_list)} results")
        return metrics_list
