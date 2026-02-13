"""
Generative Memory: LLM-based compression and reconstruction

Core hypothesis: LLM parameters contain consensus knowledge.
We only store the "diff" - what makes this memory unique.
"""
from typing import List, Dict, Any, Optional
import json
from datetime import datetime


class GenerativeMemory:
    """
    Generative memory compression using LLM
    
    Storage strategy:
    1. Consensus knowledge → 0 bytes (in model)
    2. Private memory → Store diff only
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
    
    def compress(
        self,
        experiences: List[str],
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compress experiences into generative memory
        
        Returns:
            {
                'summary': str,      # High-level summary
                'diff': List[str],   # Unique details
                'metadata': dict,    # Compression stats
            }
        """
        
        # Calculate original size
        original_text = "\n".join(experiences)
        original_size = len(original_text.encode('utf-8'))
        
        # For now, simple compression (will add LLM later)
        summary = self._extract_summary(experiences)
        diff = self._extract_diff(experiences, summary)
        
        compressed_size = len(json.dumps({
            'summary': summary,
            'diff': diff
        }).encode('utf-8'))
        
        compression_ratio = original_size / compressed_size if compressed_size > 0 else 0
        
        return {
            'summary': summary,
            'diff': diff,
            'metadata': {
                'original_size': original_size,
                'compressed_size': compressed_size,
                'compression_ratio': compression_ratio,
                'timestamp': datetime.now().isoformat(),
            }
        }
    
    def reconstruct(
        self,
        compressed: Dict[str, Any],
        query: Optional[str] = None
    ) -> str:
        """
        Reconstruct memory from compressed form
        
        Uses LLM to regenerate full experience from summary + diff
        """
        
        summary = compressed['summary']
        diff = compressed['diff']
        
        # Simple reconstruction (will add LLM later)
        reconstructed = f"{summary}\n\nKey details:\n"
        reconstructed += "\n".join(f"- {d}" for d in diff)
        
        return reconstructed
    
    def _extract_summary(self, experiences: List[str]) -> str:
        """Extract high-level summary"""
        
        # Simple heuristic: first sentence of each experience
        summaries = []
        for exp in experiences:
            sentences = exp.split('.')
            if sentences:
                summaries.append(sentences[0].strip())
        
        return ". ".join(summaries[:3]) + "."
    
    def _extract_diff(
        self,
        experiences: List[str],
        summary: str
    ) -> List[str]:
        """Extract unique details not in summary"""
        
        # Simple heuristic: extract specific names, numbers, dates
        diff = []
        
        for exp in experiences:
            # Extract quoted text
            if '"' in exp:
                parts = exp.split('"')
                for i in range(1, len(parts), 2):
                    diff.append(f'Quote: "{parts[i]}"')
            
            # Extract numbers
            words = exp.split()
            for word in words:
                if any(c.isdigit() for c in word):
                    diff.append(f'Data: {word}')
        
        return diff[:10]  # Limit to top 10 details


class CompressionBenchmark:
    """Benchmark compression performance"""
    
    def __init__(self):
        self.generative = GenerativeMemory()
    
    def run(self, test_data: List[str]) -> Dict[str, Any]:
        """Run compression benchmark"""
        
        results = []
        
        for i, data in enumerate(test_data):
            experiences = [data]  # Single experience for now
            
            compressed = self.generative.compress(experiences)
            reconstructed = self.generative.reconstruct(compressed)
            
            results.append({
                'test_id': i,
                'original_size': compressed['metadata']['original_size'],
                'compressed_size': compressed['metadata']['compressed_size'],
                'ratio': compressed['metadata']['compression_ratio'],
                'quality_score': self._calculate_quality(data, reconstructed),
            })
        
        # Aggregate stats
        avg_ratio = sum(r['ratio'] for r in results) / len(results)
        avg_quality = sum(r['quality_score'] for r in results) / len(results)
        
        return {
            'total_tests': len(results),
            'avg_compression_ratio': avg_ratio,
            'avg_quality_score': avg_quality,
            'details': results,
        }
    
    def _calculate_quality(self, original: str, reconstructed: str) -> float:
        """Calculate reconstruction quality (0-1)"""
        
        # Simple metric: word overlap
        original_words = set(original.lower().split())
        reconstructed_words = set(reconstructed.lower().split())
        
        if not original_words:
            return 0.0
        
        overlap = len(original_words & reconstructed_words)
        return overlap / len(original_words)
