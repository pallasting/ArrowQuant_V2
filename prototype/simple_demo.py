"""
Simplified compression demo (no external dependencies)

Demonstrates core compression concepts without requiring PyArrow/Polars
"""
import json
import time
from typing import List, Dict, Any
from pathlib import Path


class SimpleMemoryStore:
    """Simple JSON-based memory storage (for demo)"""
    
    def __init__(self, storage_path: str = "data/memories"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memories_file = self.storage_path / "memories.json"
        self.memories = self._load()
    
    def _load(self) -> List[Dict]:
        if self.memories_file.exists():
            with open(self.memories_file) as f:
                return json.load(f)
        return []
    
    def _save(self):
        with open(self.memories_file, 'w') as f:
            json.dump(self.memories, f, indent=2)
    
    def store(self, memory: Dict[str, Any]):
        self.memories.append(memory)
        self._save()
    
    def retrieve(self, limit: int = 100) -> List[Dict]:
        return self.memories[-limit:]
    
    def get_size(self) -> int:
        if self.memories_file.exists():
            return self.memories_file.stat().st_size
        return 0


class SimpleCompressor:
    """Simple compression without LLM (for demo)"""
    
    def compress(self, text: str) -> Dict[str, Any]:
        """
        Simple compression strategy:
        1. Extract key entities (names, numbers, dates)
        2. Create summary (first + last sentence)
        3. Store diff (unique details)
        """
        
        original_size = len(text.encode('utf-8'))
        
        # Extract sentences
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Summary: first and last sentence
        if len(sentences) >= 2:
            summary = f"{sentences[0]}. {sentences[-1]}."
        elif sentences:
            summary = sentences[0] + "."
        else:
            summary = text[:100]
        
        # Extract entities
        entities = self._extract_entities(text)
        
        # Compressed form
        compressed = {
            'summary': summary,
            'entities': entities,
            'sentence_count': len(sentences),
        }
        
        compressed_size = len(json.dumps(compressed).encode('utf-8'))
        
        return {
            'compressed': compressed,
            'original_size': original_size,
            'compressed_size': compressed_size,
            'ratio': original_size / compressed_size if compressed_size > 0 else 0,
        }
    
    def _extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract key entities"""
        
        words = text.split()
        entities = {
            'numbers': [],
            'capitalized': [],
            'quoted': [],
        }
        
        # Extract numbers
        for word in words:
            if any(c.isdigit() for c in word):
                entities['numbers'].append(word)
        
        # Extract capitalized words (potential names)
        for word in words:
            if word and word[0].isupper() and len(word) > 1:
                entities['capitalized'].append(word)
        
        # Extract quoted text
        if '"' in text:
            parts = text.split('"')
            for i in range(1, len(parts), 2):
                entities['quoted'].append(parts[i])
        
        # Deduplicate and limit
        for key in entities:
            entities[key] = list(set(entities[key]))[:10]
        
        return entities
    
    def reconstruct(self, compressed: Dict) -> str:
        """Reconstruct from compressed form"""
        
        summary = compressed['summary']
        entities = compressed['entities']
        
        # Build reconstruction
        parts = [summary]
        
        if entities.get('numbers'):
            parts.append(f"Key data: {', '.join(entities['numbers'])}")
        
        if entities.get('capitalized'):
            parts.append(f"Mentioned: {', '.join(entities['capitalized'][:5])}")
        
        if entities.get('quoted'):
            parts.append(f"Quotes: {', '.join(entities['quoted'][:3])}")
        
        return "\n".join(parts)


def run_demo():
    """Run compression demo"""
    
    print("="*80)
    print("AI-OS MEMORY COMPRESSION DEMO")
    print("="*80)
    print()
    
    # Test data
    test_cases = [
        {
            'name': 'Short conversation',
            'text': 'Met with John at 3pm to discuss the AI-OS project. He suggested using Arrow format for better performance.',
        },
        {
            'name': 'Meeting notes',
            'text': '''
            Team meeting on 2026-02-13 with Alice, Bob, and Carol.
            Discussed Q1 metrics showing 25% growth.
            Decided to hire 3 engineers and increase budget by $50,000.
            Action items: Alice prepares JDs, Bob schedules interviews.
            Next meeting scheduled for 2026-02-20 at 10am.
            ''',
        },
        {
            'name': 'Long document',
            'text': '''
            Project Status Report - Q1 2026
            
            The AI-OS Memory project has made significant progress.
            We completed the architecture design with 100x compression target.
            Team size grew from 3 to 5 engineers.
            Budget utilization at 85% of allocated $200,000.
            
            Key achievements include generative memory system,
            Arrow-based storage engine, and scene replay prototype.
            
            Challenges: LLM API costs averaging $500/month,
            latency optimization needed for sub-100ms retrieval,
            and privacy compliance requirements.
            
            Next quarter focus: Complete Phase 0 validation,
            begin Rust implementation, and design OpenClaw integration.
            Timeline slightly ahead of schedule.
            ''',
        },
    ]
    
    compressor = SimpleCompressor()
    total_original = 0
    total_compressed = 0
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n{'─'*80}")
        print(f"Test {i}: {case['name']}")
        print(f"{'─'*80}")
        
        text = case['text'].strip()
        result = compressor.compress(text)
        
        total_original += result['original_size']
        total_compressed += result['compressed_size']
        
        print(f"\nOriginal ({result['original_size']} bytes):")
        print(text[:200] + "..." if len(text) > 200 else text)
        
        print(f"\nCompressed ({result['compressed_size']} bytes):")
        print(json.dumps(result['compressed'], indent=2))
        
        print(f"\nReconstructed:")
        reconstructed = compressor.reconstruct(result['compressed'])
        print(reconstructed)
        
        print(f"\n📊 Compression ratio: {result['ratio']:.2f}x")
        print(f"💾 Space saved: {result['original_size'] - result['compressed_size']} bytes "
              f"({(1 - result['compressed_size']/result['original_size'])*100:.1f}%)")
    
    # Overall stats
    print(f"\n{'='*80}")
    print("OVERALL RESULTS")
    print(f"{'='*80}")
    print(f"\nTotal original: {total_original:,} bytes ({total_original/1024:.2f} KB)")
    print(f"Total compressed: {total_compressed:,} bytes ({total_compressed/1024:.2f} KB)")
    print(f"Overall ratio: {total_original/total_compressed:.2f}x")
    print(f"Space saved: {total_original - total_compressed:,} bytes "
          f"({(1 - total_compressed/total_original)*100:.1f}%)")
    
    # Assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}")
    
    ratio = total_original / total_compressed
    
    print(f"\n📈 Current compression: {ratio:.2f}x")
    print(f"🎯 Target compression: 100-1000x")
    
    if ratio >= 100:
        print("✅ Target achieved!")
    elif ratio >= 10:
        print("⚠️  Good progress, but needs LLM integration to reach target")
    else:
        print("❌ Needs significant improvement")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS TO REACH 100x+")
    print(f"{'='*80}")
    
    print("""
1. ✅ Basic compression working (current: ~{:.1f}x)
   
2. 🔄 Integrate real LLM (Claude/GPT)
   - Use model's world knowledge
   - Store only unique "diff"
   - Expected: 10-50x improvement
   
3. 🔄 Add semantic deduplication
   - Identify redundant information
   - Reference existing memories
   - Expected: 2-5x improvement
   
4. 🔄 Implement scene replay for visual memories
   - Store 3D scene params instead of video
   - Expected: 1000x+ for video content
   
5. 🔄 Optimize storage format
   - Use Arrow/Parquet instead of JSON
   - Binary embeddings
   - Expected: 2-3x improvement
   
Combined potential: 10x × 3x × 2x = 60x+ (conservative)
With LLM optimization: 100-1000x achievable
    """.format(ratio))
    
    print(f"{'='*80}\n")


if __name__ == '__main__':
    run_demo()
