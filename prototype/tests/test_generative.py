"""
Test generative memory compression
"""
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from generative import GenerativeMemory, CompressionBenchmark


@pytest.fixture
def generative():
    """Create generative memory instance"""
    return GenerativeMemory()


def test_basic_compression(generative):
    """Test basic compression"""
    
    experiences = [
        "Today I met with John at 3pm to discuss the AI-OS project.",
        "We talked about memory compression using LLMs.",
        "John suggested using Arrow format for storage.",
    ]
    
    compressed = generative.compress(experiences)
    
    assert 'summary' in compressed
    assert 'diff' in compressed
    assert 'metadata' in compressed
    
    # Check compression ratio
    ratio = compressed['metadata']['compression_ratio']
    print(f"\nCompression ratio: {ratio:.2f}x")
    
    assert ratio > 1.0  # Should achieve some compression


def test_reconstruction(generative):
    """Test memory reconstruction"""
    
    experiences = [
        "Had lunch at Mario's Pizza at 12:30pm.",
        "Ordered the Margherita pizza and a Coke.",
        "The waiter's name was Tony.",
    ]
    
    compressed = generative.compress(experiences)
    reconstructed = generative.reconstruct(compressed)
    
    print(f"\nOriginal: {experiences}")
    print(f"Compressed: {compressed}")
    print(f"Reconstructed: {reconstructed}")
    
    # Check that key details are preserved
    assert "pizza" in reconstructed.lower()
    assert "12:30" in reconstructed or "Tony" in reconstructed


def test_compression_benchmark():
    """Test compression benchmark"""
    
    test_data = [
        "Met with Sarah at Starbucks to discuss the quarterly report. She mentioned sales are up 15% and we need to hire 3 new engineers.",
        "Attended the team standup at 9am. Mike reported he finished the authentication module. Lisa is working on the dashboard.",
        "Reviewed pull request #234 from Alex. The code looks good but needs more tests. Left 5 comments for improvement.",
    ]
    
    benchmark = CompressionBenchmark()
    results = benchmark.run(test_data)
    
    print(f"\n{'='*60}")
    print("COMPRESSION BENCHMARK RESULTS")
    print(f"{'='*60}")
    print(f"Total tests: {results['total_tests']}")
    print(f"Avg compression ratio: {results['avg_compression_ratio']:.2f}x")
    print(f"Avg quality score: {results['avg_quality_score']:.2%}")
    print(f"{'='*60}")
    
    for detail in results['details']:
        print(f"\nTest {detail['test_id']}:")
        print(f"  Original: {detail['original_size']} bytes")
        print(f"  Compressed: {detail['compressed_size']} bytes")
        print(f"  Ratio: {detail['ratio']:.2f}x")
        print(f"  Quality: {detail['quality_score']:.2%}")
    
    # Should achieve reasonable compression
    assert results['avg_compression_ratio'] > 1.5
    
    # Should maintain reasonable quality
    assert results['avg_quality_score'] > 0.5


def test_large_dataset():
    """Test with larger dataset"""
    
    # Generate synthetic data
    test_data = []
    for i in range(20):
        test_data.append(
            f"Day {i}: Worked on task #{i*10} with team member Person{i}. "
            f"Completed {i*2} items and scheduled {i+1} meetings. "
            f"Key decision: Approved budget of ${i*1000}."
        )
    
    benchmark = CompressionBenchmark()
    results = benchmark.run(test_data)
    
    print(f"\n{'='*60}")
    print("LARGE DATASET BENCHMARK")
    print(f"{'='*60}")
    print(f"Dataset size: {len(test_data)} experiences")
    print(f"Avg compression: {results['avg_compression_ratio']:.2f}x")
    print(f"Avg quality: {results['avg_quality_score']:.2%}")
    
    total_original = sum(d['original_size'] for d in results['details'])
    total_compressed = sum(d['compressed_size'] for d in results['details'])
    
    print(f"\nTotal original: {total_original:,} bytes ({total_original/1024:.1f} KB)")
    print(f"Total compressed: {total_compressed:,} bytes ({total_compressed/1024:.1f} KB)")
    print(f"Overall ratio: {total_original/total_compressed:.2f}x")
    print(f"{'='*60}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
