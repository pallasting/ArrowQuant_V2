"""
Comprehensive compression benchmark

Tests compression performance across different scenarios:
1. Short conversations
2. Long documents
3. Structured data
4. Mixed content
"""
import sys
from pathlib import Path
import time
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from generative import CompressionBenchmark
from memory_core import MemoryStore


def generate_test_data():
    """Generate diverse test data"""
    
    return {
        'short_conversations': [
            "Quick sync with Tom about the bug fix.",
            "Lunch with Maria at noon.",
            "Called customer support about billing issue.",
        ],
        
        'long_documents': [
            """
            Project Status Report - Q1 2026
            
            Executive Summary:
            The AI-OS Memory project has made significant progress in Q1. We've completed
            the initial architecture design and validated core compression hypotheses.
            
            Key Achievements:
            - Designed generative memory system with 100x+ compression target
            - Implemented Arrow-based storage engine
            - Created scene replay prototype
            - Established privacy-layered architecture
            
            Challenges:
            - LLM API costs for reconstruction
            - Latency optimization needed
            - Privacy compliance requirements
            
            Next Steps:
            - Complete Phase 0 validation
            - Begin Rust implementation
            - Design OpenClaw integration
            
            Team: 5 engineers, 2 researchers
            Budget: On track
            Timeline: Slightly ahead of schedule
            """,
        ],
        
        'structured_data': [
            """
            Meeting Notes - 2026-02-13
            
            Attendees: Alice, Bob, Carol, Dave
            Duration: 60 minutes
            Location: Conference Room B
            
            Agenda:
            1. Review Q1 metrics
            2. Discuss hiring plan
            3. Plan Q2 roadmap
            
            Action Items:
            - Alice: Prepare hiring JDs by Friday
            - Bob: Schedule interviews next week
            - Carol: Update roadmap doc
            - Dave: Send meeting summary
            
            Decisions:
            - Approved budget increase of 20%
            - Agreed to hire 3 engineers
            - Moved feature X to Q3
            """,
        ],
        
        'mixed_content': [
            """
            Today was productive. Started at 9am with standup where Mike mentioned
            the authentication bug is fixed (PR #456). Then had 1:1 with Sarah - she's
            interested in the ML team. Lunch at "Joe's Diner" with the crew, got the
            burger special ($12.99). Afternoon: reviewed 3 PRs, wrote 200 lines of code,
            fixed 2 bugs. Team happy hour at 5pm, discussed weekend plans. Home by 7pm.
            """,
        ],
    }


def run_benchmark():
    """Run comprehensive benchmark"""
    
    print("="*80)
    print("AI-OS MEMORY COMPRESSION BENCHMARK")
    print("="*80)
    print()
    
    test_data = generate_test_data()
    benchmark = CompressionBenchmark()
    
    all_results = {}
    
    for category, data in test_data.items():
        print(f"\n{'─'*80}")
        print(f"Category: {category.upper().replace('_', ' ')}")
        print(f"{'─'*80}")
        
        start_time = time.time()
        results = benchmark.run(data)
        elapsed = time.time() - start_time
        
        all_results[category] = results
        
        print(f"\nTests: {results['total_tests']}")
        print(f"Avg Compression: {results['avg_compression_ratio']:.2f}x")
        print(f"Avg Quality: {results['avg_quality_score']:.2%}")
        print(f"Time: {elapsed:.3f}s")
        
        # Detailed breakdown
        for i, detail in enumerate(results['details']):
            print(f"\n  Test {i+1}:")
            print(f"    Original: {detail['original_size']:,} bytes")
            print(f"    Compressed: {detail['compressed_size']:,} bytes")
            print(f"    Ratio: {detail['ratio']:.2f}x")
            print(f"    Quality: {detail['quality_score']:.2%}")
    
    # Overall summary
    print(f"\n{'='*80}")
    print("OVERALL SUMMARY")
    print(f"{'='*80}")
    
    total_tests = sum(r['total_tests'] for r in all_results.values())
    avg_ratio = sum(
        r['avg_compression_ratio'] * r['total_tests'] 
        for r in all_results.values()
    ) / total_tests
    avg_quality = sum(
        r['avg_quality_score'] * r['total_tests']
        for r in all_results.values()
    ) / total_tests
    
    print(f"\nTotal Tests: {total_tests}")
    print(f"Overall Avg Compression: {avg_ratio:.2f}x")
    print(f"Overall Avg Quality: {avg_quality:.2%}")
    
    # Assessment
    print(f"\n{'='*80}")
    print("ASSESSMENT")
    print(f"{'='*80}")
    
    if avg_ratio >= 100:
        status = "✅ EXCELLENT - Target achieved!"
    elif avg_ratio >= 50:
        status = "✓ GOOD - Close to target"
    elif avg_ratio >= 10:
        status = "⚠ MODERATE - Needs improvement"
    else:
        status = "❌ POOR - Significant work needed"
    
    print(f"\nCompression: {status}")
    
    if avg_quality >= 0.9:
        status = "✅ EXCELLENT - High fidelity"
    elif avg_quality >= 0.7:
        status = "✓ GOOD - Acceptable quality"
    elif avg_quality >= 0.5:
        status = "⚠ MODERATE - Some loss"
    else:
        status = "❌ POOR - Too much loss"
    
    print(f"Quality: {status}")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    
    if avg_ratio < 100:
        print("\n1. Integrate real LLM for compression")
        print("   - Use Claude/GPT for summarization")
        print("   - Implement proper diff extraction")
    
    if avg_quality < 0.9:
        print("\n2. Improve reconstruction quality")
        print("   - Better prompt engineering")
        print("   - Store more critical details in diff")
    
    print("\n3. Add vector search")
    print("   - Integrate HNSW for semantic retrieval")
    print("   - Test retrieval latency")
    
    print("\n4. Test with real data")
    print("   - Use actual conversation logs")
    print("   - Measure user satisfaction")
    
    print(f"\n{'='*80}")
    
    # Save results
    output_file = Path(__file__).parent.parent / "data" / "benchmark_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': all_results,
            'summary': {
                'total_tests': total_tests,
                'avg_compression_ratio': avg_ratio,
                'avg_quality_score': avg_quality,
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    print()


if __name__ == '__main__':
    run_benchmark()
