"""
Quality Evaluator Example

Demonstrates how to use the QualityEvaluator to assess compression quality.
"""

from llm_compression import QualityEvaluator, QualityMetrics


def main():
    """Demonstrate QualityEvaluator usage"""
    
    print("=" * 70)
    print("Quality Evaluator Example")
    print("=" * 70)
    
    # Initialize evaluator
    print("\n1. Initializing QualityEvaluator...")
    evaluator = QualityEvaluator(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        semantic_threshold=0.85,
        entity_threshold=0.95,
        failure_log_path="./quality_failures.jsonl"
    )
    print("✓ Evaluator initialized")
    
    # Example 1: High quality reconstruction
    print("\n2. Evaluating high quality reconstruction...")
    original_text = """
    John Smith met Mary Johnson on 2024-01-15 at 3pm to discuss the $1,000 budget 
    for the new project. They agreed to allocate 50% to development and 30% to marketing.
    The meeting took place at Google headquarters in Mountain View.
    """
    
    reconstructed_text = """
    John Smith met Mary Johnson on 2024-01-15 at 3pm to discuss the $1,000 budget 
    for the new project. They agreed to allocate 50% to development and 30% to marketing.
    The meeting took place at Google headquarters in Mountain View.
    """
    
    compressed_size = 100  # bytes
    reconstruction_latency = 250.0  # ms
    
    metrics = evaluator.evaluate(
        original_text,
        reconstructed_text,
        compressed_size,
        reconstruction_latency
    )
    
    print("\nQuality Metrics:")
    print(f"  Compression Ratio: {metrics.compression_ratio:.2f}x")
    print(f"  Semantic Similarity: {metrics.semantic_similarity:.3f}")
    print(f"  Entity Accuracy: {metrics.entity_accuracy:.3f}")
    print(f"  BLEU Score: {metrics.bleu_score:.3f}")
    print(f"  Overall Score: {metrics.overall_score:.3f}")
    print(f"  Warnings: {len(metrics.warnings)}")
    
    # Generate report
    print("\n" + evaluator.generate_report(metrics))
    
    # Example 2: Low quality reconstruction
    print("\n" + "=" * 70)
    print("3. Evaluating low quality reconstruction...")
    
    poor_reconstruction = """
    Someone had a meeting about money. They talked about a project.
    """
    
    metrics2 = evaluator.evaluate(
        original_text,
        poor_reconstruction,
        compressed_size,
        reconstruction_latency
    )
    
    print("\nQuality Metrics:")
    print(f"  Compression Ratio: {metrics2.compression_ratio:.2f}x")
    print(f"  Semantic Similarity: {metrics2.semantic_similarity:.3f}")
    print(f"  Entity Accuracy: {metrics2.entity_accuracy:.3f}")
    print(f"  BLEU Score: {metrics2.bleu_score:.3f}")
    print(f"  Overall Score: {metrics2.overall_score:.3f}")
    print(f"  Warnings: {len(metrics2.warnings)}")
    
    if metrics2.warnings:
        print("\nWarnings:")
        for warning in metrics2.warnings:
            print(f"  ⚠ {warning}")
    
    # Generate report
    print("\n" + evaluator.generate_report(metrics2))
    
    # Example 3: Entity extraction
    print("\n" + "=" * 70)
    print("4. Demonstrating entity extraction...")
    
    sample_text = """
    Dr. Alice Brown and Professor Bob Wilson met on January 15, 2024 at 2:30pm.
    They discussed the $50,000 grant and the 75% completion rate of the research.
    The meeting was held at Stanford University in California.
    """
    
    entities = evaluator._extract_entities(sample_text)
    
    print("\nExtracted Entities:")
    for entity_type, entity_list in entities.items():
        if entity_list:
            print(f"  {entity_type.capitalize()}: {', '.join(entity_list)}")
    
    # Example 4: Batch evaluation
    print("\n" + "=" * 70)
    print("5. Demonstrating batch evaluation...")
    
    originals = [
        "John met Mary on 2024-01-15.",
        "The budget is $1,000.",
        "Alice works at Google in California."
    ]
    
    reconstructed_list = [
        "John met Mary on 2024-01-15.",
        "The budget is $1,000.",
        "Alice works at Google in California."
    ]
    
    compressed_sizes = [20, 15, 25]
    latencies = [100.0, 90.0, 110.0]
    
    batch_metrics = evaluator.evaluate_batch(
        originals,
        reconstructed_list,
        compressed_sizes,
        latencies
    )
    
    print(f"\nEvaluated {len(batch_metrics)} items:")
    for i, metrics in enumerate(batch_metrics):
        print(f"\n  Item {i+1}:")
        print(f"    Compression Ratio: {metrics.compression_ratio:.2f}x")
        print(f"    Semantic Similarity: {metrics.semantic_similarity:.3f}")
        print(f"    Overall Score: {metrics.overall_score:.3f}")
    
    # Calculate average metrics
    avg_compression = sum(m.compression_ratio for m in batch_metrics) / len(batch_metrics)
    avg_similarity = sum(m.semantic_similarity for m in batch_metrics) / len(batch_metrics)
    avg_overall = sum(m.overall_score for m in batch_metrics) / len(batch_metrics)
    
    print(f"\n  Average Metrics:")
    print(f"    Compression Ratio: {avg_compression:.2f}x")
    print(f"    Semantic Similarity: {avg_similarity:.3f}")
    print(f"    Overall Score: {avg_overall:.3f}")
    
    print("\n" + "=" * 70)
    print("✓ Quality evaluation examples completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
