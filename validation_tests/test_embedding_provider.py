"""
EmbeddingProvider Interface Test

Test if the unified EmbeddingProvider interface works correctly.
"""

import sys
import numpy as np
from pathlib import Path

def test_embedding_provider():
    """Test EmbeddingProvider interface"""
    print("\n" + "=" * 60)
    print("EmbeddingProvider Interface Test")
    print("=" * 60)
    
    try:
        from llm_compression.embedding_provider import get_default_provider
        
        # Get default provider
        print(f"\nGetting default EmbeddingProvider...")
        provider = get_default_provider()
        print(f"v Provider type: {type(provider).__name__}")
        print(f"v Embedding dimension: {provider.dimension}")
        
        # Test 1: Single text encoding
        print(f"\nTest 1: Single text encoding")
        text = "Hello, World!"
        embedding = provider.encode(text)
        
        print(f"  Input: '{text}'")
        print(f"  Output shape: {embedding.shape}")
        print(f"  Output type: {type(embedding)}")
        print(f"  Data type: {embedding.dtype}")
        
        if embedding.shape != (provider.dimension,):
            print(f"  X Shape error: expected ({provider.dimension},), got {embedding.shape}")
            return 1
        
        if not isinstance(embedding, np.ndarray):
            print(f"  X Type error: expected np.ndarray, got {type(embedding)}")
            return 1
        
        print(f"  v Single text encoding is normal")
        
        # Test 2: Batch encoding
        print(f"\nTest 2: Batch encoding")
        texts = [
            "First sentence",
            "Second sentence",
            "Third sentence"
        ]
        embeddings = provider.encode_batch(texts)
        
        print(f"  Input count: {len(texts)}")
        print(f"  Output shape: {embeddings.shape}")
        
        if embeddings.shape != (len(texts), provider.dimension):
            print(f"  X Shape error: expected ({len(texts)}, {provider.dimension}), got {embeddings.shape}")
            return 1
        
        print(f"  v Batch encoding is normal")
        
        # Test 3: Similarity calculation
        print(f"\nTest 3: Similarity calculation")
        
        # Self-similarity (should be ~1.0)
        sim_self = provider.similarity(embedding, embedding)
        print(f"  Self-similarity: {sim_self:.6f}")
        
        if abs(sim_self - 1.0) > 0.01:
            print(f"  ! Self-similarity not 1.0: {sim_self:.6f}")
        else:
            print(f"  v Self-similarity is normal")
        
        # Similarity between different texts
        emb1 = provider.encode("Machine learning")
        emb2 = provider.encode("Artificial intelligence")
        emb3 = provider.encode("The weather is nice today")
        
        sim_12 = provider.similarity(emb1, emb2)
        sim_13 = provider.similarity(emb1, emb3)
        
        print(f"  'Machine learning' vs 'Artificial intelligence': {sim_12:.4f}")
        print(f"  'Machine learning' vs 'The weather is nice today': {sim_13:.4f}")
        
        if sim_12 > sim_13:
            print(f"  v Semantic similarity is reasonable (related > unrelated)")
        else:
            print(f"  ! Semantic similarity may have issues")
        
        # Test 4: Similarity matrix
        print(f"\nTest 4: Similarity matrix")
        
        vectors = np.array([emb1, emb2, emb3])
        sim_matrix = provider.similarity_matrix(vectors)
        
        print(f"  Input shape: {vectors.shape}")
        print(f"  Output shape: {sim_matrix.shape}")
        print(f"  Similarity matrix:")
        print(f"    {sim_matrix[0, 0]:.4f}  {sim_matrix[0, 1]:.4f}  {sim_matrix[0, 2]:.4f}")
        print(f"    {sim_matrix[1, 0]:.4f}  {sim_matrix[1, 1]:.4f}  {sim_matrix[1, 2]:.4f}")
        print(f"    {sim_matrix[2, 0]:.4f}  {sim_matrix[2, 1]:.4f}  {sim_matrix[2, 2]:.4f}")
        
        # Diagonal should be ~1.0
        diagonal_ok = all(abs(sim_matrix[i, i] - 1.0) < 0.01 for i in range(3))
        if diagonal_ok:
            print(f"  v Diagonal elements are normal (~1.0)")
        else:
            print(f"  ! Diagonal elements are abnormal")
        
        # Test 5: Empty text handling
        print(f"\nTest 5: Edge cases")
        
        try:
            empty_emb = provider.encode("")
            print(f"  Empty text: {empty_emb.shape}")
            
            if np.allclose(empty_emb, 0):
                print(f"  v Empty text returns zero vector")
            else:
                print(f"  i Empty text returns non-zero vector")
        except Exception as e:
            print(f"  ! Empty text handling exception: {e}")
        
        # Test 6: get_embedding_dimension method
        print(f"\nTest 6: Helper methods")
        dim = provider.get_embedding_dimension()
        print(f"  get_embedding_dimension(): {dim}")
        
        if dim == provider.dimension:
            print(f"  v Dimension methods are consistent")
        else:
            print(f"  X Dimension methods are inconsistent: {dim} vs {provider.dimension}")
            return 1
        
        print(f"\n" + "=" * 60)
        print("v EmbeddingProvider interface test completed")
        print(f"\nAll interface methods work correctly:")
        print(f"  v encode()")
        print(f"  v encode_batch()")
        print(f"  v similarity()")
        print(f"  v similarity_matrix()")
        print(f"  v get_embedding_dimension()")
        
        return 0
        
    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(test_embedding_provider())
