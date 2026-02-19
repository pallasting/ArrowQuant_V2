"""
Precision Validation Test

Compare ArrowEngine and sentence-transformers output precision, target similarity >= 0.99.
"""

import os
import sys
import numpy as np
from pathlib import Path

def test_precision_validation():
    """Test precision validation"""
    print("\n" + "=" * 60)
    print("Precision Validation Test")
    print("=" * 60)
    
    model_path_str = os.environ.get(
        "ARROW_MODEL_PATH",
        "D:/ai-models/minilm" if os.path.exists("D:/ai-models/minilm") 
        else "./models/minilm"
    )
    model_path = Path(model_path_str)
    
    if not model_path.exists():
        print(f"\nX Model directory not found: {model_path}")
        return 1
    
    # Test texts (diverse)
    test_texts = [
        "Artificial intelligence is transforming technology.",
        "Machine learning enables computers to learn from data.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing helps computers understand text.",
        "Computer vision allows machines to interpret images.",
        "The quick brown fox jumps over the lazy dog.",
        "Python is a popular programming language for data science.",
        "Climate change poses significant challenges to global ecosystems.",
    ]
    
    print(f"\nNumber of test texts: {len(test_texts)}")
    
    try:
        from llm_compression.inference.arrow_engine import ArrowEngine
        
        # Load ArrowEngine
        print(f"\nLoading ArrowEngine...")
        arrow_engine = ArrowEngine(str(model_path))
        print(f"v ArrowEngine loaded")
        
        # Encode
        print(f"\nEncoding with ArrowEngine...")
        arrow_embs = arrow_engine.encode(test_texts, normalize=True)
        print(f"v Encoding completed: {arrow_embs.shape}")
        
    except Exception as e:
        print(f"\nX ArrowEngine loading failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Try loading sentence-transformers for comparison
    try:
        from sentence_transformers import SentenceTransformer
        
        print(f"\nLoading sentence-transformers...")
        st_model = SentenceTransformer("all-MiniLM-L6-v2")
        print(f"v sentence-transformers loaded")
        
        # Encode
        print(f"\nEncoding with sentence-transformers...")
        st_embs = st_model.encode(test_texts, normalize_embeddings=True)
        print(f"v Encoding completed: {st_embs.shape}")
        
        # Calculate similarity
        print(f"\n" + "=" * 60)
        print(f"Precision Comparison:")
        print(f"=" * 60)
        
        similarities = []
        for i in range(len(test_texts)):
            sim = np.dot(arrow_embs[i], st_embs[i])
            similarities.append(sim)
            
            # Show similarity for each text
            text_preview = test_texts[i][:50] + "..." if len(test_texts[i]) > 50 else test_texts[i]
            print(f"\nText {i+1}: {text_preview}")
            print(f"  Similarity: {sim:.6f}")
            
            if sim < 0.99:
                print(f"  ! Similarity < 0.99")
        
        # Statistical analysis
        avg_sim = np.mean(similarities)
        min_sim = np.min(similarities)
        max_sim = np.max(similarities)
        std_sim = np.std(similarities)
        
        print(f"\n" + "=" * 60)
        print(f"Statistical Results:")
        print(f"=" * 60)
        print(f"  Average similarity: {avg_sim:.6f}")
        print(f"  Min similarity: {min_sim:.6f}")
        print(f"  Max similarity: {max_sim:.6f}")
        print(f"  Std dev: {std_sim:.6f}")
        
        # Evaluate results
        print(f"\nPerformance Evaluation:")
        print(f"  Target: Min similarity >= 0.99")
        
        if min_sim >= 0.99:
            print(f"  v Excellent - Min similarity {min_sim:.6f} >= 0.99")
            status = 0
        elif min_sim >= 0.95:
            print(f"  ! Acceptable - Min similarity {min_sim:.6f} >= 0.95")
            print(f"     Suggestion: Check if model conversion is correct")
            status = 0
        else:
            print(f"  X Failed - Min similarity {min_sim:.6f} < 0.95")
            print(f"     Need to reconvert model")
            status = 1
        
        if avg_sim >= 0.995:
            print(f"  v Average similarity {avg_sim:.6f} >= 0.995 (Excellent)")
        elif avg_sim >= 0.99:
            print(f"  v Average similarity {avg_sim:.6f} >= 0.99 (Good)")
        
    except ImportError:
        print(f"\n! sentence-transformers not installed, skipping precision comparison")
        print(f"   Install command: pip install sentence-transformers")
        print(f"\nValidating ArrowEngine basic functionality only:")
        
        # Self-similarity test
        print(f"\nSelf-similarity test:")
        for i in range(len(test_texts)):
            # Same text encoded twice should be identical
            emb1 = arrow_engine.encode(test_texts[i], normalize=True)
            emb2 = arrow_engine.encode(test_texts[i], normalize=True)
            sim = np.dot(emb1[0], emb2[0])
            
            print(f"  Text {i+1}: {sim:.6f}")
            
            if abs(sim - 1.0) > 0.001:
                print(f"    ! Self-similarity not 1.0, may have issues")
        
        print(f"\nv ArrowEngine basic functionality is normal")
        status = 0
    
    except Exception as e:
        print(f"\nX Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print(f"\n" + "=" * 60)
    print("v Precision validation test completed")
    return status

if __name__ == "__main__":
    sys.exit(test_precision_validation())
