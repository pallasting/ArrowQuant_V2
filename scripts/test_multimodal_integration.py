"""
Test Multimodal Integration

Validates that all multimodal components work together:
- MultimodalEmbeddingProvider
- MultimodalStorage
- Text, vision, and audio encoding
- Cross-modal similarity (CLIP)

Usage:
    python scripts/test_multimodal_integration.py
"""

import tempfile
from pathlib import Path

import numpy as np

from llm_compression.multimodal import (
    MultimodalEmbeddingProvider,
    MultimodalStorage,
    get_multimodal_provider,
)
from llm_compression.logger import logger


def test_multimodal_provider():
    """Test MultimodalEmbeddingProvider initialization and encoding."""
    print("\n" + "=" * 60)
    print("Test 1: MultimodalEmbeddingProvider")
    print("=" * 60)
    
    try:
        # Initialize provider
        print("\nInitializing MultimodalEmbeddingProvider...")
        provider = get_multimodal_provider()
        
        print(f"Provider: {provider}")
        print(f"Available modalities: {provider.get_available_modalities()}")
        
        # Test text encoding (backward compatible)
        print("\n--- Text Encoding ---")
        text = "Hello, multimodal world!"
        text_emb = provider.encode(text, normalize=True)
        print(f"Text: '{text}'")
        print(f"Embedding shape: {text_emb.shape}")
        print(f"Embedding norm: {np.linalg.norm(text_emb):.4f}")
        
        # Test batch text encoding
        texts = ["First text", "Second text", "Third text"]
        text_batch_emb = provider.encode_batch(texts, normalize=True)
        print(f"\nBatch text encoding: {len(texts)} texts")
        print(f"Batch embedding shape: {text_batch_emb.shape}")
        
        # Test vision encoding (if available)
        if 'vision' in provider.get_available_modalities():
            print("\n--- Vision Encoding ---")
            # Create synthetic test image
            test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            vision_emb = provider.encode_image(test_image, normalize=True)
            print(f"Image shape: {test_image.shape}")
            print(f"Vision embedding shape: {vision_emb.shape}")
            print(f"Vision embedding norm: {np.linalg.norm(vision_emb):.4f}")
            
            # Test batch vision encoding
            test_images = np.random.randint(0, 255, (3, 224, 224, 3), dtype=np.uint8)
            vision_batch_emb = provider.encode_image(test_images, normalize=True)
            print(f"\nBatch vision encoding: {len(test_images)} images")
            print(f"Batch embedding shape: {vision_batch_emb.shape}")
        else:
            print("\n--- Vision Encoding ---")
            print("Vision encoder not available (model not found)")
        
        # Test audio encoding (if available)
        if 'audio' in provider.get_available_modalities():
            print("\n--- Audio Encoding ---")
            # Create synthetic test audio (3 seconds at 16kHz)
            test_audio = np.random.randn(48000).astype(np.float32)
            audio_emb = provider.encode_audio(test_audio, normalize=True)
            print(f"Audio shape: {test_audio.shape}")
            print(f"Audio embedding shape: {audio_emb.shape}")
            print(f"Audio embedding norm: {np.linalg.norm(audio_emb):.4f}")
            
            # Test batch audio encoding
            test_audios = np.random.randn(3, 48000).astype(np.float32)
            audio_batch_emb = provider.encode_audio(test_audios, normalize=True)
            print(f"\nBatch audio encoding: {len(test_audios)} audio clips")
            print(f"Batch embedding shape: {audio_batch_emb.shape}")
        else:
            print("\n--- Audio Encoding ---")
            print("Audio encoder not available (model not found)")
        
        # Test multimodal encoding
        print("\n--- Multimodal Encoding ---")
        result = provider.encode_multimodal(
            text=["cat", "dog"],
            images=np.random.randint(0, 255, (2, 224, 224, 3), dtype=np.uint8) if 'vision' in provider.get_available_modalities() else None,
            audio=np.random.randn(2, 48000).astype(np.float32) if 'audio' in provider.get_available_modalities() else None,
            normalize=True,
        )
        print(f"Encoded modalities: {list(result.keys())}")
        for modality, emb in result.items():
            print(f"  {modality}: {emb.shape}")
        
        # Test cross-modal similarity (if CLIP available)
        if 'clip' in provider.get_available_modalities() and provider.clip_engine is not None:
            print("\n--- Cross-Modal Similarity (CLIP) ---")
            texts = ["a cat", "a dog", "a car"]
            images = np.random.randint(0, 255, (3, 224, 224, 3), dtype=np.uint8)
            
            similarity = provider.compute_cross_modal_similarity(texts, images)
            print(f"Text queries: {texts}")
            print(f"Number of images: {len(images)}")
            print(f"Similarity matrix shape: {similarity.shape}")
            print(f"Similarity matrix:\n{similarity}")
            
            # Test text-to-image retrieval
            best_matches = provider.find_best_image_matches("a cat", images, top_k=2)
            print(f"\nBest image matches for 'a cat': {best_matches}")
            
            # Test zero-shot classification
            class_names = ["cat", "dog", "car", "tree"]
            probs = provider.zero_shot_classification(images[0], class_names)
            print(f"\nZero-shot classification for image 0:")
            for class_name, prob in probs.items():
                print(f"  {class_name}: {prob:.4f}")
        else:
            print("\n--- Cross-Modal Similarity (CLIP) ---")
            print("CLIP engine not available (requires separate text/vision model structure)")
        
        print("\n" + "=" * 60)
        print("PASSED: MultimodalEmbeddingProvider test passed")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nFAILED: MultimodalEmbeddingProvider test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multimodal_storage():
    """Test MultimodalStorage save and load operations."""
    print("\n" + "=" * 60)
    print("Test 2: MultimodalStorage")
    print("=" * 60)
    
    try:
        # Create temporary storage
        with tempfile.TemporaryDirectory() as tmpdir:
            print(f"\nCreating storage at: {tmpdir}")
            storage = MultimodalStorage(storage_path=tmpdir)
            
            print(f"Storage: {storage}")
            
            # Store vision embeddings
            print("\n--- Vision Embedding Storage ---")
            for i in range(5):
                vision_emb = np.random.randn(512).astype(np.float32)
                vision_emb = vision_emb / np.linalg.norm(vision_emb)  # Normalize
                
                storage.store_vision_embedding(
                    embedding_id=f"vision_{i:03d}",
                    image_id=f"image_{i:03d}",
                    embedding=vision_emb,
                    model="clip-vit-b32",
                    metadata={"source": "test", "index": i},
                )
            
            print(f"Stored 5 vision embeddings")
            
            # Query vision embeddings
            vision_results = storage.query_vision_embeddings(limit=3)
            print(f"Queried {len(vision_results)} vision embeddings")
            for result in vision_results:
                print(f"  {result['embedding_id']}: {result['image_id']}")
            
            # Store audio embeddings
            print("\n--- Audio Embedding Storage ---")
            for i in range(5):
                audio_emb = np.random.randn(512).astype(np.float32)
                audio_emb = audio_emb / np.linalg.norm(audio_emb)  # Normalize
                
                storage.store_audio_embedding(
                    embedding_id=f"audio_{i:03d}",
                    audio_id=f"audio_{i:03d}",
                    embedding=audio_emb,
                    model="whisper-base",
                    duration_seconds=3.0,
                    metadata={"source": "test", "index": i},
                )
            
            print(f"Stored 5 audio embeddings")
            
            # Query audio embeddings
            audio_results = storage.query_audio_embeddings(limit=3)
            print(f"Queried {len(audio_results)} audio embeddings")
            for result in audio_results:
                print(f"  {result['embedding_id']}: {result['audio_id']}")
            
            # Store CLIP embeddings
            print("\n--- CLIP Embedding Storage ---")
            for i in range(5):
                # Text embeddings
                text_emb = np.random.randn(512).astype(np.float32)
                text_emb = text_emb / np.linalg.norm(text_emb)
                
                storage.store_clip_embedding(
                    embedding_id=f"clip_text_{i:03d}",
                    source_id=f"text_{i:03d}",
                    modality="text",
                    embedding=text_emb,
                    model="clip-vit-b32",
                    metadata={"source": "test", "index": i},
                )
                
                # Image embeddings
                image_emb = np.random.randn(512).astype(np.float32)
                image_emb = image_emb / np.linalg.norm(image_emb)
                
                storage.store_clip_embedding(
                    embedding_id=f"clip_image_{i:03d}",
                    source_id=f"image_{i:03d}",
                    modality="image",
                    embedding=image_emb,
                    model="clip-vit-b32",
                    metadata={"source": "test", "index": i},
                )
            
            print(f"Stored 10 CLIP embeddings (5 text + 5 image)")
            
            # Query CLIP embeddings
            clip_text_results = storage.query_clip_embeddings(modality="text", limit=3)
            print(f"Queried {len(clip_text_results)} CLIP text embeddings")
            for result in clip_text_results:
                print(f"  {result['embedding_id']}: {result['source_id']}")
            
            clip_image_results = storage.query_clip_embeddings(modality="image", limit=3)
            print(f"Queried {len(clip_image_results)} CLIP image embeddings")
            for result in clip_image_results:
                print(f"  {result['embedding_id']}: {result['source_id']}")
            
            # Get storage stats
            print("\n--- Storage Statistics ---")
            stats = storage.get_storage_stats()
            print(f"Vision: {stats['vision']['count']} embeddings, {stats['vision']['size_mb']:.2f} MB")
            print(f"Audio: {stats['audio']['count']} embeddings, {stats['audio']['size_mb']:.2f} MB")
            print(f"CLIP: {stats['clip']['count']} embeddings, {stats['clip']['size_mb']:.2f} MB")
            print(f"Total: {stats['total']['count']} embeddings, {stats['total']['size_mb']:.2f} MB")
            
            print("\n" + "=" * 60)
            print("PASSED: MultimodalStorage test passed")
            print("=" * 60)
            
            return True
            
    except Exception as e:
        print(f"\nFAILED: MultimodalStorage test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests."""
    print("\n" + "=" * 60)
    print("Multimodal Integration Tests")
    print("=" * 60)
    
    results = []
    
    # Test 1: MultimodalEmbeddingProvider
    results.append(("MultimodalEmbeddingProvider", test_multimodal_provider()))
    
    # Test 2: MultimodalStorage
    results.append(("MultimodalStorage", test_multimodal_storage()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for test_name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"{status}: {test_name}")
    
    all_passed = all(passed for _, passed in results)
    
    if all_passed:
        print("\n" + "=" * 60)
        print("All tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("Some tests failed")
        print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
