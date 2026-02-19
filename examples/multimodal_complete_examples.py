"""
Multimodal Encoder System - Complete Usage Examples

This script demonstrates all features of the multimodal encoder system.
"""

import numpy as np
from PIL import Image
import librosa

from llm_compression.multimodal import get_multimodal_provider, MultimodalStorage


def example_1_text_encoding():
    """Example 1: Text encoding (backward compatible)"""
    print("\n=== Example 1: Text Encoding ===")
    
    provider = get_multimodal_provider(
        text_model_path="D:/ai-models/bert-base-uncased"
    )
    
    # Single text
    text = "Hello, world!"
    embedding = provider.encode(text)
    print(f"Text: '{text}'")
    print(f"Embedding shape: {embedding.shape}")  # (384,)
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")
    
    # Batch of texts
    texts = ["cat", "dog", "bird"]
    embeddings = provider.encode_batch(texts)
    print(f"\nBatch encoding: {len(texts)} texts")
    print(f"Embeddings shape: {embeddings.shape}")  # (3, 384)


def example_2_image_encoding():
    """Example 2: Image encoding with CLIP Vision Transformer"""
    print("\n=== Example 2: Image Encoding ===")
    
    provider = get_multimodal_provider(
        vision_model_path="D:/ai-models/clip-vit-b32"
    )
    
    # Load and preprocess image
    image = Image.open("test_image.jpg").resize((224, 224))
    image_array = np.array(image)
    
    # Add batch dimension
    image_batch = image_array[np.newaxis, ...]
    
    # Encode
    embedding = provider.encode_image(image_batch, normalize=True)
    print(f"Image shape: {image_array.shape}")
    print(f"Embedding shape: {embedding.shape}")  # (1, 512)
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")


def example_3_audio_encoding():
    """Example 3: Audio encoding with Whisper"""
    print("\n=== Example 3: Audio Encoding ===")
    
    provider = get_multimodal_provider(
        audio_model_path="D:/ai-models/whisper-base"
    )
    
    # Load audio (16kHz required)
    audio, sr = librosa.load("test_audio.wav", sr=16000)
    print(f"Audio duration: {len(audio) / sr:.2f} seconds")
    
    # Encode
    embedding = provider.encode_audio(audio, normalize=True)
    print(f"Embedding shape: {embedding.shape}")  # (512,)
    print(f"Embedding norm: {np.linalg.norm(embedding):.4f}")


def example_4_multimodal_encoding():
    """Example 4: Encode multiple modalities at once"""
    print("\n=== Example 4: Multimodal Encoding ===")
    
    provider = get_multimodal_provider(
        text_model_path="D:/ai-models/bert-base-uncased",
        vision_model_path="D:/ai-models/clip-vit-b32",
        audio_model_path="D:/ai-models/whisper-base"
    )
    
    # Prepare data
    texts = ["cat", "dog"]
    image = np.random.randint(0, 255, (1, 224, 224, 3), dtype=np.uint8)
    audio = np.random.randn(16000).astype(np.float32)
    
    # Encode all modalities
    result = provider.encode_multimodal(
        text=texts,
        images=image,
        audio=audio
    )
    
    print(f"Text embeddings: {result['text'].shape}")
    print(f"Vision embeddings: {result['vision'].shape}")
    print(f"Audio embeddings: {result['audio'].shape}")


def example_5_cross_modal_similarity():
    """Example 5: Text-image similarity with CLIP"""
    print("\n=== Example 5: Cross-Modal Similarity ===")
    
    provider = get_multimodal_provider(
        text_model_path="D:/ai-models/bert-base-uncased",
        vision_model_path="D:/ai-models/clip-vit-b32"
    )
    
    # Prepare data
    texts = ["a cat", "a dog", "a bird"]
    images = np.random.randint(0, 255, (5, 224, 224, 3), dtype=np.uint8)
    
    # Compute similarity
    similarity = provider.compute_cross_modal_similarity(texts, images)
    print(f"Similarity matrix shape: {similarity.shape}")  # (3, 5)
    print(f"Similarity range: [{similarity.min():.4f}, {similarity.max():.4f}]")


def example_6_image_retrieval():
    """Example 6: Text-to-image retrieval"""
    print("\n=== Example 6: Image Retrieval ===")
    
    provider = get_multimodal_provider(
        text_model_path="D:/ai-models/bert-base-uncased",
        vision_model_path="D:/ai-models/clip-vit-b32"
    )
    
    # Prepare data
    texts = ["a cat sitting on a couch"]
    images = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)
    
    # Find best matches
    matches = provider.find_best_image_matches(texts, images, top_k=3)
    print(f"Query: '{texts[0]}'")
    print(f"Top 3 matching image indices: {matches[0]}")


def example_7_zero_shot_classification():
    """Example 7: Zero-shot image classification"""
    print("\n=== Example 7: Zero-Shot Classification ===")
    
    provider = get_multimodal_provider(
        text_model_path="D:/ai-models/bert-base-uncased",
        vision_model_path="D:/ai-models/clip-vit-b32"
    )
    
    # Prepare data
    images = np.random.randint(0, 255, (3, 224, 224, 3), dtype=np.uint8)
    class_labels = ["cat", "dog", "bird", "car", "tree"]
    
    # Classify
    predictions, probabilities = provider.zero_shot_classification(
        images, class_labels
    )
    
    print(f"Number of images: {len(images)}")
    print(f"Class labels: {class_labels}")
    print(f"Predictions: {predictions}")
    print(f"Probabilities shape: {probabilities.shape}")  # (3, 5)


def example_8_storage():
    """Example 8: Store and query embeddings"""
    print("\n=== Example 8: Storage ===")
    
    storage = MultimodalStorage(
        storage_path="./test_embeddings",
        compression_level=3
    )
    
    # Store vision embedding
    vision_emb = np.random.randn(512).astype(np.float32)
    storage.store_vision_embedding(
        embedding_id="emb_001",
        image_id="img_123",
        embedding=vision_emb,
        model="clip-vit-b32",
        metadata={"source": "camera"}
    )
    print("Stored vision embedding")
    
    # Store audio embedding
    audio_emb = np.random.randn(512).astype(np.float32)
    storage.store_audio_embedding(
        embedding_id="emb_002",
        audio_id="aud_456",
        embedding=audio_emb,
        model="whisper-base",
        duration=3.5
    )
    print("Stored audio embedding")
    
    # Query embeddings
    vision_results = storage.query_vision_embeddings(limit=10)
    print(f"Vision embeddings count: {len(vision_results)}")
    
    # Get statistics
    stats = storage.get_storage_stats()
    print(f"Storage stats: {stats}")


def example_9_batch_processing():
    """Example 9: Efficient batch processing"""
    print("\n=== Example 9: Batch Processing ===")
    
    provider = get_multimodal_provider(
        vision_model_path="D:/ai-models/clip-vit-b32"
    )
    
    # Process large batch of images
    batch_size = 16
    num_images = 100
    
    all_embeddings = []
    for i in range(0, num_images, batch_size):
        # Generate batch
        batch = np.random.randint(
            0, 255, 
            (min(batch_size, num_images - i), 224, 224, 3),
            dtype=np.uint8
        )
        
        # Encode batch
        embeddings = provider.encode_image(batch)
        all_embeddings.append(embeddings)
        
        print(f"Processed batch {i//batch_size + 1}: {len(batch)} images")
    
    # Concatenate all embeddings
    all_embeddings = np.vstack(all_embeddings)
    print(f"Total embeddings: {all_embeddings.shape}")  # (100, 512)


def example_10_available_modalities():
    """Example 10: Check available modalities"""
    print("\n=== Example 10: Available Modalities ===")
    
    provider = get_multimodal_provider(
        text_model_path="D:/ai-models/bert-base-uncased",
        vision_model_path="D:/ai-models/clip-vit-b32"
    )
    
    modalities = provider.get_available_modalities()
    print(f"Available modalities: {modalities}")
    
    print(f"Text dimension: {provider.dimension}")
    print(f"Vision dimension: {provider.vision_dimension}")


if __name__ == "__main__":
    print("Multimodal Encoder System - Complete Examples")
    print("=" * 50)
    
    # Run all examples
    try:
        example_1_text_encoding()
    except Exception as e:
        print(f"Example 1 failed: {e}")
    
    try:
        example_2_image_encoding()
    except Exception as e:
        print(f"Example 2 failed: {e}")
    
    try:
        example_3_audio_encoding()
    except Exception as e:
        print(f"Example 3 failed: {e}")
    
    try:
        example_4_multimodal_encoding()
    except Exception as e:
        print(f"Example 4 failed: {e}")
    
    try:
        example_5_cross_modal_similarity()
    except Exception as e:
        print(f"Example 5 failed: {e}")
    
    try:
        example_6_image_retrieval()
    except Exception as e:
        print(f"Example 6 failed: {e}")
    
    try:
        example_7_zero_shot_classification()
    except Exception as e:
        print(f"Example 7 failed: {e}")
    
    try:
        example_8_storage()
    except Exception as e:
        print(f"Example 8 failed: {e}")
    
    try:
        example_9_batch_processing()
    except Exception as e:
        print(f"Example 9 failed: {e}")
    
    try:
        example_10_available_modalities()
    except Exception as e:
        print(f"Example 10 failed: {e}")
    
    print("\n" + "=" * 50)
    print("Examples completed!")
