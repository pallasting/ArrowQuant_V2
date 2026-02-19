"""
Generate test data for multimodal encoder tests

Creates synthetic images and audio for testing purposes.
"""

import numpy as np
from PIL import Image
from pathlib import Path


def generate_test_images(output_dir: Path, count: int = 10):
    """
    Generate synthetic test images.
    
    Args:
        output_dir: Output directory for images
        count: Number of images to generate
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i in range(count):
        # Create random RGB image
        img_array = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        
        # Save image
        img.save(output_dir / f"test_image_{i:03d}.jpg")
    
    print(f"Generated {count} test images in {output_dir}")


def generate_test_audio(output_dir: Path, count: int = 10):
    """
    Generate synthetic test audio.
    
    Args:
        output_dir: Output directory for audio
        count: Number of audio files to generate
    """
    try:
        import soundfile as sf
    except ImportError:
        print("soundfile not installed. Skipping audio generation.")
        print("Install with: pip install soundfile")
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    sample_rate = 16000
    duration = 3  # seconds
    
    for i in range(count):
        # Create random waveform
        waveform = np.random.randn(sample_rate * duration).astype(np.float32)
        
        # Normalize to [-1, 1]
        waveform = waveform / np.abs(waveform).max()
        
        # Save audio
        sf.write(
            output_dir / f"test_audio_{i:03d}.wav",
            waveform,
            sample_rate
        )
    
    print(f"Generated {count} test audio files in {output_dir}")


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent
    
    # Generate test images
    generate_test_images(fixtures_dir / "images", count=10)
    
    # Generate test audio
    generate_test_audio(fixtures_dir / "audio", count=10)
    
    print("\nTest data generation complete!")
