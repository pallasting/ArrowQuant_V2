#!/usr/bin/env python3
"""
Debug Mel-Spectrogram Computation

Compare mel-spectrogram computation between HuggingFace and ArrowEngine.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger


def debug_mel_computation():
    """Debug mel-spectrogram computation."""
    
    print("\n" + "="*70)
    print("  Mel-Spectrogram Debug Tool")
    print("="*70)
    
    # Generate test audio
    print("\n[1/4] Generating test audio...")
    sample_rate = 16000
    duration = 3.0
    num_samples = int(sample_rate * duration)
    audio = np.random.randn(num_samples).astype(np.float32)
    print(f"✅ Audio: shape={audio.shape}, mean={audio.mean():.6f}, std={audio.std():.6f}")
    
    # HuggingFace preprocessing
    print("\n[2/4] HuggingFace preprocessing...")
    from transformers import WhisperProcessor
    
    hf_processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    hf_inputs = hf_processor(audio, sampling_rate=sample_rate, return_tensors="pt")
    hf_mel = hf_inputs.input_features.numpy()
    
    print(f"HF mel-spectrogram:")
    print(f"  Shape: {hf_mel.shape}")
    print(f"  Mean: {hf_mel.mean():.6f}")
    print(f"  Std: {hf_mel.std():.6f}")
    print(f"  Min: {hf_mel.min():.6f}")
    print(f"  Max: {hf_mel.max():.6f}")
    
    # Check HuggingFace feature extractor details
    print(f"\nHF Feature Extractor Config:")
    feature_extractor = hf_processor.feature_extractor
    print(f"  n_fft: {feature_extractor.n_fft}")
    print(f"  hop_length: {feature_extractor.hop_length}")
    print(f"  n_mels: {feature_extractor.feature_size}")
    print(f"  sampling_rate: {feature_extractor.sampling_rate}")
    
    # Check if there's normalization
    if hasattr(feature_extractor, 'do_normalize'):
        print(f"  do_normalize: {feature_extractor.do_normalize}")
    if hasattr(feature_extractor, 'mean'):
        print(f"  mean: {feature_extractor.mean}")
    if hasattr(feature_extractor, 'std'):
        print(f"  std: {feature_extractor.std}")
    
    # ArrowEngine preprocessing
    print("\n[3/4] ArrowEngine preprocessing...")
    from llm_compression.multimodal.audio_processor import AudioProcessor
    
    arrow_processor = AudioProcessor(
        sample_rate=16000,
        max_audio_length=30,
        n_mels=80,
        n_fft=400,
        hop_length=160
    )
    
    preprocessed = arrow_processor.preprocess_batch([audio], pad_or_trim=True)
    arrow_mel = arrow_processor.compute_mel_spectrogram(preprocessed[0])
    
    print(f"Arrow mel-spectrogram:")
    print(f"  Shape: {arrow_mel.shape}")
    print(f"  Mean: {arrow_mel.mean():.6f}")
    print(f"  Std: {arrow_mel.std():.6f}")
    print(f"  Min: {arrow_mel.min():.6f}")
    print(f"  Max: {arrow_mel.max():.6f}")
    
    # Compare
    print("\n[4/4] Comparison...")
    print(f"Mean difference: {abs(hf_mel.mean() - arrow_mel.mean()):.6f}")
    print(f"Std difference: {abs(hf_mel.std() - arrow_mel.std()):.6f}")
    
    # Check if HF applies normalization
    print("\n" + "="*70)
    print("  Analysis")
    print("="*70)
    
    if abs(hf_mel.mean()) < 1.0 and abs(arrow_mel.mean()) > 10.0:
        print("\n⚠️  ISSUE FOUND: HuggingFace applies normalization!")
        print("   HF mel values are normalized (mean ~0, std ~0.5)")
        print("   Arrow mel values are in log scale (mean ~-17)")
        print("\n   Solution: Apply same normalization in ArrowEngine")
        
        # Try to reverse-engineer the normalization
        print("\n   Attempting to match HF normalization...")
        
        # Common normalization: (x - mean) / std
        arrow_normalized = (arrow_mel - arrow_mel.mean()) / arrow_mel.std()
        print(f"\n   Arrow normalized (z-score):")
        print(f"     Mean: {arrow_normalized.mean():.6f}")
        print(f"     Std: {arrow_normalized.std():.6f}")
        
        # Check if this matches HF
        diff = abs(hf_mel.mean() - arrow_normalized.mean())
        if diff < 0.1:
            print(f"   ✅ Z-score normalization matches! (diff={diff:.6f})")
        else:
            print(f"   ❌ Z-score doesn't match (diff={diff:.6f})")
            
            # Try other normalizations
            # Min-max normalization
            arrow_minmax = (arrow_mel - arrow_mel.min()) / (arrow_mel.max() - arrow_mel.min())
            arrow_minmax = arrow_minmax * 2 - 1  # Scale to [-1, 1]
            print(f"\n   Arrow min-max normalized:")
            print(f"     Mean: {arrow_minmax.mean():.6f}")
            print(f"     Std: {arrow_minmax.std():.6f}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    debug_mel_computation()
