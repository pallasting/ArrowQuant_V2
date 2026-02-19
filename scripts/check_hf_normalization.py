#!/usr/bin/env python3
"""Check HuggingFace Whisper normalization behavior."""

import numpy as np
from transformers import WhisperProcessor

# Generate test audio
audio = np.random.randn(48000).astype(np.float32)

processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

# Test with do_normalize=False (default)
print("Testing with do_normalize=False:")
inputs_no_norm = processor(audio, sampling_rate=16000, return_tensors="pt", do_normalize=False)
mel_no_norm = inputs_no_norm.input_features.numpy()
print(f"  Mean: {mel_no_norm.mean():.6f}")
print(f"  Std: {mel_no_norm.std():.6f}")
print(f"  Min: {mel_no_norm.min():.6f}")
print(f"  Max: {mel_no_norm.max():.6f}")

# Test with do_normalize=True
print("\nTesting with do_normalize=True:")
inputs_norm = processor(audio, sampling_rate=16000, return_tensors="pt", do_normalize=True)
mel_norm = inputs_norm.input_features.numpy()
print(f"  Mean: {mel_norm.mean():.6f}")
print(f"  Std: {mel_norm.std():.6f}")
print(f"  Min: {mel_norm.min():.6f}")
print(f"  Max: {mel_norm.max():.6f}")

# Test default behavior (no do_normalize specified)
print("\nTesting default behavior (no do_normalize):")
inputs_default = processor(audio, sampling_rate=16000, return_tensors="pt")
mel_default = inputs_default.input_features.numpy()
print(f"  Mean: {mel_default.mean():.6f}")
print(f"  Std: {mel_default.std():.6f}")
print(f"  Min: {mel_default.min():.6f}")
print(f"  Max: {mel_default.max():.6f}")

# Check if default matches normalized
if np.allclose(mel_default, mel_norm, rtol=1e-5):
    print("\n✅ Default behavior uses normalization!")
elif np.allclose(mel_default, mel_no_norm, rtol=1e-5):
    print("\n✅ Default behavior does NOT use normalization!")
else:
    print("\n⚠️  Default behavior is different from both!")
