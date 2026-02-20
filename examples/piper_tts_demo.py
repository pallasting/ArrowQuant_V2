"""
Piper TTS Backend Demo

This example demonstrates the Piper TTS backend for fast, local speech synthesis.
Piper provides high-quality speech synthesis without requiring cloud APIs.

Requirements: 1.1, 1.2, 1.3
"""

import numpy as np
from pathlib import Path

from llm_compression.expression.tts import TTSEngine
from llm_compression.expression.expression_types import (
    TTSConfig,
    VoiceConfig,
    TTSBackend
)


def demo_basic_synthesis():
    """Demonstrate basic Piper TTS synthesis."""
    print("=" * 60)
    print("Demo 1: Basic Piper TTS Synthesis")
    print("=" * 60)
    
    # Create TTS engine with Piper backend
    config = TTSConfig(
        backend=TTSBackend.PIPER,
        sample_rate=22050,
        streaming=False,
        cache_enabled=True
    )
    
    engine = TTSEngine(config)
    
    # Synthesize speech
    text = "Hello! This is a demonstration of the Piper text-to-speech engine."
    print(f"\nSynthesizing: '{text}'")
    
    audio_chunks = list(engine.synthesize(text, streaming=False))
    audio = audio_chunks[0]
    
    print(f"Generated audio: {len(audio)} samples, {len(audio) / config.sample_rate:.2f} seconds")
    print(f"Audio shape: {audio.shape}, dtype: {audio.dtype}")
    print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")


def demo_emotion_control():
    """Demonstrate emotion control with Piper TTS."""
    print("\n" + "=" * 60)
    print("Demo 2: Emotion Control")
    print("=" * 60)
    
    config = TTSConfig(backend=TTSBackend.PIPER)
    engine = TTSEngine(config)
    
    text = "I am so excited to show you this feature!"
    
    # Test different emotions
    emotions = [
        ("neutral", 0.5),
        ("joy", 1.0),
        ("sadness", 0.8),
        ("excitement", 0.9)
    ]
    
    for emotion, intensity in emotions:
        voice = VoiceConfig(
            voice_id="en_US-lessac-medium",
            emotion=emotion,
            emotion_intensity=intensity
        )
        
        print(f"\nEmotion: {emotion} (intensity={intensity})")
        print(f"  Voice params: speed={voice.speed:.2f}, pitch={voice.pitch:.2f}")
        
        audio_chunks = list(engine.synthesize(text, voice_config=voice, streaming=False))
        audio = audio_chunks[0]
        
        print(f"  Generated: {len(audio)} samples, {len(audio) / config.sample_rate:.2f}s")


def demo_voice_parameters():
    """Demonstrate voice parameter control."""
    print("\n" + "=" * 60)
    print("Demo 3: Voice Parameter Control")
    print("=" * 60)
    
    config = TTSConfig(backend=TTSBackend.PIPER)
    engine = TTSEngine(config)
    
    text = "Testing different voice parameters."
    
    # Test different parameter combinations
    configs = [
        ("Normal", 1.0, 1.0, 1.0),
        ("Fast", 1.5, 1.0, 1.0),
        ("Slow", 0.7, 1.0, 1.0),
        ("High pitch", 1.0, 1.3, 1.0),
        ("Low pitch", 1.0, 0.8, 1.0),
        ("Quiet", 1.0, 1.0, 0.5),
    ]
    
    for name, speed, pitch, volume in configs:
        voice = VoiceConfig(
            voice_id="en_US-lessac-medium",
            speed=speed,
            pitch=pitch,
            volume=volume
        )
        
        print(f"\n{name}: speed={speed}, pitch={pitch}, volume={volume}")
        
        audio_chunks = list(engine.synthesize(text, voice_config=voice, streaming=False))
        audio = audio_chunks[0]
        
        print(f"  Generated: {len(audio)} samples")
        print(f"  Duration: {len(audio) / config.sample_rate:.2f}s")
        print(f"  Volume range: [{audio.min():.3f}, {audio.max():.3f}]")


def demo_streaming_synthesis():
    """Demonstrate streaming synthesis for long text."""
    print("\n" + "=" * 60)
    print("Demo 4: Streaming Synthesis")
    print("=" * 60)
    
    config = TTSConfig(backend=TTSBackend.PIPER, streaming=True)
    engine = TTSEngine(config)
    
    text = (
        "This is a longer text that will be synthesized in streaming mode. "
        "Streaming allows for lower latency by generating audio sentence by sentence. "
        "This is especially useful for real-time applications and long-form content."
    )
    
    print(f"\nSynthesizing (streaming): '{text[:60]}...'")
    print("\nStreaming chunks:")
    
    total_samples = 0
    for i, audio_chunk in enumerate(engine.synthesize(text, streaming=True)):
        chunk_duration = len(audio_chunk) / config.sample_rate
        total_samples += len(audio_chunk)
        print(f"  Chunk {i+1}: {len(audio_chunk)} samples, {chunk_duration:.2f}s")
    
    total_duration = total_samples / config.sample_rate
    print(f"\nTotal: {total_samples} samples, {total_duration:.2f}s")


def demo_caching():
    """Demonstrate TTS caching for performance."""
    print("\n" + "=" * 60)
    print("Demo 5: TTS Caching")
    print("=" * 60)
    
    config = TTSConfig(backend=TTSBackend.PIPER, cache_enabled=True)
    engine = TTSEngine(config)
    
    text = "This text will be cached for faster repeated synthesis."
    voice = VoiceConfig(voice_id="en_US-lessac-medium")
    
    # First synthesis (not cached)
    print("\nFirst synthesis (not cached):")
    audio1 = list(engine.synthesize(text, voice_config=voice, streaming=False))[0]
    print(f"  Generated: {len(audio1)} samples")
    
    # Second synthesis (should use cache)
    print("\nSecond synthesis (should use cache):")
    audio2 = list(engine.synthesize(text, voice_config=voice, streaming=False))[0]
    print(f"  Generated: {len(audio2)} samples")
    
    # Verify same audio
    if np.array_equal(audio1, audio2):
        print("\n✓ Cache working correctly - same audio returned")
    else:
        print("\n✗ Cache issue - different audio returned")
    
    # Different voice should not use cache
    print("\nThird synthesis (different voice, not cached):")
    voice2 = VoiceConfig(voice_id="en_US-lessac-medium", speed=1.5)
    audio3 = list(engine.synthesize(text, voice_config=voice2, streaming=False))[0]
    print(f"  Generated: {len(audio3)} samples")


def demo_error_handling():
    """Demonstrate graceful error handling."""
    print("\n" + "=" * 60)
    print("Demo 6: Error Handling")
    print("=" * 60)
    
    config = TTSConfig(backend=TTSBackend.PIPER)
    engine = TTSEngine(config)
    
    # Force backend to None to simulate error
    print("\nSimulating backend failure...")
    engine.backend = None
    
    text = "This should handle errors gracefully."
    audio_chunks = list(engine.synthesize(text, streaming=False))
    audio = audio_chunks[0]
    
    print(f"Fallback audio generated: {len(audio)} samples")
    print("✓ Error handled gracefully - returned silence instead of crashing")


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("PIPER TTS BACKEND DEMONSTRATION")
    print("=" * 60)
    print("\nThis demo shows the Piper TTS backend capabilities:")
    print("- Fast local speech synthesis")
    print("- Emotion control")
    print("- Voice parameter adjustment")
    print("- Streaming synthesis")
    print("- Output caching")
    print("- Graceful error handling")
    print("\nNote: If piper-tts is not installed, mock synthesis will be used.")
    print("Install with: pip install piper-tts")
    
    try:
        demo_basic_synthesis()
        demo_emotion_control()
        demo_voice_parameters()
        demo_streaming_synthesis()
        demo_caching()
        demo_error_handling()
        
        print("\n" + "=" * 60)
        print("All demos completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\nError running demos: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
