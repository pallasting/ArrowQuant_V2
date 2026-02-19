"""
Unit tests for AudioProcessor

Tests Arrow-native audio preprocessing functionality.
"""

import pytest
import numpy as np

from llm_compression.multimodal.audio_processor import (
    AudioProcessor,
    MelSpectrogramProcessor
)


def _has_librosa() -> bool:
    """Check if librosa is installed."""
    try:
        import librosa
        return True
    except ImportError:
        return False


class TestMelSpectrogramProcessor:
    """Test MelSpectrogramProcessor functionality."""
    
    def test_initialization(self):
        """Test MelSpectrogramProcessor initialization."""
        processor = MelSpectrogramProcessor(
            n_mels=80,
            n_fft=400,
            hop_length=160,
            sample_rate=16000
        )
        
        assert processor.n_mels == 80
        assert processor.n_fft == 400
        assert processor.hop_length == 160
        assert processor.sample_rate == 16000
    
    @pytest.mark.skipif(
        not _has_librosa(),
        reason="librosa not installed"
    )
    def test_compute_mel_spectrogram(self):
        """Test mel-spectrogram computation."""
        processor = MelSpectrogramProcessor()
        
        # Create random waveform (1 second at 16kHz)
        waveform = np.random.randn(16000).astype(np.float32)
        
        # Compute mel-spectrogram
        mel_spec = processor.compute_mel_spectrogram(waveform)
        
        # Check output shape
        assert mel_spec.shape[0] == 80  # n_mels
        assert mel_spec.shape[1] > 0  # n_frames
        assert mel_spec.dtype == np.float32
    
    @pytest.mark.skipif(
        not _has_librosa(),
        reason="librosa not installed"
    )
    def test_compute_batch(self):
        """Test batch mel-spectrogram computation."""
        processor = MelSpectrogramProcessor()
        
        # Create batch of waveforms
        waveforms = [
            np.random.randn(16000).astype(np.float32)
            for _ in range(4)
        ]
        
        # Compute batch
        mel_specs = processor.compute_batch(waveforms)
        
        # Check output
        assert len(mel_specs) == 4
        for mel_spec in mel_specs:
            assert mel_spec.shape[0] == 80


class TestAudioProcessor:
    """Test AudioProcessor functionality."""
    
    def test_initialization(self):
        """Test AudioProcessor initialization."""
        processor = AudioProcessor(
            sample_rate=16000,
            max_audio_length=30
        )
        
        assert processor.sample_rate == 16000
        assert processor.max_audio_length == 30
        assert processor.max_samples == 16000 * 30
    
    def test_preprocess_pad(self):
        """Test preprocessing with padding."""
        processor = AudioProcessor(sample_rate=16000, max_audio_length=30)
        
        # Create short waveform
        waveform = np.random.randn(8000).astype(np.float32)
        
        # Preprocess with padding
        processed = processor.preprocess(waveform, pad_or_trim=True)
        
        # Check output
        assert processed.shape == (processor.max_samples,)
        assert processed.dtype == np.float32
        
        # Check padding (last part should be zeros)
        assert np.all(processed[8000:] == 0)
    
    def test_preprocess_trim(self):
        """Test preprocessing with trimming."""
        processor = AudioProcessor(sample_rate=16000, max_audio_length=30)
        
        # Create long waveform
        waveform = np.random.randn(processor.max_samples + 10000).astype(np.float32)
        
        # Preprocess with trimming
        processed = processor.preprocess(waveform, pad_or_trim=True)
        
        # Check output
        assert processed.shape == (processor.max_samples,)
    
    def test_preprocess_no_pad_trim(self):
        """Test preprocessing without padding/trimming."""
        processor = AudioProcessor(sample_rate=16000, max_audio_length=30)
        
        # Create waveform
        waveform = np.random.randn(8000).astype(np.float32)
        
        # Preprocess without padding
        processed = processor.preprocess(waveform, pad_or_trim=False)
        
        # Check output (should be unchanged)
        assert processed.shape == (8000,)
        np.testing.assert_array_equal(processed, waveform)
    
    def test_preprocess_batch(self):
        """Test batch preprocessing."""
        processor = AudioProcessor(sample_rate=16000, max_audio_length=30)
        
        # Create batch of waveforms
        waveforms = [
            np.random.randn(8000).astype(np.float32)
            for _ in range(4)
        ]
        
        # Preprocess batch
        processed = processor.preprocess_batch(waveforms, pad_or_trim=True)
        
        # Check output
        assert processed.shape == (4, processor.max_samples)
        assert processed.dtype == np.float32
    
    def test_arrow_roundtrip(self):
        """Test Arrow Binary array roundtrip."""
        processor = AudioProcessor(sample_rate=16000, max_audio_length=30)
        
        # Create batch of waveforms
        waveforms = np.random.randn(4, processor.max_samples).astype(np.float32)
        
        # Convert to Arrow
        arrow_array = processor.to_arrow(waveforms)
        
        # Convert back
        recovered = processor.from_arrow(arrow_array)
        
        # Check roundtrip accuracy
        assert recovered.shape == waveforms.shape
        assert recovered.dtype == waveforms.dtype
        np.testing.assert_array_almost_equal(recovered, waveforms, decimal=6)

