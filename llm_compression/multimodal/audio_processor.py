"""
AudioProcessor - Arrow-native audio preprocessing

Zero-copy audio preprocessing for Whisper encoder input.
Computes mel-spectrograms from audio waveforms.
"""

from typing import Union, List, Optional
import numpy as np
import pyarrow as pa

from llm_compression.logger import logger


class MelSpectrogramProcessor:
    """
    Compute mel-spectrograms from audio waveforms.
    
    Uses optimized FFT operations with cached filter banks
    for high-performance audio preprocessing.
    """
    
    def __init__(
        self,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        sample_rate: int = 16000
    ):
        """
        Initialize mel-spectrogram processor.
        
        Args:
            n_mels: Number of mel-frequency bins (default: 80 for Whisper)
            n_fft: FFT window size (default: 400)
            hop_length: Hop length for STFT (default: 160)
            sample_rate: Audio sample rate (default: 16000 Hz)
        """
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate
        
        # Pre-compute mel filter banks (cached for performance)
        self._mel_filters: Optional[np.ndarray] = None
        self._initialize_mel_filters()
        
        logger.info(
            f"Initialized MelSpectrogramProcessor: "
            f"n_mels={n_mels}, n_fft={n_fft}, hop_length={hop_length}, sr={sample_rate}"
        )
    
    def _initialize_mel_filters(self) -> None:
        """Pre-compute mel filter banks."""
        try:
            import librosa
            self._mel_filters = librosa.filters.mel(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels
            )
            logger.debug(f"Initialized mel filters: shape={self._mel_filters.shape}")
        except ImportError:
            logger.warning(
                "librosa not installed. Mel-spectrogram computation will be unavailable. "
                "Install with: pip install librosa"
            )
            self._mel_filters = None
    
    def compute_mel_spectrogram(
        self,
        waveform: np.ndarray,
        max_frames: Optional[int] = None
    ) -> np.ndarray:
        """
        Compute mel-spectrogram from audio waveform with Whisper-style normalization.
        
        Args:
            waveform: Audio waveform (n_samples,) float32
            max_frames: Maximum number of frames (default: None = no limit)
            
        Returns:
            Mel-spectrogram: (n_mels, n_frames) float32
        """
        if self._mel_filters is None:
            raise RuntimeError(
                "Mel filters not initialized. Install librosa: pip install librosa"
            )
        
        import librosa
        
        # Compute STFT
        stft = librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Magnitude spectrogram (power spectrum)
        magnitude = np.abs(stft) ** 2
        
        # Apply mel filter banks (cached, zero-copy)
        mel_spec = self._mel_filters @ magnitude
        
        # Log scale (log10 for Whisper compatibility)
        mel_spec = np.log10(np.clip(mel_spec, a_min=1e-10, a_max=None))
        
        # Whisper-specific normalization:
        # 1. Clip to max - 8.0 (dynamic range compression)
        # 2. Normalize to roughly [-1, 1] range
        mel_spec = np.maximum(mel_spec, mel_spec.max() - 8.0)
        mel_spec = (mel_spec + 4.0) / 4.0
        
        # Truncate or pad to max_frames if specified
        if max_frames is not None:
            n_frames = mel_spec.shape[1]
            if n_frames > max_frames:
                mel_spec = mel_spec[:, :max_frames]
            elif n_frames < max_frames:
                pad_width = max_frames - n_frames
                mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
        
        return mel_spec.astype(np.float32)
    
    def compute_batch(
        self,
        waveforms: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Compute mel-spectrograms for batch of waveforms.
        
        Args:
            waveforms: List of audio waveforms
            
        Returns:
            List of mel-spectrograms
        """
        mel_specs = []
        for waveform in waveforms:
            try:
                mel_spec = self.compute_mel_spectrogram(waveform)
                mel_specs.append(mel_spec)
            except Exception as e:
                logger.error(f"Failed to compute mel-spectrogram: {e}")
                # Add zero spectrogram as placeholder
                mel_specs.append(np.zeros((self.n_mels, 1), dtype=np.float32))
        
        return mel_specs


class AudioProcessor:
    """
    Arrow-native audio preprocessing for Whisper encoder.
    
    Features:
    - Zero-copy operations where possible
    - Mel-spectrogram computation
    - Batch processing support
    - Arrow Binary array I/O
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        max_audio_length: int = 30,
        n_mels: int = 80,
        n_fft: int = 400,
        hop_length: int = 160,
        max_mel_frames: Optional[int] = None
    ):
        """
        Initialize audio processor.
        
        Args:
            sample_rate: Target sample rate (default: 16000 Hz for Whisper)
            max_audio_length: Maximum audio length in seconds (default: 30)
            n_mels: Number of mel-frequency bins (default: 80)
            n_fft: FFT window size (default: 400)
            hop_length: Hop length for STFT (default: 160)
            max_mel_frames: Maximum mel-spectrogram frames (default: None = auto-calculate)
        """
        self.sample_rate = sample_rate
        self.max_audio_length = max_audio_length
        self.max_samples = sample_rate * max_audio_length
        
        # Calculate max mel frames if not specified
        # For Whisper: 30s audio @ 16kHz with hop_length=160 → 3000 frames
        # After conv2 (stride=2): 1500 frames (matches max_positions)
        # To ensure we don't exceed max_positions after conv2, we need:
        # max_mel_frames such that (max_mel_frames // 2) <= max_positions
        # So max_mel_frames = max_positions * 2 = 3000
        if max_mel_frames is None:
            # Default to 3000 frames (will be 1500 after conv2 with stride=2)
            self.max_mel_frames = 3000
        else:
            self.max_mel_frames = max_mel_frames
        
        # Initialize mel-spectrogram processor
        self.mel_processor = MelSpectrogramProcessor(
            n_mels=n_mels,
            n_fft=n_fft,
            hop_length=hop_length,
            sample_rate=sample_rate
        )
        
        logger.info(
            f"Initialized AudioProcessor: "
            f"sr={sample_rate}, max_length={max_audio_length}s, "
            f"max_mel_frames={self.max_mel_frames}"
        )
    
    def load_audio(
        self,
        audio_path: str,
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> np.ndarray:
        """
        Load audio file and resample to target sample rate.
        
        Args:
            audio_path: Path to audio file
            offset: Start time in seconds (default: 0.0)
            duration: Duration in seconds (default: None = full file)
            
        Returns:
            Audio waveform: (n_samples,) float32
        """
        try:
            import librosa
            
            # Load and resample audio
            waveform, sr = librosa.load(
                audio_path,
                sr=self.sample_rate,
                offset=offset,
                duration=duration or self.max_audio_length
            )
            
            return waveform.astype(np.float32)
            
        except ImportError:
            raise RuntimeError(
                "librosa not installed. Install with: pip install librosa"
            )
        except Exception as e:
            logger.error(f"Failed to load audio {audio_path}: {e}")
            raise
    
    def preprocess(
        self,
        waveform: np.ndarray,
        pad_or_trim: bool = True
    ) -> np.ndarray:
        """
        Preprocess audio waveform.
        
        Args:
            waveform: Audio waveform (n_samples,) float32
            pad_or_trim: Whether to pad/trim to max_samples (default: True)
            
        Returns:
            Preprocessed waveform: (max_samples,) float32
        """
        # Pad or trim to max length
        if pad_or_trim:
            if len(waveform) > self.max_samples:
                waveform = waveform[:self.max_samples]
            elif len(waveform) < self.max_samples:
                waveform = np.pad(
                    waveform,
                    (0, self.max_samples - len(waveform)),
                    mode='constant'
                )
        
        return waveform
    
    def preprocess_batch(
        self,
        waveforms: List[np.ndarray],
        pad_or_trim: bool = True
    ) -> np.ndarray:
        """
        Preprocess batch of audio waveforms.
        
        Args:
            waveforms: List of audio waveforms
            pad_or_trim: Whether to pad/trim to max_samples
            
        Returns:
            Batch of preprocessed waveforms: (batch, max_samples) float32
        """
        processed = []
        for waveform in waveforms:
            processed.append(self.preprocess(waveform, pad_or_trim))
        
        return np.stack(processed, axis=0)
    
    def compute_mel_spectrogram(
        self,
        waveform: np.ndarray
    ) -> np.ndarray:
        """
        Compute mel-spectrogram from waveform.
        
        Args:
            waveform: Audio waveform (n_samples,) float32
            
        Returns:
            Mel-spectrogram: (n_mels, n_frames) float32
        """
        return self.mel_processor.compute_mel_spectrogram(
            waveform,
            max_frames=self.max_mel_frames
        )
    
    def to_arrow(self, waveforms: np.ndarray) -> pa.Array:
        """
        Convert waveforms to Arrow Binary array.
        
        Args:
            waveforms: (batch, n_samples) float32 array
            
        Returns:
            Arrow Binary array (zero-copy when possible)
        """
        binary_data = []
        for waveform in waveforms:
            binary_data.append(waveform.tobytes())
        
        return pa.array(binary_data, type=pa.binary())
    
    def from_arrow(self, arrow_array: pa.Array) -> np.ndarray:
        """
        Load waveforms from Arrow Binary array.
        
        Args:
            arrow_array: Arrow Binary array
            
        Returns:
            Batch of waveforms: (batch, n_samples) float32 array
        """
        waveforms = []
        for audio_bytes in arrow_array:
            waveform = np.frombuffer(audio_bytes.as_py(), dtype=np.float32)
            waveforms.append(waveform)
        
        return np.stack(waveforms, axis=0)
