"""
AudioEncoder - High-level Whisper Audio Encoder

Wraps AudioInferenceCore with preprocessing, weight loading, and
a unified interface compatible with EmbeddingProvider protocol.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union, List
import numpy as np
import torch

from llm_compression.logger import logger
from llm_compression.inference.audio_core import AudioInferenceCore
from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.multimodal.audio_processor import AudioProcessor


@dataclass
class AudioConfig:
    """Whisper audio encoder configuration."""
    n_mels: int = 80
    n_fft: int = 400
    hop_length: int = 160
    sample_rate: int = 16000
    hidden_size: int = 512
    num_layers: int = 6
    num_attention_heads: int = 8
    intermediate_size: int = 2048
    layer_norm_eps: float = 1e-5
    max_positions: int = 1500
    max_audio_length: int = 30  # seconds
    
    def to_dict(self):
        """Convert to dictionary for AudioInferenceCore."""
        return {
            "n_mels": self.n_mels,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "layer_norm_eps": self.layer_norm_eps,
            "max_positions": self.max_positions,
        }


class AudioEncoder:
    """
    Whisper audio encoder.
    
    High-level interface that combines:
    - AudioProcessor for preprocessing (mel-spectrogram)
    - AudioInferenceCore for encoding
    - WeightLoader for loading from Parquet
    
    Features:
    - Fast loading (<500ms target)
    - Fast inference (<200ms target)
    - Arrow-native architecture
    - Reuses TransformerLayer from InferenceCore
    
    Example:
        encoder = AudioEncoder("models/whisper-base")
        embedding = encoder.encode(waveform)  # (512,) float32
    """
    
    def __init__(
        self,
        model_path: Union[str, Path],
        device: Optional[str] = None,
        config: Optional[AudioConfig] = None
    ):
        """
        Initialize Audio Encoder.
        
        Args:
            model_path: Path to model directory with weights.parquet
            device: Device for inference ('cpu', 'cuda', or None for auto)
            config: Audio configuration (default: Whisper base)
        """
        self.model_path = Path(model_path)
        self.device = device or self._get_default_device()
        self.config = config or AudioConfig()
        
        logger.info(f"Initializing AudioEncoder from {self.model_path}")
        
        # Initialize audio processor
        self.audio_processor = AudioProcessor(
            sample_rate=self.config.sample_rate,
            max_audio_length=self.config.max_audio_length,
            n_mels=self.config.n_mels,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length
        )
        
        # Load weights
        weights_path = self.model_path / "weights.parquet"
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
        
        self.weight_loader = WeightLoader(str(weights_path))
        weights = self.weight_loader.load_weights()
        
        # Initialize core
        self.core = AudioInferenceCore(
            weights=weights,
            config=self.config.to_dict(),
            device=self.device
        )
        
        logger.info(
            f"AudioEncoder initialized: "
            f"n_mels={self.config.n_mels}, "
            f"hidden_size={self.config.hidden_size}, "
            f"sample_rate={self.config.sample_rate}, "
            f"device={self.device}"
        )
    
    def _get_default_device(self) -> str:
        """Get default device (CPU or CUDA)."""
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    
    @property
    def embedding_dimension(self) -> int:
        """Get embedding dimension."""
        return self.config.hidden_size
    
    def get_embedding_dimension(self) -> int:
        """Get embedding dimension (compatibility method)."""
        return self.embedding_dimension
    
    def encode(
        self,
        audio: Union[np.ndarray, List[np.ndarray], str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode audio to embeddings.
        
        Args:
            audio: Single audio or batch of audio
                  Can be: numpy array (waveform), file path, or list of either
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            Embeddings: (batch, hidden_size) float32 numpy array
        """
        # Handle single audio
        if isinstance(audio, (str, np.ndarray)):
            if isinstance(audio, str) or (isinstance(audio, np.ndarray) and audio.ndim == 1):
                audio = [audio]
        
        # Load audio files if needed
        waveforms = []
        for item in audio:
            if isinstance(item, str):
                waveform = self.audio_processor.load_audio(item)
            else:
                waveform = item
            waveforms.append(waveform)
        
        # Preprocess waveforms
        preprocessed = self.audio_processor.preprocess_batch(waveforms, pad_or_trim=True)
        
        # Compute mel-spectrograms
        mel_specs = []
        for waveform in preprocessed:
            mel_spec = self.audio_processor.compute_mel_spectrogram(waveform)
            mel_specs.append(mel_spec)
        
        # Stack mel-spectrograms: (batch, n_mels, n_frames)
        mel_tensor = torch.from_numpy(np.stack(mel_specs, axis=0)).to(self.device)
        
        # Encode
        with torch.no_grad():
            embeddings = self.core(mel_tensor)
        
        # Normalize if requested
        if normalize:
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        
        # Convert to numpy
        embeddings_np = embeddings.cpu().numpy().astype(np.float32)
        
        # Return single embedding if single audio
        if len(embeddings_np) == 1:
            return embeddings_np[0]
        
        return embeddings_np
    
    def encode_batch(
        self,
        audio: List[Union[np.ndarray, str]],
        batch_size: int = 16,
        normalize: bool = True,
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Encode batch of audio with optional batching.
        
        Args:
            audio: List of audio (numpy arrays or file paths)
            batch_size: Batch size for processing
            normalize: Whether to L2-normalize embeddings
            show_progress: Whether to show progress (not implemented yet)
            
        Returns:
            Embeddings: (n_audio, hidden_size) float32 numpy array
        """
        if len(audio) == 0:
            return np.zeros((0, self.embedding_dimension), dtype=np.float32)
        
        # Process in batches
        all_embeddings = []
        
        for i in range(0, len(audio), batch_size):
            batch = audio[i:i + batch_size]
            batch_embeddings = self.encode(batch, normalize=normalize)
            
            # Handle single vs batch output
            if batch_embeddings.ndim == 1:
                batch_embeddings = batch_embeddings[np.newaxis, :]
            
            all_embeddings.append(batch_embeddings)
        
        return np.vstack(all_embeddings)
    
    def __repr__(self) -> str:
        return (
            f"AudioEncoder("
            f"model_path={self.model_path}, "
            f"device={self.device}, "
            f"embedding_dim={self.embedding_dimension})"
        )
