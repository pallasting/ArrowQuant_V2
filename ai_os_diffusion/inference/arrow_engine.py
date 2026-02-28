"""
ArrowEngine - High-performance inference engine for AI-OS Diffusion.

Unified entry point for both autoregressive and diffusion generation.

Example:
    >>> from ..inference import ArrowEngine
    >>> 
    >>> # Autoregressive mode (embedding)
    >>> engine = ArrowEngine("./models/minilm")
    >>> embeddings = engine.encode(["Hello, world!"])
    >>> 
    >>> # Diffusion mode (text generation)
    >>> result = engine.diffuse(
    ...     prompt="A beautiful sunset",
    ...     modality="text",
    ...     num_steps=4
    ... )
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from ..utils.logger import logger
from ..utils.errors import ModelLoadError, InferenceError
from .device_utils import get_best_device, get_device_info
from .weight_loader import WeightLoader, LazyWeightDict
from .fast_tokenizer import FastTokenizer
from .inference_core import InferenceCore


class ArrowEngine:
    """
    High-performance inference engine for AI-OS Diffusion.
    
    Unified entry point supporting both:
    - Autoregressive mode: encode() for embeddings
    - Diffusion mode: diffuse() for generation
    
    Architecture:
    - 🦴 Rust Skeleton: ArrowStorage, ArrowQuant, FastTokenizer (10-100x speedup)
    - 🧠 Python Brain: DiffusionCore, EvolutionRouter (flexible learning)
    
    Performance Targets:
    - Model load time: < 100ms (vs 2-5s traditional)
    - Inference latency: < 5ms per sequence (AR mode)
    - Diffusion latency: < 500ms for 4-step generation (text)
    - Memory usage: < 50% of original model size (INT2 quantization)
    
    Example:
        >>> engine = ArrowEngine("./models/unified-diffusion")
        >>> 
        >>> # AR mode: embeddings
        >>> embeddings = engine.encode(["AI", "ML"])
        >>> print(embeddings.shape)
        (2, 384)
        >>> 
        >>> # Diffusion mode: text generation
        >>> result = engine.diffuse(
        ...     prompt="Explain quantum computing",
        ...     modality="text",
        ...     num_steps=4
        ... )
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_batch_size: int = 32,
        normalize_embeddings: bool = True,
        enable_intel_optimizations: bool = True,
        lazy_load: bool = True,
    ):
        """
        Initialize ArrowEngine.
        
        Args:
            model_path: Path to converted model directory
            device: Device for inference ("cpu", "cuda", "mps", or None for auto)
            max_batch_size: Maximum batch size for inference
            normalize_embeddings: L2-normalize embeddings by default
            enable_intel_optimizations: Enable Intel CPU optimizations (MKL, threading)
            lazy_load: Load weights on-demand to reduce startup time
        """
        self.model_path = Path(model_path)
        self.max_batch_size = max_batch_size
        self.normalize_embeddings = normalize_embeddings
        self.lazy_load = lazy_load
        
        if not self.model_path.exists():
            raise ModelLoadError(
                f"Model path not found: {model_path}",
                context={"model_path": str(model_path)}
            )
        
        # Auto-detect device
        self.device = device or self._auto_detect_device()
        
        # Apply Intel CPU optimizations
        if enable_intel_optimizations and self.device == "cpu":
            self._apply_intel_optimizations()
        
        start_time = time.time()
        
        # Load model components
        self._load_metadata()
        self._load_weights()
        self._load_tokenizer()
        self._initialize_inference_core()
        
        load_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"ArrowEngine loaded in {load_time_ms:.2f}ms")
        logger.info(f"Model: {self.metadata.get('model_name', 'unknown')}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Mode: Dual (AR + Diffusion)")

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """
        Encode sentences to embeddings (Autoregressive mode).
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for processing (default: self.max_batch_size)
            show_progress: Show progress bar for large batches
            normalize: L2-normalize embeddings (default: self.normalize_embeddings)
            
        Returns:
            np.ndarray: Embeddings of shape (N, embedding_dim)
            
        Example:
            >>> embeddings = engine.encode(["Hello", "World"])
            >>> print(embeddings.shape)
            (2, 384)
        """
        # Handle single sentence
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]
        
        # Use default batch size if not specified
        batch_size = batch_size or self.max_batch_size
        normalize = normalize if normalize is not None else self.normalize_embeddings
        
        all_embeddings = []
        
        # Process in batches
        num_batches = (len(sentences) + batch_size - 1) // batch_size
        iterator = range(num_batches)
        
        if show_progress and num_batches > 1:
            iterator = tqdm(iterator, desc="Encoding", total=num_batches)
        
        for i in iterator:
            batch_start = i * batch_size
            batch_end = min((i + 1) * batch_size, len(sentences))
            batch_sentences = sentences[batch_start:batch_end]
            
            # Tokenize
            encoded = self.tokenizer.encode(
                batch_sentences,
                add_special_tokens=True,
                return_tensors="pt"
            )
            
            input_ids = encoded['input_ids']
            attention_mask = encoded['attention_mask']
            
            # Forward pass
            with torch.no_grad():
                embeddings = self.inference_core(input_ids, attention_mask)
            
            # Normalize if requested
            if normalize:
                embeddings = self.inference_core.normalize_embeddings(embeddings)
            
            # Convert to numpy
            embeddings_np = embeddings.cpu().numpy()
            all_embeddings.append(embeddings_np)
        
        # Concatenate all batches
        all_embeddings = np.vstack(all_embeddings)
        
        # Return single embedding if input was single sentence
        if is_single:
            return all_embeddings[0]
        
        return all_embeddings

    def diffuse(
        self,
        prompt: Union[str, List[str]],
        modality: str = "text",
        num_steps: int = 4,
        guidance_scale: float = 7.5,
        memory_guided: bool = False,
        **kwargs
    ) -> Union[str, np.ndarray, Dict]:
        """
        Generate content using diffusion (Diffusion mode).
        
        Args:
            prompt: Text prompt or list of prompts
            modality: Output modality ("text", "image", "audio")
            num_steps: Number of diffusion steps (4 for distilled, 50 for full)
            guidance_scale: Classifier-free guidance scale
            memory_guided: Use memory conditioning from ArrowStorage
            **kwargs: Additional modality-specific parameters
            
        Returns:
            Generated content (format depends on modality):
            - text: str or List[str]
            - image: np.ndarray of shape (H, W, 3)
            - audio: np.ndarray of shape (samples,)
            
        Example:
            >>> # Text generation
            >>> text = engine.diffuse(
            ...     prompt="Explain AI",
            ...     modality="text",
            ...     num_steps=4
            ... )
            >>> 
            >>> # Image generation
            >>> image = engine.diffuse(
            ...     prompt="A sunset",
            ...     modality="image",
            ...     num_steps=50
            ... )
        """
        # TODO: Implement after creating DiffusionCore
        raise NotImplementedError(
            "diffuse() will be implemented in Phase 1 after creating DiffusionCore"
        )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text autoregressively (legacy AR mode for decoder models).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            
        Returns:
            str: Generated text
            
        Note:
            This is the legacy AR generation method.
            For diffusion-based generation, use diffuse() instead.
        """
        # TODO: Implement after migrating inference_core
        raise NotImplementedError(
            "generate() will be implemented after migrating inference_core.py"
        )

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension from metadata."""
        return self.metadata.get('embedding_dimension', 384)

    def _auto_detect_device(self) -> str:
        """Auto-detect best available device."""
        return get_best_device()
    
    def _apply_intel_optimizations(self):
        """Apply Intel CPU optimizations for better performance."""
        import os
        
        try:
            import psutil
            physical_cores = psutil.cpu_count(logical=False) or 4
        except ImportError:
            physical_cores = 4
        
        # Set optimal thread count
        try:
            torch.set_num_threads(physical_cores)
            torch.set_num_interop_threads(2)
        except RuntimeError:
            logger.warning("Could not set thread counts (already initialized)")
        
        # Enable MKL-DNN optimizations
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        # Set MKL environment variables
        os.environ['MKL_NUM_THREADS'] = str(physical_cores)
        os.environ['OMP_NUM_THREADS'] = str(physical_cores)
        os.environ['KMP_BLOCKTIME'] = '1'
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        
        logger.info(f"Intel CPU optimizations enabled (threads={physical_cores})")
    
    def _load_metadata(self):
        """Load model metadata from metadata.json."""
        if self.model_path.is_file():
            metadata_path = self.model_path.parent / "metadata.json"
        else:
            metadata_path = self.model_path / "metadata.json"
        
        # Log device info
        info = get_device_info(self.device)
        logger.info(f"Initialized on {info.get('name', self.device)} ({self.device})")
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found: {metadata_path}, using defaults")
            self.metadata = {
                'model_name': 'unknown',
                'embedding_dimension': 384,
                'architecture': 'unified-diffusion'
            }
            return
        
        try:
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.debug(f"Loaded metadata: {len(self.metadata)} fields")
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load metadata: {e}",
                context={"metadata_path": str(metadata_path)},
                original_exception=e
            )

    def _load_weights(self):
        """Load model weights using WeightLoader."""
        if self.model_path.is_file():
            weights_path = self.model_path
        else:
            weights_path = self.model_path / "weights.parquet"
        
        if not weights_path.exists():
            raise ModelLoadError(
                f"Weights file not found: {weights_path}",
                context={"weights_path": str(weights_path)}
            )
        
        try:
            self.weight_loader = WeightLoader(
                parquet_path=str(weights_path),
                use_memory_map=True,
                device=self.device,
                cache_weights=not self.lazy_load,
                force_float32=True,
            )
            
            # Load weights (lazy or eager based on config)
            if self.lazy_load:
                self.weights = LazyWeightDict(self.weight_loader)
                logger.info("Weights loaded lazily (on-demand)")
            else:
                self.weights = self.weight_loader.load_weights()
                logger.info(f"Weights loaded eagerly ({len(self.weights)} layers)")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load weights: {e}",
                context={"weights_path": str(weights_path)},
                original_exception=e
            )
    
    def _load_tokenizer(self):
        """Load tokenizer using FastTokenizer."""
        if self.model_path.is_file():
            tokenizer_path = self.model_path.parent
        else:
            tokenizer_path = self.model_path
        
        tokenizer_file = tokenizer_path / "tokenizer.json"
        
        if not tokenizer_file.exists():
            raise ModelLoadError(
                f"Tokenizer file not found: {tokenizer_file}",
                context={"tokenizer_path": str(tokenizer_path)}
            )
        
        try:
            max_length = self.metadata.get('max_position_embeddings', 512)
            
            self.tokenizer = FastTokenizer(
                tokenizer_path=str(tokenizer_path),
                max_length=max_length,
                padding="max_length",
                truncation=True,
            )
            
            logger.info(f"Tokenizer loaded (vocab_size={self.tokenizer.get_vocab_size()})")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to load tokenizer: {e}",
                context={"tokenizer_path": str(tokenizer_path)},
                original_exception=e
            )
    
    def _initialize_inference_core(self):
        """Initialize InferenceCore with loaded weights and config."""
        try:
            # Build config from metadata
            config = {
                'hidden_size': self.metadata.get('hidden_size', 384),
                'num_layers': self.metadata.get('num_layers', 6),
                'num_attention_heads': self.metadata.get('num_attention_heads', 6),
                'intermediate_size': self.metadata.get('intermediate_size', 1536),
                'max_position_embeddings': self.metadata.get('max_position_embeddings', 512),
                'vocab_size': self.metadata.get('vocab_size', 30522),
                'layer_norm_eps': self.metadata.get('layer_norm_eps', 1e-12),
                'architecture': self.metadata.get('architecture', ''),
                'rope_theta': self.metadata.get('rope_theta'),
                'num_key_value_heads': self.metadata.get('num_key_value_heads'),
                'rms_norm_eps': self.metadata.get('rms_norm_eps', 1e-6),
            }
            
            self.inference_core = InferenceCore(
                weights=self.weights,
                config=config,
                device=self.device,
            )
            
            logger.info(f"InferenceCore initialized (dim={config['hidden_size']})")
            
        except Exception as e:
            raise ModelLoadError(
                f"Failed to initialize inference core: {e}",
                context={"config": config},
                original_exception=e
            )
