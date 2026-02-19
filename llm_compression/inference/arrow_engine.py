"""
ArrowEngine - High-performance embedding inference engine.

Main API class that integrates all components for fast model loading
and inference using Arrow/Parquet storage and Rust tokenizers.

Example:
    >>> from llm_compression.inference import ArrowEngine
    >>> 
    >>> engine = ArrowEngine("./models/minilm")  # < 100ms
    >>> embeddings = engine.encode(["Hello, world!"])
    >>> print(embeddings.shape)
    (1, 384)
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.inference.fast_tokenizer import FastTokenizer
from llm_compression.inference.inference_core import InferenceCore
from llm_compression.inference.device_utils import get_best_device, get_device_info
from llm_compression.inference.intel_opt import optimize_for_intel
from llm_compression.logger import logger


class ArrowEngine:
    """
    High-performance embedding inference engine using Arrow/Parquet storage.
    
    Combines zero-copy weight loading, Rust tokenization, and optimized
    PyTorch inference for 10-100x performance improvements over traditional
    approaches.
    
    Performance Targets:
    - Model load time: < 100ms (vs 2-5s traditional)
    - Inference latency: < 5ms per sequence
    - Throughput: > 2000 sequences/s (batch_size=32)
    - Memory usage: < 50% of original model size
    
    Features:
    - Zero-copy weight loading from Parquet
    - Fast Rust tokenization (20x speedup)
    - Batch processing with progress bars
    - Multi-device support (CPU/CUDA/MPS)
    - L2 normalization for similarity search
    
    Example:
        >>> engine = ArrowEngine("./models/minilm")
        >>> 
        >>> embeddings = engine.encode([
        ...     "Artificial intelligence",
        ...     "Machine learning"
        ... ])
        >>> print(embeddings.shape)
        (2, 384)
        >>> 
        >>> similarity = np.dot(embeddings[0], embeddings[1])
        >>> print(f"Similarity: {similarity:.3f}")
        Similarity: 0.856
    """
    
    def __init__(
        self,
        model_path: str,
        device: Optional[str] = None,
        max_batch_size: int = 32,
        normalize_embeddings: bool = True,
        enable_intel_optimizations: bool = True,
    ):
        """
        Initialize ArrowEngine.
        
        Args:
            model_path: Path to converted model directory
            device: Device for inference ("cpu", "cuda", "mps", or None for auto)
            max_batch_size: Maximum batch size for inference
            normalize_embeddings: L2-normalize embeddings by default
            enable_intel_optimizations: Enable Intel CPU optimizations (MKL, threading)
        """
        self.model_path = Path(model_path)
        self.max_batch_size = max_batch_size
        self.normalize_embeddings = normalize_embeddings
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        self.device = device or self._auto_detect_device()
        
        # Apply Intel CPU optimizations
        if enable_intel_optimizations and self.device == "cpu":
            self._apply_intel_optimizations()
        
        start_time = time.time()
        
        self._load_metadata()
        self._load_weights()
        self._load_tokenizer()
        self._initialize_inference_core()
        
        load_time_ms = (time.time() - start_time) * 1000
        
        # LoRA Auto-Router
        self.lora_router = None
        if hasattr(self.inference_core, 'lora_manager') and self.inference_core.lora_manager:
            try:
                from llm_compression.inference.lora_router import LoRARouter
                # Use self.encode (bound method) as the embedder for the router
                # Warning: encode -> router -> encode loop?
                # No, router uses embedder only during registration (offline) or selection (online).
                # During selection, router calls embedder. embedding is base model.
                self.lora_router = LoRARouter(
                    self.inference_core.lora_manager,
                    embedder_func=lambda text: self.encode(text, normalize=True)[0]
                )
                logger.info("LoRA Router initialized.")
            except ImportError:
                pass

        self.federation = None
        self.distiller = None
        self.sensors = None
        self._evolution_thread = None
        self.last_used_lora = None
        self.last_router_confidence = 0.0

        logger.info(f"ArrowEngine loaded in {load_time_ms:.2f}ms")
        logger.info(f"Model: {self.metadata.get('model_name', 'unknown')}")
        logger.info(f"Embedding dimension: {self.get_embedding_dimension()}")
        logger.info(f"Device: {self.device}")

    def register_lora(self, path: str):
        """Register a LoRA for auto-routing."""
        if not hasattr(self, 'lora_registry'):
             self.lora_registry = {}
             
        try:
            # We need to import LoRAFormat here to load
            from llm_compression.inference.lora_format import LoRAFormat
            card = LoRAFormat.load(path)
            self.lora_registry[card.name] = card
            
            if self.lora_router:
                self.lora_router.register_card(card)
            logger.info(f"Registered LoRA: {card.name}")
        except Exception as e:
            logger.error(f"Failed to register LoRA from {path}: {e}")

    def encode_with_lora(
        self,
        sentences: Union[str, List[str]],
        intent_query: Optional[str] = None,
        **kwargs
    ):
        """
        Encode with automatic LoRA selection based on intent.
        
        If intent_query is provided, Router selects best LoRA.
        Similar query is used to select the LoRA, then `sentences` are encoded with it.
        """
        if not intent_query or not self.lora_router:
            return self.encode(sentences, **kwargs)
            
        # 1. Select LoRA — get scores to detect confidence level
        selected_names = self.lora_router.select(intent_query, top_k=1)
        top_score = self._get_router_confidence(intent_query)
        self.last_router_confidence = top_score
        
        active_name = None
        
        if selected_names:
            active_name = selected_names[0]
            card = getattr(self, 'lora_registry', {}).get(active_name)
            
            # Check Remote if not found locally
            if not card and self.federation:
                try:
                    logger.info(f"Auto-fetching remote skill: {active_name}")
                    path = self.federation.fetch_skill(active_name)
                    if path:
                        from llm_compression.inference.lora_format import LoRAFormat
                        card = LoRAFormat.load(str(path))
                        # Register locally
                        if not hasattr(self, 'lora_registry'):
                            self.lora_registry = {}
                        self.lora_registry[card.name] = card
                except Exception as e:
                    logger.error(f"Failed to fetch remote skill {active_name}: {e}")

            if card and self.inference_core.lora_manager:
                # Hot-swap Apply
                self.inference_core.lora_manager.apply_card(card)
                self.last_used_lora = active_name
            else:
                active_name = None # Failed to load or find
        
        # Cognitive Dissonance Detection:
        # If confidence is below threshold, trigger background evolution
        if self.distiller and top_score < self.distiller._confidence_threshold:
            self._trigger_evolution(intent_query, top_score)

        try:
            # 2. Encode
            return self.encode(sentences, **kwargs)
        finally:
            # 3. Restore (Unload)
            if active_name and self.inference_core.lora_manager:
                self.inference_core.lora_manager.remove_card(active_name)
    
    def start_federation(self, port: int = 9000, node_name: str = "ai-os-node"):
        """Initialize Federation Manager."""
        try:
            from llm_compression.federation import FederationManager
            lora_storage = self.model_path / "lora_skills"
            self.federation = FederationManager(node_name, port, str(lora_storage))
            self.federation.start()
            logger.info(f"Federation started on port {port}")
        except Exception as e:
             logger.error(f"Failed to start federation: {e}")

    def sync_remote_skills(self):
        """Sync skills from federation to router."""
        if not self.federation or not self.lora_router:
            return
            
        try:
            remote_skills = self.federation.scan_remote_skills()
            for name, info in remote_skills.items():
                if hasattr(self, 'lora_registry') and name in self.lora_registry:
                    continue
                # Register virtual candidate
                self.lora_router.register_virtual_candidate(name, name)
                
            logger.info(f"Synced {len(remote_skills)} remote skills to router.")
        except Exception as e:
            logger.error(f"Federation sync failed: {e}")

    def enable_evolution(
        self,
        cloud_providers: Optional[Dict] = None,
        confidence_threshold: float = 0.3,
        rank: int = 8,
    ):
        """
        Enable self-evolution capability.
        
        When enabled, the engine will automatically detect cognitive
        dissonance (low router confidence) and trigger background
        learning to acquire new skills.
        
        Args:
            cloud_providers: Dict of {name: CloudProvider} for Tier 2 learning.
            confidence_threshold: Router confidence below this triggers evolution.
            rank: LoRA rank for extracted skills.
        """
        try:
            from llm_compression.evolution.skill_distiller import SkillDistiller, NodeTier
            
            lora_dir = self.model_path / "lora_skills"
            self.distiller = SkillDistiller(
                engine=self,
                node_tier=NodeTier.HUB,
                lora_output_dir=str(lora_dir),
                cloud_providers=cloud_providers or {},
                rank=rank,
            )
            self.distiller._confidence_threshold = confidence_threshold
            
            logger.info(
                f"Self-evolution enabled: threshold={confidence_threshold}, "
                f"rank={rank}, output={lora_dir}"
            )
        except Exception as e:
            logger.error(f"Failed to enable evolution: {e}")

    def enable_actions(self, workspace_dir: Optional[str] = None):
        """Initialize Embodied Action Manager."""
        try:
            from llm_compression.action.manager import ActionManager
            ws = workspace_dir or str(self.model_path / "actions")
            sensor_link = getattr(self, "sensors", None)
            self.actions = ActionManager(ws, sensor_manager=sensor_link)
            logger.info(f"Embodied Actions enabled at {ws}")
        except Exception as e:
            logger.error(f"Action init failed: {e}")

    def enable_sensors(self, workspace_dir: Optional[str] = None, start_hardware: bool = False):
        """Initialize Sensor Manager."""
        try:
            from llm_compression.sensors.manager import SensorManager
            ws = workspace_dir or str(self.model_path / "sensors")
            self.sensors = SensorManager(ws)
            if start_hardware:
                self.sensors.start_hardware()
            logger.info(f"Sensors enabled at {ws} (Hardware={'ON' if start_hardware else 'OFF'})")
        except ImportError:
            logger.warning("Sensor module not found.")
    
    def _get_router_confidence(self, query: str) -> float:
        """Get the top confidence score from the router for a query."""
        if not self.lora_router or not self.lora_router.index:
            return 0.0
            
        query_vec = self.lora_router.embedder(query)
        best_score = 0.0
        
        for name, doc_vec in self.lora_router.index.items():
            import numpy as np
            norm_q = np.linalg.norm(query_vec)
            norm_d = np.linalg.norm(doc_vec)
            if norm_q > 0 and norm_d > 0:
                sim = float(np.dot(query_vec, doc_vec) / (norm_q * norm_d))
                best_score = max(best_score, sim)
        
        return best_score
    
    def _trigger_evolution(self, query: str, confidence: float):
        """
        Trigger background evolution when cognitive dissonance is detected.
        
        Currently uses Tier 3 (weight extraction from local model).
        If cloud providers are configured, also tries Tier 2.
        """
        import threading
        
        if self._evolution_thread and self._evolution_thread.is_alive():
            logger.debug("Evolution already in progress, skipping.")
            return
        
        logger.info(
            f"Cognitive dissonance detected! confidence={confidence:.3f} "
            f"Query: '{query[:80]}...'"
        )
        
        def _evolve():
            try:
                # Generate a skill name from the query
                skill_name = query.lower()[:40].replace(" ", "_")
                skill_name = "".join(c for c in skill_name if c.isalnum() or c == "_")
                skill_name = f"evolved_{skill_name}_v1"
                
                # Try Tier 2 first if cloud providers available
                if self.distiller.cloud_providers:
                    from llm_compression.evolution.cloud_distiller import CloudDistiller
                    for pname, provider in self.distiller.cloud_providers.items():
                        cd = CloudDistiller(
                            engine=self,
                            output_dir=str(self.distiller.output_dir),
                            rank=self.distiller.extractor.rank,
                        )
                        card = cd.distill_topic(
                            topic=query,
                            provider=provider,
                            num_pairs=10,
                            skill_name=skill_name,
                        )
                        if card and len(card.weights_A) > 0:
                            self.register_lora(
                                str(self.distiller.output_dir / f"{skill_name}.lora.arrow")
                            )
                            logger.info(f"Evolution SUCCESS (Tier 2): {skill_name}")
                            return
                
                # Fallback: Tier 3 — extract from local model weights
                card = self.distiller.extract_skill_from_engine(
                    name=skill_name,
                    test_queries=[query],
                    description=f"Auto-evolved skill for: {query[:100]}",
                )
                
                if card and len(card.weights_A) > 0:
                    self.register_lora(
                        str(self.distiller.output_dir / f"{skill_name}.lora.arrow")
                    )
                    logger.info(f"Evolution SUCCESS (Tier 3): {skill_name}")
                else:
                    logger.warning(f"Evolution produced empty skill for: {query[:60]}")
                    
            except Exception as e:
                logger.error(f"Background evolution failed: {e}")
        
        self._evolution_thread = threading.Thread(
            target=_evolve, daemon=True, name="ai-os-evolution"
        )
        self._evolution_thread.start()

    def _auto_detect_device(self) -> str:
        """Auto-detect best available device."""
        return get_best_device()
    
    def _apply_intel_optimizations(self):
        """Apply Intel CPU optimizations for better performance."""
        import os
        
        try:
            import psutil
            # Get physical core count
            physical_cores = psutil.cpu_count(logical=False) or 4
        except ImportError:
            # Fallback if psutil not available
            physical_cores = 4
        
        # Set optimal thread count
        # Rule: Use physical cores for intra-op, 2 threads for inter-op
        # Note: Must be called before any parallel work starts
        try:
            torch.set_num_threads(physical_cores)
        except RuntimeError:
            logger.warning("Could not set intra-op threads (already initialized)")
        
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            logger.warning("Could not set inter-op threads (already initialized)")
        
        # Enable MKL-DNN (oneDNN) optimizations
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        
        # Set MKL environment variables for optimal performance
        os.environ['MKL_NUM_THREADS'] = str(physical_cores)
        os.environ['OMP_NUM_THREADS'] = str(physical_cores)
        os.environ['KMP_BLOCKTIME'] = '1'  # Low latency
        os.environ['KMP_AFFINITY'] = 'granularity=fine,compact,1,0'
        
        logger.info(f"Intel CPU optimizations enabled:")
        logger.info(f"  - Intra-op threads: {physical_cores}")
        logger.info(f"  - Inter-op threads: 2")
        logger.info(f"  - MKL-DNN: {torch.backends.mkldnn.is_available()}")
    
    def _load_metadata(self):
        """Load model metadata from metadata.json."""
        metadata_path = self.model_path / "metadata.json"
        
        info = get_device_info(self.device)
        logger.info(f"Initialized on {info.get('name', self.device)} ({self.device})")
        
        if not metadata_path.exists():
            logger.warning(f"Metadata not found: {metadata_path}")
            self.metadata = {}
            return
        
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        logger.debug(f"Loaded metadata: {len(self.metadata)} fields")
    
    def _load_weights(self):
        """Load model weights using WeightLoader."""
        parquet_path = self.model_path / "weights.parquet"
        
        if not parquet_path.exists():
            raise FileNotFoundError(f"Weights not found: {parquet_path}")
        
        self.weight_loader = WeightLoader(
            parquet_path=str(parquet_path),
            device=self.device,
            cache_weights=True,
        )
        
        self.weights = self.weight_loader.load_weights()
        
        logger.info(f"Loaded {len(self.weights)} weight tensors")
    
    def _load_tokenizer(self):
        """Load tokenizer using FastTokenizer."""
        tokenizer_path = self.model_path / "tokenizer"
        
        if not tokenizer_path.exists():
            tokenizer_path = self.model_path / "tokenizer.json"
        
        if not tokenizer_path.exists():
            raise FileNotFoundError(f"Tokenizer not found in {self.model_path}")
        
        max_length = self.metadata.get('model_info', {}).get('max_seq_length', 512)
        
        self.tokenizer = FastTokenizer(
            tokenizer_path=str(tokenizer_path),
            max_length=max_length,
        )
        
        logger.info(f"Loaded tokenizer with max_length={max_length}")
    
    def _initialize_inference_core(self):
        """Initialize inference core from metadata + weight shapes."""
        model_info = self.metadata.get('model_info', {})
        
        # === Prefer metadata values (saved by ModelConverter from model.config) ===
        # Fall back to weight-shape detection when metadata is absent (legacy models)
        
        hidden_size = (
            model_info.get('hidden_size') or
            model_info.get('embedding_dimension') or
            384
        )
        
        # num_layers: count unique encoder.layer.N indices from weight keys
        layer_indices = set()
        for key in self.weights:
            if key.startswith("encoder.layer."):
                parts = key.split(".")
                if len(parts) > 2:
                    try:
                        layer_indices.add(int(parts[2]))
                    except ValueError:
                        pass
        num_layers = (
            len(layer_indices) if layer_indices else
            model_info.get('num_hidden_layers') or
            6
        )
        
        # num_attention_heads: read from metadata first (most reliable)
        num_attention_heads = model_info.get('num_attention_heads')
        if not num_attention_heads:
            # Fallback: can't reliably infer from weight shape alone since
            # all_head_size == hidden_size for standard BERT. Use common default.
            num_attention_heads = 12  # Most BERT models use 12 heads
            logger.warning(
                "num_attention_heads not found in metadata, defaulting to 12. "
                "Re-convert model with updated ModelConverter for accurate config."
            )
        
        # intermediate_size: from metadata or weight shape
        intermediate_size = model_info.get('intermediate_size')
        if not intermediate_size:
            ffn_key = "encoder.layer.0.intermediate.dense.weight"
            if ffn_key in self.weights:
                intermediate_size = self.weights[ffn_key].shape[0]
            else:
                intermediate_size = hidden_size * 4
        
        # max_position_embeddings: from metadata or weight shape
        max_position = model_info.get('max_position_embeddings')
        if not max_position:
            pos_key = "embeddings.position_embeddings.weight"
            if pos_key in self.weights:
                max_position = self.weights[pos_key].shape[0]
            else:
                max_position = 512
        
        # vocab_size: from metadata or weight shape
        vocab_size = model_info.get('vocab_size')
        if not vocab_size:
            vocab_key = "embeddings.word_embeddings.weight"
            if vocab_key in self.weights:
                vocab_size = self.weights[vocab_key].shape[0]
            else:
                vocab_size = 30522
        
        layer_norm_eps = model_info.get('layer_norm_eps', 1e-12)
        
        config = {
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'num_attention_heads': num_attention_heads,
            'intermediate_size': intermediate_size,
            'max_position_embeddings': max_position,
            'vocab_size': vocab_size,
            'layer_norm_eps': layer_norm_eps,
        }
        
        self.inference_core = InferenceCore(
            weights=self.weights,
            config=config,
            device=self.device,
        )
        
        # Apply Intel/Hardware specific optimizations
        dtype = torch.float16 if model_info.get('use_float16') else torch.float32
        self.inference_core = optimize_for_intel(self.inference_core, dtype=dtype)
        
        logger.info(
            f"InferenceCore: hidden={hidden_size}, layers={num_layers}, "
            f"heads={num_attention_heads}, intermediate={intermediate_size}"
        )
    
    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: Optional[int] = None,
        show_progress: bool = False,
        normalize: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, List[Tuple[torch.Tensor, ...]]]]:
        """
        Encode sentences to embeddings.
        
        Args:
            sentences: Single sentence or list of sentences
            batch_size: Batch size for processing (default: max_batch_size)
            show_progress: Show progress bar for large batches
            normalize: L2-normalize embeddings (default: self.normalize_embeddings)
            output_attentions: Whether to output attention weights
            
        Returns:
            If output_attentions is False:
                Embeddings as numpy array, shape (n_sentences, embedding_dim)
            If output_attentions is True:
                Tuple of (embeddings, list_of_attentions_per_batch)
                
        Example:
            >>> engine = ArrowEngine("./models/minilm")
            >>> 
            >>> emb = engine.encode("Hello, world!")
            >>> print(emb.shape)
            (1, 384)
            >>> 
            >>> embs = engine.encode([
            ...     "First sentence",
            ...     "Second sentence"
            ... ])
            >>> print(embs.shape)
            (2, 384)
        """
        is_single = isinstance(sentences, str)
        if is_single:
            sentences = [sentences]
        
        batch_size = batch_size or self.max_batch_size
        normalize = normalize if normalize is not None else self.normalize_embeddings
        
        all_embeddings = []
        all_attentions = []
        
        iterator = range(0, len(sentences), batch_size)
        if show_progress and len(sentences) > batch_size:
            iterator = tqdm(iterator, desc="Encoding", total=len(sentences)//batch_size + 1)
        
        for i in iterator:
            batch_sentences = sentences[i:i + batch_size]
            result = self._encode_batch(
                batch_sentences, 
                normalize=normalize,
                output_attentions=output_attentions
            )
            
            if output_attentions:
                embeddings, attentions = result
                all_embeddings.append(embeddings)
                all_attentions.append(attentions)
            else:
                all_embeddings.append(result)
        
        embeddings = np.vstack(all_embeddings)
        
        if is_single:
            embeddings = embeddings[0:1]
            if output_attentions:
                # Return attentions for the single item (might be batched though)
                pass
        
        if output_attentions:
            return embeddings, all_attentions
            
        return embeddings
    
    def _encode_batch(
        self,
        sentences: List[str],
        normalize: bool = True,
        output_attentions: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[torch.Tensor, ...]]]:
        """
        Encode a batch of sentences.
        
        Args:
            sentences: List of sentences
            normalize: L2-normalize embeddings
            output_attentions: Whether to output attention weights
            
        Returns:
            If output_attentions is False:
                Embeddings as numpy array, shape (batch_size, embedding_dim)
            If output_attentions is True:
                Tuple of (embeddings, attentions)
        """
        encoded = self.tokenizer.encode(sentences)
        
        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        # Pre-allocate inputs on device
        input_ids = torch.as_tensor(input_ids).to(self.device).long()
        attention_mask = torch.as_tensor(attention_mask).to(self.device).long()

        with torch.no_grad():
            outputs = self.inference_core(
                input_ids, 
                attention_mask,
                output_attentions=output_attentions
            )
            
            if output_attentions:
                embeddings, attentions = outputs
            else:
                embeddings = outputs
                attentions = None
            
            if normalize:
                embeddings = self.inference_core.normalize_embeddings(embeddings)
            
            embeddings = embeddings.cpu().numpy()
            
            if output_attentions:
                # Return attentions as tuple of tensors on CPU
                cpu_attentions = tuple(att.cpu() for att in attentions)
                return embeddings, cpu_attentions
        
        return embeddings
    
    def encode_batch(
        self,
        sentences: List[str],
        normalize: Optional[bool] = None,
        output_attentions: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[torch.Tensor, ...]]]:
        """
        Encode a batch of sentences (optimized for throughput).
        
        Args:
            sentences: List of sentences (up to max_batch_size)
            normalize: L2-normalize embeddings
            output_attentions: Whether to output attention weights
            
        Returns:
            Embeddings as numpy array, shape (batch_size, embedding_dim)
        """
        if len(sentences) > self.max_batch_size:
            logger.warning(
                f"Batch size {len(sentences)} exceeds max_batch_size {self.max_batch_size}. "
                f"Consider using encode() with automatic batching."
            )
        
        normalize = normalize if normalize is not None else self.normalize_embeddings
        return self._encode_batch(sentences, normalize=normalize, output_attentions=output_attentions)
    
    def get_embedding_dimension(self) -> int:
        """
        Get embedding dimension.
        
        Returns:
            Embedding dimension (e.g., 384 for all-MiniLM-L6-v2)
        """
        return self.inference_core.get_embedding_dimension()
    
    def get_max_seq_length(self) -> int:
        """
        Get maximum sequence length.
        
        Returns:
            Maximum sequence length (e.g., 512)
        """
        return self.tokenizer.max_length
    
    def similarity(
        self,
        sentences1: Union[str, List[str]],
        sentences2: Union[str, List[str]],
    ) -> np.ndarray:
        """
        Compute cosine similarity between two sets of sentences.
        
        Args:
            sentences1: First sentence(s)
            sentences2: Second sentence(s)
            
        Returns:
            Similarity matrix, shape (len(sentences1), len(sentences2))
            
        Example:
            >>> engine = ArrowEngine("./models/minilm")
            >>> 
            >>> sim = engine.similarity(
            ...     "Machine learning",
            ...     "Artificial intelligence"
            ... )
            >>> print(f"Similarity: {sim[0, 0]:.3f}")
            Similarity: 0.856
        """
        emb1 = self.encode(sentences1, normalize=True)
        emb2 = self.encode(sentences2, normalize=True)
        
        similarity_matrix = np.dot(emb1, emb2.T)
        
        return similarity_matrix
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ArrowEngine("
            f"model={self.model_path.name}, "
            f"dim={self.get_embedding_dimension()}, "
            f"device={self.device}, "
            f"batch_size={self.max_batch_size})"
        )
