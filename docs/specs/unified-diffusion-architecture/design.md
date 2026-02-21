# Design Document: Unified Diffusion Architecture

## Overview

The Unified Diffusion Architecture replaces autoregressive (AR) models with thermodynamic diffusion models to enable unified multi-modal generation. The core principle is:

```
Unified Diffusion = Shared Score Network + Modality Projection Heads + Thermodynamic Samplers
```

All modality generation is unified as a single denoising process: starting from noise/[MASK] state, iteratively converging to the data distribution through learned score functions (log probability gradients), with modality-specific projection heads decoding to concrete outputs.

### Thermodynamic Correspondence

| Physical Concept | Architecture Mapping | Code Mapping |
|-----------------|---------------------|--------------|
| Thermodynamic Equilibrium | Pure noise N(0,I) / Full [MASK] | Sampling initial state |
| Langevin Dynamics | Continuous modality denoising | `ContinuousSampler.step()` |
| CTMC Jump Process | Discrete modality unmasking | `DiscreteSampler.step()` |
| Free Energy Minimization | Score Matching Loss | Training objective |
| Entropy Production Rate | Noise residual \|x_t - x̂_0\| | Uncertainty metric |

### Key Design Decisions

1. **Shared Backbone**: 90%+ parameters in SharedTransformer, <10% in modality heads
2. **Thermodynamic Framework**: Diffusion as physical process with measurable uncertainty
3. **Memory Integration**: ArrowStorage retrieval as conditioning vectors
4. **Progressive Evolution**: 5-level adaptation from score mixing to full fine-tuning
5. **Edge-First**: Consistency distillation for 4-step generation on mobile devices

## Architecture

### Global Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     User API Layer                        │
│  .encode()  .generate()  .diffuse()  .render_avatar()    │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                  ArrowEngine (Router)                     │
│  mode="ar"  → InferenceCore (existing AR path)           │
│  mode="diffusion" → DiffusionCore (new diffusion path)   │
└─────────────────────────┬────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────┐
│                DiffusionCore (New)                        │
│  ┌──────────────────────────────────────────────────┐    │
│  │  UnifiedScoreNetwork                              │    │
│  │  ┌──────────┐  ┌───────────┐  ┌───────────────┐  │    │
│  │  │Modality  │  │Sinusoidal │  │  Condition     │  │    │
│  │  │Embedding │  │Time Embed │  │  Encoder       │  │    │
│  │  └────┬─────┘  └─────┬─────┘  └──────┬────────┘  │    │
│  │       └──────────────┼───────────────┘            │    │
│  │                      ▼                            │    │
│  │           SharedTransformer (N layers)             │    │
│  │                      │                            │    │
│  │       ┌─────┬────────┼────────┬──────┐            │    │
│  │       ▼     ▼        ▼        ▼      ▼            │    │
│  │    TextH ImageH  AudioH  CodeH  AvatarH           │    │
│  └──────────────────────────────────────────────────┘    │
│  ┌──────────────────────────────────────────────────┐    │
│  │  EvolutionRouter (5-Level Self-Evolution)        │    │
│  │   ├── L0: Score Mixer                             │    │
│  │   ├── L1: ControlNet Bank                         │    │
│  │   ├── L2: LoRA Router                             │    │
│  │   └── L3: Selective Finetuner                     │    │
│  └──────────────────────────────────────────────────┘    │
│                          │                                │
│  ┌──────────┐  ┌────────▼─────────┐  ┌──────────────┐  │
│  │ Discrete │  │ NoiseScheduler   │  │ Continuous   │  │
│  │ Sampler  │  │ (Unified)        │  │ Sampler      │  │
│  │(text/code│  │ + Energy Models  │  │(image/audio) │  │
│  └──────────┘  └───────────────────┘  └──────────────┘  │
└──────────────────────────┬───────────────────────────────┘
                           │ weights
┌──────────────────────────▼───────────────────────────────┐
│  WeightLoader V2 + ArrowQuant (Parquet V2, zero-copy)   │
└──────────────────────────────────────────────────────────┘
```

### Integration with Existing Architecture

```
ArrowEngine
  ├── mode="ar" (existing, preserved)
  │     ├── InferenceCore
  │     ├── LoRA Router
  │     └── WeightLoader V1/V2
  │
  └── mode="diffusion" (new)
        ├── DiffusionCore (core of this design)
        ├── EvolutionRouter (L0-L4 progressive evolution)
        │     ├── ControlNet Bank (L1: behavior control)
        │     ├── LoRA Router (L2: knowledge injection)
        │     └── Selective Finetuner (L3: partial unfreezing)
        ├── EnergyModelValidator (post-processing & constraints)
        └── WeightLoader V2 (reused, with ArrowQuant)
```

## Components and Interfaces

### UnifiedScoreNetwork

The core neural network that learns score functions across all modalities.

```python
class UnifiedScoreNetwork(nn.Module):
    """
    Unified Score Network: all modalities share Transformer backbone.
    
    Parameter distribution:
      SharedTransformer: ~90% (backbone for "understanding")
      Projection Heads: ~10% (for "expression")
    """
    def __init__(self, config: DiffusionConfig):
        # Shared components
        self.modality_embed = nn.Embedding(5, config.hidden_dim)  
        self.time_embed = SinusoidalTimeEmbedding(config.hidden_dim)
        self.condition_proj = nn.Linear(config.condition_dim, config.hidden_dim)
        self.shared_transformer = TransformerStack(
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            intermediate_dim=config.intermediate_dim,
        )
        
        # Modality projection heads (each < 10M params)
        self.heads = nn.ModuleDict({
            "text": TextProjectionHead(config.hidden_dim, config.vocab_size),
            "code": TextProjectionHead(config.hidden_dim, config.vocab_size),
            "image": PatchProjectionHead(config.hidden_dim, patch_size=16, channels=4),
            "audio": WaveformProjectionHead(config.hidden_dim, hop_length=256),
        })
    
    def forward(self, x_t, t, modality, condition=None):
        """
        Args:
            x_t: Current noisy state (projected to hidden space by Head.encode)
            t: Timestep [0, T]
            modality: "text" | "code" | "image" | "audio"
            condition: Condition vector (from memory retrieval or CLIP)
        Returns:
            score or ε prediction (projected back to modality space by Head.decode)
        """
        # 1. Input Head: modality data → hidden space
        h = self.heads[modality].encode(x_t)
        
        # 2. Add time and modality embeddings
        h = h + self.time_embed(t) + self.modality_embed(MODALITY_IDS[modality])
        
        # 3. Condition injection (Cross-Attention)
        if condition is not None:
            condition = self.condition_proj(condition)
        
        # 4. Shared Transformer forward
        h = self.shared_transformer(h, condition=condition)
        
        # 5. Output Head: hidden space → modality space (score/ε/logits)
        base_score = self.heads[modality].decode(h)
        
        return base_score
```

**Interface:**
- `forward(x_t, t, modality, condition) -> score`: Main forward pass
- `encode(x, modality) -> h`: Modality-specific encoding
- `decode(h, modality) -> x`: Modality-specific decoding

**Key Properties:**
- Shared parameters: 90%+ in SharedTransformer
- Modality-specific: <10% in projection heads
- Extensible: Add new modality by training new head only

### EvolutionRouter

Progressive self-evolution hub that combines scores from multiple sources.

```python
class EvolutionRouter:
    """
    5-level progressive self-evolution hub.
    Overlays ControlNet, LoRA, and energy constraints on base score.
    """
    def __init__(self, score_network, control_nets, lora_manager, energy_models):
        self.score_net = score_network
        self.control_nets = control_nets  # L1
        self.loras = lora_manager         # L2
        self.energy_models = energy_models # EBM fusion
        
    def get_fused_score(self, x_t, t, modality, condition, active_profiles):
        """
        L0 Score Combination: ∇p_final = ∇p_base + α∇p_control - η∇E
        """
        # Base Score
        score = self.score_net(x_t, t, modality, condition)
        
        # L1: ControlNet behavior constraint injection
        for profile in active_profiles.control_nets:
            c_net = self.control_nets[profile.id]
            score += profile.weight * c_net(x_t, t, profile.condition_template)
            
        # EBM: Energy model constraint gradients
        for e_model in active_profiles.energy_models:
            energy_grad = torch.autograd.grad(e_model(x_t, t).sum(), x_t)[0]
            score -= e_model.weight * energy_grad
            
        return score
```

**Interface:**
- `get_fused_score(x_t, t, modality, condition, profiles) -> score`: Combined score
- `select_evolution_level(uncertainty) -> level`: Choose L0-L4 based on uncertainty
- `apply_evolution(level, data) -> adapted_model`: Execute evolution at selected level

**Evolution Levels:**
- L0: Score mixing (real-time, no training)
- L1: ControlNet (10% params, structural constraints)
- L2: LoRA (1% params, domain knowledge)
- L3: Selective unfreezing (uncertainty-driven layers)
- L4: Full fine-tuning (long-term consolidation)

### DiffusionCore

Main inference engine managing the denoising loop.

```python
class DiffusionCore:
    """
    Diffusion inference core: manages denoising loop.
    
    Equivalent to InferenceCore for diffusion paradigm.
    """
    def __init__(self, score_network, scheduler, config):
        self.score_net = score_network
        self.scheduler = scheduler    # NoiseScheduler
        self.config = config
    
    def generate(self, condition, modality, num_steps=4):
        """
        Unified generation entry point.
        
        1. Initialize noise state
        2. Iterative denoising
        3. Projection head decoding
        """
        # Initialize
        if modality in ("text", "code"):
            x_t = self._init_masked_sequence(condition)  # Full [MASK]
            sampler = DiscreteSampler(self.scheduler)
        else:
            x_t = torch.randn(self._get_latent_shape(modality))  # Gaussian noise
            sampler = ContinuousSampler(self.scheduler)
        
        # Denoising loop
        for t in self.scheduler.timesteps(num_steps):
            score = self.score_net(x_t, t, modality, condition)
            x_t = sampler.step(score, t, x_t)
        
        return x_t  # Final result
```

**Interface:**
- `generate(condition, modality, num_steps) -> output`: Main generation method
- `_init_masked_sequence(condition) -> x_t`: Initialize discrete modality
- `_get_latent_shape(modality) -> shape`: Get continuous modality shape

### NoiseScheduler

Unified noise scheduler for both discrete and continuous modalities.

```python
class NoiseScheduler:
    """
    Unified noise scheduler supporting discrete and continuous modes.
    
    Discrete mode (text/code):  β(t) = mask probability
    Continuous mode (image/audio):  σ(t) = noise standard deviation
    """
    def __init__(self, schedule_type="cosine", num_train_steps=1000):
        self.schedule_type = schedule_type
        self.num_train_steps = num_train_steps
    
    def timesteps(self, num_inference_steps):
        """Return inference timestep sequence (uniform or non-uniform sampling)"""
        if self.schedule_type == "cosine":
            return self._cosine_schedule(num_inference_steps)
        elif self.schedule_type == "linear":
            return self._linear_schedule(num_inference_steps)
        else:
            return self._custom_schedule(num_inference_steps)
    
    def add_noise(self, x_0, t, mode="continuous"):
        """Forward noising"""
        if mode == "discrete":
            return self._mask_tokens(x_0, t)
        else:
            return x_0 + self.sigma(t) * torch.randn_like(x_0)
    
    def sigma(self, t):
        """Continuous noise standard deviation σ(t)"""
        if self.schedule_type == "cosine":
            return self._cosine_sigma(t)
        return self._linear_sigma(t)
    
    def mask_rate(self, t):
        """Discrete mask probability β(t)"""
        if self.schedule_type == "cosine":
            return self._cosine_mask_rate(t)
        return self._linear_mask_rate(t)
```

**Interface:**
- `timesteps(num_steps) -> [t_0, ..., t_N]`: Generate timestep sequence
- `add_noise(x_0, t, mode) -> x_t`: Forward noising process
- `sigma(t) -> float`: Continuous noise level
- `mask_rate(t) -> float`: Discrete mask probability

### MemoryConditioner

Converts ArrowStorage retrieval results into diffusion conditioning vectors.

```python
class MemoryConditioner:
    """
    Memory-guided conditioner: converts ArrowStorage retrieval to diffusion conditions.
    
    This is the AI-OS differentiator—personal memory drives generation.
    """
    def __init__(self, arrow_storage, condition_dim):
        self.storage = arrow_storage
        self.projector = nn.Linear(384, condition_dim)  # MiniLM dim → condition dim
    
    def get_condition(self, query, top_k=5):
        """
        Retrieve relevant memories and project to condition vectors.
        
        Returns:
            condition: [K, condition_dim] condition matrix
        """
        # 1. Vector retrieval Top-K memories
        results = self.storage.search(query, limit=top_k)
        
        # 2. Extract memory embedding vectors
        memory_vectors = torch.stack([r.embedding for r in results])
        
        # 3. Project to diffusion condition space
        condition = self.projector(memory_vectors)
        
        return condition
```

**Interface:**
- `get_condition(query, top_k) -> condition`: Retrieve and project memories
- `project_embedding(embedding) -> condition`: Project single embedding

### UncertaintyEstimator

Measures uncertainty based on denoising residuals.

```python
class UncertaintyEstimator:
    """
    Uncertainty measurement based on denoising residuals.
    
    Replaces LoRA Router's heuristic confidence thresholds.
    """
    def estimate(self, x_t, x_0_pred, t):
        """
        Compute uncertainty at current denoising step.
        
        High: Model uncertain, should trigger self-evolution
        Low: Model confident, normal output
        """
        # Denoising residual (should approach 0 as t→0)
        residual = (x_t - x_0_pred).norm(dim=-1).mean()
        
        # Normalize to [0, 1]
        expected_residual = self.scheduler.sigma(t)
        uncertainty = residual / (expected_residual + 1e-8)
        
        return uncertainty.item()
    
    def should_evolve(self, uncertainty, threshold=1.5):
        """Trigger self-evolution when uncertainty exceeds threshold"""
        return uncertainty > threshold
```

**Interface:**
- `estimate(x_t, x_0_pred, t) -> float`: Compute uncertainty metric
- `should_evolve(uncertainty, threshold) -> bool`: Decide if evolution needed

### DiscreteSampler and ContinuousSampler

Samplers for discrete and continuous modalities.

```python
class DiscreteSampler:
    """Sampler for discrete modalities (text, code) using mask-based denoising."""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def step(self, logits, t, x_t):
        """
        One denoising step for discrete tokens.
        
        Unmasks tokens with highest confidence.
        """
        # Get mask rate for current timestep
        mask_rate = self.scheduler.mask_rate(t)
        
        # Compute confidence scores
        confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
        
        # Unmask top (1 - mask_rate) fraction
        num_unmask = int((1 - mask_rate) * x_t.size(1))
        unmask_indices = confidence.topk(num_unmask).indices
        
        # Update masked positions
        x_t[unmask_indices] = logits[unmask_indices].argmax(dim=-1)
        
        return x_t

class ContinuousSampler:
    """Sampler for continuous modalities (image, audio) using Gaussian denoising."""
    
    def __init__(self, scheduler):
        self.scheduler = scheduler
    
    def step(self, score, t, x_t):
        """
        One denoising step for continuous data.
        
        Applies Langevin dynamics update.
        """
        # Get noise level
        sigma_t = self.scheduler.sigma(t)
        sigma_next = self.scheduler.sigma(t - 1) if t > 0 else 0
        
        # Langevin update
        x_t = x_t + (sigma_t ** 2 - sigma_next ** 2) * score
        
        # Add noise for next step (if not final)
        if t > 0:
            x_t = x_t + torch.randn_like(x_t) * torch.sqrt(sigma_next ** 2 - sigma_t ** 2)
        
        return x_t
```

**Interfaces:**
- `DiscreteSampler.step(logits, t, x_t) -> x_t`: Unmask tokens
- `ContinuousSampler.step(score, t, x_t) -> x_t`: Langevin update

## Data Models

### DiffusionConfig

```python
@dataclass
class DiffusionConfig:
    """Configuration for unified diffusion system."""
    
    # Score Network architecture
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_dim: int = 3072
    condition_dim: int = 768
    
    # Modality-specific
    vocab_size: int = 32000
    max_seq_length: int = 2048
    image_size: int = 512
    patch_size: int = 16
    audio_sample_rate: int = 16000
    
    # Scheduler
    schedule_type: str = "cosine"
    num_train_steps: int = 1000
    num_inference_steps: int = 4
    
    # Quantization
    quantization_method: str = "arrowquant_v1"
    bit_width: int = 2
    
    # Evolution
    uncertainty_threshold: float = 1.5
    evolution_levels: List[str] = field(default_factory=lambda: ["L0", "L1", "L2", "L3", "L4"])
```

### ModelMetadata

```python
@dataclass
class ModelMetadata:
    """Metadata for diffusion models stored in metadata.json."""
    
    model_type: str = "unified_diffusion"
    diffusion_config: DiffusionConfig
    supported_modalities: List[str]
    consistency_distilled: bool
    quantization: Dict[str, Any]
    
    @classmethod
    def from_json(cls, path: Path) -> "ModelMetadata":
        """Load metadata from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
    
    def to_json(self, path: Path):
        """Save metadata to JSON file."""
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
```

### EvolutionProfile

```python
@dataclass
class EvolutionProfile:
    """Profile for active evolution components."""
    
    control_nets: List[ControlNetConfig] = field(default_factory=list)
    loras: List[LoRAConfig] = field(default_factory=list)
    energy_models: List[EnergyModelConfig] = field(default_factory=list)
    
@dataclass
class ControlNetConfig:
    """Configuration for a ControlNet."""
    id: str
    weight: float
    condition_template: Optional[str] = None

@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter."""
    id: str
    rank: int
    alpha: float
    target_modules: List[str]

@dataclass
class EnergyModelConfig:
    """Configuration for an energy-based model."""
    id: str
    weight: float
    constraint_type: str  # "physical", "logical", "format"
```

## Data Flow Examples

### Text Generation Flow

```
"Write a poem about spring"
    │
    ▼ MemoryConditioner
ArrowStorage.search("spring") → [memories: last spring photos, spring poems read]
    │
    ▼ condition = projector(memory_vectors)
    │
    ▼ DiscreteSampler initialization
x_T = [MASK] [MASK] [MASK] [MASK] ... [MASK]  (L=128)
    │
    ▼ 4-step denoising (Consistency Distillation)
    │  t=4: score_net(x_4, t=4, "text", condition)
    │       → unmask top 20% positions by confidence
    │  t=3: score_net(x_3, t=3, "text", condition)
    │       → unmask another 30%
    │  t=2 → t=1 → ...
    ▼
x_0 = "Spring breeze caresses flowers blooming, memories of that cherry blossom field..."
```

### Multi-Modal Parallel Generation Flow

```
User voice input: "Help me recall my last trip to Japan"
    │
    ▼ Unified condition preparation
condition = MemoryConditioner.get_condition("Japan trip")
    │
    ▼ Single SharedTransformer forward
hidden_states = shared_transformer(x_t, t, condition)
    │
    ├── TextHead.decode(h) → "Last time in Japan was..."     (text reply)
    ├── AudioHead.decode(h) → [16kHz waveform]                (voice synthesis)
    ├── AvatarHead.decode(h) → [blendshape params]            (lip sync + expression)
    └── ImageHead.decode(h) → [512×512 image]                 (related photo)
    
    ▲ All outputs naturally synchronized, no cascading delay
```

## Storage Design

### Model Directory Structure

```
models/
  diffusion-base/                   # Unified diffusion base
    metadata.json                   # Contains diffusion config
    shared_transformer.parquet      # Shared backbone weights (ArrowQuant INT2)
    tokenizer/                      # Text tokenizer
    heads/
      text_head.parquet             # Text projection head
      code_head.parquet             # Code projection head (can share with text)
      image_head.parquet            # Image projection head
      audio_head.parquet            # Audio projection head
    vae/                            # Image VAE (optional)
      encoder.parquet
      decoder.parquet
  lora_cards/
    writing_style.parquet           # LoRA fine-tuned on diffusion base
    code_python.parquet
  control_nets/
    cot_reasoning.parquet           # CoT-ControlNet
    tool_schema.parquet             # ToolSchema-ControlNet
```

### metadata.json Format

```json
{
  "model_type": "unified_diffusion",
  "diffusion_config": {
    "score_network": {
      "hidden_dim": 768,
      "num_layers": 12,
      "num_heads": 12,
      "intermediate_dim": 3072
    },
    "scheduler": {
      "type": "cosine",
      "num_train_steps": 1000,
      "num_inference_steps": 4
    },
    "supported_modalities": ["text", "code", "image", "audio"],
    "consistency_distilled": true
  },
  "quantization": {
    "method": "arrowquant_v1",
    "bit_width": 2
  }
}
```

## ArrowEngine Integration

```python
class ArrowEngine:
    # Existing interfaces (unchanged)
    def encode(self, sentences, **kwargs): ...
    def generate(self, prompt, **kwargs): ...
    
    # New unified diffusion interface
    def diffuse(
        self,
        prompt: str,
        modality: Literal["text", "code", "image", "audio"] = "text",
        num_steps: int = 4,
        guidance_scale: float = 3.0,
        memory_guided: bool = True,
        **kwargs,
    ) -> Union[str, np.ndarray, torch.Tensor]:
        """
        Unified diffusion generation.
        
        Args:
            prompt: Input prompt
            modality: Target generation modality
            num_steps: Denoising steps (1-50, default 4 for consistency)
            guidance_scale: Classifier-Free Guidance strength
            memory_guided: Whether to use ArrowStorage memory conditioning
        """
        # 1. Condition encoding
        condition = None
        if memory_guided:
            condition = self.memory_conditioner.get_condition(prompt)
        
        # 2. Diffusion generation
        output = self.diffusion_core.generate(
            condition=condition,
            modality=modality,
            num_steps=num_steps,
        )
        
        # 3. Post-processing
        if modality in ("text", "code"):
            return self.tokenizer.decode(output)
        elif modality == "image":
            return self.vae_decoder(output)  # latent → pixel
        elif modality == "audio":
            return output.numpy()  # waveform
```

**API Examples:**

```python
# Text generation
text = engine.diffuse("Write a haiku", modality="text", num_steps=4)

# Image generation
image = engine.diffuse("A serene mountain landscape", modality="image", num_steps=20)

# Audio generation
audio = engine.diffuse("Hello, how are you?", modality="audio", voice_id="default")

# Multi-modal generation
outputs = engine.diffuse_multimodal(
    prompt="Describe my last vacation",
    modalities=["text", "audio", "image"]
)
```


## Correctness Properties

A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.

### Property 1: Discrete Diffusion Initialization Consistency

*For any* text or code generation request, initializing the diffusion process should produce a fully masked token sequence of the requested length.

**Validates: Requirements 1.1**

### Property 2: Forward Noising Schedule Compliance

*For any* timestep t and input sequence, applying forward noising should produce a mask rate that matches the noise schedule's mask_rate(t) within numerical precision.

**Validates: Requirements 1.2**

### Property 3: Confidence-Based Unmasking

*For any* denoising step with discrete tokens, the tokens unmasked should be those with the highest confidence scores, and the number unmasked should match the schedule.

**Validates: Requirements 1.3**

### Property 4: Infilling Context Preservation

*For any* text generation with provided prefix and suffix, the generated output should contain the prefix at the start and suffix at the end unchanged.

**Validates: Requirements 1.4**

### Property 5: Modality Process Consistency

*For any* code generation request and text generation request with equivalent parameters, both should use the same DiscreteSampler and NoiseScheduler instances.

**Validates: Requirements 1.6**

### Property 6: Gaussian Initialization for Continuous Modalities

*For any* image or audio generation request, the initial latent state should follow a Gaussian distribution N(0, I) with mean near 0 and standard deviation near 1.

**Validates: Requirements 2.1**

### Property 7: VAE Dimensionality Reduction

*For any* image processed by the VAE Encoder, the latent representation dimension should be smaller than the original image dimension (height × width × channels).

**Validates: Requirements 2.2**

### Property 8: Score-Based Denoising Update

*For any* continuous denoising step, the update to x_t should follow the Langevin dynamics formula: x_{t-1} = x_t + (σ_t² - σ_{t-1}²) * score + noise.

**Validates: Requirements 2.3**

### Property 9: VAE Round-Trip Structural Preservation

*For any* valid image, encoding to latent space and then decoding back should preserve the overall structure (measured by SSIM > 0.8 or similar metric).

**Validates: Requirements 2.4**

### Property 10: Memory Conditioning Injection

*For any* generation request with memory_guided=True, the conditioning vector should be non-zero and derived from ArrowStorage retrieval results.

**Validates: Requirements 2.7, 5.1, 5.2, 5.3, 5.4**

### Property 11: Audio Sampling Rate Constraint

*For any* audio generated by the AudioProjectionHead, the sampling rate should be at least 16000 Hz.

**Validates: Requirements 3.2**

### Property 12: Audio Noise Initialization

*For any* TTS request, the initial audio state should be Gaussian noise in waveform or latent space.

**Validates: Requirements 3.1**

### Property 13: Shared Transformer Parameter Dominance

*For any* UnifiedScoreNetwork instance, the SharedTransformer parameters should account for at least 90% of the total model parameters.

**Validates: Requirements 4.2**

### Property 14: Modality-Specific Embeddings

*For any* two different modalities processed by the Score_Network, their modality embeddings should be distinct (non-equal).

**Validates: Requirements 4.3**

### Property 15: Time Embedding Variation

*For any* two different timesteps t1 and t2, the sinusoidal time embeddings should be different.

**Validates: Requirements 4.4**

### Property 16: Projection Head Parameter Constraint

*For any* new Projection_Head added to the system, its parameter count should be less than 10 million.

**Validates: Requirements 4.5**

### Property 17: Multi-Modal Single Forward Pass

*For any* multi-modal generation request, the SharedTransformer.forward() method should be called exactly once, producing hidden states for all requested modalities.

**Validates: Requirements 4.6, 16.1, 16.2**

### Property 18: Memory Conditioner Dimensionality Transformation

*For any* memory embeddings extracted from ArrowStorage, projecting them through MemoryConditioner should produce vectors with dimension equal to condition_dim.

**Validates: Requirements 5.3**

### Property 19: Uncertainty Computation Formula

*For any* denoising step, the uncertainty should be computed as: uncertainty = ||x_t - x_0_pred|| / (σ(t) + ε), where σ(t) is the expected noise level.

**Validates: Requirements 6.1, 6.2**

### Property 20: Uncertainty-Based Evolution Triggering

*For any* generation with computed uncertainty, evolution should be triggered if and only if uncertainty exceeds the threshold (default 1.5).

**Validates: Requirements 6.3, 6.4**

### Property 21: Variable Step Count Support

*For any* consistency-distilled model, it should accept step counts from 1 to 50 and produce valid outputs for all values.

**Validates: Requirements 7.4**

### Property 22: ControlNet Zero Initialization

*For any* newly loaded ControlNet, its residual connection weights should be initialized to zero or near-zero values.

**Validates: Requirements 8.1**

### Property 23: ControlNet Weighted Combination

*For any* set of active ControlNets with weights [w1, w2, ..., wn], the combined score should equal: base_score + Σ(wi * controlnet_i(x_t, t)).

**Validates: Requirements 8.2**

### Property 24: ControlNet Parameter Budget

*For any* ControlNet, its parameter count should be less than 10% of the base Score_Network parameter count.

**Validates: Requirements 8.3**

### Property 25: JSON Schema Compliance with ToolSchema-ControlNet

*For any* tool calling output generated with ToolSchema-ControlNet active, the output should parse as valid JSON and validate against the specified schema.

**Validates: Requirements 8.5**

### Property 26: Evolution Level Score Combination (L0)

*For any* L0 evolution request, the final score should be computed as a weighted sum of multiple score predictions without any parameter updates.

**Validates: Requirements 9.1**

### Property 27: Evolution Level Parameter Budgets

*For any* evolution level, the trainable parameters should match the expected budget: L1 ≈ 10%, L2 ≈ 1%, L3 variable, L4 = 100%.

**Validates: Requirements 9.2, 9.3, 9.4**

### Property 28: Uncertainty-Driven Level Selection

*For any* detected uncertainty value, the EvolutionRouter should select evolution levels in increasing order as uncertainty increases (higher uncertainty → higher level).

**Validates: Requirements 9.6**

### Property 29: Evolution Validation Before Persistence

*For any* completed evolution at any level, validation should be performed and pass before adaptations are persisted to storage.

**Validates: Requirements 9.7**

### Property 30: EBM Score Combination Formula

*For any* active EBM with weight η, the final score should be computed as: final_score = diffusion_score - η * ∇E(x_t, t).

**Validates: Requirements 10.1, 10.2**

### Property 31: Multi-EBM Support

*For any* set of active EBMs, the system should support applying all of them simultaneously with independent weights.

**Validates: Requirements 10.3**

### Property 32: EBM Retry with Increased Weight

*For any* EBM validation failure, the system should retry generation with increased constraint weight.

**Validates: Requirements 10.4**

### Property 33: Dual-Mode Noise Scheduling

*For any* NoiseScheduler instance, it should support both discrete (masking) and continuous (Gaussian) modes and produce appropriate noise for each.

**Validates: Requirements 11.1, 11.6**

### Property 34: Discrete Schedule Timestep Dependency

*For any* two different timesteps t1 < t2 in discrete mode, the mask rate should satisfy: mask_rate(t1) < mask_rate(t2) (more masking at higher noise levels).

**Validates: Requirements 11.2**

### Property 35: Continuous Schedule Timestep Dependency

*For any* two different timesteps t1 < t2 in continuous mode, the noise standard deviation should satisfy: σ(t1) < σ(t2) (more noise at higher noise levels).

**Validates: Requirements 11.3**

### Property 36: Schedule Type Support

*For any* NoiseScheduler, it should support at least three schedule types: "cosine", "linear", and "custom".

**Validates: Requirements 11.4**

### Property 37: Sampling Strategy Support

*For any* NoiseScheduler generating inference timesteps, it should support both uniform and non-uniform sampling strategies.

**Validates: Requirements 11.5**

### Property 38: Mode-Based Routing

*For any* ArrowEngine request, when mode="ar" it should route to InferenceCore, and when mode="diffusion" it should route to DiffusionCore.

**Validates: Requirements 12.2, 17.1**

### Property 39: Backward API Compatibility

*For any* existing code using ArrowEngine.encode() or ArrowEngine.generate(), it should continue to work without modification after diffusion integration.

**Validates: Requirements 12.1**

### Property 40: Memory-Guided Storage Query

*For any* diffuse() call with memory_guided=True, ArrowStorage.search() should be invoked with the prompt as query.

**Validates: Requirements 12.4**

### Property 41: Modality-Specific Output Types

*For any* diffuse() call, the output type should match the modality: text/code → str, image → np.ndarray/torch.Tensor, audio → np.ndarray.

**Validates: Requirements 12.5, 12.6, 12.7**

### Property 42: Parquet V2 Storage Format

*For any* diffusion model component (Score_Network, Projection_Heads, ControlNets, VAE), weights should be stored in Parquet V2 format.

**Validates: Requirements 13.1, 13.2, 13.3, 13.4**

### Property 43: Quantization Support

*For any* stored weights, the system should support both INT2 and INT4 ArrowQuant quantization.

**Validates: Requirements 13.5**

### Property 44: Zero-Copy Memory Mapping

*For any* weight loading operation, the system should use memory mapping to avoid copying data into process memory.

**Validates: Requirements 13.6**

### Property 45: Lazy Dequantization

*For any* quantized weight access, dequantization should occur on-demand when the weight is accessed, not during initial loading.

**Validates: Requirements 13.7**

### Property 46: Metadata Configuration Parsing

*For any* model with metadata.json, loading the model should configure DiffusionCore with parameters matching the metadata specification.

**Validates: Requirements 15.7**

### Property 47: Multi-Modal Output Timing Alignment

*For any* multi-modal generation producing text, audio, and avatar outputs, all outputs should include timing information that allows synchronization.

**Validates: Requirements 16.4**

### Property 48: Dual-Model Loading Support

*For any* system instance, it should support loading both an AR model and a diffusion model simultaneously without conflicts.

**Validates: Requirements 17.2**

### Property 49: AR LoRA Functionality Preservation

*For any* AR model with LoRA adapters, the existing LoRA Router functionality should continue to work correctly.

**Validates: Requirements 17.3**

### Property 50: Cross-Model LoRA Isolation

*For any* attempt to apply an AR model LoRA to a diffusion model or vice versa, the system should reject the operation with an error.

**Validates: Requirements 17.4**

### Property 51: Mode-Consistent API Parameters

*For any* common parameter (e.g., temperature, top_k) supported in both AR and diffusion modes, it should have consistent behavior and interpretation.

**Validates: Requirements 17.5**

### Property 52: Uncertainty-Triggered Evolution

*For any* generation completing with uncertainty above threshold, the system should trigger the appropriate evolution level based on uncertainty magnitude.

**Validates: Requirements 18.1**

### Property 53: Adaptation Persistence After Validation

*For any* evolution-produced adaptation, it should be stored to ArrowStorage only after validation confirms improvement.

**Validates: Requirements 18.2, 18.3**

### Property 54: Adaptation Retrieval and Application

*For any* generation context similar to a previously learned adaptation, the system should retrieve and apply the relevant adaptation from ArrowStorage.

**Validates: Requirements 18.4**

## Error Handling

### Error Categories

1. **Configuration Errors**: Invalid model metadata, missing components, incompatible quantization
2. **Generation Errors**: Denoising divergence, invalid outputs, constraint violations
3. **Memory Errors**: ArrowStorage unavailable, conditioning projection failures
4. **Evolution Errors**: Validation failures, adaptation conflicts, insufficient resources
5. **Storage Errors**: Parquet loading failures, quantization errors, corrupted weights

### Error Handling Strategy

```python
class DiffusionError(Exception):
    """Base exception for diffusion system errors."""
    pass

class ConfigurationError(DiffusionError):
    """Raised when model configuration is invalid."""
    pass

class GenerationError(DiffusionError):
    """Raised when generation fails or produces invalid output."""
    pass

class MemoryConditioningError(DiffusionError):
    """Raised when memory conditioning fails."""
    pass

class EvolutionError(DiffusionError):
    """Raised when self-evolution fails."""
    pass

class StorageError(DiffusionError):
    """Raised when weight loading or storage operations fail."""
    pass
```

### Error Recovery

1. **Graceful Degradation**: Fall back to AR mode if diffusion fails
2. **Retry with Adjusted Parameters**: Reduce step count, disable conditioning
3. **Logging and Monitoring**: Track error rates and patterns
4. **User Notification**: Provide clear error messages with recovery suggestions

### Validation Points

1. **Pre-Generation**: Validate configuration, check model availability
2. **During Generation**: Monitor denoising convergence, check uncertainty
3. **Post-Generation**: Validate output format, check constraints
4. **Evolution**: Validate improvements before persistence

## Testing Strategy

### Dual Testing Approach

The system requires both unit tests and property-based tests for comprehensive coverage:

- **Unit tests**: Verify specific examples, edge cases, and error conditions
- **Property tests**: Verify universal properties across all inputs using randomization

Both approaches are complementary and necessary. Unit tests catch concrete bugs in specific scenarios, while property tests verify general correctness across the input space.

### Property-Based Testing

**Framework**: Use `hypothesis` library for Python property-based testing

**Configuration**:
- Minimum 100 iterations per property test (due to randomization)
- Each test must reference its design document property
- Tag format: `# Feature: unified-diffusion-architecture, Property {N}: {property_text}`

**Example Property Test**:

```python
from hypothesis import given, strategies as st
import pytest

@given(
    modality=st.sampled_from(["text", "code"]),
    seq_length=st.integers(min_value=10, max_value=512)
)
@pytest.mark.property
def test_discrete_initialization_consistency(modality, seq_length):
    """
    Feature: unified-diffusion-architecture, Property 1: Discrete Diffusion Initialization Consistency
    
    For any text or code generation request, initializing the diffusion process
    should produce a fully masked token sequence of the requested length.
    """
    core = DiffusionCore(score_network, scheduler, config)
    x_t = core._init_masked_sequence(condition=None, length=seq_length)
    
    # Verify fully masked
    assert x_t.shape[0] == seq_length
    assert torch.all(x_t == MASK_TOKEN_ID)
```

### Unit Testing

**Focus Areas**:
- Component initialization and configuration
- Specific edge cases (empty inputs, boundary values)
- Error conditions and exception handling
- Integration points between components

**Example Unit Test**:

```python
def test_vae_encoder_output_shape():
    """Test that VAE encoder produces expected latent shape."""
    vae = VAEEncoder(config)
    image = torch.randn(1, 3, 512, 512)
    latent = vae.encode(image)
    
    expected_shape = (1, 4, 64, 64)  # 8x downsampling, 4 channels
    assert latent.shape == expected_shape
```

### Integration Testing

**Scenarios**:
1. End-to-end text generation with memory guidance
2. Multi-modal parallel generation
3. Evolution triggering and adaptation persistence
4. Mode switching between AR and diffusion
5. ControlNet and LoRA application

### Performance Testing

**Benchmarks**:
1. Text generation latency (target: <500ms, 350M params, INT2, CPU)
2. Image generation latency (target: <30s, 600M params, INT4, CPU)
3. TTS latency (target: <2s end-to-end)
4. Memory conditioning overhead (target: <10ms)
5. Multi-modal synchronization accuracy

### Test Organization

```
tests/
├── unit/
│   ├── test_score_network.py
│   ├── test_diffusion_core.py
│   ├── test_noise_scheduler.py
│   ├── test_memory_conditioner.py
│   ├── test_uncertainty_estimator.py
│   └── test_evolution_router.py
├── property/
│   ├── test_diffusion_properties.py
│   ├── test_evolution_properties.py
│   ├── test_memory_properties.py
│   └── test_storage_properties.py
├── integration/
│   ├── test_end_to_end_generation.py
│   ├── test_multimodal_generation.py
│   ├── test_evolution_loop.py
│   └── test_arrowengine_integration.py
└── performance/
    ├── benchmark_generation_latency.py
    ├── benchmark_memory_usage.py
    └── benchmark_multimodal_sync.py
```

## Implementation Notes

### Phase 3a: Discrete Diffusion Text PoC (2 weeks)

1. Implement DiffusionCore + NoiseScheduler + DiscreteSampler
2. Implement TextProjectionHead
3. Convert open-source MDLM weights → Parquet V2 + ArrowQuant
4. Implement ArrowEngine.diffuse(modality="text")
5. Validate infilling quality and latency benchmarks

### Phase 3b: Unified Score Network + Memory Conditioning (2 weeks)

1. Implement UnifiedScoreNetwork (shared Transformer)
2. Implement MemoryConditioner (ArrowStorage → condition vectors)
3. Implement UncertaintyEstimator (uncertainty-driven evolution trigger)
4. Cross-modal end-to-end testing

### Phase 3c: Image/Audio Diffusion + Virtual Embodiment (3 weeks)

1. Implement ImageProjectionHead + VAE integration
2. Implement AudioProjectionHead (WaveGrad)
3. Consistency Distillation training (4-step compression)
4. Multi-modal parallel generation validation (virtual embodiment scenario)
5. Edge deployment validation (ARM + INT2)

### Risks and Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Open-source MDLM weights insufficient quality | Medium | High | Fallback to SEDD; or initialize from BERT weights |
| Consistency Distillation quality degradation | Medium | Medium | Keep multi-step inference as high-quality fallback |
| Unified backbone cross-modal interference | Medium | High | Train heads independently initially, unfreeze backbone gradually |
| Edge device insufficient compute | Low | Medium | INT2 + minimal backbone (~50M) subset deployment |

### Integration with ArrowQuant

The unified diffusion architecture forms a perfect synergy with Phase 2 ArrowQuant:

```
ArrowQuant INT2 quantization (Phase 2)
        ↓ compress weights
UnifiedScoreNetwork.shared_transformer (Phase 3)
        ↓ ~200MB (INT2)
Edge device deployable
```

ArrowQuant provides the storage infrastructure for diffusion models, while diffusion models provide the largest application scenario for ArrowQuant.
