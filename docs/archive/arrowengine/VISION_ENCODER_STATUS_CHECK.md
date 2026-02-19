# Vision Encoder 实现状态检查

## 检查结果：❌ 未实现

### 现有实现分析

#### 1. 现有文件：`llm_compression/multimodal/vision_provider.py`

**CLIPVisionProvider 类**：
```python
class CLIPVisionProvider(VisionProvider):
    """Implementation using OpenAI CLIP (via transformers or open_clip)."""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: str = "cpu"):
        from transformers import CLIPProcessor, CLIPModel
        self._processor = CLIPProcessor.from_pretrained(model_name)
        self._model = CLIPModel.from_pretrained(model_name).to(device)
```

**问题**：
- ❌ 使用 HuggingFace transformers（不是 Arrow-native）
- ❌ 没有复用 InferenceCore
- ❌ 没有使用 WeightLoader（Parquet 权重）
- ❌ 加载速度慢（~9s vs 目标 <500ms）
- ❌ 内存占用高（~2GB vs 目标 <1GB）

### Spec 要求的实现

根据 `.kiro/specs/multimodal-encoder-system/design.md`，需要实现：

#### 1. PatchEmbedding 模块
```python
class PatchEmbedding(nn.Module):
    """
    Convert images to patch embeddings.
    
    Process:
    1. Split image into 16x16 patches (196 patches for 224x224)
    2. Flatten each patch to 768-dim vector
    3. Project through linear layer
    """
    
    def __init__(self, config: VisionConfig):
        super().__init__()
        # Conv2d acts as patch extraction + linear projection
        self.projection = nn.Conv2d(
            in_channels=3,
            out_channels=config.hidden_size,  # 768
            kernel_size=config.patch_size,    # 16
            stride=config.patch_size,         # 16
            bias=False
        )
```

**状态**: ❌ 未实现

#### 2. VisionEncoder 类
```python
class VisionEncoder:
    """
    CLIP Vision Transformer encoder.
    
    Architecture:
    1. Patch Embedding (16x16 patches)
    2. Add [CLS] token
    3. Add position embeddings
    4. 12 Transformer layers (via InferenceCore)  # 关键：复用 InferenceCore
    5. Extract [CLS] token output
    6. Project to output dimension
    """
    
    def __init__(self, model_path: str, device: Optional[str] = None):
        # Load weights from Parquet
        self.weight_loader = WeightLoader(model_path)
        
        # Initialize components
        self.patch_embedding = PatchEmbedding(self.config)
        self.cls_token = nn.Parameter(...)
        self.position_embedding = nn.Parameter(...)
        
        # Reuse InferenceCore for Transformer layers
        self.transformer = InferenceCore(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072
        )
```

**状态**: ❌ 未实现

#### 3. 权重加载
```python
def _load_weights(self, weights: dict):
    """Load weights from Parquet into PyTorch modules."""
    # Load patch embedding weights
    self.patch_embedding.projection.weight.data = torch.from_numpy(
        weights["vision_model.embeddings.patch_embedding.weight"]
    )
    
    # Load CLS token and position embeddings
    self.cls_token.data = torch.from_numpy(...)
    self.position_embedding.data = torch.from_numpy(...)
```

**状态**: ❌ 未实现

### 关键差异对比

| 特性 | 现有 CLIPVisionProvider | Spec 要求的 VisionEncoder |
|------|------------------------|--------------------------|
| **基础框架** | HuggingFace transformers | Arrow-native (自研) |
| **Transformer** | CLIP 内置 | InferenceCore (复用) |
| **权重加载** | HF checkpoint | Parquet (WeightLoader) |
| **加载速度** | ~9s | <500ms (目标) |
| **内存占用** | ~2GB | <1GB (目标) |
| **推理速度** | ~150ms | <100ms (目标) |
| **依赖** | transformers, torch | 仅 torch + Arrow |
| **零拷贝** | ❌ | ✅ |

### 为什么需要重新实现？

#### 1. 性能目标
- **加载速度**: 9s → <500ms (18x 加速)
- **推理延迟**: 150ms → <100ms (1.5x 加速)
- **内存占用**: 2GB → <1GB (2x 减少)

#### 2. 架构一致性
- 复用 InferenceCore（与 TextEncoder 一致）
- 复用 WeightLoader（与 ArrowEngine 一致）
- 零拷贝 Arrow 架构（与整体系统一致）

#### 3. 依赖简化
- 移除 transformers 依赖
- 统一权重格式（Parquet）
- 统一接口（EmbeddingProvider）

### 实现计划

#### Task 2.1: PatchEmbedding 模块
```python
# 文件: llm_compression/multimodal/vision_encoder.py

class PatchEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.projection = nn.Conv2d(3, 768, kernel_size=16, stride=16, bias=False)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # (B, 3, 224, 224) -> (B, 768, 14, 14) -> (B, 196, 768)
        patches = self.projection(images)
        patches = patches.flatten(2).transpose(1, 2)
        return patches
```

#### Task 2.2: VisionEncoder 类
```python
class VisionEncoder:
    def __init__(self, model_path: str, device: Optional[str] = None):
        # 1. Load config
        self.config = VisionConfig()
        
        # 2. Load weights from Parquet
        self.weight_loader = WeightLoader(model_path)
        weights = self.weight_loader.load_weights()
        
        # 3. Initialize components
        self.patch_embedding = PatchEmbedding(self.config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.position_embedding = nn.Parameter(torch.zeros(1, 197, 768))
        
        # 4. Reuse InferenceCore
        self.transformer = InferenceCore(
            hidden_size=768,
            num_layers=12,
            num_heads=12,
            intermediate_size=3072
        )
        
        # 5. Load weights
        self._load_weights(weights)
    
    def encode(self, images: np.ndarray, normalize: bool = True) -> np.ndarray:
        # Preprocess -> Forward -> Normalize
        ...
```

#### Task 2.3: 权重加载
```python
def _load_weights(self, weights: dict):
    # Patch embedding
    self.patch_embedding.projection.weight.data = torch.from_numpy(
        weights["vision_model.embeddings.patch_embedding.weight"]
    )
    
    # CLS token & position embeddings
    self.cls_token.data = torch.from_numpy(
        weights["vision_model.embeddings.class_embedding"]
    ).unsqueeze(0).unsqueeze(0)
    
    self.position_embedding.data = torch.from_numpy(
        weights["vision_model.embeddings.position_embedding.weight"]
    ).unsqueeze(0)
    
    # LayerNorm weights
    self.pre_layernorm.weight.data = torch.from_numpy(...)
    self.post_layernorm.weight.data = torch.from_numpy(...)
```

### 依赖检查

#### 需要的组件（已存在）
- ✅ InferenceCore - `llm_compression/inference/inference_core.py`
- ✅ WeightLoader - `llm_compression/inference/weight_loader.py`
- ✅ ImageProcessor - `llm_compression/multimodal/image_processor.py`

#### 需要创建的文件
- ❌ `llm_compression/multimodal/vision_encoder.py` - VisionEncoder 实现
- ❌ `llm_compression/multimodal/vision_config.py` - VisionConfig 配置
- ❌ `tests/unit/test_vision_encoder.py` - 单元测试

### 下一步行动

1. **创建 VisionConfig** - 配置类
2. **实现 PatchEmbedding** - Patch 提取模块
3. **实现 VisionEncoder** - 主编码器类
4. **实现权重加载** - 从 Parquet 加载
5. **编写单元测试** - 验证功能
6. **性能基准测试** - 验证性能目标

### 总结

**现状**: 
- ❌ Vision Encoder 未按 spec 实现
- ⚠️ 现有 CLIPVisionProvider 是 HuggingFace 包装器
- ⚠️ 不符合 Arrow-native 架构要求

**需要**: 
- 从零开始实现 Arrow-native VisionEncoder
- 复用 InferenceCore 和 WeightLoader
- 达到性能目标（<500ms 加载，<100ms 推理）

**预计工作量**: 
- 核心实现：2-3 小时
- 测试验证：1-2 小时
- 总计：3-5 小时

