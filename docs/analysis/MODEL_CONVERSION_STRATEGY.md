# 模型转换与 ArrowEngine 扩展策略

**日期**: 2026-02-18  
**目标**: 转换多个开源模型并评估 CLIP 等多模态模型的集成策略

---

## 第一部分：初始模型转换计划

### 1.1 目标模型列表

| 模型 | 类型 | 架构 | 参数量 | 用途 | 优先级 |
|------|------|------|--------|------|--------|
| **MiniCPM-o 4.5** | 多模态 | Transformer | ~4.5B | 图文理解 | P1 |
| **Step 3.5 Flash** | 文本 | Transformer | ~3.5B | 快速推理 | P1 |
| **Stable-DiffCoder** | 代码 | Transformer | ~1-3B | 代码理解 | P1 |
| **Intern-S1-Pro** | 文本 | Transformer | ~7B | 长上下文 | P2 |
| **CLIP** | 多模态 | Dual-Encoder | ~400M | 图文对齐 | P1 |

### 1.2 转换优先级策略

**Phase 1: 文本模型（Week 1-2）**
- ✅ all-MiniLM-L6-v2 (已完成)
- 🔄 Step 3.5 Flash
- 🔄 Stable-DiffCoder

**Phase 2: 多模态模型（Week 3-4）**
- 🔄 CLIP (图文对齐)
- 🔄 MiniCPM-o 4.5 (完整多模态)

**Phase 3: 大模型（Week 5-6）**
- 🔄 Intern-S1-Pro (长上下文)

---

## 第二部分：CLIP 与 Sentence-Transformer 的架构对比

### 2.1 架构差异分析

#### Sentence-Transformer (BERT-based)
```
输入: 文本
    ↓
Tokenizer → Token IDs
    ↓
Embedding Layer (word + position + token_type)
    ↓
12 x Transformer Layers
    ├─ MultiHeadAttention
    ├─ LayerNorm
    ├─ FeedForward (GELU)
    └─ LayerNorm
    ↓
Mean Pooling
    ↓
L2 Normalization
    ↓
输出: 384-dim embedding
```

#### CLIP (Dual-Encoder)
```
文本分支:                      图像分支:
输入: 文本                     输入: 图像
    ↓                             ↓
Text Tokenizer              Image Patches (16x16)
    ↓                             ↓
Text Embedding              Patch Embedding
    ↓                             ↓
12 x Text Transformer       12 x Vision Transformer
    ↓                             ↓
[CLS] Token Pooling         [CLS] Token Pooling
    ↓                             ↓
Text Projection             Image Projection
    ↓                             ↓
512-dim embedding           512-dim embedding
         ↓                   ↓
         └─── 对比学习空间 ───┘
              (Contrastive Learning)
```

### 2.2 关键差异

| 维度 | Sentence-Transformer | CLIP |
|------|---------------------|------|
| **架构** | 单编码器 | 双编码器（文本+图像） |
| **输入** | 仅文本 | 文本 + 图像 |
| **Pooling** | Mean Pooling | [CLS] Token |
| **投影层** | 无 | 有（降维到共享空间） |
| **训练目标** | 句子相似度 | 图文对比学习 |
| **输出维度** | 384/768 | 512 |

---

## 第三部分：CLIP 是否需要 ArrowEngine 原生支持？

### 3.1 答案：是的，强烈建议！

**原因**:

1. **架构复杂度更高**
   - 双编码器架构
   - 需要同时处理文本和图像
   - 投影层和对比学习空间

2. **性能优化空间大**
   - 图像编码器计算密集
   - Vision Transformer 的 patch embedding
   - 大量矩阵运算

3. **零拷贝优势明显**
   - 图像数据量大（224x224x3）
   - Arrow 零拷贝可显著减少内存占用
   - 批处理效率提升

4. **多模态融合需求**
   - 需要高效的文本-图像对齐
   - 跨模态检索性能关键
   - 实时性要求高

### 3.2 性能对比预测

| 指标 | HuggingFace CLIP | ArrowEngine CLIP | 提升 |
|------|-----------------|------------------|------|
| 模型加载 | ~5s | ~500ms | 10x |
| 文本编码 | ~50ms | ~20ms | 2.5x |
| 图像编码 | ~100ms | ~40ms | 2.5x |
| 批量吞吐 | ~50 img/s | ~150 img/s | 3x |
| 内存占用 | ~2GB | ~800MB | 2.5x |

---

## 第四部分：ArrowEngine 扩展架构

### 4.1 模块化设计

```python
# 当前架构
llm_compression/
├── inference/
│   ├── inference_core.py      # BERT Transformer ✅
│   ├── arrow_engine.py         # 文本编码器 ✅
│   ├── weight_loader.py        # 权重加载 ✅
│   └── fast_tokenizer.py       # 文本分词 ✅

# 扩展架构
llm_compression/
├── inference/
│   ├── inference_core.py      # 基础 Transformer ✅
│   ├── arrow_engine.py         # 统一接口 ✅
│   ├── weight_loader.py        # 权重加载 ✅
│   ├── fast_tokenizer.py       # 文本分词 ✅
│   │
│   ├── text_encoder.py         # 文本编码器（BERT/GPT）🆕
│   ├── vision_encoder.py       # 视觉编码器（ViT）🆕
│   ├── clip_engine.py          # CLIP 双编码器 🆕
│   ├── multimodal_fusion.py    # 多模态融合 🆕
│   └── image_processor.py      # 图像预处理 🆕
```

### 4.2 CLIP ArrowEngine 实现

```python
class VisionTransformer:
    """
    Vision Transformer 核心实现
    
    架构:
    1. Patch Embedding (16x16 patches)
    2. Position Embedding
    3. 12 x Transformer Layers
    4. [CLS] Token Pooling
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12
    ):
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embedding = nn.Conv2d(
            in_channels=3,
            out_channels=hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        
        # Position Embedding
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_size)
        )
        
        # [CLS] Token
        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, hidden_size)
        )
        
        # Transformer Layers (复用 InferenceCore)
        self.transformer = InferenceCore(
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            intermediate_size=hidden_size * 4
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        输入: (batch_size, 3, 224, 224)
        输出: (batch_size, hidden_size)
        """
        batch_size = images.shape[0]
        
        # 1. Patch Embedding
        # (B, 3, 224, 224) → (B, 768, 14, 14) → (B, 196, 768)
        patches = self.patch_embedding(images)
        patches = patches.flatten(2).transpose(1, 2)
        
        # 2. 添加 [CLS] Token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 197, 768)
        
        # 3. 添加 Position Embedding
        x = x + self.position_embedding
        
        # 4. Transformer Layers
        x = self.transformer(x)
        
        # 5. 提取 [CLS] Token
        cls_output = x[:, 0]  # (B, 768)
        
        return cls_output


class CLIPEngine:
    """
    CLIP ArrowEngine 实现
    
    核心能力:
    1. 文本编码（复用 InferenceCore）
    2. 图像编码（VisionTransformer）
    3. 投影到共享空间
    4. 零拷贝 Arrow 数据流
    """
    
    def __init__(self, model_path: str):
        # 加载权重
        self.weight_loader = WeightLoader(model_path)
        
        # 文本编码器（复用现有实现）
        self.text_encoder = InferenceCore(
            hidden_size=512,
            num_layers=12,
            num_heads=8
        )
        
        # 图像编码器
        self.vision_encoder = VisionTransformer(
            image_size=224,
            patch_size=16,
            hidden_size=768,
            num_layers=12,
            num_heads=12
        )
        
        # 投影层
        self.text_projection = nn.Linear(512, 512)
        self.vision_projection = nn.Linear(768, 512)
        
        # 加载权重
        self._load_weights()
    
    def encode_text(
        self, 
        texts: List[str],
        normalize: bool = True
    ) -> np.ndarray:
        """
        编码文本
        
        输入: 文本列表
        输出: (N, 512) embedding
        """
        # 1. Tokenize
        tokens = self.tokenizer(texts)
        
        # 2. 文本编码
        text_features = self.text_encoder(tokens)
        
        # 3. 投影
        text_embeddings = self.text_projection(text_features)
        
        # 4. L2 归一化
        if normalize:
            text_embeddings = F.normalize(text_embeddings, dim=-1)
        
        return text_embeddings.cpu().numpy()
    
    def encode_image(
        self,
        images: np.ndarray,  # Arrow Array
        normalize: bool = True
    ) -> np.ndarray:
        """
        编码图像（零拷贝）
        
        输入: Arrow Array (N, 224, 224, 3)
        输出: (N, 512) embedding
        """
        # 1. 零拷贝转换为 Tensor
        image_tensor = torch.from_numpy(images).permute(0, 3, 1, 2)
        
        # 2. 图像编码
        vision_features = self.vision_encoder(image_tensor)
        
        # 3. 投影
        image_embeddings = self.vision_projection(vision_features)
        
        # 4. L2 归一化
        if normalize:
            image_embeddings = F.normalize(image_embeddings, dim=-1)
        
        return image_embeddings.cpu().numpy()
    
    def compute_similarity(
        self,
        text_embeddings: np.ndarray,
        image_embeddings: np.ndarray
    ) -> np.ndarray:
        """
        计算文本-图像相似度
        
        输入:
        - text_embeddings: (N, 512)
        - image_embeddings: (M, 512)
        
        输出: (N, M) 相似度矩阵
        """
        # 向量化计算（零拷贝）
        similarity = np.dot(text_embeddings, image_embeddings.T)
        return similarity
```

### 4.3 零拷贝图像处理

```python
class ArrowImageProcessor:
    """
    Arrow 原生图像处理
    
    核心能力:
    1. 零拷贝图像加载
    2. 向量化预处理
    3. 批处理优化
    """
    
    def __init__(self):
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
    
    def preprocess_arrow(
        self,
        images: pa.Array  # Arrow Binary Array
    ) -> np.ndarray:
        """
        零拷贝图像预处理
        
        流程:
        1. Arrow Binary → NumPy (零拷贝)
        2. Resize (向量化)
        3. Normalize (向量化)
        4. 返回 NumPy Array
        """
        # 1. 零拷贝转换
        image_arrays = []
        for img_bytes in images:
            # 从 bytes 解码图像（零拷贝）
            img = np.frombuffer(img_bytes.as_py(), dtype=np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            image_arrays.append(img)
        
        # 2. 批量 resize（向量化）
        resized = np.stack([
            cv2.resize(img, (224, 224)) 
            for img in image_arrays
        ])
        
        # 3. 归一化（向量化）
        normalized = (resized / 255.0 - self.mean) / self.std
        
        return normalized.astype(np.float32)
```

---

## 第五部分：模型转换实施计划

### 5.1 转换脚本模板

```python
# scripts/convert_clip_to_arrow.py

import torch
import pyarrow as pa
import pyarrow.parquet as pq
from transformers import CLIPModel, CLIPProcessor

def convert_clip_to_arrow(
    model_name: str = "openai/clip-vit-base-patch16",
    output_dir: str = "./models/clip"
):
    """
    转换 CLIP 模型到 Arrow 格式
    
    步骤:
    1. 加载 HuggingFace CLIP
    2. 提取权重
    3. 转换为 Arrow Table
    4. 保存为 Parquet
    """
    print(f"Loading {model_name}...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    
    # 提取权重
    weights = {}
    
    # 文本编码器权重
    for name, param in model.text_model.named_parameters():
        weights[f"text.{name}"] = param.detach().cpu().numpy()
    
    # 图像编码器权重
    for name, param in model.vision_model.named_parameters():
        weights[f"vision.{name}"] = param.detach().cpu().numpy()
    
    # 投影层权重
    weights["text_projection"] = model.text_projection.weight.detach().cpu().numpy()
    weights["visual_projection"] = model.visual_projection.weight.detach().cpu().numpy()
    
    # 转换为 Arrow Table
    print("Converting to Arrow format...")
    arrow_table = convert_weights_to_arrow(weights)
    
    # 保存
    output_path = f"{output_dir}/weights.parquet"
    pq.write_table(arrow_table, output_path, compression='zstd')
    
    # 保存 tokenizer
    processor.save_pretrained(f"{output_dir}/tokenizer")
    
    print(f"✅ Conversion complete: {output_path}")
    print(f"   Original size: {get_model_size(model):.2f} MB")
    print(f"   Arrow size: {get_file_size(output_path):.2f} MB")
    print(f"   Compression ratio: {get_compression_ratio(model, output_path):.2f}x")


def convert_weights_to_arrow(weights: Dict) -> pa.Table:
    """转换权重字典为 Arrow Table"""
    arrays = []
    names = []
    
    for name, weight in weights.items():
        # 展平权重
        flat_weight = weight.flatten()
        
        # 创建 Arrow Array
        arrow_array = pa.array(flat_weight, type=pa.float32())
        
        arrays.append(arrow_array)
        names.append(name)
    
    # 创建 Table
    table = pa.table({
        'layer_name': pa.array(names),
        'weights': arrays,
        'shape': pa.array([w.shape for w in weights.values()]),
        'dtype': pa.array([str(w.dtype) for w in weights.values()])
    })
    
    return table
```

### 5.2 批量转换脚本

```python
# scripts/batch_convert_models.py

MODELS_TO_CONVERT = [
    {
        'name': 'Step 3.5 Flash',
        'hf_name': 'stepfun-ai/step-3.5-flash',
        'type': 'text',
        'output_dir': './models/step-flash'
    },
    {
        'name': 'Stable-DiffCoder',
        'hf_name': 'bytedance/stable-diffcoder',
        'type': 'code',
        'output_dir': './models/diffcoder'
    },
    {
        'name': 'CLIP',
        'hf_name': 'openai/clip-vit-base-patch16',
        'type': 'multimodal',
        'output_dir': './models/clip'
    },
    {
        'name': 'MiniCPM-o 4.5',
        'hf_name': 'openbmb/MiniCPM-o-4_5',
        'type': 'multimodal',
        'output_dir': './models/minicpm'
    }
]

def batch_convert():
    """批量转换模型"""
    for model_config in MODELS_TO_CONVERT:
        print(f"\n{'='*60}")
        print(f"Converting: {model_config['name']}")
        print(f"{'='*60}")
        
        try:
            if model_config['type'] == 'text':
                convert_text_model(
                    model_config['hf_name'],
                    model_config['output_dir']
                )
            elif model_config['type'] == 'multimodal':
                convert_multimodal_model(
                    model_config['hf_name'],
                    model_config['output_dir']
                )
            elif model_config['type'] == 'code':
                convert_code_model(
                    model_config['hf_name'],
                    model_config['output_dir']
                )
            
            print(f"✅ {model_config['name']} converted successfully!")
            
        except Exception as e:
            print(f"❌ Failed to convert {model_config['name']}: {e}")
            continue

if __name__ == "__main__":
    batch_convert()
```

---

## 第六部分：实施时间表

### Week 1: 文本模型转换
- Day 1-2: Step 3.5 Flash 转换和验证
- Day 3-4: Stable-DiffCoder 转换和验证
- Day 5: 性能基准测试

### Week 2: CLIP 扩展开发
- Day 1-2: VisionTransformer 实现
- Day 3-4: CLIPEngine 实现
- Day 5: 端到端测试

### Week 3: CLIP 转换和优化
- Day 1-2: CLIP 模型转换
- Day 3-4: 性能优化
- Day 5: 精度验证

### Week 4: MiniCPM-o 集成
- Day 1-3: MiniCPM-o 架构分析
- Day 4-5: 转换和初步测试

---

## 第七部分：建议与总结

### 7.1 CLIP 需要 ArrowEngine 原生支持吗？

**答案：强烈建议！**

**理由**:
1. ✅ 架构复杂度高（双编码器）
2. ✅ 性能优化空间大（10x+ 提升）
3. ✅ 零拷贝优势明显（图像数据大）
4. ✅ 多模态融合需求（实时性关键）
5. ✅ 与现有架构高度兼容（复用 InferenceCore）

### 7.2 实施优先级

**P0 (立即开始)**:
1. CLIP VisionTransformer 实现
2. CLIP 模型转换
3. 端到端验证

**P1 (Week 2-3)**:
1. Step 3.5 Flash 转换
2. Stable-DiffCoder 转换
3. 性能基准测试

**P2 (Week 4+)**:
1. MiniCPM-o 集成
2. Intern-S1-Pro 转换
3. 动态权重组合

### 7.3 预期成果

转换完成后，我们将拥有:
- ✅ 5+ 个高性能本地模型
- ✅ 文本 + 图像 + 代码能力
- ✅ 完整的多模态支持
- ✅ 动态权重组合基础

这将为动态权重组合系统提供坚实的基础！

---

**文档日期**: 2026-02-18  
**状态**: 实施计划  
**下一步**: 开始 CLIP VisionTransformer 实现
