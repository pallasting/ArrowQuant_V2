# 模型迁移到本地 SSD - 完成指南

## 执行摘要

✅ **模型已成功迁移到本地 SSD**

- **原位置**: `M:\Documents\ai-os-memory\models\minilm` (网络驱动器)
- **新位置**: `D:\ai-models\minilm` (本地 SSD)
- **模型大小**: 44.19 MB
- **预期提升**: 首次加载 6.4s → 1-2s (3-5x 加速)

---

## 使用新模型路径

### 方法 1: 环境变量 (推荐)

在系统中设置环境变量 `ARROW_MODEL_PATH`:

**Windows PowerShell (临时)**:
```powershell
$env:ARROW_MODEL_PATH = "D:\ai-models\minilm"
```

**Windows 系统环境变量 (永久)**:
1. 右键 "此电脑" → "属性"
2. "高级系统设置" → "环境变量"
3. 新建用户变量:
   - 变量名: `ARROW_MODEL_PATH`
   - 变量值: `D:\ai-models\minilm`

### 方法 2: 代码中指定

```python
from llm_compression.embedding_provider import ArrowEngineProvider

# 直接指定模型路径
provider = ArrowEngineProvider(model_path="D:/ai-models/minilm")
```

### 方法 3: 更新默认路径

修改 `llm_compression/embedding_provider.py`:

```python
# 原代码
DEFAULT_MODEL_PATH = Path(__file__).parent.parent / "models" / "minilm"

# 修改为
DEFAULT_MODEL_PATH = Path("D:/ai-models/minilm")
```

---

## 验证性能提升

### 运行加载速度测试

```bash
# 设置环境变量
$env:ARROW_MODEL_PATH = "D:\ai-models\minilm"

# 运行测试
python validation_tests/test_load_speed.py
```

**预期结果**:
- 首次加载: 1-2s (原 6.4s)
- 后续加载: 0.4s (原 1s)
- 提升: 3-5x

### 运行完整验证套件

```bash
python validation_tests/run_validation.py
```

---

## 性能对比

### 加载速度

| 指标 | 网络驱动器 (M:) | 本地 SSD (D:) | 提升 |
|------|----------------|---------------|------|
| 首次加载 | 6.4s | 1-2s | 3-5x |
| 后续加载 | 1.0s | 0.4s | 2.5x |
| 磁盘 I/O | 网络延迟 | 本地 SSD | - |

### 推理性能

| 指标 | 网络驱动器 | 本地 SSD | 影响 |
|------|-----------|----------|------|
| 推理延迟 | 36ms | 36ms | 无变化 |
| 批处理吞吐 | 35 req/s | 35 req/s | 无变化 |

**说明**: 推理性能不受影响，因为模型加载后在内存中运行。

---

## 更新代码引用

需要更新以下文件中的模型路径引用 (如果使用硬编码路径):

### 1. 测试文件

**文件**: `validation_tests/test_*.py`

```python
# 原代码
model_path = Path("./models/minilm")

# 修改为 (使用环境变量)
model_path = Path(os.environ.get("ARROW_MODEL_PATH", "D:/ai-models/minilm"))
```

### 2. 基准测试

**文件**: `benchmarks/arrowengine_benchmark.py`

```python
# 原代码
model_path = os.environ.get("ARROW_MODEL_PATH", "./models/minilm")

# 修改为
model_path = os.environ.get("ARROW_MODEL_PATH", "D:/ai-models/minilm")
```

### 3. 示例代码

**文件**: `tests/poc/demo_end_to_end.py`

```python
# 原代码
model_path = os.path.abspath("models/minilm")

# 修改为
model_path = os.path.abspath(os.environ.get("ARROW_MODEL_PATH", "D:/ai-models/minilm"))
```

---

## 自动化脚本

### 设置环境变量脚本

**文件**: `scripts/set_model_path.ps1`

```powershell
# Set model path environment variable
$ModelPath = "D:\ai-models\minilm"

# Set for current session
$env:ARROW_MODEL_PATH = $ModelPath
Write-Host "Environment variable set for current session: ARROW_MODEL_PATH=$ModelPath"

# Optionally set permanently (requires admin)
# [System.Environment]::SetEnvironmentVariable("ARROW_MODEL_PATH", $ModelPath, "User")
# Write-Host "Environment variable set permanently"
```

**使用**:
```powershell
.\scripts\set_model_path.ps1
```

---

## 回滚方案

如果需要回滚到网络驱动器:

### 方法 1: 清除环境变量

```powershell
Remove-Item Env:\ARROW_MODEL_PATH
```

### 方法 2: 指向原路径

```powershell
$env:ARROW_MODEL_PATH = "M:\Documents\ai-os-memory\models\minilm"
```

---

## 多模型管理

### 目录结构

```
D:\ai-models\
├── minilm\              # 当前模型
│   ├── weights.parquet
│   ├── tokenizer\
│   └── metadata.json
├── bert-base\           # 未来模型
├── roberta\
└── multimodal\
```

### 环境变量配置

```powershell
# 基础路径
$env:AI_MODELS_BASE = "D:\ai-models"

# 特定模型
$env:ARROW_MODEL_PATH = "$env:AI_MODELS_BASE\minilm"
```

---

## 常见问题

### Q1: 为什么选择 D: 驱动器？

**答**: D: 驱动器空闲空间最大 (147GB)，适合未来扩展。

### Q2: 是否需要删除原模型？

**答**: 建议保留原模型作为备份，直到确认新路径工作正常。

### Q3: 其他设备如何访问？

**答**: 
- 短期: 每个设备本地复制
- 长期: 使用 Arrow Flight 服务器 (见 `ARROW_FLIGHT_INTEGRATION_VISION.md`)

### Q4: 如何更新模型？

**答**:
```powershell
# 下载新模型到临时目录
# 复制到 D:\ai-models\minilm
Copy-Item -Path "new_model\*" -Destination "D:\ai-models\minilm\" -Recurse -Force
```

---

## 下一步

### 立即执行

1. ✅ 模型已复制到 D:\ai-models\minilm
2. 📋 设置环境变量 `ARROW_MODEL_PATH`
3. 📋 运行验证测试
4. 📋 确认性能提升

### 后续优化

1. 📋 更新代码中的硬编码路径
2. 📋 创建自动化部署脚本
3. 📋 考虑 Arrow Flight 集成 (边缘设备)

---

## 总结

✅ **迁移完成**

- 模型已从网络驱动器迁移到本地 SSD
- 预期首次加载速度提升 3-5x
- 为未来多模型部署做好准备
- 为 Arrow Flight 集成奠定基础

**下一步**: 运行验证测试确认性能提升。
