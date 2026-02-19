# ArrowEngine 硬件环境部署验证

这个目录包含了 ArrowEngine 在你的硬件环境下的完整验证测试套件。

## 快速开始

### 1. 确保模型已转换

如果还没有转换模型，先运行：

```bash
python -m llm_compression.tools.cli convert \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output ./models/minilm
```

### 2. 运行所有验证测试

```bash
cd validation_tests
python run_validation.py
```

这将运行所有 8 个测试并生成详细报告。

## 测试套件

### 必需测试（P0）

1. **环境检查** (`test_environment.py`)
   - 检查 Python 版本、依赖库、系统资源
   - 预计时间: 10 秒

2. **模型加载速度** (`test_load_speed.py`)
   - 测试模型加载时间
   - 目标: < 500ms
   - 预计时间: 30 秒

3. **推理延迟** (`test_inference_latency.py`)
   - 测试单次推理延迟
   - 目标: 中位数 < 10ms
   - 预计时间: 1 分钟

4. **批量吞吐量** (`test_batch_throughput.py`)
   - 测试批量处理吞吐量
   - 目标: > 1000 请求/秒
   - 预计时间: 2 分钟

5. **EmbeddingProvider 接口** (`test_embedding_provider.py`)
   - 测试统一接口
   - 预计时间: 30 秒

6. **ArrowStorage 集成** (`test_arrow_storage_integration.py`)
   - 测试存储集成
   - 预计时间: 30 秒

### 可选测试（P1）

7. **内存占用** (`test_memory_usage.py`)
   - 测试内存占用
   - 目标: < 150MB
   - 需要: psutil
   - 预计时间: 30 秒

8. **精度验证** (`test_precision_validation.py`)
   - 对比 sentence-transformers
   - 目标: 相似度 ≥ 0.99
   - 需要: sentence-transformers
   - 预计时间: 1 分钟

## 单独运行测试

你也可以单独运行某个测试：

```bash
# 只运行环境检查
python test_environment.py

# 只运行性能测试
python test_inference_latency.py
```

## 验收标准

### 必须通过（P0）
- [ ] 环境检查通过
- [ ] 模型加载时间 < 500ms
- [ ] 推理延迟中位数 < 10ms
- [ ] EmbeddingProvider 接口正常
- [ ] ArrowStorage 集成正常

### 应该通过（P1）
- [ ] 模型加载时间 < 100ms
- [ ] 推理延迟中位数 < 5ms
- [ ] 批量吞吐量 > 2000 rps
- [ ] 内存占用 < 100MB
- [ ] 精度相似度 ≥ 0.99

## 故障排查

### 问题 1: 模型文件不存在

**错误**: `FileNotFoundError: Model path not found`

**解决方案**:
```bash
python -m llm_compression.tools.cli convert \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output ./models/minilm
```

### 问题 2: 依赖库缺失

**错误**: `ModuleNotFoundError: No module named 'xxx'`

**解决方案**:
```bash
pip install -r requirements.txt
pip install -e .
```

### 问题 3: 性能不达标

**可能原因**:
- CPU 性能不足
- 磁盘 I/O 慢（HDD vs SSD）
- 后台程序占用资源
- 防病毒软件扫描

**解决方案**:
1. 使用 SSD 存储模型
2. 关闭不必要的后台程序
3. 将模型目录添加到防病毒白名单
4. 多次运行取平均值

### 问题 4: 精度不匹配

**可能原因**:
- 模型转换错误
- 权重损坏

**解决方案**:
```bash
# 重新转换模型
rm -rf ./models/minilm
python -m llm_compression.tools.cli convert \
    --model sentence-transformers/all-MiniLM-L6-v2 \
    --output ./models/minilm
```

## 输出报告

测试完成后会生成：

1. **控制台输出**: 实时测试结果
2. **VALIDATION_REPORT.md**: 详细测试报告（保存在项目根目录）

## 下一步

### 如果所有必需测试通过 ✅

恭喜！ArrowEngine 已准备好在你的硬件环境下运行。

**选择下一步**:

**选项 A**: 完成 phase-2-quality-optimization 剩余任务
- Task 13: 文档完善
- Task 14: 生产部署

**选项 B**: 开始 multimodal-encoder-system 实现
- 扩展 ArrowEngine 支持视觉和音频
- 13 个主要任务

### 如果测试失败 ❌

1. 查看失败的测试输出
2. 参考上面的故障排查部分
3. 解决问题后重新运行
4. 如需帮助，提供详细错误日志

## 预计总时间

- **最快**: 5-10 分钟（所有测试通过）
- **典型**: 15-20 分钟（包括故障排查）
- **最长**: 30-40 分钟（需要重新转换模型）

## 技术支持

如果遇到问题：

1. 检查 `ARROWENGINE_DEPLOYMENT_VALIDATION_PLAN.md` 的详细说明
2. 查看 `ARROWENGINE_VERIFICATION_REPORT.md` 了解系统状态
3. 运行 `python test_environment.py` 检查环境配置

---

**准备好了吗？** 运行 `python run_validation.py` 开始验证！
