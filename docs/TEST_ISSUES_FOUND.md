# 测试问题追踪

**日期**: 2026-02-17  
**状态**: 调查中

---

## NetworkNavigator Arrow 测试问题

### 观察到的问题

**测试文件**: `tests/unit/test_network_navigator_arrow.py`

**运行状态**:
- 总测试数: 15
- 通过: 13
- 失败: 2
- 超时: 测试运行超过 2 分钟

### 失败的测试

#### 1. test_find_similar_vectorized_empty_result
**状态**: 超时/失败  
**位置**: Line ~145  
**可能原因**:
- sentence-transformers 模型加载时间过长
- 测试逻辑可能有无限循环
- 高阈值导致计算时间过长

#### 2. test_large_scale_retrieve  
**状态**: 失败  
**位置**: TestPerformance 类  
**可能原因**:
- 性能测试阈值设置过严格
- 1000 个记忆的处理时间超出预期
- 内存或 CPU 资源限制

---

## 根本原因分析

### 可能的问题

1. **模型加载开销**
   - sentence-transformers 首次加载需要 3-5 秒
   - 每个测试都可能重新加载模型
   - 建议: 使用 pytest fixture 缓存模型

2. **性能测试阈值**
   - 测试可能设置了过于严格的时间限制
   - Windows 环境性能可能不同于开发环境
   - 建议: 放宽性能阈值或使用相对比较

3. **测试隔离问题**
   - 测试之间可能有状态共享
   - 建议: 检查 fixture 的 scope

---

## 建议的修复方案

### 方案 1: 优化测试 Fixture（推荐）

```python
# 在 conftest.py 中添加 session 级别的 embedder
@pytest.fixture(scope="session")
def embedder_arrow_session():
    """Session-scoped embedder to avoid reloading model"""
    return LocalEmbedderArrow()
```

### 方案 2: 放宽性能阈值

```python
# 修改 test_large_scale_retrieve
def test_large_scale_retrieve(self, navigator_arrow, embedder_arrow):
    # ...
    # 原来: assert elapsed_ms < 100
    # 修改为:
    assert elapsed_ms < 200  # 放宽到 200ms
```

### 方案 3: 跳过慢速测试

```python
@pytest.mark.slow
def test_find_similar_vectorized_empty_result(...):
    # 标记为慢速测试，可选择性运行
```

---

## 下一步行动

### 立即执行

1. **检查测试代码**
   - 查看 `test_find_similar_vectorized_empty_result` 实现
   - 确认是否有无限循环或死锁

2. **优化 Fixture**
   - 添加 session-scoped embedder fixture
   - 减少模型重复加载

3. **调整性能阈值**
   - 根据实际环境调整时间限制
   - 使用相对性能比较而非绝对值

### 后续工作

4. **运行其他测试**
   - BatchProcessor Arrow
   - CognitiveLoop Arrow
   - 查看是否有类似问题

5. **性能优化**
   - 如果测试确实太慢，考虑优化实现
   - 或者将慢速测试移到性能测试套件

---

## 临时解决方案

如果需要快速验证功能，可以：

1. **跳过失败的测试**
   ```bash
   pytest tests/unit/test_network_navigator_arrow.py -v -k "not empty_result and not large_scale"
   ```

2. **增加超时时间**
   ```bash
   pytest tests/unit/test_network_navigator_arrow.py -v --timeout=300
   ```

3. **单独运行快速测试**
   ```bash
   pytest tests/unit/test_network_navigator_arrow.py::TestNetworkNavigatorArrow::test_initialization -v
   ```

---

**文档版本**: 1.0  
**创建日期**: 2026-02-17  
**负责人**: AI-OS 团队  
**状态**: 待修复
