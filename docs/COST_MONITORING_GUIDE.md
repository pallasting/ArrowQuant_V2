# 成本监控系统使用指南

**版本**: 1.0  
**最后更新**: 2026-02-17  
**状态**: ✅ 已完成 (Task 11)

---

## 概述

Phase 2.0 成本监控系统提供全面的 API 成本跟踪、GPU 使用监控和成本优化建议。

### 核心功能

1. **实时成本跟踪** - 记录每次 API 调用和本地模型使用的成本
2. **GPU 成本监控** - 跟踪 GPU 使用时间和电力成本
3. **成本汇总报告** - 生成日/周/月成本报告
4. **优化建议** - 基于使用模式提供成本优化建议
5. **成本告警** - 当成本超过阈值时发出警告

---

## 快速开始

### 基础使用

```python
from llm_compression.cost_monitor import CostMonitor, ModelType

# 创建成本监控器
monitor = CostMonitor(
    log_file="logs/cost/cost.jsonl",
    enable_logging=True
)

# 记录云端 API 操作
monitor.record_operation(
    model_type=ModelType.CLOUD_API,
    model_name="claude-opus-4",
    tokens_used=1500,
    operation="compress",
    success=True
)

# 记录本地模型操作
monitor.record_operation(
    model_type=ModelType.LOCAL_MODEL,
    model_name="qwen2.5:7b-instruct",
    tokens_used=1200,
    operation="compress",
    success=True
)

# 获取成本汇总
summary = monitor.get_summary()
print(f"总成本: ${summary.total_cost:.6f}")
print(f"成本节省: {summary.savings_percentage:.1f}%")
```

---

## 核心组件

### 1. CostMonitor 类

主要的成本监控类，负责记录和分析成本。

#### 初始化参数

```python
CostMonitor(
    log_file: Optional[str] = None,      # 成本日志文件路径
    enable_logging: bool = True          # 是否启用日志记录
)
```

#### 主要方法

##### record_operation()

记录单次操作的成本。

```python
monitor.record_operation(
    model_type: ModelType,      # 模型类型（CLOUD_API/LOCAL_MODEL/SIMPLE_COMPRESSION）
    model_name: str,            # 模型名称
    tokens_used: int,           # 使用的 token 数
    operation: str = "compress", # 操作类型
    success: bool = True        # 是否成功
)
```

##### get_summary()

获取指定时间范围的成本汇总。

```python
summary = monitor.get_summary(
    start_time: Optional[float] = None,  # 开始时间（Unix 时间戳）
    end_time: Optional[float] = None     # 结束时间（Unix 时间戳）
)

# 返回 CostSummary 对象
print(f"总成本: ${summary.total_cost:.6f}")
print(f"云端成本: ${summary.cloud_cost:.6f}")
print(f"本地成本: ${summary.local_cost:.6f}")
print(f"总 tokens: {summary.total_tokens:,}")
print(f"成本节省: ${summary.savings:.6f} ({summary.savings_percentage:.1f}%)")
```

##### generate_report()

生成成本报告。

```python
report = monitor.generate_report(
    period: str = "week",              # 时间周期（day/week/month）
    output_file: Optional[str] = None  # 输出文件路径（可选）
)

print(report)
```

##### optimize_model_selection()

获取成本优化建议。

```python
recommendations = monitor.optimize_model_selection()

print(f"当前策略: {recommendations['current_strategy']}")
print(f"潜在节省: ${recommendations['potential_savings']:.6f}")

for rec in recommendations['recommendations']:
    print(f"类型: {rec['type']}")
    print(f"原因: {rec['reason']}")
    print(f"行动: {rec['action']}")
    print(f"潜在节省: ${rec['potential_savings']:.6f}")
```

---

### 2. GPU 成本跟踪

跟踪 GPU 使用时间和电力成本。

```python
# 开始 GPU 跟踪
monitor.start_gpu_tracking()

# 执行 GPU 密集型操作
# ... 本地模型推理 ...

# 停止 GPU 跟踪
monitor.stop_gpu_tracking()

# 获取 GPU 成本
gpu_cost = monitor.get_gpu_cost()
print(f"GPU 成本: ${gpu_cost:.6f}")
```

**GPU 成本计算**:
- 默认成本: $0.50/小时（AMD Mi50 电费估算）
- 可通过修改 `CostMonitor.GPU_COST_PER_HOUR` 调整

---

### 3. 成本常量

系统使用以下成本常量（美元/1K tokens）:

```python
CLOUD_API_COST_PER_1K = 0.001      # 云端 API
LOCAL_MODEL_COST_PER_1K = 0.0001   # 本地模型（电费）
SIMPLE_COMPRESSION_COST_PER_1K = 0.0  # 简单压缩（无 LLM）
```

**实际成本示例**:
- Claude Opus-4: 1000 tokens = $0.001
- 本地 Qwen2.5-7B: 1000 tokens = $0.0001
- Arrow 压缩: 无 LLM 成本

---

## 集成示例

### 与 ProtocolAdapter 集成

```python
from llm_compression.protocol_adapter import ProtocolAdapter
from llm_compression.cost_monitor import CostMonitor, ModelType

class CostAwareAdapter:
    def __init__(self, api_key: str):
        self.adapter = ProtocolAdapter(api_key=api_key)
        self.cost_monitor = CostMonitor()
    
    async def complete_with_monitoring(self, prompt: str, model: str):
        # 调用 API
        result = await self.adapter.complete(prompt, model=model)
        
        # 记录成本
        tokens_used = len(prompt.split()) + len(result.split())
        self.cost_monitor.record_operation(
            model_type=ModelType.CLOUD_API,
            model_name=model,
            tokens_used=tokens_used,
            operation="complete",
            success=True
        )
        
        return result
```

### 与 ModelRouter 集成

```python
from llm_compression.model_router import ModelRouter
from llm_compression.cost_monitor import CostMonitor, ModelType

class CostOptimizedRouter:
    def __init__(self):
        self.router = ModelRouter()
        self.cost_monitor = CostMonitor()
    
    def select_model_with_cost_awareness(self, text: str):
        # 获取成本汇总
        summary = self.cost_monitor.get_summary()
        
        # 如果云端成本过高，优先使用本地模型
        if summary.cloud_cost > 1.0:  # $1 阈值
            return "qwen2.5:7b-instruct"  # 本地模型
        
        # 否则使用路由器选择
        model_info = self.router.select_model(
            text_length=len(text),
            quality_requirement=0.85
        )
        
        return model_info["model"]
```

---

## 成本报告

### 日报告

```python
report = monitor.generate_report(period="day")
```

**输出示例**:
```
============================================================
每日成本报告
============================================================
生成时间: 2026-02-17 14:30:00

成本汇总:
  - 总成本: $0.1234
  - 云端 API 成本: $0.0800
  - 本地模型成本: $0.0434
  - GPU 成本: $0.0250

Token 使用:
  - 总 tokens: 123,456
  - 云端 API tokens: 80,000
  - 本地模型 tokens: 43,456

操作统计:
  - 总操作数: 150
  - 云端 API 操作: 50
  - 本地模型操作: 100

成本节省:
  - 节省金额: $0.0456
  - 节省比例: 27.0%

============================================================
```

### 周报告

```python
report = monitor.generate_report(period="week")
```

### 月报告

```python
report = monitor.generate_report(period="month")
```

### 保存报告到文件

```python
report = monitor.generate_report(
    period="week",
    output_file="reports/weekly_cost_report.txt"
)
```

---

## 成本优化建议

系统会自动分析使用模式并提供优化建议。

### 建议类型

#### 1. 增加本地模型使用

**触发条件**: 云端 API 使用率 > 50%

```python
{
    "type": "increase_local_usage",
    "reason": "云端 API 使用率过高 (65.0%)",
    "action": "增加本地模型优先级",
    "potential_savings": 0.0450
}
```

#### 2. 优化策略

**触发条件**: 成本节省 < 80%

```python
{
    "type": "optimize_strategy",
    "reason": "成本节省低于预期 (45.0% < 80%)",
    "action": "优先使用本地模型，仅在高质量要求时使用云端 API",
    "potential_savings": 0.0300
}
```

#### 3. 优化 GPU 使用

**触发条件**: GPU 成本 > 本地模型成本的 50%

```python
{
    "type": "optimize_gpu_usage",
    "reason": "GPU 成本过高 ($0.0500)",
    "action": "优化批量处理，减少 GPU 空闲时间",
    "potential_savings": 0.0150
}
```

---

## 成本告警

### 设置成本阈值

```python
class CostAlertMonitor(CostMonitor):
    def __init__(self, daily_budget: float = 1.0, **kwargs):
        super().__init__(**kwargs)
        self.daily_budget = daily_budget
    
    def check_budget(self):
        """检查是否超出预算"""
        summary = self.get_summary(
            start_time=time.time() - 86400  # 最近 24 小时
        )
        
        if summary.total_cost > self.daily_budget:
            self.send_alert(
                f"⚠️ 成本告警: 日成本 ${summary.total_cost:.4f} "
                f"超出预算 ${self.daily_budget:.4f}"
            )
    
    def send_alert(self, message: str):
        """发送告警（可以集成邮件、Slack 等）"""
        print(f"[ALERT] {message}")
        # 这里可以集成实际的告警系统
```

---

## 最佳实践

### 1. 定期生成报告

```python
import schedule

def generate_daily_report():
    monitor = CostMonitor()
    report = monitor.generate_report(
        period="day",
        output_file=f"reports/cost_{datetime.now().strftime('%Y%m%d')}.txt"
    )

# 每天凌晨 1 点生成报告
schedule.every().day.at("01:00").do(generate_daily_report)
```

### 2. 实时成本监控

```python
class RealtimeCostMonitor:
    def __init__(self, alert_threshold: float = 0.1):
        self.monitor = CostMonitor()
        self.alert_threshold = alert_threshold
        self.last_check = time.time()
    
    def check_and_alert(self):
        """每小时检查成本"""
        now = time.time()
        if now - self.last_check > 3600:  # 1 小时
            summary = self.monitor.get_summary(
                start_time=self.last_check,
                end_time=now
            )
            
            if summary.total_cost > self.alert_threshold:
                print(f"⚠️ 小时成本: ${summary.total_cost:.4f}")
            
            self.last_check = now
```

### 3. 成本优化循环

```python
def optimize_cost_continuously():
    """持续优化成本"""
    monitor = CostMonitor()
    
    while True:
        # 每小时检查一次
        time.sleep(3600)
        
        # 获取优化建议
        recommendations = monitor.optimize_model_selection()
        
        # 应用建议
        if recommendations['recommendations']:
            print("应用成本优化建议...")
            for rec in recommendations['recommendations']:
                apply_recommendation(rec)
```

---

## 性能影响

成本监控系统设计为低开销：

- **记录操作**: < 1ms
- **生成汇总**: < 10ms (1000 条记录)
- **生成报告**: < 50ms
- **内存占用**: ~1MB (10,000 条记录)

---

## 故障排查

### 问题 1: 日志文件未创建

**原因**: 日志目录不存在

**解决方案**:
```python
from pathlib import Path

log_file = Path("logs/cost/cost.jsonl")
log_file.parent.mkdir(parents=True, exist_ok=True)

monitor = CostMonitor(log_file=str(log_file))
```

### 问题 2: GPU 成本计算不准确

**原因**: GPU 跟踪未正确启动/停止

**解决方案**:
```python
try:
    monitor.start_gpu_tracking()
    # GPU 操作
    ...
finally:
    monitor.stop_gpu_tracking()
```

### 问题 3: 成本汇总为 0

**原因**: 操作标记为失败（success=False）

**解决方案**:
```python
# 确保成功的操作标记为 True
monitor.record_operation(
    model_type=ModelType.CLOUD_API,
    model_name="claude-opus-4",
    tokens_used=1000,
    operation="compress",
    success=True  # ← 确保为 True
)
```

---

## API 参考

### ModelType 枚举

```python
class ModelType(Enum):
    CLOUD_API = "cloud_api"              # 云端 API（Claude/GPT/Gemini）
    LOCAL_MODEL = "local_model"          # 本地模型（Qwen/Llama）
    SIMPLE_COMPRESSION = "simple_compression"  # 简单压缩（Arrow/ZSTD）
```

### CostEntry 数据类

```python
@dataclass
class CostEntry:
    timestamp: float        # Unix 时间戳
    model_type: ModelType   # 模型类型
    model_name: str         # 模型名称
    tokens_used: int        # 使用的 token 数
    cost: float             # 成本（美元）
    operation: str          # 操作类型
    success: bool           # 是否成功
```

### CostSummary 数据类

```python
@dataclass
class CostSummary:
    total_cost: float = 0.0           # 总成本
    cloud_cost: float = 0.0           # 云端成本
    local_cost: float = 0.0           # 本地成本
    
    total_tokens: int = 0             # 总 tokens
    cloud_tokens: int = 0             # 云端 tokens
    local_tokens: int = 0             # 本地 tokens
    
    total_operations: int = 0         # 总操作数
    cloud_operations: int = 0         # 云端操作数
    local_operations: int = 0         # 本地操作数
    
    savings: float = 0.0              # 节省金额
    savings_percentage: float = 0.0   # 节省比例
```

---

## 相关文档

- [Phase 2.0 实施计划](../specs/PHASE_2.0_SPEC/IMPLEMENTATION_PLAN.md)
- [压缩策略决策](COMPRESSION_STRATEGY_DECISION.md)
- [性能监控指南](PERFORMANCE_MONITORING_GUIDE.md)

---

## 更新日志

| 日期 | 版本 | 变更内容 |
|-----|------|---------|
| 2026-02-17 | 1.0 | 初始版本，Task 11 完成 |

---

**文档维护者**: AI-OS 团队  
**最后审核**: 2026-02-17  
**状态**: ✅ 已完成
