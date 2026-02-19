"""
成本监控和优化模块

跟踪云端 API 和本地模型的成本，提供成本优化策略。
Phase 1.1 成本监控。
"""

import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from enum import Enum
import json
from pathlib import Path

from llm_compression.logger import logger


class ModelType(Enum):
    """模型类型"""
    CLOUD_API = "cloud_api"
    LOCAL_MODEL = "local_model"
    SIMPLE_COMPRESSION = "simple_compression"


@dataclass
class CostEntry:
    """成本记录"""
    timestamp: float
    model_type: ModelType
    model_name: str
    tokens_used: int
    cost: float  # 美元
    operation: str  # compress/reconstruct
    success: bool
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "model_type": self.model_type.value,
            "model_name": self.model_name,
            "tokens_used": self.tokens_used,
            "cost": self.cost,
            "operation": self.operation,
            "success": self.success
        }


@dataclass
class CostSummary:
    """成本汇总"""
    total_cost: float = 0.0
    cloud_cost: float = 0.0
    local_cost: float = 0.0
    
    total_tokens: int = 0
    cloud_tokens: int = 0
    local_tokens: int = 0
    
    total_operations: int = 0
    cloud_operations: int = 0
    local_operations: int = 0
    
    savings: float = 0.0  # 相比全部使用云端 API 的节省
    savings_percentage: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "total_cost": self.total_cost,
            "cloud_cost": self.cloud_cost,
            "local_cost": self.local_cost,
            "total_tokens": self.total_tokens,
            "cloud_tokens": self.cloud_tokens,
            "local_tokens": self.local_tokens,
            "total_operations": self.total_operations,
            "cloud_operations": self.cloud_operations,
            "local_operations": self.local_operations,
            "savings": self.savings,
            "savings_percentage": self.savings_percentage
        }


class CostMonitor:
    """成本监控器"""
    
    # 成本常量（美元/1K tokens）
    CLOUD_API_COST_PER_1K = 0.001  # 云端 API
    LOCAL_MODEL_COST_PER_1K = 0.0001  # 本地模型（电费）
    SIMPLE_COMPRESSION_COST_PER_1K = 0.0  # 简单压缩（无 LLM）
    
    # GPU 成本（美元/小时）
    GPU_COST_PER_HOUR = 0.50  # AMD Mi50 电费估算
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        enable_logging: bool = True
    ):
        """
        初始化成本监控器
        
        Args:
            log_file: 成本日志文件路径
            enable_logging: 是否启用日志记录
        """
        self.log_file = Path(log_file) if log_file else None
        self.enable_logging = enable_logging
        
        # 成本记录
        self.cost_entries: List[CostEntry] = []
        
        # GPU 使用时间跟踪
        self.gpu_start_time: Optional[float] = None
        self.total_gpu_hours: float = 0.0
        
        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"CostMonitor initialized: log_file={log_file}, "
            f"enable_logging={enable_logging}"
        )
    
    def record_operation(
        self,
        model_type: ModelType,
        model_name: str,
        tokens_used: int,
        operation: str = "compress",
        success: bool = True
    ):
        """
        记录操作成本
        
        Args:
            model_type: 模型类型
            model_name: 模型名称
            tokens_used: 使用的 token 数
            operation: 操作类型
            success: 是否成功
        """
        # 计算成本
        cost = self._calculate_cost(model_type, tokens_used)
        
        # 创建成本记录
        entry = CostEntry(
            timestamp=time.time(),
            model_type=model_type,
            model_name=model_name,
            tokens_used=tokens_used,
            cost=cost,
            operation=operation,
            success=success
        )
        
        self.cost_entries.append(entry)
        
        # 写入日志
        if self.enable_logging and self.log_file:
            self._write_log(entry)
        
        logger.debug(
            f"Cost recorded: {model_type.value} - {model_name} - "
            f"{tokens_used} tokens - ${cost:.6f}"
        )
    
    def _calculate_cost(self, model_type: ModelType, tokens: int) -> float:
        """
        计算成本
        
        Args:
            model_type: 模型类型
            tokens: token 数量
            
        Returns:
            float: 成本（美元）
        """
        if model_type == ModelType.CLOUD_API:
            return (tokens / 1000) * self.CLOUD_API_COST_PER_1K
        elif model_type == ModelType.LOCAL_MODEL:
            return (tokens / 1000) * self.LOCAL_MODEL_COST_PER_1K
        else:  # SIMPLE_COMPRESSION
            return 0.0
    
    def start_gpu_tracking(self):
        """开始跟踪 GPU 使用时间"""
        if self.gpu_start_time is None:
            self.gpu_start_time = time.time()
            logger.info("GPU tracking started")
    
    def stop_gpu_tracking(self):
        """停止跟踪 GPU 使用时间"""
        if self.gpu_start_time is not None:
            elapsed_hours = (time.time() - self.gpu_start_time) / 3600
            self.total_gpu_hours += elapsed_hours
            self.gpu_start_time = None
            logger.info(f"GPU tracking stopped: {elapsed_hours:.4f} hours")
    
    def get_gpu_cost(self) -> float:
        """
        获取 GPU 成本
        
        Returns:
            float: GPU 成本（美元）
        """
        return self.total_gpu_hours * self.GPU_COST_PER_HOUR
    
    def get_summary(
        self,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None
    ) -> CostSummary:
        """
        获取成本汇总
        
        Args:
            start_time: 开始时间（Unix 时间戳）
            end_time: 结束时间（Unix 时间戳）
            
        Returns:
            CostSummary: 成本汇总
        """
        # 过滤时间范围
        entries = self.cost_entries
        if start_time:
            entries = [e for e in entries if e.timestamp >= start_time]
        if end_time:
            entries = [e for e in entries if e.timestamp <= end_time]
        
        summary = CostSummary()
        
        for entry in entries:
            if not entry.success:
                continue
            
            summary.total_cost += entry.cost
            summary.total_tokens += entry.tokens_used
            summary.total_operations += 1
            
            if entry.model_type == ModelType.CLOUD_API:
                summary.cloud_cost += entry.cost
                summary.cloud_tokens += entry.tokens_used
                summary.cloud_operations += 1
            elif entry.model_type == ModelType.LOCAL_MODEL:
                summary.local_cost += entry.cost
                summary.local_tokens += entry.tokens_used
                summary.local_operations += 1
        
        # 添加 GPU 成本到本地模型成本
        gpu_cost = self.get_gpu_cost()
        summary.local_cost += gpu_cost
        summary.total_cost += gpu_cost
        
        # 计算节省
        # 假设全部使用云端 API 的成本
        total_tokens = summary.total_tokens
        if total_tokens > 0:
            cloud_only_cost = (total_tokens / 1000) * self.CLOUD_API_COST_PER_1K
            summary.savings = cloud_only_cost - summary.total_cost
            summary.savings_percentage = (summary.savings / cloud_only_cost) * 100
        
        return summary
    
    def get_daily_summary(self, days: int = 7) -> Dict[str, CostSummary]:
        """
        获取每日成本汇总
        
        Args:
            days: 天数
            
        Returns:
            Dict[str, CostSummary]: 日期 -> 成本汇总
        """
        now = time.time()
        summaries = {}
        
        for i in range(days):
            day_start = now - (i + 1) * 86400
            day_end = now - i * 86400
            
            date_str = datetime.fromtimestamp(day_start).strftime("%Y-%m-%d")
            summaries[date_str] = self.get_summary(day_start, day_end)
        
        return summaries
    
    def get_weekly_summary(self) -> CostSummary:
        """
        获取本周成本汇总
        
        Returns:
            CostSummary: 本周成本汇总
        """
        now = time.time()
        week_start = now - 7 * 86400
        return self.get_summary(week_start, now)
    
    def get_monthly_summary(self) -> CostSummary:
        """
        获取本月成本汇总
        
        Returns:
            CostSummary: 本月成本汇总
        """
        now = time.time()
        month_start = now - 30 * 86400
        return self.get_summary(month_start, now)
    
    def generate_report(
        self,
        period: str = "week",
        output_file: Optional[str] = None
    ) -> str:
        """
        生成成本报告
        
        Args:
            period: 时间周期（day/week/month）
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        if period == "day":
            summary = self.get_summary(time.time() - 86400, time.time())
            title = "每日成本报告"
        elif period == "week":
            summary = self.get_weekly_summary()
            title = "每周成本报告"
        else:  # month
            summary = self.get_monthly_summary()
            title = "每月成本报告"
        
        # 生成报告
        report_lines = [
            "=" * 60,
            title,
            "=" * 60,
            f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "成本汇总:",
            f"  - 总成本: ${summary.total_cost:.4f}",
            f"  - 云端 API 成本: ${summary.cloud_cost:.4f}",
            f"  - 本地模型成本: ${summary.local_cost:.4f}",
            f"  - GPU 成本: ${self.get_gpu_cost():.4f}",
            "",
            "Token 使用:",
            f"  - 总 tokens: {summary.total_tokens:,}",
            f"  - 云端 API tokens: {summary.cloud_tokens:,}",
            f"  - 本地模型 tokens: {summary.local_tokens:,}",
            "",
            "操作统计:",
            f"  - 总操作数: {summary.total_operations:,}",
            f"  - 云端 API 操作: {summary.cloud_operations:,}",
            f"  - 本地模型操作: {summary.local_operations:,}",
            "",
            "成本节省:",
            f"  - 节省金额: ${summary.savings:.4f}",
            f"  - 节省比例: {summary.savings_percentage:.1f}%",
            "",
            "=" * 60
        ]
        
        report = "\n".join(report_lines)
        
        # 写入文件
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info(f"Cost report saved to {output_file}")
        
        return report
    
    def _write_log(self, entry: CostEntry):
        """写入日志文件"""
        if not self.log_file:
            return
        
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to write cost log: {e}")
    
    def optimize_model_selection(self) -> Dict[str, any]:
        """
        优化模型选择策略
        
        Returns:
            Dict: 优化建议
        """
        summary = self.get_summary()
        
        recommendations = {
            "current_strategy": "hybrid",
            "recommendations": [],
            "potential_savings": 0.0
        }
        
        # 如果云端 API 使用过多，建议增加本地模型使用
        if summary.cloud_operations > summary.local_operations:
            cloud_ratio = summary.cloud_operations / summary.total_operations
            if cloud_ratio > 0.5:
                potential_savings = summary.cloud_cost * 0.9  # 假设可以节省 90%
                recommendations["recommendations"].append({
                    "type": "increase_local_usage",
                    "reason": f"云端 API 使用率过高 ({cloud_ratio:.1%})",
                    "action": "增加本地模型优先级",
                    "potential_savings": potential_savings
                })
                recommendations["potential_savings"] += potential_savings
        
        # 如果成本节省低于预期，建议优化
        if summary.savings_percentage < 80:
            recommendations["recommendations"].append({
                "type": "optimize_strategy",
                "reason": f"成本节省低于预期 ({summary.savings_percentage:.1f}% < 80%)",
                "action": "优先使用本地模型，仅在高质量要求时使用云端 API",
                "potential_savings": summary.cloud_cost * 0.5
            })
        
        # 如果 GPU 成本过高，建议优化
        gpu_cost = self.get_gpu_cost()
        if gpu_cost > summary.local_cost * 0.5:
            recommendations["recommendations"].append({
                "type": "optimize_gpu_usage",
                "reason": f"GPU 成本过高 (${gpu_cost:.4f})",
                "action": "优化批量处理，减少 GPU 空闲时间",
                "potential_savings": gpu_cost * 0.3
            })
        
        return recommendations
    
    def clear(self):
        """清除所有成本记录"""
        self.cost_entries.clear()
        self.total_gpu_hours = 0.0
        self.gpu_start_time = None
        logger.info("Cost records cleared")
