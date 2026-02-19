"""
测试成本监控模块

测试 CostMonitor 的所有功能，包括：
- 成本记录
- 成本汇总
- 报告生成
- 优化建议
"""

import pytest
import time
import tempfile
from pathlib import Path
from datetime import datetime

from llm_compression.cost_monitor import (
    CostMonitor,
    ModelType,
    CostEntry,
    CostSummary
)


class TestCostEntry:
    """测试 CostEntry 数据类"""
    
    def test_cost_entry_creation(self):
        """测试创建成本记录"""
        entry = CostEntry(
            timestamp=time.time(),
            model_type=ModelType.CLOUD_API,
            model_name="claude-opus-4",
            tokens_used=1000,
            cost=0.001,
            operation="compress",
            success=True
        )
        
        assert entry.model_type == ModelType.CLOUD_API
        assert entry.model_name == "claude-opus-4"
        assert entry.tokens_used == 1000
        assert entry.cost == 0.001
        assert entry.operation == "compress"
        assert entry.success is True
    
    def test_cost_entry_to_dict(self):
        """测试成本记录转换为字典"""
        entry = CostEntry(
            timestamp=1234567890.0,
            model_type=ModelType.LOCAL_MODEL,
            model_name="llama-3-8b",
            tokens_used=500,
            cost=0.00005,
            operation="reconstruct",
            success=True
        )
        
        entry_dict = entry.to_dict()
        
        assert entry_dict["timestamp"] == 1234567890.0
        assert entry_dict["model_type"] == "local_model"
        assert entry_dict["model_name"] == "llama-3-8b"
        assert entry_dict["tokens_used"] == 500
        assert entry_dict["cost"] == 0.00005
        assert entry_dict["operation"] == "reconstruct"
        assert entry_dict["success"] is True


class TestCostSummary:
    """测试 CostSummary 数据类"""
    
    def test_cost_summary_creation(self):
        """测试创建成本汇总"""
        summary = CostSummary(
            total_cost=1.5,
            cloud_cost=1.0,
            local_cost=0.5,
            total_tokens=150000,
            cloud_tokens=100000,
            local_tokens=50000,
            total_operations=150,
            cloud_operations=100,
            local_operations=50,
            savings=0.5,
            savings_percentage=25.0
        )
        
        assert summary.total_cost == 1.5
        assert summary.cloud_cost == 1.0
        assert summary.local_cost == 0.5
        assert summary.total_tokens == 150000
        assert summary.savings == 0.5
        assert summary.savings_percentage == 25.0
    
    def test_cost_summary_to_dict(self):
        """测试成本汇总转换为字典"""
        summary = CostSummary(
            total_cost=2.0,
            cloud_cost=1.5,
            local_cost=0.5
        )
        
        summary_dict = summary.to_dict()
        
        assert summary_dict["total_cost"] == 2.0
        assert summary_dict["cloud_cost"] == 1.5
        assert summary_dict["local_cost"] == 0.5
        assert "total_tokens" in summary_dict
        assert "savings" in summary_dict


class TestCostMonitor:
    """测试 CostMonitor 类"""
    
    @pytest.fixture
    def monitor(self):
        """创建测试用的成本监控器"""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_file = Path(tmpdir) / "cost.log"
            monitor = CostMonitor(log_file=str(log_file), enable_logging=True)
            yield monitor
    
    @pytest.fixture
    def monitor_no_log(self):
        """创建不记录日志的成本监控器"""
        return CostMonitor(enable_logging=False)
    
    def test_monitor_initialization(self, monitor):
        """测试监控器初始化"""
        assert monitor.enable_logging is True
        assert monitor.log_file is not None
        assert len(monitor.cost_entries) == 0
        assert monitor.total_gpu_hours == 0.0
    
    def test_record_cloud_api_operation(self, monitor_no_log):
        """测试记录云端 API 操作"""
        monitor_no_log.record_operation(
            model_type=ModelType.CLOUD_API,
            model_name="claude-opus-4",
            tokens_used=1000,
            operation="compress",
            success=True
        )
        
        assert len(monitor_no_log.cost_entries) == 1
        entry = monitor_no_log.cost_entries[0]
        
        assert entry.model_type == ModelType.CLOUD_API
        assert entry.model_name == "claude-opus-4"
        assert entry.tokens_used == 1000
        assert entry.cost == 0.001  # 1000 tokens * $0.001/1K
        assert entry.operation == "compress"
        assert entry.success is True
    
    def test_record_local_model_operation(self, monitor_no_log):
        """测试记录本地模型操作"""
        monitor_no_log.record_operation(
            model_type=ModelType.LOCAL_MODEL,
            model_name="llama-3-8b",
            tokens_used=2000,
            operation="reconstruct",
            success=True
        )
        
        assert len(monitor_no_log.cost_entries) == 1
        entry = monitor_no_log.cost_entries[0]
        
        assert entry.model_type == ModelType.LOCAL_MODEL
        assert entry.tokens_used == 2000
        assert entry.cost == 0.0002  # 2000 tokens * $0.0001/1K
    
    def test_record_simple_compression_operation(self, monitor_no_log):
        """测试记录简单压缩操作"""
        monitor_no_log.record_operation(
            model_type=ModelType.SIMPLE_COMPRESSION,
            model_name="arrow",
            tokens_used=1000,
            operation="compress",
            success=True
        )
        
        assert len(monitor_no_log.cost_entries) == 1
        entry = monitor_no_log.cost_entries[0]
        
        assert entry.model_type == ModelType.SIMPLE_COMPRESSION
        assert entry.cost == 0.0  # 简单压缩无成本
    
    def test_record_failed_operation(self, monitor_no_log):
        """测试记录失败的操作"""
        monitor_no_log.record_operation(
            model_type=ModelType.CLOUD_API,
            model_name="claude-opus-4",
            tokens_used=1000,
            operation="compress",
            success=False
        )
        
        assert len(monitor_no_log.cost_entries) == 1
        entry = monitor_no_log.cost_entries[0]
        assert entry.success is False
    
    def test_gpu_tracking(self, monitor_no_log):
        """测试 GPU 使用时间跟踪"""
        # 开始跟踪
        monitor_no_log.start_gpu_tracking()
        assert monitor_no_log.gpu_start_time is not None
        
        # 模拟 GPU 使用
        time.sleep(0.1)
        
        # 停止跟踪
        monitor_no_log.stop_gpu_tracking()
        assert monitor_no_log.gpu_start_time is None
        assert monitor_no_log.total_gpu_hours > 0
        
        # 计算 GPU 成本
        gpu_cost = monitor_no_log.get_gpu_cost()
        assert gpu_cost > 0
        assert gpu_cost == monitor_no_log.total_gpu_hours * monitor_no_log.GPU_COST_PER_HOUR
    
    def test_get_summary_empty(self, monitor_no_log):
        """测试获取空汇总"""
        summary = monitor_no_log.get_summary()
        
        assert summary.total_cost == 0.0
        assert summary.total_tokens == 0
        assert summary.total_operations == 0
    
    def test_get_summary_with_operations(self, monitor_no_log):
        """测试获取包含操作的汇总"""
        # 记录多个操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        monitor_no_log.record_operation(
            ModelType.LOCAL_MODEL, "llama-3-8b", 2000, "compress", True
        )
        monitor_no_log.record_operation(
            ModelType.SIMPLE_COMPRESSION, "arrow", 500, "compress", True
        )
        
        summary = monitor_no_log.get_summary()
        
        assert summary.total_operations == 3
        assert summary.cloud_operations == 1
        assert summary.local_operations == 1
        
        assert summary.total_tokens == 3500
        assert summary.cloud_tokens == 1000
        assert summary.local_tokens == 2000
        
        assert summary.cloud_cost == 0.001
        assert summary.local_cost == 0.0002
        assert abs(summary.total_cost - 0.0012) < 1e-10  # 使用浮点数容差
    
    def test_get_summary_excludes_failed_operations(self, monitor_no_log):
        """测试汇总排除失败的操作"""
        # 记录成功和失败的操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", False
        )
        
        summary = monitor_no_log.get_summary()
        
        # 只计算成功的操作
        assert summary.total_operations == 1
        assert summary.total_tokens == 1000
        assert summary.cloud_cost == 0.001
    
    def test_get_summary_with_time_range(self, monitor_no_log):
        """测试获取指定时间范围的汇总"""
        now = time.time()
        
        # 记录不同时间的操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        
        # 获取未来时间范围的汇总（应该为空）
        summary = monitor_no_log.get_summary(
            start_time=now + 100,
            end_time=now + 200
        )
        assert summary.total_operations == 0
        
        # 获取包含操作的时间范围
        summary = monitor_no_log.get_summary(
            start_time=now - 100,
            end_time=now + 100
        )
        assert summary.total_operations == 1
    
    def test_get_summary_calculates_savings(self, monitor_no_log):
        """测试汇总计算节省金额"""
        # 记录混合操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        monitor_no_log.record_operation(
            ModelType.LOCAL_MODEL, "llama-3-8b", 9000, "compress", True
        )
        
        summary = monitor_no_log.get_summary()
        
        # 全部使用云端 API 的成本: 10000 * 0.001 / 1000 = 0.01
        # 实际成本: 0.001 + 0.0009 = 0.0019
        # 节省: 0.01 - 0.0019 = 0.0081
        assert summary.total_tokens == 10000
        assert summary.savings > 0
        assert summary.savings_percentage > 0
    
    def test_get_daily_summary(self, monitor_no_log):
        """测试获取每日汇总"""
        # 记录操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        
        daily_summaries = monitor_no_log.get_daily_summary(days=7)
        
        assert len(daily_summaries) == 7
        assert all(isinstance(date, str) for date in daily_summaries.keys())
        assert all(isinstance(summary, CostSummary) for summary in daily_summaries.values())
    
    def test_get_weekly_summary(self, monitor_no_log):
        """测试获取每周汇总"""
        # 记录操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        
        summary = monitor_no_log.get_weekly_summary()
        
        assert isinstance(summary, CostSummary)
        assert summary.total_operations == 1
    
    def test_get_monthly_summary(self, monitor_no_log):
        """测试获取每月汇总"""
        # 记录操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        
        summary = monitor_no_log.get_monthly_summary()
        
        assert isinstance(summary, CostSummary)
        assert summary.total_operations == 1
    
    def test_generate_report_day(self, monitor_no_log):
        """测试生成每日报告"""
        # 记录操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        
        report = monitor_no_log.generate_report(period="day")
        
        assert "每日成本报告" in report
        assert "总成本" in report
        assert "云端 API 成本" in report
        assert "本地模型成本" in report
        assert "Token 使用" in report
        assert "操作统计" in report
        assert "成本节省" in report
    
    def test_generate_report_week(self, monitor_no_log):
        """测试生成每周报告"""
        report = monitor_no_log.generate_report(period="week")
        assert "每周成本报告" in report
    
    def test_generate_report_month(self, monitor_no_log):
        """测试生成每月报告"""
        report = monitor_no_log.generate_report(period="month")
        assert "每月成本报告" in report
    
    def test_generate_report_to_file(self, monitor_no_log):
        """测试生成报告到文件"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "report.txt"
            
            monitor_no_log.record_operation(
                ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
            )
            
            report = monitor_no_log.generate_report(
                period="day",
                output_file=str(output_file)
            )
            
            assert output_file.exists()
            content = output_file.read_text(encoding="utf-8")
            assert content == report
    
    def test_optimize_model_selection_high_cloud_usage(self, monitor_no_log):
        """测试优化建议 - 云端使用过高"""
        # 记录大量云端 API 操作
        for _ in range(10):
            monitor_no_log.record_operation(
                ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
            )
        
        # 记录少量本地模型操作
        for _ in range(2):
            monitor_no_log.record_operation(
                ModelType.LOCAL_MODEL, "llama-3-8b", 1000, "compress", True
            )
        
        recommendations = monitor_no_log.optimize_model_selection()
        
        assert "recommendations" in recommendations
        assert len(recommendations["recommendations"]) > 0
        
        # 应该建议增加本地模型使用
        rec_types = [r["type"] for r in recommendations["recommendations"]]
        assert "increase_local_usage" in rec_types
    
    def test_optimize_model_selection_low_savings(self, monitor_no_log):
        """测试优化建议 - 节省低于预期"""
        # 记录操作，使节省比例低于 80%
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 8000, "compress", True
        )
        monitor_no_log.record_operation(
            ModelType.LOCAL_MODEL, "llama-3-8b", 2000, "compress", True
        )
        
        recommendations = monitor_no_log.optimize_model_selection()
        
        # 应该有优化建议
        assert len(recommendations["recommendations"]) > 0
    
    def test_optimize_model_selection_high_gpu_cost(self, monitor_no_log):
        """测试优化建议 - GPU 成本过高"""
        # 模拟长时间 GPU 使用
        monitor_no_log.total_gpu_hours = 10.0  # 10 小时
        
        # 记录少量本地模型操作
        monitor_no_log.record_operation(
            ModelType.LOCAL_MODEL, "llama-3-8b", 1000, "compress", True
        )
        
        recommendations = monitor_no_log.optimize_model_selection()
        
        # 应该建议优化 GPU 使用
        rec_types = [r["type"] for r in recommendations["recommendations"]]
        assert "optimize_gpu_usage" in rec_types
    
    def test_clear(self, monitor_no_log):
        """测试清除成本记录"""
        # 记录操作
        monitor_no_log.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        monitor_no_log.total_gpu_hours = 5.0
        
        assert len(monitor_no_log.cost_entries) == 1
        assert monitor_no_log.total_gpu_hours == 5.0
        
        # 清除
        monitor_no_log.clear()
        
        assert len(monitor_no_log.cost_entries) == 0
        assert monitor_no_log.total_gpu_hours == 0.0
        assert monitor_no_log.gpu_start_time is None
    
    def test_log_file_writing(self, monitor):
        """测试日志文件写入"""
        # 记录操作
        monitor.record_operation(
            ModelType.CLOUD_API, "claude-opus-4", 1000, "compress", True
        )
        
        # 检查日志文件
        assert monitor.log_file.exists()
        content = monitor.log_file.read_text()
        assert "claude-opus-4" in content
        assert "1000" in content


class TestCostMonitorIntegration:
    """集成测试"""
    
    def test_realistic_usage_scenario(self):
        """测试真实使用场景"""
        monitor = CostMonitor(enable_logging=False)
        
        # 模拟一天的使用
        # 早上：使用云端 API 处理重要任务
        for _ in range(10):
            monitor.record_operation(
                ModelType.CLOUD_API, "claude-opus-4", 1500, "compress", True
            )
        
        # 白天：主要使用本地模型
        for _ in range(50):
            monitor.record_operation(
                ModelType.LOCAL_MODEL, "llama-3-8b", 1000, "compress", True
            )
        
        # 晚上：使用简单压缩
        for _ in range(20):
            monitor.record_operation(
                ModelType.SIMPLE_COMPRESSION, "arrow", 500, "compress", True
            )
        
        # 获取汇总
        summary = monitor.get_summary()
        
        assert summary.total_operations == 80
        assert summary.cloud_operations == 10
        assert summary.local_operations == 50
        
        # 验证成本计算
        expected_cloud_cost = (10 * 1500 / 1000) * 0.001
        expected_local_cost = (50 * 1000 / 1000) * 0.0001
        
        assert abs(summary.cloud_cost - expected_cloud_cost) < 0.0001
        assert abs(summary.local_cost - expected_local_cost) < 0.0001
        
        # 验证节省
        assert summary.savings > 0
        assert summary.savings_percentage > 50  # 应该节省超过 50%
        
        # 生成报告
        report = monitor.generate_report(period="day")
        assert len(report) > 0
        
        # 获取优化建议
        recommendations = monitor.optimize_model_selection()
        assert "recommendations" in recommendations


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
