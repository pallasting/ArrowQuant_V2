"""
成本监控集成示例

演示如何将 CostMonitor 集成到 Phase 2.0 系统中。
展示与 ProtocolAdapter、ModelRouter 和 SemanticIndexer 的集成。
"""

import asyncio
from pathlib import Path
from datetime import datetime

from llm_compression.cost_monitor import CostMonitor, ModelType
from llm_compression.protocol_adapter import ProtocolAdapter
from llm_compression.model_router import ModelRouter
from llm_compression.logger import logger


class CostAwareMemorySystem:
    """带成本监控的记忆系统"""
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8045",
        cost_log_dir: str = "logs/cost"
    ):
        """
        初始化带成本监控的记忆系统
        
        Args:
            api_key: API 密钥
            base_url: API 基础 URL
            cost_log_dir: 成本日志目录
        """
        # 初始化核心组件
        self.protocol_adapter = ProtocolAdapter(
            base_url=base_url,
            api_key=api_key
        )
        self.model_router = ModelRouter()
        
        # 初始化成本监控器
        cost_log_path = Path(cost_log_dir) / f"cost_{datetime.now().strftime('%Y%m%d')}.jsonl"
        self.cost_monitor = CostMonitor(
            log_file=str(cost_log_path),
            enable_logging=True
        )
        
        logger.info("CostAwareMemorySystem initialized with cost monitoring")
    
    async def compress_with_monitoring(
        self,
        text: str,
        quality_requirement: float = 0.85
    ) -> dict:
        """
        压缩文本并监控成本
        
        Args:
            text: 要压缩的文本
            quality_requirement: 质量要求
            
        Returns:
            dict: 压缩结果和成本信息
        """
        # 1. 选择模型
        model_info = self.model_router.select_model(
            text_length=len(text),
            quality_requirement=quality_requirement
        )
        
        model_name = model_info["model"]
        model_type = self._get_model_type(model_name)
        
        logger.info(f"Selected model: {model_name} (type: {model_type.value})")
        
        # 2. 执行压缩
        try:
            # 这里简化处理，实际应该调用真实的压缩逻辑
            result = await self._compress_text(text, model_name)
            
            # 3. 记录成本
            tokens_used = self._estimate_tokens(text)
            self.cost_monitor.record_operation(
                model_type=model_type,
                model_name=model_name,
                tokens_used=tokens_used,
                operation="compress",
                success=True
            )
            
            return {
                "compressed": result,
                "model_used": model_name,
                "tokens_used": tokens_used,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            
            # 记录失败的操作
            tokens_used = self._estimate_tokens(text)
            self.cost_monitor.record_operation(
                model_type=model_type,
                model_name=model_name,
                tokens_used=tokens_used,
                operation="compress",
                success=False
            )
            
            return {
                "compressed": None,
                "model_used": model_name,
                "tokens_used": tokens_used,
                "success": False,
                "error": str(e)
            }
    
    async def batch_compress_with_monitoring(
        self,
        texts: list[str],
        quality_requirement: float = 0.85
    ) -> dict:
        """
        批量压缩并监控成本
        
        Args:
            texts: 文本列表
            quality_requirement: 质量要求
            
        Returns:
            dict: 批量压缩结果和成本汇总
        """
        logger.info(f"Starting batch compression of {len(texts)} texts")
        
        # 开始 GPU 跟踪（如果使用本地模型）
        self.cost_monitor.start_gpu_tracking()
        
        results = []
        for i, text in enumerate(texts):
            result = await self.compress_with_monitoring(text, quality_requirement)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
        
        # 停止 GPU 跟踪
        self.cost_monitor.stop_gpu_tracking()
        
        # 获取成本汇总
        summary = self.cost_monitor.get_summary()
        
        return {
            "results": results,
            "total_texts": len(texts),
            "successful": sum(1 for r in results if r["success"]),
            "failed": sum(1 for r in results if not r["success"]),
            "cost_summary": {
                "total_cost": summary.total_cost,
                "cloud_cost": summary.cloud_cost,
                "local_cost": summary.local_cost,
                "gpu_cost": self.cost_monitor.get_gpu_cost(),
                "total_tokens": summary.total_tokens,
                "savings": summary.savings,
                "savings_percentage": summary.savings_percentage
            }
        }
    
    def get_cost_report(self, period: str = "day") -> str:
        """
        生成成本报告
        
        Args:
            period: 时间周期（day/week/month）
            
        Returns:
            str: 成本报告
        """
        return self.cost_monitor.generate_report(period=period)
    
    def get_optimization_recommendations(self) -> dict:
        """
        获取成本优化建议
        
        Returns:
            dict: 优化建议
        """
        return self.cost_monitor.optimize_model_selection()
    
    def _get_model_type(self, model_name: str) -> ModelType:
        """
        根据模型名称判断模型类型
        
        Args:
            model_name: 模型名称
            
        Returns:
            ModelType: 模型类型
        """
        if "claude" in model_name.lower() or "gpt" in model_name.lower() or "gemini" in model_name.lower():
            return ModelType.CLOUD_API
        elif "qwen" in model_name.lower() or "llama" in model_name.lower():
            return ModelType.LOCAL_MODEL
        else:
            return ModelType.SIMPLE_COMPRESSION
    
    async def _compress_text(self, text: str, model_name: str) -> str:
        """
        实际的压缩逻辑（简化版）
        
        Args:
            text: 文本
            model_name: 模型名称
            
        Returns:
            str: 压缩结果
        """
        # 这里应该调用实际的压缩逻辑
        # 为了示例，我们只返回一个简化的结果
        return f"[Compressed by {model_name}]: {text[:50]}..."
    
    def _estimate_tokens(self, text: str) -> int:
        """
        估算 token 数量
        
        Args:
            text: 文本
            
        Returns:
            int: 估算的 token 数量
        """
        # 简单估算：1 token ≈ 4 字符
        return len(text) // 4


async def main():
    """主函数 - 演示成本监控集成"""
    
    print("=" * 80)
    print("Phase 2.0 成本监控集成示例")
    print("=" * 80)
    
    # 初始化系统
    system = CostAwareMemorySystem(
        api_key="test_key",
        base_url="http://localhost:8045",
        cost_log_dir="logs/cost"
    )
    
    # 示例 1: 单个文本压缩
    print("\n示例 1: 单个文本压缩")
    print("-" * 80)
    
    text = "这是一段测试文本，用于演示成本监控功能。" * 10
    result = await system.compress_with_monitoring(text, quality_requirement=0.85)
    
    print(f"压缩结果:")
    print(f"  - 模型: {result['model_used']}")
    print(f"  - Tokens: {result['tokens_used']}")
    print(f"  - 成功: {result['success']}")
    
    # 示例 2: 批量压缩
    print("\n示例 2: 批量压缩")
    print("-" * 80)
    
    texts = [
        "短文本示例 " + str(i) for i in range(5)
    ] + [
        "这是一段较长的文本，包含更多的信息和内容。" * 20 + str(i) for i in range(5)
    ]
    
    batch_result = await system.batch_compress_with_monitoring(texts, quality_requirement=0.85)
    
    print(f"\n批量压缩结果:")
    print(f"  - 总文本数: {batch_result['total_texts']}")
    print(f"  - 成功: {batch_result['successful']}")
    print(f"  - 失败: {batch_result['failed']}")
    
    cost_summary = batch_result['cost_summary']
    print(f"\n成本汇总:")
    print(f"  - 总成本: ${cost_summary['total_cost']:.6f}")
    print(f"  - 云端 API 成本: ${cost_summary['cloud_cost']:.6f}")
    print(f"  - 本地模型成本: ${cost_summary['local_cost']:.6f}")
    print(f"  - GPU 成本: ${cost_summary['gpu_cost']:.6f}")
    print(f"  - 总 Tokens: {cost_summary['total_tokens']:,}")
    print(f"  - 成本节省: ${cost_summary['savings']:.6f} ({cost_summary['savings_percentage']:.1f}%)")
    
    # 示例 3: 生成成本报告
    print("\n示例 3: 成本报告")
    print("-" * 80)
    
    report = system.get_cost_report(period="day")
    print(report)
    
    # 示例 4: 获取优化建议
    print("\n示例 4: 优化建议")
    print("-" * 80)
    
    recommendations = system.get_optimization_recommendations()
    
    print(f"当前策略: {recommendations['current_strategy']}")
    print(f"潜在节省: ${recommendations['potential_savings']:.6f}")
    
    if recommendations['recommendations']:
        print("\n优化建议:")
        for i, rec in enumerate(recommendations['recommendations'], 1):
            print(f"\n  {i}. {rec['type']}")
            print(f"     原因: {rec['reason']}")
            print(f"     行动: {rec['action']}")
            print(f"     潜在节省: ${rec['potential_savings']:.6f}")
    else:
        print("\n当前策略已优化，无需调整。")
    
    print("\n" + "=" * 80)
    print("示例完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
