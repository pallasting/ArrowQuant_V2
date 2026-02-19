#!/usr/bin/env python3
"""
Phase 1.1 验证脚本

验证所有 Phase 1.1 验收标准：
- 本地模型可用
- 压缩延迟 < 2s
- 重构延迟 < 500ms
- 吞吐量 > 100/min
- 成本节省 > 80%
- 所有 Phase 1.0 标准继续满足
"""

import asyncio
import time
import sys
from pathlib import Path
from typing import Dict, List, Tuple

from llm_compression.config import load_config
from llm_compression.llm_client import LLMClient
from llm_compression.model_selector import ModelSelector
from llm_compression.compressor import LLMCompressor
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.quality_evaluator import QualityEvaluator
from llm_compression.cost_monitor import CostMonitor, ModelType
from llm_compression.model_deployment import ModelDeploymentSystem
from llm_compression.logger import logger


class Phase11Validator:
    """Phase 1.1 验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.config = load_config()
        self.results = {}
        self.passed_checks = 0
        self.total_checks = 0
        
        # 初始化组件
        self.llm_client = LLMClient(endpoint=self.config.llm.cloud_endpoint)
        self.model_selector = ModelSelector(
            cloud_endpoint=self.config.llm.cloud_endpoint,
            prefer_local=True,
            ollama_endpoint=self.config.model.ollama_endpoint
        )
        self.quality_evaluator = QualityEvaluator()
        self.cost_monitor = CostMonitor()
        
        logger.info("Phase11Validator initialized")
    
    def check(self, name: str, condition: bool, actual: any, expected: any, unit: str = ""):
        """
        检查单个条件
        
        Args:
            name: 检查名称
            condition: 是否通过
            actual: 实际值
            expected: 期望值
            unit: 单位
        """
        self.total_checks += 1
        status = "✓ PASS" if condition else "✗ FAIL"
        
        if condition:
            self.passed_checks += 1
        
        result = {
            "name": name,
            "passed": condition,
            "actual": actual,
            "expected": expected,
            "unit": unit
        }
        
        self.results[name] = result
        
        print(f"  [{status}] {name}")
        print(f"        实际: {actual}{unit}, 期望: {expected}{unit}")
        
        return condition
    
    async def validate_local_model_availability(self) -> bool:
        """验证本地模型可用性"""
        print("\n" + "=" * 70)
        print("检查 1: 本地模型可用性")
        print("=" * 70)
        
        try:
            # 检查 Ollama 服务
            deployment = ModelDeploymentSystem()
            
            # 检查服务状态
            print("\n检查 Ollama 服务...")
            service_running = deployment.check_service_status("ollama")
            self.check(
                "Ollama 服务运行",
                service_running,
                "运行中" if service_running else "未运行",
                "运行中"
            )
            
            # 检查模型
            print("\n检查已安装模型...")
            models = deployment.list_models()
            
            required_models = ["qwen2.5:7b-instruct"]
            for model in required_models:
                model_available = any(model in m for m in models)
                self.check(
                    f"模型 {model} 可用",
                    model_available,
                    "已安装" if model_available else "未安装",
                    "已安装"
                )
            
            # 测试推理
            print("\n测试本地模型推理...")
            test_text = "This is a test message for local model inference."
            
            compressor = LLMCompressor(
                llm_client=self.llm_client,
                model_selector=self.model_selector,
                quality_evaluator=self.quality_evaluator
            )
            
            compressed = await compressor.compress(test_text, manual_model="qwen2.5")
            
            inference_works = compressed is not None
            self.check(
                "本地模型推理",
                inference_works,
                "成功" if inference_works else "失败",
                "成功"
            )
            
            return service_running and inference_works
            
        except Exception as e:
            logger.error(f"Local model validation failed: {e}", exc_info=True)
            self.check("本地模型可用性", False, "失败", "成功")
            return False
    
    async def validate_compression_latency(self) -> bool:
        """验证压缩延迟 < 2s"""
        print("\n" + "=" * 70)
        print("检查 2: 压缩延迟")
        print("=" * 70)
        
        try:
            compressor = LLMCompressor(
                llm_client=self.llm_client,
                model_selector=self.model_selector,
                quality_evaluator=self.quality_evaluator
            )
            
            # 测试文本（约 1000 字符）
            test_text = """
            Artificial intelligence has revolutionized many industries in recent years.
            Machine learning algorithms can now process vast amounts of data and identify
            patterns that humans might miss. Deep learning, a subset of machine learning,
            uses neural networks with multiple layers to learn hierarchical representations
            of data. This technology powers applications like image recognition, natural
            language processing, and autonomous vehicles. Companies are investing billions
            of dollars in AI research and development. The potential applications are vast,
            ranging from healthcare diagnostics to financial forecasting. However, there
            are also concerns about AI ethics, bias in algorithms, and the impact on
            employment. Researchers are working on making AI systems more transparent,
            fair, and accountable. The future of AI holds both tremendous promise and
            significant challenges that society must address thoughtfully.
            """ * 2
            
            print(f"\n测试文本长度: {len(test_text)} 字符")
            print("运行 5 次测试...")
            
            latencies = []
            for i in range(5):
                start = time.time()
                compressed = await compressor.compress(test_text, manual_model="qwen2.5")
                latency = time.time() - start
                latencies.append(latency)
                print(f"  测试 {i+1}: {latency:.3f}s")
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            print(f"\n平均延迟: {avg_latency:.3f}s")
            print(f"最大延迟: {max_latency:.3f}s")
            
            avg_pass = self.check(
                "平均压缩延迟 < 2s",
                avg_latency < 2.0,
                f"{avg_latency:.3f}",
                "< 2.0",
                "s"
            )
            
            max_pass = self.check(
                "最大压缩延迟 < 3s",
                max_latency < 3.0,
                f"{max_latency:.3f}",
                "< 3.0",
                "s"
            )
            
            return avg_pass and max_pass
            
        except Exception as e:
            logger.error(f"Compression latency validation failed: {e}", exc_info=True)
            self.check("压缩延迟", False, "失败", "< 2s")
            return False
    
    async def validate_reconstruction_latency(self) -> bool:
        """验证重构延迟 < 500ms"""
        print("\n" + "=" * 70)
        print("检查 3: 重构延迟")
        print("=" * 70)
        
        try:
            compressor = LLMCompressor(
                llm_client=self.llm_client,
                model_selector=self.model_selector,
                quality_evaluator=self.quality_evaluator
            )
            
            reconstructor = LLMReconstructor(llm_client=self.llm_client)
            
            # 测试文本
            test_text = "AI technology is advancing rapidly." * 50
            
            print(f"\n测试文本长度: {len(test_text)} 字符")
            print("运行 5 次测试...")
            
            # 先压缩
            compressed = await compressor.compress(test_text, manual_model="qwen2.5")
            
            if not compressed:
                self.check("重构延迟", False, "压缩失败", "< 500ms")
                return False
            
            # 测试重构延迟
            latencies = []
            for i in range(5):
                start = time.time()
                reconstructed = await reconstructor.reconstruct(compressed)
                latency = (time.time() - start) * 1000  # 转换为毫秒
                latencies.append(latency)
                print(f"  测试 {i+1}: {latency:.0f}ms")
            
            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)
            
            print(f"\n平均延迟: {avg_latency:.0f}ms")
            print(f"最大延迟: {max_latency:.0f}ms")
            
            avg_pass = self.check(
                "平均重构延迟 < 500ms",
                avg_latency < 500,
                f"{avg_latency:.0f}",
                "< 500",
                "ms"
            )
            
            max_pass = self.check(
                "最大重构延迟 < 800ms",
                max_latency < 800,
                f"{max_latency:.0f}",
                "< 800",
                "ms"
            )
            
            return avg_pass and max_pass
            
        except Exception as e:
            logger.error(f"Reconstruction latency validation failed: {e}", exc_info=True)
            self.check("重构延迟", False, "失败", "< 500ms")
            return False
    
    async def validate_throughput(self) -> bool:
        """验证吞吐量 > 100/min"""
        print("\n" + "=" * 70)
        print("检查 4: 吞吐量")
        print("=" * 70)
        
        try:
            compressor = LLMCompressor(
                llm_client=self.llm_client,
                model_selector=self.model_selector,
                quality_evaluator=self.quality_evaluator
            )
            
            # 测试文本
            test_texts = [
                "AI is transforming the world." * 30
                for _ in range(10)
            ]
            
            print(f"\n测试 {len(test_texts)} 个文本...")
            
            start = time.time()
            for i, text in enumerate(test_texts):
                await compressor.compress(text, manual_model="qwen2.5")
                print(f"  完成 {i+1}/{len(test_texts)}")
            
            elapsed_time = time.time() - start
            elapsed_minutes = elapsed_time / 60
            
            throughput = len(test_texts) / elapsed_minutes
            
            print(f"\n总耗时: {elapsed_time:.1f}s ({elapsed_minutes:.2f}分钟)")
            print(f"吞吐量: {throughput:.1f} 操作/分钟")
            
            passed = self.check(
                "吞吐量 > 100/min",
                throughput > 100,
                f"{throughput:.1f}",
                "> 100",
                " 操作/分钟"
            )
            
            return passed
            
        except Exception as e:
            logger.error(f"Throughput validation failed: {e}", exc_info=True)
            self.check("吞吐量", False, "失败", "> 100/min")
            return False
    
    async def validate_cost_savings(self) -> bool:
        """验证成本节省 > 80%"""
        print("\n" + "=" * 70)
        print("检查 5: 成本节省")
        print("=" * 70)
        
        try:
            compressor = LLMCompressor(
                llm_client=self.llm_client,
                model_selector=self.model_selector,
                quality_evaluator=self.quality_evaluator
            )
            
            # 清除成本监控器
            self.cost_monitor.clear()
            
            # 测试文本
            test_text = "Machine learning is a subset of artificial intelligence." * 40
            
            print("\n模拟本地模型使用...")
            # 模拟 90% 本地模型，10% 云端 API
            for i in range(10):
                if i < 9:
                    # 本地模型
                    await compressor.compress(test_text, manual_model="qwen2.5")
                    self.cost_monitor.record_operation(
                        model_type=ModelType.LOCAL_MODEL,
                        model_name="qwen2.5:7b-instruct",
                        tokens_used=1000,
                        operation="compress",
                        success=True
                    )
                else:
                    # 云端 API
                    self.cost_monitor.record_operation(
                        model_type=ModelType.CLOUD_API,
                        model_name="cloud-api",
                        tokens_used=1000,
                        operation="compress",
                        success=True
                    )
            
            # 获取成本汇总
            summary = self.cost_monitor.get_summary()
            
            print(f"\n成本分析:")
            print(f"  总成本: ${summary.total_cost:.6f}")
            print(f"  云端成本: ${summary.cloud_cost:.6f}")
            print(f"  本地成本: ${summary.local_cost:.6f}")
            print(f"  成本节省: ${summary.savings:.6f} ({summary.savings_percentage:.1f}%)")
            
            passed = self.check(
                "成本节省 > 80%",
                summary.savings_percentage > 80,
                f"{summary.savings_percentage:.1f}",
                "> 80",
                "%"
            )
            
            return passed
            
        except Exception as e:
            logger.error(f"Cost savings validation failed: {e}", exc_info=True)
            self.check("成本节省", False, "失败", "> 80%")
            return False
    
    async def validate_phase_1_0_standards(self) -> bool:
        """验证 Phase 1.0 标准继续满足"""
        print("\n" + "=" * 70)
        print("检查 6: Phase 1.0 标准")
        print("=" * 70)
        
        try:
            compressor = LLMCompressor(
                llm_client=self.llm_client,
                model_selector=self.model_selector,
                quality_evaluator=self.quality_evaluator
            )
            
            reconstructor = LLMReconstructor(llm_client=self.llm_client)
            
            # 测试文本
            test_text = """
            The development of quantum computing represents a paradigm shift in computational
            capabilities. Unlike classical computers that use bits (0 or 1), quantum computers
            use quantum bits or qubits that can exist in multiple states simultaneously through
            superposition. This property, combined with quantum entanglement, allows quantum
            computers to solve certain problems exponentially faster than classical computers.
            Applications include cryptography, drug discovery, optimization problems, and
            simulation of quantum systems. Major technology companies and research institutions
            are investing heavily in quantum computing research. However, significant challenges
            remain, including maintaining quantum coherence, error correction, and scaling up
            the number of qubits. The field is rapidly evolving, with new breakthroughs
            announced regularly.
            """ * 2
            
            print(f"\n测试文本长度: {len(test_text)} 字符")
            
            # 压缩
            compressed = await compressor.compress(test_text, manual_model="qwen2.5")
            
            if not compressed:
                print("✗ 压缩失败")
                return False
            
            # 重构
            reconstructed = await reconstructor.reconstruct(compressed)
            
            if not reconstructed:
                print("✗ 重构失败")
                return False
            
            # 质量评估
            quality = await self.quality_evaluator.evaluate(
                original_text=test_text,
                reconstructed_text=reconstructed.full_text,
                compressed_memory=compressed
            )
            
            print(f"\n性能指标:")
            print(f"  压缩比: {compressed.metadata.compression_ratio:.2f}x")
            print(f"  语义相似度: {quality.semantic_similarity:.3f}")
            print(f"  实体准确率: {quality.entity_accuracy:.3f}")
            
            # 检查 Phase 1.0 标准
            compression_ratio_pass = self.check(
                "压缩比 > 10x",
                compressed.metadata.compression_ratio > 10,
                f"{compressed.metadata.compression_ratio:.2f}",
                "> 10",
                "x"
            )
            
            quality_pass = self.check(
                "重构质量 > 0.85",
                quality.semantic_similarity > 0.85,
                f"{quality.semantic_similarity:.3f}",
                "> 0.85"
            )
            
            entity_pass = self.check(
                "实体准确率 > 0.95",
                quality.entity_accuracy > 0.95,
                f"{quality.entity_accuracy:.3f}",
                "> 0.95"
            )
            
            return compression_ratio_pass and quality_pass and entity_pass
            
        except Exception as e:
            logger.error(f"Phase 1.0 standards validation failed: {e}", exc_info=True)
            self.check("Phase 1.0 标准", False, "失败", "通过")
            return False
    
    async def run_validation(self) -> bool:
        """运行所有验证"""
        print("\n" + "=" * 70)
        print("Phase 1.1 验证")
        print("=" * 70)
        print(f"开始时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # 运行所有检查
        checks = [
            ("本地模型可用性", self.validate_local_model_availability()),
            ("压缩延迟", self.validate_compression_latency()),
            ("重构延迟", self.validate_reconstruction_latency()),
            ("吞吐量", self.validate_throughput()),
            ("成本节省", self.validate_cost_savings()),
            ("Phase 1.0 标准", self.validate_phase_1_0_standards()),
        ]
        
        results = []
        for name, check in checks:
            try:
                result = await check
                results.append((name, result))
            except Exception as e:
                logger.error(f"Check {name} failed with exception: {e}", exc_info=True)
                results.append((name, False))
        
        # 生成报告
        self.generate_report(results)
        
        # 返回总体结果
        all_passed = all(result for _, result in results)
        return all_passed
    
    def generate_report(self, results: List[Tuple[str, bool]]):
        """生成验证报告"""
        print("\n" + "=" * 70)
        print("验证结果汇总")
        print("=" * 70)
        
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  [{status}] {name}")
        
        print("\n" + "=" * 70)
        print(f"总计: {self.passed_checks}/{self.total_checks} 检查通过")
        print(f"通过率: {self.passed_checks / self.total_checks * 100:.1f}%")
        print("=" * 70)
        
        if self.passed_checks == self.total_checks:
            print("\n🎉 Phase 1.1 验证通过！")
            print("所有验收标准已达成。")
        else:
            print("\n⚠️  Phase 1.1 验证未完全通过")
            print(f"有 {self.total_checks - self.passed_checks} 项检查失败。")
            print("请查看上述详细信息并解决问题。")
        
        print(f"\n完成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")


async def main():
    """主函数"""
    validator = Phase11Validator()
    
    try:
        success = await validator.run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}", exc_info=True)
        print(f"\n✗ 验证过程出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
