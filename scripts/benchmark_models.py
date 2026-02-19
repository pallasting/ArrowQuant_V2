#!/usr/bin/env python3
"""
模型性能基准测试工具

对比所有模型（云端 API 和本地模型）的性能指标：
- 压缩比
- 重构质量
- 延迟
- 吞吐量
- 成本

Phase 1.1 基准测试工具。
"""

import asyncio
import time
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import statistics

from llm_compression.config import Config
from llm_compression.llm_client import LLMClient
from llm_compression.model_selector import ModelSelector, MemoryType, QualityLevel
from llm_compression.compressor import LLMCompressor
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.quality_evaluator import QualityEvaluator
from llm_compression.cost_monitor import CostMonitor, ModelType
from llm_compression.logger import logger


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    model_name: str
    is_local: bool
    
    # 压缩指标
    avg_compression_ratio: float
    min_compression_ratio: float
    max_compression_ratio: float
    p95_compression_ratio: float
    
    # 质量指标
    avg_quality_score: float
    min_quality_score: float
    max_quality_score: float
    p95_quality_score: float
    
    # 延迟指标（毫秒）
    avg_compression_latency_ms: float
    p95_compression_latency_ms: float
    avg_reconstruction_latency_ms: float
    p95_reconstruction_latency_ms: float
    
    # 吞吐量指标
    compression_throughput_per_min: float
    reconstruction_throughput_per_min: float
    
    # 成本指标
    cost_per_1k_operations: float
    cost_per_gb_compressed: float
    
    # 成功率
    success_rate: float
    
    # 测试样本数
    num_samples: int
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return asdict(self)


class ModelBenchmark:
    """模型基准测试器"""
    
    def __init__(
        self,
        config: Optional[Config] = None,
        num_samples: int = 50,
        text_lengths: List[int] = None
    ):
        """
        初始化基准测试器
        
        Args:
            config: 配置对象
            num_samples: 每个模型的测试样本数
            text_lengths: 测试文本长度列表
        """
        self.config = config or Config.load()
        self.num_samples = num_samples
        self.text_lengths = text_lengths or [100, 200, 500, 1000, 2000]
        
        # 初始化组件
        self.llm_client = LLMClient(
            endpoint=self.config.llm_endpoint,
            timeout=self.config.llm_timeout
        )
        
        self.model_selector = ModelSelector(
            cloud_endpoint=self.config.llm_endpoint,
            prefer_local=True,
            ollama_endpoint=self.config.ollama_endpoint
        )
        
        self.quality_evaluator = QualityEvaluator()
        self.cost_monitor = CostMonitor()
        
        # 测试数据
        self.test_texts: List[str] = []
        
        logger.info(
            f"ModelBenchmark initialized: num_samples={num_samples}, "
            f"text_lengths={text_lengths}"
        )
    
    def generate_test_data(self):
        """生成测试数据"""
        logger.info("Generating test data...")
        
        self.test_texts = []
        
        # 为每个长度生成测试文本
        for length in self.text_lengths:
            for i in range(self.num_samples // len(self.text_lengths)):
                # 生成包含实体的测试文本
                text = self._generate_text_with_entities(length, i)
                self.test_texts.append(text)
        
        logger.info(f"Generated {len(self.test_texts)} test texts")
    
    def _generate_text_with_entities(self, length: int, seed: int) -> str:
        """生成包含实体的测试文本"""
        # 基础文本模板
        templates = [
            "On {date}, {person} visited {location} and discovered {number} ancient artifacts. "
            "The expedition was led by Dr. {person2} from {organization}. "
            "They found evidence of a civilization that existed {number2} years ago. "
            "The artifacts included pottery, tools, and inscriptions in an unknown language. "
            "This discovery could rewrite our understanding of ancient history. ",
            
            "In {year}, {company} announced a breakthrough in {technology}. "
            "CEO {person} stated that this innovation would increase efficiency by {percentage}%. "
            "The company invested ${amount} million in research and development. "
            "Experts predict this will revolutionize the {industry} industry. "
            "The technology is expected to be commercially available by {future_year}. ",
            
            "The conference held in {city} on {date} attracted {number} participants from {number2} countries. "
            "Keynote speaker {person} discussed the future of {topic}. "
            "Attendees included representatives from {organization}, {organization2}, and {organization3}. "
            "The event generated ${revenue} million in economic activity for the local area. "
            "Next year's conference is scheduled for {future_date} in {future_city}. "
        ]
        
        # 实体数据
        dates = ["2024-01-15", "2023-06-20", "2025-03-10", "2022-11-05"]
        persons = ["Alice Johnson", "Bob Smith", "Carol Williams", "David Brown"]
        locations = ["Cairo", "Beijing", "London", "New York"]
        organizations = ["UNESCO", "MIT", "Stanford University", "Google Research"]
        cities = ["Paris", "Tokyo", "Berlin", "Sydney"]
        technologies = ["artificial intelligence", "quantum computing", "renewable energy"]
        industries = ["healthcare", "transportation", "manufacturing"]
        topics = ["climate change", "digital transformation", "sustainable development"]
        
        # 选择模板
        template = templates[seed % len(templates)]
        
        # 填充实体
        text = template.format(
            date=dates[seed % len(dates)],
            person=persons[seed % len(persons)],
            person2=persons[(seed + 1) % len(persons)],
            location=locations[seed % len(locations)],
            number=100 + seed * 10,
            number2=1000 + seed * 100,
            year=2020 + (seed % 5),
            future_year=2025 + (seed % 5),
            company=organizations[seed % len(organizations)],
            technology=technologies[seed % len(technologies)],
            percentage=10 + seed % 50,
            amount=10 + seed % 100,
            industry=industries[seed % len(industries)],
            city=cities[seed % len(cities)],
            future_city=cities[(seed + 1) % len(cities)],
            organization=organizations[seed % len(organizations)],
            organization2=organizations[(seed + 1) % len(organizations)],
            organization3=organizations[(seed + 2) % len(organizations)],
            revenue=5 + seed % 20,
            future_date=dates[(seed + 1) % len(dates)],
            topic=topics[seed % len(topics)]
        )
        
        # 重复文本直到达到目标长度
        while len(text) < length:
            text += text
        
        return text[:length]
    
    async def benchmark_model(
        self,
        model_name: str,
        is_local: bool
    ) -> BenchmarkResult:
        """
        对单个模型进行基准测试
        
        Args:
            model_name: 模型名称
            is_local: 是否本地模型
            
        Returns:
            BenchmarkResult: 基准测试结果
        """
        logger.info(f"Benchmarking model: {model_name} (local={is_local})")
        
        # 创建压缩器和重构器
        compressor = LLMCompressor(
            llm_client=self.llm_client,
            model_selector=self.model_selector,
            quality_evaluator=self.quality_evaluator
        )
        
        reconstructor = LLMReconstructor(
            llm_client=self.llm_client
        )
        
        # 测试指标
        compression_ratios = []
        quality_scores = []
        compression_latencies = []
        reconstruction_latencies = []
        successes = 0
        total_cost = 0.0
        total_size_compressed = 0
        
        # 运行测试
        for i, text in enumerate(self.test_texts):
            try:
                # 压缩测试
                start_time = time.time()
                compressed = await compressor.compress(
                    text,
                    manual_model=model_name
                )
                compression_time = (time.time() - start_time) * 1000  # 转换为毫秒
                
                if compressed:
                    compression_latencies.append(compression_time)
                    compression_ratios.append(compressed.metadata.compression_ratio)
                    total_size_compressed += len(text)
                    
                    # 重构测试
                    start_time = time.time()
                    reconstructed = await reconstructor.reconstruct(compressed)
                    reconstruction_time = (time.time() - start_time) * 1000
                    
                    if reconstructed:
                        reconstruction_latencies.append(reconstruction_time)
                        
                        # 质量评估
                        quality = await self.quality_evaluator.evaluate(
                            original_text=text,
                            reconstructed_text=reconstructed.full_text,
                            compressed_memory=compressed
                        )
                        quality_scores.append(quality.semantic_similarity)
                        
                        # 记录成本
                        model_type = ModelType.LOCAL_MODEL if is_local else ModelType.CLOUD_API
                        tokens_used = compressed.metadata.tokens_used or 1000
                        self.cost_monitor.record_operation(
                            model_type=model_type,
                            model_name=model_name,
                            tokens_used=tokens_used,
                            operation="compress",
                            success=True
                        )
                        
                        successes += 1
                
                # 进度显示
                if (i + 1) % 10 == 0:
                    logger.info(f"  Progress: {i + 1}/{len(self.test_texts)}")
                    
            except Exception as e:
                logger.warning(f"  Test {i} failed: {e}")
                continue
        
        # 计算统计指标
        if not compression_ratios:
            logger.error(f"No successful tests for {model_name}")
            return None
        
        # 吞吐量（每分钟）
        total_time_min = sum(compression_latencies) / 1000 / 60
        compression_throughput = len(compression_latencies) / total_time_min if total_time_min > 0 else 0
        
        total_recon_time_min = sum(reconstruction_latencies) / 1000 / 60
        reconstruction_throughput = len(reconstruction_latencies) / total_recon_time_min if total_recon_time_min > 0 else 0
        
        # 成本
        summary = self.cost_monitor.get_summary()
        cost_per_1k = (summary.total_cost / successes * 1000) if successes > 0 else 0
        cost_per_gb = (summary.total_cost / (total_size_compressed / 1e9)) if total_size_compressed > 0 else 0
        
        # 创建结果
        result = BenchmarkResult(
            model_name=model_name,
            is_local=is_local,
            avg_compression_ratio=statistics.mean(compression_ratios),
            min_compression_ratio=min(compression_ratios),
            max_compression_ratio=max(compression_ratios),
            p95_compression_ratio=self._percentile(compression_ratios, 95),
            avg_quality_score=statistics.mean(quality_scores) if quality_scores else 0.0,
            min_quality_score=min(quality_scores) if quality_scores else 0.0,
            max_quality_score=max(quality_scores) if quality_scores else 0.0,
            p95_quality_score=self._percentile(quality_scores, 95) if quality_scores else 0.0,
            avg_compression_latency_ms=statistics.mean(compression_latencies),
            p95_compression_latency_ms=self._percentile(compression_latencies, 95),
            avg_reconstruction_latency_ms=statistics.mean(reconstruction_latencies) if reconstruction_latencies else 0.0,
            p95_reconstruction_latency_ms=self._percentile(reconstruction_latencies, 95) if reconstruction_latencies else 0.0,
            compression_throughput_per_min=compression_throughput,
            reconstruction_throughput_per_min=reconstruction_throughput,
            cost_per_1k_operations=cost_per_1k,
            cost_per_gb_compressed=cost_per_gb,
            success_rate=successes / len(self.test_texts),
            num_samples=successes
        )
        
        logger.info(f"Benchmark complete for {model_name}: {successes}/{len(self.test_texts)} successful")
        
        return result
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    async def run_all_benchmarks(self) -> Dict[str, BenchmarkResult]:
        """运行所有模型的基准测试"""
        logger.info("Starting comprehensive benchmark...")
        
        # 生成测试数据
        self.generate_test_data()
        
        results = {}
        
        # 测试模型列表
        models_to_test = [
            ("qwen2.5:7b-instruct", True),
            ("llama3.1:8b-instruct-q4_K_M", True),
            ("gemma3:4b", True),
            ("cloud-api", False),
        ]
        
        for model_name, is_local in models_to_test:
            logger.info(f"\n{'='*60}")
            logger.info(f"Testing: {model_name}")
            logger.info(f"{'='*60}")
            
            # 清除成本监控器（为每个模型单独计算）
            self.cost_monitor.clear()
            
            try:
                result = await self.benchmark_model(model_name, is_local)
                if result:
                    results[model_name] = result
            except Exception as e:
                logger.error(f"Benchmark failed for {model_name}: {e}", exc_info=True)
        
        return results
    
    def generate_report(
        self,
        results: Dict[str, BenchmarkResult],
        output_file: Optional[str] = None
    ) -> str:
        """
        生成基准测试报告
        
        Args:
            results: 基准测试结果
            output_file: 输出文件路径
            
        Returns:
            str: 报告内容
        """
        lines = [
            "=" * 80,
            "模型性能基准测试报告",
            "=" * 80,
            f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"测试样本数: {self.num_samples}",
            f"文本长度范围: {min(self.text_lengths)}-{max(self.text_lengths)} 字符",
            "",
            "=" * 80,
            "压缩比对比",
            "=" * 80,
            f"{'模型':<30} {'平均':<10} {'最小':<10} {'最大':<10} {'P95':<10}",
            "-" * 80,
        ]
        
        for model_name, result in results.items():
            lines.append(
                f"{model_name:<30} "
                f"{result.avg_compression_ratio:<10.2f} "
                f"{result.min_compression_ratio:<10.2f} "
                f"{result.max_compression_ratio:<10.2f} "
                f"{result.p95_compression_ratio:<10.2f}"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "重构质量对比",
            "=" * 80,
            f"{'模型':<30} {'平均':<10} {'最小':<10} {'最大':<10} {'P95':<10}",
            "-" * 80,
        ])
        
        for model_name, result in results.items():
            lines.append(
                f"{model_name:<30} "
                f"{result.avg_quality_score:<10.3f} "
                f"{result.min_quality_score:<10.3f} "
                f"{result.max_quality_score:<10.3f} "
                f"{result.p95_quality_score:<10.3f}"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "延迟对比（毫秒）",
            "=" * 80,
            f"{'模型':<30} {'压缩(平均)':<15} {'压缩(P95)':<15} {'重构(平均)':<15} {'重构(P95)':<15}",
            "-" * 80,
        ])
        
        for model_name, result in results.items():
            lines.append(
                f"{model_name:<30} "
                f"{result.avg_compression_latency_ms:<15.1f} "
                f"{result.p95_compression_latency_ms:<15.1f} "
                f"{result.avg_reconstruction_latency_ms:<15.1f} "
                f"{result.p95_reconstruction_latency_ms:<15.1f}"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "吞吐量对比（操作/分钟）",
            "=" * 80,
            f"{'模型':<30} {'压缩':<15} {'重构':<15}",
            "-" * 80,
        ])
        
        for model_name, result in results.items():
            lines.append(
                f"{model_name:<30} "
                f"{result.compression_throughput_per_min:<15.1f} "
                f"{result.reconstruction_throughput_per_min:<15.1f}"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "成本对比",
            "=" * 80,
            f"{'模型':<30} {'$/1K操作':<15} {'$/GB压缩':<15}",
            "-" * 80,
        ])
        
        for model_name, result in results.items():
            lines.append(
                f"{model_name:<30} "
                f"${result.cost_per_1k_operations:<14.4f} "
                f"${result.cost_per_gb_compressed:<14.4f}"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "成功率对比",
            "=" * 80,
            f"{'模型':<30} {'成功率':<15} {'成功样本数':<15}",
            "-" * 80,
        ])
        
        for model_name, result in results.items():
            lines.append(
                f"{model_name:<30} "
                f"{result.success_rate * 100:<14.1f}% "
                f"{result.num_samples:<15}"
            )
        
        lines.extend([
            "",
            "=" * 80,
            "推荐模型",
            "=" * 80,
        ])
        
        # 找出最佳模型
        best_compression = max(results.items(), key=lambda x: x[1].avg_compression_ratio)
        best_quality = max(results.items(), key=lambda x: x[1].avg_quality_score)
        best_speed = min(results.items(), key=lambda x: x[1].avg_compression_latency_ms)
        best_cost = min(results.items(), key=lambda x: x[1].cost_per_1k_operations)
        
        lines.extend([
            f"最佳压缩比: {best_compression[0]} ({best_compression[1].avg_compression_ratio:.2f}x)",
            f"最佳质量: {best_quality[0]} ({best_quality[1].avg_quality_score:.3f})",
            f"最快速度: {best_speed[0]} ({best_speed[1].avg_compression_latency_ms:.1f}ms)",
            f"最低成本: {best_cost[0]} (${best_cost[1].cost_per_1k_operations:.4f}/1K)",
            "",
            "=" * 80,
        ])
        
        report = "\n".join(lines)
        
        # 保存到文件
        if output_file:
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(report, encoding="utf-8")
            logger.info(f"Report saved to {output_file}")
        
        return report
    
    def save_results_json(
        self,
        results: Dict[str, BenchmarkResult],
        output_file: str
    ):
        """保存结果为 JSON 格式"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        json_data = {
            "timestamp": time.time(),
            "test_config": {
                "num_samples": self.num_samples,
                "text_lengths": self.text_lengths
            },
            "results": {
                model_name: result.to_dict()
                for model_name, result in results.items()
            }
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved to {output_file}")


async def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("LLM 压缩系统 - 模型性能基准测试")
    print("Phase 1.1 基准测试工具")
    print("=" * 80)
    
    # 创建基准测试器
    benchmark = ModelBenchmark(
        num_samples=50,  # 每个模型 50 个样本
        text_lengths=[100, 200, 500, 1000, 2000]
    )
    
    # 运行基准测试
    results = await benchmark.run_all_benchmarks()
    
    # 生成报告
    report = benchmark.generate_report(
        results,
        output_file="benchmark_results/model_comparison_report.txt"
    )
    
    print("\n" + report)
    
    # 保存 JSON 结果
    benchmark.save_results_json(
        results,
        output_file="benchmark_results/model_comparison_results.json"
    )
    
    print("\n" + "=" * 80)
    print("基准测试完成！")
    print("=" * 80)
    print(f"\n报告已保存到: benchmark_results/model_comparison_report.txt")
    print(f"JSON 结果已保存到: benchmark_results/model_comparison_results.json")


if __name__ == "__main__":
    asyncio.run(main())
