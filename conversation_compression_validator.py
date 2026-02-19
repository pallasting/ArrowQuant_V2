#!/usr/bin/env python3
"""
会话压缩验证器
使用真实的 Windsurf 会话数据验证压缩系统
"""

import os
import re
import json
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
import asyncio

# 添加项目路径
import sys
sys.path.insert(0, '/Media/Ubuntu/Documents/Surface-Memory/Documents/ai-os-memory')

from llm_compression.compressor import LLMCompressor
from llm_compression.reconstructor import LLMReconstructor
from llm_compression.quality_evaluator import QualityEvaluator
from llm_compression.model_selector import ModelSelector, MemoryType, QualityLevel


class ConversationValidator:
    """会话压缩验证器"""
    
    def __init__(
        self,
        data_dir: str = "/Data/CascadeProjects/TalkingWithU",
        model_name: str = "gemma3",  # 默认使用 Gemma 3 4B
        output_dir: str = "validation_results"
    ):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 初始化组件
        print(f"初始化压缩系统 (模型: {model_name})...")
        self.model_name = model_name
        
        # 创建依赖组件
        from llm_compression.llm_client import LLMClient
        from llm_compression.config import load_config
        
        config = load_config()
        
        # 使用本地GPU模型（Ollama + Vulkan）
        endpoint = 'http://localhost:11434'
        print(f"使用本地GPU模型: {endpoint} (Vulkan加速)")
        
        self.llm_client = LLMClient(endpoint)
        self.model_selector = ModelSelector(
            cloud_endpoint=config.llm.cloud_endpoint,
            ollama_endpoint='http://localhost:11434',
            prefer_local=True  # 优先本地GPU
        )
        
        self.compressor = LLMCompressor(self.llm_client, self.model_selector)
        self.reconstructor = LLMReconstructor(self.llm_client, quality_threshold=0.85)
        self.evaluator = QualityEvaluator()
        
        # 结果存储
        self.results = []
        self.errors = []
    
    def load_conversations(self) -> List[Path]:
        """加载所有会话文件"""
        files = sorted(self.data_dir.glob("*.txt.md"))
        print(f"\n找到 {len(files)} 个会话文件")
        return files
    
    def parse_conversation(self, file_path: Path) -> List[Dict[str, str]]:
        """解析会话文件"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"  读取文件失败: {e}")
            return []
        
        messages = []
        lines = content.split('\n')
        
        current_role = None
        current_content = []
        
        for line in lines:
            # 检测角色切换
            if line.startswith('Assistant') or line.startswith('assistant'):
                if current_role and current_content:
                    messages.append({
                        'role': current_role,
                        'content': '\n'.join(current_content).strip()
                    })
                current_role = 'assistant'
                current_content = []
            elif line.startswith('Human') or line.startswith('human') or line.startswith('User'):
                if current_role and current_content:
                    messages.append({
                        'role': current_role,
                        'content': '\n'.join(current_content).strip()
                    })
                current_role = 'user'
                current_content = []
            elif line.strip():
                # 跳过时间戳行
                if not re.match(r'^\d{4}-\d{2}-\d{2}', line):
                    current_content.append(line)
        
        # 添加最后一条消息
        if current_role and current_content:
            messages.append({
                'role': current_role,
                'content': '\n'.join(current_content).strip()
            })
        
        # 过滤空消息
        messages = [m for m in messages if m['content'].strip()]
        
        return messages
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """格式化消息为文本"""
        return '\n\n'.join([
            f"[{msg['role']}]: {msg['content']}"
            for msg in messages
        ])
    
    async def validate_file(self, file_path: Path) -> Dict[str, Any]:
        """验证单个文件"""
        print(f"\n{'='*60}")
        print(f"处理: {file_path.name}")
        print(f"{'='*60}")
        
        result = {
            'file': file_path.name,
            'timestamp': datetime.now().isoformat(),
            'model': self.model_name,
            'success': False
        }
        
        try:
            # 1. 解析会话
            messages = self.parse_conversation(file_path)
            if not messages:
                print("  ⚠️  无法解析会话内容")
                result['error'] = "无法解析会话"
                self.errors.append(result)
                return result
            
            result['message_count'] = len(messages)
            print(f"  消息数: {len(messages)}")
            
            # 2. 格式化文本
            text = self.format_messages(messages)
            original_length = len(text)
            result['original_length'] = original_length
            result['original_chars'] = original_length
            print(f"  原始长度: {original_length:,} 字符")
            
            # 3. 压缩
            print(f"  压缩中...")
            start_time = time.time()
            
            compressed = await self.compressor.compress(text)
            
            compress_time = time.time() - start_time
            result['compress_time'] = compress_time
            
            # 计算压缩后大小（summary_hash + entities + diff_data）
            compressed_size = compressed.compression_metadata.compressed_size
            result['compressed_length'] = compressed_size
            result['compressed_chars'] = compressed_size
            
            compression_ratio = compressed.compression_metadata.compression_ratio
            result['compression_ratio'] = compression_ratio
            
            print(f"  ✅ 压缩完成")
            print(f"     压缩后: {compressed_size:,} 字节")
            print(f"     压缩比: {compression_ratio:.2f}x")
            print(f"     耗时: {compress_time:.2f}s")
            
            # 4. 重构
            print(f"  重构中...")
            start_time = time.time()
            
            reconstructed = await self.reconstructor.reconstruct(compressed)
            
            reconstruct_time = time.time() - start_time
            result['reconstruct_time'] = reconstruct_time
            result['reconstructed_length'] = len(reconstructed.full_text)
            
            print(f"  ✅ 重构完成")
            print(f"     耗时: {reconstruct_time:.2f}s")
            
            # 5. 质量评估
            print(f"  评估质量...")
            quality = self.evaluator.evaluate(
                text, 
                reconstructed.full_text,
                compressed_size=compressed.compression_metadata.compressed_size,
                reconstruction_latency_ms=reconstruct_time * 1000
            )
            
            result['quality_score'] = quality.overall_score
            result['semantic_similarity'] = quality.semantic_similarity
            result['entity_accuracy'] = getattr(quality, 'entity_accuracy', 0.0)
            
            print(f"  ✅ 质量评估完成")
            print(f"     总分: {quality.overall_score:.3f}")
            print(f"     语义相似度: {quality.semantic_similarity:.3f}")
            
            # 6. 检查关键信息保留
            sample_keywords = self._extract_keywords(text)
            preserved_keywords = sum(1 for kw in sample_keywords if kw in reconstructed.full_text)
            keyword_retention = preserved_keywords / len(sample_keywords) if sample_keywords else 0
            result['keyword_retention'] = keyword_retention
            
            print(f"     关键词保留: {keyword_retention:.1%} ({preserved_keywords}/{len(sample_keywords)})")
            
            result['success'] = True
            self.results.append(result)
            
            # 保存详细结果
            self._save_detail(file_path.stem, {
                'original': text[:500] + '...' if len(text) > 500 else text,
                'compressed_hash': compressed.summary_hash,
                'reconstructed': reconstructed.full_text[:500] + '...' if len(reconstructed.full_text) > 500 else reconstructed.full_text,
                'metrics': result
            })
            
        except Exception as e:
            print(f"  ❌ 错误: {e}")
            result['error'] = str(e)
            self.errors.append(result)
        
        return result
    
    def _extract_keywords(self, text: str, max_keywords: int = 20) -> List[str]:
        """提取关键词（简单实现）"""
        # 提取长度 > 3 的中文词和英文单词
        words = re.findall(r'[\u4e00-\u9fff]{2,}|[a-zA-Z]{4,}', text)
        # 去重并取前 N 个
        unique_words = list(dict.fromkeys(words))[:max_keywords]
        return unique_words
    
    def _save_detail(self, name: str, data: Dict[str, Any]):
        """保存详细结果"""
        detail_file = self.output_dir / f"{name}_detail.json"
        with open(detail_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    async def validate_all(self):
        """验证所有文件"""
        files = self.load_conversations()
        
        if not files:
            print("没有找到会话文件")
            return
        
        print(f"\n开始验证 {len(files)} 个文件...")
        print(f"模型: {self.model_name}")
        print(f"输出目录: {self.output_dir}")
        
        for i, file_path in enumerate(files, 1):
            print(f"\n[{i}/{len(files)}]")
            await self.validate_file(file_path)
        
        # 生成汇总报告
        self.generate_report()
    
    def generate_report(self):
        """生成验证报告"""
        if not self.results:
            print("\n没有成功的验证结果")
            return
        
        print(f"\n{'='*60}")
        print("验证报告汇总")
        print(f"{'='*60}")
        
        # 统计
        total_files = len(self.results) + len(self.errors)
        success_count = len(self.results)
        error_count = len(self.errors)
        
        print(f"\n📊 总体统计:")
        print(f"  总文件数: {total_files}")
        print(f"  成功: {success_count} ({success_count/total_files*100:.1f}%)")
        print(f"  失败: {error_count} ({error_count/total_files*100:.1f}%)")
        
        if not self.results:
            return
        
        # 计算平均值
        avg_compression_ratio = sum(r['compression_ratio'] for r in self.results) / len(self.results)
        avg_compress_time = sum(r['compress_time'] for r in self.results) / len(self.results)
        avg_reconstruct_time = sum(r['reconstruct_time'] for r in self.results) / len(self.results)
        avg_quality = sum(r['quality_score'] for r in self.results) / len(self.results)
        avg_similarity = sum(r['semantic_similarity'] for r in self.results) / len(self.results)
        avg_keyword_retention = sum(r.get('keyword_retention', 0) for r in self.results) / len(self.results)
        
        # 计算吞吐量
        total_time = sum(r['compress_time'] + r['reconstruct_time'] for r in self.results)
        throughput = (len(self.results) * 60) / total_time if total_time > 0 else 0
        
        print(f"\n📈 性能指标:")
        print(f"  平均压缩比: {avg_compression_ratio:.2f}x")
        print(f"  平均压缩耗时: {avg_compress_time:.2f}s")
        print(f"  平均重构耗时: {avg_reconstruct_time:.2f}s")
        print(f"  吞吐量: {throughput:.1f} 文件/分钟")
        
        print(f"\n🎯 质量指标:")
        print(f"  平均质量分数: {avg_quality:.3f}")
        print(f"  平均语义相似度: {avg_similarity:.3f}")
        print(f"  平均关键词保留: {avg_keyword_retention:.1%}")
        
        # 目标对比
        print(f"\n✅ 目标达成情况:")
        print(f"  压缩比 > 10x: {'✅' if avg_compression_ratio > 10 else '❌'} ({avg_compression_ratio:.2f}x)")
        print(f"  压缩延迟 < 10s: {'✅' if avg_compress_time < 10 else '❌'} ({avg_compress_time:.2f}s)")
        print(f"  重构延迟 < 500ms: {'✅' if avg_reconstruct_time < 0.5 else '❌'} ({avg_reconstruct_time*1000:.0f}ms)")
        print(f"  质量 > 0.85: {'✅' if avg_quality > 0.85 else '❌'} ({avg_quality:.3f})")
        print(f"  吞吐量 > 10/min: {'✅' if throughput > 10 else '❌'} ({throughput:.1f}/min)")
        
        # 保存报告
        report = {
            'summary': {
                'model': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'total_files': total_files,
                'success_count': success_count,
                'error_count': error_count,
                'avg_compression_ratio': avg_compression_ratio,
                'avg_compress_time': avg_compress_time,
                'avg_reconstruct_time': avg_reconstruct_time,
                'avg_quality_score': avg_quality,
                'avg_semantic_similarity': avg_similarity,
                'avg_keyword_retention': avg_keyword_retention,
                'throughput': throughput
            },
            'results': self.results,
            'errors': self.errors
        }
        
        report_file = self.output_dir / f"validation_report_{self.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n📄 详细报告已保存到: {report_file}")
        
        # 生成 Markdown 报告
        self._generate_markdown_report(report, report_file.with_suffix('.md'))
    
    def _generate_markdown_report(self, report: Dict[str, Any], output_file: Path):
        """生成 Markdown 格式报告"""
        summary = report['summary']
        
        md = f"""# 会话压缩验证报告

**模型**: {summary['model']}  
**时间**: {summary['timestamp']}  
**数据源**: Windsurf 会话记录

---

## 执行摘要

### 总体统计

- **总文件数**: {summary['total_files']}
- **成功**: {summary['success_count']} ({summary['success_count']/summary['total_files']*100:.1f}%)
- **失败**: {summary['error_count']} ({summary['error_count']/summary['total_files']*100:.1f}%)

### 性能指标

| 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|
| 平均压缩比 | {summary['avg_compression_ratio']:.2f}x | > 10x | {'✅' if summary['avg_compression_ratio'] > 10 else '❌'} |
| 平均压缩耗时 | {summary['avg_compress_time']:.2f}s | < 10s | {'✅' if summary['avg_compress_time'] < 10 else '❌'} |
| 平均重构耗时 | {summary['avg_reconstruct_time']*1000:.0f}ms | < 500ms | {'✅' if summary['avg_reconstruct_time'] < 0.5 else '❌'} |
| 吞吐量 | {summary['throughput']:.1f}/min | > 10/min | {'✅' if summary['throughput'] > 10 else '❌'} |

### 质量指标

| 指标 | 结果 | 目标 | 状态 |
|------|------|------|------|
| 平均质量分数 | {summary['avg_quality_score']:.3f} | > 0.85 | {'✅' if summary['avg_quality_score'] > 0.85 else '❌'} |
| 平均语义相似度 | {summary['avg_semantic_similarity']:.3f} | > 0.85 | {'✅' if summary['avg_semantic_similarity'] > 0.85 else '❌'} |
| 平均关键词保留 | {summary['avg_keyword_retention']:.1%} | > 90% | {'✅' if summary['avg_keyword_retention'] > 0.9 else '❌'} |

---

## 详细结果

"""
        
        for i, result in enumerate(report['results'], 1):
            md += f"""
### {i}. {result['file']}

- **消息数**: {result['message_count']}
- **原始长度**: {result['original_length']:,} 字符
- **压缩后**: {result['compressed_length']:,} 字符
- **压缩比**: {result['compression_ratio']:.2f}x
- **压缩耗时**: {result['compress_time']:.2f}s
- **重构耗时**: {result['reconstruct_time']:.2f}s
- **质量分数**: {result['quality_score']:.3f}
- **语义相似度**: {result['semantic_similarity']:.3f}
- **关键词保留**: {result.get('keyword_retention', 0):.1%}

"""
        
        if report['errors']:
            md += "\n---\n\n## 错误记录\n\n"
            for error in report['errors']:
                md += f"- **{error['file']}**: {error.get('error', '未知错误')}\n"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(md)
        
        print(f"📄 Markdown 报告已保存到: {output_file}")


async def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='会话压缩验证器')
    parser.add_argument('--data-dir', default='/Data/CascadeProjects/TalkingWithU',
                        help='会话数据目录')
    parser.add_argument('--model', default='gemma3',
                        choices=['gemma3', 'qwen2.5', 'tinyllama', 'cloud'],
                        help='使用的模型')
    parser.add_argument('--output-dir', default='validation_results',
                        help='输出目录')
    
    args = parser.parse_args()
    
    print("="*60)
    print("会话压缩验证器")
    print("="*60)
    print(f"数据目录: {args.data_dir}")
    print(f"模型: {args.model}")
    print(f"输出目录: {args.output_dir}")
    
    validator = ConversationValidator(
        data_dir=args.data_dir,
        model_name=args.model,
        output_dir=args.output_dir
    )
    
    await validator.validate_all()
    
    print("\n✅ 验证完成！")


if __name__ == "__main__":
    asyncio.run(main())
