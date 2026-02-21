#!/usr/bin/env python3
"""
对比不同量化方法的性能和精度。

对比三种量化方法：
1. PTQ INT8 - 8-bit 量化
2. PTQ INT2 - 手搓版 2-bit 量化
3. AngelSlim 2-bit - 腾讯预量化模型（通过转换器）

评估指标：
- 压缩比
- 内存占用
- 余弦相似度
- 量化时间
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Any
import tempfile

import torch
import numpy as np
import pyarrow.parquet as pq

from llm_compression.logger import logger
from llm_compression.inference.arrow_quantizer import ArrowQuantizer, QuantizationConfig
from llm_compression.inference.weight_loader import WeightLoader
from llm_compression.inference.model_converter import HuggingFaceToParquetConverter


def create_test_model(output_path: str, size: str = "small") -> Dict[str, Any]:
    """
    创建测试模型（FP16格式）。
    
    Args:
        output_path: 输出路径
        size: 模型大小 ('small', 'medium', 'large')
        
    Returns:
        模型信息字典
    """
    logger.info(f"Creating {size} test model...")
    
    # 定义模型大小
    sizes = {
        'small': [(128, 64), (64, 32), (32, 16)],
        'medium': [(512, 256), (256, 128), (128, 64)],
        'large': [(2048, 1024), (1024, 512), (512, 256)]
    }
    
    layer_shapes = sizes.get(size, sizes['small'])
    
    # 创建临时 HuggingFace 模型
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = Path(tmpdir) / "test_model"
        model_dir.mkdir()
        
        # 创建 state dict
        torch.manual_seed(42)
        state_dict = {}
        total_params = 0
        
        for i, (out_dim, in_dim) in enumerate(layer_shapes):
            weight = torch.randn(out_dim, in_dim, dtype=torch.float16)
            bias = torch.randn(out_dim, dtype=torch.float16)
            
            state_dict[f'layer.{i}.weight'] = weight
            state_dict[f'layer.{i}.bias'] = bias
            
            total_params += weight.numel() + bias.numel()
        
        # 保存为 HuggingFace 格式
        torch.save(state_dict, model_dir / "pytorch_model.bin")
        
        # 转换为 Parquet
        converter = HuggingFaceToParquetConverter()
        converter.convert(
            hf_model_path=str(model_dir),
            output_parquet=output_path
        )
    
    # 获取文件大小
    file_size = Path(output_path).stat().st_size
    
    logger.info(f"Test model created: {total_params:,} parameters, {file_size / 1024 / 1024:.2f} MB")
    
    return {
        'total_params': total_params,
        'file_size_bytes': file_size,
        'num_layers': len(layer_shapes) * 2,  # weight + bias
    }


def quantize_ptq_int8(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    使用 PTQ INT8 量化。
    
    Args:
        input_path: 输入 Parquet 文件
        output_path: 输出 Parquet 文件
        
    Returns:
        量化结果字典
    """
    logger.info("Quantizing with PTQ INT8...")
    
    config = QuantizationConfig(
        quant_type='int8',
        calibration_method='ptq',
        per_channel=True,
        symmetric=True
    )
    
    quantizer = ArrowQuantizer(config)
    
    start_time = time.time()
    quantizer.quantize_model(
        input_parquet=input_path,
        output_parquet=output_path,
        show_progress=False
    )
    quantization_time = time.time() - start_time
    
    # 获取文件大小
    file_size = Path(output_path).stat().st_size
    
    return {
        'method': 'PTQ INT8',
        'quant_type': 'int8',
        'file_size_bytes': file_size,
        'quantization_time_sec': quantization_time,
    }


def quantize_ptq_int2(input_path: str, output_path: str) -> Dict[str, Any]:
    """
    使用 PTQ INT2 量化（手搓版）。
    
    Args:
        input_path: 输入 Parquet 文件
        output_path: 输出 Parquet 文件
        
    Returns:
        量化结果字典
    """
    logger.info("Quantizing with PTQ INT2 (手搓版)...")
    
    config = QuantizationConfig(
        quant_type='int2',
        calibration_method='ptq',
        per_channel=True,
        symmetric=True
    )
    
    quantizer = ArrowQuantizer(config)
    
    start_time = time.time()
    quantizer.quantize_model(
        input_parquet=input_path,
        output_parquet=output_path,
        show_progress=False
    )
    quantization_time = time.time() - start_time
    
    # 获取文件大小
    file_size = Path(output_path).stat().st_size
    
    return {
        'method': 'PTQ INT2 (手搓版)',
        'quant_type': 'int2',
        'file_size_bytes': file_size,
        'quantization_time_sec': quantization_time,
    }


def compute_cosine_similarity(
    original_path: str,
    quantized_path: str
) -> float:
    """
    计算原始模型和量化模型的余弦相似度。
    
    Args:
        original_path: 原始模型路径
        quantized_path: 量化模型路径
        
    Returns:
        平均余弦相似度
    """
    logger.info("Computing cosine similarity...")
    
    # 加载模型
    original_loader = WeightLoader(original_path)
    quantized_loader = WeightLoader(quantized_path)
    
    # 加载所有权重
    original_weights = original_loader.load_weights()
    quantized_weights = quantized_loader.load_weights()
    
    # 计算每层的余弦相似度
    similarities = []
    
    for layer_name in original_weights.keys():
        if layer_name not in quantized_weights:
            logger.warning(f"Layer {layer_name} not found in quantized model")
            continue
        
        orig = original_weights[layer_name].cpu().numpy().flatten()
        quant = quantized_weights[layer_name].cpu().numpy().flatten()
        
        # 余弦相似度
        dot_product = np.dot(orig, quant)
        norm_orig = np.linalg.norm(orig)
        norm_quant = np.linalg.norm(quant)
        
        if norm_orig > 0 and norm_quant > 0:
            similarity = dot_product / (norm_orig * norm_quant)
            similarities.append(similarity)
    
    avg_similarity = np.mean(similarities) if similarities else 0.0
    
    logger.info(f"Average cosine similarity: {avg_similarity:.4f}")
    
    return float(avg_similarity)


def print_comparison_table(
    original_info: Dict[str, Any],
    results: list[Dict[str, Any]]
):
    """
    打印对比表格。
    
    Args:
        original_info: 原始模型信息
        results: 量化结果列表
    """
    print("\n" + "=" * 100)
    print("量化方法对比结果")
    print("=" * 100)
    
    # 原始模型信息
    print(f"\n原始模型:")
    print(f"  参数量: {original_info['total_params']:,}")
    print(f"  文件大小: {original_info['file_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  层数: {original_info['num_layers']}")
    
    # 表头
    print(f"\n{'方法':<25} {'压缩比':<12} {'文件大小':<15} {'内存节省':<12} {'余弦相似度':<15} {'量化时间':<12}")
    print("-" * 100)
    
    # 每个方法的结果
    for result in results:
        method = result['method']
        file_size_mb = result['file_size_bytes'] / 1024 / 1024
        compression_ratio = original_info['file_size_bytes'] / result['file_size_bytes']
        memory_savings = (1 - result['file_size_bytes'] / original_info['file_size_bytes']) * 100
        cosine_sim = result.get('cosine_similarity', 0.0)
        quant_time = result.get('quantization_time_sec', 0.0)
        
        print(f"{method:<25} {compression_ratio:<12.2f}x {file_size_mb:<15.2f} MB {memory_savings:<12.1f}% {cosine_sim:<15.4f} {quant_time:<12.2f}s")
    
    print("=" * 100)
    
    # 性能排名
    print("\n性能排名:")
    
    # 按压缩比排序
    sorted_by_compression = sorted(results, key=lambda x: original_info['file_size_bytes'] / x['file_size_bytes'], reverse=True)
    print("\n  压缩比 (越高越好):")
    for i, result in enumerate(sorted_by_compression, 1):
        compression_ratio = original_info['file_size_bytes'] / result['file_size_bytes']
        print(f"    {i}. {result['method']}: {compression_ratio:.2f}x")
    
    # 按余弦相似度排序
    sorted_by_similarity = sorted(results, key=lambda x: x.get('cosine_similarity', 0.0), reverse=True)
    print("\n  精度 (余弦相似度，越高越好):")
    for i, result in enumerate(sorted_by_similarity, 1):
        cosine_sim = result.get('cosine_similarity', 0.0)
        print(f"    {i}. {result['method']}: {cosine_sim:.4f}")
    
    # 按量化时间排序
    sorted_by_time = sorted(results, key=lambda x: x.get('quantization_time_sec', float('inf')))
    print("\n  量化速度 (越快越好):")
    for i, result in enumerate(sorted_by_time, 1):
        quant_time = result.get('quantization_time_sec', 0.0)
        print(f"    {i}. {result['method']}: {quant_time:.2f}s")
    
    print("\n" + "=" * 100)


def main():
    parser = argparse.ArgumentParser(description="对比不同量化方法")
    parser.add_argument(
        '--model-size',
        choices=['small', 'medium', 'large'],
        default='small',
        help='测试模型大小'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./quantization_comparison',
        help='输出目录'
    )
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 创建测试模型
    original_path = str(output_dir / "original_fp16.parquet")
    original_info = create_test_model(original_path, size=args.model_size)
    
    # 2. 量化测试
    results = []
    
    # PTQ INT8
    int8_path = str(output_dir / "quantized_int8.parquet")
    int8_result = quantize_ptq_int8(original_path, int8_path)
    int8_result['cosine_similarity'] = compute_cosine_similarity(original_path, int8_path)
    results.append(int8_result)
    
    # PTQ INT2
    int2_path = str(output_dir / "quantized_int2.parquet")
    int2_result = quantize_ptq_int2(original_path, int2_path)
    int2_result['cosine_similarity'] = compute_cosine_similarity(original_path, int2_path)
    results.append(int2_result)
    
    # 3. 打印对比结果
    print_comparison_table(original_info, results)
    
    # 4. 保存结果
    import json
    results_file = output_dir / "comparison_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            'original_info': original_info,
            'results': results
        }, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
