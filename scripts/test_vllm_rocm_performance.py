#!/usr/bin/env python3
"""
vLLM (ROCm) vs Ollama (Vulkan) 性能对比测试
"""

import time
import requests
import json
from typing import List, Dict
import statistics

# 测试配置
VLLM_URL = "http://localhost:8000/v1/completions"
OLLAMA_URL = "http://localhost:11434/api/generate"

# 测试提示词（与之前 Ollama 测试相同）
TEST_PROMPTS = [
    "你好",  # 短提示
    "请解释一下什么是机器学习，以及它在现代社会中的应用。",  # 中等提示
    "请详细说明深度学习神经网络的工作原理，包括前向传播、反向传播和梯度下降算法。",  # 长提示
]

def test_vllm_inference(prompt: str, max_tokens: int = 100) -> Dict:
    """测试 vLLM 推理性能"""
    payload = {
        "model": "Qwen/Qwen2.5-1.5B-Instruct",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    
    start_time = time.time()
    try:
        response = requests.post(VLLM_URL, json=payload, timeout=30)
        response.raise_for_status()
        elapsed = time.time() - start_time
        
        result = response.json()
        return {
            "success": True,
            "latency": elapsed,
            "tokens": len(result["choices"][0]["text"].split()),
            "response": result["choices"][0]["text"][:100]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.time() - start_time
        }

def test_ollama_inference(prompt: str) -> Dict:
    """测试 Ollama 推理性能"""
    payload = {
        "model": "qwen2.5:7b-instruct",
        "prompt": prompt,
        "stream": False
    }
    
    start_time = time.time()
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=30)
        response.raise_for_status()
        elapsed = time.time() - start_time
        
        result = response.json()
        return {
            "success": True,
            "latency": elapsed,
            "tokens": len(result["response"].split()),
            "response": result["response"][:100]
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "latency": time.time() - start_time
        }

def run_comparison_test(num_runs: int = 3):
    """运行对比测试"""
    print("=" * 80)
    print("vLLM (ROCm) vs Ollama (Vulkan) 性能对比测试")
    print("=" * 80)
    print()
    
    results = {
        "vllm": [],
        "ollama": []
    }
    
    for i, prompt in enumerate(TEST_PROMPTS, 1):
        print(f"\n测试 {i}: {prompt[:50]}...")
        print("-" * 80)
        
        # 测试 vLLM
        print("\n[vLLM (ROCm)]")
        vllm_times = []
        for run in range(num_runs):
            result = test_vllm_inference(prompt)
            if result["success"]:
                vllm_times.append(result["latency"])
                print(f"  运行 {run+1}: {result['latency']:.3f}s ({result['tokens']} tokens)")
            else:
                print(f"  运行 {run+1}: 失败 - {result['error']}")
        
        if vllm_times:
            avg_vllm = statistics.mean(vllm_times)
            results["vllm"].append(avg_vllm)
            print(f"  平均: {avg_vllm:.3f}s")
        
        # 测试 Ollama
        print("\n[Ollama (Vulkan)]")
        ollama_times = []
        for run in range(num_runs):
            result = test_ollama_inference(prompt)
            if result["success"]:
                ollama_times.append(result["latency"])
                print(f"  运行 {run+1}: {result['latency']:.3f}s ({result['tokens']} tokens)")
            else:
                print(f"  运行 {run+1}: 失败 - {result['error']}")
        
        if ollama_times:
            avg_ollama = statistics.mean(ollama_times)
            results["ollama"].append(avg_ollama)
            print(f"  平均: {avg_ollama:.3f}s")
        
        # 对比
        if vllm_times and ollama_times:
            speedup = avg_ollama / avg_vllm
            print(f"\n  对比: vLLM 比 Ollama {'快' if speedup > 1 else '慢'} {abs(speedup-1)*100:.1f}%")
    
    # 总结
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    
    if results["vllm"] and results["ollama"]:
        avg_vllm_total = statistics.mean(results["vllm"])
        avg_ollama_total = statistics.mean(results["ollama"])
        overall_speedup = avg_ollama_total / avg_vllm_total
        
        print(f"\nvLLM (ROCm) 平均延迟: {avg_vllm_total:.3f}s")
        print(f"Ollama (Vulkan) 平均延迟: {avg_ollama_total:.3f}s")
        print(f"\n总体性能: vLLM 比 Ollama {'快' if overall_speedup > 1 else '慢'} {abs(overall_speedup-1)*100:.1f}%")
        
        if overall_speedup > 1:
            print(f"\n✅ ROCm 胜出！比 Vulkan 快 {(overall_speedup-1)*100:.1f}%")
        elif overall_speedup < 0.95:
            print(f"\n⚠️ Vulkan 胜出！比 ROCm 快 {(1/overall_speedup-1)*100:.1f}%")
        else:
            print(f"\n⚖️ 性能相当，差异在 5% 以内")

if __name__ == "__main__":
    print("等待 5 秒让服务准备就绪...")
    time.sleep(5)
    run_comparison_test(num_runs=3)
