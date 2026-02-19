#!/usr/bin/env python3
"""
Phase 1.1 简化验证脚本

验证核心 Phase 1.1 验收标准：
- 本地模型可用
- 基本功能正常
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from llm_compression.logger import logger


class SimplePhase11Validator:
    """简化的 Phase 1.1 验证器"""
    
    def __init__(self):
        """初始化验证器"""
        self.passed_checks = 0
        self.total_checks = 0
        logger.info("SimplePhase11Validator initialized")
    
    def check(self, name: str, condition: bool, details: str = ""):
        """检查单个条件"""
        self.total_checks += 1
        status = "✓ PASS" if condition else "✗ FAIL"
        
        if condition:
            self.passed_checks += 1
        
        print(f"  [{status}] {name}")
        if details:
            print(f"        {details}")
        
        return condition
    
    async def validate_ollama_service(self) -> bool:
        """验证 Ollama 服务"""
        print("\n" + "=" * 70)
        print("检查 1: Ollama 服务")
        print("=" * 70)
        
        try:
            # 检查 ollama 命令是否存在
            result = subprocess.run(
                ["which", "ollama"],
                capture_output=True,
                timeout=5
            )
            
            ollama_installed = result.returncode == 0
            self.check(
                "Ollama 已安装",
                ollama_installed,
                f"路径: {result.stdout.decode().strip()}" if ollama_installed else "未找到"
            )
            
            if not ollama_installed:
                return False
            
            # 检查 ollama 进程
            result = subprocess.run(
                ["pgrep", "-x", "ollama"],
                capture_output=True,
                timeout=5
            )
            
            ollama_running = result.returncode == 0
            self.check(
                "Ollama 服务运行中",
                ollama_running,
                f"PID: {result.stdout.decode().strip()}" if ollama_running else "未运行"
            )
            
            return ollama_installed and ollama_running
            
        except Exception as e:
            logger.error(f"Ollama service validation failed: {e}")
            self.check("Ollama 服务", False, f"错误: {e}")
            return False
    
    async def validate_models_installed(self) -> bool:
        """验证模型已安装"""
        print("\n" + "=" * 70)
        print("检查 2: 已安装模型")
        print("=" * 70)
        
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=10,
                text=True
            )
            
            if result.returncode != 0:
                self.check("列出模型", False, "命令失败")
                return False
            
            output = result.stdout
            print(f"\n{output}")
            
            # 检查 Qwen2.5 模型
            qwen_installed = "qwen2.5" in output.lower()
            self.check(
                "Qwen2.5 模型已安装",
                qwen_installed,
                "已找到" if qwen_installed else "未找到"
            )
            
            return qwen_installed
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            self.check("模型检查", False, f"错误: {e}")
            return False
    
    async def validate_gpu_backend(self) -> bool:
        """验证 GPU 后端"""
        print("\n" + "=" * 70)
        print("检查 3: GPU 后端")
        print("=" * 70)
        
        backends_available = []
        
        # 检查 ROCm
        try:
            result = subprocess.run(
                ["rocm-smi"],
                capture_output=True,
                timeout=10
            )
            rocm_available = result.returncode == 0
            if rocm_available:
                backends_available.append("ROCm")
            self.check("ROCm 可用", rocm_available)
        except Exception:
            self.check("ROCm 可用", False)
        
        # 检查 Vulkan
        try:
            result = subprocess.run(
                ["vulkaninfo", "--summary"],
                capture_output=True,
                timeout=10
            )
            vulkan_available = result.returncode == 0
            if vulkan_available:
                backends_available.append("Vulkan")
            self.check("Vulkan 可用", vulkan_available)
        except Exception:
            self.check("Vulkan 可用", False)
        
        # 检查 OpenCL
        try:
            result = subprocess.run(
                ["clinfo"],
                capture_output=True,
                timeout=10
            )
            opencl_available = result.returncode == 0
            if opencl_available:
                backends_available.append("OpenCL")
            self.check("OpenCL 可用", opencl_available)
        except Exception:
            self.check("OpenCL 可用", False)
        
        has_gpu = len(backends_available) > 0
        if has_gpu:
            print(f"\n  可用后端: {', '.join(backends_available)}")
        
        return has_gpu
    
    async def validate_basic_inference(self) -> bool:
        """验证基本推理功能"""
        print("\n" + "=" * 70)
        print("检查 4: 基本推理")
        print("=" * 70)
        
        try:
            print("\n  测试推理: 'Hello, how are you?'")
            
            result = subprocess.run(
                ["ollama", "run", "qwen2.5:7b-instruct", "Hello, how are you?"],
                capture_output=True,
                timeout=30,
                text=True
            )
            
            inference_works = result.returncode == 0 and len(result.stdout) > 0
            
            self.check(
                "推理成功",
                inference_works,
                f"输出长度: {len(result.stdout)} 字符" if inference_works else "失败"
            )
            
            if inference_works:
                print(f"\n  响应预览: {result.stdout[:200]}...")
            
            return inference_works
            
        except subprocess.TimeoutExpired:
            self.check("推理成功", False, "超时 (30s)")
            return False
        except Exception as e:
            logger.error(f"Inference validation failed: {e}")
            self.check("推理成功", False, f"错误: {e}")
            return False
    
    async def run_validation(self) -> bool:
        """运行所有验证"""
        print("\n" + "=" * 70)
        print("Phase 1.1 简化验证")
        print("=" * 70)
        
        # 运行所有检查
        checks = [
            ("Ollama 服务", self.validate_ollama_service()),
            ("已安装模型", self.validate_models_installed()),
            ("GPU 后端", self.validate_gpu_backend()),
            ("基本推理", self.validate_basic_inference()),
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
    
    def generate_report(self, results):
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
            print("\n🎉 Phase 1.1 基础验证通过！")
            print("本地模型部署系统就绪。")
        else:
            print("\n⚠️  Phase 1.1 验证未完全通过")
            print(f"有 {self.total_checks - self.passed_checks} 项检查失败。")


async def main():
    """主函数"""
    validator = SimplePhase11Validator()
    
    try:
        success = await validator.run_validation()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Validation failed with exception: {e}", exc_info=True)
        print(f"\n✗ 验证过程出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
