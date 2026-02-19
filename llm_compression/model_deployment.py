"""
本地模型部署系统

管理本地 LLM 模型的下载、部署、量化和服务启动。
支持 Ollama 和 vLLM 两种部署框架，以及 ROCm、Vulkan、OpenCL 三层 GPU 后端。

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5
"""

import os
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import asyncio
import aiohttp

from llm_compression.logger import logger


class DeploymentFramework(Enum):
    """部署框架"""
    OLLAMA = "ollama"
    VLLM = "vllm"


class GPUBackend(Enum):
    """GPU 后端"""
    ROCM = "rocm"
    VULKAN = "vulkan"
    OPENCL = "opencl"
    CPU = "cpu"


class QuantizationType(Enum):
    """量化类型"""
    NONE = "none"        # 无量化（FP16/FP32）
    INT8 = "int8"        # 8-bit 量化
    INT4 = "int4"        # 4-bit 量化
    Q4_K_M = "q4_k_m"    # Ollama Q4_K_M 量化
    Q5_K_M = "q5_k_m"    # Ollama Q5_K_M 量化
    Q8_0 = "q8_0"        # Ollama Q8_0 量化


@dataclass
class ModelInfo:
    """模型信息"""
    name: str                          # 模型名称（如 "qwen2.5:7b-instruct"）
    display_name: str                  # 显示名称（如 "Qwen2.5-7B-Instruct"）
    size_gb: float                     # 模型大小（GB）
    parameters: str                    # 参数量（如 "7B"）
    context_length: int                # 上下文长度
    quantization: QuantizationType     # 量化类型
    framework: DeploymentFramework     # 部署框架
    endpoint: str                      # API 端点
    is_downloaded: bool = False        # 是否已下载
    is_running: bool = False           # 是否正在运行


@dataclass
class DeploymentConfig:
    """部署配置"""
    framework: DeploymentFramework = DeploymentFramework.OLLAMA
    gpu_backend: GPUBackend = GPUBackend.ROCM
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    tensor_parallel_size: int = 1
    port: int = 11434  # Ollama 默认端口
    
    # Ollama 特定配置
    ollama_num_parallel: int = 4
    ollama_max_queue: int = 512
    ollama_max_loaded_models: int = 2
    
    # ROCm 特定配置
    hsa_override_gfx_version: str = "9.0.6"  # Mi50 = gfx906


class ModelDeploymentSystem:
    """本地模型部署系统"""
    
    # 预定义的模型配置
    SUPPORTED_MODELS = {
        "qwen2.5:7b-instruct": ModelInfo(
            name="qwen2.5:7b-instruct",
            display_name="Qwen2.5-7B-Instruct",
            size_gb=4.7,
            parameters="7B",
            context_length=32768,
            quantization=QuantizationType.Q4_K_M,
            framework=DeploymentFramework.OLLAMA,
            endpoint="http://localhost:11434/v1"
        ),
        "qwen2.5:7b-instruct-q4_k_m": ModelInfo(
            name="qwen2.5:7b-instruct-q4_k_m",
            display_name="Qwen2.5-7B-Instruct (Q4_K_M)",
            size_gb=4.7,
            parameters="7B",
            context_length=32768,
            quantization=QuantizationType.Q4_K_M,
            framework=DeploymentFramework.OLLAMA,
            endpoint="http://localhost:11434/v1"
        ),
        "llama3.1:8b-instruct-q4_k_m": ModelInfo(
            name="llama3.1:8b-instruct-q4_k_m",
            display_name="Llama 3.1 8B Instruct (Q4_K_M)",
            size_gb=4.9,
            parameters="8B",
            context_length=8192,
            quantization=QuantizationType.Q4_K_M,
            framework=DeploymentFramework.OLLAMA,
            endpoint="http://localhost:11434/v1"
        ),
        "gemma3:4b": ModelInfo(
            name="gemma3:4b",
            display_name="Gemma 3 4B",
            size_gb=3.3,
            parameters="4B",
            context_length=8192,
            quantization=QuantizationType.Q4_K_M,
            framework=DeploymentFramework.OLLAMA,
            endpoint="http://localhost:11434/v1"
        ),
    }
    
    def __init__(self, config: Optional[DeploymentConfig] = None):
        """
        初始化模型部署系统
        
        Args:
            config: 部署配置（可选，使用默认配置）
        """
        self.config = config or DeploymentConfig()
        self.deployed_models: Dict[str, ModelInfo] = {}
        
        logger.info(
            f"ModelDeploymentSystem initialized: "
            f"framework={self.config.framework.value}, "
            f"gpu_backend={self.config.gpu_backend.value}"
        )
    
    async def check_prerequisites(self) -> Tuple[bool, List[str]]:
        """
        检查部署前提条件
        
        Returns:
            Tuple[bool, List[str]]: (是否满足条件, 缺失的组件列表)
        """
        missing = []
        
        # 检查部署框架
        if self.config.framework == DeploymentFramework.OLLAMA:
            if not self._check_command("ollama"):
                missing.append("Ollama (install: curl -fsSL https://ollama.com/install.sh | sh)")
        elif self.config.framework == DeploymentFramework.VLLM:
            if not self._check_python_package("vllm"):
                missing.append("vLLM (install: pip install vllm)")
        
        # 检查 GPU 后端
        if self.config.gpu_backend == GPUBackend.ROCM:
            if not self._check_command("rocm-smi"):
                missing.append("ROCm (install: sudo apt install rocm-smi)")
        elif self.config.gpu_backend == GPUBackend.VULKAN:
            if not self._check_command("vulkaninfo"):
                missing.append("Vulkan (install: sudo apt install vulkan-tools)")
        elif self.config.gpu_backend == GPUBackend.OPENCL:
            if not self._check_command("clinfo"):
                missing.append("OpenCL (install: sudo apt install clinfo)")
        
        # 检查 GPU 可用性
        if self.config.gpu_backend != GPUBackend.CPU:
            gpu_available = await self._check_gpu_available()
            if not gpu_available:
                logger.warning(f"GPU backend {self.config.gpu_backend.value} not available")
        
        is_ready = len(missing) == 0
        
        if is_ready:
            logger.info("✅ All prerequisites satisfied")
        else:
            logger.warning(f"❌ Missing prerequisites: {missing}")
        
        return is_ready, missing
    
    def _check_command(self, command: str) -> bool:
        """检查命令是否可用"""
        try:
            subprocess.run(
                ["which", command],
                check=True,
                capture_output=True,
                timeout=5
            )
            return True
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def _check_python_package(self, package: str) -> bool:
        """检查 Python 包是否已安装"""
        try:
            __import__(package)
            return True
        except ImportError:
            return False
    
    async def _check_gpu_available(self) -> bool:
        """检查 GPU 是否可用"""
        if self.config.gpu_backend == GPUBackend.ROCM:
            try:
                result = subprocess.run(
                    ["rocm-smi"],
                    capture_output=True,
                    timeout=10,
                    text=True
                )
                return result.returncode == 0
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
        
        elif self.config.gpu_backend == GPUBackend.VULKAN:
            try:
                result = subprocess.run(
                    ["vulkaninfo", "--summary"],
                    capture_output=True,
                    timeout=10,
                    text=True
                )
                return result.returncode == 0 and "deviceName" in result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
        
        elif self.config.gpu_backend == GPUBackend.OPENCL:
            try:
                result = subprocess.run(
                    ["clinfo"],
                    capture_output=True,
                    timeout=10,
                    text=True
                )
                return result.returncode == 0 and "Platform Name" in result.stdout
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
        
        return True
    
    async def list_available_models(self) -> List[ModelInfo]:
        """
        列出可用的模型
        
        Returns:
            List[ModelInfo]: 可用模型列表
        """
        models = []
        
        if self.config.framework == DeploymentFramework.OLLAMA:
            # 检查 Ollama 已下载的模型
            try:
                result = subprocess.run(
                    ["ollama", "list"],
                    capture_output=True,
                    timeout=10,
                    text=True
                )
                
                if result.returncode == 0:
                    # 解析输出
                    lines = result.stdout.strip().split('\n')[1:]  # 跳过标题行
                    for line in lines:
                        if line.strip():
                            parts = line.split()
                            if len(parts) >= 1:
                                model_name = parts[0]
                                # 检查是否在支持列表中
                                if model_name in self.SUPPORTED_MODELS:
                                    model_info = self.SUPPORTED_MODELS[model_name]
                                    model_info.is_downloaded = True
                                    models.append(model_info)
            except (subprocess.TimeoutExpired, FileNotFoundError) as e:
                logger.error(f"Failed to list Ollama models: {e}")
        
        # 添加未下载的支持模型
        for model_name, model_info in self.SUPPORTED_MODELS.items():
            if not any(m.name == model_name for m in models):
                models.append(model_info)
        
        return models
    
    async def download_model(self, model_name: str) -> bool:
        """
        下载模型
        
        Args:
            model_name: 模型名称
            
        Returns:
            bool: 是否成功
        """
        if model_name not in self.SUPPORTED_MODELS:
            logger.error(f"Unsupported model: {model_name}")
            return False
        
        model_info = self.SUPPORTED_MODELS[model_name]
        
        if self.config.framework == DeploymentFramework.OLLAMA:
            logger.info(f"Downloading model {model_name} via Ollama...")
            
            try:
                # 使用 ollama pull 下载模型
                process = subprocess.Popen(
                    ["ollama", "pull", model_name],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )
                
                # 实时输出进度
                for line in process.stdout:
                    logger.info(line.strip())
                
                process.wait()
                
                if process.returncode == 0:
                    logger.info(f"✅ Model {model_name} downloaded successfully")
                    model_info.is_downloaded = True
                    return True
                else:
                    logger.error(f"❌ Failed to download model {model_name}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error downloading model {model_name}: {e}")
                return False
        
        elif self.config.framework == DeploymentFramework.VLLM:
            # vLLM 使用 Hugging Face 模型，自动下载
            logger.info(f"vLLM will auto-download model {model_name} on first use")
            return True
        
        return False
    
    async def start_service(self, model_name: Optional[str] = None) -> bool:
        """
        启动模型服务
        
        Args:
            model_name: 要加载的模型名称（可选，Ollama 可以不指定）
            
        Returns:
            bool: 是否成功启动
        """
        if self.config.framework == DeploymentFramework.OLLAMA:
            return await self._start_ollama_service(model_name)
        elif self.config.framework == DeploymentFramework.VLLM:
            if not model_name:
                logger.error("vLLM requires model_name to be specified")
                return False
            return await self._start_vllm_service(model_name)
        
        return False
    
    async def _start_ollama_service(self, model_name: Optional[str] = None) -> bool:
        """启动 Ollama 服务"""
        # 检查服务是否已运行
        if await self._check_ollama_running():
            logger.info("Ollama service is already running")
            return True
        
        logger.info("Starting Ollama service...")
        
        # 设置环境变量
        env = os.environ.copy()
        
        # GPU 后端配置
        if self.config.gpu_backend == GPUBackend.ROCM:
            env["OLLAMA_GPU_DRIVER"] = "rocm"
            env["HSA_OVERRIDE_GFX_VERSION"] = self.config.hsa_override_gfx_version
            env["ROCM_PATH"] = "/opt/rocm"
            logger.info(f"Using ROCm backend with GFX version {self.config.hsa_override_gfx_version}")
        elif self.config.gpu_backend == GPUBackend.VULKAN:
            env["OLLAMA_GPU_DRIVER"] = "vulkan"
            logger.info("Using Vulkan backend")
        elif self.config.gpu_backend == GPUBackend.CPU:
            env["OLLAMA_GPU_DRIVER"] = "cpu"
            logger.info("Using CPU backend")
        
        # Ollama 性能配置
        env["OLLAMA_MAX_LOADED_MODELS"] = str(self.config.ollama_max_loaded_models)
        env["OLLAMA_NUM_PARALLEL"] = str(self.config.ollama_num_parallel)
        env["OLLAMA_MAX_QUEUE"] = str(self.config.ollama_max_queue)
        
        try:
            # 启动 Ollama 服务（后台运行）
            subprocess.Popen(
                ["ollama", "serve"],
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 等待服务启动
            for i in range(30):  # 最多等待 30 秒
                await asyncio.sleep(1)
                if await self._check_ollama_running():
                    logger.info("✅ Ollama service started successfully")
                    return True
            
            logger.error("❌ Ollama service failed to start within 30 seconds")
            return False
            
        except Exception as e:
            logger.error(f"Error starting Ollama service: {e}")
            return False
    
    async def _start_vllm_service(self, model_name: str) -> bool:
        """启动 vLLM 服务"""
        logger.info(f"Starting vLLM service with model {model_name}...")
        
        # vLLM 命令行参数
        cmd = [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_name,
            "--tensor-parallel-size", str(self.config.tensor_parallel_size),
            "--gpu-memory-utilization", str(self.config.gpu_memory_utilization),
            "--max-model-len", str(self.config.max_model_len),
            "--port", str(self.config.port)
        ]
        
        # GPU 后端配置
        env = os.environ.copy()
        if self.config.gpu_backend == GPUBackend.ROCM:
            env["HIP_VISIBLE_DEVICES"] = "0"
            env["HSA_OVERRIDE_GFX_VERSION"] = self.config.hsa_override_gfx_version
        
        try:
            # 启动 vLLM 服务（后台运行）
            subprocess.Popen(
                cmd,
                env=env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL
            )
            
            # 等待服务启动
            for i in range(60):  # 最多等待 60 秒（vLLM 启动较慢）
                await asyncio.sleep(1)
                if await self._check_service_health(f"http://localhost:{self.config.port}/health"):
                    logger.info("✅ vLLM service started successfully")
                    return True
            
            logger.error("❌ vLLM service failed to start within 60 seconds")
            return False
            
        except Exception as e:
            logger.error(f"Error starting vLLM service: {e}")
            return False
    
    async def _check_ollama_running(self) -> bool:
        """检查 Ollama 服务是否运行"""
        try:
            # 检查进程
            result = subprocess.run(
                ["pgrep", "-x", "ollama"],
                capture_output=True,
                timeout=5
            )
            if result.returncode != 0:
                return False
            
            # 检查 API 健康
            return await self._check_service_health("http://localhost:11434/api/tags")
        except Exception:
            return False
    
    async def _check_service_health(self, url: str) -> bool:
        """检查服务健康状态"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def get_model_info(self, model_name: str) -> Optional[ModelInfo]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            Optional[ModelInfo]: 模型信息，如果不存在则返回 None
        """
        if model_name in self.SUPPORTED_MODELS:
            model_info = self.SUPPORTED_MODELS[model_name]
            
            # 检查是否已下载
            if self.config.framework == DeploymentFramework.OLLAMA:
                try:
                    result = subprocess.run(
                        ["ollama", "list"],
                        capture_output=True,
                        timeout=10,
                        text=True
                    )
                    model_info.is_downloaded = model_name in result.stdout
                except Exception:
                    pass
            
            # 检查是否正在运行
            model_info.is_running = await self._check_ollama_running()
            
            return model_info
        
        return None
    
    def get_endpoint(self, model_name: str) -> str:
        """
        获取模型的 API 端点
        
        Args:
            model_name: 模型名称
            
        Returns:
            str: API 端点 URL
        """
        if model_name in self.SUPPORTED_MODELS:
            return self.SUPPORTED_MODELS[model_name].endpoint
        
        # 默认端点
        if self.config.framework == DeploymentFramework.OLLAMA:
            return "http://localhost:11434/v1"
        else:
            return f"http://localhost:{self.config.port}/v1"
    
    async def stop_service(self) -> bool:
        """
        停止模型服务
        
        Returns:
            bool: 是否成功停止
        """
        if self.config.framework == DeploymentFramework.OLLAMA:
            try:
                subprocess.run(
                    ["pkill", "-x", "ollama"],
                    timeout=10
                )
                logger.info("Ollama service stopped")
                return True
            except Exception as e:
                logger.error(f"Error stopping Ollama service: {e}")
                return False
        
        # vLLM 需要手动停止进程
        logger.warning("vLLM service must be stopped manually")
        return False
    
    def get_quantization_recommendation(self, gpu_memory_gb: float) -> QuantizationType:
        """
        根据 GPU 内存推荐量化类型
        
        Args:
            gpu_memory_gb: GPU 内存大小（GB）
            
        Returns:
            QuantizationType: 推荐的量化类型
        """
        if gpu_memory_gb >= 24:
            return QuantizationType.Q8_0  # 高质量
        elif gpu_memory_gb >= 16:
            return QuantizationType.Q5_K_M  # 中等质量
        elif gpu_memory_gb >= 8:
            return QuantizationType.Q4_K_M  # 标准质量
        else:
            return QuantizationType.INT4  # 最小内存
