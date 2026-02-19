#!/bin/bash
# 使用 ModelScope 镜像启动 vLLM 服务

echo "=========================================="
echo "vLLM (ROCm) 性能测试"
echo "模型源: ModelScope.cn"
echo "模型: Qwen/Qwen2.5-7B-Instruct"
echo "=========================================="
echo ""

# ModelScope 模型 ID
# HuggingFace: Qwen/Qwen2.5-7B-Instruct
# ModelScope: qwen/Qwen2.5-7B-Instruct

echo "配置 ModelScope 环境变量..."
echo "启动 vLLM 服务（端口 8000）..."
echo ""

sudo docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=4g \
    -p 8000:8000 \
    -v ~/.cache/modelscope:/root/.cache/modelscope \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e VLLM_USE_MODELSCOPE=True \
    -e MODELSCOPE_CACHE=/root/.cache/modelscope \
    nalanzeyu/vllm-gfx906 \
    vllm serve qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048 \
    --trust-remote-code

echo ""
echo "vLLM 服务已停止"
