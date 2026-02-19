#!/bin/bash
# 使用 HuggingFace 镜像站启动 vLLM 服务

echo "=========================================="
echo "vLLM (ROCm) 性能测试"
echo "模型源: hf-mirror.com (国内镜像)"
echo "模型: Qwen/Qwen2.5-7B-Instruct"
echo "=========================================="
echo ""

echo "启动 vLLM 服务（端口 8000）..."
echo "使用 HuggingFace 国内镜像加速下载..."
echo ""

sudo docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=4g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_ENDPOINT=https://hf-mirror.com \
    nalanzeyu/vllm-gfx906 \
    vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048 \
    --trust-remote-code

echo ""
echo "vLLM 服务已停止"
