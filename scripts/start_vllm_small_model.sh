#!/bin/bash
# 使用小模型启动 vLLM（适配 MI50 16GB 显存）
# Qwen2.5-1.5B-Instruct 模型大小约 3GB，适合 16GB 显存

echo "启动 vLLM (ROCm) 服务 - 小模型配置..."
echo "模型: Qwen/Qwen2.5-1.5B-Instruct (~3GB)"
echo "GPU 内存利用率: 0.9"
echo "最大模型长度: 4096"
echo ""

sudo docker run --rm \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=4g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nalanzeyu/vllm-gfx906 \
    vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code
