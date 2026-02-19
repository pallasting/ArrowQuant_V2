#!/bin/bash
# 使用低内存配置启动 vLLM（适配 MI50 16GB 显存）

echo "启动 vLLM (ROCm) 服务 - 低内存配置..."
echo "GPU 内存利用率: 0.85 (默认 0.9)"
echo "最大并发序列数: 16 (默认 256)"
echo "最大模型长度: 2048"
echo ""

sudo docker run --rm \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=4g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    nalanzeyu/vllm-gfx906 \
    vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048 \
    --max-num-seqs 16 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code
