#!/bin/bash
# 使用本地缓存的模型启动 vLLM

echo "启动 vLLM (ROCm) 服务..."
echo "使用本地缓存模型"
echo ""

sudo docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=4g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    nalanzeyu/vllm-gfx906 \
    vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048 \
    --trust-remote-code
