#!/bin/bash
# 使用宿主机网络启动 vLLM（可以访问代理）

echo "启动 vLLM (ROCm) 服务..."
echo "使用宿主机网络模式"
echo ""

sudo docker run --rm \
    --network=host \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=4g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e http_proxy="${http_proxy}" \
    -e https_proxy="${https_proxy}" \
    -e HTTP_PROXY="${HTTP_PROXY}" \
    -e HTTPS_PROXY="${HTTPS_PROXY}" \
    nalanzeyu/vllm-gfx906 \
    vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048 \
    --trust-remote-code
