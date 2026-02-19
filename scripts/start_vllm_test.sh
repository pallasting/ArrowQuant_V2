#!/bin/bash
# 启动 vLLM 服务进行性能测试

echo "启动 vLLM (ROCm) 服务..."
echo "模型: Qwen/Qwen2.5-7B-Instruct"
echo "端口: 8000"
echo ""

# 使用 Docker 启动 vLLM
sudo docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=4g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_HUB_OFFLINE=0 \
    nalanzeyu/vllm-gfx906 \
    vllm serve Qwen/Qwen2.5-7B-Instruct \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 2048

echo ""
echo "vLLM 服务已停止"
