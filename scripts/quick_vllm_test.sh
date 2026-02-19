#!/bin/bash
# 使用小模型快速测试 vLLM 性能

echo "=========================================="
echo "vLLM (ROCm) 快速性能测试"
echo "使用模型: TinyLlama-1.1B"
echo "=========================================="
echo ""

# 方案：使用宿主机已有的 HuggingFace 缓存，或者使用镜像站点
# 如果模型不存在，会自动下载（可能很慢）

echo "提示：如果下载很慢，请按 Ctrl+C 停止"
echo "我们可以改用其他方案"
echo ""
echo "启动 vLLM 服务..."

sudo docker run --rm \
    --device=/dev/kfd \
    --device=/dev/dri \
    --group-add video \
    --shm-size=2g \
    -p 8000:8000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_ENDPOINT=https://hf-mirror.com \
    nalanzeyu/vllm-gfx906 \
    vllm serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype float16 \
    --max-model-len 1024
