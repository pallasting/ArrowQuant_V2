#!/usr/bin/env python3
"""
在宿主机上下载模型供 vLLM 使用
使用 HuggingFace 镜像加速
"""

import os
from huggingface_hub import snapshot_download

# 设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

model_id = "Qwen/Qwen2.5-7B-Instruct"
cache_dir = os.path.expanduser("~/.cache/huggingface")

print(f"下载模型: {model_id}")
print(f"缓存目录: {cache_dir}")
print(f"使用镜像: {os.environ.get('HF_ENDPOINT', 'huggingface.co')}")
print()

try:
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        resume_download=True,
    )
    print(f"\n✅ 模型下载完成!")
    print(f"本地路径: {local_dir}")
except Exception as e:
    print(f"\n❌ 下载失败: {e}")
    print("\n提示: 请确保已安装 huggingface_hub:")
    print("  pip install huggingface_hub")
