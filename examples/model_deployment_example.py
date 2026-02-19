"""
本地模型部署示例

演示如何使用 ModelDeploymentSystem 部署和管理本地 LLM 模型。
"""

import asyncio
from llm_compression.model_deployment import (
    ModelDeploymentSystem,
    DeploymentConfig,
    DeploymentFramework,
    GPUBackend,
    QuantizationType
)
from llm_compression.logger import logger


async def main():
    """主函数"""
    
    print("=" * 60)
    print("本地模型部署示例")
    print("=" * 60)
    print()
    
    # 1. 创建部署配置
    print("1. 创建部署配置...")
    config = DeploymentConfig(
        framework=DeploymentFramework.OLLAMA,
        gpu_backend=GPUBackend.ROCM,
        hsa_override_gfx_version="9.0.6",  # Mi50 = gfx906
        ollama_num_parallel=4,
        ollama_max_queue=512,
        ollama_max_loaded_models=2
    )
    print(f"   框架: {config.framework.value}")
    print(f"   GPU 后端: {config.gpu_backend.value}")
    print(f"   GFX 版本: {config.hsa_override_gfx_version}")
    print()
    
    # 2. 初始化部署系统
    print("2. 初始化部署系统...")
    deployment = ModelDeploymentSystem(config)
    print("   ✅ 部署系统已初始化")
    print()
    
    # 3. 检查前提条件
    print("3. 检查前提条件...")
    is_ready, missing = await deployment.check_prerequisites()
    if is_ready:
        print("   ✅ 所有前提条件已满足")
    else:
        print("   ❌ 缺少以下组件:")
        for component in missing:
            print(f"      - {component}")
        print()
        print("   请安装缺失的组件后重试")
        return
    print()
    
    # 4. 列出可用模型
    print("4. 列出可用模型...")
    models = await deployment.list_available_models()
    print(f"   找到 {len(models)} 个模型:")
    for model in models:
        status = "✅ 已下载" if model.is_downloaded else "⬇️  未下载"
        print(f"   - {model.display_name} ({model.parameters}, {model.size_gb}GB) {status}")
    print()
    
    # 5. 获取模型信息
    print("5. 获取 Qwen2.5-7B 模型信息...")
    model_name = "qwen2.5:7b-instruct"
    model_info = await deployment.get_model_info(model_name)
    if model_info:
        print(f"   名称: {model_info.display_name}")
        print(f"   参数: {model_info.parameters}")
        print(f"   大小: {model_info.size_gb} GB")
        print(f"   上下文长度: {model_info.context_length}")
        print(f"   量化类型: {model_info.quantization.value}")
        print(f"   已下载: {'是' if model_info.is_downloaded else '否'}")
        print(f"   端点: {model_info.endpoint}")
    else:
        print(f"   ❌ 模型 {model_name} 不存在")
    print()
    
    # 6. 下载模型（如果未下载）
    if model_info and not model_info.is_downloaded:
        print(f"6. 下载模型 {model_name}...")
        success = await deployment.download_model(model_name)
        if success:
            print("   ✅ 模型下载成功")
        else:
            print("   ❌ 模型下载失败")
        print()
    else:
        print(f"6. 模型 {model_name} 已下载，跳过下载步骤")
        print()
    
    # 7. 启动服务
    print("7. 启动 Ollama 服务...")
    service_running = await deployment._check_ollama_running()
    if service_running:
        print("   ✅ Ollama 服务已在运行")
    else:
        print("   启动服务中...")
        success = await deployment.start_service()
        if success:
            print("   ✅ Ollama 服务启动成功")
        else:
            print("   ❌ Ollama 服务启动失败")
    print()
    
    # 8. 获取端点
    print("8. 获取模型端点...")
    endpoint = deployment.get_endpoint(model_name)
    print(f"   端点: {endpoint}")
    print()
    
    # 9. 量化推荐
    print("9. 获取量化推荐...")
    gpu_memory_gb = 16.0  # Mi50 有 16GB HBM2
    recommended_quant = deployment.get_quantization_recommendation(gpu_memory_gb)
    print(f"   GPU 内存: {gpu_memory_gb} GB")
    print(f"   推荐量化: {recommended_quant.value}")
    print()
    
    # 10. 总结
    print("=" * 60)
    print("部署完成！")
    print("=" * 60)
    print()
    print("下一步:")
    print("1. 使用端点进行推理:")
    print(f"   curl {endpoint}/chat/completions \\")
    print('     -H "Content-Type: application/json" \\')
    print('     -d \'{"model": "' + model_name + '", "messages": [{"role": "user", "content": "Hello!"}]}\'')
    print()
    print("2. 或在 Python 中使用:")
    print("   from llm_compression.llm_client import LLMClient")
    print(f"   client = LLMClient(endpoint='{endpoint}')")
    print('   response = await client.generate("Hello, how are you?")')
    print()


if __name__ == "__main__":
    asyncio.run(main())
