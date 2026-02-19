"""
LLM 客户端使用示例

演示如何使用 LLM 客户端进行文本生成
"""

import asyncio
from llm_compression import LLMClient, Config


async def basic_example():
    """基础使用示例"""
    print("=== 基础使用示例 ===\n")
    
    # 创建客户端
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0,
        max_retries=3,
        rate_limit=60
    )
    
    try:
        # 单个请求
        print("发送单个请求...")
        response = await client.generate(
            prompt="Summarize the following text: The quick brown fox jumps over the lazy dog.",
            max_tokens=50,
            temperature=0.3
        )
        
        print(f"响应: {response.text}")
        print(f"使用 tokens: {response.tokens_used}")
        print(f"延迟: {response.latency_ms:.2f}ms")
        print(f"模型: {response.model}")
        print()
        
    finally:
        await client.close()


async def batch_example():
    """批量请求示例"""
    print("=== 批量请求示例 ===\n")
    
    # 创建客户端
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0,
        max_retries=3,
        rate_limit=60
    )
    
    try:
        # 批量请求
        prompts = [
            "Summarize: Text about AI",
            "Summarize: Text about machine learning",
            "Summarize: Text about deep learning"
        ]
        
        print(f"发送 {len(prompts)} 个批量请求...")
        responses = await client.batch_generate(
            prompts=prompts,
            max_tokens=50,
            temperature=0.3
        )
        
        for i, response in enumerate(responses):
            print(f"\n请求 {i+1}:")
            print(f"  响应: {response.text}")
            print(f"  Tokens: {response.tokens_used}")
            print(f"  延迟: {response.latency_ms:.2f}ms")
        
        print()
        
    finally:
        await client.close()


async def metrics_example():
    """指标跟踪示例"""
    print("=== 指标跟踪示例 ===\n")
    
    # 创建客户端
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=30.0,
        max_retries=3,
        rate_limit=60
    )
    
    try:
        # 执行多个请求
        print("执行多个请求...")
        for i in range(5):
            await client.generate(
                prompt=f"Test request {i}",
                max_tokens=50
            )
        
        # 获取指标
        metrics = client.get_metrics()
        
        print("\n指标统计:")
        print(f"  总请求数: {metrics['total_requests']}")
        print(f"  成功请求数: {metrics['successful_requests']}")
        print(f"  失败请求数: {metrics['failed_requests']}")
        print(f"  成功率: {metrics['success_rate']:.2%}")
        print(f"  总 tokens: {metrics['total_tokens']}")
        print(f"  平均 tokens/请求: {metrics['avg_tokens_per_request']:.2f}")
        print(f"  平均延迟: {metrics['avg_latency_ms']:.2f}ms")
        print(f"  最近延迟: {[f'{l:.2f}ms' for l in metrics['recent_latencies']]}")
        print()
        
    finally:
        await client.close()


async def config_example():
    """使用配置文件示例"""
    print("=== 使用配置文件示例 ===\n")
    
    # 从配置文件加载
    config = Config.from_yaml("config.yaml")
    
    # 创建客户端
    client = LLMClient(
        endpoint=config.llm.cloud_endpoint,
        api_key=config.llm.cloud_api_key,
        timeout=config.llm.timeout,
        max_retries=config.llm.max_retries,
        rate_limit=config.llm.rate_limit
    )
    
    try:
        print(f"使用端点: {config.llm.cloud_endpoint}")
        print(f"超时: {config.llm.timeout}s")
        print(f"最大重试: {config.llm.max_retries}")
        print(f"速率限制: {config.llm.rate_limit} 请求/分钟")
        print()
        
        # 发送请求
        response = await client.generate(
            prompt="Hello, world!",
            max_tokens=50
        )
        
        print(f"响应: {response.text}")
        print()
        
    finally:
        await client.close()


async def error_handling_example():
    """错误处理示例"""
    print("=== 错误处理示例 ===\n")
    
    from llm_compression.llm_client import LLMAPIError, LLMTimeoutError
    
    # 创建客户端（使用较短的超时）
    client = LLMClient(
        endpoint="http://localhost:8045",
        timeout=5.0,
        max_retries=2
    )
    
    try:
        # 尝试发送请求
        print("尝试发送请求...")
        response = await client.generate(
            prompt="Test prompt",
            max_tokens=50
        )
        print(f"成功: {response.text}")
        
    except LLMTimeoutError as e:
        print(f"超时错误: {e}")
        
    except LLMAPIError as e:
        print(f"API 错误: {e}")
        
    except Exception as e:
        print(f"未知错误: {e}")
        
    finally:
        await client.close()
        print()


async def main():
    """主函数"""
    print("LLM 客户端使用示例\n")
    print("=" * 50)
    print()
    
    # 注意：这些示例需要实际的 LLM API 服务器运行在 localhost:8045
    # 如果没有服务器，请使用 mock 或跳过这些示例
    
    print("提示: 这些示例需要 LLM API 服务器运行在 http://localhost:8045")
    print("如果没有服务器，示例将失败。")
    print()
    
    try:
        # 运行示例
        # await basic_example()
        # await batch_example()
        # await metrics_example()
        # await config_example()
        # await error_handling_example()
        
        print("示例已准备就绪。")
        print("取消注释上面的函数调用以运行特定示例。")
        
    except Exception as e:
        print(f"示例执行失败: {e}")
        print("请确保 LLM API 服务器正在运行。")


if __name__ == "__main__":
    asyncio.run(main())
