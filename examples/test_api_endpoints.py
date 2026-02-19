#!/usr/bin/env python3
"""
API 端点测试脚本

测试所有可用的 API 端点，找出哪些可以用于 Chat Agent
"""

import asyncio
import aiohttp
import time


# API 配置
API_KEY = "sk-0437c02b1560470981866f50b05759e3"
BASE_URL = "http://192.168.1.99:8046"

# 测试端点配置
ENDPOINTS = {
    "Gemini CLI OAuth (Claude)": {
        "url": f"{BASE_URL}/gemini-cli-oauth/v1/messages",
        "protocol": "claude",
        "models": ["gemini-3-pro-preview", "gemini-3-flash-preview"]
    },
    "Gemini CLI OAuth (OpenAI)": {
        "url": f"{BASE_URL}/gemini-cli-oauth/v1/chat/completions",
        "protocol": "openai",
        "models": ["gemini-3-flash-preview", "gemini-3-pro-preview"]
    },
    "Gemini Antigravity (Claude)": {
        "url": f"{BASE_URL}/gemini-antigravity/v1/messages",
        "protocol": "claude",
        "models": ["gemini-claude-opus-4-6-thinking", "gemini-3-pro-preview"]
    },
    "Gemini Antigravity (OpenAI)": {
        "url": f"{BASE_URL}/gemini-antigravity/v1/chat/completions",
        "protocol": "openai",
        "models": ["gemini-3-pro-preview"]
    },
    "Claude Kiro OAuth (Claude)": {
        "url": f"{BASE_URL}/claude-kiro-oauth/v1/messages",
        "protocol": "claude",
        "models": ["claude-opus-4-6", "claude-opus-4-5"]
    },
    "Claude Kiro OAuth (OpenAI)": {
        "url": f"{BASE_URL}/claude-kiro-oauth/v1/chat/completions",
        "protocol": "openai",
        "models": ["claude-opus-4-5"]
    },
    "Qwen OAuth (Claude)": {
        "url": f"{BASE_URL}/openai-qwen-oauth/v1/messages",
        "protocol": "claude",
        "models": ["qwen3-coder-plus"]
    },
    "Qwen OAuth (OpenAI)": {
        "url": f"{BASE_URL}/openai-qwen-oauth/v1/chat/completions",
        "protocol": "openai",
        "models": ["qwen3-coder-plus"]
    },
    "iFlow OAuth (Claude)": {
        "url": f"{BASE_URL}/openai-iflow/v1/messages",
        "protocol": "claude",
        "models": ["minimax-m2.5"]
    },
    "iFlow OAuth (OpenAI)": {
        "url": f"{BASE_URL}/openai-iflow/v1/chat/completions",
        "protocol": "openai",
        "models": ["kimi-k2.5", "minimax-m2.5"]
    }
}


async def test_claude_endpoint(session, name, config, model):
    """测试 Claude 协议端点"""
    url = config["url"]
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": API_KEY
    }
    data = {
        "model": model,
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "Hello! Please respond with 'OK'."}
        ]
    }
    
    try:
        start = time.time()
        async with session.post(url, json=data, headers=headers, timeout=30) as response:
            elapsed = time.time() - start
            
            if response.status == 200:
                result = await response.json()
                content = result.get("content", [{}])[0].get("text", "")
                return {
                    "status": "✅ 成功",
                    "model": model,
                    "response": content[:50],
                    "time": f"{elapsed:.2f}s",
                    "endpoint": name
                }
            else:
                error_text = await response.text()
                return {
                    "status": "❌ 失败",
                    "model": model,
                    "error": f"HTTP {response.status}: {error_text[:100]}",
                    "endpoint": name
                }
    except asyncio.TimeoutError:
        return {
            "status": "⏱️ 超时",
            "model": model,
            "error": "请求超时（30s）",
            "endpoint": name
        }
    except Exception as e:
        return {
            "status": "❌ 错误",
            "model": model,
            "error": str(e)[:100],
            "endpoint": name
        }


async def test_openai_endpoint(session, name, config, model):
    """测试 OpenAI 协议端点"""
    url = config["url"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    data = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Hello! Please respond with 'OK'."}
        ],
        "max_tokens": 100
    }
    
    try:
        start = time.time()
        async with session.post(url, json=data, headers=headers, timeout=30) as response:
            elapsed = time.time() - start
            
            if response.status == 200:
                result = await response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                return {
                    "status": "✅ 成功",
                    "model": model,
                    "response": content[:50],
                    "time": f"{elapsed:.2f}s",
                    "endpoint": name
                }
            else:
                error_text = await response.text()
                return {
                    "status": "❌ 失败",
                    "model": model,
                    "error": f"HTTP {response.status}: {error_text[:100]}",
                    "endpoint": name
                }
    except asyncio.TimeoutError:
        return {
            "status": "⏱️ 超时",
            "model": model,
            "error": "请求超时（30s）",
            "endpoint": name
        }
    except Exception as e:
        return {
            "status": "❌ 错误",
            "model": model,
            "error": str(e)[:100],
            "endpoint": name
        }


async def test_all_endpoints():
    """测试所有端点"""
    print("\n" + "="*70)
    print("API 端点测试")
    print("="*70)
    print(f"\n基础 URL: {BASE_URL}")
    print(f"API Key: {API_KEY[:20]}...")
    print(f"\n测试 {len(ENDPOINTS)} 个端点...\n")
    
    results = []
    
    async with aiohttp.ClientSession() as session:
        tasks = []
        
        for name, config in ENDPOINTS.items():
            protocol = config["protocol"]
            models = config["models"]
            
            # 只测试每个端点的第一个模型
            model = models[0]
            
            if protocol == "claude":
                task = test_claude_endpoint(session, name, config, model)
            else:
                task = test_openai_endpoint(session, name, config, model)
            
            tasks.append(task)
        
        # 并发测试所有端点
        results = await asyncio.gather(*tasks)
    
    # 打印结果
    print("\n" + "="*70)
    print("测试结果")
    print("="*70)
    
    successful = []
    failed = []
    
    for result in results:
        status = result["status"]
        endpoint = result["endpoint"]
        model = result["model"]
        
        if "成功" in status:
            successful.append(result)
            print(f"\n{status} {endpoint}")
            print(f"  模型: {model}")
            print(f"  响应: {result.get('response', 'N/A')}")
            print(f"  时间: {result.get('time', 'N/A')}")
        else:
            failed.append(result)
            print(f"\n{status} {endpoint}")
            print(f"  模型: {model}")
            print(f"  错误: {result.get('error', 'N/A')}")
    
    # 总结
    print("\n" + "="*70)
    print("总结")
    print("="*70)
    print(f"\n✅ 成功: {len(successful)}/{len(results)}")
    print(f"❌ 失败: {len(failed)}/{len(results)}")
    
    if successful:
        print("\n推荐使用的端点:")
        for i, result in enumerate(successful[:3], 1):
            print(f"\n{i}. {result['endpoint']}")
            print(f"   模型: {result['model']}")
            print(f"   响应时间: {result.get('time', 'N/A')}")
            
            # 提取协议和 URL
            endpoint_name = result['endpoint']
            config = ENDPOINTS[endpoint_name]
            protocol = config['protocol']
            url = config['url']
            
            print(f"   协议: {protocol}")
            print(f"   URL: {url}")
    
    return successful, failed


async def main():
    """主函数"""
    try:
        successful, failed = await test_all_endpoints()
        
        if successful:
            print("\n" + "="*70)
            print("配置建议")
            print("="*70)
            
            # 选择最快的端点
            best = min(successful, key=lambda x: float(x.get('time', '999').rstrip('s')))
            
            print(f"\n推荐使用: {best['endpoint']}")
            print(f"模型: {best['model']}")
            
            # 提取配置
            config = ENDPOINTS[best['endpoint']]
            protocol = config['protocol']
            url = config['url']
            
            print(f"\n更新 test_chat_agent.py 配置:")
            print(f"```python")
            if protocol == "openai":
                print(f"llm_client = LLMClient(")
                print(f"    endpoint=\"{url}\",")
                print(f"    api_key=\"{API_KEY}\",")
                print(f"    api_type=\"openai\",")
                print(f"    timeout=30.0")
                print(f")")
            else:
                print(f"llm_client = LLMClient(")
                print(f"    endpoint=\"{url}\",")
                print(f"    api_key=\"{API_KEY}\",")
                print(f"    api_type=\"claude\",")
                print(f"    timeout=30.0")
                print(f")")
            print(f"```")
            
            return 0
        else:
            print("\n❌ 没有可用的端点")
            return 1
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
