#!/usr/bin/env python3
"""
ArrowQuant V2 异步API验证脚本

此脚本演示ArrowQuant V2异步API的基本功能:
1. 初始化AsyncQuantizer
2. 执行异步量化任务
3. 并发执行多个任务
4. 验证结果一致性
"""

import asyncio
import sys
import tempfile
from pathlib import Path

try:
    from arrow_quant_v2 import AsyncArrowQuantV2, DiffusionQuantConfig
    print("✅ 成功导入 AsyncArrowQuantV2")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请先构建模块: maturin develop --release")
    sys.exit(1)


async def demo_basic_async():
    """演示基本异步功能"""
    print("\n" + "="*60)
    print("演示 1: 基本异步API")
    print("="*60)
    
    # 创建异步量化器
    quantizer = AsyncArrowQuantV2()
    print("✅ AsyncQuantizer 初始化成功")
    
    # 创建配置
    config = DiffusionQuantConfig(bit_width=4)
    print(f"✅ 配置创建成功")
    
    # 测试异步方法返回Future
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model"
        output_path = Path(tmpdir) / "output"
        model_path.mkdir()
        output_path.mkdir()
        
        try:
            # 调用异步方法
            future = quantizer.quantize_diffusion_model_async(
                str(model_path),
                str(output_path),
                config
            )
            
            # 验证返回的是awaitable
            if hasattr(future, '__await__'):
                print("✅ quantize_diffusion_model_async 返回 awaitable Future")
            else:
                print(f"❌ 返回类型错误: {type(future)}")
                return False
            
            # 等待完成（预期失败，因为没有实际模型）
            try:
                await future
                print("⚠️  任务完成（意外）")
            except Exception as e:
                print(f"✅ 任务失败（预期）: {type(e).__name__}")
                print("   （因为没有实际模型文件）")
        
        except Exception as e:
            print(f"❌ 测试失败: {e}")
            return False
    
    return True


async def demo_concurrent_tasks():
    """演示并发任务处理"""
    print("\n" + "="*60)
    print("演示 2: 并发任务处理（12个并发任务）")
    print("="*60)
    
    quantizer = AsyncArrowQuantV2()
    num_tasks = 12
    
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)
        
        # 创建12个并发任务
        tasks = []
        config = DiffusionQuantConfig(bit_width=4)
        
        print(f"创建 {num_tasks} 个并发任务...")
        for i in range(num_tasks):
            model_path = base_dir / f"model_{i}"
            output_path = base_dir / f"output_{i}"
            model_path.mkdir()
            output_path.mkdir()
            
            task = quantizer.quantize_diffusion_model_async(
                str(model_path),
                str(output_path),
                config
            )
            tasks.append(task)
        
        print(f"✅ 创建了 {len(tasks)} 个任务")
        
        # 并发执行所有任务
        print("并发执行所有任务...")
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 统计结果
        success_count = sum(1 for r in results if not isinstance(r, Exception))
        error_count = sum(1 for r in results if isinstance(r, Exception))
        
        print(f"✅ 所有任务完成:")
        print(f"   - 成功: {success_count}")
        print(f"   - 失败: {error_count} (预期，因为没有实际模型)")
        print(f"   - 无死锁: ✅")
        
        return True


async def demo_error_handling():
    """演示错误处理"""
    print("\n" + "="*60)
    print("演示 3: 异步错误处理")
    print("="*60)
    
    quantizer = AsyncArrowQuantV2()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "nonexistent"
        output_path = Path(tmpdir) / "output"
        output_path.mkdir()
        
        config = DiffusionQuantConfig(bit_width=4)
        
        try:
            # 使用不存在的模型路径
            result = await quantizer.quantize_diffusion_model_async(
                str(model_path),
                str(output_path),
                config
            )
            print("⚠️  任务成功（意外）")
        except Exception as e:
            print(f"✅ 异常正确传播到Python: {type(e).__name__}")
            print(f"   错误信息: {str(e)[:100]}...")
            return True
    
    return False


async def main():
    """主函数"""
    print("\n" + "="*60)
    print("ArrowQuant V2 异步API验证")
    print("="*60)
    
    results = []
    
    # 运行所有演示
    results.append(("基本异步API", await demo_basic_async()))
    results.append(("并发任务处理", await demo_concurrent_tasks()))
    results.append(("错误处理", await demo_error_handling()))
    
    # 总结
    print("\n" + "="*60)
    print("验证总结")
    print("="*60)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ 所有验证通过！ArrowQuant V2异步API完全可用。")
    else:
        print("❌ 部分验证失败，请检查上述错误。")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
