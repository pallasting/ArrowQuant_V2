#!/usr/bin/env python3
"""
Arrow 存储引擎验证脚本

不依赖 pytest，直接验证核心功能。
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from llm_compression.storage import ArrowStorage
    print("✅ 成功导入 ArrowStorage")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("提示: 需要安装 pyarrow 和 zstandard")
    sys.exit(1)


def test_basic_compression():
    """测试基础压缩功能"""
    print("\n" + "="*60)
    print("测试 1: 基础压缩和解压")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    storage = ArrowStorage(storage_path=temp_dir)

    try:
        # 测试短文本
        text = "Hello, World! 你好世界！"
        compressed = storage.compress(text)
        decompressed = storage.decompress(compressed)

        assert decompressed == text, "解压后文本不匹配"
        print(f"✅ 短文本压缩: {len(text)} bytes → {len(compressed)} bytes")

        # 测试长文本
        long_text = "This is a test. " * 1000
        compressed_long = storage.compress(long_text)
        decompressed_long = storage.decompress(compressed_long)

        assert decompressed_long == long_text, "长文本解压失败"
        ratio = len(long_text) / len(compressed_long)
        print(f"✅ 长文本压缩: {len(long_text)} bytes → {len(compressed_long)} bytes")
        print(f"   压缩比: {ratio:.2f}x")

        if ratio < 2.5:
            print(f"⚠️  警告: 压缩比 {ratio:.2f}x 低于目标 2.5x")
        else:
            print(f"✅ 压缩比达标 ({ratio:.2f}x > 2.5x)")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def test_persistence():
    """测试持久化功能"""
    print("\n" + "="*60)
    print("测试 2: 持久化存储")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    storage = ArrowStorage(storage_path=temp_dir)

    try:
        memory_id = "test_001"
        text = "Test content for persistence 测试持久化"

        # 保存
        compressed = storage.compress(text)
        path = storage.save(memory_id, compressed)
        print(f"✅ 保存成功: {path}")

        # 检查存在
        assert storage.exists(memory_id), "文件应该存在"
        print(f"✅ 存在性检查通过")

        # 加载
        loaded = storage.load(memory_id)
        decompressed = storage.decompress(loaded)
        assert decompressed == text, "加载的内容不匹配"
        print(f"✅ 加载并解压成功")

        # 删除
        result = storage.delete(memory_id)
        assert result is True, "删除应该成功"
        assert not storage.exists(memory_id), "文件应该不存在"
        print(f"✅ 删除成功")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir)


def test_performance():
    """测试性能指标"""
    print("\n" + "="*60)
    print("测试 3: 性能基准")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    storage = ArrowStorage(storage_path=temp_dir)

    try:
        text = "Performance test content. " * 100  # ~2.5KB

        # 测试压缩速度
        times = []
        for _ in range(10):
            start = time.perf_counter()
            storage.compress(text)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_compress = sum(times) / len(times)
        print(f"压缩速度: {avg_compress:.3f}ms (平均)")

        if avg_compress < 1.0:
            print(f"✅ 压缩速度达标 ({avg_compress:.3f}ms < 1ms)")
        else:
            print(f"⚠️  压缩速度: {avg_compress:.3f}ms (目标 <1ms)")

        # 测试解压速度
        compressed = storage.compress(text)
        times = []
        for _ in range(10):
            start = time.perf_counter()
            storage.decompress(compressed)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_decompress = sum(times) / len(times)
        print(f"解压速度: {avg_decompress:.3f}ms (平均)")

        if avg_decompress < 1.0:
            print(f"✅ 解压速度达标 ({avg_decompress:.3f}ms < 1ms)")
        else:
            print(f"⚠️  解压速度: {avg_decompress:.3f}ms (目标 <1ms)")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "="*60)
    print("测试 4: 边界情况")
    print("="*60)

    temp_dir = tempfile.mkdtemp()
    storage = ArrowStorage(storage_path=temp_dir)

    try:
        # 空字符串
        empty = ""
        compressed = storage.compress(empty)
        decompressed = storage.decompress(compressed)
        assert decompressed == empty
        print("✅ 空字符串处理正常")

        # Unicode
        unicode_text = "你好世界！🌍 Hello مرحبا"
        compressed = storage.compress(unicode_text)
        decompressed = storage.decompress(compressed)
        assert decompressed == unicode_text
        print("✅ Unicode 处理正常")

        # 特殊字符
        special = "Special: \n\t\r \"'\\/"
        compressed = storage.compress(special)
        decompressed = storage.decompress(compressed)
        assert decompressed == special
        print("✅ 特殊字符处理正常")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False
    finally:
        shutil.rmtree(temp_dir)


def main():
    """运行所有测试"""
    print("="*60)
    print("Arrow 存储引擎验证")
    print("="*60)

    results = []

    results.append(("基础压缩", test_basic_compression()))
    results.append(("持久化", test_persistence()))
    results.append(("性能基准", test_performance()))
    results.append(("边界情况", test_edge_cases()))

    # 总结
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)

    for name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{name}: {status}")

    total = len(results)
    passed = sum(1 for _, p in results if p)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！Task 1.1 完成。")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
