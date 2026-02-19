"""
Arrow 压缩存储引擎测试

测试 ArrowStorage 的压缩、解压、持久化等功能。
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from llm_compression.storage import ArrowStorage


@pytest.fixture
def temp_storage():
    """创建临时存储目录"""
    temp_dir = tempfile.mkdtemp()
    storage = ArrowStorage(storage_path=temp_dir)
    yield storage
    # 清理
    shutil.rmtree(temp_dir)


class TestArrowCompression:
    """测试压缩和解压功能"""

    def test_compress_decompress_short_text(self, temp_storage):
        """测试短文本压缩"""
        text = "Hello, World!"
        compressed = temp_storage.compress(text)
        decompressed = temp_storage.decompress(compressed)

        assert decompressed == text
        assert isinstance(compressed, bytes)

    def test_compress_decompress_long_text(self, temp_storage):
        """测试长文本压缩"""
        text = "This is a test. " * 1000  # 16,000 字符
        compressed = temp_storage.compress(text)
        decompressed = temp_storage.decompress(compressed)

        assert decompressed == text

    def test_compress_decompress_unicode(self, temp_storage):
        """测试 Unicode 文本压缩"""
        text = "你好世界！🌍 Hello World! مرحبا بالعالم"
        compressed = temp_storage.compress(text)
        decompressed = temp_storage.decompress(compressed)

        assert decompressed == text

    def test_compress_empty_string(self, temp_storage):
        """测试空字符串压缩"""
        text = ""
        compressed = temp_storage.compress(text)
        decompressed = temp_storage.decompress(compressed)

        assert decompressed == text

    def test_compression_ratio(self, temp_storage):
        """测试压缩比"""
        # 重复文本应该有很好的压缩比
        text = "Python is a programming language. " * 100

        ratio = temp_storage.get_compression_ratio(text)

        # 验证压缩比 > 2.5x
        assert ratio > 2.5, f"Compression ratio {ratio:.2f}x is below target 2.5x"

    def test_compression_ratio_random_text(self, temp_storage):
        """测试随机文本的压缩比"""
        # 随机文本压缩比较低，可能小于 1.0（因为熵高）
        import random
        import string
        text = ''.join(random.choices(string.ascii_letters + string.digits, k=1000))

        ratio = temp_storage.get_compression_ratio(text)

        # 随机文本压缩比可能小于 1.0，只要不是异常值即可
        assert 0.5 < ratio < 2.0

    def test_decompress_invalid_data(self, temp_storage):
        """测试解压无效数据"""
        invalid_data = b"invalid compressed data"

        with pytest.raises(ValueError):
            temp_storage.decompress(invalid_data)


class TestArrowPersistence:
    """测试持久化功能"""

    def test_save_and_load(self, temp_storage):
        """测试保存和加载"""
        memory_id = "test_001"
        text = "Test content for persistence"

        # 压缩并保存
        compressed = temp_storage.compress(text)
        path = temp_storage.save(memory_id, compressed)

        # 验证文件存在
        assert path.exists()

        # 加载并解压
        loaded = temp_storage.load(memory_id)
        decompressed = temp_storage.decompress(loaded)

        assert decompressed == text

    def test_load_nonexistent(self, temp_storage):
        """测试加载不存在的记忆"""
        with pytest.raises(FileNotFoundError):
            temp_storage.load("nonexistent_id")

    def test_exists(self, temp_storage):
        """测试存在性检查"""
        memory_id = "test_002"

        # 初始不存在
        assert not temp_storage.exists(memory_id)

        # 保存后存在
        compressed = temp_storage.compress("Test")
        temp_storage.save(memory_id, compressed)
        assert temp_storage.exists(memory_id)

    def test_delete(self, temp_storage):
        """测试删除功能"""
        memory_id = "test_003"

        # 保存
        compressed = temp_storage.compress("Test")
        temp_storage.save(memory_id, compressed)
        assert temp_storage.exists(memory_id)

        # 删除
        result = temp_storage.delete(memory_id)
        assert result is True
        assert not temp_storage.exists(memory_id)

        # 删除不存在的
        result = temp_storage.delete(memory_id)
        assert result is False

    def test_list_all(self, temp_storage):
        """测试列出所有记忆"""
        # 初始为空
        assert len(temp_storage.list_all()) == 0

        # 添加多个记忆
        memory_ids = ["mem_001", "mem_002", "mem_003"]
        for memory_id in memory_ids:
            compressed = temp_storage.compress(f"Content {memory_id}")
            temp_storage.save(memory_id, compressed)

        # 验证列表
        all_ids = temp_storage.list_all()
        assert len(all_ids) == 3
        assert set(all_ids) == set(memory_ids)


class TestArrowPerformance:
    """测试性能指标"""

    def test_compression_speed(self, temp_storage):
        """测试压缩速度"""
        import time

        text = "Performance test content. " * 100  # ~2.5KB

        # 测试 10 次取平均
        times = []
        for _ in range(10):
            start = time.perf_counter()
            temp_storage.compress(text)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # 转换为毫秒

        avg_time = sum(times) / len(times)

        # 验证 < 1ms
        assert avg_time < 1.0, f"Compression took {avg_time:.2f}ms, target is <1ms"

    def test_decompression_speed(self, temp_storage):
        """测试解压速度"""
        import time

        text = "Performance test content. " * 100
        compressed = temp_storage.compress(text)

        # 测试 10 次取平均
        times = []
        for _ in range(10):
            start = time.perf_counter()
            temp_storage.decompress(compressed)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        avg_time = sum(times) / len(times)

        # 验证 < 1ms
        assert avg_time < 1.0, f"Decompression took {avg_time:.2f}ms, target is <1ms"

    def test_roundtrip_speed(self, temp_storage):
        """测试完整往返速度"""
        import time

        text = "Roundtrip test content. " * 100

        start = time.perf_counter()
        compressed = temp_storage.compress(text)
        decompressed = temp_storage.decompress(compressed)
        end = time.perf_counter()

        roundtrip_time = (end - start) * 1000

        # 验证往返 < 2ms
        assert roundtrip_time < 2.0, f"Roundtrip took {roundtrip_time:.2f}ms"
        assert decompressed == text


class TestArrowEdgeCases:
    """测试边界情况"""

    def test_very_long_text(self, temp_storage):
        """测试超长文本"""
        # 1MB 文本
        text = "A" * (1024 * 1024)

        compressed = temp_storage.compress(text)
        decompressed = temp_storage.decompress(compressed)

        assert decompressed == text

    def test_special_characters(self, temp_storage):
        """测试特殊字符"""
        text = "Special chars: \n\t\r\0 \"'\\/"

        compressed = temp_storage.compress(text)
        decompressed = temp_storage.decompress(compressed)

        assert decompressed == text

    def test_multiple_saves_same_id(self, temp_storage):
        """测试覆盖保存"""
        memory_id = "test_overwrite"

        # 第一次保存
        text1 = "First version"
        compressed1 = temp_storage.compress(text1)
        temp_storage.save(memory_id, compressed1)

        # 第二次保存（覆盖）
        text2 = "Second version"
        compressed2 = temp_storage.compress(text2)
        temp_storage.save(memory_id, compressed2)

        # 验证加载的是第二个版本
        loaded = temp_storage.load(memory_id)
        decompressed = temp_storage.decompress(loaded)

        assert decompressed == text2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
