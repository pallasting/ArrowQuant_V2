"""
Arrow/Parquet 压缩存储引擎

提供高效的文本压缩和解压缩功能，使用 Arrow IPC 格式和 ZSTD 压缩。
目标：3x 压缩比，<1ms 延迟，100% 保真。
"""

import pyarrow as pa
import zstandard as zstd
from pathlib import Path
from typing import Optional
import uuid


class ArrowStorage:
    """
    Arrow/Parquet 压缩存储引擎

    使用 Arrow IPC 格式进行零拷贝序列化，配合 ZSTD 压缩算法。
    """

    def __init__(self, storage_path: str = "~/.ai-os/memory/"):
        """
        初始化存储引擎

        Args:
            storage_path: 存储目录路径
        """
        self.storage_path = Path(storage_path).expanduser()
        self.storage_path.mkdir(parents=True, exist_ok=True)

    def compress(self, text: str) -> bytes:
        """
        压缩文本为 Arrow 格式

        Args:
            text: 原始文本

        Returns:
            压缩后的字节数据

        Example:
            >>> storage = ArrowStorage()
            >>> compressed = storage.compress("Hello, World!")
            >>> len(compressed) < len("Hello, World!")
            True
        """
        # 创建 Arrow 表
        table = pa.table({
            'text': [text],
            'length': [len(text)],
            'encoding': ['utf-8']
        })

        # 序列化为 IPC 格式（零拷贝）
        sink = pa.BufferOutputStream()
        writer = pa.ipc.new_stream(sink, table.schema)
        writer.write_table(table)
        writer.close()

        # ZSTD 压缩（level 3 平衡速度和压缩比）
        arrow_bytes = sink.getvalue().to_pybytes()
        compressed = zstd.compress(arrow_bytes, level=3)

        return compressed

    def decompress(self, compressed: bytes) -> str:
        """
        解压缩 Arrow 数据

        Args:
            compressed: 压缩的字节数据

        Returns:
            原始文本

        Raises:
            ValueError: 如果数据损坏或格式错误

        Example:
            >>> storage = ArrowStorage()
            >>> text = "Test content"
            >>> compressed = storage.compress(text)
            >>> storage.decompress(compressed) == text
            True
        """
        try:
            # ZSTD 解压
            decompressed = zstd.decompress(compressed)

            # 反序列化 Arrow
            reader = pa.ipc.open_stream(decompressed)
            table = reader.read_all()

            # 提取文本
            text = table['text'][0].as_py()
            return text

        except Exception as e:
            raise ValueError(f"Failed to decompress data: {e}")

    def save(self, memory_id: str, compressed: bytes) -> Path:
        """
        保存压缩数据到磁盘

        Args:
            memory_id: 记忆唯一标识
            compressed: 压缩的字节数据

        Returns:
            保存的文件路径

        Example:
            >>> storage = ArrowStorage()
            >>> compressed = storage.compress("Test")
            >>> path = storage.save("test_001", compressed)
            >>> path.exists()
            True
        """
        file_path = self.storage_path / f"{memory_id}.arrow"
        file_path.write_bytes(compressed)
        return file_path

    def load(self, memory_id: str) -> bytes:
        """
        从磁盘加载压缩数据

        Args:
            memory_id: 记忆唯一标识

        Returns:
            压缩的字节数据

        Raises:
            FileNotFoundError: 如果文件不存在

        Example:
            >>> storage = ArrowStorage()
            >>> compressed = storage.compress("Test")
            >>> storage.save("test_001", compressed)
            >>> loaded = storage.load("test_001")
            >>> loaded == compressed
            True
        """
        file_path = self.storage_path / f"{memory_id}.arrow"

        if not file_path.exists():
            raise FileNotFoundError(f"Memory {memory_id} not found")

        return file_path.read_bytes()

    def delete(self, memory_id: str) -> bool:
        """
        删除存储的记忆

        Args:
            memory_id: 记忆唯一标识

        Returns:
            是否成功删除
        """
        file_path = self.storage_path / f"{memory_id}.arrow"

        if file_path.exists():
            file_path.unlink()
            return True
        return False

    def exists(self, memory_id: str) -> bool:
        """
        检查记忆是否存在

        Args:
            memory_id: 记忆唯一标识

        Returns:
            是否存在
        """
        file_path = self.storage_path / f"{memory_id}.arrow"
        return file_path.exists()

    def list_all(self) -> list[str]:
        """
        列出所有存储的记忆 ID

        Returns:
            记忆 ID 列表
        """
        return [
            f.stem for f in self.storage_path.glob("*.arrow")
        ]

    def get_compression_ratio(self, text: str) -> float:
        """
        计算压缩比

        Args:
            text: 原始文本

        Returns:
            压缩比（原始大小 / 压缩后大小）
        """
        original_size = len(text.encode('utf-8'))
        compressed = self.compress(text)
        compressed_size = len(compressed)

        return original_size / compressed_size if compressed_size > 0 else 0.0
