"""
文件导入器 - 将文档转换为记忆
"""
import asyncio
from pathlib import Path
from typing import List, Optional
from llm_compression.compressor import LLMCompressor


class FileImporter:
    """将文件内容导入为记忆"""
    
    def __init__(self, compressor: LLMCompressor):
        self.compressor = compressor
    
    async def import_file(
        self,
        file_path: str,
        chunk_size: int = 1000,
        overlap: int = 200
    ) -> List:
        """
        导入单个文件
        
        Args:
            file_path: 文件路径
            chunk_size: 分块大小（字符）
            overlap: 重叠大小（保持上下文连续性）
            
        Returns:
            CompressedMemory 对象列表
        """
        import time
        total_start = time.time()
        
        print(f"  [1/4] 检查文件...", flush=True)
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        print(f"  [2/4] 读取文件...", flush=True)
        # 读取文件
        content = path.read_text(encoding='utf-8')
        print(f"        文件大小: {len(content)} 字符", flush=True)
        
        print(f"  [3/4] 分块处理...", end='', flush=True)
        # 分块
        chunks = self._chunk_text(content, chunk_size, overlap)
        print(f" 完成", flush=True)
        
        print(f"        分块数量: {len(chunks)} 块", flush=True)
        print(f"        预计时间: ~{len(chunks) * 1.6:.0f} 秒", flush=True)
        
        print(f"  [4/4] 压缩存储...", flush=True)
        # 压缩每个块
        compressed_memories = []
        start_time = time.time()
        
        for i, chunk in enumerate(chunks):
            chunk_start = time.time()
            
            # 添加元数据
            annotated = f"[Source: {path.name}, Part {i+1}/{len(chunks)}]\n{chunk}"
            
            # 压缩
            compressed = await self.compressor.compress(annotated)
            compressed_memories.append(compressed)
            
            # 显示进度
            elapsed = time.time() - chunk_start
            total_elapsed = time.time() - start_time
            avg_time = total_elapsed / (i + 1)
            remaining = avg_time * (len(chunks) - i - 1)
            
            print(f"        [{i+1}/{len(chunks)}] {elapsed:.1f}s | "
                  f"总计 {total_elapsed:.0f}s | "
                  f"剩余 ~{remaining:.0f}s", flush=True)
        
        total_time = time.time() - total_start
        print(f"  ✓ 完成！总用时 {total_time:.1f}s (平均 {total_time/len(chunks):.1f}s/块)", flush=True)
        
        return compressed_memories
    
    def _chunk_text(
        self,
        text: str,
        chunk_size: int,
        overlap: int
    ) -> List[str]:
        """
        简单分块（固定大小，快速）
        
        Args:
            text: 原始文本
            chunk_size: 目标块大小
            overlap: 重叠大小
            
        Returns:
            文本块列表
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if chunk:  # 只添加非空块
                chunks.append(chunk)
            start = end - overlap if end < len(text) else len(text)
        
        return chunks
    
    async def import_directory(
        self,
        dir_path: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = False
    ) -> dict:
        """
        导入目录中的所有文件
        
        Args:
            dir_path: 目录路径
            extensions: 文件扩展名列表（如 ['.txt', '.md']）
            recursive: 是否递归子目录
            
        Returns:
            {文件名: [记忆ID列表]}
        """
        if extensions is None:
            extensions = ['.txt', '.md', '.py', '.json', '.yaml']
        
        path = Path(dir_path)
        if not path.is_dir():
            raise NotADirectoryError(f"Not a directory: {dir_path}")
        
        # 查找文件
        pattern = '**/*' if recursive else '*'
        files = [
            f for f in path.glob(pattern)
            if f.is_file() and f.suffix in extensions
        ]
        
        # 导入每个文件
        results = {}
        for file in files:
            print(f"\nImporting: {file.name}")
            try:
                memory_ids = await self.import_file(str(file))
                results[file.name] = memory_ids
            except Exception as e:
                print(f"  Error: {e}")
                results[file.name] = []
        
        return results
