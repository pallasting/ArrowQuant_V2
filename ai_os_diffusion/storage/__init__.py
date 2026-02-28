"""
Storage module for AI-OS Diffusion.

Provides Arrow-based storage with vector search capabilities.
Will use Rust backend for 10-50x performance improvement.
"""

from ai_os_diffusion.storage.arrow_storage import ArrowStorage, StorageError

__all__ = [
    "ArrowStorage",
    "StorageError",
]
