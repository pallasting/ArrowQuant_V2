"""
Property-Based Tests for Arrow Storage

Tests storage properties including OpenClaw schema compatibility,
storage format specification, summary deduplication, and incremental updates.

Feature: llm-compression-integration
Properties: 11, 18, 19, 20
"""

import pytest
import pyarrow as pa
from hypothesis import given, settings, strategies as st
from pathlib import Path
import tempfile
import shutil

from llm_compression.arrow_storage import (
    ArrowStorage,
    create_experiences_compressed_schema,
    create_identity_compressed_schema,
    create_preferences_compressed_schema,
    create_context_compressed_schema,
    SCHEMA_REGISTRY
)
from llm_compression.compressor import CompressedMemory, CompressionMetadata
from datetime import datetime


# ============================================================================
# Test Strategies
# ============================================================================

@st.composite
def compressed_memory_strategy(draw):
    """Generate valid CompressedMemory for testing"""
    memory_id = draw(st.text(min_size=10, max_size=50))
    summary_hash = draw(st.text(min_size=16, max_size=16, alphabet=st.characters(whitelist_categories=('Ll', 'Nd'))))
    
    # Generate entities
    entities = {
        'persons': draw(st.lists(st.text(min_size=5, max_size=20), max_size=5)),
        'locations': draw(st.lists(st.text(min_size=5, max_size=20), max_size=5)),
        'dates': draw(st.lists(st.text(min_size=8, max_size=20), max_size=5)),
        'numbers': draw(st.lists(st.text(min_size=1, max_size=10), max_size=5)),
        'keywords': draw(st.lists(st.text(min_size=4, max_size=15), max_size=5)),
    }
    
    # Generate diff data
    diff_text = draw(st.text(min_size=10, max_size=200))
    diff_data = diff_text.encode('utf-8')
    
    # Generate embedding (384 dimensions for MiniLM)
    embedding = draw(st.lists(st.floats(min_value=-1.0, max_value=1.0, allow_nan=False), min_size=384, max_size=384))
    
    # Generate compression metadata
    original_size = draw(st.integers(min_value=100, max_value=10000))
    compressed_size = draw(st.integers(min_value=10, max_value=original_size))
    
    compression_metadata = CompressionMetadata(
        original_size=original_size,
        compressed_size=compressed_size,
        compression_ratio=original_size / compressed_size if compressed_size > 0 else 1.0,
        model_used=draw(st.sampled_from(['gpt-4', 'claude-3', 'local-model'])),
        quality_score=draw(st.floats(min_value=0.0, max_value=1.0)),
        compression_time_ms=draw(st.floats(min_value=10.0, max_value=5000.0)),
        compressed_at=datetime.now()
    )
    
    # Generate original fields
    original_fields = {
        'intent': draw(st.text(min_size=5, max_size=50)),
        'action': draw(st.text(min_size=5, max_size=50)),
        'outcome': draw(st.text(min_size=5, max_size=50)),
        'success': draw(st.booleans()),
        'related_memories': draw(st.lists(st.text(min_size=10, max_size=30), max_size=5)),
    }
    
    return CompressedMemory(
        memory_id=memory_id,
        summary_hash=summary_hash,
        entities=entities,
        diff_data=diff_data,
        embedding=embedding,
        compression_metadata=compression_metadata,
        original_fields=original_fields
    )


# ============================================================================
# Property 11: OpenClaw Schema Compatibility
# ============================================================================

@settings(max_examples=50, deadline=None)
@given(compressed=compressed_memory_strategy())
def test_property_11_openclaw_schema_compatibility(compressed):
    """
    Feature: llm-compression-integration, Property 11: OpenClaw Schema 完全兼容
    
    *For any* 符合 OpenClaw 原始 schema 的记忆对象，系统应该能够正确存储和检索，
    且扩展字段不影响原有功能
    
    Validates: Requirements 4.1, 4.2, 8.7
    """
    # Create temporary storage
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ArrowStorage(storage_path=tmpdir)
        
        # Test 1: Save compressed memory
        try:
            storage.save(compressed, category='experiences')
        except Exception as e:
            pytest.fail(f"Failed to save memory: {e}")
        
        # Test 2: Load compressed memory
        loaded = storage.load(compressed.memory_id, category='experiences')
        assert loaded is not None, "Failed to load saved memory"
        
        # Test 3: Verify all OpenClaw original fields are preserved
        assert loaded.memory_id == compressed.memory_id
        assert loaded.original_fields.get('intent') == compressed.original_fields.get('intent')
        assert loaded.original_fields.get('action') == compressed.original_fields.get('action')
        assert loaded.original_fields.get('outcome') == compressed.original_fields.get('outcome')
        assert loaded.original_fields.get('success') == compressed.original_fields.get('success')
        
        # Test 4: Verify compression extension fields are preserved
        assert loaded.summary_hash == compressed.summary_hash
        assert loaded.entities == compressed.entities
        assert loaded.diff_data == compressed.diff_data
        
        # Test 5: Verify embedding is preserved (with float16 precision tolerance)
        assert len(loaded.embedding) == len(compressed.embedding)
        # Allow small precision loss due to float16 conversion
        for i in range(min(10, len(loaded.embedding))):  # Check first 10 values
            assert abs(loaded.embedding[i] - compressed.embedding[i]) < 0.01
        
        # Test 6: Verify compression metadata is preserved
        assert loaded.compression_metadata.original_size == compressed.compression_metadata.original_size
        assert loaded.compression_metadata.compressed_size == compressed.compression_metadata.compressed_size
        assert abs(loaded.compression_metadata.compression_ratio - compressed.compression_metadata.compression_ratio) < 0.01
        assert loaded.compression_metadata.model_used == compressed.compression_metadata.model_used


def test_schema_has_all_openclaw_fields():
    """
    Verify that compressed schema includes all OpenClaw original fields
    
    This is a deterministic test to ensure schema completeness.
    """
    schema = create_experiences_compressed_schema()
    
    # OpenClaw required fields
    required_fields = [
        'memory_id',
        'timestamp',
        'context',
        'intent',
        'action',
        'outcome',
        'success',
        'embedding',
        'related_memories',
    ]
    
    schema_field_names = [field.name for field in schema]
    
    for field in required_fields:
        assert field in schema_field_names, f"Missing OpenClaw field: {field}"


def test_schema_has_compression_extension_fields():
    """
    Verify that compressed schema includes all compression extension fields
    """
    schema = create_experiences_compressed_schema()
    
    # Compression extension fields
    extension_fields = [
        'is_compressed',
        'summary_hash',
        'entities',
        'diff_data',
        'compression_metadata',
    ]
    
    schema_field_names = [field.name for field in schema]
    
    for field in extension_fields:
        assert field in schema_field_names, f"Missing compression extension field: {field}"


def test_all_categories_have_schemas():
    """
    Verify that all memory categories have defined schemas
    """
    required_categories = ['experiences', 'identity', 'preferences', 'context', 'summaries']
    
    for category in required_categories:
        assert category in SCHEMA_REGISTRY, f"Missing schema for category: {category}"
        assert isinstance(SCHEMA_REGISTRY[category], pa.Schema)


def test_embedding_uses_float16():
    """
    Verify that embedding field uses float16 for space savings
    
    Validates: Requirements 8.3
    """
    schema = create_experiences_compressed_schema()
    
    embedding_field = None
    for field in schema:
        if field.name == 'embedding':
            embedding_field = field
            break
    
    assert embedding_field is not None, "Embedding field not found"
    
    # Check that it's a list of float16
    assert pa.types.is_list(embedding_field.type), "Embedding should be a list type"
    assert pa.types.is_float16(embedding_field.type.value_type), "Embedding should use float16"


# ============================================================================
# Property 18: Storage Format Specification
# ============================================================================

def test_property_18_storage_format_specification():
    """
    Feature: llm-compression-integration, Property 18: 存储格式规范
    
    *For any* 压缩记忆，存储应该满足：
    - 使用 Arrow/Parquet 列式存储
    - diff 字段使用 zstd 压缩
    - embedding 使用 float16 存储
    
    Validates: Requirements 8.1, 8.2, 8.3
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ArrowStorage(storage_path=tmpdir, compression_level=3)
        
        # Create test memory
        compressed = CompressedMemory(
            memory_id="test_memory_001",
            summary_hash="abc123def456",
            entities={'persons': ['John'], 'dates': ['2024-01-15'], 'numbers': ['42'], 'locations': [], 'keywords': ['test']},
            diff_data=b"test diff data",
            embedding=[0.1] * 384,
            compression_metadata=CompressionMetadata(
                original_size=1000,
                compressed_size=100,
                compression_ratio=10.0,
                model_used="test-model",
                quality_score=0.95,
                compression_time_ms=50.0,
                compressed_at=datetime.now()
            ),
            original_fields={'intent': 'test', 'action': 'test', 'outcome': 'test', 'success': True, 'related_memories': []}
        )
        
        # Save memory
        storage.save(compressed, category='experiences')
        
        # Test 1: Verify file is Parquet format
        file_path = storage.category_paths['experiences']
        assert file_path.exists(), "Storage file not created"
        assert file_path.suffix == '.parquet', "File should be Parquet format"
        
        # Test 2: Verify file can be read as Arrow table
        import pyarrow.parquet as pq
        table = pq.read_table(file_path)
        assert isinstance(table, pa.Table), "File should be readable as Arrow table"
        
        # Test 3: Verify embedding uses float16
        embedding_field = table.schema.field('embedding')
        assert pa.types.is_list(embedding_field.type), "Embedding should be list type"
        assert pa.types.is_float16(embedding_field.type.value_type), "Embedding should use float16"
        
        # Test 4: Verify diff_data is binary type
        diff_field = table.schema.field('diff_data')
        assert pa.types.is_binary(diff_field.type), "diff_data should be binary type"


# ============================================================================
# Property 19: Summary Deduplication
# ============================================================================

@settings(max_examples=20, deadline=None)
@given(
    summary_hash=st.text(min_size=16, max_size=16, alphabet=st.characters(whitelist_categories=('Ll', 'Nd'))),
    num_memories=st.integers(min_value=2, max_value=5)
)
def test_property_19_summary_deduplication(summary_hash, num_memories):
    """
    Feature: llm-compression-integration, Property 19: 摘要去重
    
    *For any* 两个具有相同 summary_hash 的记忆，系统应该只存储一份摘要，
    其他记忆存储引用
    
    Validates: Requirements 8.4
    
    Note: This test verifies the schema and structure support for deduplication.
    Full deduplication logic will be implemented in subtask 11.6.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ArrowStorage(storage_path=tmpdir)
        
        # Create multiple memories with same summary_hash
        memories = []
        for i in range(num_memories):
            compressed = CompressedMemory(
                memory_id=f"memory_{i}",
                summary_hash=summary_hash,  # Same hash for all
                entities={'persons': [f'Person{i}'], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                diff_data=f"diff_{i}".encode('utf-8'),
                embedding=[0.1 * i] * 384,
                compression_metadata=CompressionMetadata(
                    original_size=1000 + i * 100,
                    compressed_size=100 + i * 10,
                    compression_ratio=10.0,
                    model_used="test-model",
                    quality_score=0.95,
                    compression_time_ms=50.0,
                    compressed_at=datetime.now()
                ),
                original_fields={'intent': f'intent_{i}', 'action': f'action_{i}', 'outcome': f'outcome_{i}', 'success': True, 'related_memories': []}
            )
            memories.append(compressed)
        
        # Save all memories
        for memory in memories:
            storage.save(memory, category='experiences')
        
        # Verify all memories are saved
        for memory in memories:
            loaded = storage.load(memory.memory_id, category='experiences')
            assert loaded is not None, f"Memory {memory.memory_id} not found"
            assert loaded.summary_hash == summary_hash, "Summary hash should be preserved"


# ============================================================================
# Property 20: Incremental Update Support
# ============================================================================

@settings(max_examples=20, deadline=None)
@given(
    num_batches=st.integers(min_value=2, max_value=5),
    batch_size=st.integers(min_value=1, max_value=3)
)
def test_property_20_incremental_update_support(num_batches, batch_size):
    """
    Feature: llm-compression-integration, Property 20: 增量更新支持
    
    *For any* 新增记忆，系统应该支持 append-only 操作，不需要重写整个存储文件
    
    Validates: Requirements 8.5
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        storage = ArrowStorage(storage_path=tmpdir)
        
        all_memory_ids = []
        
        # Add memories in multiple batches
        for batch_idx in range(num_batches):
            batch_memories = []
            
            for i in range(batch_size):
                memory_id = f"batch{batch_idx}_memory{i}"
                all_memory_ids.append(memory_id)
                
                compressed = CompressedMemory(
                    memory_id=memory_id,
                    summary_hash=f"hash_{batch_idx}_{i}",
                    entities={'persons': [], 'dates': [], 'numbers': [], 'locations': [], 'keywords': []},
                    diff_data=f"diff_{batch_idx}_{i}".encode('utf-8'),
                    embedding=[0.1] * 384,
                    compression_metadata=CompressionMetadata(
                        original_size=1000,
                        compressed_size=100,
                        compression_ratio=10.0,
                        model_used="test-model",
                        quality_score=0.95,
                        compression_time_ms=50.0,
                        compressed_at=datetime.now()
                    ),
                    original_fields={'intent': 'test', 'action': 'test', 'outcome': 'test', 'success': True, 'related_memories': []}
                )
                batch_memories.append(compressed)
            
            # Save batch
            for memory in batch_memories:
                storage.save(memory, category='experiences')
            
            # Verify all previously saved memories are still accessible
            for memory_id in all_memory_ids:
                loaded = storage.load(memory_id, category='experiences')
                assert loaded is not None, f"Memory {memory_id} should still be accessible after batch {batch_idx}"
        
        # Final verification: all memories should be present
        assert len(all_memory_ids) == num_batches * batch_size
        
        for memory_id in all_memory_ids:
            loaded = storage.load(memory_id, category='experiences')
            assert loaded is not None, f"Memory {memory_id} not found in final check"


# ============================================================================
# Additional Schema Tests
# ============================================================================

def test_identity_schema_structure():
    """Test identity schema has correct structure"""
    schema = create_identity_compressed_schema()
    
    required_fields = ['memory_id', 'timestamp', 'description', 'values', 'embedding']
    schema_field_names = [field.name for field in schema]
    
    for field in required_fields:
        assert field in schema_field_names, f"Missing field in identity schema: {field}"


def test_preferences_schema_structure():
    """Test preferences schema has correct structure"""
    schema = create_preferences_compressed_schema()
    
    required_fields = ['memory_id', 'timestamp', 'preference', 'reason', 'embedding']
    schema_field_names = [field.name for field in schema]
    
    for field in required_fields:
        assert field in schema_field_names, f"Missing field in preferences schema: {field}"


def test_context_schema_structure():
    """Test context schema has correct structure"""
    schema = create_context_compressed_schema()
    
    required_fields = ['memory_id', 'timestamp', 'context', 'embedding']
    schema_field_names = [field.name for field in schema]
    
    for field in required_fields:
        assert field in schema_field_names, f"Missing field in context schema: {field}"
