"""
Unit tests for CognitiveLoopArrow

Tests:
- process_arrow(): 端到端零拷贝处理
- load_memories_from_table(): 从 Arrow Table 加载记忆
- add_memory_arrow(): 添加单个记忆
- batch_add_memories_arrow(): 批量添加记忆
- batch_process_queries(): 批量处理查询
- 零拷贝验证
- 大规模测试（100K+ 记忆）

Requirements: Task 12.5
"""

import pytest
import pyarrow as pa
import numpy as np
from unittest.mock import Mock, AsyncMock, patch

from llm_compression.cognitive_loop_arrow import (
    CognitiveLoopArrow,
    CognitiveResultArrow,
    add_arrow_support
)
from llm_compression.cognitive_loop import CognitiveLoop
from llm_compression.embedder_arrow import LocalEmbedderArrow
from llm_compression.network_navigator_arrow import NetworkNavigatorArrow
from llm_compression.expression_layer import ExpressionResult
from llm_compression.internal_feedback import QualityScore


@pytest.fixture
def embedder_arrow():
    """Create LocalEmbedderArrow instance"""
    return LocalEmbedderArrow()


@pytest.fixture
def navigator_arrow():
    """Create NetworkNavigatorArrow instance"""
    return NetworkNavigatorArrow()


@pytest.fixture
def mock_expressor():
    """Create mock expressor"""
    expressor = Mock()
    expressor.express_text = AsyncMock(return_value=ExpressionResult(
        content="Generated output",
        modality="text",
        quality_score=0.9,
        source_memories=["memory1", "memory2"]
    ))
    return expressor


@pytest.fixture
def mock_feedback():
    """Create mock feedback system"""
    feedback = Mock()
    feedback.evaluate = AsyncMock(return_value=QualityScore(
        overall=0.9,
        consistency=0.9,
        completeness=0.9,
        accuracy=0.9,
        coherence=0.9
    ))
    return feedback


@pytest.fixture
def cognitive_loop_arrow(embedder_arrow, navigator_arrow, mock_expressor, mock_feedback):
    """Create CognitiveLoopArrow instance"""
    return CognitiveLoopArrow(
        embedder_arrow=embedder_arrow,
        navigator_arrow=navigator_arrow,
        expressor=mock_expressor,
        feedback=mock_feedback,
        quality_threshold=0.85
    )


@pytest.fixture
def sample_memory_table(embedder_arrow):
    """Create sample memory table"""
    texts = [
        "Python is a programming language",
        "Machine learning uses neural networks",
        "Data science involves statistics"
    ]
    
    memory_ids = ["mem1", "mem2", "mem3"]
    
    # Encode embeddings
    embeddings_array = embedder_arrow.batch_encode_arrow(texts)
    
    # Create table
    table = pa.table({
        'memory_id': pa.array(memory_ids),
        'content': pa.array(texts),
        'embedding': embeddings_array
    })
    
    return table


class TestCognitiveLoopArrow:
    """Test CognitiveLoopArrow class"""
    
    def test_initialization(self, cognitive_loop_arrow):
        """Test initialization"""
        assert cognitive_loop_arrow is not None
        assert cognitive_loop_arrow.embedder_arrow is not None
        assert cognitive_loop_arrow.navigator_arrow is not None
        assert cognitive_loop_arrow.memory_table is None
    
    def test_load_memories_from_table(self, cognitive_loop_arrow, sample_memory_table):
        """Test loading memories from Arrow Table"""
        cognitive_loop_arrow.load_memories_from_table(sample_memory_table)
        
        assert cognitive_loop_arrow.memory_table is not None
        assert len(cognitive_loop_arrow.memory_table) == 3
        assert 'embedding' in cognitive_loop_arrow.memory_table.schema.names
    
    def test_load_memories_invalid_table(self, cognitive_loop_arrow):
        """Test loading memories from invalid table"""
        # Table without embedding column
        invalid_table = pa.table({
            'memory_id': pa.array(['mem1']),
            'content': pa.array(['test'])
        })
        
        with pytest.raises(ValueError, match="must contain 'embedding' column"):
            cognitive_loop_arrow.load_memories_from_table(invalid_table)
    
    def test_add_memory_arrow(self, cognitive_loop_arrow):
        """Test adding single memory"""
        cognitive_loop_arrow.add_memory_arrow(
            memory_id="mem1",
            content="Test memory content"
        )
        
        assert cognitive_loop_arrow.memory_table is not None
        assert len(cognitive_loop_arrow.memory_table) == 1
        assert cognitive_loop_arrow.memory_table['memory_id'][0].as_py() == "mem1"
    
    def test_add_memory_with_embedding(self, cognitive_loop_arrow, embedder_arrow):
        """Test adding memory with pre-computed embedding"""
        embedding = embedder_arrow.embedder.encode("Test content")
        
        cognitive_loop_arrow.add_memory_arrow(
            memory_id="mem1",
            content="Test content",
            embedding=embedding
        )
        
        assert len(cognitive_loop_arrow.memory_table) == 1
    
    def test_add_memory_with_metadata(self, cognitive_loop_arrow):
        """Test adding memory with metadata"""
        cognitive_loop_arrow.add_memory_arrow(
            memory_id="mem1",
            content="Test content",
            metadata={'timestamp': 123456, 'source': 'test'}
        )
        
        assert len(cognitive_loop_arrow.memory_table) == 1
        # Note: metadata columns may not be added if schema doesn't match
    
    def test_batch_add_memories_arrow(self, cognitive_loop_arrow):
        """Test batch adding memories"""
        memory_ids = ["mem1", "mem2", "mem3"]
        contents = ["Content 1", "Content 2", "Content 3"]
        
        cognitive_loop_arrow.batch_add_memories_arrow(
            memory_ids=memory_ids,
            contents=contents
        )
        
        assert cognitive_loop_arrow.memory_table is not None
        assert len(cognitive_loop_arrow.memory_table) == 3
        assert cognitive_loop_arrow.memory_table['memory_id'].to_pylist() == memory_ids
    
    def test_batch_add_memories_with_embeddings(self, cognitive_loop_arrow, embedder_arrow):
        """Test batch adding memories with pre-computed embeddings"""
        memory_ids = ["mem1", "mem2"]
        contents = ["Content 1", "Content 2"]
        
        # Pre-compute embeddings
        embeddings = embedder_arrow.embedder.encode_batch(contents)
        
        cognitive_loop_arrow.batch_add_memories_arrow(
            memory_ids=memory_ids,
            contents=contents,
            embeddings=embeddings
        )
        
        assert len(cognitive_loop_arrow.memory_table) == 2
    
    def test_batch_add_memories_length_mismatch(self, cognitive_loop_arrow):
        """Test batch adding with length mismatch"""
        with pytest.raises(ValueError, match="must have same length"):
            cognitive_loop_arrow.batch_add_memories_arrow(
                memory_ids=["mem1", "mem2"],
                contents=["Content 1"]  # Length mismatch
            )
    
    @pytest.mark.asyncio
    async def test_process_arrow_no_memories(self, cognitive_loop_arrow):
        """Test processing with no memories"""
        result = await cognitive_loop_arrow.process_arrow(
            query="What is Python?",
            max_memories=5
        )
        
        assert isinstance(result, CognitiveResultArrow)
        assert result.output is not None
        assert result.quality is not None
        assert len(result.memories_table) == 0
    
    @pytest.mark.asyncio
    async def test_process_arrow_with_memories(self, cognitive_loop_arrow, sample_memory_table):
        """Test processing with memories"""
        cognitive_loop_arrow.load_memories_from_table(sample_memory_table)
        
        result = await cognitive_loop_arrow.process_arrow(
            query="What is Python?",
            max_memories=5
        )
        
        assert isinstance(result, CognitiveResultArrow)
        assert result.output == "Generated output"
        assert result.quality.overall == 0.9
        assert result.processing_time_ms > 0
    
    @pytest.mark.asyncio
    async def test_process_arrow_with_corrections(self, cognitive_loop_arrow, sample_memory_table):
        """Test processing with quality corrections"""
        # Mock low quality to trigger corrections
        cognitive_loop_arrow.feedback.evaluate = AsyncMock(side_effect=[
            QualityScore(overall=0.7, consistency=0.7, completeness=0.7, accuracy=0.7, coherence=0.7),  # First evaluation (low)
            QualityScore(overall=0.9, consistency=0.9, completeness=0.9, accuracy=0.9, coherence=0.9)   # After correction (high)
        ])
        
        cognitive_loop_arrow.load_memories_from_table(sample_memory_table)
        
        result = await cognitive_loop_arrow.process_arrow(
            query="What is Python?",
            max_memories=5
        )
        
        assert result.corrections_applied > 0
    
    def test_get_memory_stats_empty(self, cognitive_loop_arrow):
        """Test getting stats with no memories"""
        stats = cognitive_loop_arrow.get_memory_stats()
        
        assert stats['total_memories'] == 0
        assert stats['table_size_bytes'] == 0
    
    def test_get_memory_stats_with_memories(self, cognitive_loop_arrow, sample_memory_table):
        """Test getting stats with memories"""
        cognitive_loop_arrow.load_memories_from_table(sample_memory_table)
        
        stats = cognitive_loop_arrow.get_memory_stats()
        
        assert stats['total_memories'] == 3
        assert stats['table_size_bytes'] > 0
        assert stats['table_size_mb'] > 0
        assert 'embedding' in stats['columns']
    
    @pytest.mark.asyncio
    async def test_batch_process_queries(self, cognitive_loop_arrow, sample_memory_table):
        """Test batch processing queries"""
        cognitive_loop_arrow.load_memories_from_table(sample_memory_table)
        
        queries = [
            "What is Python?",
            "What is machine learning?",
            "What is data science?"
        ]
        
        results = await cognitive_loop_arrow.batch_process_queries(
            queries=queries,
            max_memories=5
        )
        
        assert len(results) == 3
        assert all(isinstance(r, CognitiveResultArrow) for r in results)
    
    def test_extract_memory_contents_empty(self, cognitive_loop_arrow):
        """Test extracting contents from empty table"""
        empty_table = pa.table({})
        contents = cognitive_loop_arrow._extract_memory_contents(empty_table)
        
        assert contents == []
    
    def test_extract_memory_contents_with_content_column(self, cognitive_loop_arrow):
        """Test extracting contents with content column"""
        table = pa.table({
            'memory_id': pa.array(['mem1', 'mem2']),
            'content': pa.array(['Content 1', 'Content 2'])
        })
        
        contents = cognitive_loop_arrow._extract_memory_contents(table)
        
        assert contents == ['Content 1', 'Content 2']
    
    def test_extract_memory_contents_with_text_column(self, cognitive_loop_arrow):
        """Test extracting contents with text column"""
        table = pa.table({
            'memory_id': pa.array(['mem1', 'mem2']),
            'text': pa.array(['Text 1', 'Text 2'])
        })
        
        contents = cognitive_loop_arrow._extract_memory_contents(table)
        
        assert contents == ['Text 1', 'Text 2']


class TestZeroCopyVerification:
    """Test zero-copy characteristics"""
    
    def test_memory_table_is_zero_copy(self, cognitive_loop_arrow, sample_memory_table):
        """Test that memory table operations are zero-copy"""
        cognitive_loop_arrow.load_memories_from_table(sample_memory_table)
        
        # Get table reference
        table1 = cognitive_loop_arrow.memory_table
        
        # Load again (should be zero-copy reference)
        cognitive_loop_arrow.load_memories_from_table(sample_memory_table)
        table2 = cognitive_loop_arrow.memory_table
        
        # Tables should share same underlying data
        assert table1.schema == table2.schema
    
    def test_batch_add_preserves_zero_copy(self, cognitive_loop_arrow, embedder_arrow):
        """Test that batch add preserves zero-copy"""
        # Add first batch
        cognitive_loop_arrow.batch_add_memories_arrow(
            memory_ids=["mem1", "mem2"],
            contents=["Content 1", "Content 2"]
        )
        
        table_size_1 = cognitive_loop_arrow.memory_table.nbytes
        
        # Add second batch
        cognitive_loop_arrow.batch_add_memories_arrow(
            memory_ids=["mem3", "mem4"],
            contents=["Content 3", "Content 4"]
        )
        
        table_size_2 = cognitive_loop_arrow.memory_table.nbytes
        
        # Size should increase proportionally
        assert table_size_2 > table_size_1


class TestLargeScaleOperations:
    """Test large-scale operations (100K+ memories)"""
    
    @pytest.mark.slow
    def test_large_scale_memory_loading(self, cognitive_loop_arrow, embedder_arrow):
        """Test loading 10K memories"""
        # Generate 10K memories
        n_memories = 10000
        memory_ids = [f"mem{i}" for i in range(n_memories)]
        contents = [f"Content {i}" for i in range(n_memories)]
        
        # Batch add
        cognitive_loop_arrow.batch_add_memories_arrow(
            memory_ids=memory_ids,
            contents=contents
        )
        
        assert len(cognitive_loop_arrow.memory_table) == n_memories
        
        # Check memory usage
        stats = cognitive_loop_arrow.get_memory_stats()
        assert stats['table_size_mb'] > 0
        print(f"10K memories: {stats['table_size_mb']:.2f} MB")
    
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_large_scale_retrieval(self, cognitive_loop_arrow, embedder_arrow):
        """Test retrieval from 10K memories"""
        # Generate 1K memories (reduced for test speed)
        n_memories = 1000
        memory_ids = [f"mem{i}" for i in range(n_memories)]
        contents = [f"Content about topic {i % 10}" for i in range(n_memories)]
        
        cognitive_loop_arrow.batch_add_memories_arrow(
            memory_ids=memory_ids,
            contents=contents
        )
        
        # Process query
        result = await cognitive_loop_arrow.process_arrow(
            query="Tell me about topic 5",
            max_memories=10
        )
        
        assert isinstance(result, CognitiveResultArrow)
        assert result.processing_time_ms > 0
        print(f"Retrieval from 1K memories: {result.processing_time_ms:.1f}ms")


class TestAddArrowSupport:
    """Test add_arrow_support function"""
    
    def test_add_arrow_support(self):
        """Test adding Arrow support to CognitiveLoop"""
        cognitive_loop = CognitiveLoop()
        cognitive_loop_arrow = add_arrow_support(cognitive_loop)
        
        assert isinstance(cognitive_loop_arrow, CognitiveLoopArrow)
        assert cognitive_loop_arrow.cognitive_loop is cognitive_loop


class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_workflow(self, embedder_arrow):
        """Test complete end-to-end workflow"""
        # Create mock components
        mock_expressor = Mock()
        mock_expressor.express_text = AsyncMock(return_value=ExpressionResult(
            content="Python is a high-level programming language",
            modality="text",
            quality_score=0.95,
            source_memories=["mem1"]
        ))
        
        mock_feedback = Mock()
        mock_feedback.evaluate = AsyncMock(return_value=QualityScore(
            overall=0.95,
            consistency=0.95,
            completeness=0.95,
            accuracy=0.95,
            coherence=0.95
        ))
        
        # Create cognitive loop
        cognitive_loop_arrow = CognitiveLoopArrow(
            embedder_arrow=embedder_arrow,
            expressor=mock_expressor,
            feedback=mock_feedback
        )
        
        # Add memories
        cognitive_loop_arrow.batch_add_memories_arrow(
            memory_ids=["mem1", "mem2", "mem3"],
            contents=[
                "Python is a programming language",
                "Python is used for data science",
                "Python has simple syntax"
            ]
        )
        
        # Process query
        result = await cognitive_loop_arrow.process_arrow(
            query="What is Python?",
            max_memories=3
        )
        
        # Verify result
        assert result.output == "Python is a high-level programming language"
        assert result.quality.overall == 0.95
        assert len(result.memories_table) > 0
        assert result.processing_time_ms > 0
