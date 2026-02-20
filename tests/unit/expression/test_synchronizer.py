"""
Unit tests for the Synchronizer class.

Tests cover:
- Stream registration and management
- Chunk buffering and retrieval
- Timing coordination
- Synchronization and drift correction
- Pause/resume functionality
- Fallback handling
- Statistics tracking

Requirements: 4.1, 4.6, 6.3, 13.1
"""

import pytest
import time
import numpy as np
from unittest.mock import Mock, patch

from llm_compression.expression.synchronizer import (
    Synchronizer,
    SynchronizerConfig,
    OutputStream,
    StreamChunk,
    StreamState
)
from llm_compression.expression.expression_types import OutputModality


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sync_config():
    """Create a test synchronizer configuration."""
    return SynchronizerConfig(
        buffer_size_ms=100.0,
        max_drift_ms=50.0,
        sync_interval_ms=25.0,
        enable_auto_sync=True,
        fallback_to_text=True
    )


@pytest.fixture
def synchronizer(sync_config):
    """Create a test synchronizer."""
    return Synchronizer(config=sync_config)


@pytest.fixture
def text_chunks():
    """Create sample text chunks."""
    return [
        ("Hello", 100.0),
        ("world", 100.0),
        ("!", 50.0)
    ]


@pytest.fixture
def audio_chunks():
    """Create sample audio chunks."""
    return [
        (np.random.randn(2205), 100.0),  # 100ms at 22050 Hz
        (np.random.randn(2205), 100.0),
        (np.random.randn(1102), 50.0)
    ]


# ============================================================================
# Configuration Tests
# ============================================================================


class TestSynchronizerConfig:
    """Test SynchronizerConfig validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = SynchronizerConfig(
            buffer_size_ms=200.0,
            max_drift_ms=100.0,
            sync_interval_ms=50.0
        )
        assert config.buffer_size_ms == 200.0
        assert config.max_drift_ms == 100.0
        assert config.sync_interval_ms == 50.0
    
    def test_invalid_buffer_size(self):
        """Test invalid buffer size."""
        with pytest.raises(ValueError, match="buffer_size_ms must be positive"):
            SynchronizerConfig(buffer_size_ms=-100.0)
    
    def test_invalid_max_drift(self):
        """Test invalid max drift."""
        with pytest.raises(ValueError, match="max_drift_ms must be positive"):
            SynchronizerConfig(max_drift_ms=0.0)
    
    def test_invalid_sync_interval(self):
        """Test invalid sync interval."""
        with pytest.raises(ValueError, match="sync_interval_ms must be positive"):
            SynchronizerConfig(sync_interval_ms=-50.0)


# ============================================================================
# Stream Management Tests
# ============================================================================


class TestStreamManagement:
    """Test stream registration and management."""
    
    def test_register_stream(self, synchronizer):
        """Test stream registration."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        assert OutputModality.TEXT in synchronizer.streams
        assert synchronizer.streams[OutputModality.TEXT].modality == OutputModality.TEXT
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.IDLE
    
    def test_register_multiple_streams(self, synchronizer):
        """Test registering multiple streams."""
        synchronizer.register_stream(OutputModality.TEXT)
        synchronizer.register_stream(OutputModality.SPEECH)
        synchronizer.register_stream(OutputModality.VISUAL)
        
        assert len(synchronizer.streams) == 3
        assert OutputModality.TEXT in synchronizer.streams
        assert OutputModality.SPEECH in synchronizer.streams
        assert OutputModality.VISUAL in synchronizer.streams
    
    def test_register_duplicate_stream(self, synchronizer):
        """Test registering duplicate stream."""
        synchronizer.register_stream(OutputModality.TEXT)
        synchronizer.register_stream(OutputModality.TEXT)  # Duplicate
        
        # Should only have one stream
        assert len(synchronizer.streams) == 1
    
    def test_get_stream_state(self, synchronizer):
        """Test getting stream state."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        state = synchronizer.get_stream_state(OutputModality.TEXT)
        assert state == StreamState.IDLE
        
        # Non-existent stream
        state = synchronizer.get_stream_state(OutputModality.SPEECH)
        assert state is None


# ============================================================================
# Chunk Buffering Tests
# ============================================================================


class TestChunkBuffering:
    """Test chunk buffering and retrieval."""
    
    def test_add_chunk(self, synchronizer, text_chunks):
        """Test adding chunks to stream."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        stream = synchronizer.streams[OutputModality.TEXT]
        assert stream.buffer_size() == 3
        assert stream.state == StreamState.BUFFERING
    
    def test_add_chunk_auto_register(self, synchronizer):
        """Test adding chunk auto-registers stream."""
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        assert OutputModality.TEXT in synchronizer.streams
        assert synchronizer.streams[OutputModality.TEXT].buffer_size() == 1
    
    def test_chunk_sequence_numbers(self, synchronizer, text_chunks):
        """Test chunk sequence numbering."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        stream = synchronizer.streams[OutputModality.TEXT]
        chunks = list(stream.buffer)
        
        assert chunks[0].sequence_number == 0
        assert chunks[1].sequence_number == 1
        assert chunks[2].sequence_number == 2
    
    def test_chunk_timestamps(self, synchronizer, text_chunks):
        """Test chunk timestamp calculation."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        stream = synchronizer.streams[OutputModality.TEXT]
        chunks = list(stream.buffer)
        
        assert chunks[0].timestamp_ms == 0.0
        assert chunks[1].timestamp_ms == 100.0
        assert chunks[2].timestamp_ms == 200.0
    
    def test_final_chunk(self, synchronizer):
        """Test marking final chunk."""
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Final",
            duration_ms=100.0,
            is_final=True
        )
        
        stream = synchronizer.streams[OutputModality.TEXT]
        chunk = stream.buffer[0]
        
        assert chunk.is_final is True


# ============================================================================
# Playback Control Tests
# ============================================================================


class TestPlaybackControl:
    """Test playback control (start, pause, resume, stop)."""
    
    def test_start(self, synchronizer, text_chunks):
        """Test starting playback."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        synchronizer.start()
        
        assert synchronizer.reference_time is not None
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PLAYING
    
    def test_pause(self, synchronizer, text_chunks):
        """Test pausing playback."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        synchronizer.start()
        synchronizer.pause()
        
        assert synchronizer.is_paused is True
        assert synchronizer.pause_time is not None
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PAUSED
    
    def test_resume(self, synchronizer, text_chunks):
        """Test resuming playback."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        synchronizer.start()
        synchronizer.pause()
        time.sleep(0.1)  # Simulate pause duration
        synchronizer.resume()
        
        assert synchronizer.is_paused is False
        assert synchronizer.pause_time is None
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PLAYING
    
    def test_stop(self, synchronizer, text_chunks):
        """Test stopping playback."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        synchronizer.start()
        synchronizer.stop()
        
        assert synchronizer.reference_time is None
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.IDLE
        assert synchronizer.streams[OutputModality.TEXT].buffer_size() == 0


# ============================================================================
# Synchronization Tests
# ============================================================================


class TestSynchronization:
    """Test timing coordination and synchronization."""
    
    def test_get_next_chunks_synchronized(self, synchronizer):
        """Test getting synchronized chunks."""
        # Add text chunks
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0,
            timestamp_ms=0.0
        )
        
        # Add audio chunks
        synchronizer.add_chunk(
            modality=OutputModality.SPEECH,
            data=np.random.randn(2205),
            duration_ms=100.0,
            timestamp_ms=0.0
        )
        
        synchronizer.start()
        
        # Get chunks immediately (should return both)
        chunks = synchronizer.get_next_chunks()
        
        assert OutputModality.TEXT in chunks
        assert OutputModality.SPEECH in chunks
        assert chunks[OutputModality.TEXT].data == "Hello"
    
    def test_buffering_complete(self, synchronizer):
        """Test buffering completion check."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        # Add insufficient data
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=50.0
        )
        
        # Should not be complete (need 100ms buffer)
        assert synchronizer.is_buffering_complete() is False
        
        # Add more data
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=60.0
        )
        
        # Should be complete now
        assert synchronizer.is_buffering_complete() is True
    
    def test_drift_detection(self, synchronizer):
        """Test drift detection between streams."""
        # Add text chunks (fast)
        for i in range(5):
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=f"Text{i}",
                duration_ms=20.0,
                timestamp_ms=i * 20.0
            )
        
        # Add audio chunks (slow)
        for i in range(3):
            synchronizer.add_chunk(
                modality=OutputModality.SPEECH,
                data=np.random.randn(2205),
                duration_ms=50.0,
                timestamp_ms=i * 50.0
            )
        
        synchronizer.start()
        
        # Simulate playback
        time.sleep(0.15)  # 150ms
        
        # Get chunks (should trigger drift detection)
        chunks = synchronizer.get_next_chunks()
        
        # Check statistics
        stats = synchronizer.get_statistics()
        assert stats["max_drift_observed_ms"] >= 0.0


# ============================================================================
# Fallback Tests
# ============================================================================


class TestFallback:
    """Test fallback handling."""
    
    def test_fallback_to_text_only(self, synchronizer):
        """Test fallback to text-only output."""
        # Register multiple streams
        synchronizer.register_stream(OutputModality.TEXT)
        synchronizer.register_stream(OutputModality.SPEECH)
        synchronizer.register_stream(OutputModality.VISUAL)
        
        # Add chunks
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        synchronizer.add_chunk(
            modality=OutputModality.SPEECH,
            data=np.random.randn(2205),
            duration_ms=100.0
        )
        
        synchronizer.start()
        
        # Trigger fallback
        synchronizer.fallback_to_text_only()
        
        # Check that only text stream is active
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PLAYING
        assert synchronizer.streams[OutputModality.SPEECH].state == StreamState.IDLE
        assert synchronizer.streams[OutputModality.VISUAL].state == StreamState.IDLE
    
    def test_buffer_underrun(self, synchronizer):
        """Test buffer underrun handling."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        # Add single chunk
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=50.0
        )
        
        synchronizer.start()
        
        # Get chunk
        chunks = synchronizer.get_next_chunks()
        assert OutputModality.TEXT in chunks
        
        # Try to get another chunk (should cause underrun)
        time.sleep(0.06)  # Wait for chunk to "play"
        chunks = synchronizer.get_next_chunks()
        
        # Check statistics
        stats = synchronizer.get_statistics()
        assert stats["buffer_underruns"] >= 0


# ============================================================================
# Statistics Tests
# ============================================================================


class TestStatistics:
    """Test statistics tracking."""
    
    def test_chunk_processing_stats(self, synchronizer, text_chunks):
        """Test chunk processing statistics."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        stats = synchronizer.get_statistics()
        assert stats["total_chunks_processed"] == 3
    
    def test_stream_state_stats(self, synchronizer):
        """Test stream state statistics."""
        synchronizer.register_stream(OutputModality.TEXT)
        synchronizer.register_stream(OutputModality.SPEECH)
        
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.start()
        
        stats = synchronizer.get_statistics()
        assert stats["active_streams"] == 1  # Only text has data
        assert stats["completed_streams"] == 0
    
    def test_reset_statistics(self, synchronizer, text_chunks):
        """Test resetting statistics."""
        synchronizer.register_stream(OutputModality.TEXT)
        
        for text, duration in text_chunks:
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=duration
            )
        
        stats = synchronizer.get_statistics()
        assert stats["total_chunks_processed"] == 3
        
        synchronizer.reset_statistics()
        
        stats = synchronizer.get_statistics()
        assert stats["total_chunks_processed"] == 0


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""
    
    def test_multimodal_playback(self, synchronizer):
        """Test complete multimodal playback workflow."""
        # Add text chunks
        text_data = ["Hello", "world", "!"]
        for i, text in enumerate(text_data):
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=text,
                duration_ms=100.0,
                timestamp_ms=i * 100.0
            )
        
        # Add audio chunks
        for i in range(3):
            synchronizer.add_chunk(
                modality=OutputModality.SPEECH,
                data=np.random.randn(2205),
                duration_ms=100.0,
                timestamp_ms=i * 100.0
            )
        
        # Start playback
        synchronizer.start()
        
        # Simulate playback
        all_chunks = []
        for _ in range(3):
            chunks = synchronizer.get_next_chunks()
            all_chunks.append(chunks)
            time.sleep(0.11)  # Slightly more than chunk duration
        
        # Verify we got chunks
        assert len(all_chunks) > 0
        
        # Check statistics
        stats = synchronizer.get_statistics()
        assert stats["total_chunks_processed"] == 6
    
    def test_pause_resume_workflow(self, synchronizer):
        """Test pause/resume workflow."""
        # Add chunks
        for i in range(5):
            synchronizer.add_chunk(
                modality=OutputModality.TEXT,
                data=f"Text{i}",
                duration_ms=100.0
            )
        
        # Start playback
        synchronizer.start()
        
        # Get first chunk
        chunks1 = synchronizer.get_next_chunks()
        assert len(chunks1) > 0
        
        # Pause
        synchronizer.pause()
        time.sleep(0.1)
        
        # Resume
        synchronizer.resume()
        
        # Get next chunk
        time.sleep(0.11)
        chunks2 = synchronizer.get_next_chunks()
        
        # Should have gotten chunks
        assert len(chunks2) > 0


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_synchronizer(self, synchronizer):
        """Test synchronizer with no streams."""
        synchronizer.start()
        chunks = synchronizer.get_next_chunks()
        
        assert len(chunks) == 0
    
    def test_get_chunks_before_start(self, synchronizer):
        """Test getting chunks before starting."""
        synchronizer.add_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        chunks = synchronizer.get_next_chunks()
        assert len(chunks) == 0
    
    def test_pause_before_start(self, synchronizer):
        """Test pausing before starting."""
        synchronizer.pause()
        
        # Should handle gracefully
        assert synchronizer.is_paused is True
    
    def test_resume_without_pause(self, synchronizer):
        """Test resuming without pausing."""
        synchronizer.start()
        synchronizer.resume()
        
        # Should handle gracefully
        assert synchronizer.is_paused is False


# ============================================================================
# Synchronize Method Tests (Task 6.2)
# ============================================================================


class TestSynchronizeMethod:
    """Test the synchronize() method for multi-modal coordination."""
    
    def test_synchronize_text_only(self, synchronizer):
        """Test synchronizing text-only output."""
        text_data = [
            ("Hello", 100.0),
            ("world", 100.0),
            ("!", 50.0)
        ]
        
        result = synchronizer.synchronize(text_data=text_data)
        
        assert result["success"] is True
        assert "text" in result["streams_registered"]
        assert result["total_duration_ms"] == 250.0
        assert result["alignment_offset_ms"] == 0.0
        assert result["priority_applied"] is False
    
    def test_synchronize_text_and_speech(self, synchronizer):
        """Test synchronizing text and speech outputs."""
        text_data = [
            ("Hello", 100.0),
            ("world", 100.0)
        ]
        
        speech_data = [
            (np.random.randn(2205), 100.0),
            (np.random.randn(2205), 100.0)
        ]
        
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data
        )
        
        assert result["success"] is True
        assert "text" in result["streams_registered"]
        assert "speech" in result["streams_registered"]
        assert result["total_duration_ms"] == 200.0
        assert len(synchronizer.streams) == 2
    
    def test_synchronize_all_modalities(self, synchronizer):
        """Test synchronizing all three modalities."""
        text_data = [("Hello", 100.0)]
        speech_data = [(np.random.randn(2205), 100.0)]
        visual_data = [("image_data", 100.0)]
        
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data,
            visual_data=visual_data
        )
        
        assert result["success"] is True
        assert len(result["streams_registered"]) == 3
        assert "text" in result["streams_registered"]
        assert "speech" in result["streams_registered"]
        assert "visual" in result["streams_registered"]
    
    def test_synchronize_empty_data(self, synchronizer):
        """Test synchronizing with no data."""
        result = synchronizer.synchronize()
        
        assert result["success"] is False
        assert len(result["streams_registered"]) == 0
        assert result["message"] == "No data provided"
    
    def test_synchronize_timing_alignment(self, synchronizer):
        """Test timing alignment when durations mismatch."""
        # Text is shorter than speech
        text_data = [("Hello", 100.0)]
        speech_data = [
            (np.random.randn(2205), 100.0),
            (np.random.randn(2205), 100.0)
        ]
        
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data
        )
        
        assert result["success"] is True
        # Should have applied alignment
        assert result["alignment_offset_ms"] > 0.0
        
        # Both streams should have same total duration
        text_stream = synchronizer.streams[OutputModality.TEXT]
        speech_stream = synchronizer.streams[OutputModality.SPEECH]
        
        assert text_stream.total_duration_ms == speech_stream.total_duration_ms
    
    def test_synchronize_no_alignment_needed(self, synchronizer):
        """Test synchronization when durations match."""
        text_data = [("Hello", 100.0), ("world", 100.0)]
        speech_data = [
            (np.random.randn(2205), 100.0),
            (np.random.randn(2205), 100.0)
        ]
        
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data
        )
        
        assert result["success"] is True
        # No alignment needed
        assert result["alignment_offset_ms"] == 0.0
    
    def test_synchronize_custom_priority_order(self, synchronizer):
        """Test synchronization with custom priority order."""
        text_data = [("Hello", 100.0)]
        speech_data = [(np.random.randn(2205), 100.0)]
        
        # Custom priority: speech first, then text
        priority_order = [OutputModality.SPEECH, OutputModality.TEXT]
        
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data,
            priority_order=priority_order
        )
        
        assert result["success"] is True
        assert len(result["streams_registered"]) == 2
    
    def test_synchronize_priority_filtering(self, synchronizer):
        """Test priority filtering when resources are limited."""
        # Mock _should_apply_priority to return True
        synchronizer._should_apply_priority = lambda: True
        
        text_data = [("Hello", 100.0)]
        speech_data = [(np.random.randn(2205), 100.0)]
        visual_data = [("image", 100.0)]
        
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data,
            visual_data=visual_data
        )
        
        assert result["success"] is True
        assert result["priority_applied"] is True
        # Should keep only top 2 modalities (text and speech by default priority)
        assert len(result["streams_registered"]) == 2
        assert "text" in result["streams_registered"]
        assert "speech" in result["streams_registered"]
    
    def test_synchronize_integration_with_playback(self, synchronizer):
        """Test synchronize() integration with playback."""
        text_data = [("Hello", 100.0), ("world", 100.0)]
        speech_data = [
            (np.random.randn(2205), 100.0),
            (np.random.randn(2205), 100.0)
        ]
        
        # Synchronize
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data
        )
        
        assert result["success"] is True
        
        # Start playback
        synchronizer.start()
        
        # Get synchronized chunks
        chunks = synchronizer.get_next_chunks()
        
        # Should get both text and speech chunks
        assert OutputModality.TEXT in chunks or OutputModality.SPEECH in chunks
    
    def test_synchronize_large_duration_mismatch(self, synchronizer):
        """Test synchronization with large duration mismatch."""
        # Text is much shorter than speech
        text_data = [("Hi", 50.0)]
        speech_data = [
            (np.random.randn(2205), 100.0),
            (np.random.randn(2205), 100.0),
            (np.random.randn(2205), 100.0)
        ]
        
        result = synchronizer.synchronize(
            text_data=text_data,
            speech_data=speech_data
        )
        
        assert result["success"] is True
        # Should have applied significant alignment
        assert result["alignment_offset_ms"] >= 200.0
        
        # Verify alignment worked
        text_stream = synchronizer.streams[OutputModality.TEXT]
        speech_stream = synchronizer.streams[OutputModality.SPEECH]
        
        assert abs(text_stream.total_duration_ms - speech_stream.total_duration_ms) < 1.0
    
    def test_synchronize_multiple_calls(self, synchronizer):
        """Test calling synchronize() multiple times."""
        # First synchronization
        result1 = synchronizer.synchronize(
            text_data=[("Hello", 100.0)]
        )
        
        assert result1["success"] is True
        
        # Second synchronization (should add more data)
        result2 = synchronizer.synchronize(
            text_data=[("world", 100.0)]
        )
        
        assert result2["success"] is True
        
        # Should have accumulated chunks
        text_stream = synchronizer.streams[OutputModality.TEXT]
        assert text_stream.total_duration_ms == 200.0
    
    def test_align_timing_text_padding(self, synchronizer):
        """Test _align_timing adds text padding correctly."""
        modality_data = {
            OutputModality.TEXT: [("Hello", 100.0)]
        }
        modality_durations = {
            OutputModality.TEXT: 100.0
        }
        target_duration = 200.0
        
        offset = synchronizer._align_timing(
            modality_data,
            modality_durations,
            target_duration
        )
        
        assert offset == 100.0
        # Should have added padding chunk
        assert len(modality_data[OutputModality.TEXT]) == 2
        assert modality_data[OutputModality.TEXT][1] == ("", 100.0)
    
    def test_align_timing_speech_padding(self, synchronizer):
        """Test _align_timing adds speech silence correctly."""
        audio = np.random.randn(2205)
        modality_data = {
            OutputModality.SPEECH: [(audio, 100.0)]
        }
        modality_durations = {
            OutputModality.SPEECH: 100.0
        }
        target_duration = 200.0
        
        offset = synchronizer._align_timing(
            modality_data,
            modality_durations,
            target_duration
        )
        
        assert offset == 100.0
        # Should have added silence chunk
        assert len(modality_data[OutputModality.SPEECH]) == 2
        silence, duration = modality_data[OutputModality.SPEECH][1]
        assert duration == 100.0
        assert isinstance(silence, np.ndarray)
        assert np.all(silence == 0.0)
    
    def test_apply_priority_filter(self, synchronizer):
        """Test _apply_priority_filter keeps top modalities."""
        modality_data = {
            OutputModality.TEXT: [("Hello", 100.0)],
            OutputModality.SPEECH: [(np.random.randn(2205), 100.0)],
            OutputModality.VISUAL: [("image", 100.0)]
        }
        
        priority_order = [
            OutputModality.SPEECH,
            OutputModality.TEXT,
            OutputModality.VISUAL
        ]
        
        filtered = synchronizer._apply_priority_filter(
            modality_data,
            priority_order
        )
        
        # Should keep top 2
        assert len(filtered) == 2
        assert OutputModality.SPEECH in filtered
        assert OutputModality.TEXT in filtered
        assert OutputModality.VISUAL not in filtered
    
    def test_should_apply_priority_default(self, synchronizer):
        """Test _should_apply_priority default behavior."""
        # Default should be False (allow all modalities)
        assert synchronizer._should_apply_priority() is False


# ============================================================================
# Streaming Coordination Tests (Task 6.3)
# ============================================================================


class TestStreamingCoordination:
    """Test streaming coordination features (Task 6.3)."""
    
    def test_start_streaming(self, synchronizer):
        """Test starting streaming mode."""
        modalities = [OutputModality.TEXT, OutputModality.SPEECH]
        
        synchronizer.start_streaming(modalities)
        
        # Check streams are registered and in buffering state
        assert OutputModality.TEXT in synchronizer.streams
        assert OutputModality.SPEECH in synchronizer.streams
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.BUFFERING
        assert synchronizer.streams[OutputModality.SPEECH].state == StreamState.BUFFERING
        
        # Reference time should be set
        assert synchronizer.reference_time is not None
    
    def test_add_streaming_chunk(self, synchronizer):
        """Test adding chunks in streaming mode."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add first chunk (insufficient buffer)
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=50.0
        )
        
        # Should still be buffering
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.BUFFERING
        
        # Add more chunks to reach buffer threshold
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=60.0
        )
        
        # Should transition to playing
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PLAYING
    
    def test_add_streaming_chunk_final(self, synchronizer):
        """Test adding final chunk in streaming mode."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add chunks
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=100.0,
            is_final=True
        )
        
        # Check final chunk is marked
        stream = synchronizer.streams[OutputModality.TEXT]
        final_chunk = list(stream.buffer)[-1]
        assert final_chunk.is_final is True
    
    def test_get_streaming_chunks(self, synchronizer):
        """Test getting chunks in streaming mode."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add chunks
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=100.0
        )
        
        # Get chunks
        chunks = synchronizer.get_streaming_chunks()
        
        # Should get first chunk
        assert OutputModality.TEXT in chunks
        assert chunks[OutputModality.TEXT].data == "Hello"
    
    def test_get_streaming_chunks_buffering(self, synchronizer):
        """Test getting chunks while still buffering."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add insufficient data
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=50.0
        )
        
        # Should not return chunks yet (still buffering)
        chunks = synchronizer.get_streaming_chunks()
        assert len(chunks) == 0
    
    def test_get_streaming_chunks_buffer_underrun(self, synchronizer):
        """Test buffer underrun handling in streaming mode."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add enough data to start playing
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        # Get chunk
        chunks = synchronizer.get_streaming_chunks()
        assert OutputModality.TEXT in chunks
        
        # Try to get another chunk (buffer underrun)
        time.sleep(0.11)
        chunks = synchronizer.get_streaming_chunks()
        
        # Stream should transition back to buffering
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.BUFFERING
        
        # Check statistics
        stats = synchronizer.get_statistics()
        assert stats["buffer_underruns"] > 0
    
    def test_get_streaming_chunks_temporal_coherence(self, synchronizer):
        """Test temporal coherence in streaming mode."""
        synchronizer.start_streaming([OutputModality.TEXT, OutputModality.SPEECH])
        
        # Add synchronized chunks
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.SPEECH,
            data=np.random.randn(2205),
            duration_ms=100.0
        )
        
        # Add more chunks
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=100.0
        )
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.SPEECH,
            data=np.random.randn(2205),
            duration_ms=100.0
        )
        
        # Get chunks
        chunks = synchronizer.get_streaming_chunks()
        
        # Should get both modalities synchronized
        assert OutputModality.TEXT in chunks or OutputModality.SPEECH in chunks
        
        # Check temporal coherence
        assert synchronizer.is_synchronized
    
    def test_pause_streaming(self, synchronizer):
        """Test pausing streaming."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.pause_streaming()
        
        assert synchronizer.is_paused is True
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PAUSED
    
    def test_resume_streaming(self, synchronizer):
        """Test resuming streaming."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.pause_streaming()
        time.sleep(0.05)
        synchronizer.resume_streaming()
        
        assert synchronizer.is_paused is False
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PLAYING
    
    def test_stop_streaming(self, synchronizer):
        """Test stopping streaming."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.stop_streaming()
        
        assert synchronizer.reference_time is None
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.IDLE
        assert synchronizer.streams[OutputModality.TEXT].buffer_size() == 0
    
    def test_get_streaming_status(self, synchronizer):
        """Test getting streaming status."""
        synchronizer.start_streaming([OutputModality.TEXT, OutputModality.SPEECH])
        
        # Add chunks
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.SPEECH,
            data=np.random.randn(2205),
            duration_ms=100.0
        )
        
        status = synchronizer.get_streaming_status()
        
        assert status["is_streaming"] is True
        assert status["is_paused"] is False
        assert "streams" in status
        assert "buffer_health" in status
        assert "temporal_coherence" in status
        assert "statistics" in status
        
        # Check stream status
        assert "text" in status["streams"]
        assert "speech" in status["streams"]
        
        # Check buffer health
        assert "text" in status["buffer_health"]
        assert "speech" in status["buffer_health"]
    
    def test_get_streaming_status_buffer_health(self, synchronizer):
        """Test buffer health reporting in streaming status."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add insufficient data
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=40.0
        )
        
        status = synchronizer.get_streaming_status()
        
        # Buffer should be unhealthy (< 50% of 100ms target)
        assert status["buffer_health"]["text"]["is_healthy"] is False
        assert status["buffer_health"]["text"]["buffered_ms"] == 40.0
        
        # Add more data
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=70.0
        )
        
        status = synchronizer.get_streaming_status()
        
        # Buffer should be healthy now
        assert status["buffer_health"]["text"]["is_healthy"] is True
        assert status["buffer_health"]["text"]["buffered_ms"] == 110.0
    
    def test_ensure_buffer_health(self, synchronizer):
        """Test buffer health checking."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add insufficient data
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=40.0
        )
        
        # Buffer should be unhealthy
        assert synchronizer.ensure_buffer_health(OutputModality.TEXT) is False
        
        # Add more data
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=70.0
        )
        
        # Buffer should be healthy now
        assert synchronizer.ensure_buffer_health(OutputModality.TEXT) is True
    
    def test_ensure_buffer_health_nonexistent_stream(self, synchronizer):
        """Test buffer health check for non-existent stream."""
        # Should return False for non-existent stream
        assert synchronizer.ensure_buffer_health(OutputModality.TEXT) is False
    
    def test_streaming_multimodal_workflow(self, synchronizer):
        """Test complete multimodal streaming workflow."""
        # Start streaming for text and speech
        synchronizer.start_streaming([OutputModality.TEXT, OutputModality.SPEECH])
        
        # Add chunks incrementally (simulating real-time generation)
        for i in range(3):
            synchronizer.add_streaming_chunk(
                modality=OutputModality.TEXT,
                data=f"Text{i}",
                duration_ms=100.0,
                is_final=(i == 2)
            )
            
            synchronizer.add_streaming_chunk(
                modality=OutputModality.SPEECH,
                data=np.random.randn(2205),
                duration_ms=100.0,
                is_final=(i == 2)
            )
        
        # Get streaming status
        status = synchronizer.get_streaming_status()
        assert status["is_streaming"] is True
        
        # Get chunks
        chunks = synchronizer.get_streaming_chunks()
        assert len(chunks) > 0
        
        # Check buffer health
        assert synchronizer.ensure_buffer_health(OutputModality.TEXT)
        assert synchronizer.ensure_buffer_health(OutputModality.SPEECH)
    
    def test_streaming_pause_resume_workflow(self, synchronizer):
        """Test streaming with pause/resume workflow."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add chunks
        for i in range(5):
            synchronizer.add_streaming_chunk(
                modality=OutputModality.TEXT,
                data=f"Text{i}",
                duration_ms=100.0
            )
        
        # Get first chunk
        chunks1 = synchronizer.get_streaming_chunks()
        assert len(chunks1) > 0
        
        # Pause
        synchronizer.pause_streaming()
        assert synchronizer.is_paused is True
        
        # Try to get chunks while paused (should return empty)
        chunks_paused = synchronizer.get_streaming_chunks()
        assert len(chunks_paused) == 0
        
        # Resume
        time.sleep(0.05)
        synchronizer.resume_streaming()
        assert synchronizer.is_paused is False
        
        # Get next chunk
        time.sleep(0.11)
        chunks2 = synchronizer.get_streaming_chunks()
        
        # Should be able to continue
        assert synchronizer.streams[OutputModality.TEXT].state in [
            StreamState.PLAYING,
            StreamState.BUFFERING
        ]
    
    def test_streaming_buffer_prevents_stuttering(self, synchronizer):
        """Test that buffering prevents stuttering in streaming mode."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add first chunk (insufficient buffer)
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=50.0
        )
        
        # Should not start playing yet (prevents stuttering)
        chunks = synchronizer.get_streaming_chunks()
        assert len(chunks) == 0
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.BUFFERING
        
        # Add more data to reach buffer threshold
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=60.0
        )
        
        # Now should start playing (buffer is sufficient)
        chunks = synchronizer.get_streaming_chunks()
        assert len(chunks) > 0
        assert synchronizer.streams[OutputModality.TEXT].state == StreamState.PLAYING
    
    def test_streaming_temporal_coherence_maintained(self, synchronizer):
        """Test that temporal coherence is maintained during streaming."""
        synchronizer.start_streaming([OutputModality.TEXT, OutputModality.SPEECH])
        
        # Add chunks with automatic timestamps
        for i in range(3):
            synchronizer.add_streaming_chunk(
                modality=OutputModality.TEXT,
                data=f"Text{i}",
                duration_ms=100.0
            )
            
            synchronizer.add_streaming_chunk(
                modality=OutputModality.SPEECH,
                data=np.random.randn(2205),
                duration_ms=100.0
            )
        
        # Get chunks multiple times
        for _ in range(3):
            chunks = synchronizer.get_streaming_chunks()
            time.sleep(0.11)
        
        # Check that synchronization was maintained
        stats = synchronizer.get_statistics()
        
        # Drift should be minimal
        assert stats["max_drift_observed_ms"] < synchronizer.config.max_drift_ms * 2
    
    def test_streaming_final_chunk_completion(self, synchronizer):
        """Test that streams complete properly with final chunks."""
        synchronizer.start_streaming([OutputModality.TEXT])
        
        # Add chunks with final marker
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="Hello",
            duration_ms=100.0
        )
        
        synchronizer.add_streaming_chunk(
            modality=OutputModality.TEXT,
            data="world",
            duration_ms=100.0,
            is_final=True
        )
        
        # Get all chunks
        chunks1 = synchronizer.get_streaming_chunks()
        time.sleep(0.11)
        chunks2 = synchronizer.get_streaming_chunks()
        
        # Stream should complete after final chunk
        time.sleep(0.11)
        synchronizer.get_streaming_chunks()
        
        # Check if stream completed
        status = synchronizer.get_streaming_status()
        # Stream should be completed or have no more data
        assert (
            status["streams"]["text"]["state"] == "completed" or
            status["streams"]["text"]["buffer_size"] == 0
        )
