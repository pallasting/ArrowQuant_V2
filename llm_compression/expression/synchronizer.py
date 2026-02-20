"""
Multi-modal output synchronizer for the Expression & Presentation Layer.

This module implements the Synchronizer class which coordinates timing and
buffering across multiple output modalities (text, audio, visual) to maintain
temporal coherence and prevent stuttering.

Requirements: 4.1, 4.6, 6.3
"""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Iterator
import numpy as np

from llm_compression.expression.expression_types import OutputModality

logger = logging.getLogger(__name__)


# ============================================================================
# Stream Types and Data Structures
# ============================================================================


class StreamState(Enum):
    """State of an output stream."""
    IDLE = "idle"
    BUFFERING = "buffering"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class StreamChunk:
    """
    A chunk of output data for a specific modality.
    
    Attributes:
        modality: Output modality (text, speech, visual)
        data: Chunk data (str for text, np.ndarray for audio, etc.)
        timestamp_ms: Timestamp in milliseconds from stream start
        duration_ms: Duration of this chunk in milliseconds
        sequence_number: Sequence number for ordering
        is_final: Whether this is the final chunk
    """
    modality: OutputModality
    data: Any
    timestamp_ms: float
    duration_ms: float
    sequence_number: int
    is_final: bool = False


@dataclass
class OutputStream:
    """
    Represents a single output stream (text, audio, or visual).
    
    Attributes:
        modality: Output modality
        state: Current stream state
        buffer: Buffer of chunks waiting to be played
        current_position_ms: Current playback position in milliseconds
        total_duration_ms: Total duration of stream in milliseconds
        next_sequence: Next expected sequence number
        start_time: Stream start time (for synchronization)
    """
    modality: OutputModality
    state: StreamState = StreamState.IDLE
    buffer: deque = field(default_factory=deque)
    current_position_ms: float = 0.0
    total_duration_ms: float = 0.0
    next_sequence: int = 0
    start_time: Optional[float] = None
    
    def add_chunk(self, chunk: StreamChunk):
        """Add a chunk to the buffer."""
        self.buffer.append(chunk)
        self.total_duration_ms += chunk.duration_ms
    
    def get_next_chunk(self) -> Optional[StreamChunk]:
        """Get the next chunk from buffer."""
        if self.buffer:
            return self.buffer.popleft()
        return None
    
    def has_data(self) -> bool:
        """Check if stream has buffered data."""
        return len(self.buffer) > 0
    
    def buffer_size(self) -> int:
        """Get number of chunks in buffer."""
        return len(self.buffer)


# ============================================================================
# Synchronizer Configuration
# ============================================================================


@dataclass
class SynchronizerConfig:
    """
    Configuration for the Synchronizer.
    
    Attributes:
        buffer_size_ms: Buffer size in milliseconds (prevents stuttering)
        max_drift_ms: Maximum allowed drift between streams in milliseconds
        sync_interval_ms: Interval for synchronization checks in milliseconds
        enable_auto_sync: Enable automatic synchronization
        fallback_to_text: Fallback to text-only on sync failures
    """
    buffer_size_ms: float = 200.0  # 200ms buffer
    max_drift_ms: float = 100.0  # 100ms max drift
    sync_interval_ms: float = 50.0  # Check sync every 50ms
    enable_auto_sync: bool = True
    fallback_to_text: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.buffer_size_ms <= 0:
            raise ValueError(f"buffer_size_ms must be positive, got {self.buffer_size_ms}")
        
        if self.max_drift_ms <= 0:
            raise ValueError(f"max_drift_ms must be positive, got {self.max_drift_ms}")
        
        if self.sync_interval_ms <= 0:
            raise ValueError(f"sync_interval_ms must be positive, got {self.sync_interval_ms}")


# ============================================================================
# Synchronizer Class
# ============================================================================


class Synchronizer:
    """
    Multi-modal output synchronizer.
    
    Coordinates timing and buffering across multiple output modalities
    (text, audio, visual) to maintain temporal coherence and prevent
    stuttering.
    
    Key responsibilities:
    - Track output streams (text, audio, visual)
    - Implement timing coordination between streams
    - Implement buffer management to prevent stuttering
    - Maintain temporal coherence across modalities
    - Handle output priority when resources are limited
    - Support pause/resume functionality
    - Provide fallback to text-only when needed
    
    Requirements: 4.1, 4.6, 6.3
    """
    
    def __init__(self, config: Optional[SynchronizerConfig] = None):
        """
        Initialize the Synchronizer.
        
        Args:
            config: Synchronizer configuration (uses defaults if None)
        """
        self.config = config or SynchronizerConfig()
        
        # Output streams
        self.streams: Dict[OutputModality, OutputStream] = {}
        
        # Synchronization state
        self.is_synchronized = True
        self.reference_time: Optional[float] = None
        self.is_paused = False
        self.pause_time: Optional[float] = None
        
        # Statistics
        self.stats = {
            "total_chunks_processed": 0,
            "sync_corrections": 0,
            "buffer_underruns": 0,
            "max_drift_observed_ms": 0.0
        }
        
        # Fallback state
        self._fallback_active = False
        self._fallback_mode = None
        self._fallback_config = {
            "enable_text_fallback": True,
            "enable_single_modality_fallback": True,
            "enable_auto_recovery": False,
            "recovery_retry_delay_ms": 1000.0
        }
        
        logger.info(f"Synchronizer initialized with config: {self.config}")
    
    def register_stream(self, modality: OutputModality):
        """
        Register an output stream.
        
        Args:
            modality: Output modality to register
        """
        if modality not in self.streams:
            self.streams[modality] = OutputStream(modality=modality)
            logger.info(f"Registered stream: {modality.value}")
        else:
            logger.warning(f"Stream {modality.value} already registered")
    
    def add_chunk(
        self,
        modality: OutputModality,
        data: Any,
        duration_ms: float,
        timestamp_ms: Optional[float] = None,
        is_final: bool = False
    ):
        """
        Add a chunk to a stream.
        
        Args:
            modality: Output modality
            data: Chunk data
            duration_ms: Duration of chunk in milliseconds
            timestamp_ms: Optional timestamp (auto-calculated if None)
            is_final: Whether this is the final chunk
        """
        # Ensure stream is registered
        if modality not in self.streams:
            self.register_stream(modality)
        
        stream = self.streams[modality]
        
        # Calculate timestamp if not provided
        if timestamp_ms is None:
            timestamp_ms = stream.total_duration_ms
        
        # Create chunk
        chunk = StreamChunk(
            modality=modality,
            data=data,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            sequence_number=stream.next_sequence,
            is_final=is_final
        )
        
        # Add to stream
        stream.add_chunk(chunk)
        stream.next_sequence += 1
        
        # Update state
        if stream.state == StreamState.IDLE:
            stream.state = StreamState.BUFFERING
        
        self.stats["total_chunks_processed"] += 1
        
        logger.debug(
            f"Added chunk to {modality.value}: "
            f"seq={chunk.sequence_number}, "
            f"duration={duration_ms}ms, "
            f"final={is_final}"
        )
    
    def start(self):
        """Start synchronized playback."""
        if self.reference_time is not None:
            logger.warning("Synchronizer already started")
            return
        
        self.reference_time = time.time()
        
        # Start all streams
        for stream in self.streams.values():
            if stream.state == StreamState.BUFFERING:
                stream.state = StreamState.PLAYING
                stream.start_time = self.reference_time
        
        logger.info("Synchronizer started")
    
    def pause(self):
        """Pause all streams."""
        if self.is_paused:
            logger.warning("Synchronizer already paused")
            return
        
        self.is_paused = True
        self.pause_time = time.time()
        
        # Pause all streams
        for stream in self.streams.values():
            if stream.state == StreamState.PLAYING:
                stream.state = StreamState.PAUSED
        
        logger.info("Synchronizer paused")
    
    def resume(self):
        """Resume all streams."""
        if not self.is_paused:
            logger.warning("Synchronizer not paused")
            return
        
        # Calculate pause duration
        pause_duration = time.time() - self.pause_time if self.pause_time else 0
        
        # Adjust reference time
        if self.reference_time:
            self.reference_time += pause_duration
        
        self.is_paused = False
        self.pause_time = None
        
        # Resume all streams
        for stream in self.streams.values():
            if stream.state == StreamState.PAUSED:
                stream.state = StreamState.PLAYING
        
        logger.info(f"Synchronizer resumed (pause duration: {pause_duration:.2f}s)")
    
    def stop(self):
        """Stop all streams and reset."""
        self.reference_time = None
        self.is_paused = False
        self.pause_time = None
        
        # Stop all streams
        for stream in self.streams.values():
            stream.state = StreamState.IDLE
            stream.buffer.clear()
            stream.current_position_ms = 0.0
            stream.total_duration_ms = 0.0
            stream.next_sequence = 0
            stream.start_time = None
        
        logger.info("Synchronizer stopped")
    
    def get_next_chunks(self) -> Dict[OutputModality, Optional[StreamChunk]]:
        """
        Get the next synchronized chunks from all streams.
        
        This method ensures temporal coherence by only returning chunks
        that should be played at the current time.
        
        Returns:
            Dictionary mapping modality to next chunk (or None)
        """
        if self.is_paused:
            return {}
        
        if self.reference_time is None:
            logger.warning("Synchronizer not started")
            return {}
        
        # Calculate current playback time
        current_time_ms = (time.time() - self.reference_time) * 1000
        
        # Check synchronization
        if self.config.enable_auto_sync:
            self._check_synchronization(current_time_ms)
        
        # Get chunks that should be played now
        chunks = {}
        for modality, stream in self.streams.items():
            if stream.state != StreamState.PLAYING:
                continue
            
            # Check if we have buffered data
            if not stream.has_data():
                # Buffer underrun
                self.stats["buffer_underruns"] += 1
                logger.warning(f"Buffer underrun for {modality.value}")
                continue
            
            # Check if next chunk should be played
            next_chunk = stream.buffer[0] if stream.buffer else None
            if next_chunk and next_chunk.timestamp_ms <= current_time_ms:
                chunk = stream.get_next_chunk()
                stream.current_position_ms = chunk.timestamp_ms + chunk.duration_ms
                chunks[modality] = chunk
                
                # Check if stream completed
                if chunk.is_final and not stream.has_data():
                    stream.state = StreamState.COMPLETED
                    logger.info(f"Stream {modality.value} completed")
        
        return chunks
    
    def _check_synchronization(self, current_time_ms: float):
        """
        Check synchronization between streams and correct drift.
        
        Args:
            current_time_ms: Current playback time in milliseconds
        """
        # Get positions of all playing streams
        positions = []
        for stream in self.streams.values():
            if stream.state == StreamState.PLAYING:
                positions.append(stream.current_position_ms)
        
        if len(positions) < 2:
            # Nothing to synchronize
            return
        
        # Calculate drift (max - min position)
        max_pos = max(positions)
        min_pos = min(positions)
        drift = max_pos - min_pos
        
        # Update statistics
        self.stats["max_drift_observed_ms"] = max(
            self.stats["max_drift_observed_ms"],
            drift
        )
        
        # Check if drift exceeds threshold
        if drift > self.config.max_drift_ms:
            logger.warning(f"Drift detected: {drift:.2f}ms (max: {self.config.max_drift_ms}ms)")
            self._correct_drift(max_pos, min_pos)
            self.stats["sync_corrections"] += 1
            self.is_synchronized = False
        else:
            self.is_synchronized = True
    
    def _correct_drift(self, max_pos: float, min_pos: float):
        """
        Correct drift between streams.
        
        Strategy: Adjust slower streams to catch up with faster streams.
        
        Args:
            max_pos: Maximum position across streams
            min_pos: Minimum position across streams
        """
        target_pos = (max_pos + min_pos) / 2
        
        for stream in self.streams.values():
            if stream.state != StreamState.PLAYING:
                continue
            
            # Adjust position
            if stream.current_position_ms < target_pos:
                # Stream is behind, skip ahead
                logger.debug(
                    f"Adjusting {stream.modality.value} from "
                    f"{stream.current_position_ms:.2f}ms to {target_pos:.2f}ms"
                )
                stream.current_position_ms = target_pos
    
    def is_buffering_complete(self) -> bool:
        """
        Check if all streams have sufficient buffered data.
        
        Returns:
            True if all streams have enough data buffered
        """
        for stream in self.streams.values():
            if stream.state in [StreamState.BUFFERING, StreamState.PLAYING]:
                # Calculate buffered duration
                buffered_ms = sum(
                    chunk.duration_ms for chunk in stream.buffer
                )
                
                if buffered_ms < self.config.buffer_size_ms:
                    return False
        
        return True
    
    def get_stream_state(self, modality: OutputModality) -> Optional[StreamState]:
        """
        Get the state of a stream.
        
        Args:
            modality: Output modality
            
        Returns:
            Stream state or None if stream not registered
        """
        stream = self.streams.get(modality)
        return stream.state if stream else None
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get synchronization statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "is_synchronized": self.is_synchronized,
            "is_paused": self.is_paused,
            "active_streams": len([
                s for s in self.streams.values()
                if s.state == StreamState.PLAYING
            ]),
            "completed_streams": len([
                s for s in self.streams.values()
                if s.state == StreamState.COMPLETED
            ])
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.stats = {
            "total_chunks_processed": 0,
            "sync_corrections": 0,
            "buffer_underruns": 0,
            "max_drift_observed_ms": 0.0
        }
        logger.info("Statistics reset")
    
    def synchronize(
        self,
        text_data: Optional[List[tuple]] = None,
        speech_data: Optional[List[tuple]] = None,
        visual_data: Optional[List[tuple]] = None,
        priority_order: Optional[List[OutputModality]] = None
    ) -> Dict[str, Any]:
        """
        Synchronize multi-modal outputs with timing alignment and priority handling.
        
        This is the main entry point for coordinating multiple output modalities.
        It handles:
        - Text and speech timing alignment
        - Multi-modal output coordination
        - Output priority when resources are limited
        
        Args:
            text_data: List of (text, duration_ms) tuples for text output
            speech_data: List of (audio_array, duration_ms) tuples for speech output
            visual_data: List of (visual_data, duration_ms) tuples for visual output
            priority_order: List of modalities in priority order (highest first)
                           Defaults to [TEXT, SPEECH, VISUAL]
        
        Returns:
            Dictionary with synchronization results:
            - success: Whether synchronization succeeded
            - streams_registered: List of registered modalities
            - total_duration_ms: Total synchronized duration
            - alignment_offset_ms: Timing offset applied for alignment
            - priority_applied: Whether priority filtering was applied
            - message: Status message
        
        Requirements: 4.3 (Multi-modal coordination), 4.5 (Output priority)
        
        Example:
            >>> sync = Synchronizer()
            >>> result = sync.synchronize(
            ...     text_data=[("Hello", 100.0), ("world", 100.0)],
            ...     speech_data=[(audio1, 100.0), (audio2, 100.0)]
            ... )
            >>> sync.start()
            >>> chunks = sync.get_next_chunks()
        """
        # Set default priority order
        if priority_order is None:
            priority_order = [
                OutputModality.TEXT,
                OutputModality.SPEECH,
                OutputModality.VISUAL
            ]
        
        # Collect all modality data
        modality_data = {}
        if text_data:
            modality_data[OutputModality.TEXT] = text_data
        if speech_data:
            modality_data[OutputModality.SPEECH] = speech_data
        if visual_data:
            modality_data[OutputModality.VISUAL] = visual_data
        
        if not modality_data:
            logger.warning("No data provided for synchronization")
            return {
                "success": False,
                "streams_registered": [],
                "total_duration_ms": 0.0,
                "alignment_offset_ms": 0.0,
                "priority_applied": False,
                "message": "No data provided"
            }
        
        # Calculate total duration for each modality
        modality_durations = {}
        for modality, data in modality_data.items():
            total_duration = sum(duration for _, duration in data)
            modality_durations[modality] = total_duration
        
        # Check for duration mismatches (timing alignment)
        max_duration = max(modality_durations.values())
        min_duration = min(modality_durations.values())
        duration_mismatch = max_duration - min_duration
        
        # Apply timing alignment if mismatch exceeds threshold
        alignment_offset = 0.0
        if duration_mismatch > self.config.max_drift_ms:
            logger.warning(
                f"Duration mismatch detected: {duration_mismatch:.2f}ms "
                f"(max: {self.config.max_drift_ms}ms)"
            )
            
            # Align to the longest duration
            alignment_offset = self._align_timing(modality_data, modality_durations, max_duration)
            
            logger.info(f"Applied timing alignment: offset={alignment_offset:.2f}ms")
        
        # Handle output priority (resource limitation)
        priority_applied = False
        if len(modality_data) > 2:  # More than 2 modalities
            # Check if we should apply priority filtering
            # (In a real system, this would check resource availability)
            if self._should_apply_priority():
                logger.info("Applying output priority filtering")
                modality_data = self._apply_priority_filter(modality_data, priority_order)
                priority_applied = True
        
        # Register streams and add chunks
        streams_registered = []
        for modality, data in modality_data.items():
            self.register_stream(modality)
            streams_registered.append(modality)
            
            # Add chunks to stream
            for chunk_data, duration in data:
                self.add_chunk(
                    modality=modality,
                    data=chunk_data,
                    duration_ms=duration
                )
            
            logger.info(
                f"Registered {modality.value} stream with {len(data)} chunks, "
                f"total duration: {modality_durations[modality]:.2f}ms"
            )
        
        # Calculate final synchronized duration
        final_duration = max(
            self.streams[m].total_duration_ms
            for m in streams_registered
        )
        
        return {
            "success": True,
            "streams_registered": [m.value for m in streams_registered],
            "total_duration_ms": final_duration,
            "alignment_offset_ms": alignment_offset,
            "priority_applied": priority_applied,
            "message": f"Synchronized {len(streams_registered)} modalities successfully"
        }
    
    def _align_timing(
        self,
        modality_data: Dict[OutputModality, List[tuple]],
        modality_durations: Dict[OutputModality, float],
        target_duration: float
    ) -> float:
        """
        Align timing across modalities to match target duration.
        
        Strategy: Pad shorter modalities with silence/pauses to match longest.
        
        Args:
            modality_data: Dictionary of modality data
            modality_durations: Dictionary of modality durations
            target_duration: Target duration to align to
        
        Returns:
            Maximum alignment offset applied
        """
        max_offset = 0.0
        
        for modality, duration in modality_durations.items():
            if duration < target_duration:
                # Calculate padding needed
                padding_needed = target_duration - duration
                
                # Add padding chunk at the end
                if modality == OutputModality.TEXT:
                    # Add empty text chunk
                    modality_data[modality].append(("", padding_needed))
                elif modality == OutputModality.SPEECH:
                    # Add silence
                    sample_rate = 22050
                    silence_samples = int((padding_needed / 1000.0) * sample_rate)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    modality_data[modality].append((silence, padding_needed))
                elif modality == OutputModality.VISUAL:
                    # Add blank frame
                    modality_data[modality].append((None, padding_needed))
                
                max_offset = max(max_offset, padding_needed)
                
                logger.debug(
                    f"Padded {modality.value} with {padding_needed:.2f}ms "
                    f"to reach {target_duration:.2f}ms"
                )
        
        return max_offset
    
    def _should_apply_priority(self) -> bool:
        """
        Determine if priority filtering should be applied.
        
        In a real system, this would check:
        - Available CPU/GPU resources
        - Network bandwidth
        - Memory usage
        - User preferences
        
        For now, returns False (allow all modalities).
        
        Returns:
            True if priority filtering should be applied
        """
        # Placeholder: In production, check actual resource constraints
        # For example:
        # - Check CPU usage > 80%
        # - Check memory usage > 1.5GB
        # - Check network latency > 200ms
        
        return False
    
    def _apply_priority_filter(
        self,
        modality_data: Dict[OutputModality, List[tuple]],
        priority_order: List[OutputModality]
    ) -> Dict[OutputModality, List[tuple]]:
        """
        Filter modalities based on priority order.
        
        Keeps only the top N modalities based on priority and resource constraints.
        
        Args:
            modality_data: Dictionary of modality data
            priority_order: List of modalities in priority order
        
        Returns:
            Filtered modality data
        """
        # Keep top 2 modalities by priority
        max_modalities = 2
        
        filtered_data = {}
        for modality in priority_order:
            if modality in modality_data:
                filtered_data[modality] = modality_data[modality]
                
                if len(filtered_data) >= max_modalities:
                    break
        
        # Log which modalities were dropped
        dropped = set(modality_data.keys()) - set(filtered_data.keys())
        if dropped:
            logger.warning(
                f"Dropped modalities due to priority: "
                f"{[m.value for m in dropped]}"
            )
        
        return filtered_data
    
    def fallback_to_text_only(self):
        """
        Fallback to text-only output.
        
        Disables all non-text streams and continues with text only.
        This is used when synchronization fails or resources are limited.
        
        Requirements: 4.7, 11.1
        """
        logger.warning("Falling back to text-only output")
        
        for modality, stream in self.streams.items():
            if modality != OutputModality.TEXT:
                stream.state = StreamState.ERROR
                stream.buffer.clear()
        
        # Ensure text stream is active
        if OutputModality.TEXT in self.streams:
            text_stream = self.streams[OutputModality.TEXT]
            if text_stream.state in [StreamState.IDLE, StreamState.ERROR, StreamState.BUFFERING]:
                text_stream.state = StreamState.PLAYING
        
        # Update fallback status
        self._fallback_active = True
        self._fallback_mode = "text_only"
    
    def fallback_to_single_modality(self, preferred_modality: Optional[OutputModality] = None):
        """
        Fallback to single modality when resources are limited.
        
        Selects the best available modality based on priority and resource constraints.
        Disables all other modalities to conserve resources.
        
        Args:
            preferred_modality: Preferred modality to keep (auto-selected if None)
        
        Requirements: 4.7, 11.1
        """
        logger.warning("Falling back to single modality due to resource constraints")
        
        # Determine which modality to keep
        if preferred_modality and preferred_modality in self.streams:
            selected_modality = preferred_modality
        else:
            # Auto-select based on priority: TEXT > SPEECH > VISUAL
            priority_order = [OutputModality.TEXT, OutputModality.SPEECH, OutputModality.VISUAL]
            selected_modality = None
            for modality in priority_order:
                if modality in self.streams and self.streams[modality].state != StreamState.ERROR:
                    selected_modality = modality
                    break
            
            if selected_modality is None:
                # No valid streams, fallback to text
                logger.error("No valid streams available, creating text stream")
                self.register_stream(OutputModality.TEXT)
                selected_modality = OutputModality.TEXT
        
        logger.info(f"Selected modality for fallback: {selected_modality.value}")
        
        # Disable all other modalities
        for modality, stream in self.streams.items():
            if modality != selected_modality:
                stream.state = StreamState.ERROR
                stream.buffer.clear()
            else:
                # Ensure selected stream is active
                if stream.state in [StreamState.IDLE, StreamState.ERROR]:
                    stream.state = StreamState.PLAYING
        
        # Update fallback status
        self._fallback_active = True
        self._fallback_mode = f"single_{selected_modality.value}"
    
    def apply_graceful_degradation(self, error_modality: OutputModality):
        """
        Apply graceful degradation when a specific modality fails.
        
        This method implements a degradation strategy that:
        1. Disables the failed modality
        2. Continues with remaining modalities
        3. Falls back to text-only if all other modalities fail
        
        Args:
            error_modality: The modality that encountered an error
        
        Requirements: 11.1 (Graceful degradation)
        """
        logger.warning(f"Applying graceful degradation for failed modality: {error_modality.value}")
        
        # Mark the failed modality as error
        if error_modality in self.streams:
            stream = self.streams[error_modality]
            stream.state = StreamState.ERROR
            stream.buffer.clear()
        
        # Count remaining active modalities
        active_modalities = [
            m for m, s in self.streams.items()
            if s.state not in [StreamState.ERROR, StreamState.IDLE, StreamState.COMPLETED]
        ]
        
        logger.info(f"Active modalities after degradation: {[m.value for m in active_modalities]}")
        
        # Apply appropriate fallback strategy
        if len(active_modalities) == 0:
            # No active modalities, fallback to text
            logger.error("All modalities failed, falling back to text-only")
            self.fallback_to_text_only()
        elif len(active_modalities) == 1:
            # Already at single modality
            logger.info(f"Continuing with single modality: {active_modalities[0].value}")
            self._fallback_active = True
            self._fallback_mode = f"degraded_{active_modalities[0].value}"
        else:
            # Multiple modalities still active
            logger.info(f"Continuing with {len(active_modalities)} modalities")
            self._fallback_active = True
            self._fallback_mode = "partial_degradation"
    
    def recover_from_error(self, modality: OutputModality) -> bool:
        """
        Attempt to recover a failed modality.
        
        This method tries to restart a modality that previously failed.
        It's useful for transient errors (network issues, temporary resource constraints).
        
        Args:
            modality: The modality to recover
        
        Returns:
            True if recovery was successful, False otherwise
        
        Requirements: 11.1 (Error recovery)
        """
        if modality not in self.streams:
            logger.error(f"Cannot recover {modality.value}: stream not registered")
            return False
        
        stream = self.streams[modality]
        
        if stream.state != StreamState.ERROR:
            logger.warning(f"Stream {modality.value} is not in error state (state: {stream.state.value})")
            return False
        
        logger.info(f"Attempting to recover stream: {modality.value}")
        
        try:
            # Clear error state
            stream.state = StreamState.IDLE
            stream.buffer.clear()
            stream.current_position_ms = 0.0
            stream.next_sequence = 0
            
            # If we're currently in fallback mode, check if we can exit it
            if self._fallback_active:
                # Count active modalities (not in error or idle state)
                active_modalities = [
                    m for m, s in self.streams.items()
                    if s.state not in [StreamState.ERROR, StreamState.IDLE, StreamState.COMPLETED]
                ]
                
                # Only exit fallback if we now have multiple active modalities
                # (the recovered stream is still IDLE, so it won't be counted yet)
                if len(active_modalities) >= 1:
                    # We have at least one active modality, and the recovered one can be activated
                    # Exit fallback mode
                    logger.info("Exiting fallback mode after recovery")
                    self._fallback_active = False
                    self._fallback_mode = None
            
            logger.info(f"Successfully recovered stream: {modality.value}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to recover stream {modality.value}: {e}")
            stream.state = StreamState.ERROR
            return False
    
    def configure_fallback(
        self,
        enable_text_fallback: bool = True,
        enable_single_modality_fallback: bool = True,
        enable_auto_recovery: bool = False,
        recovery_retry_delay_ms: float = 1000.0
    ):
        """
        Configure fallback behavior.
        
        Args:
            enable_text_fallback: Enable automatic fallback to text-only
            enable_single_modality_fallback: Enable fallback to single modality
            enable_auto_recovery: Enable automatic error recovery attempts
            recovery_retry_delay_ms: Delay between recovery attempts in milliseconds
        
        Requirements: 4.7, 11.1
        """
        self._fallback_config = {
            "enable_text_fallback": enable_text_fallback,
            "enable_single_modality_fallback": enable_single_modality_fallback,
            "enable_auto_recovery": enable_auto_recovery,
            "recovery_retry_delay_ms": recovery_retry_delay_ms
        }
        
        logger.info(f"Fallback configuration updated: {self._fallback_config}")
    
    def get_fallback_status(self) -> Dict[str, Any]:
        """
        Get current fallback status.
        
        Returns:
            Dictionary with fallback status information:
            - fallback_active: Whether fallback mode is active
            - fallback_mode: Current fallback mode (text_only, single_*, degraded_*, etc.)
            - failed_modalities: List of modalities in error state
            - active_modalities: List of currently active modalities
            - fallback_config: Current fallback configuration
        
        Requirements: 4.7, 11.1
        """
        failed_modalities = [
            m.value for m, s in self.streams.items()
            if s.state == StreamState.ERROR
        ]
        
        active_modalities = [
            m.value for m, s in self.streams.items()
            if s.state in [StreamState.PLAYING, StreamState.BUFFERING]
        ]
        
        return {
            "fallback_active": getattr(self, "_fallback_active", False),
            "fallback_mode": getattr(self, "_fallback_mode", None),
            "failed_modalities": failed_modalities,
            "active_modalities": active_modalities,
            "fallback_config": getattr(self, "_fallback_config", {
                "enable_text_fallback": True,
                "enable_single_modality_fallback": True,
                "enable_auto_recovery": False,
                "recovery_retry_delay_ms": 1000.0
            })
        }
    
    # ========================================================================
    # Streaming Coordination Methods (Task 6.3)
    # ========================================================================
    
    def start_streaming(self, modalities: List[OutputModality]):
        """
        Start streaming mode for specified modalities.
        
        Streaming mode enables continuous data flow with automatic buffering
        and temporal coherence maintenance. This is optimized for real-time
        generation scenarios where chunks arrive incrementally.
        
        Args:
            modalities: List of modalities to enable for streaming
        
        Requirements: 6.3 (Streaming coordination)
        """
        for modality in modalities:
            if modality not in self.streams:
                self.register_stream(modality)
            
            stream = self.streams[modality]
            stream.state = StreamState.BUFFERING
        
        # Start reference time if not already started
        if self.reference_time is None:
            self.reference_time = time.time()
        
        logger.info(f"Started streaming for modalities: {[m.value for m in modalities]}")
    
    def add_streaming_chunk(
        self,
        modality: OutputModality,
        data: Any,
        duration_ms: float,
        is_final: bool = False
    ):
        """
        Add a chunk in streaming mode with automatic buffering.
        
        This method is optimized for streaming scenarios where chunks arrive
        incrementally. It automatically manages buffering to prevent stuttering
        and maintains temporal coherence.
        
        Args:
            modality: Output modality
            data: Chunk data
            duration_ms: Duration of chunk in milliseconds
            is_final: Whether this is the final chunk in the stream
        
        Requirements: 6.3 (Buffer outputs to prevent stuttering)
        """
        # Add chunk using existing method
        self.add_chunk(
            modality=modality,
            data=data,
            duration_ms=duration_ms,
            is_final=is_final
        )
        
        # Check if we have enough buffered data to start playing
        if modality in self.streams:
            stream = self.streams[modality]
            
            if stream.state == StreamState.BUFFERING:
                # Calculate buffered duration
                buffered_ms = sum(chunk.duration_ms for chunk in stream.buffer)
                
                # Start playing if buffer is sufficient
                if buffered_ms >= self.config.buffer_size_ms:
                    stream.state = StreamState.PLAYING
                    stream.start_time = time.time()
                    logger.debug(
                        f"Stream {modality.value} ready to play "
                        f"(buffered: {buffered_ms:.2f}ms)"
                    )
    
    def get_streaming_chunks(self) -> Dict[OutputModality, Optional[StreamChunk]]:
        """
        Get next chunks in streaming mode with temporal coherence.
        
        This method is optimized for streaming scenarios. It ensures:
        - Chunks are delivered continuously without gaps
        - Temporal coherence is maintained across modalities
        - Buffering prevents stuttering
        - Automatic synchronization corrections
        
        Returns:
            Dictionary mapping modality to next chunk (or None)
        
        Requirements: 6.3 (Maintain temporal coherence, prevent stuttering)
        """
        if self.is_paused:
            return {}
        
        if self.reference_time is None:
            logger.warning("Streaming not started")
            return {}
        
        # Calculate current playback time
        current_time_ms = (time.time() - self.reference_time) * 1000
        
        # Check synchronization more frequently in streaming mode
        if self.config.enable_auto_sync:
            self._check_synchronization(current_time_ms)
        
        # Get chunks that should be played now
        chunks = {}
        for modality, stream in self.streams.items():
            if stream.state not in [StreamState.PLAYING, StreamState.BUFFERING]:
                continue
            
            # Check if we have buffered data
            if not stream.has_data():
                # In streaming mode, this is expected - more data may arrive
                if stream.state == StreamState.BUFFERING:
                    logger.debug(f"Stream {modality.value} still buffering")
                else:
                    # Buffer underrun during playback
                    self.stats["buffer_underruns"] += 1
                    logger.warning(
                        f"Buffer underrun for {modality.value} in streaming mode"
                    )
                    # Transition back to buffering
                    stream.state = StreamState.BUFFERING
                continue
            
            # Check if stream is ready to play
            if stream.state == StreamState.BUFFERING:
                buffered_ms = sum(chunk.duration_ms for chunk in stream.buffer)
                if buffered_ms < self.config.buffer_size_ms:
                    # Not enough buffer yet
                    continue
                else:
                    # Start playing
                    stream.state = StreamState.PLAYING
                    stream.start_time = time.time()
            
            # Get next chunk if it's time to play it
            next_chunk = stream.buffer[0] if stream.buffer else None
            if next_chunk and next_chunk.timestamp_ms <= current_time_ms:
                chunk = stream.get_next_chunk()
                stream.current_position_ms = chunk.timestamp_ms + chunk.duration_ms
                chunks[modality] = chunk
                
                # Check if stream completed
                if chunk.is_final and not stream.has_data():
                    stream.state = StreamState.COMPLETED
                    logger.info(f"Stream {modality.value} completed")
        
        return chunks
    
    def pause_streaming(self):
        """
        Pause streaming for all active streams.
        
        This maintains buffer state and allows resuming from the same position.
        Useful for long outputs where user may want to pause.
        
        Requirements: 6.4 (Support pause/resume for long outputs)
        """
        self.pause()
        logger.info("Streaming paused")
    
    def resume_streaming(self):
        """
        Resume streaming from paused state.
        
        This continues playback from where it was paused, maintaining
        temporal coherence across all streams.
        
        Requirements: 6.4 (Support pause/resume for long outputs)
        """
        self.resume()
        logger.info("Streaming resumed")
    
    def stop_streaming(self):
        """
        Stop streaming and clear all buffers.
        
        This ends the streaming session and resets all streams.
        """
        self.stop()
        logger.info("Streaming stopped")
    
    def get_streaming_status(self) -> Dict[str, Any]:
        """
        Get current streaming status for all streams.
        
        Returns:
            Dictionary with streaming status information:
            - is_streaming: Whether streaming is active
            - is_paused: Whether streaming is paused
            - streams: Status of each stream
            - buffer_health: Buffer health for each stream
            - temporal_coherence: Whether streams are synchronized
        """
        streams_status = {}
        buffer_health = {}
        
        for modality, stream in self.streams.items():
            # Calculate buffered duration
            buffered_ms = sum(chunk.duration_ms for chunk in stream.buffer)
            buffer_percentage = (buffered_ms / self.config.buffer_size_ms) * 100
            
            streams_status[modality.value] = {
                "state": stream.state.value,
                "position_ms": stream.current_position_ms,
                "total_duration_ms": stream.total_duration_ms,
                "buffer_size": stream.buffer_size(),
                "buffered_ms": buffered_ms
            }
            
            buffer_health[modality.value] = {
                "buffered_ms": buffered_ms,
                "buffer_percentage": min(100.0, buffer_percentage),
                "is_healthy": buffered_ms >= self.config.buffer_size_ms * 0.5
            }
        
        return {
            "is_streaming": self.reference_time is not None,
            "is_paused": self.is_paused,
            "streams": streams_status,
            "buffer_health": buffer_health,
            "temporal_coherence": self.is_synchronized,
            "statistics": self.get_statistics()
        }
    
    def ensure_buffer_health(self, modality: OutputModality) -> bool:
        """
        Check if a stream has healthy buffer levels.
        
        A healthy buffer prevents stuttering during playback.
        
        Args:
            modality: Output modality to check
        
        Returns:
            True if buffer is healthy (>50% of target buffer size)
        
        Requirements: 6.3 (Buffer outputs to prevent stuttering)
        """
        if modality not in self.streams:
            return False
        
        stream = self.streams[modality]
        buffered_ms = sum(chunk.duration_ms for chunk in stream.buffer)
        
        # Healthy if buffer is at least 50% full
        threshold = self.config.buffer_size_ms * 0.5
        is_healthy = buffered_ms >= threshold
        
        if not is_healthy:
            logger.warning(
                f"Buffer health low for {modality.value}: "
                f"{buffered_ms:.2f}ms (threshold: {threshold:.2f}ms)"
            )
        
        return is_healthy
    def synchronize(
        self,
        text_data: Optional[List[tuple]] = None,
        speech_data: Optional[List[tuple]] = None,
        visual_data: Optional[List[tuple]] = None,
        priority_order: Optional[List[OutputModality]] = None
    ) -> Dict[str, Any]:
        """
        Synchronize multi-modal outputs with timing alignment and priority handling.

        This is the main entry point for coordinating multiple output modalities.
        It handles:
        - Text and speech timing alignment
        - Multi-modal output coordination
        - Output priority when resources are limited

        Args:
            text_data: List of (text, duration_ms) tuples for text output
            speech_data: List of (audio_array, duration_ms) tuples for speech output
            visual_data: List of (visual_data, duration_ms) tuples for visual output
            priority_order: List of modalities in priority order (highest first)
                           Defaults to [TEXT, SPEECH, VISUAL]

        Returns:
            Dictionary with synchronization results:
            - success: Whether synchronization succeeded
            - streams_registered: List of registered modalities
            - total_duration_ms: Total synchronized duration
            - alignment_offset_ms: Timing offset applied for alignment
            - priority_applied: Whether priority filtering was applied
            - message: Status message

        Requirements: 4.3 (Multi-modal coordination), 4.5 (Output priority)

        Example:
            >>> sync = Synchronizer()
            >>> result = sync.synchronize(
            ...     text_data=[("Hello", 100.0), ("world", 100.0)],
            ...     speech_data=[(audio1, 100.0), (audio2, 100.0)]
            ... )
            >>> sync.start()
            >>> chunks = sync.get_next_chunks()
        """
        # Set default priority order
        if priority_order is None:
            priority_order = [
                OutputModality.TEXT,
                OutputModality.SPEECH,
                OutputModality.VISUAL
            ]

        # Collect all modality data
        modality_data = {}
        if text_data:
            modality_data[OutputModality.TEXT] = text_data
        if speech_data:
            modality_data[OutputModality.SPEECH] = speech_data
        if visual_data:
            modality_data[OutputModality.VISUAL] = visual_data

        if not modality_data:
            logger.warning("No data provided for synchronization")
            return {
                "success": False,
                "streams_registered": [],
                "total_duration_ms": 0.0,
                "alignment_offset_ms": 0.0,
                "priority_applied": False,
                "message": "No data provided"
            }

        # Calculate total duration for each modality
        modality_durations = {}
        for modality, data in modality_data.items():
            total_duration = sum(duration for _, duration in data)
            modality_durations[modality] = total_duration

        # Check for duration mismatches (timing alignment)
        max_duration = max(modality_durations.values())
        min_duration = min(modality_durations.values())
        duration_mismatch = max_duration - min_duration

        # Apply timing alignment if mismatch exceeds threshold
        alignment_offset = 0.0
        if duration_mismatch > self.config.max_drift_ms:
            logger.warning(
                f"Duration mismatch detected: {duration_mismatch:.2f}ms "
                f"(max: {self.config.max_drift_ms}ms)"
            )

            # Align to the longest duration
            alignment_offset = self._align_timing(modality_data, modality_durations, max_duration)

            logger.info(f"Applied timing alignment: offset={alignment_offset:.2f}ms")

        # Handle output priority (resource limitation)
        priority_applied = False
        if len(modality_data) > 2:  # More than 2 modalities
            # Check if we should apply priority filtering
            # (In a real system, this would check resource availability)
            if self._should_apply_priority():
                logger.info("Applying output priority filtering")
                modality_data = self._apply_priority_filter(modality_data, priority_order)
                priority_applied = True

        # Register streams and add chunks
        streams_registered = []
        for modality, data in modality_data.items():
            self.register_stream(modality)
            streams_registered.append(modality)

            # Add chunks to stream
            for chunk_data, duration in data:
                self.add_chunk(
                    modality=modality,
                    data=chunk_data,
                    duration_ms=duration
                )

            logger.info(
                f"Registered {modality.value} stream with {len(data)} chunks, "
                f"total duration: {modality_durations[modality]:.2f}ms"
            )

        # Calculate final synchronized duration
        final_duration = max(
            self.streams[m].total_duration_ms
            for m in streams_registered
        )

        return {
            "success": True,
            "streams_registered": [m.value for m in streams_registered],
            "total_duration_ms": final_duration,
            "alignment_offset_ms": alignment_offset,
            "priority_applied": priority_applied,
            "message": f"Synchronized {len(streams_registered)} modalities successfully"
        }

    def _align_timing(
        self,
        modality_data: Dict[OutputModality, List[tuple]],
        modality_durations: Dict[OutputModality, float],
        target_duration: float
    ) -> float:
        """
        Align timing across modalities to match target duration.

        Strategy: Pad shorter modalities with silence/pauses to match longest.

        Args:
            modality_data: Dictionary of modality data
            modality_durations: Dictionary of modality durations
            target_duration: Target duration to align to

        Returns:
            Maximum alignment offset applied
        """
        max_offset = 0.0

        for modality, duration in modality_durations.items():
            if duration < target_duration:
                # Calculate padding needed
                padding_needed = target_duration - duration

                # Add padding chunk at the end
                if modality == OutputModality.TEXT:
                    # Add empty text chunk
                    modality_data[modality].append(("", padding_needed))
                elif modality == OutputModality.SPEECH:
                    # Add silence
                    sample_rate = 22050
                    silence_samples = int((padding_needed / 1000.0) * sample_rate)
                    silence = np.zeros(silence_samples, dtype=np.float32)
                    modality_data[modality].append((silence, padding_needed))
                elif modality == OutputModality.VISUAL:
                    # Add blank frame
                    modality_data[modality].append((None, padding_needed))

                max_offset = max(max_offset, padding_needed)

                logger.debug(
                    f"Padded {modality.value} with {padding_needed:.2f}ms "
                    f"to reach {target_duration:.2f}ms"
                )

        return max_offset

    def _should_apply_priority(self) -> bool:
        """
        Determine if priority filtering should be applied.

        In a real system, this would check:
        - Available CPU/GPU resources
        - Network bandwidth
        - Memory usage
        - User preferences

        For now, returns False (allow all modalities).

        Returns:
            True if priority filtering should be applied
        """
        # Placeholder: In production, check actual resource constraints
        # For example:
        # - Check CPU usage > 80%
        # - Check memory usage > 1.5GB
        # - Check network latency > 200ms

        return False

    def _apply_priority_filter(
        self,
        modality_data: Dict[OutputModality, List[tuple]],
        priority_order: List[OutputModality]
    ) -> Dict[OutputModality, List[tuple]]:
        """
        Filter modalities based on priority order.

        Keeps only the top N modalities based on priority and resource constraints.

        Args:
            modality_data: Dictionary of modality data
            priority_order: List of modalities in priority order

        Returns:
            Filtered modality data
        """
        # Keep top 2 modalities by priority
        max_modalities = 2

        filtered_data = {}
        for modality in priority_order:
            if modality in modality_data:
                filtered_data[modality] = modality_data[modality]

                if len(filtered_data) >= max_modalities:
                    break

        # Log which modalities were dropped
        dropped = set(modality_data.keys()) - set(filtered_data.keys())
        if dropped:
            logger.warning(
                f"Dropped modalities due to priority: "
                f"{[m.value for m in dropped]}"
            )

        return filtered_data

