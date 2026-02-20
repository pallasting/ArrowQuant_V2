"""
Demo of the Synchronizer for multi-modal output coordination.

This example demonstrates:
1. Using synchronize() method for easy multi-modal coordination
2. Registering multiple output streams (text, audio)
3. Adding chunks to streams
4. Synchronized playback with timing alignment
5. Output priority handling
6. Pause/resume functionality
7. Statistics tracking

Requirements: 4.1, 4.3, 4.5, 4.6, 6.3
"""

import time
import numpy as np

from llm_compression.expression import (
    Synchronizer,
    SynchronizerConfig,
    OutputModality,
)


def demo_synchronize_method():
    """Demo the synchronize() method (Task 6.2)."""
    print("=" * 70)
    print("Demo 1: synchronize() Method - Easy Multi-modal Coordination")
    print("=" * 70)
    print()
    
    # Create synchronizer
    sync = Synchronizer()
    print("✓ Created synchronizer")
    print()
    
    # Prepare text data
    text_data = [
        ("Hello", 100.0),
        ("world", 100.0),
        ("!", 50.0)
    ]
    
    # Prepare speech data (simulated audio)
    speech_data = []
    for i in range(3):
        sample_rate = 22050
        duration_ms = 100.0 if i < 2 else 50.0
        samples = int(sample_rate * duration_ms / 1000)
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        speech_data.append((audio, duration_ms))
    
    print("Synchronizing text and speech outputs...")
    result = sync.synchronize(
        text_data=text_data,
        speech_data=speech_data
    )
    
    print(f"✓ Synchronization result:")
    print(f"  Success: {result['success']}")
    print(f"  Streams registered: {result['streams_registered']}")
    print(f"  Total duration: {result['total_duration_ms']}ms")
    print(f"  Alignment offset: {result['alignment_offset_ms']}ms")
    print(f"  Priority applied: {result['priority_applied']}")
    print(f"  Message: {result['message']}")
    print()
    
    # Start playback
    print("Starting playback...")
    sync.start()
    
    # Get a few chunks
    print("Playing synchronized chunks:")
    for _ in range(3):
        chunks = sync.get_next_chunks()
        for modality, chunk in chunks.items():
            if modality == OutputModality.TEXT:
                print(f"  [{chunk.timestamp_ms:6.1f}ms] TEXT:   '{chunk.data}'")
            elif modality == OutputModality.SPEECH:
                print(f"  [{chunk.timestamp_ms:6.1f}ms] SPEECH: {len(chunk.data)} samples")
        time.sleep(0.11)
    
    print()
    print("=" * 70)
    print()


def demo_timing_alignment():
    """Demo timing alignment when durations mismatch."""
    print("=" * 70)
    print("Demo 2: Timing Alignment - Handling Duration Mismatches")
    print("=" * 70)
    print()
    
    sync = Synchronizer()
    
    # Text is shorter than speech
    text_data = [
        ("Short", 100.0),
        ("text", 100.0)
    ]
    
    # Speech is longer
    speech_data = []
    for i in range(4):
        audio = np.random.randn(2205).astype(np.float32) * 0.1
        speech_data.append((audio, 100.0))
    
    print("Text duration: 200ms")
    print("Speech duration: 400ms")
    print("Mismatch: 200ms")
    print()
    
    print("Synchronizing with automatic alignment...")
    result = sync.synchronize(
        text_data=text_data,
        speech_data=speech_data
    )
    
    print(f"✓ Alignment applied: {result['alignment_offset_ms']}ms")
    print(f"✓ Final duration: {result['total_duration_ms']}ms")
    print()
    
    # Verify alignment
    text_stream = sync.streams[OutputModality.TEXT]
    speech_stream = sync.streams[OutputModality.SPEECH]
    
    print("Stream durations after alignment:")
    print(f"  TEXT:   {text_stream.total_duration_ms}ms")
    print(f"  SPEECH: {speech_stream.total_duration_ms}ms")
    print(f"  Difference: {abs(text_stream.total_duration_ms - speech_stream.total_duration_ms):.2f}ms")
    print()
    
    print("=" * 70)
    print()


def demo_priority_handling():
    """Demo output priority when resources are limited."""
    print("=" * 70)
    print("Demo 3: Output Priority - Resource-Limited Scenarios")
    print("=" * 70)
    print()
    
    sync = Synchronizer()
    
    # Mock resource limitation
    sync._should_apply_priority = lambda: True
    
    # Prepare all three modalities
    text_data = [("Hello", 100.0)]
    speech_data = [(np.random.randn(2205).astype(np.float32), 100.0)]
    visual_data = [("image_data", 100.0)]
    
    print("Attempting to synchronize 3 modalities with limited resources...")
    print("Priority order: TEXT > SPEECH > VISUAL")
    print()
    
    result = sync.synchronize(
        text_data=text_data,
        speech_data=speech_data,
        visual_data=visual_data
    )
    
    print(f"✓ Priority filtering applied: {result['priority_applied']}")
    print(f"✓ Streams registered: {result['streams_registered']}")
    print(f"  (VISUAL was dropped due to priority)")
    print()
    
    print("=" * 70)
    print()


def demo_manual_workflow():
    """Demo manual workflow (original approach)."""
    print("=" * 70)
    print("Demo 4: Manual Workflow - Traditional Approach")
    print("=" * 70)
    print()
    
    # Create synchronizer with custom config
    config = SynchronizerConfig(
        buffer_size_ms=150.0,
        max_drift_ms=50.0,
        sync_interval_ms=25.0,
        enable_auto_sync=True
    )
    
    sync = Synchronizer(config=config)
    print(f"✓ Created synchronizer with {config.buffer_size_ms}ms buffer")
    print()
    
    # Register streams
    print("Registering output streams...")
    sync.register_stream(OutputModality.TEXT)
    sync.register_stream(OutputModality.SPEECH)
    print(f"✓ Registered TEXT stream")
    print(f"✓ Registered SPEECH stream")
    print()
    
    # Add text chunks
    print("Adding text chunks...")
    text_chunks = [
        ("Hello", 100.0),
        ("world", 100.0),
        ("from", 100.0),
        ("the", 100.0),
        ("synchronizer", 150.0),
    ]
    
    for i, (text, duration) in enumerate(text_chunks):
        sync.add_chunk(
            modality=OutputModality.TEXT,
            data=text,
            duration_ms=duration,
            is_final=(i == len(text_chunks) - 1)
        )
        print(f"  Added: '{text}' ({duration}ms)")
    print()
    
    # Add audio chunks (simulated)
    print("Adding audio chunks...")
    for i in range(5):
        # Simulate audio data (random noise)
        sample_rate = 22050
        duration_ms = 100.0 if i < 4 else 150.0
        samples = int(sample_rate * duration_ms / 1000)
        audio_data = np.random.randn(samples).astype(np.float32) * 0.1
        
        sync.add_chunk(
            modality=OutputModality.SPEECH,
            data=audio_data,
            duration_ms=duration_ms,
            is_final=(i == 4)
        )
        print(f"  Added: audio chunk {i} ({duration_ms}ms, {samples} samples)")
    print()
    
    # Check buffering
    print("Checking buffer status...")
    if sync.is_buffering_complete():
        print("✓ Buffering complete - ready to play")
    else:
        print("⚠ Buffering incomplete - need more data")
    print()
    
    # Start playback
    print("Starting synchronized playback...")
    sync.start()
    print("✓ Playback started")
    print()
    
    # Simulate playback
    print("Playing synchronized chunks:")
    print("-" * 70)
    
    chunk_count = 0
    while True:
        # Get next synchronized chunks
        chunks = sync.get_next_chunks()
        
        if not chunks:
            # Check if all streams completed
            text_state = sync.get_stream_state(OutputModality.TEXT)
            speech_state = sync.get_stream_state(OutputModality.SPEECH)
            
            if text_state.value == "completed" and speech_state.value == "completed":
                print("✓ All streams completed")
                break
            
            # Wait a bit and try again
            time.sleep(0.05)
            continue
        
        # Display chunks
        for modality, chunk in chunks.items():
            if modality == OutputModality.TEXT:
                print(f"  [{chunk.timestamp_ms:6.1f}ms] TEXT:   '{chunk.data}'")
            elif modality == OutputModality.SPEECH:
                print(f"  [{chunk.timestamp_ms:6.1f}ms] SPEECH: {len(chunk.data)} samples")
        
        chunk_count += 1
        
        # Simulate playback delay
        time.sleep(0.11)  # Slightly more than chunk duration
        
        # Demo pause/resume after 2 chunks
        if chunk_count == 2:
            print()
            print("⏸ Pausing playback...")
            sync.pause()
            time.sleep(0.3)  # Pause for 300ms
            print("▶ Resuming playback...")
            sync.resume()
            print()
    
    print("-" * 70)
    print()
    
    # Show statistics
    print("Synchronization Statistics:")
    stats = sync.get_statistics()
    print(f"  Total chunks processed: {stats['total_chunks_processed']}")
    print(f"  Sync corrections: {stats['sync_corrections']}")
    print(f"  Buffer underruns: {stats['buffer_underruns']}")
    print(f"  Max drift observed: {stats['max_drift_observed_ms']:.2f}ms")
    print(f"  Is synchronized: {stats['is_synchronized']}")
    print(f"  Active streams: {stats['active_streams']}")
    print(f"  Completed streams: {stats['completed_streams']}")
    print()
    
    # Demo fallback
    print("Demonstrating fallback to text-only...")
    sync.stop()
    
    # Add new chunks
    sync.add_chunk(
        modality=OutputModality.TEXT,
        data="Fallback mode",
        duration_ms=100.0
    )
    sync.add_chunk(
        modality=OutputModality.SPEECH,
        data=np.random.randn(2205).astype(np.float32),
        duration_ms=100.0
    )
    
    sync.start()
    sync.fallback_to_text_only()
    
    print("✓ Fallback activated - only text stream active")
    print(f"  TEXT state: {sync.get_stream_state(OutputModality.TEXT).value}")
    print(f"  SPEECH state: {sync.get_stream_state(OutputModality.SPEECH).value}")
    print()
    
    print("=" * 70)
    print()


def main():
    """Run all synchronizer demos."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SYNCHRONIZER DEMO SUITE" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
def main():
    """Run all synchronizer demos."""
    print()
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "SYNCHRONIZER DEMO SUITE" + " " * 30 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Demo 1: synchronize() method (Task 6.2)
    demo_synchronize_method()
    
    # Demo 2: Timing alignment
    demo_timing_alignment()
    
    # Demo 3: Priority handling
    demo_priority_handling()
    
    # Demo 4: Manual workflow
    demo_manual_workflow()
    
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 22 + "ALL DEMOS COMPLETE!" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")
    print()


if __name__ == "__main__":
    main()
