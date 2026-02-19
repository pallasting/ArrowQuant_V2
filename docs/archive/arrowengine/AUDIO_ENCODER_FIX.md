# Audio Encoder Position Embedding Fix

## Problem

AudioEncoder was failing with `IndexError: index out of range in self` when encoding audio.

**Root Cause**: 
- 3-second audio (48,000 samples @ 16kHz) → mel-spectrogram with ~3,001 frames
- After Conv2 (stride=2): 1,501 frames
- Position embedding only supports max_positions=1,500
- Index out of range error when trying to access position 1,500+

## Solution

### 1. AudioInferenceCore - Add Sequence Truncation
Added safety check in `forward()` to truncate sequences exceeding max_positions:

```python
if seq_len > self.max_positions:
    logger.warning(f"Sequence length {seq_len} exceeds max_positions {self.max_positions}. Truncating to {self.max_positions}.")
    x = x[:, :self.max_positions, :]
    seq_len = self.max_positions
```

### 2. AudioProcessor - Limit Mel-Spectrogram Frames
Set `max_mel_frames=3000` to ensure after Conv2 (stride=2) we get exactly 1,500 frames:

```python
# Default to 3000 frames (will be 1500 after conv2 with stride=2)
self.max_mel_frames = 3000
```

### 3. MelSpectrogramProcessor - Add Frame Limiting
Added `max_frames` parameter to truncate/pad mel-spectrograms:

```python
def compute_mel_spectrogram(self, waveform: np.ndarray, max_frames: Optional[int] = None) -> np.ndarray:
    # ... compute mel-spectrogram ...
    
    # Truncate or pad to max_frames if specified
    if max_frames is not None:
        if n_frames > max_frames:
            mel_spec = mel_spec[:, :max_frames]
        elif n_frames < max_frames:
            pad_width = max_frames - n_frames
            mel_spec = np.pad(mel_spec, ((0, 0), (0, pad_width)), mode='constant')
```

## Test Results

Both encoders now pass all tests:

```
✅ VisionEncoder test PASSED
  - Load time: ~1.3s
  - Single image encoding: (512,) float32, L2 norm = 1.0
  - Batch encoding: (4, 512)

✅ AudioEncoder test PASSED
  - Load time: ~0.2s
  - Single audio encoding: (512,) float32, L2 norm = 1.0
  - Batch encoding: (4, 512)
  - No warnings or errors
```

## Files Modified

- `llm_compression/inference/audio_core.py` - Added sequence truncation
- `llm_compression/multimodal/audio_processor.py` - Added max_mel_frames parameter
- `llm_compression/multimodal/audio_processor.py` - Added max_frames to MelSpectrogramProcessor

## Task Status

- ✅ Task 3.1: Implement MelSpectrogramProcessor
- ✅ Task 3.2: Implement AudioEncoder class
- ✅ Task 3.3: Implement weight loading for audio encoder
- ✅ Task 3: Implement Audio Encoder (Whisper)

## Next Steps

Proceed to Task 4: Checkpoint - Ensure basic encoders work
