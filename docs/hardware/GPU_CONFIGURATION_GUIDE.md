# Ollama GPU Configuration Guide

## Current Status
- **Problem**: Ollama is using CPU instead of GPU (AMD Mi50)
- **Impact**: Inference time is 6-11s per request (should be 2-3s on GPU)
- **Evidence**: `ollama ps` shows "100% CPU", `rocm-smi` shows 0% GPU usage

## Root Cause Analysis

The AMD Mi50 GPU (gfx906) is available and ROCm 7.2.0 is installed, but Ollama is not configured to use it.

### GPU Information
```bash
$ rocminfo | grep "Marketing Name"
Marketing Name:          AMD Instinct MI50/MI60

$ rocm-smi --showuse
GPU[0]          : GPU use (%): 0  # ← GPU not being used
```

### Ollama Status
```bash
$ ollama ps
NAME                   PROCESSOR    
qwen2.5:7b-instruct    100% CPU     # ← Should show GPU!
```

## Solution: Configure Ollama for ROCm/GPU

### Option 1: Environment Variables (Recommended)

Ollama needs specific ROCm environment variables to use the GPU:

1. **Stop Ollama** (requires sudo):
   ```bash
   sudo pkill ollama
   ```

2. **Set ROCm environment variables**:
   ```bash
   export HSA_OVERRIDE_GFX_VERSION=9.0.6  # For gfx906 (MI50)
   export ROCR_VISIBLE_DEVICES=0          # Use GPU 0
   export HIP_VISIBLE_DEVICES=0           # Use GPU 0
   ```

3. **Start Ollama with GPU support** (requires sudo):
   ```bash
   sudo -E HSA_OVERRIDE_GFX_VERSION=9.0.6 \
        ROCR_VISIBLE_DEVICES=0 \
        HIP_VISIBLE_DEVICES=0 \
        /usr/local/bin/ollama serve &
   ```

4. **Reload the model**:
   ```bash
   ollama stop qwen2.5:7b-instruct
   ollama run qwen2.5:7b-instruct "test"
   ```

5. **Verify GPU usage**:
   ```bash
   ollama ps  # Should show "GPU" instead of "CPU"
   rocm-smi --showuse  # Should show GPU usage > 0%
   ```

### Option 2: Systemd Service (Persistent)

If Ollama runs as a systemd service, configure it permanently:

1. **Create systemd override**:
   ```bash
   sudo mkdir -p /etc/systemd/system/ollama.service.d/
   sudo tee /etc/systemd/system/ollama.service.d/override.conf << EOF
   [Service]
   Environment="HSA_OVERRIDE_GFX_VERSION=9.0.6"
   Environment="ROCR_VISIBLE_DEVICES=0"
   Environment="HIP_VISIBLE_DEVICES=0"
   EOF
   ```

2. **Reload and restart**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl restart ollama
   ```

### Option 3: User Service (No Sudo)

If you don't have sudo access, run Ollama as a user service:

1. **Stop existing Ollama** (if you have permission):
   ```bash
   pkill ollama  # May fail without sudo
   ```

2. **Start Ollama with environment variables**:
   ```bash
   HSA_OVERRIDE_GFX_VERSION=9.0.6 \
   ROCR_VISIBLE_DEVICES=0 \
   HIP_VISIBLE_DEVICES=0 \
   ollama serve &
   ```

## Verification Steps

After configuration, verify GPU is working:

### 1. Check Ollama Process
```bash
ollama ps
```
Expected output:
```
NAME                   PROCESSOR    
qwen2.5:7b-instruct    100% GPU     # ← Should show GPU!
```

### 2. Check GPU Usage
```bash
rocm-smi --showuse
```
Expected output:
```
GPU[0]          : GPU use (%): 50-90  # ← Should show activity
```

### 3. Test Inference Speed
```bash
time ollama run qwen2.5:7b-instruct "Summarize: test message"
```
Expected: 2-3 seconds (vs 6-11 seconds on CPU)

### 4. Run Acceptance Tests
```bash
python scripts/phase_1_1_final_acceptance.py
```
Expected results:
- Compression latency: < 2s (vs current 8-16s)
- Throughput: > 100/min (vs current ~8/min)

## Expected Performance Improvements

| Metric | CPU (Current) | GPU (Expected) | Improvement |
|--------|---------------|----------------|-------------|
| Inference time | 6-11s | 2-3s | 3-4x faster |
| Compression latency | 8-16s | 2-4s | 4x faster |
| Throughput | ~8/min | 100-150/min | 12-18x faster |

## Troubleshooting

### GPU Still Not Used

1. **Check ROCm installation**:
   ```bash
   rocminfo | grep "Marketing Name"
   ```

2. **Check GPU visibility**:
   ```bash
   rocm-smi
   ```

3. **Check Ollama logs**:
   ```bash
   journalctl -u ollama -f  # If systemd service
   # OR
   ps aux | grep ollama  # Check process
   ```

4. **Verify model supports GPU**:
   ```bash
   ollama show qwen2.5:7b-instruct
   ```

### Model Still Slow

1. **Check GPU memory**:
   ```bash
   rocm-smi --showmeminfo vram
   ```

2. **Try smaller context**:
   ```bash
   ollama run qwen2.5:7b-instruct --ctx-size 2048
   ```

3. **Check quantization**:
   - Q4_K_M (current): Good balance
   - Q8_0: Higher quality, slower
   - Q4_0: Faster, lower quality

## Next Steps

1. **Configure GPU** using Option 1 above
2. **Verify GPU usage** with verification steps
3. **Re-run acceptance tests**: `python scripts/phase_1_1_final_acceptance.py`
4. **Update Phase 1.1 report** with final GPU-accelerated results

## References

- [Ollama GPU Support](https://github.com/ollama/ollama/blob/main/docs/gpu.md)
- [ROCm Documentation](https://rocm.docs.amd.com/)
- [AMD MI50 Specifications](https://www.amd.com/en/products/server-accelerators/instinct-mi50)
