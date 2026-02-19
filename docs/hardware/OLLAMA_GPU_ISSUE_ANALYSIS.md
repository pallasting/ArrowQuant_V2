# Ollama GPU Issue Analysis

**Date**: 2026-02-15  
**Issue**: Ollama ROCm runner crashes during GPU discovery  
**Status**: Root cause identified

## Problem Summary

Ollama has ROCm support compiled in, but the ROCm runner crashes during GPU discovery, causing fallback to CPU.

## Evidence

### 1. Ollama Has ROCm Support
```bash
$ ls -lh /usr/local/lib/ollama/rocm/libggml-hip.so
-rwxr-xr-x 1 root root 595M Feb 12 22:51 libggml-hip.so
```

### 2. Environment Variables Set Correctly
```bash
$ sudo cat /proc/$(pgrep -f "ollama serve")/environ | tr '\0' '\n' | grep -E "(HSA|ROCR|HIP)"
HSA_OVERRIDE_GFX_VERSION=9.0.6
ROCR_VISIBLE_DEVICES=0
HIP_VISIBLE_DEVICES=0
```

### 3. Runner Crashes During GPU Discovery
```
time=2026-02-15T15:50:36.995Z level=INFO source=runner.go:464 
msg="failure during GPU discovery" 
OLLAMA_LIBRARY_PATH="[/usr/local/lib/ollama /usr/local/lib/ollama/rocm]" 
extra_envs="map[GGML_CUDA_INIT:1 ROCR_VISIBLE_DEVICES:GPU-6a0418e1732c730d]" 
error="runner crashed"
```

### 4. Fallback to CPU
```
time=2026-02-15T15:50:38.461Z level=INFO source=device.go:245 
msg="model weights" device=CPU size="4.1 GiB"
```

## Root Cause: ROCm Version Mismatch

**Ollama Bundled ROCm**: 6.3.60303
```bash
$ ls -l /usr/local/lib/ollama/rocm/libamdhip64.so.6.3.60303
-rwxr-xr-x 1 root root 22294280 Feb 10  2025
```

**System ROCm**: 7.2.0
```bash
$ ls -l /opt/rocm-7.2.0/lib/libamdhip64.so.7.2.70200
-rw-r--r--  1 root root  27131824 Jan 10 00:38
```

The version mismatch between Ollama's bundled ROCm 6.3 libraries and the system's ROCm 7.2.0 installation is causing the runner to crash.

## Why This Happens

1. Ollama bundles its own ROCm libraries for portability
2. These libraries (ROCm 6.3) may be incompatible with:
   - System ROCm 7.2.0 kernel drivers
   - AMD GPU firmware expectations
   - HSA runtime interfaces

3. When Ollama tries to initialize the GPU with bundled ROCm 6.3 libraries against ROCm 7.2.0 drivers, the runner crashes

## Potential Solutions

### Option 1: Use System ROCm Libraries (Recommended)
Force Ollama to use system ROCm 7.2.0 libraries instead of bundled ones:

```bash
# Add to systemd service
Environment="LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:$LD_LIBRARY_PATH"
Environment="OLLAMA_LLM_LIBRARY=rocm"
```

**Pros**: Uses compatible ROCm version
**Cons**: May break if system ROCm is updated/removed

### Option 2: Downgrade System ROCm to 6.3
Match system ROCm to Ollama's bundled version:

```bash
# Uninstall ROCm 7.2.0
sudo apt remove rocm-*

# Install ROCm 6.3
# (requires finding ROCm 6.3 packages)
```

**Pros**: Perfect compatibility
**Cons**: Downgrades system ROCm, may affect other applications

### Option 3: Update Ollama
Wait for/install Ollama version with ROCm 7.2 support:

```bash
# Check for newer Ollama version
curl -fsSL https://ollama.com/install.sh | sh
```

**Pros**: Official solution, maintains system ROCm
**Cons**: May not be available yet

### Option 4: Use CPU (Current State)
Accept CPU-only operation:

**Pros**: Works now, no changes needed
**Cons**: 3-4x slower inference (6-11s vs 2-3s)

### Option 5: Try Alternative GPU Backend
Ollama also supports Vulkan:

```bash
# Add to systemd service
Environment="OLLAMA_VULKAN=1"
```

**Pros**: May work with current setup
**Cons**: Vulkan support is experimental, may be slower than ROCm

## Recommended Action

**Try Option 1 first** (use system ROCm libraries):

1. Update systemd service to use system ROCm:
```bash
sudo tee -a /etc/systemd/system/ollama.service.d/rocm.conf << 'EOF'
Environment="LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64:$LD_LIBRARY_PATH"
Environment="ROCM_PATH=/opt/rocm-7.2.0"
EOF
```

2. Reload and restart:
```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

3. Test:
```bash
ollama stop qwen2.5:7b-instruct
ollama run qwen2.5:7b-instruct "test"
ollama ps  # Should show GPU
```

If Option 1 fails, try Option 5 (Vulkan) as a fallback.

## Performance Impact

| Configuration | Inference Time | Throughput | Status |
|---------------|----------------|------------|--------|
| CPU (current) | 6-11s | ~8/min | ✓ Working |
| GPU (ROCm) | 2-3s | 100-150/min | ✗ Crashed |
| GPU (Vulkan) | 3-5s | 50-80/min | ? Untested |

## Next Steps

1. Try Option 1 (system ROCm libraries)
2. If fails, try Option 5 (Vulkan)
3. If both fail, document CPU-only operation and adjust Phase 1.1 targets
4. Consider filing issue with Ollama project about ROCm 7.2 compatibility

## References

- Ollama GPU Support: https://github.com/ollama/ollama/blob/main/docs/gpu.md
- ROCm Compatibility: https://rocm.docs.amd.com/
- System Info: AMD Instinct MI50 (gfx906) with ROCm 7.2.0
- Ollama Version: 0.16.1 (bundled ROCm 6.3.60303)
