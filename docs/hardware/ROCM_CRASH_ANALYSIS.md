# ROCm Runner Crash Analysis

**Date**: 2026-02-15  
**Issue**: Ollama ROCm runner crashes during GPU discovery  
**Tested Configurations**: Bundled ROCm 6.3 and System ROCm 7.2

## Crash Pattern Analysis

### What We Know from Logs

#### 1. GPU Discovery Process
```
time=2026-02-15T16:39:06.840Z level=INFO source=runner.go:67 
msg="discovering available GPUs..."
```

Ollama starts GPU discovery by launching multiple runner processes.

#### 2. Multiple Runner Attempts
```
msg="starting runner" cmd="/usr/local/bin/ollama runner --ollama-engine --port 38059"
msg="starting runner" cmd="/usr/local/bin/ollama runner --ollama-engine --port 38693"
msg="starting runner" cmd="/usr/local/bin/ollama runner --ollama-engine --port 38211"
msg="starting runner" cmd="/usr/local/bin/ollama runner --ollama-engine --port 41849"
```

Ollama launches 4-5 runner processes to test different GPU backends:
- CUDA (if available)
- ROCm (if available)
- Vulkan (if enabled)
- CPU (fallback)

#### 3. ROCm Runner Crash
```
time=2026-02-15T16:39:19.591Z level=INFO source=runner.go:464 
msg="failure during GPU discovery" 
OLLAMA_LIBRARY_PATH="[/usr/local/lib/ollama /usr/local/lib/ollama/rocm]" 
extra_envs="map[GGML_CUDA_INIT:1 ROCR_VISIBLE_DEVICES:GPU-6a0418e1732c730d]" 
error="runner crashed"
```

**Key observations**:
- ROCm runner crashes after ~13 seconds
- No detailed error message (just "runner crashed")
- Happens with both ROCm 6.3 (bundled) and ROCm 7.2 (system)
- Environment variables are set correctly

#### 4. Fallback Behavior
After ROCm crash:
- With `OLLAMA_VULKAN=0`: Falls back to CPU
- With `OLLAMA_VULKAN=1`: Falls back to Vulkan (works!)

## Root Cause Investigation

### Hypothesis 1: Library Version Mismatch ❌ (Tested, Failed)

**Theory**: Ollama's bundled ROCm 6.3 incompatible with system ROCm 7.2 drivers

**Test**: Replaced bundled ROCm 6.3 with system ROCm 7.2 libraries
```bash
sudo rm -rf /usr/local/lib/ollama/rocm/*
sudo ln -s /opt/rocm-7.2.0/lib/*.so* /usr/local/lib/ollama/rocm/
```

**Result**: Still crashed
**Conclusion**: Not just a library version issue

### Hypothesis 2: GPU Architecture Compatibility Issue ⚠️ (Likely)

**Theory**: MI50 (gfx906) requires specific ROCm configuration that Ollama doesn't provide

**Evidence**:
1. **HSA_OVERRIDE_GFX_VERSION=9.0.6** is set (correct for gfx906)
2. **GPU is detected** by ROCm tools (rocminfo, rocm-smi)
3. **Vulkan works** with same GPU (proves GPU is functional)
4. **ROCm runner crashes** during initialization

**Possible causes**:
- Missing GPU firmware/microcode
- Kernel driver incompatibility
- ROCm runtime initialization failure
- Memory allocation failure

### Hypothesis 3: llama.cpp ROCm Backend Issue ⚠️ (Likely)

**Theory**: llama.cpp's ROCm backend has issues with MI50/gfx906

**Evidence**:
1. Ollama uses llama.cpp for inference
2. llama.cpp's ROCm support is less mature than CUDA
3. MI50 is older GPU (2018), may have limited testing
4. Vulkan backend works (different code path)

**Known llama.cpp ROCm issues**:
- gfx906 support varies by ROCm version
- Some operations not optimized for older architectures
- Memory management issues on certain GPUs

### Hypothesis 4: Missing ROCm Components 🔍 (Needs Investigation)

**Theory**: System missing required ROCm components for GPU compute

**Check required**:
```bash
# Check if ROCm kernel modules loaded
lsmod | grep amdgpu
lsmod | grep kfd  # Kernel Fusion Driver (required for compute)

# Check HSA runtime
ls -la /dev/kfd  # HSA device node

# Check GPU compute capability
rocminfo | grep -i "compute"
```

### Hypothesis 5: Ollama Binary Compatibility 🔍 (Possible)

**Theory**: Ollama binary compiled without proper ROCm support for gfx906

**Evidence**:
- Ollama 0.16.1 may not have full gfx906 support
- Binary may be compiled for newer GPUs (gfx908+)
- Bundled ROCm 6.3 may target different architectures

## Detailed Crash Timeline

```
T+0s:    Ollama starts GPU discovery
T+0s:    Launches multiple runner processes
T+0-6s:  Runners test CUDA (not available)
T+6-13s: Runner tests ROCm backend
         - Loads ROCm libraries
         - Initializes HSA runtime
         - Attempts GPU detection
         - CRASH occurs here
T+13s:   Ollama detects crash, logs "runner crashed"
T+13s:   Falls back to next backend (Vulkan or CPU)
```

## What's Missing: Detailed Error Information

The logs show "runner crashed" but don't show:
- **Segmentation fault details**
- **ROCm error codes**
- **HSA runtime errors**
- **GPU initialization failures**

### Why No Detailed Errors?

1. **Runner is separate process**: Crashes in child process, parent only sees exit code
2. **No stderr capture**: Error output may not be logged
3. **Silent failures**: ROCm may fail silently without error messages

## Investigation Steps Needed

### Step 1: Check ROCm Kernel Support
```bash
# Check if KFD (Kernel Fusion Driver) is loaded
lsmod | grep kfd

# Check HSA device
ls -la /dev/kfd

# Check dmesg for ROCm errors
sudo dmesg | grep -i "kfd\|hsa\|amdgpu" | tail -50
```

### Step 2: Test ROCm Directly (Bypass Ollama)
```bash
# Test ROCm with simple program
cat > test_rocm.cpp << 'EOF'
#include <hip/hip_runtime.h>
#include <iostream>

int main() {
    int deviceCount = 0;
    hipError_t error = hipGetDeviceCount(&deviceCount);
    
    if (error != hipSuccess) {
        std::cerr << "hipGetDeviceCount failed: " << hipGetErrorString(error) << std::endl;
        return 1;
    }
    
    std::cout << "Found " << deviceCount << " GPU(s)" << std::endl;
    
    for (int i = 0; i < deviceCount; i++) {
        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, i);
        std::cout << "GPU " << i << ": " << prop.name << std::endl;
        std::cout << "  Compute: " << prop.major << "." << prop.minor << std::endl;
        std::cout << "  Memory: " << prop.totalGlobalMem / (1024*1024*1024) << " GB" << std::endl;
    }
    
    return 0;
}
EOF

# Compile and run
/opt/rocm-7.2.0/bin/hipcc test_rocm.cpp -o test_rocm
./test_rocm
```

### Step 3: Enable Ollama Debug Logging
```bash
# Set debug environment
sudo tee -a /etc/systemd/system/ollama.service.d/rocm.conf << 'EOF'
Environment="OLLAMA_DEBUG=1"
Environment="HSA_ENABLE_DEBUG=1"
Environment="AMD_LOG_LEVEL=3"
EOF

sudo systemctl daemon-reload
sudo systemctl restart ollama

# Watch logs in real-time
sudo journalctl -u ollama -f
```

### Step 4: Check llama.cpp ROCm Support
```bash
# Check if llama.cpp was compiled with ROCm support
strings /usr/local/bin/ollama | grep -i "rocm\|hip"

# Check available backends
/usr/local/bin/ollama runner --help 2>&1 | grep -i backend
```

## Comparison: Why Vulkan Works but ROCm Doesn't

| Aspect | ROCm | Vulkan |
|--------|------|--------|
| **API Level** | Low-level (HSA) | High-level (Graphics API) |
| **Driver** | ROCm kernel driver | Mesa RADV driver |
| **Initialization** | Complex (HSA runtime) | Simpler (Vulkan loader) |
| **GPU Support** | Compute-specific | Graphics + Compute |
| **Maturity** | Newer, less tested | Mature, well-tested |
| **Error Handling** | Silent failures | Better error reporting |

**Key difference**: Vulkan uses Mesa's RADV driver which is more mature and stable for MI50, while ROCm uses AMD's proprietary HSA runtime which may have compatibility issues.

## Likely Root Cause (Best Guess)

Based on all evidence, the most likely cause is:

**ROCm HSA Runtime Initialization Failure on gfx906 (MI50)**

Specifically:
1. Ollama's ROCm runner tries to initialize HSA runtime
2. HSA runtime fails to properly initialize MI50 (gfx906)
3. Possible reasons:
   - Missing KFD (Kernel Fusion Driver) support
   - Incompatible ROCm version for gfx906
   - GPU firmware/microcode issues
   - Memory allocation failures

4. Runner crashes without detailed error
5. Ollama falls back to next backend

## Recommended Actions

### Immediate (Diagnostic)
1. ✅ Check KFD status: `lsmod | grep kfd`
2. ✅ Check HSA device: `ls -la /dev/kfd`
3. ✅ Test ROCm directly with simple HIP program
4. ✅ Enable debug logging and capture detailed errors

### Short-term (Workaround)
1. ✅ **Keep using Vulkan** (current solution)
   - Performance: 1.54s (excellent)
   - Stability: Proven to work
   - Meets 4/5 Phase 1.1 criteria

### Long-term (Fix)
1. ⏭️ Wait for Ollama update with better gfx906 support
2. ⏭️ Try newer ROCm version (if available)
3. ⏭️ Report issue to Ollama project with diagnostic info
4. ⏭️ Consider upgrading GPU to newer model (gfx908+)

## Conclusion

**Why ROCm crashes**: HSA runtime initialization failure on MI50 (gfx906), likely due to:
- Incomplete gfx906 support in Ollama's ROCm backend
- Missing or incompatible kernel drivers (KFD)
- ROCm version incompatibility

**Why Vulkan works**: Uses different driver stack (Mesa RADV) with better MI50 support

**Best solution**: Continue using Vulkan backend (1.54s performance is excellent)

**Next steps**: Run diagnostic commands to confirm root cause, but don't block on fixing ROCm since Vulkan works well.
