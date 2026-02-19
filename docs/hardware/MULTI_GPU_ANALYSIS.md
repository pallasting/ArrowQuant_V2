# Multi-GPU Configuration Analysis

**Date**: 2026-02-15  
**System**: 2x AMD GPUs (different models)

## Current GPU Configuration

### GPU 0: AMD Instinct MI50/MI60 (Compute GPU)
- **Model**: Vega 20 (gfx906)
- **VRAM**: 16 GB HBM2
- **PCI**: 3d:00.0 (Display controller)
- **Purpose**: High-performance compute
- **Status**: Currently used by Ollama (Vulkan)
- **Vulkan**: Supported (RADV VEGA20)

### GPU 1: AMD FirePro W7000 (Display GPU)
- **Model**: Pitcairn XT (GCN 1.0)
- **VRAM**: ~4 GB GDDR5
- **PCI**: af:00.0 (VGA compatible controller)
- **Purpose**: Display output
- **Status**: Not used by Ollama
- **Vulkan**: Likely supported but not detected

## Multi-GPU Parallel Processing

### Question: Can Vulkan use both GPUs in parallel?

**Short Answer**: Yes, but with significant limitations for LLM inference.

### Vulkan Multi-GPU Capabilities

Vulkan supports several multi-GPU modes:

#### 1. **Device Groups** (Explicit Multi-GPU)
- Application explicitly manages multiple GPUs
- Can distribute work across GPUs
- Requires application support
- **Ollama Support**: ❌ Not implemented

#### 2. **Implicit Multi-GPU** (SLI/CrossFire-like)
- Driver automatically distributes work
- Transparent to application
- **Vulkan Support**: Limited (not like gaming SLI)
- **Ollama Support**: ❌ Not available

#### 3. **Separate Workloads**
- Different models on different GPUs
- Manual load balancing
- **Ollama Support**: ⚠️ Possible with multiple instances

### Why Multi-GPU is Challenging for LLM Inference

#### Problem 1: Model Size vs GPU Memory
```
MI50: 16 GB VRAM
W7000: ~4 GB VRAM
Qwen2.5-7B: 4.9 GB loaded

✓ MI50 can hold entire model
✗ W7000 too small for full model
```

#### Problem 2: Sequential Nature of LLM Inference
- LLM inference is largely sequential (token-by-token)
- Each token depends on previous tokens
- Hard to parallelize across GPUs
- Multi-GPU helps more with:
  - Training (batch parallelism)
  - Serving multiple requests (request parallelism)
  - Very large models (model parallelism)

#### Problem 3: GPU Performance Mismatch
```
MI50 (Vega 20):
- Compute: 13.3 TFLOPS (FP32)
- Memory: 1024 GB/s bandwidth
- Architecture: GCN 5.0 (2018)

W7000 (Pitcairn):
- Compute: 2.4 TFLOPS (FP32)
- Memory: 154 GB/s bandwidth
- Architecture: GCN 1.0 (2012)

Performance ratio: MI50 is ~5.5x faster
```

**Result**: W7000 would be a bottleneck, not a boost.

## Practical Multi-GPU Strategies for Ollama

### Strategy 1: Single GPU (Current - Recommended)
**Use MI50 only**

```bash
# Current Vulkan config
Environment="OLLAMA_VULKAN=1"
# Implicitly uses GPU 0 (MI50)
```

**Pros**:
- ✅ Simple configuration
- ✅ Best single-GPU performance
- ✅ No synchronization overhead
- ✅ Currently working (1.54s inference)

**Cons**:
- ❌ W7000 sits idle

**Performance**: 1.54s per inference

---

### Strategy 2: Multiple Ollama Instances (Load Balancing)
**Run separate Ollama instances on each GPU**

```bash
# Instance 1 on MI50 (port 11434)
OLLAMA_VULKAN=1 GGML_VK_VISIBLE_DEVICES=0 ollama serve

# Instance 2 on W7000 (port 11435)
OLLAMA_VULKAN=1 GGML_VK_VISIBLE_DEVICES=1 OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

**Pros**:
- ✅ Both GPUs utilized
- ✅ Can serve more concurrent requests
- ✅ Load balancing across GPUs

**Cons**:
- ❌ W7000 much slower (~5x)
- ❌ Requires 2x model memory (4.9 GB × 2)
- ❌ Complex load balancing needed
- ❌ W7000 may not have enough VRAM

**Performance**: 
- MI50: 1.54s per inference
- W7000: ~7-8s per inference (estimated)
- Combined throughput: ~50/min (not much better)

---

### Strategy 3: Dedicated Roles
**MI50 for inference, W7000 for display only**

```bash
# Keep current config
# MI50 handles all Ollama workload
# W7000 handles display output
```

**Pros**:
- ✅ Clean separation of duties
- ✅ MI50 not burdened by display
- ✅ Simple and stable

**Cons**:
- ❌ W7000 compute power unused

**Performance**: Same as Strategy 1 (1.54s)

---

### Strategy 4: Model Parallelism (Not Supported)
**Split model across GPUs**

**Status**: ❌ Not supported by Ollama/llama.cpp

Requires:
- Framework support (like DeepSpeed, Megatron)
- High-speed GPU interconnect (NVLink, Infinity Fabric)
- Complex tensor sharding

**Not feasible** for current setup.

---

## Recommendation

### Primary Recommendation: **Strategy 1** (Current Setup)

**Keep using MI50 only via Vulkan**

**Reasoning**:
1. **Performance**: MI50 is 5.5x faster than W7000
2. **Simplicity**: Single GPU = no synchronization overhead
3. **Memory**: W7000's 4 GB may not fit model + context
4. **Current Results**: Already achieving 1.54s (excellent)
5. **Diminishing Returns**: Multi-GPU won't significantly improve single-request latency

### When Multi-GPU Would Help

Multi-GPU (Strategy 2) would be beneficial if:
1. **High concurrent load**: Serving 10+ simultaneous requests
2. **Multiple models**: Different models on different GPUs
3. **Batch processing**: Processing many texts in parallel

**For current use case** (Phase 1.1 acceptance):
- Single requests: MI50 alone is optimal
- Batch processing: Already fast enough with MI50

### W7000 Alternative Uses

Since W7000 is underutilized, consider:
1. **Display only**: Current role (good)
2. **Development/testing**: Test different models
3. **Backup**: Fallback if MI50 fails
4. **Other workloads**: Non-Ollama GPU tasks

---

## Vulkan Multi-GPU Configuration (If Needed)

### Check Available Vulkan Devices
```bash
vulkaninfo --summary | grep -A 10 "GPU"
```

Current output:
```
GPU0: AMD Instinct MI50/MI60 (RADV VEGA20)  ← Used by Ollama
GPU1: llvmpipe (CPU fallback)                ← Not a real GPU
```

**Note**: W7000 not showing up in Vulkan devices. Possible reasons:
1. Driver not loaded for W7000
2. W7000 being used for display (X11/Wayland)
3. Vulkan not enabled for W7000

### Enable W7000 for Vulkan (If Desired)
```bash
# Check if W7000 has Vulkan support
lspci -v -s af:00.0 | grep -i driver

# May need to configure X11 to free W7000 for compute
# This is complex and may break display
```

### Configure Ollama for Specific GPU
```bash
# Use only MI50 (GPU 0) - Current
Environment="GGML_VK_VISIBLE_DEVICES=0"

# Use only W7000 (if available as GPU 1)
Environment="GGML_VK_VISIBLE_DEVICES=1"

# Use both (not recommended - Ollama doesn't support)
Environment="GGML_VK_VISIBLE_DEVICES=0,1"
```

---

## Performance Comparison

| Strategy | Latency | Throughput | Complexity | Recommended |
|----------|---------|------------|------------|-------------|
| 1. MI50 only | 1.54s | 48/min | Low | ✅ Yes |
| 2. Both GPUs | 1.54s (MI50)<br>7-8s (W7000) | ~55/min | High | ❌ No |
| 3. Dedicated roles | 1.54s | 48/min | Low | ✅ Yes |
| 4. Model parallel | N/A | N/A | Very High | ❌ Not supported |

---

## Conclusion

### Multi-GPU Answer

**Can Vulkan use both GPUs in parallel?**
- **Technically**: Yes, Vulkan supports multi-GPU
- **In Ollama**: No, not implemented
- **For your setup**: Not beneficial

**Reasons**:
1. W7000 is 5.5x slower than MI50
2. W7000 has insufficient VRAM (4 GB vs 16 GB)
3. LLM inference doesn't parallelize well across GPUs
4. Current single-GPU performance is already excellent (1.54s)

### Recommendation

**Keep current configuration** (MI50 only via Vulkan):
- Best performance for single requests
- Simplest configuration
- Already meeting 4/5 Phase 1.1 criteria
- W7000 serves its purpose as display GPU

### When to Revisit Multi-GPU

Consider multi-GPU if:
1. Serving 20+ concurrent users
2. Running multiple different models
3. Need higher aggregate throughput (not latency)
4. Upgrade W7000 to another MI50/MI60

For now, focus on **batch processing** (already implemented) to improve throughput, not multi-GPU.

---

## Next Steps

1. ✅ Proceed with Option B (upgrade Ollama's ROCm to 7.2)
2. ✅ Keep using MI50 only (current Vulkan config)
3. ⏭️ Consider multi-GPU only if concurrent load increases significantly

Would you like me to proceed with Option B testing now?
