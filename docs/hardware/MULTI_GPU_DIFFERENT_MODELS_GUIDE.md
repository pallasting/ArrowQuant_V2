# Multi-GPU Setup: Different Models on Different GPUs

**Date**: 2026-02-15  
**Scenario**: Run different models on MI50 and W7000 simultaneously  
**Status**: Feasible with multiple Ollama instances

## Your GPU Configuration

### GPU 0: AMD Instinct MI50
- **VRAM**: 16 GB
- **Performance**: High (13.3 TFLOPS)
- **Current**: Running Qwen2.5-7B (4.9 GB)
- **Capacity**: Can run larger models (up to ~13 GB)

### GPU 1: AMD FirePro W7000
- **VRAM**: 4 GB
- **Performance**: Medium (2.4 TFLOPS, 5.5x slower than MI50)
- **Current**: Idle (display only)
- **Capacity**: Can run small models (up to ~3 GB)

## Strategy: Multiple Ollama Instances

### Concept

Run **two separate Ollama instances**:
- **Instance 1** (MI50): Large/medium models on port 11434
- **Instance 2** (W7000): Small models on port 11435

### Architecture

```
┌─────────────────────────────────────────┐
│         Application Layer               │
│  (Load balancer / Smart router)         │
└─────────────┬───────────────────────────┘
              │
      ┌───────┴────────┐
      │                │
┌─────▼─────┐    ┌────▼──────┐
│ Ollama 1  │    │ Ollama 2  │
│ Port 11434│    │ Port 11435│
│           │    │           │
│ MI50 GPU  │    │ W7000 GPU │
│ 16 GB     │    │ 4 GB      │
│           │    │           │
│ Qwen2.5   │    │ Qwen2.5   │
│ 7B (4.9GB)│    │ 1.5B (1GB)│
└───────────┘    └───────────┘
```

## Implementation Guide

### Step 1: Check Available Small Models

Models that fit in W7000's 4 GB VRAM:

| Model | Size | VRAM | Use Case |
|-------|------|------|----------|
| qwen2.5:0.5b | 0.5 GB | ~0.8 GB | Ultra-fast, simple tasks |
| qwen2.5:1.5b | 1 GB | ~1.3 GB | Fast, good quality |
| qwen2.5:3b | 2 GB | ~2.5 GB | Balanced |
| llama3.2:1b | 1 GB | ~1.3 GB | Fast, Meta model |
| llama3.2:3b | 2 GB | ~2.5 GB | Good quality |
| gemma2:2b | 1.5 GB | ~2 GB | Google model |

**Recommended for W7000**: qwen2.5:1.5b or qwen2.5:3b

### Step 2: Setup Second Ollama Instance

#### Option A: Systemd Service (Recommended)

Create second Ollama service for W7000:

```bash
# 1. Create service file
sudo tee /etc/systemd/system/ollama-w7000.service << 'EOF'
[Unit]
Description=Ollama Service (W7000 GPU)
After=network-online.target ollama.service

[Service]
ExecStart=/usr/local/bin/ollama serve
User=ollama
Group=ollama
Restart=always
RestartSec=3
Environment="OLLAMA_HOST=0.0.0.0:11435"
Environment="OLLAMA_MODELS=/usr/share/ollama/.ollama/models-w7000"
Environment="OLLAMA_VULKAN=1"
Environment="GGML_VK_VISIBLE_DEVICES=1"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

[Install]
WantedBy=default.target
EOF

# 2. Create separate model directory
sudo mkdir -p /usr/share/ollama/.ollama/models-w7000
sudo chown -R ollama:ollama /usr/share/ollama/.ollama/models-w7000

# 3. Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable ollama-w7000
sudo systemctl start ollama-w7000

# 4. Check status
sudo systemctl status ollama-w7000
```

#### Option B: Manual Start (Testing)

```bash
# Terminal 1: MI50 (already running on port 11434)
# No action needed

# Terminal 2: W7000 on port 11435
OLLAMA_HOST=0.0.0.0:11435 \
OLLAMA_VULKAN=1 \
GGML_VK_VISIBLE_DEVICES=1 \
ollama serve
```

### Step 3: Install Small Model on W7000

```bash
# Pull small model to W7000 instance
OLLAMA_HOST=localhost:11435 ollama pull qwen2.5:1.5b

# Or even smaller
OLLAMA_HOST=localhost:11435 ollama pull qwen2.5:0.5b

# Verify
OLLAMA_HOST=localhost:11435 ollama list
```

### Step 4: Test Both Instances

```bash
# Test MI50 (large model)
curl http://localhost:11434/api/generate -d '{
  "model": "qwen2.5:7b-instruct",
  "prompt": "Explain quantum computing",
  "stream": false
}'

# Test W7000 (small model)
curl http://localhost:11435/api/generate -d '{
  "model": "qwen2.5:1.5b",
  "prompt": "Hello, how are you?",
  "stream": false
}'
```

### Step 5: Update Application to Use Both

Update `llm_compression/llm_client.py` to support multiple endpoints:

```python
class MultiGPULLMClient:
    def __init__(self):
        self.mi50_client = LLMClient(endpoint="http://localhost:11434")
        self.w7000_client = LLMClient(endpoint="http://localhost:11435")
    
    def compress(self, text, quality="standard"):
        # Route based on text length or quality
        if len(text) < 500 or quality == "fast":
            # Use W7000 for short/fast requests
            return self.w7000_client.compress(text)
        else:
            # Use MI50 for long/quality requests
            return self.mi50_client.compress(text)
```

## Performance Expectations

### Single GPU (Current)

| GPU | Model | Inference | Throughput |
|-----|-------|-----------|------------|
| MI50 | Qwen2.5-7B | 1.54s | 48/min |
| W7000 | Idle | N/A | 0/min |
| **Total** | | | **48/min** |

### Dual GPU (Proposed)

| GPU | Model | Inference | Throughput | Use Case |
|-----|-------|-----------|------------|----------|
| MI50 | Qwen2.5-7B | 1.54s | 48/min | Long texts, high quality |
| W7000 | Qwen2.5-1.5B | 3-4s | 15-20/min | Short texts, fast response |
| **Total** | | | **63-68/min** | Combined |

**Gain**: 40% more throughput by utilizing W7000

### With Smart Routing

If 30% of requests are short (< 500 chars):
- 70% → MI50: 48 × 0.7 = 33.6/min
- 30% → W7000: 20 × 0.3 = 6/min
- **Total**: ~40/min effective

**Note**: Gain depends on workload distribution

## Use Case Examples

### Scenario 1: Quality vs Speed Tiers

```python
# High quality (MI50 - 7B model)
compress(text, quality="high")  # → localhost:11434

# Fast response (W7000 - 1.5B model)
compress(text, quality="fast")  # → localhost:11435
```

### Scenario 2: Length-Based Routing

```python
if len(text) < 500:
    # Short text → W7000 (faster, good enough)
    endpoint = "localhost:11435"
else:
    # Long text → MI50 (better quality)
    endpoint = "localhost:11434"
```

### Scenario 3: Different Models for Different Tasks

```python
# MI50: Compression (needs quality)
compress_endpoint = "localhost:11434"  # Qwen2.5-7B

# W7000: Classification (simpler task)
classify_endpoint = "localhost:11435"  # Qwen2.5-1.5B
```

## Advantages

1. **Utilizes W7000**: No longer idle
2. **More throughput**: 40% increase
3. **Flexibility**: Different models for different needs
4. **Load balancing**: Distribute work across GPUs
5. **Redundancy**: If one fails, other continues

## Disadvantages

1. **Complexity**: Two services to manage
2. **Memory overhead**: 2x model storage
3. **W7000 slower**: 5.5x slower than MI50
4. **Quality difference**: Small model = lower quality
5. **Routing logic**: Need smart load balancer

## Recommended Configuration

### For Your Use Case (Phase 1.1)

**Start simple**: Keep single GPU (MI50) for now

**Reasons**:
1. MI50 alone meets 4/5 criteria
2. Batching gives bigger gains (3-5x)
3. Simpler to maintain
4. W7000's 40% gain < batching's 300% gain

### When to Add W7000

Add W7000 when:
1. ✅ MI50 at capacity (> 80% utilization)
2. ✅ Need more concurrent requests
3. ✅ Have fast/slow request tiers
4. ✅ Want redundancy/failover

## Implementation Script

I can create a setup script if you want to try dual-GPU:

```bash
# scripts/setup_dual_gpu.sh
# - Creates second Ollama service
# - Installs small model on W7000
# - Tests both GPUs
# - Updates application config
```

Would you like me to create this script?

## Conclusion

### Vulkan vs ROCm Performance

**Difference**: 10-15% (ROCm would be faster)
**Impact**: Minimal (1.54s → 1.3s)
**Recommendation**: Keep Vulkan (stable, fast enough)

### Multi-GPU Different Models

**Feasible**: Yes, with multiple Ollama instances
**Gain**: 40% more throughput
**Complexity**: Medium (two services)
**Recommendation**: 
- **Now**: Keep single GPU (MI50)
- **Later**: Add W7000 when MI50 at capacity

### Priority Order

1. 🔥 **Batching** (3-5x gain, already implemented)
2. ⭐ **Single GPU optimization** (tune current setup)
3. ⭐ **Add W7000** (40% gain, medium effort)
4. ❄️ **Fix ROCm** (10-15% gain, high effort)

**Next step**: Focus on batching to reach 100/min throughput target, not multi-GPU.
