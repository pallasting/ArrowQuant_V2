#!/bin/bash
# Configure Ollama to use AMD GPU (ROCm)

echo "=== Ollama GPU Configuration Script ==="
echo ""

# Check current Ollama status
echo "1. Current Ollama Status:"
ollama ps
echo ""

# Check GPU availability
echo "2. GPU Information:"
rocm-smi --showuse
echo ""

# Check ROCm environment
echo "3. ROCm Environment:"
echo "HSA_OVERRIDE_GFX_VERSION: ${HSA_OVERRIDE_GFX_VERSION:-not set}"
echo "ROCR_VISIBLE_DEVICES: ${ROCR_VISIBLE_DEVICES:-not set}"
echo "HIP_VISIBLE_DEVICES: ${HIP_VISIBLE_DEVICES:-not set}"
echo ""

# Stop Ollama (requires sudo)
echo "4. Stopping Ollama service..."
echo "   Note: This requires sudo access. You may need to enter your password."
sudo pkill -9 ollama
sleep 2

# Set ROCm environment variables and restart Ollama
echo "5. Starting Ollama with GPU support..."
echo "   Setting environment variables:"
echo "   - HSA_OVERRIDE_GFX_VERSION=9.0.6 (for gfx906/MI50)"
echo "   - ROCR_VISIBLE_DEVICES=0"
echo ""

# Start Ollama with ROCm environment
sudo -E HSA_OVERRIDE_GFX_VERSION=9.0.6 ROCR_VISIBLE_DEVICES=0 /usr/local/bin/ollama serve &
OLLAMA_PID=$!

echo "   Ollama started with PID: $OLLAMA_PID"
echo "   Waiting 5 seconds for service to initialize..."
sleep 5

# Verify GPU is being used
echo ""
echo "6. Verifying GPU Configuration:"
echo "   Loading model to check GPU usage..."
ollama run qwen2.5:7b-instruct "Hello" --verbose 2>&1 | head -20 &
sleep 3

echo ""
echo "7. Checking GPU usage:"
rocm-smi --showuse
echo ""

echo "8. Checking Ollama process status:"
ollama ps
echo ""

echo "=== Configuration Complete ==="
echo ""
echo "If you see 'GPU' in the PROCESSOR column above, GPU is working!"
echo "If you still see 'CPU', try the following:"
echo "  1. Check that ROCm is properly installed: rocminfo"
echo "  2. Verify GPU is visible: rocm-smi"
echo "  3. Check Ollama logs: journalctl -u ollama -f"
echo "  4. Try unloading and reloading the model:"
echo "     ollama stop qwen2.5:7b-instruct"
echo "     ollama run qwen2.5:7b-instruct"
