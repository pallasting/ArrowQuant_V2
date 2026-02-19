#!/bin/bash
# Diagnose Ollama ROCm Configuration

echo "=========================================="
echo "Ollama ROCm Configuration Diagnosis"
echo "=========================================="
echo ""

echo "1. Ollama Version:"
ollama --version
echo ""

echo "2. Ollama Binary Info:"
file /usr/local/bin/ollama
echo ""

echo "3. ROCm Libraries in Ollama:"
ls -lh /usr/local/lib/ollama/rocm/ | grep -E "\.so"
echo ""

echo "4. Check if ROCm libraries exist:"
if [ -f "/usr/local/lib/ollama/rocm/libggml-hip.so" ]; then
    echo "✓ libggml-hip.so found (ROCm GPU support library)"
    ls -lh /usr/local/lib/ollama/rocm/libggml-hip.so
else
    echo "✗ libggml-hip.so NOT found"
fi
echo ""

echo "5. System ROCm Installation:"
if command -v rocminfo &> /dev/null; then
    echo "✓ ROCm installed"
    rocminfo | grep -A 3 "Marketing Name" | grep -E "(Marketing|Name:)"
else
    echo "✗ ROCm not found in PATH"
fi
echo ""

echo "6. GPU Detection:"
if command -v rocm-smi &> /dev/null; then
    echo "✓ rocm-smi available"
    rocm-smi --showproductname
else
    echo "✗ rocm-smi not found"
fi
echo ""

echo "7. Current Ollama Process:"
ps aux | grep "ollama serve" | grep -v grep
echo ""

echo "8. Ollama Model Status:"
ollama ps
echo ""

echo "9. Environment Variables (current shell):"
env | grep -E "(ROCM|HIP|HSA|GPU)" | sort
echo ""

echo "10. Check Ollama Library Path:"
if [ -d "/usr/local/lib/ollama" ]; then
    echo "Available GPU backends:"
    ls -d /usr/local/lib/ollama/*/ 2>/dev/null | xargs -n1 basename
fi
echo ""

echo "=========================================="
echo "Diagnosis Summary"
echo "=========================================="
echo ""

# Check if ROCm support is compiled in
if [ -f "/usr/local/lib/ollama/rocm/libggml-hip.so" ]; then
    echo "✓ Ollama has ROCm support compiled in"
    echo ""
    echo "Issue: Ollama is not using the GPU despite having ROCm libraries."
    echo ""
    echo "Possible causes:"
    echo "1. Missing environment variables (HSA_OVERRIDE_GFX_VERSION)"
    echo "2. Ollama not configured to prefer GPU over CPU"
    echo "3. GPU not detected at runtime"
    echo ""
    echo "Solution:"
    echo "Stop Ollama and restart with ROCm environment variables:"
    echo ""
    echo "  sudo pkill ollama"
    echo "  sudo HSA_OVERRIDE_GFX_VERSION=9.0.6 ROCR_VISIBLE_DEVICES=0 /usr/local/bin/ollama serve &"
    echo ""
    echo "Then reload the model:"
    echo "  ollama stop qwen2.5:7b-instruct"
    echo "  ollama run qwen2.5:7b-instruct 'test'"
else
    echo "✗ Ollama does NOT have ROCm support"
    echo ""
    echo "You may need to install the ROCm-specific version of Ollama."
    echo "However, the rocm directory exists, so ROCm support should be available."
fi
