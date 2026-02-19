#!/bin/bash
# Test ROCm runner manually to capture crash details

echo "=== Testing ROCm Runner Manually ==="
echo ""

# Set ROCm environment
export HSA_OVERRIDE_GFX_VERSION=9.0.6
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export LD_LIBRARY_PATH=/opt/rocm-7.2.0/lib:/opt/rocm-7.2.0/lib64
export ROCM_PATH=/opt/rocm-7.2.0

echo "Environment:"
echo "  HSA_OVERRIDE_GFX_VERSION=$HSA_OVERRIDE_GFX_VERSION"
echo "  ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
echo "  LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo ""

# Check if ROCm libraries are accessible
echo "Checking ROCm libraries:"
if [ -f "/opt/rocm-7.2.0/lib/libamdhip64.so" ]; then
    echo "  ✓ libamdhip64.so found"
    ldd /opt/rocm-7.2.0/lib/libamdhip64.so | head -5
else
    echo "  ✗ libamdhip64.so not found"
fi
echo ""

# Check GPU visibility
echo "Checking GPU visibility:"
rocminfo | grep -A 3 "Marketing Name" | grep -E "(Marketing|Name:)"
echo ""

# Try to run ollama runner with ROCm
echo "Attempting to run ollama runner with ROCm..."
echo "This will likely crash - we want to see the error"
echo ""

timeout 10 /usr/local/bin/ollama runner --ollama-engine --port 50000 2>&1 | head -50 &
RUNNER_PID=$!

sleep 5

if ps -p $RUNNER_PID > /dev/null; then
    echo "Runner is still running (PID: $RUNNER_PID)"
    kill $RUNNER_PID 2>/dev/null
else
    echo "Runner crashed or exited"
fi

echo ""
echo "=== Test Complete ==="
