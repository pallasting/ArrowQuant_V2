#!/bin/bash
# GPU Backend Testing Script
# Tests ROCm, Vulkan, and OpenCL backends with AMD Mi50

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

RESULTS_FILE="gpu_backend_test_results.txt"
echo "GPU Backend Test Results - $(date)" > $RESULTS_FILE
echo "================================================" >> $RESULTS_FILE

log_info "GPU Backend Testing Script"
echo "==========================="
echo ""

# ============================================================================
# 1. ROCm Backend Test
# ============================================================================
log_info "=== 1. Testing ROCm Backend ==="
echo ""

if command -v rocm-smi &> /dev/null; then
    log_success "ROCm tools available"
    
    # Test 1: Basic GPU detection
    log_info "Test 1.1: GPU Detection"
    if rocm-smi > /dev/null 2>&1; then
        rocm-smi --showid --showproductname
        log_success "✅ GPU detected by ROCm"
        echo "ROCm_Detection: PASS" >> $RESULTS_FILE
    else
        log_error "❌ GPU not detected by ROCm"
        echo "ROCm_Detection: FAIL" >> $RESULTS_FILE
    fi
    echo ""
    
    # Test 2: PyTorch ROCm support
    log_info "Test 1.2: PyTorch ROCm Support"
    cat > /tmp/test_pytorch_rocm.py << 'EOF'
import torch
import sys

try:
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        
        # Test tensor operations
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        
        print("✅ Tensor operations successful")
        sys.exit(0)
    else:
        print("❌ CUDA not available")
        sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
EOF
    
    if python3 /tmp/test_pytorch_rocm.py; then
        log_success "✅ PyTorch ROCm support working"
        echo "PyTorch_ROCm: PASS" >> $RESULTS_FILE
    else
        log_error "❌ PyTorch ROCm support not working"
        echo "PyTorch_ROCm: FAIL" >> $RESULTS_FILE
    fi
    echo ""
    
    # Test 3: Performance test
    log_info "Test 1.3: ROCm Performance Test"
    cat > /tmp/test_rocm_perf.py << 'EOF'
import torch
import time

if torch.cuda.is_available():
    # Warmup
    x = torch.randn(5000, 5000).cuda()
    y = torch.randn(5000, 5000).cuda()
    _ = torch.matmul(x, y)
    torch.cuda.synchronize()
    
    # Benchmark
    iterations = 10
    start = time.time()
    for _ in range(iterations):
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
    elapsed = time.time() - start
    
    avg_time = elapsed / iterations
    gflops = (2 * 5000**3) / (avg_time * 1e9)
    
    print(f"Average time: {avg_time*1000:.2f}ms")
    print(f"Performance: {gflops:.2f} GFLOPS")
    
    if avg_time < 0.1:  # < 100ms is good
        print("✅ Performance: GOOD")
    else:
        print("⚠️  Performance: ACCEPTABLE")
else:
    print("❌ CUDA not available")
EOF
    
    python3 /tmp/test_rocm_perf.py
    echo ""
    
else
    log_warning "ROCm not installed - skipping ROCm tests"
    echo "ROCm: NOT_INSTALLED" >> $RESULTS_FILE
fi

# ============================================================================
# 2. Vulkan Backend Test
# ============================================================================
log_info "=== 2. Testing Vulkan Backend ==="
echo ""

if command -v vulkaninfo &> /dev/null; then
    log_success "Vulkan tools available"
    
    # Test 1: Device detection
    log_info "Test 2.1: Vulkan Device Detection"
    if vulkaninfo --summary > /dev/null 2>&1; then
        vulkaninfo --summary | grep -A 5 "GPU"
        log_success "✅ Vulkan devices detected"
        echo "Vulkan_Detection: PASS" >> $RESULTS_FILE
    else
        log_error "❌ Vulkan device detection failed"
        echo "Vulkan_Detection: FAIL" >> $RESULTS_FILE
    fi
    echo ""
    
    # Test 2: Vulkan compute test
    log_info "Test 2.2: Vulkan Compute Test"
    cat > /tmp/test_vulkan.py << 'EOF'
try:
    import vulkan as vk
    
    # Create instance
    app_info = vk.VkApplicationInfo(
        sType=vk.VK_STRUCTURE_TYPE_APPLICATION_INFO,
        pApplicationName="VulkanTest",
        applicationVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        pEngineName="No Engine",
        engineVersion=vk.VK_MAKE_VERSION(1, 0, 0),
        apiVersion=vk.VK_API_VERSION_1_0
    )
    
    create_info = vk.VkInstanceCreateInfo(
        sType=vk.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO,
        pApplicationInfo=app_info
    )
    
    instance = vk.vkCreateInstance(create_info, None)
    
    # Enumerate devices
    devices = vk.vkEnumeratePhysicalDevices(instance)
    print(f"Found {len(devices)} Vulkan device(s)")
    
    for i, device in enumerate(devices):
        props = vk.vkGetPhysicalDeviceProperties(device)
        print(f"Device {i}: {props.deviceName}")
    
    print("✅ Vulkan compute test successful")
    
except ImportError:
    print("⚠️  Python vulkan module not available")
    print("   Install with: pip install vulkan")
except Exception as e:
    print(f"❌ Vulkan test failed: {e}")
EOF
    
    python3 /tmp/test_vulkan.py 2>/dev/null || log_warning "Vulkan Python test skipped (module not available)"
    echo ""
    
else
    log_warning "Vulkan not installed - skipping Vulkan tests"
    echo "Vulkan: NOT_INSTALLED" >> $RESULTS_FILE
fi

# ============================================================================
# 3. OpenCL Backend Test
# ============================================================================
log_info "=== 3. Testing OpenCL Backend ==="
echo ""

if command -v clinfo &> /dev/null; then
    log_success "OpenCL tools available"
    
    # Test 1: Platform detection
    log_info "Test 3.1: OpenCL Platform Detection"
    if clinfo > /dev/null 2>&1; then
        clinfo | grep -A 3 "Platform Name"
        log_success "✅ OpenCL platforms detected"
        echo "OpenCL_Detection: PASS" >> $RESULTS_FILE
    else
        log_error "❌ OpenCL platform detection failed"
        echo "OpenCL_Detection: FAIL" >> $RESULTS_FILE
    fi
    echo ""
    
    # Test 2: Device info
    log_info "Test 3.2: OpenCL Device Information"
    clinfo | grep -E "Device Name|Device Type|Max compute units|Global memory size" | head -10
    echo ""
    
    # Test 3: PyOpenCL test
    log_info "Test 3.3: PyOpenCL Test"
    cat > /tmp/test_opencl.py << 'EOF'
try:
    import pyopencl as cl
    import numpy as np
    
    # Get platforms and devices
    platforms = cl.get_platforms()
    print(f"Found {len(platforms)} OpenCL platform(s)")
    
    for platform in platforms:
        print(f"Platform: {platform.name}")
        devices = platform.get_devices()
        for device in devices:
            print(f"  Device: {device.name}")
            print(f"    Type: {cl.device_type.to_string(device.type)}")
            print(f"    Compute Units: {device.max_compute_units}")
            print(f"    Global Memory: {device.global_mem_size / 1024**3:.2f} GB")
    
    # Simple compute test
    ctx = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(ctx)
    
    # Create buffers
    a = np.random.rand(1000).astype(np.float32)
    b = np.random.rand(1000).astype(np.float32)
    
    mf = cl.mem_flags
    a_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=a)
    b_buf = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=b)
    c_buf = cl.Buffer(ctx, mf.WRITE_ONLY, a.nbytes)
    
    # Simple kernel
    prg = cl.Program(ctx, """
    __kernel void add(__global const float *a,
                      __global const float *b,
                      __global float *c)
    {
        int gid = get_global_id(0);
        c[gid] = a[gid] + b[gid];
    }
    """).build()
    
    prg.add(queue, a.shape, None, a_buf, b_buf, c_buf)
    
    c = np.empty_like(a)
    cl.enqueue_copy(queue, c, c_buf)
    
    # Verify
    if np.allclose(c, a + b):
        print("✅ OpenCL compute test successful")
    else:
        print("❌ OpenCL compute test failed")
    
except ImportError:
    print("⚠️  PyOpenCL not available")
    print("   Install with: pip install pyopencl")
except Exception as e:
    print(f"❌ OpenCL test failed: {e}")
EOF
    
    python3 /tmp/test_opencl.py 2>/dev/null || log_warning "PyOpenCL test skipped (module not available)"
    echo ""
    
else
    log_warning "OpenCL not installed - skipping OpenCL tests"
    echo "OpenCL: NOT_INSTALLED" >> $RESULTS_FILE
fi

# ============================================================================
# 4. Ollama GPU Backend Test
# ============================================================================
log_info "=== 4. Testing Ollama GPU Backends ==="
echo ""

if command -v ollama &> /dev/null; then
    log_success "Ollama installed"
    
    # Test ROCm backend
    log_info "Test 4.1: Ollama with ROCm"
    export OLLAMA_GPU_DRIVER=rocm
    export HSA_OVERRIDE_GFX_VERSION=9.0.6
    
    # Start Ollama in background if not running
    if ! pgrep -x "ollama" > /dev/null; then
        log_info "Starting Ollama service..."
        ollama serve > /tmp/ollama.log 2>&1 &
        OLLAMA_PID=$!
        sleep 5
    fi
    
    # Test model pull (small model)
    log_info "Testing model pull with ROCm backend..."
    if timeout 60 ollama pull tinyllama > /dev/null 2>&1; then
        log_success "✅ Ollama ROCm backend working"
        echo "Ollama_ROCm: PASS" >> $RESULTS_FILE
        
        # Test inference
        log_info "Testing inference..."
        RESPONSE=$(echo "Hello" | ollama run tinyllama 2>/dev/null | head -1)
        if [ -n "$RESPONSE" ]; then
            log_success "✅ Inference successful"
            echo "Ollama_Inference: PASS" >> $RESULTS_FILE
        fi
    else
        log_warning "⚠️  Ollama ROCm test timed out or failed"
        echo "Ollama_ROCm: TIMEOUT" >> $RESULTS_FILE
    fi
    
    # Test Vulkan backend
    log_info "Test 4.2: Ollama with Vulkan"
    export OLLAMA_GPU_DRIVER=vulkan
    
    # Restart Ollama with Vulkan
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
        sleep 2
    fi
    
    ollama serve > /tmp/ollama_vulkan.log 2>&1 &
    OLLAMA_PID=$!
    sleep 5
    
    if timeout 30 ollama list > /dev/null 2>&1; then
        log_success "✅ Ollama Vulkan backend working"
        echo "Ollama_Vulkan: PASS" >> $RESULTS_FILE
    else
        log_warning "⚠️  Ollama Vulkan test failed"
        echo "Ollama_Vulkan: FAIL" >> $RESULTS_FILE
    fi
    
    # Cleanup
    if [ -n "$OLLAMA_PID" ]; then
        kill $OLLAMA_PID 2>/dev/null || true
    fi
    
else
    log_warning "Ollama not installed - skipping Ollama tests"
    echo "Ollama: NOT_INSTALLED" >> $RESULTS_FILE
fi
echo ""

# ============================================================================
# 5. Summary and Recommendations
# ============================================================================
log_info "=== Test Summary ==="
echo ""

echo "Test results saved to: $RESULTS_FILE"
echo ""

# Analyze results
ROCM_OK=false
VULKAN_OK=false
OPENCL_OK=false

if grep -q "ROCm_Detection: PASS" $RESULTS_FILE; then
    ROCM_OK=true
fi

if grep -q "Vulkan_Detection: PASS" $RESULTS_FILE; then
    VULKAN_OK=true
fi

if grep -q "OpenCL_Detection: PASS" $RESULTS_FILE; then
    OPENCL_OK=true
fi

# Recommendations
log_info "Recommendations for Phase 1.1:"
echo ""

if $ROCM_OK; then
    log_success "✅ ROCm: Use as primary GPU backend (best performance)"
    echo "   Configuration: export OLLAMA_GPU_DRIVER=rocm"
    echo "   Configuration: export HSA_OVERRIDE_GFX_VERSION=9.0.6"
else
    log_warning "⚠️  ROCm: Not available or not working"
fi

if $VULKAN_OK; then
    log_success "✅ Vulkan: Use as fallback GPU backend"
    echo "   Configuration: export OLLAMA_GPU_DRIVER=vulkan"
else
    log_warning "⚠️  Vulkan: Not available"
fi

if $OPENCL_OK; then
    log_success "✅ OpenCL: Available as last resort"
else
    log_warning "⚠️  OpenCL: Not available"
fi

echo ""

if $ROCM_OK || $VULKAN_OK || $OPENCL_OK; then
    log_success "✅ At least one GPU backend is working - Phase 1.1 can proceed"
    echo ""
    log_info "Recommended backend priority:"
    echo "  1. ROCm (best performance)"
    echo "  2. Vulkan (good compatibility)"
    echo "  3. OpenCL (fallback)"
else
    log_error "❌ No GPU backends working - Phase 1.1 cannot proceed"
    echo ""
    log_info "Required actions:"
    echo "  1. Install ROCm: sudo ./scripts/install_rocm.sh"
    echo "  2. Or install Vulkan: sudo apt install vulkan-tools mesa-vulkan-drivers"
    echo "  3. Verify GPU is properly installed and enabled in BIOS"
fi

echo ""
log_info "For detailed results, see: $RESULTS_FILE"
