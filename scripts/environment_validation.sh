#!/bin/bash
# Environment Validation Script for Phase 1.1
# Validates AMD GPU, Intel QAT, and deployment frameworks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Log functions
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

# Results tracking
RESULTS_FILE="environment_validation_results.txt"
echo "Environment Validation Results - $(date)" > $RESULTS_FILE
echo "================================================" >> $RESULTS_FILE

# Function to record result
record_result() {
    echo "$1: $2" >> $RESULTS_FILE
}

log_info "Starting environment validation..."
echo ""

# ============================================================================
# 1. System Information
# ============================================================================
log_info "=== 1. System Information ==="
echo ""

log_info "Checking OS and kernel version..."
OS_INFO=$(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)
KERNEL_VERSION=$(uname -r)
log_success "OS: $OS_INFO"
log_success "Kernel: $KERNEL_VERSION"
record_result "OS" "$OS_INFO"
record_result "Kernel" "$KERNEL_VERSION"
echo ""

# ============================================================================
# 2. AMD GPU Detection and Validation
# ============================================================================
log_info "=== 2. AMD GPU Detection ==="
echo ""

# Check for AMD GPU
if lspci | grep -i "VGA.*AMD" > /dev/null || lspci | grep -i "Display.*AMD" > /dev/null; then
    GPU_INFO=$(lspci | grep -i "VGA.*AMD\|Display.*AMD")
    log_success "AMD GPU detected:"
    echo "$GPU_INFO"
    record_result "AMD_GPU" "Detected"
    
    # Check for Mi50 specifically
    if echo "$GPU_INFO" | grep -i "MI50\|Vega 20" > /dev/null; then
        log_success "AMD Mi50 (Vega 20) confirmed!"
        record_result "AMD_MI50" "Confirmed"
    else
        log_warning "GPU detected but not Mi50. Detected: $GPU_INFO"
        record_result "AMD_MI50" "Not detected"
    fi
else
    log_error "No AMD GPU detected!"
    record_result "AMD_GPU" "Not detected"
fi
echo ""

# ============================================================================
# 3. ROCm Validation
# ============================================================================
log_info "=== 3. ROCm Validation ==="
echo ""

# Check if ROCm is installed
if command -v rocm-smi &> /dev/null; then
    log_success "ROCm tools found"
    
    # Get ROCm version
    if [ -f /opt/rocm/.info/version ]; then
        ROCM_VERSION=$(cat /opt/rocm/.info/version)
        log_success "ROCm version: $ROCM_VERSION"
        record_result "ROCm_Version" "$ROCM_VERSION"
    fi
    
    # Run rocm-smi
    log_info "Running rocm-smi..."
    if rocm-smi > /dev/null 2>&1; then
        rocm-smi
        log_success "rocm-smi executed successfully"
        record_result "rocm-smi" "Success"
    else
        log_error "rocm-smi failed to execute"
        record_result "rocm-smi" "Failed"
    fi
    
    # Check rocminfo
    log_info "Checking GPU architecture with rocminfo..."
    if command -v rocminfo &> /dev/null; then
        GPU_ARCH=$(rocminfo | grep "Name:" | head -1)
        GFX_VERSION=$(rocminfo | grep "gfx" | head -1)
        log_success "GPU Architecture: $GPU_ARCH"
        log_success "GFX Version: $GFX_VERSION"
        record_result "GPU_Architecture" "$GPU_ARCH"
        record_result "GFX_Version" "$GFX_VERSION"
    fi
else
    log_warning "ROCm not installed"
    record_result "ROCm" "Not installed"
    log_info "To install ROCm, run: sudo ./scripts/install_rocm.sh"
fi
echo ""

# ============================================================================
# 4. Vulkan Validation
# ============================================================================
log_info "=== 4. Vulkan Validation ==="
echo ""

if command -v vulkaninfo &> /dev/null; then
    log_success "Vulkan tools found"
    
    # Check Vulkan devices
    log_info "Checking Vulkan devices..."
    if vulkaninfo --summary > /dev/null 2>&1; then
        VULKAN_DEVICES=$(vulkaninfo --summary | grep "deviceName" | head -3)
        echo "$VULKAN_DEVICES"
        log_success "Vulkan validation successful"
        record_result "Vulkan" "Available"
    else
        log_warning "Vulkan validation failed"
        record_result "Vulkan" "Failed"
    fi
else
    log_warning "Vulkan not installed"
    record_result "Vulkan" "Not installed"
    log_info "To install Vulkan, run: sudo apt install vulkan-tools mesa-vulkan-drivers"
fi
echo ""

# ============================================================================
# 5. OpenCL Validation
# ============================================================================
log_info "=== 5. OpenCL Validation ==="
echo ""

if command -v clinfo &> /dev/null; then
    log_success "OpenCL tools found"
    
    # Check OpenCL platforms
    log_info "Checking OpenCL platforms..."
    if clinfo > /dev/null 2>&1; then
        OPENCL_PLATFORMS=$(clinfo | grep "Platform Name" | head -3)
        echo "$OPENCL_PLATFORMS"
        log_success "OpenCL validation successful"
        record_result "OpenCL" "Available"
    else
        log_warning "OpenCL validation failed"
        record_result "OpenCL" "Failed"
    fi
else
    log_warning "OpenCL not installed"
    record_result "OpenCL" "Not installed"
    log_info "To install OpenCL, run: sudo apt install ocl-icd-opencl-dev clinfo"
fi
echo ""

# ============================================================================
# 6. Intel QAT Detection and Validation
# ============================================================================
log_info "=== 6. Intel QAT Detection ==="
echo ""

# Check for QAT devices
if lspci | grep -i "QuickAssist\|QAT\|8086:37c8\|8086:4940\|8086:4942" > /dev/null; then
    QAT_DEVICES=$(lspci | grep -i "QuickAssist\|QAT\|8086:37c8\|8086:4940\|8086:4942")
    log_success "Intel QAT device(s) detected:"
    echo "$QAT_DEVICES"
    
    # Count QAT devices
    QAT_COUNT=$(echo "$QAT_DEVICES" | wc -l)
    log_success "Number of QAT devices: $QAT_COUNT"
    record_result "Intel_QAT" "Detected ($QAT_COUNT devices)"
    
    # Check QAT driver
    if lsmod | grep "qat" > /dev/null; then
        log_success "QAT kernel modules loaded:"
        lsmod | grep "qat"
        record_result "QAT_Driver" "Loaded"
        
        # Check QAT service status
        if systemctl is-active --quiet qat_service 2>/dev/null; then
            log_success "QAT service is running"
            record_result "QAT_Service" "Running"
        else
            log_warning "QAT service not running"
            record_result "QAT_Service" "Not running"
        fi
        
        # Check QAT devices status
        if [ -d /sys/kernel/debug/qat_* ]; then
            log_info "QAT device status:"
            for dev in /sys/kernel/debug/qat_*; do
                if [ -f "$dev/fw_counters" ]; then
                    echo "Device: $(basename $dev)"
                    cat "$dev/fw_counters" 2>/dev/null | head -5
                fi
            done
        fi
    else
        log_warning "QAT kernel modules not loaded"
        record_result "QAT_Driver" "Not loaded"
        log_info "To install QAT driver, run: sudo ./scripts/install_qat.sh"
    fi
else
    log_warning "No Intel QAT devices detected"
    record_result "Intel_QAT" "Not detected"
fi
echo ""

# ============================================================================
# 7. Python Environment Validation
# ============================================================================
log_info "=== 7. Python Environment ==="
echo ""

if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    log_success "$PYTHON_VERSION"
    record_result "Python" "$PYTHON_VERSION"
    
    # Check PyTorch
    log_info "Checking PyTorch installation..."
    if python3 -c "import torch" 2>/dev/null; then
        TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
        log_success "PyTorch version: $TORCH_VERSION"
        record_result "PyTorch" "$TORCH_VERSION"
        
        # Check CUDA/ROCm support
        CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [ "$CUDA_AVAILABLE" = "True" ]; then
            GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            log_success "PyTorch GPU support: Available"
            log_success "GPU detected by PyTorch: $GPU_NAME"
            record_result "PyTorch_GPU" "Available - $GPU_NAME"
        else
            log_warning "PyTorch GPU support: Not available"
            record_result "PyTorch_GPU" "Not available"
        fi
    else
        log_warning "PyTorch not installed"
        record_result "PyTorch" "Not installed"
    fi
else
    log_error "Python3 not found"
    record_result "Python" "Not found"
fi
echo ""

# ============================================================================
# 8. Ollama Validation
# ============================================================================
log_info "=== 8. Ollama Validation ==="
echo ""

if command -v ollama &> /dev/null; then
    OLLAMA_VERSION=$(ollama --version 2>/dev/null || echo "Unknown")
    log_success "Ollama installed: $OLLAMA_VERSION"
    record_result "Ollama" "Installed - $OLLAMA_VERSION"
    
    # Check if Ollama service is running
    if pgrep -x "ollama" > /dev/null; then
        log_success "Ollama service is running"
        record_result "Ollama_Service" "Running"
        
        # List installed models
        log_info "Installed Ollama models:"
        ollama list 2>/dev/null || log_warning "Could not list models"
    else
        log_warning "Ollama service not running"
        record_result "Ollama_Service" "Not running"
        log_info "To start Ollama: ollama serve"
    fi
else
    log_warning "Ollama not installed"
    record_result "Ollama" "Not installed"
    log_info "To install Ollama: curl -fsSL https://ollama.com/install.sh | sh"
fi
echo ""

# ============================================================================
# 9. vLLM Validation
# ============================================================================
log_info "=== 9. vLLM Validation ==="
echo ""

if python3 -c "import vllm" 2>/dev/null; then
    VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "Unknown")
    log_success "vLLM installed: $VLLM_VERSION"
    record_result "vLLM" "Installed - $VLLM_VERSION"
else
    log_warning "vLLM not installed"
    record_result "vLLM" "Not installed"
    log_info "To install vLLM: pip install vllm"
fi
echo ""

# ============================================================================
# 10. Memory and Storage Check
# ============================================================================
log_info "=== 10. System Resources ==="
echo ""

# RAM
TOTAL_RAM=$(free -h | awk '/^Mem:/ {print $2}')
AVAILABLE_RAM=$(free -h | awk '/^Mem:/ {print $7}')
log_success "Total RAM: $TOTAL_RAM"
log_success "Available RAM: $AVAILABLE_RAM"
record_result "Total_RAM" "$TOTAL_RAM"
record_result "Available_RAM" "$AVAILABLE_RAM"

# Disk space
DISK_SPACE=$(df -h / | awk 'NR==2 {print $4}')
log_success "Available disk space: $DISK_SPACE"
record_result "Disk_Space" "$DISK_SPACE"

# GPU memory (if ROCm available)
if command -v rocm-smi &> /dev/null; then
    GPU_MEM=$(rocm-smi --showmeminfo vram --csv 2>/dev/null | tail -1 | cut -d',' -f2 || echo "Unknown")
    log_success "GPU Memory: $GPU_MEM"
    record_result "GPU_Memory" "$GPU_MEM"
fi
echo ""

# ============================================================================
# 11. Network Connectivity
# ============================================================================
log_info "=== 11. Network Connectivity ==="
echo ""

# Check internet connectivity
if ping -c 1 8.8.8.8 > /dev/null 2>&1; then
    log_success "Internet connectivity: OK"
    record_result "Internet" "Connected"
else
    log_warning "Internet connectivity: Failed"
    record_result "Internet" "Not connected"
fi

# Check Hugging Face connectivity
if curl -s --head https://huggingface.co | head -n 1 | grep "200" > /dev/null; then
    log_success "Hugging Face access: OK"
    record_result "HuggingFace" "Accessible"
else
    log_warning "Hugging Face access: Failed"
    record_result "HuggingFace" "Not accessible"
fi
echo ""

# ============================================================================
# 12. Summary and Recommendations
# ============================================================================
log_info "=== 12. Validation Summary ==="
echo ""

echo "Validation complete! Results saved to: $RESULTS_FILE"
echo ""

# Generate recommendations
log_info "Recommendations:"
echo ""

# Check critical components
CRITICAL_MISSING=0

if ! lspci | grep -i "AMD" > /dev/null; then
    log_error "❌ AMD GPU not detected - Phase 1.1 cannot proceed"
    CRITICAL_MISSING=1
fi

if ! command -v rocm-smi &> /dev/null && ! command -v vulkaninfo &> /dev/null; then
    log_warning "⚠️  Neither ROCm nor Vulkan installed - Install at least one GPU backend"
    echo "   Recommended: sudo ./scripts/install_rocm.sh"
fi

if ! lspci | grep -i "QuickAssist\|QAT" > /dev/null; then
    log_warning "⚠️  Intel QAT not detected - Compression acceleration unavailable"
    echo "   This is optional but recommended for better performance"
fi

if ! command -v ollama &> /dev/null; then
    log_warning "⚠️  Ollama not installed - Install before Phase 1.1"
    echo "   Run: curl -fsSL https://ollama.com/install.sh | sh"
fi

if [ $CRITICAL_MISSING -eq 0 ]; then
    echo ""
    log_success "✅ System is ready for Phase 1.1 deployment!"
    echo ""
    log_info "Next steps:"
    echo "  1. Install missing components (if any)"
    echo "  2. Run: sudo ./scripts/install_rocm.sh (if ROCm not installed)"
    echo "  3. Run: sudo ./scripts/configure_qat.sh (to enable QAT acceleration)"
    echo "  4. Run: ./scripts/test_gpu_backends.sh (to test all GPU backends)"
    echo "  5. Proceed with Phase 1.1 Task 24"
else
    echo ""
    log_error "❌ Critical components missing - Cannot proceed with Phase 1.1"
    echo ""
    log_info "Required actions:"
    echo "  1. Verify AMD GPU is properly installed"
    echo "  2. Check BIOS settings for GPU"
    echo "  3. Install GPU drivers"
fi

echo ""
log_info "For detailed results, see: $RESULTS_FILE"
echo ""

# Exit with appropriate code
if [ $CRITICAL_MISSING -eq 1 ]; then
    exit 1
else
    exit 0
fi
