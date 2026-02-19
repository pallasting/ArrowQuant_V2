#!/bin/bash
# Intel QAT Configuration and Optimization Script
# Configures Intel QuickAssist Technology for compression acceleration

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

log_info "Intel QAT Configuration Script"
echo "================================"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    log_error "This script must be run as root (sudo)"
    exit 1
fi

# ============================================================================
# 1. Detect QAT Devices
# ============================================================================
log_info "Step 1: Detecting Intel QAT devices..."
echo ""

if ! lspci | grep -i "QuickAssist\|QAT\|8086:37c8\|8086:4940\|8086:4942" > /dev/null; then
    log_error "No Intel QAT devices detected!"
    log_info "Please verify:"
    echo "  1. QAT cards are properly installed"
    echo "  2. BIOS settings enable QAT"
    echo "  3. PCIe slots are functioning"
    exit 1
fi

QAT_DEVICES=$(lspci | grep -i "QuickAssist\|QAT\|8086:37c8\|8086:4940\|8086:4942")
QAT_COUNT=$(echo "$QAT_DEVICES" | wc -l)

log_success "Found $QAT_COUNT QAT device(s):"
echo "$QAT_DEVICES"
echo ""

# ============================================================================
# 2. Install QAT Driver (if not installed)
# ============================================================================
log_info "Step 2: Checking QAT driver installation..."
echo ""

if ! lsmod | grep "qat" > /dev/null; then
    log_warning "QAT driver not loaded. Installing..."
    
    # Install dependencies
    log_info "Installing dependencies..."
    apt-get update
    apt-get install -y \
        build-essential \
        linux-headers-$(uname -r) \
        pkg-config \
        libudev-dev \
        libssl-dev \
        wget \
        pciutils
    
    # Download and install QAT driver
    QAT_VERSION="QAT20.L.1.1.50-00003"
    QAT_URL="https://downloadmirror.intel.com/813591/${QAT_VERSION}.tar.gz"
    
    log_info "Downloading QAT driver ${QAT_VERSION}..."
    cd /tmp
    wget -q $QAT_URL -O qat_driver.tar.gz
    
    log_info "Extracting and building QAT driver..."
    tar xzf qat_driver.tar.gz
    cd ${QAT_VERSION}
    
    # Configure and build
    ./configure --enable-icp-sriov=host
    make -j$(nproc)
    make install
    
    log_success "QAT driver installed"
    
    # Load kernel modules
    log_info "Loading QAT kernel modules..."
    modprobe qat_c62x
    modprobe qat_c62xvf
    modprobe intel_qat
    
    log_success "QAT kernel modules loaded"
else
    log_success "QAT driver already loaded"
    lsmod | grep "qat"
fi
echo ""

# ============================================================================
# 3. Configure QAT Devices
# ============================================================================
log_info "Step 3: Configuring QAT devices..."
echo ""

# Create QAT configuration directory
mkdir -p /etc/qat

# Generate configuration for each device
for dev_id in $(lspci -d 8086: | grep -i "QuickAssist\|QAT" | cut -d' ' -f1); do
    log_info "Configuring device: $dev_id"
    
    # Get device number
    DEV_NUM=$(echo $dev_id | sed 's/.*://')
    
    # Create device configuration
    cat > /etc/qat/qat_dev${DEV_NUM}.conf << 'EOF'
[GENERAL]
ServicesEnabled = dc

# Data Compression - Instance 0
[DC0]
NumberDcConcurrentSymRequests = 512
NumberDcConcurrentAsymRequests = 64
NumProcesses = 1
LimitDevAccess = 0

# Data Compression - Instance 1
[DC1]
NumberDcConcurrentSymRequests = 512
NumberDcConcurrentAsymRequests = 64
NumProcesses = 1
LimitDevAccess = 0
EOF
    
    log_success "Configuration created for device $dev_id"
done
echo ""

# ============================================================================
# 4. Start QAT Service
# ============================================================================
log_info "Step 4: Starting QAT service..."
echo ""

# Create systemd service file
cat > /etc/systemd/system/qat.service << 'EOF'
[Unit]
Description=Intel QuickAssist Technology Service
After=network.target

[Service]
Type=oneshot
ExecStart=/usr/local/bin/qat_service start
ExecStop=/usr/local/bin/qat_service stop
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd
systemctl daemon-reload

# Enable and start service
systemctl enable qat.service
systemctl start qat.service

if systemctl is-active --quiet qat.service; then
    log_success "QAT service started successfully"
else
    log_error "Failed to start QAT service"
    systemctl status qat.service
    exit 1
fi
echo ""

# ============================================================================
# 5. Verify QAT Functionality
# ============================================================================
log_info "Step 5: Verifying QAT functionality..."
echo ""

# Check device status
log_info "QAT device status:"
for dev in /sys/kernel/debug/qat_*; do
    if [ -d "$dev" ]; then
        DEV_NAME=$(basename $dev)
        log_info "Device: $DEV_NAME"
        
        if [ -f "$dev/fw_counters" ]; then
            cat "$dev/fw_counters" | head -10
        fi
        echo ""
    fi
done

# Run sample test if available
if [ -f /usr/local/bin/cpa_sample_code ]; then
    log_info "Running QAT sample test..."
    /usr/local/bin/cpa_sample_code runTests=1 > /tmp/qat_test.log 2>&1
    
    if grep -q "Sample code completed successfully" /tmp/qat_test.log; then
        log_success "QAT sample test passed!"
    else
        log_warning "QAT sample test had issues. Check /tmp/qat_test.log"
    fi
else
    log_warning "QAT sample code not available for testing"
fi
echo ""

# ============================================================================
# 6. Install Python QAT Library
# ============================================================================
log_info "Step 6: Installing Python QAT library..."
echo ""

# Install qatlib Python bindings
pip3 install qat-python 2>/dev/null || log_warning "qat-python not available via pip"

# Install Intel IPP for compression
log_info "Installing Intel IPP (Integrated Performance Primitives)..."
apt-get install -y intel-ipp-dev 2>/dev/null || log_warning "Intel IPP not available"

log_success "Python libraries installation attempted"
echo ""

# ============================================================================
# 7. Performance Tuning
# ============================================================================
log_info "Step 7: Applying performance tuning..."
echo ""

# Set CPU governor to performance
log_info "Setting CPU governor to performance mode..."
for cpu in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
    echo performance > $cpu 2>/dev/null || true
done

# Disable CPU frequency scaling
log_info "Disabling CPU frequency scaling..."
systemctl disable ondemand 2>/dev/null || true

# Set IRQ affinity for QAT devices
log_info "Configuring IRQ affinity for QAT devices..."
for irq in $(grep qat /proc/interrupts | cut -d: -f1); do
    # Spread IRQs across CPUs
    CPU_MASK=$(printf '%x' $((1 << (irq % $(nproc)))))
    echo $CPU_MASK > /proc/irq/$irq/smp_affinity 2>/dev/null || true
done

log_success "Performance tuning applied"
echo ""

# ============================================================================
# 8. Create QAT Test Script
# ============================================================================
log_info "Step 8: Creating QAT test script..."
echo ""

cat > /usr/local/bin/test_qat.sh << 'EOF'
#!/bin/bash
# Quick QAT functionality test

echo "Testing Intel QAT..."
echo ""

# Check service status
echo "1. Service Status:"
systemctl status qat.service | grep Active
echo ""

# Check devices
echo "2. QAT Devices:"
lspci | grep -i "QuickAssist\|QAT"
echo ""

# Check kernel modules
echo "3. Kernel Modules:"
lsmod | grep qat
echo ""

# Check device counters
echo "4. Device Counters:"
for dev in /sys/kernel/debug/qat_*/fw_counters; do
    if [ -f "$dev" ]; then
        echo "Device: $(dirname $dev | xargs basename)"
        cat "$dev" | grep -E "Requests|Responses" | head -4
        echo ""
    fi
done

echo "QAT test complete!"
EOF

chmod +x /usr/local/bin/test_qat.sh
log_success "Test script created: /usr/local/bin/test_qat.sh"
echo ""

# ============================================================================
# 9. Summary and Next Steps
# ============================================================================
log_info "=== Configuration Summary ==="
echo ""

log_success "✅ Intel QAT configuration complete!"
echo ""

log_info "QAT Status:"
echo "  - Devices detected: $QAT_COUNT"
echo "  - Driver: Loaded"
echo "  - Service: Running"
echo "  - Configuration: /etc/qat/"
echo ""

log_info "Quick Commands:"
echo "  - Check status: systemctl status qat.service"
echo "  - Test QAT: /usr/local/bin/test_qat.sh"
echo "  - View logs: journalctl -u qat.service"
echo "  - Restart service: systemctl restart qat.service"
echo ""

log_info "Next Steps:"
echo "  1. Run: /usr/local/bin/test_qat.sh"
echo "  2. Test compression: python3 scripts/test_qat_compression.py"
echo "  3. Integrate with LLM compression system"
echo ""

log_info "For compression integration, QAT will be used for:"
echo "  - zstd compression acceleration (fallback mode)"
echo "  - diff data compression (Phase 0 algorithm)"
echo "  - Arrow/Parquet compression"
echo ""

log_success "Configuration complete! QAT is ready for use."
