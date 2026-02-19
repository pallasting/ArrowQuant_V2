#!/bin/bash

# LLM Compression System Deployment Script
# 
# This script automates the deployment process:
# 1. Checks environment (Python version, system requirements)
# 2. Creates virtual environment
# 3. Installs dependencies
# 4. Validates configuration
# 5. Runs health check
#
# Requirements: 11.5

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Print banner
echo "=========================================="
echo "  LLM Compression System Deployment"
echo "=========================================="
echo ""

# Step 1: Check Python version
log_info "Checking Python version..."
if ! command_exists python3; then
    log_error "Python 3 is not installed. Please install Python 3.10 or higher."
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 10 ]); then
    log_error "Python 3.10 or higher is required. Found: $PYTHON_VERSION"
    exit 1
fi

log_info "Python version: $PYTHON_VERSION ✓"

# Step 2: Check system requirements
log_info "Checking system requirements..."

# Check available memory
if command_exists free; then
    TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    if [ -n "$TOTAL_MEM" ] && [ "$TOTAL_MEM" -lt 4 ]; then
        log_warn "Low system memory: ${TOTAL_MEM}GB (recommended: 8GB+)"
    elif [ -n "$TOTAL_MEM" ]; then
        log_info "System memory: ${TOTAL_MEM}GB ✓"
    fi
fi

# Check disk space
DISK_SPACE=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
if [ "$DISK_SPACE" -lt 10 ]; then
    log_warn "Low disk space: ${DISK_SPACE}GB (recommended: 20GB+)"
else
    log_info "Disk space: ${DISK_SPACE}GB ✓"
fi

# Check for GPU (optional)
if command_exists nvidia-smi; then
    log_info "NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
else
    log_warn "No NVIDIA GPU detected (will run in CPU mode)"
fi

# Step 3: Create virtual environment
log_info "Creating virtual environment..."
if [ -d "venv" ]; then
    log_warn "Virtual environment already exists. Skipping creation."
else
    python3 -m venv venv
    log_info "Virtual environment created ✓"
fi

# Activate virtual environment
log_info "Activating virtual environment..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv_test/bin/activate" ]; then
    source venv_test/bin/activate
else
    log_error "Virtual environment activation script not found!"
    exit 1
fi

# Upgrade pip
log_info "Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

# Step 4: Install dependencies
log_info "Installing dependencies from requirements.txt..."
if [ ! -f "requirements.txt" ]; then
    log_error "requirements.txt not found!"
    exit 1
fi

pip install -r requirements.txt

log_info "Dependencies installed ✓"

# Step 5: Install package in development mode
log_info "Installing package in development mode..."
if [ -f "setup.py" ]; then
    pip install -e .
    log_info "Package installed ✓"
else
    log_warn "setup.py not found. Skipping package installation."
fi

# Step 6: Validate configuration
log_info "Validating configuration..."
if [ ! -f "config.yaml" ]; then
    log_warn "config.yaml not found. Creating default configuration..."
    cat > config.yaml << 'EOF'
# LLM Compression System Configuration

llm:
  cloud_endpoint: "http://localhost:8045"
  cloud_api_key: null
  timeout: 30.0
  max_retries: 3
  rate_limit: 60

model:
  prefer_local: true
  local_endpoints: {}
  quality_threshold: 0.85

compression:
  min_compress_length: 100
  max_tokens: 100
  temperature: 0.3
  auto_compress_threshold: 100

storage:
  storage_path: "~/.ai-os/memory/"
  compression_level: 3
  use_float16: true

performance:
  batch_size: 16
  max_concurrent: 4
  cache_size: 10000
  cache_ttl: 3600

monitoring:
  enable_prometheus: false
  prometheus_port: 9090
  alert_quality_threshold: 0.85
EOF
    log_info "Default config.yaml created ✓"
else
    log_info "config.yaml found ✓"
fi

# Create storage directories
log_info "Creating storage directories..."
STORAGE_PATH=$(python3 -c "import yaml; print(yaml.safe_load(open('config.yaml'))['storage']['storage_path'])" 2>/dev/null || echo "~/.ai-os/memory/")
STORAGE_PATH="${STORAGE_PATH/#\~/$HOME}"

mkdir -p "$STORAGE_PATH/core"
mkdir -p "$STORAGE_PATH/working"
mkdir -p "$STORAGE_PATH/long-term"
mkdir -p "$STORAGE_PATH/shared"

log_info "Storage directories created ✓"

# Step 7: Run tests (optional)
if [ "$1" == "--with-tests" ]; then
    log_info "Running tests..."
    python3 -m pytest tests/ -v --tb=short
    log_info "Tests passed ✓"
fi

# Step 8: Run health check
log_info "Running health check..."
if command_exists python3; then
    python3 -c "
import asyncio
import sys
sys.path.insert(0, '.')

from llm_compression.health import HealthChecker
from llm_compression.config import Config

async def check():
    try:
        config = Config.from_yaml('config.yaml')
        checker = HealthChecker(config=config)
        result = await checker.check_health()
        
        print(f'Overall Status: {result.overall_status}')
        print('Components:')
        for name, comp in result.components.items():
            print(f'  - {name}: {comp.status} ({comp.message})')
        
        if result.overall_status == 'unhealthy':
            sys.exit(1)
    except Exception as e:
        print(f'Health check failed: {e}')
        sys.exit(1)

asyncio.run(check())
"
    if [ $? -eq 0 ]; then
        log_info "Health check passed ✓"
    else
        log_warn "Health check reported issues (see above)"
    fi
else
    log_warn "Skipping health check (Python not available)"
fi

# Step 9: Print next steps
echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Activate virtual environment:"
echo "     source venv/bin/activate"
echo ""
echo "  2. Review and update config.yaml"
echo ""
echo "  3. Start the health check API:"
echo "     python3 -m llm_compression.api"
echo ""
echo "  4. Access the API at:"
echo "     http://localhost:8000/health"
echo ""
echo "  5. View API documentation:"
echo "     http://localhost:8000/docs"
echo ""
echo "For more information, see README.md"
echo ""
