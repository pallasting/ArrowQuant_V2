#!/bin/bash
#
# Setup script for Arrow Optimization development environment
#
# This script sets up the development environment for the Arrow-optimized
# embedding system.

set -e  # Exit on error

echo "=========================================="
echo "Arrow Optimization - Dev Environment Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
REQUIRED_VERSION="3.10"

if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" != "$REQUIRED_VERSION" ]; then
    echo "❌ Error: Python $REQUIRED_VERSION or higher required (found $PYTHON_VERSION)"
    exit 1
fi
echo "✅ Python $PYTHON_VERSION"
echo ""

# Create virtual environment (optional)
if [ ! -d "venv-arrow" ]; then
    echo "Creating virtual environment..."
    python -m venv venv-arrow
    echo "✅ Virtual environment created"
    echo ""
    echo "To activate:"
    echo "  source venv-arrow/bin/activate  # Linux/Mac"
    echo "  venv-arrow\\Scripts\\activate     # Windows"
    echo ""
fi

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install -r requirements-arrow.txt
echo "✅ Dependencies installed"
echo ""

# Install package in editable mode
echo "Installing llm_compression package..."
pip install -e .
echo "✅ Package installed"
echo ""

# Verify installation
echo "Verifying installation..."
python -c "import pyarrow; print(f'✅ PyArrow {pyarrow.__version__}')"
python -c "import tokenizers; print(f'✅ Tokenizers {tokenizers.__version__}')"
python -c "import fastapi; print(f'✅ FastAPI {fastapi.__version__}')"
echo ""

# Create necessary directories
echo "Creating project directories..."
mkdir -p llm_compression/models/optimized
mkdir -p logs
mkdir -p data
echo "✅ Directories created"
echo ""

# Download example model (optional)
echo "=========================================="
echo "Optional: Download example model"
echo "=========================================="
echo ""
echo "To convert the default model:"
echo "  python -m llm_compression.tools.convert_model \\"
echo "    'sentence-transformers/all-MiniLM-L6-v2' \\"
echo "    --output-dir llm_compression/models/optimized"
echo ""

# Run tests (optional)
echo "=========================================="
echo "Optional: Run tests"
echo "=========================================="
echo ""
echo "To run tests:"
echo "  pytest tests/unit/"
echo "  pytest tests/integration/"
echo "  pytest --cov=llm_compression tests/"
echo ""

# Docker setup (optional)
echo "=========================================="
echo "Optional: Docker setup"
echo "=========================================="
echo ""
echo "To build Docker image:"
echo "  docker-compose -f deployment/docker/docker-compose.yml build"
echo ""
echo "To run services:"
echo "  docker-compose -f deployment/docker/docker-compose.yml up -d"
echo ""

echo "=========================================="
echo "✅ Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Read documentation: docs/arrow-optimization/"
echo "2. Review examples: examples/arrow-optimization/"
echo "3. Start implementing: See docs/arrow-optimization/TASKS.md"
echo ""
echo "Quick reference:"
echo "- Architecture: docs/arrow-optimization/ARCHITECTURE.md"
echo "- Roadmap: docs/arrow-optimization/ROADMAP.md"
echo "- Tasks: docs/arrow-optimization/TASKS.md"
echo ""
