#!/bin/bash
# Release automation script for ArrowQuant V2
# This script helps with local release builds and testing

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default values
VERSION=""
BUILD_RUST=true
BUILD_PYTHON=true
RUN_TESTS=true
PUBLISH_PYPI=false
DRY_RUN=false

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Release automation script for ArrowQuant V2

OPTIONS:
    -v, --version VERSION    Version to release (e.g., 0.1.0)
    -r, --rust-only          Build Rust library only
    -p, --python-only        Build Python wheels only
    --no-tests               Skip running tests
    --publish                Publish to PyPI (requires credentials)
    --dry-run                Perform a dry run without publishing
    -h, --help               Show this help message

EXAMPLES:
    # Build release for version 0.1.0
    $0 --version 0.1.0

    # Build and publish to PyPI
    $0 --version 0.1.0 --publish

    # Build Rust library only
    $0 --version 0.1.0 --rust-only

    # Dry run (no publishing)
    $0 --version 0.1.0 --dry-run

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -v|--version)
            VERSION="$2"
            shift 2
            ;;
        -r|--rust-only)
            BUILD_PYTHON=false
            shift
            ;;
        -p|--python-only)
            BUILD_RUST=false
            shift
            ;;
        --no-tests)
            RUN_TESTS=false
            shift
            ;;
        --publish)
            PUBLISH_PYPI=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Validate version
if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Version is required${NC}"
    usage
fi

# Validate version format
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${RED}Error: Invalid version format. Expected: X.Y.Z${NC}"
    exit 1
fi

echo -e "${GREEN}=== ArrowQuant V2 Release Automation ===${NC}"
echo -e "Version: ${YELLOW}$VERSION${NC}"
echo -e "Build Rust: ${YELLOW}$BUILD_RUST${NC}"
echo -e "Build Python: ${YELLOW}$BUILD_PYTHON${NC}"
echo -e "Run Tests: ${YELLOW}$RUN_TESTS${NC}"
echo -e "Publish to PyPI: ${YELLOW}$PUBLISH_PYPI${NC}"
echo -e "Dry Run: ${YELLOW}$DRY_RUN${NC}"
echo ""

# Change to project directory
cd "$PROJECT_DIR"

# Step 1: Update version in Cargo.toml
if [ "$BUILD_RUST" = true ]; then
    echo -e "${GREEN}Step 1: Updating version in Cargo.toml${NC}"
    
    # Check if Cargo.toml exists
    if [ ! -f "Cargo.toml" ]; then
        echo -e "${RED}Error: Cargo.toml not found${NC}"
        exit 1
    fi
    
    # Update version (requires sed)
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
    else
        # Linux
        sed -i "s/^version = \".*\"/version = \"$VERSION\"/" Cargo.toml
    fi
    
    echo -e "${GREEN}✓ Version updated in Cargo.toml${NC}"
fi

# Step 2: Update version in pyproject.toml
if [ "$BUILD_PYTHON" = true ]; then
    echo -e "${GREEN}Step 2: Updating version in pyproject.toml${NC}"
    
    # Check if pyproject.toml exists
    if [ ! -f "pyproject.toml" ]; then
        echo -e "${RED}Error: pyproject.toml not found${NC}"
        exit 1
    fi
    
    # Update version
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    else
        # Linux
        sed -i "s/^version = \".*\"/version = \"$VERSION\"/" pyproject.toml
    fi
    
    echo -e "${GREEN}✓ Version updated in pyproject.toml${NC}"
fi

# Step 3: Run tests
if [ "$RUN_TESTS" = true ]; then
    echo -e "${GREEN}Step 3: Running tests${NC}"
    
    if [ "$BUILD_RUST" = true ]; then
        echo -e "${YELLOW}Running Rust tests...${NC}"
        cargo test --all-features --release
        echo -e "${GREEN}✓ Rust tests passed${NC}"
    fi
    
    if [ "$BUILD_PYTHON" = true ]; then
        echo -e "${YELLOW}Running Python tests...${NC}"
        if command -v pytest &> /dev/null; then
            pytest tests/test_python_bindings.py -v
            echo -e "${GREEN}✓ Python tests passed${NC}"
        else
            echo -e "${YELLOW}Warning: pytest not found, skipping Python tests${NC}"
        fi
    fi
fi

# Step 4: Build Rust library
if [ "$BUILD_RUST" = true ]; then
    echo -e "${GREEN}Step 4: Building Rust library${NC}"
    
    # Clean previous builds
    cargo clean
    
    # Build release
    echo -e "${YELLOW}Building release binary...${NC}"
    cargo build --release --all-features
    
    # Get target directory
    TARGET_DIR="target/release"
    
    # Determine library name based on OS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        LIB_NAME="libarrow_quant_v2.so"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        LIB_NAME="libarrow_quant_v2.dylib"
    elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
        LIB_NAME="arrow_quant_v2.dll"
    else
        echo -e "${RED}Error: Unsupported OS${NC}"
        exit 1
    fi
    
    # Check if library was built
    if [ -f "$TARGET_DIR/$LIB_NAME" ]; then
        echo -e "${GREEN}✓ Rust library built: $TARGET_DIR/$LIB_NAME${NC}"
        
        # Get file size
        SIZE=$(du -h "$TARGET_DIR/$LIB_NAME" | cut -f1)
        echo -e "  Size: ${YELLOW}$SIZE${NC}"
    else
        echo -e "${RED}Error: Failed to build Rust library${NC}"
        exit 1
    fi
fi

# Step 5: Build Python wheels
if [ "$BUILD_PYTHON" = true ]; then
    echo -e "${GREEN}Step 5: Building Python wheels${NC}"
    
    # Check if maturin is installed
    if ! command -v maturin &> /dev/null; then
        echo -e "${YELLOW}Installing maturin...${NC}"
        pip install maturin
    fi
    
    # Clean previous builds
    rm -rf dist/
    
    # Build wheels
    echo -e "${YELLOW}Building wheels...${NC}"
    maturin build --release --features python --out dist
    
    # Build source distribution
    echo -e "${YELLOW}Building source distribution...${NC}"
    maturin sdist --out dist
    
    # List built artifacts
    echo -e "${GREEN}✓ Python packages built:${NC}"
    ls -lh dist/
fi

# Step 6: Test wheels
if [ "$BUILD_PYTHON" = true ] && [ "$RUN_TESTS" = true ]; then
    echo -e "${GREEN}Step 6: Testing wheels${NC}"
    
    # Create virtual environment for testing
    VENV_DIR="venv-test"
    echo -e "${YELLOW}Creating test virtual environment...${NC}"
    python3 -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    
    # Install wheel
    echo -e "${YELLOW}Installing wheel...${NC}"
    pip install dist/*.whl
    
    # Test import
    echo -e "${YELLOW}Testing import...${NC}"
    python3 -c "import arrow_quant_v2; print('Import successful')"
    
    # Run basic tests
    if [ -d "tests" ]; then
        echo -e "${YELLOW}Running integration tests...${NC}"
        pip install pytest
        pytest tests/test_python_bindings.py -v
    fi
    
    # Deactivate and remove virtual environment
    deactivate
    rm -rf "$VENV_DIR"
    
    echo -e "${GREEN}✓ Wheel tests passed${NC}"
fi

# Step 7: Publish to PyPI
if [ "$PUBLISH_PYPI" = true ] && [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}Step 7: Publishing to PyPI${NC}"
    
    # Check if twine is installed
    if ! command -v twine &> /dev/null; then
        echo -e "${YELLOW}Installing twine...${NC}"
        pip install twine
    fi
    
    # Upload to PyPI
    echo -e "${YELLOW}Uploading to PyPI...${NC}"
    twine upload dist/*
    
    echo -e "${GREEN}✓ Published to PyPI${NC}"
elif [ "$PUBLISH_PYPI" = true ] && [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}Dry run: Skipping PyPI upload${NC}"
    echo -e "Would upload:"
    ls -lh dist/
fi

# Step 8: Generate checksums
echo -e "${GREEN}Step 8: Generating checksums${NC}"

CHECKSUM_FILE="checksums-$VERSION.txt"
rm -f "$CHECKSUM_FILE"

if [ "$BUILD_RUST" = true ]; then
    echo "=== Rust Library ===" >> "$CHECKSUM_FILE"
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sha256sum target/release/libarrow_quant_v2.so >> "$CHECKSUM_FILE"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        shasum -a 256 target/release/libarrow_quant_v2.dylib >> "$CHECKSUM_FILE"
    fi
    echo "" >> "$CHECKSUM_FILE"
fi

if [ "$BUILD_PYTHON" = true ]; then
    echo "=== Python Packages ===" >> "$CHECKSUM_FILE"
    if [[ "$OSTYPE" == "darwin"* ]]; then
        shasum -a 256 dist/* >> "$CHECKSUM_FILE"
    else
        sha256sum dist/* >> "$CHECKSUM_FILE"
    fi
fi

echo -e "${GREEN}✓ Checksums generated: $CHECKSUM_FILE${NC}"
cat "$CHECKSUM_FILE"

# Summary
echo ""
echo -e "${GREEN}=== Release Build Complete ===${NC}"
echo -e "Version: ${YELLOW}$VERSION${NC}"

if [ "$BUILD_RUST" = true ]; then
    echo -e "Rust library: ${YELLOW}target/release/$LIB_NAME${NC}"
fi

if [ "$BUILD_PYTHON" = true ]; then
    echo -e "Python packages: ${YELLOW}dist/${NC}"
fi

echo -e "Checksums: ${YELLOW}$CHECKSUM_FILE${NC}"

if [ "$PUBLISH_PYPI" = true ] && [ "$DRY_RUN" = false ]; then
    echo -e "${GREEN}✓ Published to PyPI${NC}"
fi

echo ""
echo -e "${GREEN}Next steps:${NC}"
echo -e "1. Review the built artifacts"
echo -e "2. Test the packages on different platforms"
echo -e "3. Create a git tag: ${YELLOW}git tag -a v$VERSION -m 'Release v$VERSION'${NC}"
echo -e "4. Push the tag: ${YELLOW}git push origin v$VERSION${NC}"
echo -e "5. GitHub Actions will automatically create a release"
