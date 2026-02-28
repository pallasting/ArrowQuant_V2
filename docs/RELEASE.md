# Release Guide - ArrowQuant V2

This guide explains how to create releases for ArrowQuant V2, including building Rust libraries, Python wheels, and publishing to PyPI.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Release Process](#release-process)
3. [Automated Release (GitHub Actions)](#automated-release-github-actions)
4. [Manual Release](#manual-release)
5. [Publishing to PyPI](#publishing-to-pypi)
6. [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Tools

- **Rust**: 1.70+ with `cargo`
- **Python**: 3.10+
- **Maturin**: Python wheel builder for Rust projects
  ```bash
  pip install maturin
  ```
- **Twine**: For uploading to PyPI (optional)
  ```bash
  pip install twine
  ```

### Optional Tools

- **cross**: For cross-compilation to ARM64
  ```bash
  cargo install cross --git https://github.com/cross-rs/cross
  ```

## Release Process

### Version Numbering

We follow [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

Example: `0.1.0` → `0.2.0` (new features) → `0.2.1` (bug fix)

### Release Checklist

Before creating a release:

- [ ] All tests passing (`cargo test --all-features`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated with release notes
- [ ] Version bumped in `Cargo.toml` and `pyproject.toml`
- [ ] Code reviewed and approved
- [ ] Performance benchmarks run (no regressions)

## Automated Release (GitHub Actions)

The recommended way to create releases is through GitHub Actions.

### Triggering a Release

#### Option 1: Git Tag (Recommended)

1. **Create and push a version tag:**
   ```bash
   git tag -a v0.1.0 -m "Release v0.1.0"
   git push origin v0.1.0
   ```

2. **GitHub Actions will automatically:**
   - Build Rust libraries for all platforms
   - Build Python wheels for all platforms
   - Run tests on all platforms
   - Create a GitHub release with artifacts
   - Optionally publish to PyPI (if configured)

#### Option 2: Manual Workflow Dispatch

1. Go to **Actions** → **Release - ArrowQuant V2**
2. Click **Run workflow**
3. Enter version (e.g., `0.1.0`)
4. Choose whether to publish to PyPI
5. Click **Run workflow**

### Release Artifacts

The automated release creates:

#### Rust Libraries
- `libarrow_quant_v2.so` (Linux x86_64)
- `libarrow_quant_v2.so` (Linux ARM64)
- `libarrow_quant_v2.dylib` (macOS x86_64)
- `libarrow_quant_v2.dylib` (macOS ARM64/Apple Silicon)
- `arrow_quant_v2.dll` (Windows x86_64)

#### Python Wheels
- Wheels for Python 3.10, 3.11, 3.12
- Platforms: Linux (x86_64, ARM64), macOS (x86_64, ARM64), Windows (x86_64)
- Source distribution (`.tar.gz`)

#### Checksums
- SHA256 checksums for all artifacts

## Manual Release

For local testing or when GitHub Actions is unavailable.

### Using the Release Script

```bash
cd ai_os_diffusion/arrow_quant_v2

# Build everything (Rust + Python)
./scripts/release.sh --version 0.1.0

# Build Rust library only
./scripts/release.sh --version 0.1.0 --rust-only

# Build Python wheels only
./scripts/release.sh --version 0.1.0 --python-only

# Skip tests (faster)
./scripts/release.sh --version 0.1.0 --no-tests

# Dry run (no publishing)
./scripts/release.sh --version 0.1.0 --dry-run

# Build and publish to PyPI
./scripts/release.sh --version 0.1.0 --publish
```

### Manual Build Steps

#### 1. Update Version

**Cargo.toml:**
```toml
[package]
version = "0.1.0"
```

**pyproject.toml:**
```toml
[project]
version = "0.1.0"
```

#### 2. Build Rust Library

```bash
cd ai_os_diffusion/arrow_quant_v2

# Clean previous builds
cargo clean

# Build release
cargo build --release --all-features

# Library location:
# - Linux: target/release/libarrow_quant_v2.so
# - macOS: target/release/libarrow_quant_v2.dylib
# - Windows: target/release/arrow_quant_v2.dll
```

#### 3. Build Python Wheels

```bash
# Install maturin
pip install maturin

# Build wheels
maturin build --release --features python --out dist

# Build source distribution
maturin sdist --out dist

# List artifacts
ls -lh dist/
```

#### 4. Test Wheels

```bash
# Create test environment
python3 -m venv venv-test
source venv-test/bin/activate  # On Windows: venv-test\Scripts\activate

# Install wheel
pip install dist/arrow_quant_v2-*.whl

# Test import
python -c "import arrow_quant_v2; print('Success')"

# Run tests
pip install pytest
pytest tests/test_python_bindings.py -v

# Cleanup
deactivate
rm -rf venv-test
```

#### 5. Generate Checksums

```bash
# Linux/macOS
sha256sum dist/* > checksums.txt

# macOS alternative
shasum -a 256 dist/* > checksums.txt

# Windows (PowerShell)
Get-FileHash dist\* -Algorithm SHA256 | Format-List > checksums.txt
```

## Publishing to PyPI

### Prerequisites

1. **PyPI Account**: Create account at [pypi.org](https://pypi.org)
2. **API Token**: Generate at [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/)
3. **Configure credentials**:
   ```bash
   # Create ~/.pypirc
   cat > ~/.pypirc << EOF
   [pypi]
   username = __token__
   password = pypi-YOUR-API-TOKEN-HERE
   EOF
   
   chmod 600 ~/.pypirc
   ```

### Publishing

#### Option 1: Automated (GitHub Actions)

Configure GitHub repository secrets:
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add secret: `PYPI_API_TOKEN` with your PyPI token
3. Push a version tag (e.g., `v0.1.0`)
4. GitHub Actions will automatically publish

#### Option 2: Manual (Twine)

```bash
# Install twine
pip install twine

# Upload to PyPI
twine upload dist/*

# Upload to Test PyPI (for testing)
twine upload --repository testpypi dist/*
```

#### Option 3: Using Release Script

```bash
./scripts/release.sh --version 0.1.0 --publish
```

### Verifying Publication

```bash
# Install from PyPI
pip install arrow-quant-v2

# Test
python -c "import arrow_quant_v2; print('Success')"
```

## Cross-Platform Builds

### Building for ARM64 (Linux)

```bash
# Install cross
cargo install cross --git https://github.com/cross-rs/cross

# Build for ARM64
cross build --release --target aarch64-unknown-linux-gnu --all-features
```

### Building for Apple Silicon (macOS)

```bash
# Add ARM64 target
rustup target add aarch64-apple-darwin

# Build
cargo build --release --target aarch64-apple-darwin --all-features
```

### Universal Binary (macOS)

```bash
# Build for both architectures
cargo build --release --target x86_64-apple-darwin --all-features
cargo build --release --target aarch64-apple-darwin --all-features

# Create universal binary
lipo -create \
  target/x86_64-apple-darwin/release/libarrow_quant_v2.dylib \
  target/aarch64-apple-darwin/release/libarrow_quant_v2.dylib \
  -output libarrow_quant_v2.dylib
```

## Troubleshooting

### Build Failures

**Problem**: Compilation errors

**Solution**:
```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release --all-features
```

**Problem**: Missing dependencies

**Solution**:
```bash
# Install build dependencies (Ubuntu/Debian)
sudo apt-get install build-essential pkg-config libssl-dev

# macOS
xcode-select --install
```

### Wheel Build Failures

**Problem**: Maturin build fails

**Solution**:
```bash
# Update maturin
pip install --upgrade maturin

# Check Rust toolchain
rustup show

# Rebuild
maturin build --release --features python
```

**Problem**: Import error after installation

**Solution**:
```bash
# Check Python version compatibility
python --version  # Should be 3.10+

# Reinstall with verbose output
pip install --force-reinstall --verbose dist/*.whl
```

### PyPI Upload Failures

**Problem**: Authentication error

**Solution**:
```bash
# Verify credentials
cat ~/.pypirc

# Use token authentication
twine upload --username __token__ --password pypi-YOUR-TOKEN dist/*
```

**Problem**: Package already exists

**Solution**:
```bash
# Bump version and rebuild
# Edit Cargo.toml and pyproject.toml
./scripts/release.sh --version 0.1.1
```

### Platform-Specific Issues

**Linux**: Missing GLIBC

**Solution**: Build on older Linux (e.g., Ubuntu 20.04) or use manylinux containers

**macOS**: Code signing issues

**Solution**:
```bash
# Remove quarantine attribute
xattr -d com.apple.quarantine libarrow_quant_v2.dylib
```

**Windows**: DLL not found

**Solution**: Ensure Visual C++ Redistributable is installed

## Release Checklist

### Pre-Release

- [ ] All tests passing
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped
- [ ] Performance benchmarks run
- [ ] Security audit passed (`cargo audit`)

### Release

- [ ] Create git tag
- [ ] Push tag to GitHub
- [ ] Verify GitHub Actions workflow
- [ ] Download and test artifacts
- [ ] Verify PyPI publication (if applicable)

### Post-Release

- [ ] Update documentation website
- [ ] Announce release (blog, social media)
- [ ] Monitor issue tracker for bug reports
- [ ] Update dependent projects

## Additional Resources

- [Maturin Documentation](https://www.maturin.rs/)
- [PyO3 Guide](https://pyo3.rs/)
- [Cargo Book](https://doc.rust-lang.org/cargo/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Semantic Versioning](https://semver.org/)

## Support

For issues or questions:
- GitHub Issues: [github.com/ai-os/arrow-quant-v2/issues](https://github.com/ai-os/arrow-quant-v2/issues)
- Documentation: [docs/](../docs/)
- Examples: [examples/](../examples/)
