# CI/CD Pipeline Documentation

## Overview

The ArrowQuant V2 project uses GitHub Actions for continuous integration and deployment. The Rust CI pipeline ensures code quality, correctness, and cross-platform compatibility.

## Workflows

### Rust CI Pipeline (`.github/workflows/rust-ci.yml`)

**Triggers:**
- Push to `main`, `master`, or `develop` branches
- Pull requests to `main` or `master` branches
- Only runs when Rust code changes (`ai_os_diffusion/arrow_quant_v2/**`)

**Jobs:**

#### 1. Test Suite (`test`)
- **Platforms:** Ubuntu, macOS, Windows
- **Rust versions:** Stable (all platforms), Nightly (Ubuntu only)
- **Actions:**
  - Runs all unit tests with `cargo test --verbose --all-features`
  - Runs release mode tests on Ubuntu stable
  - Runs documentation tests with `cargo test --doc`
- **Caching:** Cargo registry, index, and build artifacts

#### 2. Clippy Linting (`clippy`)
- **Platform:** Ubuntu
- **Actions:**
  - Runs Clippy linter with `cargo clippy --all-targets --all-features -- -D warnings`
  - Treats all warnings as errors
  - Checks both main code and tests

#### 3. Rustfmt Formatting (`fmt`)
- **Platform:** Ubuntu
- **Actions:**
  - Checks code formatting with `cargo fmt --all -- --check`
  - Ensures consistent code style across the project

#### 4. Build Check (`build`)
- **Platforms:** Ubuntu, macOS, Windows
- **Actions:**
  - Builds in debug mode: `cargo build --verbose --all-features`
  - Builds in release mode: `cargo build --release --verbose --all-features`
  - Verifies compilation on all platforms

#### 5. Code Coverage (`coverage`)
- **Platform:** Ubuntu
- **Actions:**
  - Generates coverage report using `cargo-tarpaulin`
  - Uploads to Codecov for tracking
  - Target: >85% code coverage

#### 6. Security Audit (`security-audit`)
- **Platform:** Ubuntu
- **Actions:**
  - Runs `cargo audit` to check for known vulnerabilities
  - Scans dependencies for security issues

#### 7. Python Bindings (`python-bindings`)
- **Platforms:** Ubuntu, macOS, Windows
- **Python versions:** 3.10, 3.11, 3.12
- **Actions:**
  - Builds PyO3 bindings with `maturin`
  - Installs and tests Python package
  - Runs Python integration tests

#### 8. Performance Benchmarks (`benchmark`)
- **Platform:** Ubuntu
- **Trigger:** Only on push to `main` branch
- **Actions:**
  - Runs Criterion benchmarks with `cargo bench`
  - Stores results as artifacts
  - Tracks performance over time

#### 9. All Checks Passed (`all-checks`)
- **Platform:** Ubuntu
- **Actions:**
  - Aggregates results from all required jobs
  - Fails if any required check fails
  - Provides single status check for PRs

## Local Development

### Running Tests Locally

```bash
# Navigate to project directory
cd ai_os_diffusion/arrow_quant_v2

# Run all tests
cargo test --verbose --all-features

# Run tests in release mode
cargo test --release --verbose

# Run specific test
cargo test test_name --verbose
```

### Running Clippy

```bash
# Run clippy with warnings as errors
cargo clippy --all-targets --all-features -- -D warnings

# Auto-fix some issues
cargo clippy --fix --all-targets --all-features
```

### Running Rustfmt

```bash
# Check formatting
cargo fmt --all -- --check

# Auto-format code
cargo fmt --all
```

### Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench quantization_bench
```

### Running Security Audit

```bash
# Install cargo-audit
cargo install cargo-audit

# Run audit
cargo audit
```

### Building Python Bindings

```bash
# Install maturin
pip install maturin

# Build and install
maturin develop --release

# Run Python tests
pytest tests/test_python_bindings.py -v
```

## CI/CD Best Practices

### Before Committing

1. **Format code:** `cargo fmt --all`
2. **Run clippy:** `cargo clippy --all-targets --all-features -- -D warnings`
3. **Run tests:** `cargo test --verbose --all-features`
4. **Check docs:** `cargo doc --no-deps --open`

### Pull Request Checklist

- [ ] All tests pass locally
- [ ] Code is formatted with `rustfmt`
- [ ] No clippy warnings
- [ ] Documentation is updated
- [ ] New tests added for new features
- [ ] Benchmarks run (if performance-critical changes)

### Debugging CI Failures

#### Test Failures
- Check test output in GitHub Actions logs
- Run tests locally with same configuration
- Use `RUST_BACKTRACE=1` for detailed error traces

#### Clippy Failures
- Run `cargo clippy` locally
- Fix warnings or add `#[allow(clippy::lint_name)]` with justification
- Never disable warnings globally

#### Format Failures
- Run `cargo fmt --all` locally
- Commit formatting changes separately

#### Build Failures
- Check for platform-specific issues
- Test on multiple platforms if possible
- Review dependency versions

## Performance Tracking

Benchmark results are stored as artifacts and can be downloaded from the Actions tab. Compare results across commits to track performance changes.

## Coverage Reports

Code coverage reports are uploaded to Codecov. View detailed coverage at:
- https://codecov.io/gh/YOUR_ORG/YOUR_REPO

Target: >85% code coverage for all Rust code.

## Security

Security audits run on every commit. If vulnerabilities are found:
1. Review the `cargo audit` output
2. Update vulnerable dependencies
3. If no fix available, document the risk and mitigation

## Caching

The CI pipeline uses GitHub Actions cache to speed up builds:
- Cargo registry cache
- Cargo index cache
- Build artifacts cache

Cache is invalidated when `Cargo.lock` changes.

## Matrix Testing

The pipeline tests on multiple configurations:
- **OS:** Ubuntu, macOS, Windows
- **Rust:** Stable, Nightly (Ubuntu only)
- **Python:** 3.10, 3.11, 3.12 (for bindings)

This ensures cross-platform compatibility and catches platform-specific bugs early.

## Continuous Deployment

Currently, the Rust CI pipeline focuses on testing and validation. Deployment workflows can be added for:
- Publishing to crates.io
- Building release binaries
- Publishing Python wheels to PyPI

## Troubleshooting

### Common Issues

**Issue:** Tests timeout
- **Solution:** Increase timeout in workflow or optimize slow tests

**Issue:** Cache miss
- **Solution:** Check if `Cargo.lock` changed, cache will rebuild

**Issue:** Platform-specific test failure
- **Solution:** Use conditional compilation or platform-specific test attributes

**Issue:** Python binding build fails
- **Solution:** Check Python version compatibility, ensure maturin is up to date

## Contact

For CI/CD issues or questions, please open an issue on GitHub.
