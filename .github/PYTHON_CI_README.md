# Python CI Pipeline - ArrowQuant V2

## Overview

The Python CI pipeline tests PyO3 Python bindings, integration tests, and deployment scripts across multiple Python versions and platforms.

## Pipeline Structure

### Jobs

1. **test-python-bindings** - Test PyO3 bindings on all platforms
   - Platforms: Ubuntu, macOS, Windows
   - Python: 3.10, 3.11, 3.12
   - Tests: PyO3 bindings, async quantization, scripts

2. **integration-tests** - Run integration test suite
   - Platform: Ubuntu
   - Python: 3.10, 3.11, 3.12
   - Tests: All Python tests, async tests

3. **property-based-tests** - Run property-based tests
   - Platform: Ubuntu
   - Python: 3.10, 3.11, 3.12
   - Framework: Hypothesis

4. **coverage** - Generate code coverage reports
   - Platform: Ubuntu
   - Python: 3.11
   - Tool: pytest-cov
   - Upload: Codecov

5. **lint-and-format** - Check code quality
   - Platform: Ubuntu
   - Tools: Black, Flake8, Mypy

6. **test-scripts** - Validate deployment scripts
   - Platform: Ubuntu
   - Scripts: quantize_diffusion.py, validate_quantization.py, package_model.py

7. **test-examples** - Validate example scripts
   - Platform: Ubuntu
   - Examples: async_quantization_example.py, granularity_allocation_example.py

8. **all-python-checks** - Gate check for required jobs

## Trigger Conditions

### Push Events
- Branches: main, master, develop
- Paths:
  - `ai_os_diffusion/arrow_quant_v2/**/*.py`
  - `ai_os_diffusion/arrow_quant_v2/tests/**`
  - `ai_os_diffusion/arrow_quant_v2/scripts/**`
  - `ai_os_diffusion/arrow_quant_v2/pyproject.toml`
  - `.github/workflows/python-ci.yml`

### Pull Request Events
- Target branches: main, master
- Same path filters as push events

## Running Tests Locally

### Install Dependencies
```bash
cd ai_os_diffusion/arrow_quant_v2

# Install Python dependencies
pip install maturin pytest pytest-asyncio pytest-cov numpy hypothesis pyyaml

# Build Python package
maturin develop --release
```

### Run Tests
```bash
# Run all Python tests
pytest tests/ -v

# Run specific test files
pytest tests/test_python_bindings.py -v
pytest tests/test_async_quantization.py -v
pytest tests/test_quantize_script.py -v

# Run with coverage
pytest tests/ --cov=. --cov-report=term --cov-report=xml
```

### Run Linting
```bash
# Install linting tools
pip install black flake8 mypy

# Check formatting
black --check tests/ scripts/ examples/

# Lint code
flake8 tests/ scripts/ examples/ --max-line-length=120

# Type check
mypy tests/ scripts/ examples/ --ignore-missing-imports
```

## Caching Strategy

### Pip Packages
- Path: `~/.cache/pip`
- Key: `{os}-pip-{python-version}-{pyproject.toml hash}`
- Benefit: ~30 seconds saved per job

### Cargo Registry
- Path: `~/.cargo/registry`
- Key: `{os}-cargo-registry-{Cargo.lock hash}`
- Benefit: ~1-2 minutes saved per job

### Cargo Build
- Path: `ai_os_diffusion/arrow_quant_v2/target`
- Key: `{os}-cargo-build-{python-version}-{Cargo.lock hash}`
- Benefit: ~5-10 minutes saved per job

## Test Coverage

### Current Coverage
- **Python binding tests**: 26 tests
- **Async quantization tests**: 50+ tests
- **Script tests**: 15+ tests
- **Total**: 90+ Python tests

### Coverage Targets
- **Minimum**: 80% Python code coverage
- **Goal**: 90% Python code coverage
- **Tracking**: Codecov with `python` flag

## Known Issues

### Async Tests
- Some async tests may fail intermittently
- Marked with `continue-on-error: true`
- Does not block pipeline

### Windows Builds
- Slower than Linux/macOS (~2x)
- Aggressive caching helps
- All tests still run

## Troubleshooting

### Build Failures

**Issue**: Maturin build fails
```bash
# Solution: Clean build and retry
cargo clean
maturin develop --release
```

**Issue**: Import errors
```bash
# Solution: Ensure package is installed
pip install -e .
# or
maturin develop --release
```

### Test Failures

**Issue**: Tests fail locally but pass in CI
```bash
# Solution: Check Python version
python --version  # Should be 3.10, 3.11, or 3.12

# Solution: Clean pytest cache
pytest --cache-clear
```

**Issue**: Coverage report not generated
```bash
# Solution: Install pytest-cov
pip install pytest-cov

# Run with coverage
pytest tests/ --cov=. --cov-report=term
```

## Integration with Rust CI

### Complementary Pipelines
- **Rust CI**: Tests Rust code and PyO3 from Rust side
- **Python CI**: Tests Python code and PyO3 from Python side

### Shared Resources
- Both pipelines share cargo caches
- Both upload to Codecov (different flags)
- Both run on same trigger conditions

### Coverage Tracking
- **Rust coverage**: Flag `rust`
- **Python coverage**: Flag `python`
- **Combined view**: Codecov dashboard

## Maintenance

### Regular Updates
- Update GitHub Actions versions quarterly
- Update Python package versions monthly
- Review and update caching strategy as needed

### Adding New Tests
1. Add test file to `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use pytest fixtures and markers
4. Update this README if new test category

### Modifying Pipeline
1. Edit `.github/workflows/python-ci.yml`
2. Test locally first
3. Validate YAML syntax: `python -c "import yaml; yaml.safe_load(open('.github/workflows/python-ci.yml'))"`
4. Push to feature branch and verify CI runs

## Performance

### Typical Run Times
- **test-python-bindings**: ~10-15 minutes (9 parallel jobs)
- **integration-tests**: ~8-12 minutes (3 parallel jobs)
- **property-based-tests**: ~8-12 minutes (3 parallel jobs)
- **coverage**: ~10-15 minutes
- **lint-and-format**: ~2-3 minutes
- **test-scripts**: ~8-10 minutes
- **test-examples**: ~8-10 minutes

### Total Pipeline Time
- **With cache**: ~15-20 minutes (parallel execution)
- **Without cache**: ~30-40 minutes (parallel execution)

## Future Enhancements

### Planned
1. **Benchmark CI** (Task 20.3): Performance regression testing
2. **Release automation** (Task 20.4): Automated PyPI publishing

### Potential
1. Nightly builds against Python dev versions
2. Coverage enforcement (fail if below threshold)
3. Performance benchmarks in CI
4. Automated dependency updates (Dependabot)

## Resources

- **Workflow file**: `.github/workflows/python-ci.yml`
- **Rust CI**: `.github/workflows/rust-ci.yml`
- **Test files**: `ai_os_diffusion/arrow_quant_v2/tests/`
- **Scripts**: `ai_os_diffusion/arrow_quant_v2/scripts/`
- **Examples**: `ai_os_diffusion/arrow_quant_v2/examples/`

## Support

For issues with the CI pipeline:
1. Check this README for troubleshooting
2. Review CI logs in GitHub Actions
3. Test locally to reproduce issues
4. Check Codecov for coverage reports
5. Consult Rust CI README for related issues
