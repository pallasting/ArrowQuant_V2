# Dependency Upgrade Plan - ArrowQuant V2

**Date**: 2026-02-24  
**Purpose**: Upgrade all dependencies to latest versions for production deployment

## Current Issues

1. **PyO3 Version Mismatch**: Code written for PyO3 0.20, but needs 0.22+ for production
2. **Rust Borrow Checker Errors**: 8+ errors in optimizer.rs due to simultaneous mutable/immutable borrows
3. **Outdated Dependencies**: Multiple dependencies need updates for security and compatibility

## Upgrade Strategy

### Phase 1: Core Dependencies
- PyO3: 0.20 → 0.22 (latest stable)
- Arrow/Parquet: 51.0 → 53.0 (latest)
- ndarray: 0.15 → 0.16
- tokio: 1.35 → 1.40
- rayon: 1.8 → 1.10

### Phase 2: Supporting Dependencies
- serde: 1.0 → latest patch
- thiserror: 1.0 → latest patch
- pyo3-asyncio: 0.20 → 0.22
- simsimd: 4.3 → 5.x (if available)
- lru: 0.12 → 0.13

### Phase 3: Dev Dependencies
- criterion: 0.5 → 0.6
- proptest: 1.4 → 1.5

## PyO3 0.22 API Changes

### Breaking Changes
1. `PyDict::new()` → `PyDict::new_bound()`
2. `PyList::empty()` → `PyList::empty_bound()`
3. `py.import()` → `py.import_bound()`
4. `m.add_class()` → requires new bound API
5. `m.add_function()` → requires new bound API
6. `PyObject` → `Bound<'py, PyAny>` in many contexts

### Migration Pattern
```rust
// Old (PyO3 0.20)
let dict = PyDict::new(py);
let list = PyList::empty(py);
let module = py.import("module")?;

// New (PyO3 0.22)
let dict = PyDict::new_bound(py);
let list = PyList::empty_bound(py);
let module = py.import_bound("module")?;
```

## Rust Borrow Checker Fixes

### Issue: Simultaneous Mutable/Immutable Borrows in optimizer.rs

**Problem Areas**:
1. `optimize_params()` - already fixed with buffer reuse
2. `compute_gradients()` - needs refactoring
3. `quantize_with_params_inplace()` - needs careful lifetime management

**Solution**:
- Split mutable operations into separate scopes
- Use interior mutability (RefCell/Mutex) where appropriate
- Restructure to avoid holding multiple borrows simultaneously

## Testing Strategy

1. **Unit Tests**: Run after each dependency upgrade
2. **Integration Tests**: Verify Python bindings work
3. **Benchmark Tests**: Ensure no performance regression
4. **Compilation**: Must compile without warnings

## Rollback Plan

- Keep backup of current Cargo.toml
- Use git branches for each phase
- Test thoroughly before merging

## Success Criteria

- ✅ All dependencies at latest stable versions
- ✅ Zero compilation errors
- ✅ Zero compilation warnings
- ✅ All tests passing
- ✅ Python bindings functional
- ✅ No performance regression (>5%)
- ✅ Ready for production deployment

## Timeline

- Phase 1: 2 hours (core dependencies + PyO3 migration)
- Phase 2: 1 hour (supporting dependencies)
- Phase 3: 30 minutes (dev dependencies)
- Testing: 1 hour
- **Total**: ~4.5 hours

## Next Steps

1. Backup current Cargo.toml
2. Update Cargo.toml with new versions
3. Fix PyO3 API compatibility in src/python.rs and src/python_async.rs
4. Fix borrow checker errors in src/thermodynamic/optimizer.rs
5. Run cargo check
6. Run cargo test
7. Run Python integration tests
8. Document any breaking changes
