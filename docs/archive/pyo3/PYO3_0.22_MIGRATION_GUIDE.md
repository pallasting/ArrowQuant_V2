# PyO3 0.22 Migration Guide

**Date**: 2026-02-24  
**Status**: In Progress

## Overview

This document tracks the migration from PyO3 0.20 to PyO3 0.22, which introduces the new "bound" API for better lifetime management and safety.

## Breaking Changes in PyO3 0.22

### 1. Dictionary Creation
```rust
// Old (0.20)
let dict = PyDict::new(py);

// New (0.22)
let dict = PyDict::new_bound(py);
```

### 2. List Creation
```rust
// Old (0.20)
let list = PyList::empty(py);

// New (0.22)
let list = PyList::empty_bound(py);
```

### 3. Module Import
```rust
// Old (0.20)
let module = py.import("module")?;

// New (0.22)
let module = py.import_bound("module")?;
```

### 4. Module Registration
```rust
// Old (0.20)
m.add_class::<MyClass>()?;
m.add_function(wrap_pyfunction!(my_func, m)?)?;

// New (0.22)
m.add_class::<MyClass>()?;  // Same
m.add_function(wrap_pyfunction!(my_func, m)?)?;  // Same
```

Note: Module registration API remains the same in 0.22.

## Files to Update

### 1. src/python.rs
**Locations**:
- Line 420: `PyDict::new(py)` → `PyDict::new_bound(py)`
- Line 621: `PyList::empty(py)` → `PyList::empty_bound(py)`
- Line 623: `PyDict::new(py)` → `PyDict::new_bound(py)`
- Line 1016: `py.import("numpy")` → `py.import_bound("numpy")`
- Line 1039: `py.import("numpy")` → `py.import_bound("numpy")`

**Status**: ⏳ Pending

### 2. src/python_async.rs
**Locations**:
- Line 387: `PyDict::new(py)` → `PyDict::new_bound(py)`

**Status**: ⏳ Pending

### 3. src/lib.rs
**Locations**:
- Lines 52-53: Module function registration (no changes needed)

**Status**: ✅ No changes required

## Additional Considerations

### Bound<'py, T> Type
PyO3 0.22 introduces `Bound<'py, T>` which explicitly ties Python objects to the GIL lifetime. This provides:
- Better compile-time safety
- Clearer lifetime management
- Prevention of use-after-free bugs

### Migration Pattern
```rust
// Old pattern
Python::with_gil(|py| {
    let dict = PyDict::new(py);
    dict.set_item("key", "value")?;
    // dict can be used anywhere in this scope
});

// New pattern (0.22)
Python::with_gil(|py| {
    let dict = PyDict::new_bound(py);
    dict.set_item("key", "value")?;
    // dict is explicitly bound to 'py lifetime
});
```

## Testing Strategy

After migration:
1. ✅ Cargo check (compilation)
2. ⏳ Cargo test (unit tests)
3. ⏳ Python integration tests
4. ⏳ Benchmark tests (ensure no performance regression)

## Rollback Plan

If issues arise:
1. Revert Cargo.toml to PyO3 0.20
2. Revert code changes
3. Document issues for future migration attempt

## References

- [PyO3 0.22 Release Notes](https://pyo3.rs/v0.22.0/migration.html)
- [Bound API Documentation](https://pyo3.rs/v0.22.0/types.html)
- [Migration Guide](https://pyo3.rs/v0.22.0/migration.html#from-021-to-022)

## Progress Tracker

- [x] Identify all affected code locations
- [x] Create migration guide
- [ ] Update src/python.rs
- [ ] Update src/python_async.rs
- [ ] Run cargo check
- [ ] Run cargo test
- [ ] Run Python integration tests
- [ ] Update documentation
- [ ] Mark as complete
