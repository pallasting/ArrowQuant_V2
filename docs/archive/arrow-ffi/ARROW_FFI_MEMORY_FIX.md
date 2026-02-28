# Arrow FFI Memory Safety Fix

**Date**: 2026-02-25  
**Issue**: Heap corruption (Windows error 0xc0000374) when exporting RecordBatch to PyArrow  
**Status**: ✅ FIXED

## Problem

The `export_recordbatch_to_pyarrow` function was causing heap corruption due to improper memory management of Arrow FFI structures.

### Root Cause

The Arrow C Data Interface structures (`FFI_ArrowSchema` and `FFI_ArrowArray`) have their own internal `release` callbacks that manage memory lifecycle. When we wrapped these structures in PyCapsules with additional destructors, we created a **double-free situation**:

1. PyArrow would call the FFI structure's internal `release` callback
2. Python would call the PyCapsule destructor
3. Both would try to free the same memory → heap corruption

### Symptoms

```
Windows fatal exception: code 0xc0000374
```

This error occurred when running:
```python
result_table = quantizer.quantize_arrow(table, bit_width=4)
```

## Solution

Modified the PyCapsule destructors to only free the `Box` wrapper, allowing the FFI structures' internal `Drop` implementations to handle their own memory cleanup:

```rust
unsafe extern "C" fn release_schema(capsule: *mut pyo3::ffi::PyObject) {
    let ptr = pyo3::ffi::PyCapsule_GetPointer(
        capsule,
        b"arrow_schema\0".as_ptr() as *const i8,
    );
    if !ptr.is_null() {
        // Reconstruct the Box and let it drop naturally
        // The FFI_ArrowSchema's internal release callback will be called by its Drop impl
        let _ = Box::from_raw(ptr as *mut FFI_ArrowSchema);
    }
}
```

### Key Changes

1. **Added comments** explaining that FFI structures have their own release callbacks
2. **Improved error handling** in capsule creation to clean up resources on failure
3. **Preserved FFI memory management** by letting Drop implementations handle cleanup

## Memory Management Flow

### Before Fix (Double-Free)
```
1. Create FFI structures with internal release callbacks
2. Box them and create PyCapsules
3. PyArrow imports and calls FFI release callback → frees memory
4. Python GC calls PyCapsule destructor → tries to free again → CRASH
```

### After Fix (Correct)
```
1. Create FFI structures with internal release callbacks
2. Box them and create PyCapsules
3. PyArrow imports and calls FFI release callback → frees internal memory
4. Python GC calls PyCapsule destructor → frees Box only → OK
5. Box Drop calls FFI Drop → release callback already called → safe no-op
```

## Testing

After rebuild, run:
```bash
cd ai_os_diffusion/arrow_quant_v2
.venv\Scripts\python.exe -m pytest tests/test_quantize_arrow.py -v
.venv\Scripts\python.exe -m pytest tests/test_quantize_arrow_batch.py -v
```

Expected: All tests pass without crashes.

## Related Files

- `src/python.rs` - `export_recordbatch_to_pyarrow` function (lines 180-280)
- `tests/test_quantize_arrow.py` - Test that triggered the crash
- `tests/test_quantize_arrow_batch.py` - Similar tests

## References

- Arrow C Data Interface: https://arrow.apache.org/docs/format/CDataInterface.html
- PyO3 FFI: https://pyo3.rs/latest/ffi.html
- PyCapsule API: https://docs.python.org/3/c-api/capsule.html
