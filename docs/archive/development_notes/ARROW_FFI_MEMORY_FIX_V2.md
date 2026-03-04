# Arrow FFI Memory Safety Fix V2

**Date**: 2026-02-26  
**Issue**: Heap corruption (Windows error 0xc0000374) when exporting RecordBatch to PyArrow  
**Status**: 🔄 Testing new approach

## Problem Analysis

The heap corruption was caused by improper memory management in the Arrow C Data Interface export.

### Root Cause (Deeper Analysis)

The Arrow C Data Interface uses a specific ownership model:

1. **FFI Structures** (`FFI_ArrowSchema`, `FFI_ArrowArray`):
   - Created by `arrow::ffi::to_ffi()`
   - Have internal `release` callback pointers
   - The release callback is responsible for freeing the underlying Arrow data
   - These structures are meant to be **consumed** (moved), not copied

2. **PyCapsule Ownership**:
   - PyCapsules transfer ownership of C pointers to Python
   - When Python GC releases the capsule, it calls the destructor (if provided)
   - PyArrow's `_import_from_c` expects to take ownership of the FFI structures

3. **The Double-Free Problem**:
   - Initial approach: Created PyCapsule with custom destructor that reconstructed Box and dropped it
   - PyArrow would call the FFI release callback when importing
   - Then our PyCapsule destructor would also try to drop the Box
   - Result: Double-free → heap corruption

## Solution V2: Let PyArrow Manage Memory

The correct approach is to **not provide a PyCapsule destructor** and let PyArrow's FFI import handle all memory management:

```rust
// Box the FFI structures
let schema_box = Box::new(ffi_schema);
let array_box = Box::new(ffi_array);

// Leak the boxes to get raw pointers
// PyArrow will manage the memory via FFI release callbacks
let schema_ptr = Box::leak(schema_box) as *mut FFI_ArrowSchema;
let array_ptr = Box::leak(array_box) as *mut FFI_ArrowArray;

// Create PyCapsules WITHOUT destructors
let schema_capsule = unsafe {
    pyo3::ffi::PyCapsule_New(
        schema_ptr as *mut std::ffi::c_void,
        b"arrow_schema\0".as_ptr() as *const i8,
        None,  // No destructor - FFI release callback handles it
    )
};
```

### Key Changes

1. **Use `Box::leak`** instead of `Box::into_raw`
   - More explicit about intentionally leaking memory
   - Memory will be reclaimed by FFI release callback

2. **No PyCapsule destructor** (`None` instead of `Some(release_fn)`)
   - PyArrow's `_import_from_c` takes ownership
   - FFI release callbacks handle all cleanup
   - No risk of double-free

3. **Single ownership path**:
   ```
   Rust creates FFI → Box::leak → PyCapsule (no destructor) → 
   PyArrow imports → FFI release callback frees memory
   ```

## Memory Management Flow

### New Approach (Correct)
```
1. Rust: Create FFI structures with release callbacks
2. Rust: Box them and leak to get stable pointers
3. Rust: Create PyCapsules with NO destructors
4. Python: PyArrow._import_from_c takes ownership
5. PyArrow: Calls FFI release callbacks when done
6. FFI: Release callbacks free the underlying Arrow data
```

## Testing

After rebuild, run:
```bash
cd ai_os_diffusion/arrow_quant_v2
.venv\Scripts\python.exe -m pytest tests/test_quantize_arrow.py -v
.venv\Scripts\python.exe -m pytest tests/test_quantize_arrow_batch.py -v
```

Expected: All tests pass without crashes.

## Technical References

- **Arrow C Data Interface**: https://arrow.apache.org/docs/format/CDataInterface.html
  - Section on "Memory Management" explains the release callback pattern
- **Box::leak**: https://doc.rust-lang.org/std/boxed/struct.Box.html#method.leak
  - Intentionally leak memory that will be managed elsewhere
- **PyCapsule**: https://docs.python.org/3/c-api/capsule.html
  - Destructor is optional when ownership is transferred

## Related Files

- `src/python.rs` - `export_recordbatch_to_pyarrow` function
- `tests/test_quantize_arrow.py` - Test that triggered the crash
- `ARROW_FFI_MEMORY_FIX.md` - Previous fix attempt (V1)

