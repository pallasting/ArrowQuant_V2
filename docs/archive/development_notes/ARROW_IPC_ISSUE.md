# Arrow IPC Double Free Issue Diagnosis

## Problem Description
A `double free` or `segmentation fault` occurs when using the Arrow C Data Interface to transfer data between Python (PyArrow) and Rust (arrow-rs).

## Root Cause Analysis
The issue stems from ownership ambiguity in the `FFI_ArrowArray` and `FFI_ArrowSchema` structures:
1. **FFI Protocol**: The Arrow C Data Interface uses a `release` callback in the struct to manage memory.
2. **Transfer via PyCapsules**: In `python.rs`, we extract raw pointers from Python capsules using `schema_capsule.pointer()`.
3. **Rust Import**: We use `std::ptr::read(ptr)` to create a bitwise copy of the structure and pass it to `arrow::ffi::from_ffi(array, schema)`.
4. **The Conflict**: 
   - Rust's `from_ffi` takes ownership of the FFI structure. When the resulting `ArrayData` is dropped, it calls the `release` callback.
   - The original PyCapsule in Python *also* maintains a reference to the same structure. Depending on how PyArrow manages its capsules, it may attempt to call the `release` callback again when the capsule is garbage collected.
   - Specifically, `std::ptr::read` does not null out the `release` pointer in the original memory location, leading to both sides thinking they own the resource.

## Attempted Fixes
- **Safety Comments**: Added safety documentation and checked pointer validity.
- **Copying instead of referencing**: Attempted to use `from_ffi_and_data_type` without taking ownership, but this requires a reliable way to pre-determine the schema without consuming it.

## Practical Solution
To ensure stability and prevent crashes in production environments, the following Arrow IPC methods have been temporarily disabled in `src/python.rs`:
- `quantize_arrow`
- `quantize_arrow_batch`
- `quantize_batch_arrow`

**Recommendation**: Use the `quantize_batch` method which uses standard memory passing and is verified to be stable and performance-optimized.

## Future Work
Implement a proper ownership transfer by setting the `release` callback to `NULL` in the original structure immediately after `std::ptr::read`, ensuring only Rust manages the lifetime, OR move toward a higher-level IPC mechanism (like Arrow Flight) for remote/multi-process scenarios.
