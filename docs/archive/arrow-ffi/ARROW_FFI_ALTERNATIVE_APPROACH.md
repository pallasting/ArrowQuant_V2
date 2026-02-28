# Arrow FFI Alternative Approach

**Date**: 2026-02-26  
**Issue**: Persistent heap corruption with Arrow C Data Interface export  
**Status**: 🔄 Investigating alternative approaches

## Problem Summary

Despite multiple attempts to fix the memory management in `export_recordbatch_to_pyarrow`, the heap corruption persists:

1. **V1**: Custom PyCapsule destructors → Double-free
2. **V2**: `Box::leak` + no destructors → Still crashes

This suggests the issue may be deeper than just memory management - possibly a fundamental incompatibility with how we're using the Arrow C Data Interface.

## Alternative Approach: Direct PyArrow Construction

Instead of using the C Data Interface for export, we can construct PyArrow objects directly from Python by:

1. **Export data as Python objects** (lists, bytes)
2. **Let Python construct PyArrow Table** using `pa.Table.from_pydict()`
3. **Avoid FFI export entirely** for the result

### Pros
- Avoids complex FFI memory management
- More straightforward and debuggable
- Less prone to platform-specific issues

### Cons
- Not true zero-copy for the result (but input is still zero-copy)
- Slightly more overhead for result construction
- Less elegant than pure FFI approach

## Implementation Plan

Modify `quantize_arrow()` to return Python dict instead of using FFI export:

```rust
// Instead of:
let result_pyarrow = arrow_ffi_helpers::export_recordbatch_to_pyarrow(py, &result_batch)?;

// Do this:
let result_dict = build_result_dict(py, &result_batch)?;
let pyarrow = py.import_bound("pyarrow")?;
let result_table = pyarrow.call_method1("Table.from_pydict", (result_dict,))?;
```

### Key Changes

1. **Keep FFI import** (working fine)
2. **Replace FFI export** with Python dict construction
3. **Let PyArrow handle** result table creation

## Trade-offs

| Aspect | FFI Export | Dict Export |
|--------|------------|-------------|
| Input zero-copy | ✅ Yes | ✅ Yes |
| Output zero-copy | ✅ Yes | ❌ No |
| Memory safety | ❌ Complex | ✅ Simple |
| Debugging | ❌ Hard | ✅ Easy |
| Performance | ⚡ Best | ⚡ Good |

## Decision

Given that:
- Input processing is the performance bottleneck (quantization computation)
- Output is typically much smaller than input (compressed)
- Stability is more important than marginal output performance gains

**Recommendation**: Use dict export approach for now, can optimize later if needed.

## Next Steps

1. Implement `build_result_dict()` helper function
2. Modify `quantize_arrow()` and `quantize_arrow_batch()` to use it
3. Test if this resolves the crash
4. If successful, document as the stable approach
5. Consider FFI export optimization as future enhancement

