# PyO3 Method Export Issue

## Problem

The `quantize_arrow()`, `quantize_arrow_batch()`, and `quantize_batch()` methods are implemented in `src/python.rs` but are not being exported to Python. When checking `dir(ArrowQuantV2())`, these methods don't appear.

## Root Cause

The issue is with multiple `#[pymethods]` blocks in PyO3 0.22. The code has 4 separate `#[pymethods]` blocks:

1. **First block** (lines 594-924): Contains `new()`, `quantize()`, `quantize_diffusion_model()`, etc. ✅ **EXPORTED**
2. **Second block** (lines 927-938): Contains test methods `simple_test()`, `test_method()` ❌ **NOT EXPORTED**
3. **Third block** (lines 941-943): Empty (commented out numpy methods) ❌ **REMOVED**
4. **Fourth block** (lines 944-2230): Contains `get_markov_metrics()`, `quantize_arrow()`, `quantize_arrow_batch()`, `quantize_batch()` ❌ **NOT EXPORTED**

**Only the first `#[pymethods]` block is being recognized by PyO3.**

## Solution

Consolidate all methods into a single `#[pymethods]` block. This requires:

1. Remove the second, third, and fourth `#[pymethods]` impl blocks
2. Move all methods from blocks 2-4 into the first block (before the closing brace at line 924)
3. Ensure no duplicate method names
4. Rebuild with `maturin develop --release`

## Workaround (Temporary)

Until the consolidation is complete, the methods can be accessed by:

1. Manually moving the method definitions
2. Or using the internal implementation directly (not recommended for production)

## Implementation Status

### Completed ✅
- All method implementations are complete and compile successfully
- 49 test cases written (34 Arrow IPC tests + 15 batch API tests)
- Zero-copy implementation via Arrow C Data Interface
- Batch processing with parallel execution
- Progress callback support

### Blocked ⏸️
- Test execution (methods not exported)
- Performance benchmarking (methods not exported)
- Documentation examples (methods not exported)

## Next Steps

1. **Consolidate pymethods blocks** (HIGH PRIORITY)
   - Move all methods from blocks 2-4 into block 1
   - Remove empty/duplicate blocks
   - Rebuild and verify export

2. **Run test suite** (after consolidation)
   - `pytest tests/test_arrow_ffi_integration.py -v`
   - `pytest tests/test_quantize_arrow.py -v`
   - `pytest tests/test_quantize_arrow_batch.py -v`
   - `pytest tests/test_quantize_batch.py -v`

3. **Performance benchmarking**
   - Verify 30x speedup target
   - Measure PyO3 overhead reduction

## Technical Details

### PyO3 0.22 Behavior

PyO3 0.22 supports multiple `#[pymethods]` blocks, but there may be issues with:
- Method registration order
- Module initialization
- Compilation/linking

The safest approach is to use a single `#[pymethods]` block per impl.

### File Locations

- **Implementation**: `ai_os_diffusion/arrow_quant_v2/src/python.rs`
- **Tests**: `ai_os_diffusion/arrow_quant_v2/tests/test_*.py`
- **Module registration**: `ai_os_diffusion/arrow_quant_v2/src/lib.rs`

## References

- PyO3 documentation: https://pyo3.rs/v0.22.0/class.html#pymethods
- Issue discussion: Multiple pymethods blocks not all exported
