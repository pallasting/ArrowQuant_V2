# PyO3 Export Issue - Root Cause Analysis

## Observation

Only the first 5 methods in the pymethods block are exported:
1. `new()` - line 608
2. `quantize_diffusion_model()` - line 646  
3. `validate_quality()` - line 732
4. `quantize_from_safetensors()` - line 820
5. `quantize()` - line 899

Methods after line 899 are NOT exported:
- `simple_test()` - line 926
- `test_method()` - line 931
- `get_markov_metrics()` - line 963
- `quantize_arrow()` - line 1053
- `quantize_arrow_batch()` - line 1411
- `quantize_batch()` - line 1782
- `quantize_batch_with_progress()` - line 2039

## Key Finding

**The cutoff happens exactly after the `quantize()` method at line 899.**

This suggests that something about the `quantize()` method or what comes immediately after it is causing PyO3 to stop processing methods.

## Hypothesis

Looking at the code structure, I notice that the methods added by the consolidation script (lines 926+) were simply appended after the existing methods. However, PyO3 might be:

1. **Hitting a size limit** - The pymethods block might be too large
2. **Encountering a syntax issue** - There might be a subtle syntax problem that causes PyO3's macro processor to stop
3. **Cache issue** - PyO3 might be using a cached version of the method list

## Investigation Needed

1. Check if there's a blank line or formatting issue between line 899 and 926
2. Try moving one of the non-exported methods (like `simple_test`) to before `quantize()` to see if position matters
3. Check PyO3 documentation for known limitations on pymethods block size
4. Try splitting into multiple pymethods blocks (even though this shouldn't be necessary)

## Recommended Fix

Since the consolidation approach isn't working, try a different strategy:

### Option 1: Manual Method Addition
Manually copy each method definition into the first pymethods block, ensuring proper formatting and spacing.

### Option 2: Multiple pymethods Blocks (Workaround)
Even though PyO3 should support multiple blocks in newer versions, try creating 2-3 smaller blocks:
- Block 1: Core methods (new, quantize_diffusion_model, validate_quality, quantize_from_safetensors, quantize)
- Block 2: Arrow methods (quantize_arrow, quantize_arrow_batch)  
- Block 3: Batch methods (quantize_batch, quantize_batch_with_progress)

### Option 3: Check PyO3 Version
Verify we're using PyO3 0.22.6 correctly and check if there are known issues with large pymethods blocks.

## Next Steps

1. Read lines 899-930 carefully to find any syntax issues
2. Try Option 2 (multiple blocks) as a quick workaround
3. If that fails, manually reconstruct the pymethods block from scratch
