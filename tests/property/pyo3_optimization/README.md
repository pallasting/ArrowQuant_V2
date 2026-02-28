# PyO3 Zero-Copy Optimization - Property-Based Tests

This directory contains property-based tests for the PyO3 zero-copy optimization feature using `proptest`.

## Properties Tested

### Property 1: Numpy Zero-Copy Access
**Validates: Requirements 1.1**
- For any valid numpy array, accessing through `quantize_numpy()` should not copy data
- Verifiable through memory address comparison

### Property 2: Batch Processing Single Invocation
**Validates: Requirements 2.1**
- For any dictionary of layers, `quantize_batch()` should cross Python-Rust boundary once
- Regardless of number of layers in batch

### Property 3: Batch Error Identification
**Validates: Requirements 2.3**
- For any batch with invalid layers, error messages should identify specific failed layers
- Include reason for each failure

### Property 4: Result Equivalence Across APIs
**Validates: Requirements 2.5, 5.2**
- For any weight tensor, all APIs should produce mathematically equivalent results
- Legacy, numpy, batch, and Arrow IPC APIs

### Property 5: Arrow Zero-Copy Access
**Validates: Requirements 3.1**
- For any valid PyArrow Table, accessing through `quantize_arrow()` should not copy buffers
- Verifiable through memory profiling

### Property 6: Backward Compatibility
**Validates: Requirements 5.1**
- For any existing API call, it should continue working without code changes

### Property 7: Validation Error Messages
**Validates: Requirements 6.1**
- For any invalid input, error message should clearly identify the validation failure

### Property 8-11: Memory Safety
**Validates: Requirements 9.1-9.4**
- Array lifetime validation
- Buffer reference counting
- Thread safety
- Resource cleanup

### Property 12-13: Performance Monitoring
**Validates: Requirements 10.1, 10.2, 10.4**
- Metrics reporting
- Performance degradation warnings

## Running Property Tests

Run all property tests:
```bash
cargo test --test test_zero_copy_property
cargo test --test test_equivalence_property
cargo test --test test_memory_safety_property
```

Run with more iterations:
```bash
PROPTEST_CASES=1000 cargo test --test test_zero_copy_property
```

## Test Files

- `test_zero_copy_property.rs` - Properties 1, 5: Zero-copy verification
- `test_equivalence_property.rs` - Property 4: Result equivalence
- `test_memory_safety_property.rs` - Properties 8-11: Memory safety
- `test_error_handling_property.rs` - Properties 3, 7: Error handling
- `test_performance_property.rs` - Properties 12-13: Performance monitoring
