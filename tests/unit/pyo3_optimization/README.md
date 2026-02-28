# PyO3 Zero-Copy Optimization - Unit Tests

This directory contains unit tests for the PyO3 zero-copy optimization feature.

## Test Coverage

- **Numpy Interface**: Tests for `quantize_numpy()` and `quantize_numpy_2d()` methods
- **Batch API**: Tests for `quantize_batch()` and `quantize_batch_with_progress()` methods
- **Arrow IPC**: Tests for `quantize_arrow()` and `quantize_arrow_batch()` methods
- **Error Handling**: Tests for validation and error messages
- **Edge Cases**: Tests for non-contiguous arrays, incorrect dtypes, NaN/Inf values

## Running Tests

Run all unit tests for PyO3 optimization:
```bash
cargo test --test test_numpy_interface
cargo test --test test_batch_api
cargo test --test test_arrow_ipc
```

Run specific test:
```bash
cargo test --test test_numpy_interface -- test_quantize_numpy_basic
```

## Test Files

- `test_numpy_interface.rs` - Numpy zero-copy interface tests
- `test_batch_api.rs` - Batch API tests
- `test_arrow_ipc.rs` - Arrow IPC interface tests
- `test_validation.rs` - Input validation tests
- `test_error_handling.rs` - Error handling tests
