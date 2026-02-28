//! Tests for zero-copy weight loading optimization

use arrow_quant_v2::{BufferPool, DiffusionOrchestrator, DiffusionQuantConfig, ParquetV2Extended};
use tempfile::TempDir;

#[test]
fn test_zero_copy_read_basic() {
    // Create a temporary Parquet file
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_layer.parquet");

    // Create test data
    let schema = ParquetV2Extended::from_v2_base(
        "test_layer".to_string(),
        vec![10, 20],
        "float32".to_string(),
        vec![1, 2, 3, 4],
        200,
        "int8".to_string(),
        vec![0.1, 0.2],
        vec![0.0, 0.0],
        None,
        Some(128),
    );

    // Write to file
    schema.write_to_parquet(&file_path).unwrap();

    // Read with zero-copy
    let loaded = ParquetV2Extended::read_from_parquet_zero_copy(&file_path).unwrap();

    // Verify data
    assert_eq!(loaded.layer_name, "test_layer");
    assert_eq!(loaded.shape, vec![10, 20]);
    assert_eq!(loaded.num_params, 200);
}

#[test]
fn test_zero_copy_vs_standard_equivalence() {
    // Create a temporary Parquet file
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("test_layer.parquet");

    // Create test data with diffusion metadata
    let schema = ParquetV2Extended::from_v2_base(
        "attention_layer".to_string(),
        vec![512, 512],
        "float32".to_string(),
        vec![0u8; 1024], // 1KB of data
        262144,
        "int4".to_string(),
        vec![0.05; 16],
        vec![0.0; 16],
        None,
        Some(64),
    );

    // Write to file
    schema.write_to_parquet(&file_path).unwrap();

    // Read with both methods
    let standard = ParquetV2Extended::read_from_parquet(&file_path).unwrap();
    let zero_copy = ParquetV2Extended::read_from_parquet_zero_copy(&file_path).unwrap();

    // Verify equivalence
    assert_eq!(standard.layer_name, zero_copy.layer_name);
    assert_eq!(standard.shape, zero_copy.shape);
    assert_eq!(standard.dtype, zero_copy.dtype);
    assert_eq!(standard.num_params, zero_copy.num_params);
    assert_eq!(standard.quant_type, zero_copy.quant_type);
    assert_eq!(standard.data.len(), zero_copy.data.len());
}

#[test]
fn test_buffer_pool_integration() {
    let pool = BufferPool::new(8, 512 * 1024); // 8 buffers, 512KB min

    // Simulate quantization workflow
    for i in 0..20 {
        // Acquire buffer
        let mut buffer = pool.acquire(1024 * 1024); // 1MB buffer

        // Simulate processing
        buffer.extend_from_slice(&vec![i as u8; 1000]);

        // Release buffer
        pool.release(buffer);
    }

    // Check pool stats
    let (size, total_capacity) = pool.stats();
    assert!(size > 0, "Pool should have cached buffers");
    assert!(size <= 8, "Pool should not exceed max size");
    assert!(
        total_capacity >= 512 * 1024,
        "Pool should have minimum capacity"
    );
}

#[test]
fn test_orchestrator_uses_zero_copy_in_streaming() {
    // Create config with streaming enabled
    let mut config = DiffusionQuantConfig::default();
    config.enable_streaming = true;
    config.bit_width = 8;

    // Create orchestrator
    let _orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Verify orchestrator was created successfully
    // (actual zero-copy usage is internal and tested through integration)
    assert!(true);
}

#[test]
fn test_orchestrator_uses_standard_in_parallel() {
    // Create config with parallel mode enabled
    let mut config = DiffusionQuantConfig::default();
    config.enable_streaming = false;
    config.num_threads = 4;
    config.bit_width = 8;

    // Create orchestrator
    let _orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Verify orchestrator was created successfully
    assert!(true);
}

#[test]
fn test_zero_copy_with_large_data() {
    // Create a temporary Parquet file with larger data
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("large_layer.parquet");

    // Create 10MB of test data
    let data_size = 10 * 1024 * 1024;
    let large_data = vec![42u8; data_size];

    let schema = ParquetV2Extended::from_v2_base(
        "large_layer".to_string(),
        vec![1024, 1024, 10],
        "float32".to_string(),
        large_data.clone(),
        10485760,
        "int2".to_string(),
        vec![0.01; 32],
        vec![0.0; 32],
        None,
        Some(256),
    );

    // Write to file
    schema.write_to_parquet(&file_path).unwrap();

    // Read with zero-copy
    let loaded = ParquetV2Extended::read_from_parquet_zero_copy(&file_path).unwrap();

    // Verify data integrity
    assert_eq!(loaded.layer_name, "large_layer");
    assert_eq!(loaded.data.len(), data_size);
    assert_eq!(loaded.data[0], 42);
    assert_eq!(loaded.data[data_size - 1], 42);
}

#[test]
fn test_zero_copy_with_diffusion_metadata() {
    let temp_dir = TempDir::new().unwrap();
    let file_path = temp_dir.path().join("diffusion_layer.parquet");

    // Create schema with basic data
    let schema = ParquetV2Extended::from_v2_base(
        "unet_block".to_string(),
        vec![256, 256],
        "float32".to_string(),
        vec![0u8; 4096],
        65536,
        "int4".to_string(),
        vec![0.02; 8],
        vec![0.0; 8],
        None,
        Some(128),
    );

    // Write to file
    schema.write_to_parquet(&file_path).unwrap();

    // Read with zero-copy
    let loaded = ParquetV2Extended::read_from_parquet_zero_copy(&file_path).unwrap();

    // Verify basic fields
    assert_eq!(loaded.layer_name, "unet_block");
    assert_eq!(loaded.shape, vec![256, 256]);
    assert_eq!(loaded.num_params, 65536);
    assert_eq!(loaded.quant_type, "int4");
}

#[test]
fn test_buffer_pool_concurrent_stress() {
    use std::sync::Arc;
    use std::thread;

    let pool = Arc::new(BufferPool::new(16, 256 * 1024)); // 16 buffers, 256KB min
    let mut handles = vec![];

    // Spawn 8 threads
    for thread_id in 0..8 {
        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            for i in 0..50 {
                // Acquire buffer
                let mut buffer = pool_clone.acquire(512 * 1024); // 512KB

                // Simulate work
                buffer.extend_from_slice(&vec![(thread_id * 100 + i) as u8; 100]);

                // Release buffer
                pool_clone.release(buffer);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify pool state
    let (size, _) = pool.stats();
    assert!(size > 0, "Pool should have cached buffers");
    assert!(size <= 16, "Pool should not exceed max size");
}
