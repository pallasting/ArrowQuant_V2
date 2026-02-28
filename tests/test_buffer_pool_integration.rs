//! Integration tests for buffer pool with orchestrator
//!
//! Tests buffer pool performance metrics and integration with quantization operations

use arrow_quant_v2::{BufferPool, DiffusionOrchestrator, DiffusionQuantConfig};

#[test]
fn test_buffer_pool_basic_metrics() {
    let pool = BufferPool::new(8, 1024);

    // Acquire and release buffers
    let buf1 = pool.acquire(2048);
    pool.release(buf1);

    let buf2 = pool.acquire(2048);
    pool.release(buf2);

    let metrics = pool.metrics();
    assert_eq!(metrics.total_acquires, 2);
    assert_eq!(metrics.pool_hits, 1); // Second acquire reused buffer
    assert_eq!(metrics.pool_misses, 1); // First acquire allocated new
    assert_eq!(metrics.total_releases, 2);

    // Hit rate should be 50%
    let hit_rate = metrics.hit_rate();
    assert!((hit_rate - 50.0).abs() < 0.1);
}

#[test]
fn test_buffer_pool_memory_savings() {
    let pool = BufferPool::new(8, 1024);

    // Acquire large buffer
    let buf1 = pool.acquire(1024 * 1024); // 1MB
    let capacity = buf1.capacity();
    pool.release(buf1);

    // Reacquire - should reuse
    let buf2 = pool.acquire(1024 * 1024);
    pool.release(buf2);

    let metrics = pool.metrics();

    // Should have saved approximately 1MB
    assert!(metrics.total_bytes_saved >= capacity);
    assert!(metrics.memory_savings_mb() >= 0.9); // At least 0.9 MB saved
}

#[test]
fn test_buffer_pool_high_hit_rate() {
    let pool = BufferPool::new(8, 1024);

    // Simulate many operations with same buffer size
    for _ in 0..100 {
        let buf = pool.acquire(4096);
        pool.release(buf);
    }

    let metrics = pool.metrics();

    // First is miss, rest are hits
    assert_eq!(metrics.pool_misses, 1);
    assert_eq!(metrics.pool_hits, 99);

    // Hit rate should be 99%
    let hit_rate = metrics.hit_rate();
    assert!((hit_rate - 99.0).abs() < 0.1);

    // Allocation reduction should be 99%
    let reduction = metrics.allocation_reduction();
    assert!((reduction - 99.0).abs() < 0.1);
}

#[test]
fn test_buffer_pool_multiple_sizes() {
    let pool = BufferPool::new(16, 1024);

    // Acquire buffers of different sizes
    let sizes = vec![2048, 4096, 8192, 16384];

    for &size in &sizes {
        let buf = pool.acquire(size);
        pool.release(buf);
    }

    // Reacquire same sizes - should hit
    for &size in &sizes {
        let buf = pool.acquire(size);
        pool.release(buf);
    }

    let metrics = pool.metrics();

    // 4 initial misses + 4 hits
    assert_eq!(metrics.total_acquires, 8);
    assert_eq!(metrics.pool_misses, 4);
    assert_eq!(metrics.pool_hits, 4);

    // Hit rate should be 50%
    let hit_rate = metrics.hit_rate();
    assert!((hit_rate - 50.0).abs() < 0.1);
}

#[test]
fn test_buffer_pool_with_orchestrator() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    // Get initial metrics
    let initial_metrics = orchestrator.get_buffer_pool_metrics();
    assert_eq!(initial_metrics.total_acquires, 0);

    // Use buffer pool directly
    // Note: Orchestrator's buffer pool has min_capacity = 1MB
    let pool = orchestrator.buffer_pool();

    for _ in 0..10 {
        let buf = pool.acquire(1024 * 1024); // 1MB - meets min_capacity
        pool.release(buf);
    }

    // Check metrics through orchestrator
    let metrics = orchestrator.get_buffer_pool_metrics();
    assert_eq!(metrics.total_acquires, 10);
    assert_eq!(metrics.pool_misses, 1);
    assert_eq!(metrics.pool_hits, 9);

    // Verify hit rate
    assert!((metrics.hit_rate() - 90.0).abs() < 1.0);
}

#[test]
fn test_buffer_pool_reset_metrics() {
    let config = DiffusionQuantConfig::default();
    let orchestrator = DiffusionOrchestrator::new(config).unwrap();

    let pool = orchestrator.buffer_pool();

    // Perform operations with buffer size >= min_capacity (1MB)
    for _ in 0..5 {
        let buf = pool.acquire(1024 * 1024); // 1MB
        pool.release(buf);
    }

    // Verify metrics are non-zero
    let metrics1 = orchestrator.get_buffer_pool_metrics();
    assert!(metrics1.total_acquires > 0);

    // Reset metrics
    orchestrator.reset_buffer_pool_metrics();

    // Verify metrics are zero
    let metrics2 = orchestrator.get_buffer_pool_metrics();
    assert_eq!(metrics2.total_acquires, 0);
    assert_eq!(metrics2.pool_hits, 0);
    assert_eq!(metrics2.pool_misses, 0);
}

#[test]
fn test_buffer_pool_allocation_overhead_reduction() {
    let pool = BufferPool::new(8, 1024);

    // Simulate quantization workload
    let buffer_sizes = vec![2048, 4096, 8192, 4096, 2048, 8192];

    // First pass - all misses
    for &size in &buffer_sizes {
        let buf = pool.acquire(size);
        pool.release(buf);
    }

    // Second pass - should have high hit rate
    for &size in &buffer_sizes {
        let buf = pool.acquire(size);
        pool.release(buf);
    }

    let metrics = pool.metrics();

    // Total acquires: 12
    assert_eq!(metrics.total_acquires, 12);

    // First pass: 3 unique sizes = 3 misses
    // Second pass: all hits = 6 hits
    // Total: 3 misses + 9 hits (some from first pass reuse)
    assert!(metrics.pool_hits >= 6);
    assert!(metrics.pool_misses <= 6);

    // Hit rate should be > 50%
    assert!(metrics.hit_rate() > 50.0);

    // Allocation reduction should be significant
    assert!(metrics.allocation_reduction() > 50.0);
}

#[test]
fn test_buffer_pool_concurrent_metrics() {
    use std::sync::Arc;
    use std::thread;

    let pool = Arc::new(BufferPool::new(16, 1024));
    let mut handles = vec![];

    // Spawn multiple threads
    for _ in 0..4 {
        let pool_clone = Arc::clone(&pool);
        let handle = thread::spawn(move || {
            for _ in 0..25 {
                let buf = pool_clone.acquire(4096);
                pool_clone.release(buf);
            }
        });
        handles.push(handle);
    }

    // Wait for all threads
    for handle in handles {
        handle.join().unwrap();
    }

    // Check metrics
    let metrics = pool.metrics();

    // Total acquires: 4 threads * 25 = 100
    assert_eq!(metrics.total_acquires, 100);

    // Should have high hit rate due to buffer reuse
    assert!(metrics.pool_hits > 80);
    assert!(metrics.hit_rate() > 80.0);
}

#[test]
fn test_buffer_pool_memory_savings_calculation() {
    let pool = BufferPool::new(8, 1024);

    // Acquire 1MB buffer
    let buf1 = pool.acquire(1024 * 1024);
    let capacity1 = buf1.capacity();
    pool.release(buf1);

    // Reacquire 10 times
    for _ in 0..10 {
        let buf = pool.acquire(1024 * 1024);
        pool.release(buf);
    }

    let metrics = pool.metrics();

    // Should have saved ~10MB (10 reuses of 1MB buffer)
    let expected_savings = capacity1 * 10;
    assert!(metrics.total_bytes_saved >= expected_savings);

    // Memory savings in MB
    let savings_mb = metrics.memory_savings_mb();
    assert!(savings_mb >= 9.0); // At least 9 MB saved
}

#[test]
fn test_buffer_pool_stats_consistency() {
    let pool = BufferPool::new(8, 1024);

    // Acquire and release multiple buffers
    let buf1 = pool.acquire(2048);
    let buf2 = pool.acquire(4096);
    let buf3 = pool.acquire(8192);

    pool.release(buf1);
    pool.release(buf2);
    pool.release(buf3);

    // Check pool stats
    let (pool_size, total_capacity) = pool.stats();
    assert_eq!(pool_size, 3);
    assert!(total_capacity > 0);

    // Check metrics
    let metrics = pool.metrics();
    assert_eq!(metrics.total_acquires, 3);
    assert_eq!(metrics.total_releases, 3);
    assert_eq!(metrics.pool_misses, 3); // All initial allocations
}

#[test]
fn test_buffer_pool_target_reduction() {
    let pool = BufferPool::new(16, 1024);

    // Simulate realistic quantization workload
    // Target: 20-40% allocation overhead reduction

    // Perform 100 operations with varying buffer sizes
    let sizes = vec![2048, 4096, 8192, 16384];

    for _ in 0..25 {
        for &size in &sizes {
            let buf = pool.acquire(size);
            pool.release(buf);
        }
    }

    let metrics = pool.metrics();

    // Total operations: 100
    assert_eq!(metrics.total_acquires, 100);

    // First 4 are misses (one per unique size)
    // Remaining 96 should be hits
    assert_eq!(metrics.pool_misses, 4);
    assert_eq!(metrics.pool_hits, 96);

    // Allocation reduction: 96%
    let reduction = metrics.allocation_reduction();
    assert!(reduction >= 90.0);

    // This exceeds the 20-40% target significantly
    println!("Allocation reduction: {:.2}%", reduction);
    println!("Memory savings: {:.2} MB", metrics.memory_savings_mb());
}
