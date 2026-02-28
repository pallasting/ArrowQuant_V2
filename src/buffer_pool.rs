//! Buffer pool for memory optimization
//!
//! This module provides a buffer pool for reusing allocated buffers across
//! quantization operations, reducing allocation overhead and memory fragmentation.

use std::sync::{Arc, Mutex};

/// Metrics for buffer pool performance
#[derive(Debug, Clone, Default)]
pub struct BufferPoolMetrics {
    /// Total number of acquire() calls
    pub total_acquires: usize,
    /// Number of times a buffer was reused from pool (cache hits)
    pub pool_hits: usize,
    /// Number of times a new buffer was allocated (cache misses)
    pub pool_misses: usize,
    /// Total number of release() calls
    pub total_releases: usize,
    /// Total bytes allocated (new allocations only)
    pub total_bytes_allocated: usize,
    /// Total bytes saved by reusing buffers
    pub total_bytes_saved: usize,
}

impl BufferPoolMetrics {
    /// Calculate hit rate (percentage of acquires that reused a buffer)
    pub fn hit_rate(&self) -> f64 {
        if self.total_acquires == 0 {
            0.0
        } else {
            (self.pool_hits as f64 / self.total_acquires as f64) * 100.0
        }
    }

    /// Calculate memory savings in MB
    pub fn memory_savings_mb(&self) -> f64 {
        self.total_bytes_saved as f64 / (1024.0 * 1024.0)
    }

    /// Calculate allocation overhead reduction percentage
    pub fn allocation_reduction(&self) -> f64 {
        let total_potential = self.total_acquires;
        if total_potential == 0 {
            0.0
        } else {
            (self.pool_hits as f64 / total_potential as f64) * 100.0
        }
    }
}

/// A simple buffer pool for reusing Vec<u8> allocations
#[derive(Clone)]
pub struct BufferPool {
    /// Pool of available buffers
    buffers: Arc<Mutex<Vec<Vec<u8>>>>,
    /// Maximum number of buffers to keep in pool
    max_pool_size: usize,
    /// Minimum buffer capacity to keep in pool
    min_capacity: usize,
    /// Performance metrics
    metrics: Arc<Mutex<BufferPoolMetrics>>,
}

impl BufferPool {
    /// Create a new buffer pool
    ///
    /// # Arguments
    /// * `max_pool_size` - Maximum number of buffers to keep in pool
    /// * `min_capacity` - Minimum buffer capacity to keep in pool (bytes)
    pub fn new(max_pool_size: usize, min_capacity: usize) -> Self {
        Self {
            buffers: Arc::new(Mutex::new(Vec::with_capacity(max_pool_size))),
            max_pool_size,
            min_capacity,
            metrics: Arc::new(Mutex::new(BufferPoolMetrics::default())),
        }
    }

    /// Create a default buffer pool
    ///
    /// Default settings:
    /// - max_pool_size: 16 buffers
    /// - min_capacity: 1MB
    pub fn default() -> Self {
        Self::new(16, 1024 * 1024) // 16 buffers, 1MB minimum
    }

    /// Acquire a buffer from the pool
    ///
    /// If the pool is empty, allocates a new buffer with the requested capacity.
    /// If a buffer is available, reuses it and clears its contents.
    ///
    /// # Arguments
    /// * `capacity` - Minimum capacity needed for the buffer
    ///
    /// # Returns
    /// A Vec<u8> with at least the requested capacity
    pub fn acquire(&self, capacity: usize) -> Vec<u8> {
        let mut pool = self.buffers.lock().unwrap();
        let mut metrics = self.metrics.lock().unwrap();

        metrics.total_acquires += 1;

        // Try to find a buffer with sufficient capacity
        // Prefer exact match or smallest buffer that fits
        let mut best_match: Option<(usize, usize)> = None; // (index, capacity)

        for (idx, buf) in pool.iter().enumerate() {
            let buf_capacity = buf.capacity();
            if buf_capacity >= capacity {
                match best_match {
                    None => best_match = Some((idx, buf_capacity)),
                    Some((_, best_cap)) if buf_capacity < best_cap => {
                        best_match = Some((idx, buf_capacity));
                    }
                    _ => {}
                }
            }
        }

        if let Some((idx, buf_capacity)) = best_match {
            let mut buffer = pool.swap_remove(idx);
            buffer.clear();

            // Record hit and memory savings
            metrics.pool_hits += 1;
            metrics.total_bytes_saved += buf_capacity;

            return buffer;
        }

        // No suitable buffer found, allocate new one
        metrics.pool_misses += 1;
        metrics.total_bytes_allocated += capacity;

        Vec::with_capacity(capacity)
    }

    /// Release a buffer back to the pool
    ///
    /// The buffer is only kept if:
    /// - The pool is not full
    /// - The buffer capacity is >= min_capacity
    ///
    /// # Arguments
    /// * `buffer` - Buffer to release back to pool
    pub fn release(&self, mut buffer: Vec<u8>) {
        let mut metrics = self.metrics.lock().unwrap();
        metrics.total_releases += 1;

        // Only keep buffers with sufficient capacity
        if buffer.capacity() < self.min_capacity {
            return;
        }

        let mut pool = self.buffers.lock().unwrap();

        // Only add if pool is not full
        if pool.len() < self.max_pool_size {
            buffer.clear();
            // Don't shrink - keep the original capacity for better reuse
            pool.push(buffer);
        }
    }

    /// Get current pool statistics
    ///
    /// # Returns
    /// (pool_size, total_capacity_bytes)
    pub fn stats(&self) -> (usize, usize) {
        let pool = self.buffers.lock().unwrap();
        let total_capacity: usize = pool.iter().map(|buf| buf.capacity()).sum();
        (pool.len(), total_capacity)
    }

    /// Get performance metrics
    ///
    /// # Returns
    /// BufferPoolMetrics with hit rate, memory savings, etc.
    pub fn metrics(&self) -> BufferPoolMetrics {
        let metrics = self.metrics.lock().unwrap();
        metrics.clone()
    }

    /// Reset metrics (useful for benchmarking)
    pub fn reset_metrics(&self) {
        let mut metrics = self.metrics.lock().unwrap();
        *metrics = BufferPoolMetrics::default();
    }

    /// Clear all buffers from the pool
    pub fn clear(&self) {
        let mut pool = self.buffers.lock().unwrap();
        pool.clear();
    }
}

/// RAII guard for automatic buffer release
///
/// When dropped, automatically releases the buffer back to the pool
pub struct PooledBuffer {
    buffer: Option<Vec<u8>>,
    pool: BufferPool,
}

impl PooledBuffer {
    /// Create a new pooled buffer
    pub fn new(pool: BufferPool, capacity: usize) -> Self {
        let buffer = pool.acquire(capacity);
        Self {
            buffer: Some(buffer),
            pool,
        }
    }

    /// Get mutable reference to the buffer
    pub fn as_mut(&mut self) -> &mut Vec<u8> {
        self.buffer.as_mut().unwrap()
    }

    /// Get immutable reference to the buffer
    pub fn as_ref(&self) -> &Vec<u8> {
        self.buffer.as_ref().unwrap()
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_buffer_pool_acquire_release() {
        let pool = BufferPool::new(4, 1024);

        // Acquire buffer
        let buf1 = pool.acquire(2048);
        assert!(buf1.capacity() >= 2048);

        // Release buffer
        pool.release(buf1);

        // Pool should have 1 buffer
        let (size, _) = pool.stats();
        assert_eq!(size, 1);
    }

    #[test]
    fn test_buffer_pool_reuse() {
        let pool = BufferPool::new(4, 1024);

        // Acquire and release buffer
        let buf1 = pool.acquire(2048);
        let capacity1 = buf1.capacity();
        pool.release(buf1);

        // Acquire again - should reuse same buffer
        let buf2 = pool.acquire(2048);
        assert_eq!(buf2.capacity(), capacity1);

        pool.release(buf2);
    }

    #[test]
    fn test_buffer_pool_max_size() {
        let pool = BufferPool::new(2, 1024);

        // Release 3 buffers
        pool.release(vec![0u8; 2048]);
        pool.release(vec![0u8; 2048]);
        pool.release(vec![0u8; 2048]);

        // Pool should only keep 2 buffers
        let (size, _) = pool.stats();
        assert_eq!(size, 2);
    }

    #[test]
    fn test_buffer_pool_min_capacity() {
        let pool = BufferPool::new(4, 2048);

        // Release small buffer (below min_capacity)
        pool.release(vec![0u8; 1024]);

        // Pool should not keep it
        let (size, _) = pool.stats();
        assert_eq!(size, 0);

        // Release large buffer (above min_capacity)
        pool.release(vec![0u8; 4096]);

        // Pool should keep it
        let (size, _) = pool.stats();
        assert_eq!(size, 1);
    }

    #[test]
    fn test_buffer_pool_clear() {
        let pool = BufferPool::new(4, 1024);

        // Add buffers
        pool.release(vec![0u8; 2048]);
        pool.release(vec![0u8; 2048]);

        let (size, _) = pool.stats();
        assert_eq!(size, 2);

        // Clear pool
        pool.clear();

        let (size, _) = pool.stats();
        assert_eq!(size, 0);
    }

    #[test]
    fn test_pooled_buffer_raii() {
        let pool = BufferPool::new(4, 1024);

        {
            let mut pooled = PooledBuffer::new(pool.clone(), 2048);
            pooled.as_mut().extend_from_slice(&[1, 2, 3, 4]);
            assert_eq!(pooled.as_ref().len(), 4);

            // Buffer will be automatically released when dropped
        }

        // Pool should have 1 buffer
        let (size, _) = pool.stats();
        assert_eq!(size, 1);
    }

    #[test]
    fn test_buffer_pool_stats() {
        let pool = BufferPool::new(4, 1024);

        // Add buffers of different sizes
        pool.release(Vec::with_capacity(2048));
        pool.release(Vec::with_capacity(4096));

        // Check pool stats
        let (pool_size, total_capacity) = pool.stats();
        assert_eq!(pool_size, 2);
        // Buffers keep their original capacity (no shrinking)
        assert!(total_capacity >= 2048 + 4096);
    }

    #[test]
    fn test_buffer_pool_concurrent_access() {
        use std::thread;

        let pool = BufferPool::new(8, 1024);
        let pool_clone = pool.clone();

        // Spawn thread to acquire/release buffers
        let handle = thread::spawn(move || {
            for _ in 0..10 {
                let buf = pool_clone.acquire(2048);
                pool_clone.release(buf);
            }
        });

        // Main thread also acquires/releases
        for _ in 0..10 {
            let buf = pool.acquire(2048);
            pool.release(buf);
        }

        handle.join().unwrap();

        // Pool should have some buffers
        let (size, _) = pool.stats();
        assert!(size > 0);
    }

    #[test]
    fn test_buffer_pool_metrics_hit_rate() {
        let pool = BufferPool::new(4, 1024);

        // First acquire - miss
        let buf1 = pool.acquire(2048);
        pool.release(buf1);

        // Second acquire - hit
        let buf2 = pool.acquire(2048);
        pool.release(buf2);

        // Third acquire - hit
        let buf3 = pool.acquire(2048);
        pool.release(buf3);

        let metrics = pool.metrics();
        assert_eq!(metrics.total_acquires, 3);
        assert_eq!(metrics.pool_hits, 2);
        assert_eq!(metrics.pool_misses, 1);
        assert_eq!(metrics.total_releases, 3);

        // Hit rate should be 66.67% (2 hits out of 3 acquires)
        let hit_rate = metrics.hit_rate();
        assert!((hit_rate - 66.67).abs() < 0.1);
    }

    #[test]
    fn test_buffer_pool_metrics_memory_savings() {
        let pool = BufferPool::new(4, 1024);

        // First acquire - allocates 2048 bytes
        let buf1 = pool.acquire(2048);
        let capacity = buf1.capacity();
        pool.release(buf1);

        // Second acquire - reuses buffer, saves capacity bytes
        let buf2 = pool.acquire(2048);
        pool.release(buf2);

        let metrics = pool.metrics();
        assert_eq!(metrics.total_bytes_allocated, 2048);
        assert_eq!(metrics.total_bytes_saved, capacity);

        // Memory savings should be > 0
        assert!(metrics.memory_savings_mb() > 0.0);
    }

    #[test]
    fn test_buffer_pool_metrics_allocation_reduction() {
        let pool = BufferPool::new(4, 1024);

        // Acquire and release 10 times
        for _ in 0..10 {
            let buf = pool.acquire(2048);
            pool.release(buf);
        }

        let metrics = pool.metrics();

        // First acquire is a miss, rest are hits
        assert_eq!(metrics.pool_misses, 1);
        assert_eq!(metrics.pool_hits, 9);

        // Allocation reduction should be 90%
        let reduction = metrics.allocation_reduction();
        assert!((reduction - 90.0).abs() < 0.1);
    }

    #[test]
    fn test_buffer_pool_reset_metrics() {
        let pool = BufferPool::new(4, 1024);

        // Perform some operations
        let buf = pool.acquire(2048);
        pool.release(buf);

        // Metrics should be non-zero
        let metrics1 = pool.metrics();
        assert!(metrics1.total_acquires > 0);

        // Reset metrics
        pool.reset_metrics();

        // Metrics should be zero
        let metrics2 = pool.metrics();
        assert_eq!(metrics2.total_acquires, 0);
        assert_eq!(metrics2.pool_hits, 0);
        assert_eq!(metrics2.pool_misses, 0);
    }

    #[test]
    fn test_buffer_pool_metrics_multiple_sizes() {
        let pool = BufferPool::new(8, 1024);

        // Acquire buffers of different sizes
        let buf1 = pool.acquire(2048);
        let buf2 = pool.acquire(4096);
        let buf3 = pool.acquire(8192);

        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3);

        // Reacquire - should hit for matching sizes
        let buf4 = pool.acquire(2048); // Hit
        let buf5 = pool.acquire(4096); // Hit
        let buf6 = pool.acquire(16384); // Miss (no buffer large enough)

        pool.release(buf4);
        pool.release(buf5);
        pool.release(buf6);

        let metrics = pool.metrics();
        assert_eq!(metrics.total_acquires, 6);
        assert_eq!(metrics.pool_hits, 2);
        assert_eq!(metrics.pool_misses, 4); // 3 initial + 1 for 16384
    }
}
