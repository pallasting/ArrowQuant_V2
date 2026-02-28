//! Memory scheduler for controlling concurrent memory usage during quantization.
//!
//! Provides a token-based mechanism to ensure that multiple parallel quantization
//! tasks do not exceed the available physical memory.

use std::sync::{Arc, Condvar, Mutex};
use log::{debug, trace};

/// Controls memory allocation for parallel tasks using a semaphore-like approach
/// with support for variable-sized "tokens" representing bytes of RAM.
pub struct MemoryScheduler {
    /// Maximum allowed memory in bytes
    max_memory_bytes: usize,
    /// Currently allocated memory in bytes
    current_used_bytes: Mutex<usize>,
    /// Condvar for blocking threads until memory is available
    condvar: Condvar,
}

impl MemoryScheduler {
    /// Create a new MemoryScheduler with a specific limit in bytes
    pub fn new(max_memory_bytes: usize) -> Self {
        debug!("Initializing MemoryScheduler with {} MB limit", max_memory_bytes / (1024 * 1024));
        Self {
            max_memory_bytes,
            current_used_bytes: Mutex::new(0),
            condvar: Condvar::new(),
        }
    }

    /// Acquire memory for a task. Blocks until the requested amount is available.
    ///
    /// # Arguments
    ///
    /// * `required_bytes` - Amount of RAM requested for the layer
    pub fn acquire(&self, required_bytes: usize) {
        let mut used = self.current_used_bytes.lock().unwrap();
        
        // Safety check: if a single layer is larger than the total limit,
        // we allow it to proceed (otherwise we'd deadlock), but it will block 
        // everyone else.
        if required_bytes > self.max_memory_bytes {
            debug!(
                "Requested layer ({} MB) exceeds total scheduler limit ({} MB). Processing exclusively.",
                required_bytes / (1024 * 1024),
                self.max_memory_bytes / (1024 * 1024)
            );
        }

        while *used + required_bytes > self.max_memory_bytes && *used > 0 {
            trace!(
                "Memory limit reached (Used: {} MB, Requested: {} MB). waiting...",
                *used / (1024 * 1024),
                required_bytes / (1024 * 1024)
            );
            used = self.condvar.wait(used).unwrap();
        }
        
        *used += required_bytes;
        trace!(
            "Acquired {} MB. Current total usage: {} MB",
            required_bytes / (1024 * 1024),
            *used / (1024 * 1024)
        );
    }

    /// Release memory back to the pool and notify waiting threads.
    pub fn release(&self, released_bytes: usize) {
        let mut used = self.current_used_bytes.lock().unwrap();
        
        if released_bytes > *used {
            *used = 0;
            debug!("Warning: Released more memory than currently marked as used.");
        } else {
            *used -= released_bytes;
        }
        
        trace!(
            "Released {} MB. Current total usage: {} MB",
            released_bytes / (1024 * 1024),
            *used / (1024 * 1024)
        );
        
        self.condvar.notify_all();
    }

    /// Get current memory usage in bytes
    pub fn current_usage(&self) -> usize {
        *self.current_used_bytes.lock().unwrap()
    }

    /// Get max memory limit in bytes
    pub fn max_memory(&self) -> usize {
        self.max_memory_bytes
    }

    /// Acquire a memory token (RAII guard)
    pub fn acquire_token(self: &Arc<Self>, bytes: usize) -> MemoryToken {
        self.acquire(bytes);
        MemoryToken {
            scheduler: self.clone(),
            bytes,
        }
    }
}

/// RAII guard that releases memory when dropped
pub struct MemoryToken {
    scheduler: Arc<MemoryScheduler>,
    bytes: usize,
}

impl Drop for MemoryToken {
    fn drop(&mut self) {
        self.scheduler.release(self.bytes);
    }
}

/// Shared wrapper for MemoryScheduler
pub type SharedMemoryScheduler = Arc<MemoryScheduler>;
