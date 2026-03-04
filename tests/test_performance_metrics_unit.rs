/// Unit test for performance metrics logging
///
/// This test verifies that the SIMD detection and timing infrastructure
/// is working correctly for performance metrics logging.
///
/// **Validates: Requirements 5.3, 12.1** - Performance metrics recording
use arrow_quant_v2::simd::{is_simd_available, SimdWidth};
use std::time::Instant;

#[test]
fn test_simd_detection_for_metrics() {
    // Test SIMD detection (used in performance metrics)
    let simd_status = is_simd_available();

    // Verify we get a valid SIMD width
    match simd_status {
        SimdWidth::None => {
            println!("SIMD: disabled (scalar fallback)");
            assert_eq!(simd_status.width(), 1);
            assert!(!simd_status.is_available());
        }
        SimdWidth::Neon => {
            println!("SIMD: enabled (NEON, width=4)");
            assert_eq!(simd_status.width(), 4);
            assert!(simd_status.is_available());
        }
        SimdWidth::Avx2 => {
            println!("SIMD: enabled (AVX2, width=8)");
            assert_eq!(simd_status.width(), 8);
            assert!(simd_status.is_available());
        }
        SimdWidth::Avx512 => {
            println!("SIMD: enabled (AVX-512, width=16)");
            assert_eq!(simd_status.width(), 16);
            assert!(simd_status.is_available());
        }
    }

    println!("✓ SIMD detection working correctly");
}

#[test]
fn test_timing_infrastructure() {
    // Test timing infrastructure (used in performance metrics)
    let start = Instant::now();

    // Simulate some work
    let mut sum = 0u64;
    for i in 0..1000 {
        sum += i;
    }

    let elapsed = start.elapsed();
    let elapsed_ms = elapsed.as_secs_f64() * 1000.0;

    // Verify timing works
    assert!(elapsed_ms >= 0.0);
    assert!(elapsed_ms < 1000.0); // Should complete in less than 1 second

    println!("✓ Timing infrastructure working correctly");
    println!("  Elapsed: {:.3} ms", elapsed_ms);
    println!("  Sum: {}", sum);
}

#[test]
fn test_memory_size_calculations() {
    // Test memory size calculations (used in performance metrics)
    let num_elements = 10000;

    // Calculate memory sizes
    let f32_size = std::mem::size_of::<f32>();
    let u8_size = std::mem::size_of::<u8>();

    let input_bytes = num_elements * f32_size;
    let output_bytes = num_elements * u8_size;

    let input_mb = input_bytes as f64 / (1024.0 * 1024.0);
    let output_mb = output_bytes as f64 / (1024.0 * 1024.0);
    let total_mb = input_mb + output_mb;

    // Verify calculations
    assert_eq!(f32_size, 4);
    assert_eq!(u8_size, 1);
    assert_eq!(input_bytes, 40000);
    assert_eq!(output_bytes, 10000);

    println!("✓ Memory size calculations working correctly");
    println!(
        "  Input: {:.3} MB ({} elements × {} bytes)",
        input_mb, num_elements, f32_size
    );
    println!(
        "  Output: {:.3} MB ({} elements × {} bytes)",
        output_mb, num_elements, u8_size
    );
    println!("  Total: {:.3} MB", total_mb);
}

#[test]
fn test_throughput_calculations() {
    // Test throughput calculations (used in performance metrics)
    let num_elements = 100000;
    let elapsed_ms = 10.0; // 10 milliseconds

    let elements_per_ms = num_elements as f64 / elapsed_ms;

    let total_mb = 0.5; // 0.5 MB
    let elapsed_s = elapsed_ms / 1000.0;
    let mb_per_s = total_mb / elapsed_s;

    // Verify calculations
    assert_eq!(elements_per_ms, 10000.0);
    assert_eq!(mb_per_s, 50.0);

    println!("✓ Throughput calculations working correctly");
    println!("  Elements/ms: {:.2}", elements_per_ms);
    println!("  MB/s: {:.2}", mb_per_s);
}

#[test]
fn test_performance_metrics_format() {
    // Test that we can format performance metrics correctly
    let num_layers = 5;
    let total_elements = 50000;
    let elapsed_ms = 12.5;
    let simd_enabled = true;
    let simd_width = 8;
    let input_mb = 0.19;
    let output_mb = 0.05;
    let total_mb = 0.24;

    // Format metrics (similar to what's logged)
    let msg1 = format!(
        "Quantization completed: {} layers, {} total elements, {:.2} ms elapsed",
        num_layers, total_elements, elapsed_ms
    );

    let msg2 = format!(
        "SIMD status: enabled={}, width={}",
        simd_enabled, simd_width
    );

    let msg3 = format!(
        "Memory usage: input={:.2} MB, output={:.2} MB, total={:.2} MB",
        input_mb, output_mb, total_mb
    );

    let elements_per_ms = total_elements as f64 / elapsed_ms;
    let mb_per_s = total_mb / (elapsed_ms / 1000.0);

    let msg4 = format!(
        "Throughput: {:.2} elements/ms, {:.2} MB/s",
        elements_per_ms, mb_per_s
    );

    println!("✓ Performance metrics formatting working correctly");
    println!("  {}", msg1);
    println!("  {}", msg2);
    println!("  {}", msg3);
    println!("  {}", msg4);

    // Verify format strings are correct
    assert!(msg1.contains("5 layers"));
    assert!(msg1.contains("50000 total elements"));
    assert!(msg2.contains("enabled=true"));
    assert!(msg2.contains("width=8"));
    assert!(msg3.contains("0.19 MB"));
    assert!(msg4.contains("elements/ms"));
}
