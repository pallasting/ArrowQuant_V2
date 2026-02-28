//! SIMD-accelerated quantization operations
//!
//! This module provides SIMD-optimized implementations of quantization operations
//! using AVX2 (x86_64) and NEON (ARM64) intrinsics for significant performance improvements.

/// Quantize f32 values to u8 using SIMD acceleration
///
/// This function automatically selects the best available SIMD implementation:
/// - AVX2 on x86_64 with AVX2 support
/// - NEON on ARM64 with NEON support
/// - Scalar fallback for other platforms
///
/// # Arguments
///
/// * `data` - Input f32 values to quantize
/// * `scale` - Quantization scale factor
/// * `zero_point` - Quantization zero point offset
///
/// # Returns
///
/// Vector of quantized u8 values
pub fn quantize_simd(data: &[f32], scale: f32, zero_point: f32) -> Vec<u8> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { quantize_avx2(data, scale, zero_point) }
    }

    #[cfg(all(
        not(all(target_arch = "x86_64", target_feature = "avx2")),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    {
        unsafe { quantize_neon(data, scale, zero_point) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        quantize_scalar(data, scale, zero_point)
    }
}

/// Dequantize u8 values to f32 using SIMD acceleration
///
/// # Arguments
///
/// * `data` - Input u8 values to dequantize
/// * `scale` - Quantization scale factor
/// * `zero_point` - Quantization zero point offset
///
/// # Returns
///
/// Vector of dequantized f32 values
pub fn dequantize_simd(data: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { dequantize_avx2(data, scale, zero_point) }
    }

    #[cfg(all(
        not(all(target_arch = "x86_64", target_feature = "avx2")),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    {
        unsafe { dequantize_neon(data, scale, zero_point) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        dequantize_scalar(data, scale, zero_point)
    }
}

// ============================================================================
// AVX2 Implementation (x86_64)
// ============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn quantize_avx2(data: &[f32], scale: f32, zero_point: f32) -> Vec<u8> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut result = Vec::with_capacity(data.len());
    let scale_vec = _mm256_set1_ps(1.0 / scale);
    let zp_vec = _mm256_set1_ps(zero_point);
    let zero = _mm256_setzero_ps();
    let max_val = _mm256_set1_ps(255.0);

    // Process 8 floats at a time
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 floats
        let values = _mm256_loadu_ps(data.as_ptr().add(offset));

        // Quantize: q = round(x / scale + zero_point)
        let scaled = _mm256_mul_ps(values, scale_vec);
        let shifted = _mm256_add_ps(scaled, zp_vec);

        // Round to nearest integer
        let rounded = _mm256_round_ps::<{ _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC }>(shifted);

        // Clamp to [0, 255]
        let clamped = _mm256_max_ps(_mm256_min_ps(rounded, max_val), zero);

        // Convert to i32
        let quantized_i32 = _mm256_cvtps_epi32(clamped);

        // Extract and convert to u8
        let mut temp = [0i32; 8];
        _mm256_storeu_si256(temp.as_mut_ptr() as *mut __m256i, quantized_i32);

        for &val in &temp {
            result.push(val as u8);
        }
    }

    // Handle remainder with scalar fallback
    for &value in &data[chunks * 8..] {
        let q = ((value / scale) + zero_point).round().clamp(0.0, 255.0) as u8;
        result.push(q);
    }

    result
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dequantize_avx2(data: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut result = Vec::with_capacity(data.len());
    let scale_vec = _mm256_set1_ps(scale);
    let zp_vec = _mm256_set1_ps(zero_point);

    // Process 8 u8 values at a time
    let chunks = data.len() / 8;
    for i in 0..chunks {
        let offset = i * 8;

        // Load 8 u8 values and convert to i32
        let mut temp = [0i32; 8];
        for j in 0..8 {
            temp[j] = data[offset + j] as i32;
        }
        let values_i32 = _mm256_loadu_si256(temp.as_ptr() as *const __m256i);

        // Convert i32 to f32
        let values_f32 = _mm256_cvtepi32_ps(values_i32);

        // Dequantize: x = (q - zero_point) * scale
        let shifted = _mm256_sub_ps(values_f32, zp_vec);
        let dequantized = _mm256_mul_ps(shifted, scale_vec);

        // Store results
        let mut temp_f32 = [0.0f32; 8];
        _mm256_storeu_ps(temp_f32.as_mut_ptr(), dequantized);
        result.extend_from_slice(&temp_f32);
    }

    // Handle remainder with scalar fallback
    for &value in &data[chunks * 8..] {
        let x = (value as f32 - zero_point) * scale;
        result.push(x);
    }

    result
}

// ============================================================================
// NEON Implementation (ARM64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn quantize_neon(data: &[f32], scale: f32, zero_point: f32) -> Vec<u8> {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    let mut result = Vec::with_capacity(data.len());
    let scale_vec = vdupq_n_f32(1.0 / scale);
    let zp_vec = vdupq_n_f32(zero_point);
    let zero = vdupq_n_f32(0.0);
    let max_val = vdupq_n_f32(255.0);

    // Process 4 floats at a time (NEON processes 4 f32 per register)
    let chunks = data.len() / 4;
    for i in 0..chunks {
        let offset = i * 4;

        // Load 4 floats
        let values = vld1q_f32(data.as_ptr().add(offset));

        // Quantize: q = round(x / scale + zero_point)
        let scaled = vmulq_f32(values, scale_vec);
        let shifted = vaddq_f32(scaled, zp_vec);

        // Round to nearest integer
        let rounded = vrndnq_f32(shifted);

        // Clamp to [0, 255]
        let clamped = vmaxq_f32(vminq_f32(rounded, max_val), zero);

        // Convert to u32 then to u8
        let quantized_u32 = vcvtq_u32_f32(clamped);

        // Extract and convert to u8
        let mut temp = [0u32; 4];
        vst1q_u32(temp.as_mut_ptr(), quantized_u32);

        for &val in &temp {
            result.push(val as u8);
        }
    }

    // Handle remainder with scalar fallback
    for &value in &data[chunks * 4..] {
        let q = ((value / scale) + zero_point).round().clamp(0.0, 255.0) as u8;
        result.push(q);
    }

    result
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dequantize_neon(data: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    let mut result = Vec::with_capacity(data.len());
    let scale_vec = vdupq_n_f32(scale);
    let zp_vec = vdupq_n_f32(zero_point);

    // Process 4 u8 values at a time
    let chunks = data.len() / 4;
    for i in 0..chunks {
        let offset = i * 4;

        // Load 4 u8 values and convert to u32
        let mut temp = [0u32; 4];
        for j in 0..4 {
            temp[j] = data[offset + j] as u32;
        }
        let values_u32 = vld1q_u32(temp.as_ptr());

        // Convert u32 to f32
        let values_f32 = vcvtq_f32_u32(values_u32);

        // Dequantize: x = (q - zero_point) * scale
        let shifted = vsubq_f32(values_f32, zp_vec);
        let dequantized = vmulq_f32(shifted, scale_vec);

        // Store results
        let mut temp_f32 = [0.0f32; 4];
        vst1q_f32(temp_f32.as_mut_ptr(), dequantized);
        result.extend_from_slice(&temp_f32);
    }

    // Handle remainder with scalar fallback
    for &value in &data[chunks * 4..] {
        let x = (value as f32 - zero_point) * scale;
        result.push(x);
    }

    result
}

// ============================================================================
// SIMD Cosine Similarity
// ============================================================================

/// Compute cosine similarity between two vectors using SIMD acceleration
///
/// Cosine similarity = dot(a, b) / (norm(a) * norm(b))
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// Cosine similarity value in range [-1, 1], or 0.0 if either vector is zero
pub fn cosine_similarity_simd(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    if a.is_empty() {
        return 0.0;
    }

    // Use simsimd library for optimal SIMD performance
    // simsimd automatically selects the best SIMD implementation (AVX2, NEON, etc.)
    let dot = dot_product_simd(a, b);
    let norm_a = norm_simd(a);
    let norm_b = norm_simd(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a * norm_b)
}

/// Compute dot product using SIMD acceleration
fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { dot_product_avx2(a, b) }
    }

    #[cfg(all(
        not(all(target_arch = "x86_64", target_feature = "avx2")),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    {
        unsafe { dot_product_neon(a, b) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        dot_product_scalar(a, b)
    }
}

/// Compute L2 norm using SIMD acceleration
fn norm_simd(a: &[f32]) -> f32 {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    {
        unsafe { norm_avx2(a) }
    }

    #[cfg(all(
        not(all(target_arch = "x86_64", target_feature = "avx2")),
        target_arch = "aarch64",
        target_feature = "neon"
    ))]
    {
        unsafe { norm_neon(a) }
    }

    #[cfg(not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    )))]
    {
        norm_scalar(a)
    }
}

// AVX2 implementations
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(offset));
        let prod = _mm256_mul_ps(a_vec, b_vec);
        sum = _mm256_add_ps(sum, prod);
    }

    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        total += a[i] * b[i];
    }

    total
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn norm_avx2(a: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let chunks = a.len() / 8;

    for i in 0..chunks {
        let offset = i * 8;
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(offset));
        let squared = _mm256_mul_ps(a_vec, a_vec);
        sum = _mm256_add_ps(sum, squared);
    }

    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remainder
    for i in (chunks * 8)..a.len() {
        total += a[i] * a[i];
    }

    total.sqrt()
}

// NEON implementations
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = vld1q_f32(a.as_ptr().add(offset));
        let b_vec = vld1q_f32(b.as_ptr().add(offset));
        let prod = vmulq_f32(a_vec, b_vec);
        sum = vaddq_f32(sum, prod);
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    vst1q_f32(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        total += a[i] * b[i];
    }

    total
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn norm_neon(a: &[f32]) -> f32 {
    #[cfg(target_arch = "aarch64")]
    use std::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let chunks = a.len() / 4;

    for i in 0..chunks {
        let offset = i * 4;
        let a_vec = vld1q_f32(a.as_ptr().add(offset));
        let squared = vmulq_f32(a_vec, a_vec);
        sum = vaddq_f32(sum, squared);
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    vst1q_f32(result.as_mut_ptr(), sum);
    let mut total = result.iter().sum::<f32>();

    // Handle remainder
    for i in (chunks * 4)..a.len() {
        total += a[i] * a[i];
    }

    total.sqrt()
}

// Scalar fallback implementations
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm_scalar(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// ============================================================================
// Scalar Fallback Implementation
// ============================================================================

/// Scalar quantization fallback for platforms without SIMD support
fn quantize_scalar(data: &[f32], scale: f32, zero_point: f32) -> Vec<u8> {
    data.iter()
        .map(|&value| ((value / scale) + zero_point).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Scalar dequantization fallback for platforms without SIMD support
fn dequantize_scalar(data: &[u8], scale: f32, zero_point: f32) -> Vec<f32> {
    data.iter()
        .map(|&value| (value as f32 - zero_point) * scale)
        .collect()
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_quantize_scalar_basic() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let scale = 0.1;
        let zero_point = 0.0;

        let quantized = quantize_scalar(&data, scale, zero_point);

        assert_eq!(quantized, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    fn test_quantize_scalar_clamping() {
        let data = vec![-10.0, 0.0, 10.0, 100.0];
        let scale = 1.0;
        let zero_point = 0.0;

        let quantized = quantize_scalar(&data, scale, zero_point);

        assert_eq!(quantized, vec![0, 0, 10, 100]);
    }

    #[test]
    fn test_dequantize_scalar_basic() {
        let data = vec![0, 10, 20, 30, 40];
        let scale = 0.1;
        let zero_point = 0.0;

        let dequantized = dequantize_scalar(&data, scale, zero_point);

        assert_relative_eq!(dequantized[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(dequantized[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(dequantized[2], 2.0, epsilon = 1e-5);
        assert_relative_eq!(dequantized[3], 3.0, epsilon = 1e-5);
        assert_relative_eq!(dequantized[4], 4.0, epsilon = 1e-5);
    }

    #[test]
    fn test_quantize_simd_basic() {
        let data = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let scale = 0.1;
        let zero_point = 0.0;

        let quantized = quantize_simd(&data, scale, zero_point);

        assert_eq!(quantized.len(), data.len());
        assert_eq!(quantized[0], 0);
        assert_eq!(quantized[1], 10);
        assert_eq!(quantized[8], 80);
    }

    #[test]
    fn test_dequantize_simd_basic() {
        let data = vec![0, 10, 20, 30, 40, 50, 60, 70, 80];
        let scale = 0.1;
        let zero_point = 0.0;

        let dequantized = dequantize_simd(&data, scale, zero_point);

        assert_eq!(dequantized.len(), data.len());
        assert_relative_eq!(dequantized[0], 0.0, epsilon = 1e-5);
        assert_relative_eq!(dequantized[1], 1.0, epsilon = 1e-5);
        assert_relative_eq!(dequantized[8], 8.0, epsilon = 1e-5);
    }

    #[test]
    fn test_roundtrip_simd() {
        let original = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5, 9.5];
        let scale = 0.1;
        let zero_point = 0.0;

        let quantized = quantize_simd(&original, scale, zero_point);
        let dequantized = dequantize_simd(&quantized, scale, zero_point);

        assert_eq!(dequantized.len(), original.len());
        for (orig, deq) in original.iter().zip(dequantized.iter()) {
            // Allow for quantization error
            assert!((orig - deq).abs() < 0.2, "orig: {}, deq: {}", orig, deq);
        }
    }

    #[test]
    fn test_quantize_simd_with_zero_point() {
        let data = vec![-1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let scale = 0.1;
        let zero_point = 10.0;

        let quantized = quantize_simd(&data, scale, zero_point);

        // -1.0 / 0.1 + 10.0 = 0.0 (clamped)
        assert_eq!(quantized[0], 0);
        // 0.0 / 0.1 + 10.0 = 10.0
        assert_eq!(quantized[1], 10);
        // 1.0 / 0.1 + 10.0 = 20.0
        assert_eq!(quantized[2], 20);
    }

    #[test]
    fn test_quantize_simd_large_array() {
        let data: Vec<f32> = (0..1000).map(|i| i as f32 * 0.1).collect();
        let scale = 1.0;
        let zero_point = 0.0;

        let quantized = quantize_simd(&data, scale, zero_point);

        assert_eq!(quantized.len(), data.len());
        // Check first few values
        assert_eq!(quantized[0], 0);
        assert_eq!(quantized[10], 1);
        assert_eq!(quantized[100], 10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let similarity = cosine_similarity_simd(&a, &b);

        assert_relative_eq!(similarity, 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0, 0.0];

        let similarity = cosine_similarity_simd(&a, &b);

        assert_relative_eq!(similarity, 0.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];

        let similarity = cosine_similarity_simd(&a, &b);

        assert_relative_eq!(similarity, -1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![0.0, 0.0, 0.0];

        let similarity = cosine_similarity_simd(&a, &b);

        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_mismatched_length() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];

        let similarity = cosine_similarity_simd(&a, &b);

        assert_eq!(similarity, 0.0);
    }

    #[test]
    fn test_cosine_similarity_large_vectors() {
        let a: Vec<f32> = (0..1000).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1000).map(|i| (i as f32) * 2.0).collect();

        let similarity = cosine_similarity_simd(&a, &b);

        // Vectors are parallel, so similarity should be 1.0
        assert_relative_eq!(similarity, 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_dot_product_scalar() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];

        let result = dot_product_scalar(&a, &b);

        // 1*2 + 2*3 + 3*4 + 4*5 = 2 + 6 + 12 + 20 = 40
        assert_relative_eq!(result, 40.0, epsilon = 1e-5);
    }

    #[test]
    fn test_norm_scalar() {
        let a = vec![3.0, 4.0];

        let result = norm_scalar(&a);

        // sqrt(3^2 + 4^2) = sqrt(9 + 16) = sqrt(25) = 5.0
        assert_relative_eq!(result, 5.0, epsilon = 1e-5);
    }
}
