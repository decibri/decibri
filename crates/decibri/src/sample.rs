/// Convert f32 samples (range [-1.0, 1.0]) to 16-bit signed integer little-endian bytes.
///
/// Each f32 sample produces 2 bytes (i16 LE). Values are clamped to [-1.0, 1.0].
/// Matches the JS convention: `sample * 32768` then clamp to i16 range.
pub fn f32_to_i16_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        // Scale to i16 range. Use 32768.0 (not 32767.0) to match the JS convention
        // where computeRMS divides by 32768.
        let scaled = (clamped * 32768.0) as i32;
        // Clamp to valid i16 range [-32768, 32767]
        let val = scaled.clamp(-32768, 32767) as i16;
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Convert f32 samples to 32-bit IEEE 754 float little-endian bytes.
///
/// Each f32 sample produces 4 bytes (f32 LE). No clamping — values pass through as-is.
pub fn f32_to_f32_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 4);
    for &s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

/// Convert i16 samples to f32 (range [-1.0, 1.0]).
///
/// Divides by 32768.0, matching the JS convention.
pub fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / 32768.0).collect()
}

/// Convert 16-bit signed integer little-endian bytes to f32 samples.
///
/// Each pair of bytes is interpreted as an i16 LE value, then divided by 32768.0.
/// Used by the output path: JS sends i16 LE bytes, Rust needs f32 for cpal.
pub fn i16_le_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|c| {
            let val = i16::from_le_bytes([c[0], c[1]]);
            val as f32 / 32768.0
        })
        .collect()
}

/// Convert 32-bit IEEE 754 float little-endian bytes to f32 samples.
///
/// Each group of 4 bytes is interpreted as an f32 LE value.
/// Used by the output path: JS sends f32 LE bytes, Rust needs f32 for cpal.
pub fn f32_le_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_silence() {
        let bytes = f32_to_i16_le_bytes(&[0.0]);
        assert_eq!(bytes, [0x00, 0x00]);
    }

    #[test]
    fn test_max_positive() {
        let bytes = f32_to_i16_le_bytes(&[1.0]);
        // 1.0 * 32768 = 32768, clamped to 32767 = 0x7FFF
        let val = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(val, 32767);
    }

    #[test]
    fn test_max_negative() {
        let bytes = f32_to_i16_le_bytes(&[-1.0]);
        // -1.0 * 32768 = -32768 = 0x8000
        let val = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(val, -32768);
    }

    #[test]
    fn test_half_amplitude() {
        let bytes = f32_to_i16_le_bytes(&[0.5]);
        let val = i16::from_le_bytes([bytes[0], bytes[1]]);
        // 0.5 * 32768 = 16384
        assert_eq!(val, 16384);
    }

    #[test]
    fn test_clamp_above_one() {
        let bytes = f32_to_i16_le_bytes(&[1.5]);
        let val = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(val, 32767);
    }

    #[test]
    fn test_clamp_below_negative_one() {
        let bytes = f32_to_i16_le_bytes(&[-1.5]);
        let val = i16::from_le_bytes([bytes[0], bytes[1]]);
        assert_eq!(val, -32768);
    }

    #[test]
    fn test_roundtrip_i16() {
        let original: Vec<f32> = vec![0.0, 0.25, -0.25, 0.5, -0.5, 0.99, -0.99];
        let bytes = f32_to_i16_le_bytes(&original);

        // Reconstruct i16 values
        let i16_samples: Vec<i16> = bytes
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]))
            .collect();

        // Convert back to f32
        let reconstructed = i16_to_f32(&i16_samples);

        // Verify within quantization tolerance (1/32768 ≈ 0.0000305)
        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            let diff = (orig - recon).abs();
            assert!(
                diff < 1.0 / 32768.0 + f32::EPSILON,
                "Roundtrip error too large: {orig} -> {recon} (diff: {diff})"
            );
        }
    }

    #[test]
    fn test_f32_le_roundtrip() {
        let original: Vec<f32> = vec![0.0, 0.25, -0.25, 1.0, -1.0, 0.001, -0.999];
        let bytes = f32_to_f32_le_bytes(&original);

        let reconstructed: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_multiple_samples() {
        let samples = vec![0.0, 0.5, -0.5];
        let bytes = f32_to_i16_le_bytes(&samples);
        assert_eq!(bytes.len(), 6); // 3 samples * 2 bytes each

        let f32_bytes = f32_to_f32_le_bytes(&samples);
        assert_eq!(f32_bytes.len(), 12); // 3 samples * 4 bytes each
    }

    // ── Reverse conversion tests (bytes → f32, used by output) ───────────

    #[test]
    fn test_i16_le_bytes_to_f32_silence() {
        let bytes = vec![0x00, 0x00];
        let samples = i16_le_bytes_to_f32(&bytes);
        assert_eq!(samples, vec![0.0]);
    }

    #[test]
    fn test_i16_le_bytes_to_f32_max() {
        // 32767 = 0xFF7F in LE
        let bytes = vec![0xFF, 0x7F];
        let samples = i16_le_bytes_to_f32(&bytes);
        assert!((samples[0] - 32767.0 / 32768.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_i16_le_bytes_to_f32_roundtrip() {
        // f32 → i16 LE bytes → f32: verify roundtrip within quantization tolerance
        let original: Vec<f32> = vec![0.0, 0.25, -0.25, 0.5, -0.5, 0.99, -0.99];
        let bytes = f32_to_i16_le_bytes(&original);
        let reconstructed = i16_le_bytes_to_f32(&bytes);

        for (orig, recon) in original.iter().zip(reconstructed.iter()) {
            let diff = (orig - recon).abs();
            assert!(
                diff < 1.0 / 32768.0 + f32::EPSILON,
                "i16 LE roundtrip error: {orig} -> {recon} (diff: {diff})"
            );
        }
    }

    #[test]
    fn test_f32_le_bytes_to_f32_roundtrip() {
        let original: Vec<f32> = vec![0.0, 0.25, -0.25, 1.0, -1.0, 0.001, -0.999];
        let bytes = f32_to_f32_le_bytes(&original);
        let reconstructed = f32_le_bytes_to_f32(&bytes);
        assert_eq!(original, reconstructed);
    }

    #[test]
    fn test_i16_le_bytes_to_f32_empty() {
        let bytes: Vec<u8> = vec![];
        let samples = i16_le_bytes_to_f32(&bytes);
        assert!(samples.is_empty());
    }

    #[test]
    fn test_f32_le_bytes_to_f32_empty() {
        let bytes: Vec<u8> = vec![];
        let samples = f32_le_bytes_to_f32(&bytes);
        assert!(samples.is_empty());
    }
}
