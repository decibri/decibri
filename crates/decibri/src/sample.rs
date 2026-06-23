/// Convert f32 samples (range [-1.0, 1.0]) to 16-bit signed integer little-endian bytes.
///
/// Each f32 sample produces 2 bytes (i16 LE). Values are clamped to [-1.0, 1.0].
/// Matches the JS convention: `sample * 32768` then clamp to i16 range.
pub fn f32_to_i16_le_bytes(samples: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(samples.len() * 2);
    for &s in samples {
        let clamped = s.clamp(-1.0, 1.0);
        // Scale to i16 range. Use 32768.0 (not 32767.0) so the i16 normalization
        // is the exact inverse of the `/ 32768.0` in `i16_le_bytes_to_f32`, the
        // same `[0, 1]` scale `rms` reports on.
        let scaled = (clamped * 32768.0) as i32;
        // Clamp to valid i16 range [-32768, 32767]
        let val = scaled.clamp(-32768, 32767) as i16;
        out.extend_from_slice(&val.to_le_bytes());
    }
    out
}

/// Convert f32 samples to 32-bit IEEE 754 float little-endian bytes.
///
/// Each f32 sample produces 4 bytes (f32 LE). No clamping; values pass through as-is.
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
///
/// Crate-internal (`pub(crate)`), not part of decibri's public sample API: the
/// bindings only ever convert from bytes via [`i16_le_bytes_to_f32`], never
/// from `i16` samples. Its sole caller is the `i16` roundtrip unit test below,
/// so it is `#[cfg(test)]`-gated to keep it off the public surface without
/// tripping the dead-code lint in non-test builds.
#[cfg(test)]
pub(crate) fn i16_to_f32(samples: &[i16]) -> Vec<f32> {
    samples.iter().map(|&s| s as f32 / 32768.0).collect()
}

/// Convert 16-bit signed integer little-endian bytes to f32 samples.
///
/// Each pair of bytes is interpreted as an i16 LE value, then divided by 32768.0.
/// Used by the output path: JS sends i16 LE bytes, Rust needs f32 for cpal.
///
/// A trailing odd byte (when `bytes.len()` is odd) is ignored. The public
/// bindings only ever pass even-length PCM buffers, so this is a defensive
/// note rather than a reachable case.
pub fn i16_le_bytes_to_f32(bytes: &[u8]) -> Vec<f32> {
    debug_assert!(
        bytes.len().is_multiple_of(2),
        "i16_le_bytes_to_f32 expects an even-length buffer; a trailing odd byte is dropped"
    );
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

/// Downmix interleaved multichannel audio to mono by averaging channels.
///
/// `samples` holds interleaved f32 frames of `channels` samples each
/// (`[c0, c1, .., c0, c1, ..]`). Each frame collapses to one mono sample equal
/// to the arithmetic mean of its channels, so `N` interleaved channels produce
/// `samples.len() / channels` mono samples.
///
/// `channels <= 1` returns the input unchanged (already mono, no allocation of a
/// reshaped buffer beyond the copy). Any trailing partial frame (when
/// `samples.len()` is not a multiple of `channels`) is dropped, matching the
/// frame-exact buffering the capture path guarantees.
///
/// Used to feed mono audio to the Silero VAD, which models a single channel:
/// without this, interleaved samples are misread as consecutive mono samples.
/// The interleaved data delivered to the consumer is left untouched; only the
/// VAD's private input is downmixed.
pub fn downmix_to_mono(samples: &[f32], channels: u16) -> Vec<f32> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let channels = channels as usize;
    samples
        .chunks_exact(channels)
        .map(|frame| frame.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Root-mean-square level of f32 samples, returned in `[0.0, 1.0]`.
///
/// Computes `sqrt(mean(x^2))` over `samples`, the energy-VAD score. Returns
/// `0.0` for an empty slice (no division by zero). Accumulates the sum of
/// squares in `f64` so a long block does not lose precision, then narrows the
/// result to `f32`.
///
/// The samples are expected in the normalized `[-1.0, 1.0]` domain (decibri's
/// internal representation), so the result shares the same `/32768` `[0, 1]`
/// scale as the i16 path (see [`f32_to_i16_le_bytes`]): a quarter-full-scale
/// signal returns about `0.25`, matching what the i16-normalized RMS would
/// give within quantization. Feed it the mono signal (downmix first when the
/// frame is interleaved, as [`downmix_to_mono`] does), so the score reflects
/// one channel's level.
pub fn rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = samples.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / samples.len() as f64).sqrt() as f32
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

    // ── downmix_to_mono (multichannel -> mono for the VAD feed) ───────────

    #[test]
    fn test_downmix_stereo_averages_and_halves() {
        // L R L R: two stereo frames -> two mono samples, each the average.
        let stereo = vec![0.5, 0.3, 0.4, 0.6];
        let mono = downmix_to_mono(&stereo, 2);
        // N interleaved channels produce 1/N the sample count.
        assert_eq!(mono.len(), stereo.len() / 2);
        assert!((mono[0] - 0.4).abs() < 1e-6); // (0.5 + 0.3) / 2
        assert!((mono[1] - 0.5).abs() < 1e-6); // (0.4 + 0.6) / 2
    }

    #[test]
    fn test_downmix_mono_passthrough() {
        // channels == 1 is already mono: returns the samples unchanged.
        let samples = vec![0.1, -0.2, 0.3];
        assert_eq!(downmix_to_mono(&samples, 1), samples);
        // channels == 0 is degenerate; treated as already-mono (no panic / div0).
        assert_eq!(downmix_to_mono(&samples, 0), samples);
    }

    #[test]
    fn test_downmix_six_channel() {
        // One 5.1 frame averages all six channels into a single sample.
        let frame = vec![0.0, 0.6, 0.3, -0.3, 1.2, -1.8];
        let mono = downmix_to_mono(&frame, 6);
        assert_eq!(mono.len(), 1);
        assert!((mono[0] - 0.0).abs() < 1e-6); // sum is 0.0 -> mean 0.0
    }

    #[test]
    fn test_downmix_preserves_sign_and_magnitude() {
        // Opposite-phase channels cancel; equal channels pass through.
        assert_eq!(downmix_to_mono(&[1.0, -1.0], 2), vec![0.0]);
        assert_eq!(downmix_to_mono(&[-0.5, -0.5], 2), vec![-0.5]);
    }

    #[test]
    fn test_downmix_drops_trailing_partial_frame() {
        // 5 samples at 2 channels: two full frames, the trailing sample dropped.
        let stereo = vec![0.2, 0.4, 0.6, 0.8, 0.9];
        let mono = downmix_to_mono(&stereo, 2);
        assert_eq!(mono.len(), 2);
        assert!((mono[0] - 0.3).abs() < 1e-6);
        assert!((mono[1] - 0.7).abs() < 1e-6);
    }

    #[test]
    fn test_downmix_empty() {
        let empty: Vec<f32> = vec![];
        assert!(downmix_to_mono(&empty, 2).is_empty());
    }

    // ── rms (energy-VAD score) ────────────────────────────────────────────

    #[test]
    fn test_rms_silence_is_zero() {
        assert_eq!(rms(&[0.0; 512]), 0.0);
    }

    #[test]
    fn test_rms_empty_is_zero() {
        // No samples: 0.0, not a division by zero.
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn test_rms_constant_equals_amplitude() {
        // A constant signal's RMS equals its absolute amplitude. 0.25 full scale
        // returns 0.25, the same [0, 1] scale the i16-normalized path produces.
        let rms_val = rms(&[0.25; 1000]);
        assert!((rms_val - 0.25).abs() < 1e-6, "rms {rms_val}");
    }

    #[test]
    fn test_rms_sine_is_amplitude_over_root_two() {
        // RMS of a sine of amplitude A is A / sqrt(2).
        let amplitude = 0.5_f32;
        let samples: Vec<f32> = (0..4000)
            .map(|k| amplitude * (k as f32 * 0.05).sin())
            .collect();
        let rms_val = rms(&samples);
        let expected = amplitude / 2.0_f32.sqrt();
        assert!(
            (rms_val - expected).abs() < 1e-2,
            "rms {rms_val} vs {expected}"
        );
    }

    /// The f32 RMS sits on the same `[0, 1]` scale as the legacy i16-normalized
    /// RMS (quantize by `* 32768`, normalize by `/ 32768`) within quantization,
    /// so the `0.01` energy-threshold default keeps its meaning when the score
    /// is computed on the f32 signal rather than the quantized i16 samples.
    #[test]
    fn test_rms_matches_i16_normalized_convention() {
        let samples: Vec<f32> = (0..2000).map(|k| 0.1 * (k as f32 * 0.05).sin()).collect();
        let f32_rms = rms(&samples);
        // Legacy convention: quantize to i16 (matching f32_to_i16_le_bytes),
        // then compute RMS on i16 / 32768.
        let i16_norm: Vec<f64> = samples
            .iter()
            .map(|&s| {
                let q = (s.clamp(-1.0, 1.0) * 32768.0) as i32;
                let v = q.clamp(-32768, 32767) as i16;
                v as f64 / 32768.0
            })
            .collect();
        let legacy_rms =
            (i16_norm.iter().map(|x| x * x).sum::<f64>() / i16_norm.len() as f64).sqrt() as f32;
        assert!(
            (f32_rms - legacy_rms).abs() < 1e-3,
            "f32 rms {f32_rms} vs legacy i16-normalized rms {legacy_rms}"
        );
    }
}
