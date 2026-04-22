//! Integration tests for the VAD / ORT resolution pipeline.
//!
//! These tests exercise `SileroVad` end-to-end against the bundled
//! `models/silero_vad.onnx` file. They run under `cargo test-decibri`
//! (which selects `--features ort-download-binaries`) and do not
//! require audio hardware.
//!
//! The load-failure test for a bogus `ort_library_path` lives in
//! `vad_ort_load_failure.rs` — a separate binary — because ORT
//! initialization is process-global: once any test in this binary
//! successfully initializes ORT, the `OnceLock` guard inside
//! `init_ort_once` short-circuits later callers and the load-failure
//! code path can no longer be reached.

#![cfg(feature = "vad")]

use std::path::{Path, PathBuf};

use decibri::error::DecibriError;
use decibri::vad::{SileroVad, VadConfig};

/// Absolute path to the bundled Silero VAD model, resolved from the crate
/// manifest directory. Matches the idiom used by the unit tests in
/// `crates/decibri/src/vad.rs`.
fn bundled_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join("silero_vad.onnx")
}

#[test]
fn test_ort_init_success_with_bundled_model() {
    let config = VadConfig {
        model_path: bundled_model_path(),
        ..VadConfig::default()
    };
    let vad = SileroVad::new(config);
    assert!(
        vad.is_ok(),
        "SileroVad::new should succeed with the bundled model: {:?}",
        vad.err()
    );
}

#[test]
fn test_vad_model_load_fails_for_missing_file() {
    let bogus = PathBuf::from("does-not-exist-silero-xyz.onnx");
    let config = VadConfig {
        model_path: bogus.clone(),
        ..VadConfig::default()
    };
    let err = SileroVad::new(config)
        .err()
        .expect("expected VadModelLoadFailed");
    match err {
        DecibriError::VadModelLoadFailed { path, .. } => {
            assert_eq!(
                path, bogus,
                "VadModelLoadFailed.path should match the input PathBuf"
            );
        }
        other => panic!("expected VadModelLoadFailed, got {other:?}"),
    }
}

#[test]
fn test_vad_config_validate_typed_errors() {
    // Unsupported sample rate
    let bad_rate = VadConfig {
        sample_rate: 44_100,
        ..VadConfig::default()
    };
    match bad_rate
        .validate()
        .expect_err("expected VadSampleRateUnsupported")
    {
        DecibriError::VadSampleRateUnsupported(sr) => {
            assert_eq!(sr, 44_100, "payload should carry the offending rate");
        }
        other => panic!("expected VadSampleRateUnsupported, got {other:?}"),
    }

    // Out-of-range threshold
    let bad_threshold = VadConfig {
        threshold: 1.5,
        ..VadConfig::default()
    };
    match bad_threshold
        .validate()
        .expect_err("expected VadThresholdOutOfRange")
    {
        DecibriError::VadThresholdOutOfRange(t) => {
            assert!(
                (t - 1.5).abs() < f32::EPSILON,
                "payload should carry the offending threshold, got {t}"
            );
        }
        other => panic!("expected VadThresholdOutOfRange, got {other:?}"),
    }

    // A valid config passes
    let good = VadConfig {
        model_path: bundled_model_path(),
        ..VadConfig::default()
    };
    assert!(good.validate().is_ok(), "default config should validate");
}

#[test]
fn test_vad_end_to_end_silence_inference() {
    let config = VadConfig {
        model_path: bundled_model_path(),
        ..VadConfig::default()
    };
    let mut vad = SileroVad::new(config).expect("model should load");

    // 1600 samples = three 512-sample windows + 64-sample remainder at 16 kHz.
    let silence = vec![0.0f32; 1600];
    let result = vad.process(&silence).expect("inference should succeed");

    assert!(
        result.probability < 0.5,
        "silence probability should be low, got {}",
        result.probability
    );
    assert!(
        !result.is_speech,
        "silence should not be classified as speech"
    );
}
