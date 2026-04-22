//! Integration tests for the ORT `load-dynamic` load-failure path.
//!
//! These tests live in their own file (separate test binary) because ORT
//! initialization is process-global: once `init_ort_once` has stored `Some`
//! in its `OnceLock`, any later call short-circuits and the load-failure
//! code path is unreachable. Keeping these tests alone in a binary guarantees
//! each run is the first ORT init in its process and thus actually exercises
//! the load-failure branch.
//!
//! The tests rely on the defensive pre-check added in
//! `crates/decibri/src/vad.rs::init_ort_once`, which validates that the
//! provided `ort_library_path` points at a regular file before handing it
//! to `ort::init_from`. Without that pre-check, Windows hangs indefinitely
//! inside `ort::init_from` on a nonexistent path (reproduced 2026-04-22
//! against pyke/ort 2.0.0-rc.12 + onnxruntime 1.24.4).
//!
//! Gated behind `feature = "ort-load-dynamic"` because the tests are
//! meaningful only under the dynamic-load distribution mode: under
//! `ort-download-binaries`, ORT is statically linked and any
//! `ort_library_path` argument is ignored by design.
//!
//! Under `cargo test-decibri` (which uses `ort-download-binaries`) the cfg
//! gate compiles these tests out. Run them explicitly with:
//!
//! ```text
//! cargo test -p decibri \
//!   --no-default-features \
//!   --features capture,output,vad,denoise,gain,ort-load-dynamic \
//!   --test vad_ort_load_failure
//! ```

#![cfg(all(feature = "vad", feature = "ort-load-dynamic"))]

use std::path::{Path, PathBuf};

use decibri::error::DecibriError;
use decibri::vad::{SileroVad, VadConfig};

fn bundled_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join("silero_vad.onnx")
}

/// A nonexistent `ort_library_path` must fail fast with `OrtLoadFailed`.
///
/// Prior to the defensive pre-check this test hung on Windows because
/// `ort::init_from` does not return for a nonexistent path; the pre-check
/// short-circuits that case.
#[test]
fn test_ort_load_fails_with_nonexistent_path() {
    let bogus = std::env::temp_dir().join("does-not-exist-onnxruntime-dylib-xyz");

    let config = VadConfig {
        model_path: bundled_model_path(),
        ort_library_path: Some(bogus.clone()),
        ..VadConfig::default()
    };
    let err = SileroVad::new(config)
        .err()
        .expect("expected OrtLoadFailed");

    match &err {
        DecibriError::OrtLoadFailed { path, .. } => {
            assert!(
                path.ends_with("does-not-exist-onnxruntime-dylib-xyz"),
                "OrtLoadFailed.path should end with the bogus filename, got {path}"
            );
        }
        other => panic!("expected OrtLoadFailed, got {other:?}"),
    }

    // Display message should carry the actionable guidance string.
    let msg = err.to_string();
    assert!(
        msg.contains("ORT_DYLIB_PATH"),
        "Display message should mention ORT_DYLIB_PATH, got: {msg}"
    );
}

/// An `ort_library_path` that points at a directory (e.g. a user mistakenly
/// passing the directory containing the dylib instead of the dylib itself)
/// must also fail fast with `OrtLoadFailed`. Without the pre-check, pyke/ort
/// likely hangs here on Windows as well because its dylib-load path does
/// not validate the input is a regular file.
#[test]
fn test_ort_load_fails_when_path_is_directory() {
    // `temp_dir()` is guaranteed to exist and is a directory on every platform.
    let dir_path = std::env::temp_dir();

    let config = VadConfig {
        model_path: bundled_model_path(),
        ort_library_path: Some(dir_path.clone()),
        ..VadConfig::default()
    };
    let err = SileroVad::new(config)
        .err()
        .expect("expected OrtLoadFailed when path is a directory");

    match &err {
        DecibriError::OrtLoadFailed { path, .. } => {
            assert_eq!(
                path,
                &dir_path.display().to_string(),
                "OrtLoadFailed.path should match the passed directory"
            );
        }
        other => panic!("expected OrtLoadFailed, got {other:?}"),
    }
}
