//! Integration tests for the ORT `load-dynamic` load-failure path.
//!
//! These tests live in their own file (separate test binary) because ORT
//! initialization is process-global: once `init_ort_once` has stored `Some`
//! in its `OnceLock`, any later call short-circuits and the load-failure
//! code path is unreachable. Keeping these tests alone in a binary guarantees
//! each run is the first ORT init in its process and thus actually exercises
//! the load-failure branch.
//!
//! The tests rely on the defensive pre-check in
//! `crates/decibri/src/onnx.rs::init_ort_once`, which validates that the
//! provided `ort_library_path` points at a regular file, then loads it and
//! checks it the way `ort` does (it exports `OrtGetApiBase` and reports the
//! ONNX Runtime version `ort` requires), before handing it to
//! `ort::init_from`. Without the filesystem check, Windows hangs indefinitely
//! inside `ort::init_from` on a nonexistent path (reproduced 2026-04-22
//! against pyke/ort 2.0.0-rc.12 + onnxruntime 1.24.4). That check returns
//! [`DecibriError::OrtPathInvalid`], deliberately a string-only variant that
//! does *not* construct an `ort::Error`, because `ort::Error::new` itself
//! calls `ortsys![CreateStatus]` and would re-trigger the very dylib load
//! we're avoiding. The load check returns [`DecibriError::OrtLoadFailed`]
//! carrying the `ort::LoadDynamicError` for the condition; a load that fails
//! inside `ort` cannot be retried in a process, so every attempt against a
//! file that is not a usable runtime has to fail the same typed way, which
//! the retry tests below pin.
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
//!   --features capture,playback,vad,denoise,gain,ort-load-dynamic \
//!   --test vad_ort_load_failure
//! ```

#![cfg(all(feature = "vad", feature = "ort-load-dynamic"))]

use std::path::{Path, PathBuf};
use std::sync::{Mutex, PoisonError};

use decibri::error::DecibriError;
use decibri::vad::{SileroVad, VadConfig};

/// Serializes the tests in this binary. One test mutates `ORT_DYLIB_PATH`
/// while the others read the environment (`temp_dir` consults it), and
/// concurrent environment mutation and reads are unsound on some platforms;
/// taking this lock first makes each test's environment access exclusive.
static ENV_LOCK: Mutex<()> = Mutex::new(());

fn bundled_model_path() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("models")
        .join("silero_vad.onnx")
}

/// A nonexistent `ort_library_path` must fail fast with `OrtPathInvalid`.
///
/// Prior to the defensive pre-check this test hung on Windows because
/// `ort::init_from` does not return for a nonexistent path; the pre-check
/// short-circuits that case without constructing an `ort::Error`.
#[test]
fn test_ort_load_fails_with_nonexistent_path() {
    let _env = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let bogus = std::env::temp_dir().join("does-not-exist-onnxruntime-dylib-xyz");

    // `VadConfig` is `#[non_exhaustive]`: default-construct then assign the
    // public fields rather than using a struct literal or `..default()` FRU.
    let mut config = VadConfig::default();
    config.model_path = bundled_model_path();
    config.ort_library_path = Some(bogus.clone());
    let err = SileroVad::new(config)
        .err()
        .expect("expected OrtPathInvalid");

    match &err {
        DecibriError::OrtPathInvalid { path, reason } => {
            assert!(
                path.ends_with("does-not-exist-onnxruntime-dylib-xyz"),
                "OrtPathInvalid.path should end with the bogus filename, got {}",
                path.display()
            );
            assert!(
                reason.contains("does not exist"),
                "OrtPathInvalid.reason should mention 'does not exist', got: {reason}"
            );
        }
        other => panic!("expected OrtPathInvalid, got {other:?}"),
    }

    // Display message should carry the actionable guidance string: same
    // wording as `OrtLoadFailed` so consumers see a uniform message.
    let msg = err.to_string();
    assert!(
        msg.contains("ORT_DYLIB_PATH"),
        "Display message should mention ORT_DYLIB_PATH, got: {msg}"
    );
}

/// With NO `ort_library_path` at all, the pre-check must still protect the
/// load: `ort::init()` resolves the runtime from `ORT_DYLIB_PATH` (or the
/// system loader), and a bogus `ORT_DYLIB_PATH` previously reached the
/// unguarded `ort::init()`, which hangs instead of erroring when the
/// resolution fails. The extended pre-check validates the environment
/// variable exactly as it validates an explicit path.
///
/// Environment note: this test sets `ORT_DYLIB_PATH` for its own scope. The
/// other tests in this binary pass an explicit `ort_library_path`, which
/// never consults the environment, so the mutation cannot race them.
#[test]
fn test_ort_load_fails_with_bogus_env_var_and_no_path() {
    let _env = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let bogus = std::env::temp_dir().join("does-not-exist-onnxruntime-env-xyz");
    std::env::set_var("ORT_DYLIB_PATH", &bogus);

    let mut config = VadConfig::default();
    config.model_path = bundled_model_path();
    config.ort_library_path = None;
    let result = SileroVad::new(config);
    std::env::remove_var("ORT_DYLIB_PATH");

    let err = result
        .err()
        .expect("expected OrtPathInvalid for a bogus ORT_DYLIB_PATH");
    match &err {
        DecibriError::OrtPathInvalid { path, reason } => {
            assert!(
                path.ends_with("does-not-exist-onnxruntime-env-xyz"),
                "OrtPathInvalid.path should carry the ORT_DYLIB_PATH value, got {}",
                path.display()
            );
            assert!(
                reason.contains("ORT_DYLIB_PATH"),
                "OrtPathInvalid.reason should name ORT_DYLIB_PATH, got: {reason}"
            );
        }
        other => panic!("expected OrtPathInvalid, got {other:?}"),
    }
}

/// An `ort_library_path` that points at a directory (e.g. a user mistakenly
/// passing the directory containing the dylib instead of the dylib itself)
/// must also fail fast with `OrtPathInvalid`. Without the pre-check, pyke/ort
/// likely hangs here on Windows as well because its dylib-load path does
/// not validate the input is a regular file.
#[test]
fn test_ort_load_fails_when_path_is_directory() {
    let _env = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    // `temp_dir()` is guaranteed to exist and is a directory on every platform.
    let dir_path = std::env::temp_dir();

    let mut config = VadConfig::default();
    config.model_path = bundled_model_path();
    config.ort_library_path = Some(dir_path.clone());
    let err = SileroVad::new(config)
        .err()
        .expect("expected OrtPathInvalid when path is a directory");

    match &err {
        DecibriError::OrtPathInvalid { path, .. } => {
            assert_eq!(
                path, &dir_path,
                "OrtPathInvalid.path should match the passed directory PathBuf"
            );
        }
        other => panic!("expected OrtPathInvalid, got {other:?}"),
    }
}

/// A `VadConfig` for the bundled model with the given `ort_library_path`.
fn config_for(ort_library_path: Option<PathBuf>) -> VadConfig {
    let mut config = VadConfig::default();
    config.model_path = bundled_model_path();
    config.ort_library_path = ort_library_path;
    config
}

/// A regular file in the temp directory that is not a shared library, named
/// for this process so concurrent test binaries cannot collide.
fn write_non_library(tag: &str) -> PathBuf {
    let file = std::env::temp_dir().join(format!("decibri-{tag}-{}.bin", std::process::id()));
    std::fs::write(&file, b"this is not a shared library\n").expect("write the fixture file");
    file
}

/// The `ort::LoadDynamicError` an `OrtLoadFailed` carries as its source.
fn load_dynamic_source(err: &DecibriError) -> Option<&ort::LoadDynamicError> {
    std::error::Error::source(err).and_then(|source| source.downcast_ref::<ort::LoadDynamicError>())
}

/// A regular file that is not a loadable library fails with `OrtLoadFailed`
/// carrying `ort::LoadDynamicError::Dlopen`, and a second attempt in the same
/// process fails the same way: the pre-check rejects the file before `ort`
/// loads anything, so nothing in the process changes between the attempts.
#[test]
fn test_ort_load_fails_the_same_way_twice_for_a_file_that_is_not_a_library() {
    let _env = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let file = write_non_library("not-a-library");

    for attempt in 1..=2 {
        let err = SileroVad::new(config_for(Some(file.clone())))
            .err()
            .unwrap_or_else(|| panic!("attempt {attempt}: expected OrtLoadFailed"));
        match &err {
            DecibriError::OrtLoadFailed { path, .. } => {
                assert_eq!(path, &file, "attempt {attempt}: OrtLoadFailed.path");
            }
            other => panic!("attempt {attempt}: expected OrtLoadFailed, got {other:?}"),
        }
        match load_dynamic_source(&err) {
            Some(ort::LoadDynamicError::Dlopen { path, .. }) => {
                assert_eq!(path, &file, "attempt {attempt}: Dlopen.path");
            }
            Some(other) => panic!("attempt {attempt}: expected Dlopen, got {other:?}"),
            None => {
                panic!("attempt {attempt}: OrtLoadFailed should carry an ort::LoadDynamicError")
            }
        }
    }

    let _ = std::fs::remove_file(&file);
}

/// A module that loads but is not ONNX Runtime, the test executable itself
/// where the loader accepts an executable, fails with `OrtLoadFailed` on
/// every attempt. The source is `MissingApi` where the module loads and
/// `Dlopen` where the loader refuses an executable; it is never anything
/// else, and the second attempt reports what the first did.
#[test]
fn test_ort_load_fails_the_same_way_twice_for_a_module_without_the_entry_point() {
    let _env = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let exe = std::env::current_exe().expect("current_exe");

    let mut first: Option<&'static str> = None;
    for attempt in 1..=2 {
        let err = SileroVad::new(config_for(Some(exe.clone())))
            .err()
            .unwrap_or_else(|| panic!("attempt {attempt}: expected OrtLoadFailed"));
        match &err {
            DecibriError::OrtLoadFailed { path, .. } => {
                assert_eq!(path, &exe, "attempt {attempt}: OrtLoadFailed.path");
            }
            other => panic!("attempt {attempt}: expected OrtLoadFailed, got {other:?}"),
        }
        let variant = match load_dynamic_source(&err) {
            Some(ort::LoadDynamicError::MissingApi { .. }) => "MissingApi",
            Some(ort::LoadDynamicError::Dlopen { .. }) => "Dlopen",
            Some(other) => {
                panic!("attempt {attempt}: expected MissingApi or Dlopen, got {other:?}")
            }
            None => {
                panic!("attempt {attempt}: OrtLoadFailed should carry an ort::LoadDynamicError")
            }
        };
        #[cfg(target_os = "windows")]
        assert_eq!(
            variant, "MissingApi",
            "attempt {attempt}: on Windows the executable loads as a module and lacks OrtGetApiBase"
        );
        match first {
            None => first = Some(variant),
            Some(previous) => assert_eq!(
                variant, previous,
                "the second attempt reports what the first did"
            ),
        }
    }
}

/// With no `ort_library_path`, an `ORT_DYLIB_PATH` that names a regular file
/// which is not a loadable library fails with `OrtLoadFailed` carrying
/// `Dlopen`, on the first attempt and on the second. Without the load check
/// this path reaches `ort::init()`, which loads lazily at the first C API
/// call and panics there.
///
/// Environment note: sets `ORT_DYLIB_PATH` for its own scope under `ENV_LOCK`,
/// like the bogus-path test above.
#[test]
fn test_ort_load_fails_the_same_way_twice_for_an_unusable_env_var_and_no_path() {
    let _env = ENV_LOCK.lock().unwrap_or_else(PoisonError::into_inner);
    let file = write_non_library("not-a-library-env");
    std::env::set_var("ORT_DYLIB_PATH", &file);
    let results: Vec<_> = (1..=2).map(|_| SileroVad::new(config_for(None))).collect();
    std::env::remove_var("ORT_DYLIB_PATH");
    let _ = std::fs::remove_file(&file);

    for (index, result) in results.into_iter().enumerate() {
        let attempt = index + 1;
        let err = result
            .err()
            .unwrap_or_else(|| panic!("attempt {attempt}: expected OrtLoadFailed"));
        match &err {
            DecibriError::OrtLoadFailed { path, .. } => {
                assert_eq!(
                    path, &file,
                    "attempt {attempt}: OrtLoadFailed.path is the ORT_DYLIB_PATH value"
                );
            }
            other => panic!("attempt {attempt}: expected OrtLoadFailed, got {other:?}"),
        }
        match load_dynamic_source(&err) {
            Some(ort::LoadDynamicError::Dlopen { path, .. }) => {
                assert_eq!(path, &file, "attempt {attempt}: Dlopen.path");
            }
            Some(other) => panic!("attempt {attempt}: expected Dlopen, got {other:?}"),
            None => {
                panic!("attempt {attempt}: OrtLoadFailed should carry an ort::LoadDynamicError")
            }
        }
    }
}
