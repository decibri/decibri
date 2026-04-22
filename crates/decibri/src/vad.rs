//! Voice Activity Detection using the Silero VAD v5 ONNX model.
//!
//! Silero VAD is a pre-trained ML model that detects human speech in audio.
//! It runs inference on 512-sample windows (at 16kHz) and returns a speech
//! probability between 0.0 and 1.0.
//!
//! # Model tensor specification (verified from silero_vad.onnx)
//!
//! Inputs:
//!   - `input`:  f32[batch, window_size]  (audio samples)
//!   - `state`:  f32[2, batch, 128]       (LSTM state, hidden + cell combined)
//!   - `sr`:     i64 scalar               (sample rate)
//!
//! Outputs:
//!   - `output`: f32[batch, 1]            (speech probability)
//!   - `stateN`: f32[2, batch, 128]       (updated LSTM state)

#[cfg(feature = "vad")]
use std::path::{Path, PathBuf};
#[cfg(feature = "vad")]
use std::sync::OnceLock;

use crate::error::DecibriError;

/// Configuration for Silero VAD.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Path to the silero_vad.onnx model file.
    pub model_path: PathBuf,
    /// Sample rate: 8000 or 16000 Hz.
    pub sample_rate: u32,
    /// Speech probability threshold (0.0 to 1.0). Default: 0.5.
    pub threshold: f32,
    /// Optional absolute path to the ONNX Runtime shared library.
    ///
    /// When `Some(path)`, ORT is initialized from the given library path via
    /// `ort::init_from`. When `None`, `ort::init()` is called and ORT honours
    /// the `ORT_DYLIB_PATH` environment variable if set.
    ///
    /// ORT is initialized exactly once per process. If multiple `SileroVad`
    /// instances are constructed with different `ort_library_path` values,
    /// the first path wins and later paths are silently ignored. This matches
    /// ORT's own single-global-runtime model.
    pub ort_library_path: Option<PathBuf>,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("silero_vad.onnx"),
            sample_rate: 16000,
            threshold: 0.5,
            ort_library_path: None,
        }
    }
}

impl VadConfig {
    /// Validate configuration values and return the Silero VAD window size
    /// derived from the sample rate.
    ///
    /// Called automatically by [`SileroVad::new`]; exposed publicly so
    /// callers can fail-fast before paying the cost of ORT init and
    /// ONNX model load. The returned `usize` is the `window_size` the
    /// VAD model operates on (512 for 16 kHz, 256 for 8 kHz). Direct
    /// consumers that only care about pass/fail can use `.is_ok()` or
    /// `.map(|_| ())`.
    ///
    /// Pairing the window-size derivation with the sample-rate validation
    /// means a future extension to `validate()` that accepts new sample
    /// rates must also yield a corresponding window size, turning a
    /// potential runtime `unreachable!` panic into a compile error.
    ///
    /// # Errors
    /// - [`DecibriError::VadSampleRateUnsupported`] if `sample_rate` is not
    ///   8000 or 16000.
    /// - [`DecibriError::VadThresholdOutOfRange`] if `threshold` is outside
    ///   `[0.0, 1.0]`.
    pub fn validate(&self) -> Result<usize, DecibriError> {
        let window_size = match self.sample_rate {
            8000 => 256,
            16000 => 512,
            _ => return Err(DecibriError::VadSampleRateUnsupported(self.sample_rate)),
        };
        if !(0.0..=1.0).contains(&self.threshold) {
            return Err(DecibriError::VadThresholdOutOfRange(self.threshold));
        }
        Ok(window_size)
    }
}

/// Result of processing an audio chunk through Silero VAD.
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Maximum speech probability across all windows in the chunk.
    pub probability: f32,
    /// Whether probability >= threshold.
    pub is_speech: bool,
}

/// Silero VAD v5 inference engine.
///
/// Stateful: maintains LSTM hidden/cell state across calls.
/// Call `process()` with each audio chunk. Call `reset()` to clear state.
#[cfg(feature = "vad")]
pub struct SileroVad {
    session: ort::session::Session,
    /// Combined LSTM state [2, 1, 128] = 256 floats.
    state: Vec<f32>,
    sample_rate: u32,
    threshold: f32,
    /// Carries leftover samples between process() calls.
    accumulator: Vec<f32>,
    /// 512 for 16kHz, 256 for 8kHz.
    window_size: usize,
}

/// State size: 2 (hidden + cell) * 1 (batch) * 128 (hidden_dim) = 256.
/// State size: 2 (hidden + cell layers) × batch(1) × hidden_dim(128).
const STATE_SIZE: usize = 256;

/// Process-global ORT init tracker. Stores the library path used on first
/// successful init (None if init used `ort::init()` with env-var fallback).
/// Subsequent `init_ort_once` calls see this as populated and return immediately.
#[cfg(feature = "vad")]
static ORT_INIT: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Wrap an ORT init failure with a decibri-specific actionable message.
///
/// Kept as a standalone function (rather than an inline closure in
/// `init_ort_once`) so it can be unit-tested without triggering a real ORT
/// init failure. Tests construct a synthetic `ort::Error` via
/// [`ort::Error::new`].
///
/// Returns one of two typed variants depending on whether a path was provided:
/// [`DecibriError::OrtLoadFailed`] when `path.is_some()`, or
/// [`DecibriError::OrtInitFailed`] otherwise. The inner `ort::Error` is
/// carried via `#[source]` so consumers can walk the error chain.
#[cfg(feature = "vad")]
fn wrap_init_error(path: Option<&Path>, err: ort::Error) -> DecibriError {
    match path {
        Some(p) => DecibriError::OrtLoadFailed {
            path: p.to_path_buf(),
            source: err,
        },
        None => DecibriError::OrtInitFailed { source: err },
    }
}

/// Perform the actual ORT init call. Split out by distribution-mode feature
/// so `init_ort_once` stays single-path.
///
/// Under `ort-load-dynamic`: `ort::init_from(path)` is available and is
/// fallible (validates the dylib up-front).
///
/// Under `ort-download-binaries`: `ort::init_from` does NOT exist. ORT is
/// statically linked into the binary and any path argument is meaningless.
/// The path is ignored; we call `ort::init()` to commit an
/// `EnvironmentBuilder` so our OnceLock bookkeeping fires.
#[cfg(all(feature = "vad", feature = "ort-load-dynamic"))]
fn do_ort_init(path: Option<&Path>) -> Result<bool, ort::Error> {
    match path {
        Some(p) => ort::init_from(p).map(|b| b.with_name("decibri").commit()),
        None => Ok(ort::init().with_name("decibri").commit()),
    }
}

#[cfg(all(feature = "vad", not(feature = "ort-load-dynamic")))]
fn do_ort_init(_path: Option<&Path>) -> Result<bool, ort::Error> {
    Ok(ort::init().with_name("decibri").commit())
}

/// Initialize ORT exactly once per process.
///
/// - If ORT is already initialized (by this or any prior caller), returns
///   immediately. The `path` argument is silently ignored. ORT's global
///   state cannot be re-initialized. See `VadConfig::ort_library_path` docs.
/// - Otherwise delegates to the feature-gated `do_ort_init` helper.
/// - On failure, the `OnceLock` is NOT set, so a subsequent caller can retry.
///
/// Note: `e` in the error-wrapping path below is always an `ort::Error` (ORT's
/// own error type), never a `DecibriError`. This function is the single place
/// where ORT errors enter decibri's error hierarchy. Do NOT apply the same
/// wrapping pattern elsewhere or the `decibri:` prefix and guidance string
/// will be duplicated in the message users see.
#[cfg(feature = "vad")]
fn init_ort_once(path: Option<&Path>) -> Result<(), DecibriError> {
    // Fast path: ORT already initialized.
    if ORT_INIT.get().is_some() {
        return Ok(());
    }

    // Defensive pre-check (load-dynamic only): verify the path points at a
    // regular file before handing it to `ort::init_from`. A nonexistent path
    // or a path that points at a directory causes `ort::init_from` on Windows
    // to hang indefinitely (reproduced 2026-04-22 against pyke/ort
    // 2.0.0-rc.12 + onnxruntime 1.24.4). Failing fast here turns that hang
    // into a clean typed error that Node, Python, and mobile consumers can
    // surface to users.
    //
    // The check is intentionally scoped to `load-dynamic`: under
    // `download-binaries`, ORT is statically linked and the path argument
    // is ignored, so there is nothing to pre-validate.
    #[cfg(feature = "ort-load-dynamic")]
    if let Some(p) = path {
        if !p.is_file() {
            // Use `OrtPathInvalid`, not `OrtLoadFailed`, precisely because
            // constructing an `ort::Error` here would call `ortsys![
            // CreateStatus]` — which triggers the ORT dylib load that this
            // pre-check is designed to prevent. `OrtPathInvalid` is
            // string-only and never touches ORT symbols.
            return Err(DecibriError::OrtPathInvalid {
                path: p.to_path_buf(),
                reason: "path does not exist or is not a regular file",
            });
        }
    }

    match do_ort_init(path) {
        Ok(_committed) => {
            // First-caller-wins. If another thread set it first, discard ours;
            // ORT's own global init is idempotent (first takes effect).
            let _ = ORT_INIT.set(path.map(|p| p.to_path_buf()));
            Ok(())
        }
        Err(e) => Err(wrap_init_error(path, e)),
    }
}

#[cfg(feature = "vad")]
impl SileroVad {
    /// Create a new Silero VAD instance by loading the ONNX model.
    pub fn new(config: VadConfig) -> Result<Self, DecibriError> {
        // `validate` returns the `window_size` for the accepted sample rate,
        // so `SileroVad::new` does not need its own match — and cannot panic
        // if future `validate` extensions accept new sample rates.
        let window_size = config.validate()?;

        // Initialize ORT exactly once per process. Subsequent SileroVad
        // instances pass through immediately regardless of their ort_library_path.
        init_ort_once(config.ort_library_path.as_deref())?;

        let session = ort::session::Session::builder()
            .map_err(DecibriError::OrtSessionBuildFailed)?
            // `with_intra_threads` returns `ort::Error<SessionBuilder>`; use a
            // closure with `.into()` to convert to `ort::Error<()>` via the
            // upstream `From<Error<SessionBuilder>> for Error<()>` impl.
            .with_intra_threads(1)
            .map_err(|e| DecibriError::OrtThreadsConfigFailed(e.into()))?
            .commit_from_file(&config.model_path)
            .map_err(|e| DecibriError::VadModelLoadFailed {
                path: config.model_path.clone(),
                source: e,
            })?;

        Ok(Self {
            session,
            state: vec![0.0f32; STATE_SIZE],
            sample_rate: config.sample_rate,
            threshold: config.threshold,
            accumulator: Vec::new(),
            window_size,
        })
    }

    /// Process audio samples and return a VAD result.
    ///
    /// Samples are f32 in the range [-1.0, 1.0]. The chunk can be any size;
    /// internally it is split into `window_size` windows. Leftover samples
    /// carry over to the next call.
    ///
    /// Returns the maximum speech probability across all windows processed.
    /// If no complete windows were formed (chunk too small), returns probability 0.0.
    pub fn process(&mut self, samples: &[f32]) -> Result<VadResult, DecibriError> {
        self.accumulator.extend_from_slice(samples);

        let mut max_probability: f32 = 0.0;
        let mut windows_processed = 0;

        while self.accumulator.len() >= self.window_size {
            let window: Vec<f32> = self.accumulator.drain(..self.window_size).collect();
            let probability = self.infer_window(&window)?;
            max_probability = max_probability.max(probability);
            windows_processed += 1;
        }

        // If no windows processed, return 0 probability (not enough data yet)
        if windows_processed == 0 {
            return Ok(VadResult {
                probability: 0.0,
                is_speech: false,
            });
        }

        Ok(VadResult {
            probability: max_probability,
            is_speech: max_probability >= self.threshold,
        })
    }

    /// Reset LSTM state to zeros (start of new utterance).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.accumulator.clear();
    }

    /// Run inference on a single window of exactly `window_size` samples.
    fn infer_window(&mut self, window: &[f32]) -> Result<f32, DecibriError> {
        // Create input tensors using actual model tensor names:
        //   input: f32[1, window_size]
        //   state: f32[2, 1, 128]
        //   sr:    i64 scalar
        let input_tensor =
            ort::value::Tensor::from_array(([1i64, self.window_size as i64], window.to_vec()))
                .map_err(|e| DecibriError::OrtTensorCreateFailed {
                    kind: "input",
                    source: e,
                })?;
        let state_tensor =
            ort::value::Tensor::from_array(([2i64, 1i64, 128i64], self.state.clone())).map_err(
                |e| DecibriError::OrtTensorCreateFailed {
                    kind: "state",
                    source: e,
                },
            )?;
        let sr_tensor = ort::value::Tensor::from_array(([1i64], vec![self.sample_rate as i64]))
            .map_err(|e| DecibriError::OrtTensorCreateFailed {
                kind: "sr",
                source: e,
            })?;

        let input_values = ort::inputs![
            "input" => input_tensor,
            "state" => state_tensor,
            "sr" => sr_tensor,
        ];

        let outputs = self
            .session
            .run(input_values)
            .map_err(DecibriError::OrtInferenceFailed)?;

        // Read outputs using actual model tensor names:
        //   output: f32[1, 1] (speech probability)
        //   stateN: f32[2, 1, 128] (updated state)
        let prob_tensor = outputs["output"].try_extract_tensor::<f32>().map_err(|e| {
            DecibriError::OrtTensorExtractFailed {
                kind: "output",
                source: e,
            }
        })?;
        let probability = prob_tensor.1[0];

        let state_tensor = outputs["stateN"].try_extract_tensor::<f32>().map_err(|e| {
            DecibriError::OrtTensorExtractFailed {
                kind: "state",
                source: e,
            }
        })?;
        self.state.copy_from_slice(state_tensor.1);

        Ok(probability)
    }
}

#[cfg(all(test, feature = "vad"))]
mod tests {
    use super::*;
    use std::path::Path;

    fn model_path() -> PathBuf {
        // Resolve relative to the workspace root
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir)
            .join("..")
            .join("..")
            .join("models")
            .join("silero_vad.onnx")
    }

    fn default_config() -> VadConfig {
        VadConfig {
            model_path: model_path(),
            sample_rate: 16000,
            threshold: 0.5,
            ort_library_path: None,
        }
    }

    #[test]
    fn test_wrap_init_error_with_path() {
        use std::env;

        // Platform-agnostic invalid path (guaranteed to not exist on any OS).
        let bogus_path = env::temp_dir().join("does-not-exist-onnxruntime-xyz-test");
        let err = wrap_init_error(
            Some(bogus_path.as_path()),
            ort::Error::new("simulated ort loader failure"),
        );
        let msg = err.to_string();

        assert!(
            msg.contains(&bogus_path.display().to_string()),
            "error message should contain the attempted path, got: {msg}"
        );
        assert!(
            msg.contains("If ORT_DYLIB_PATH is set"),
            "error message should contain actionable guidance phrase, got: {msg}"
        );
        assert!(
            msg.contains("simulated ort loader failure"),
            "error message should include the underlying ort error, got: {msg}"
        );
    }

    #[test]
    fn test_wrap_init_error_without_path() {
        let err = wrap_init_error(None, ort::Error::new("simulated ort init failure"));
        let msg = err.to_string();

        assert!(
            msg.contains("ort_library_path"),
            "None-path error should mention the VadConfig field, got: {msg}"
        );
        assert!(
            msg.contains("ORT_DYLIB_PATH"),
            "None-path error should mention the env var, got: {msg}"
        );
        assert!(
            msg.contains("ort-download-binaries"),
            "None-path error should mention the opt-out feature, got: {msg}"
        );
        assert!(
            msg.contains("simulated ort init failure"),
            "error message should include the underlying ort error, got: {msg}"
        );
    }

    #[test]
    fn test_vad_config_validation() {
        let bad_rate = VadConfig {
            sample_rate: 44100,
            ..default_config()
        };
        assert!(SileroVad::new(bad_rate).is_err());
    }

    /// Core guarantee of the commit 3.8 structured-error refactor:
    /// `err.source()` walks to the underlying `ort::Error` for the wrapped
    /// variants, and deliberately returns `None` for `OrtPathInvalid` (which
    /// has no underlying ORT error because constructing one would trigger
    /// the hang the pre-check prevents).
    #[test]
    fn test_ort_error_source_chain_preserved() {
        use std::error::Error;

        // OrtInitFailed (no path) carries an ort::Error source.
        let inner = ort::Error::new("simulated underlying ort error");
        let err = wrap_init_error(None, inner);
        assert!(
            err.source().is_some(),
            "OrtInitFailed should carry an ort::Error source"
        );

        // OrtLoadFailed (with path) carries an ort::Error source.
        let inner_with_path = ort::Error::new("another simulated error");
        let path_err = wrap_init_error(Some(Path::new("/tmp/bogus")), inner_with_path);
        assert!(
            path_err.source().is_some(),
            "OrtLoadFailed should carry an ort::Error source"
        );

        // OrtPathInvalid intentionally does NOT carry a source (documented
        // asymmetry — see OrtPathInvalid's rustdoc in error.rs).
        let path_invalid = DecibriError::OrtPathInvalid {
            path: PathBuf::from("/tmp/nope"),
            reason: "test",
        };
        assert!(
            path_invalid.source().is_none(),
            "OrtPathInvalid intentionally has no source (constructing ort::Error \
             would trigger the hang the pre-check prevents)"
        );
    }

    /// `DecibriError::is_ort_path_error` groups the two path-level ORT
    /// failure variants so consumers handling "the path is wrong" logic
    /// can match one predicate instead of enumerating both.
    #[test]
    fn test_is_ort_path_error() {
        let load_failed = DecibriError::OrtLoadFailed {
            path: PathBuf::from("/tmp/x"),
            source: ort::Error::new("test"),
        };
        assert!(load_failed.is_ort_path_error());

        let path_invalid = DecibriError::OrtPathInvalid {
            path: PathBuf::from("/tmp/x"),
            reason: "test",
        };
        assert!(path_invalid.is_ort_path_error());

        // Other variants return false.
        let init_failed = DecibriError::OrtInitFailed {
            source: ort::Error::new("test"),
        };
        assert!(!init_failed.is_ort_path_error());

        // `SampleRateOutOfRange` is a unit variant (applies to capture/output
        // configs, not VAD); Ross's draft had `(999999)` which wouldn't
        // compile. Using the correct unit form.
        let sample_rate = DecibriError::SampleRateOutOfRange;
        assert!(!sample_rate.is_ort_path_error());

        // VAD-specific config error also false.
        let vad_rate = DecibriError::VadSampleRateUnsupported(44_100);
        assert!(!vad_rate.is_ort_path_error());
    }

    #[test]
    fn test_vad_loads_model() {
        let vad = SileroVad::new(default_config());
        assert!(vad.is_ok(), "Model should load: {:?}", vad.err());
    }

    #[test]
    fn test_vad_silence() {
        let mut vad = SileroVad::new(default_config()).unwrap();
        let silence = vec![0.0f32; 512];
        let result = vad.process(&silence).unwrap();
        // Silence should have low probability
        assert!(
            result.probability < 0.5,
            "Silence probability should be low, got {}",
            result.probability
        );
    }

    #[test]
    fn test_vad_state_persistence() {
        let mut vad = SileroVad::new(default_config()).unwrap();
        let initial_state = vad.state.clone();

        let samples = vec![0.0f32; 512];
        vad.process(&samples).unwrap();

        // State should have changed after inference
        assert_ne!(
            vad.state, initial_state,
            "State should change after inference"
        );
    }

    #[test]
    fn test_vad_accumulator_windows() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // 1600 samples = 3 windows of 512 + 64 leftover
        let samples = vec![0.0f32; 1600];
        let result = vad.process(&samples).unwrap();
        assert!(result.probability >= 0.0); // Just verify it runs

        // 64 samples should remain in accumulator
        assert_eq!(vad.accumulator.len(), 64);
    }

    #[test]
    fn test_vad_accumulator_carry() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // First chunk: 1600 samples → 3 windows, 64 leftover
        let chunk1 = vec![0.0f32; 1600];
        vad.process(&chunk1).unwrap();
        assert_eq!(vad.accumulator.len(), 64);

        // Second chunk: 1600 samples → 64 + 1600 = 1664 → 3 windows, 128 leftover
        let chunk2 = vec![0.0f32; 1600];
        vad.process(&chunk2).unwrap();
        assert_eq!(vad.accumulator.len(), 128);
    }

    #[test]
    fn test_vad_small_chunk() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // Less than one window, so no inference should run
        let samples = vec![0.0f32; 100];
        let result = vad.process(&samples).unwrap();
        assert_eq!(result.probability, 0.0);
        assert_eq!(vad.accumulator.len(), 100);
    }

    #[test]
    fn test_vad_reset() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // Process something to change state
        let samples = vec![0.0f32; 512];
        vad.process(&samples).unwrap();
        vad.accumulator.extend_from_slice(&[0.0; 100]); // add some leftover
        vad.reset();

        assert!(vad.state.iter().all(|&v| v == 0.0), "State should be zeros");
        assert!(vad.accumulator.is_empty(), "Accumulator should be empty");
    }
}
