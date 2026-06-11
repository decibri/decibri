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
use std::path::PathBuf;

use crate::error::DecibriError;
#[cfg(feature = "vad")]
use crate::onnx::{
    OnnxInputs, OnnxSession, OnnxSessionBuilder, OnnxTensorData, OnnxTensorOwned, OnnxTensorView,
};

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
    /// ONNX session abstracted behind the internal [`OnnxSession`] trait. The
    /// ORT-backed implementation is the only backend.
    session: Box<dyn OnnxSession>,
    /// Combined LSTM state [2, 1, 128] = 256 floats.
    state: Vec<f32>,
    sample_rate: u32,
    threshold: f32,
    /// Carries leftover samples between process() calls.
    accumulator: Vec<f32>,
    /// 512 for 16kHz, 256 for 8kHz.
    window_size: usize,
    /// Silero v5 audio context: the last `context_size` samples of the previous
    /// window, prepended to each window before inference (zeros at start and
    /// after `reset`). The model is fed `context ++ window`, so the input is
    /// `context_size + window_size` samples (576 at 16 kHz). This is separate
    /// from the LSTM `state`; omitting it makes Silero score even loud speech
    /// at the non-speech floor.
    context: Vec<f32>,
    /// Number of context samples prepended per window: `window_size / 8`
    /// (64 at 16 kHz, 32 at 8 kHz).
    context_size: usize,
}

/// State size: 2 (hidden + cell) * 1 (batch) * 128 (hidden_dim) = 256.
/// State size: 2 (hidden + cell layers) × batch(1) × hidden_dim(128).
const STATE_SIZE: usize = 256;

#[cfg(feature = "vad")]
impl SileroVad {
    /// Create a new Silero VAD instance by loading the ONNX model.
    pub fn new(config: VadConfig) -> Result<Self, DecibriError> {
        // `validate` returns the `window_size` for the accepted sample rate,
        // so `SileroVad::new` does not need its own match, and cannot panic
        // if future `validate` extensions accept new sample rates.
        let window_size = config.validate()?;

        // Initialize ORT exactly once per process. Subsequent SileroVad
        // instances pass through immediately regardless of their ort_library_path.
        // ORT init lives in the capability-neutral `onnx` module so no single
        // model owns the process-global runtime state.
        crate::onnx::init_ort_once(config.ort_library_path.as_deref())?;

        // Session construction goes through the internal `OnnxSession` trait.
        // The ORT-backed implementation is the only backend. See
        // `crates/decibri/src/onnx.rs`.
        let session = OnnxSessionBuilder::from_file(config.model_path.clone())
            .with_intra_threads(1)
            .build()?;

        // Silero v5 context size is window_size / 8 (64 at 16 kHz, 32 at 8 kHz).
        let context_size = window_size / 8;

        Ok(Self {
            session,
            state: vec![0.0f32; STATE_SIZE],
            sample_rate: config.sample_rate,
            threshold: config.threshold,
            accumulator: Vec::new(),
            window_size,
            context: vec![0.0f32; context_size],
            context_size,
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
        // Detect fork()-after-ORT-init before touching
        // the inherited ONNX Runtime state. Single check at the outer
        // entry point covers every inference call; the inner
        // `infer_window` loop runs many sessions per `process` and does
        // not need its own check (any pid mismatch already failed here).
        // The fork guard lives in the `onnx` module alongside ORT init.
        crate::onnx::check_pid_for_ort()?;

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
        self.context.fill(0.0);
    }

    /// Run inference on a single window of exactly `window_size` samples.
    ///
    /// Inference goes through the internal [`OnnxSession`] trait
    /// (`Box<dyn OnnxSession>` field on `Self`) rather than calling
    /// `ort::Session` directly. The trait owns the marshaling between
    /// borrowed input slices and the backend's native tensor types.
    fn infer_window(&mut self, window: &[f32]) -> Result<f32, DecibriError> {
        // Silero v5 requires a `context_size`-sample audio context prepended to
        // each window: the model is fed `context ++ window` (576 samples at
        // 16 kHz). The context is the last `context_size` samples of the
        // previous window (zeros initially / after reset). Feeding only the bare
        // `window_size` samples makes the model score even loud speech at the
        // non-speech floor.
        let mut input_buf = Vec::with_capacity(self.context_size + self.window_size);
        input_buf.extend_from_slice(&self.context);
        input_buf.extend_from_slice(window);

        let input_shape = [1i64, (self.context_size + self.window_size) as i64];
        let state_shape = [2i64, 1i64, 128i64];
        let sr_shape = [1i64];
        let sr_data = [self.sample_rate as i64];

        let outputs = self.session.run(OnnxInputs {
            items: &[
                (
                    "input",
                    OnnxTensorView {
                        shape: &input_shape,
                        data: OnnxTensorData::F32(&input_buf),
                    },
                ),
                (
                    "state",
                    OnnxTensorView {
                        shape: &state_shape,
                        data: OnnxTensorData::F32(&self.state),
                    },
                ),
                (
                    "sr",
                    OnnxTensorView {
                        shape: &sr_shape,
                        data: OnnxTensorData::I64(&sr_data),
                    },
                ),
            ],
        })?;

        // Read outputs using actual model tensor names:
        //   output: f32[1, 1] (speech probability)
        //   stateN: f32[2, 1, 128] (updated state)
        let prob = outputs
            .get("output")
            .ok_or_else(|| DecibriError::OnnxBackendFailed {
                backend: "ort",
                source: "Silero output `output` missing from session run".into(),
            })?;
        let probability = match &prob.data {
            OnnxTensorOwned::F32(v) => {
                *v.first().ok_or_else(|| DecibriError::OnnxBackendFailed {
                    backend: "ort",
                    source: "Silero output `output` tensor is empty".into(),
                })?
            }
            OnnxTensorOwned::I64(_) => {
                return Err(DecibriError::OnnxBackendFailed {
                    backend: "ort",
                    source: "Silero output `output` must be f32".into(),
                });
            }
        };

        let state_n = outputs
            .get("stateN")
            .ok_or_else(|| DecibriError::OnnxBackendFailed {
                backend: "ort",
                source: "Silero output `stateN` missing from session run".into(),
            })?;
        match &state_n.data {
            OnnxTensorOwned::F32(v) => {
                if v.len() != self.state.len() {
                    return Err(DecibriError::OnnxBackendFailed {
                        backend: "ort",
                        source: format!(
                            "Silero output `stateN` has {} elements, expected {}",
                            v.len(),
                            self.state.len()
                        )
                        .into(),
                    });
                }
                self.state.copy_from_slice(v);
            }
            OnnxTensorOwned::I64(_) => {
                return Err(DecibriError::OnnxBackendFailed {
                    backend: "ort",
                    source: "Silero output `stateN` must be f32".into(),
                });
            }
        }

        // Carry the last context_size samples of this window forward as the
        // context for the next window (matches Silero v5's `OnnxWrapper`).
        self.context
            .copy_from_slice(&window[self.window_size - self.context_size..]);

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
    fn test_vad_config_validation() {
        let bad_rate = VadConfig {
            sample_rate: 44100,
            ..default_config()
        };
        assert!(SileroVad::new(bad_rate).is_err());
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

        // `SampleRateOutOfRange` is a unit variant (it applies to capture and
        // output configs, not VAD), so it is constructed without a payload.
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

    // ── Golden-output regression ───────────────────────────────────────
    //
    // `vad_golden_output_regression` anchors Silero VAD's exact per-window
    // speech probabilities for a fixed deterministic input. Its purpose is to
    // prove that a change to the inference path is behavior-identical, or, when
    // a change is intentional, to force a deliberate regeneration. The
    // probabilities must not move unless the model invocation genuinely changes.
    //
    // The fixture (see `golden_fixture`) alternates silent windows with windows
    // of formant-synthesized voiced audio (a diphthong with breath noise). With
    // the correct Silero v5 invocation the voiced bursts cross the 0.5 speech
    // threshold (peaks around 0.9) while the silent windows stay low (about
    // 0.002 to 0.06), so the trajectory spans both regimes and the two bursts
    // exercise the LSTM and the audio-context carry across silence-to-onset
    // transitions.
    //
    // These EXPECTED_GOLDEN values were regenerated when the Silero context bug
    // was fixed. Earlier the inference path fed only the bare 512-sample window,
    // omitting the 64-sample audio context Silero v5 requires, so every window
    // (even loud speech) scored at the non-speech floor (~0.0005) and nothing
    // crossed 0.5. The values now reflect the corrected `context ++ window`
    // invocation, where `infer_window` prepends the previous window's last 64
    // samples. The companion `vad_golden_output_regression_tts_speech` anchors
    // the same fix against generated (Kokoro TTS) speech.
    //
    // Comparison uses an absolute tolerance of 1e-4, not bit-exact equality.
    // ORT's CPU kernels are not bit-identical across platforms, so values
    // generated on one OS drift slightly on another: these were generated on
    // Windows, and Linux CI showed ~6e-8 at window 6. The tolerance is sized to
    // absorb that drift as it accumulates through the recurrent run (the LSTM
    // state plus the carried audio context) while staying orders of magnitude
    // below any real behavior change, where probabilities move by 0.01 or more.
    // The embedded literals use Rust's shortest round-tripping float formatting.
    //
    // To regenerate after a genuine, reviewed change to the model file, the ORT
    // version, the invocation, or the fixture, set the environment variable so
    // the test prints a paste-ready array and then panics:
    //
    //   DECIBRI_REGEN_VAD_GOLDEN=1 cargo test-decibri \
    //       vad_golden_output_regression -- --nocapture
    //
    // Copy the printed EXPECTED_GOLDEN over the constant below, then rerun
    // without the variable to confirm green. Never edit the numbers by hand to
    // make a failing change pass: a mismatch beyond the tolerance means the
    // inference path changed behavior.

    /// Number of 512-sample windows (at 16 kHz) in the golden fixture.
    const GOLDEN_WINDOWS: usize = 80;
    /// Silero VAD window size at 16 kHz.
    const GOLDEN_WINDOW_SIZE: usize = 512;

    /// Per-window speech probabilities produced by the current inference path
    /// over [`golden_fixture`]. Regenerate via the documented env-var path; do
    /// not hand-edit. See the module comment above.
    const EXPECTED_GOLDEN: &[f32] = &[
        0.0016697943,
        0.0068838596,
        0.008910686,
        0.007857174,
        0.005906582,
        0.0059607327,
        0.0058532655,
        0.0056402683,
        0.005431384,
        0.0051992834,
        0.005048305,
        0.0049027503,
        0.004735887,
        0.0045850873,
        0.0044474304,
        0.004327625,
        0.3420633,
        0.8894567,
        0.838452,
        0.84277743,
        0.88784015,
        0.87761295,
        0.87745064,
        0.8916172,
        0.89493644,
        0.7386559,
        0.8068055,
        0.78451455,
        0.83335185,
        0.84894687,
        0.9229615,
        0.931435,
        0.8888216,
        0.93177235,
        0.8556757,
        0.8152113,
        0.7775597,
        0.7691159,
        0.775824,
        0.7744537,
        0.8752998,
        0.74673194,
        0.6538298,
        0.6054683,
        0.6138661,
        0.6571155,
        0.57482207,
        0.44056144,
        0.06417981,
        0.025565833,
        0.015270472,
        0.012424737,
        0.012022942,
        0.011015505,
        0.0101495385,
        0.009793609,
        0.009830415,
        0.009944439,
        0.010142267,
        0.010163039,
        0.010095358,
        0.009979248,
        0.009832472,
        0.009658605,
        0.21928588,
        0.6697711,
        0.5521273,
        0.7865213,
        0.9442227,
        0.9290947,
        0.91997164,
        0.95520616,
        0.9074044,
        0.9155195,
        0.96092916,
        0.9731462,
        0.97090816,
        0.9644852,
        0.95524764,
        0.9106155,
    ];

    /// Build the deterministic golden fixture: a 16 kHz, mono f32 waveform of
    /// `GOLDEN_WINDOWS * GOLDEN_WINDOW_SIZE` samples that alternates between
    /// digital silence and formant-synthesized voiced bursts (a glottal impulse
    /// train plus breath noise exciting three resonators that sweep along a
    /// diphthong trajectory, with a pitch glide and syllabic amplitude
    /// modulation). With the corrected context-prepended invocation the silent
    /// windows stay low (about 0.002 to 0.06) and the voiced bursts cross the
    /// 0.5 speech threshold (peaks around 0.9), so the golden sequence spans
    /// both regimes. Two separate bursts also exercise the LSTM state and the
    /// audio-context carry across silence-to-onset transitions. Pure function of
    /// the sample index: no I/O, and the only randomness is a fixed-seed LCG
    /// reset per burst, so it is identical on every rerun on this machine.
    fn golden_fixture() -> Vec<f32> {
        use std::f64::consts::PI;

        let sr = 16_000.0_f64;
        let total = GOLDEN_WINDOWS * GOLDEN_WINDOW_SIZE;

        // Diphthong-like formant trajectory: each burst sweeps from an
        // /a/-vowel toward an /i/-vowel and back. Moving formants (especially
        // the F2 transition) plus breath noise are strong speech cues. Targets
        // are (center Hz, bandwidth Hz, linear gain) per formant.
        const VOWEL_A: [(f64, f64, f64); 3] = [
            (730.0, 90.0, 1.0),
            (1090.0, 110.0, 0.5),
            (2440.0, 170.0, 0.25),
        ];
        const VOWEL_I: [(f64, f64, f64); 3] = [
            (270.0, 80.0, 1.0),
            (2290.0, 110.0, 0.6),
            (3010.0, 180.0, 0.3),
        ];

        // Two voiced bursts (half-open window ranges); silent elsewhere. Each
        // burst is synthesized independently with fresh filter and glottal
        // state, exercising the LSTM across two silence-to-speech onsets.
        const BURSTS: [(usize, usize); 2] = [(16, 48), (64, 80)];

        let mut out = vec![0.0_f32; total];
        for (w0, w1) in BURSTS {
            let start = w0 * GOLDEN_WINDOW_SIZE;
            let end = w1 * GOLDEN_WINDOW_SIZE;
            let len = end - start;

            // Formant synthesis: a glottal impulse train (pitch gliding
            // 115 -> 150 Hz) plus breath noise, exciting three parallel
            // 2nd-order resonators whose centers sweep along the diphthong
            // trajectory. The periodic-pulse-plus-moving-formant structure is
            // what real voiced speech looks like, which Silero scores far
            // higher than steady additive tones. Synthesize into f64, then
            // peak-normalize so the level is strong and independent of
            // resonator gain.
            let mut raw = vec![0.0_f64; len];
            let mut z1 = [0.0_f64; 3];
            let mut z2 = [0.0_f64; 3];
            let mut phase = 0.0_f64;
            // Deterministic LCG for breath noise (fixed seed, reset per burst).
            let mut seed: u64 = 0x243F_6A88_85A3_08D3;
            for (i, slot) in raw.iter_mut().enumerate() {
                let t = i as f64 / sr;
                let frac = i as f64 / len as f64;
                // Triangle 0 -> 1 -> 0 across the burst drives the diphthong.
                let sweep = 1.0 - (2.0 * frac - 1.0).abs();

                let f0 = 115.0 + 35.0 * t; // pitch glide, Hz
                phase += f0 / sr;
                let pulse = if phase >= 1.0 {
                    phase -= 1.0;
                    1.0
                } else {
                    0.0
                };
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((seed >> 40) as f64 / (1u64 << 24) as f64) * 2.0 - 1.0;
                let exc = pulse + 0.4 * noise;

                let mut s = 0.0_f64;
                for fi in 0..3 {
                    let (fa, ba, ga) = VOWEL_A[fi];
                    let (fii, bii, gii) = VOWEL_I[fi];
                    let fc = fa + (fii - fa) * sweep;
                    let bw = ba + (bii - ba) * sweep;
                    let g = ga + (gii - ga) * sweep;
                    let r = (-PI * bw / sr).exp();
                    let a1 = 2.0 * r * (2.0 * PI * fc / sr).cos();
                    let a2 = -(r * r);
                    let y = exc + a1 * z1[fi] + a2 * z2[fi];
                    z2[fi] = z1[fi];
                    z1[fi] = y;
                    s += g * y;
                }
                // Syllabic amplitude modulation at ~4 Hz; never fully silent.
                let env = 0.55 + 0.45 * (2.0 * PI * 4.0 * t).sin().max(0.0);
                *slot = env * s;
            }

            let peak = raw.iter().fold(0.0_f64, |m, &v| m.max(v.abs())).max(1e-9);
            let scale = 0.8 / peak;
            for (i, &v) in raw.iter().enumerate() {
                out[start + i] = (v * scale).clamp(-1.0, 1.0) as f32;
            }
        }
        out
    }

    #[test]
    fn vad_golden_output_regression() {
        let fixture = golden_fixture();
        assert_eq!(
            fixture.len(),
            GOLDEN_WINDOWS * GOLDEN_WINDOW_SIZE,
            "fixture length must be an exact multiple of the window size"
        );

        let mut vad = SileroVad::new(default_config()).unwrap();

        // Feed one 512-sample window per `process` call so each call yields
        // exactly that window's probability (the accumulator starts empty and
        // is fully drained each call). This is the public API path a user
        // takes; it pins the entire per-window probability trajectory,
        // including LSTM state evolution across the two bursts.
        let mut actual: Vec<f32> = Vec::with_capacity(GOLDEN_WINDOWS);
        for window in fixture.chunks_exact(GOLDEN_WINDOW_SIZE) {
            let result = vad.process(window).unwrap();
            actual.push(result.probability);
        }
        assert_eq!(actual.len(), GOLDEN_WINDOWS);

        // Regeneration path: print a paste-ready array, then panic so it can
        // never silently pass without asserting. See the module comment above.
        if std::env::var("DECIBRI_REGEN_VAD_GOLDEN").is_ok() {
            let mut body = String::new();
            for (i, p) in actual.iter().enumerate() {
                if i % 14 == 0 {
                    body.push_str("\n        ");
                }
                body.push_str(&format!("{p:?}, "));
            }
            println!("    const EXPECTED_GOLDEN: &[f32] = &[{body}\n    ];");
            panic!(
                "DECIBRI_REGEN_VAD_GOLDEN is set: copy the printed EXPECTED_GOLDEN \
                 above into vad.rs, then rerun without the variable to confirm green."
            );
        }

        assert_eq!(
            actual.len(),
            EXPECTED_GOLDEN.len(),
            "golden window count changed: regenerate EXPECTED_GOLDEN (see module comment)"
        );
        // Absolute tolerance (not bit-exact): ORT CPU kernels drift slightly
        // across platforms. See the module comment for why 1e-4 is the bound.
        for (i, (got, want)) in actual.iter().zip(EXPECTED_GOLDEN.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4,
                "golden mismatch at window {i}: got {got:?}, expected {want:?}. \
                 A mismatch beyond tolerance means the inference path changed \
                 behavior; fix the code, not this number."
            );
        }

        // Guard the fixture itself: with the corrected invocation it must keep
        // spanning both regimes (a low silence band plus voiced bursts that
        // cross the 0.5 speech threshold), so a future fixture edit or an
        // invocation regression cannot quietly flatten the anchor. The
        // thresholds bracket the observed values (silence floor ~0.002, voiced
        // peaks ~0.9, 45 of 80 windows above 0.5) with margin.
        let lo = actual.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = actual.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let above = actual.iter().filter(|&&p| p >= 0.5).count();
        assert!(
            lo < 0.1,
            "golden fixture should include low-probability silence windows, min was {lo}"
        );
        assert!(
            hi > 0.5,
            "golden fixture voiced bursts should cross the speech threshold, max was {hi}"
        );
        assert!(
            above >= 30,
            "golden fixture should have many speech-positive windows (>= 0.5), got {above}"
        );
    }

    // ── TTS-speech golden-output regression ─────────────────────────────
    //
    // `vad_golden_output_regression_tts_speech` is the permanent efficacy
    // regression for the Silero context fix. The synthetic golden test above
    // uses formant synthesis; this one uses real generated speech, so it proves
    // the VAD detects natural speech, not just speech-like synthesis.
    //
    // The fixture (`tests/assets/vad-golden-tts-speech-16k.wav`, 6.66 s) is
    // Kokoro TTS speech, NOT a human recording: deliberately so, so that no
    // human-voice biometrics enter the repo. It was generated locally with
    // sherpa-onnx, the `kokoro-en-v0_19` model, voice "af" (sid 0, speed 1.0),
    // speaking "The quick brown fox jumps over the lazy dog." then "Decibri
    // detects speech.", assembled as lead-in silence, the sentence, a ~1 s
    // pause, the short phrase, and trailing silence, then resampled to 16 kHz
    // mono 16-bit and trimmed to a 512-sample multiple. With the corrected
    // context-prepended invocation, 123 of its 208 windows score above the 0.5
    // speech threshold and the silent windows stay near the floor. Before the
    // context fix every window (even loud speech) scored ~0.0005 and none
    // crossed 0.5, so this test could not have been satisfied at all: that is
    // exactly why it now guards the fix.
    //
    // Comparison uses an absolute tolerance of 1e-4, not bit-exact equality,
    // for the same reason as the synthetic test: ORT CPU kernels drift slightly
    // across platforms (these values were generated on Windows; Linux CI showed
    // ~6e-8 at window 6), and the tolerance absorbs that drift accumulating
    // through the recurrent run while staying far below any real behavior change.
    // To regenerate after a reviewed change to the model, the ORT version, the
    // invocation, or the fixture, set the environment variable so the test
    // prints a paste-ready array and panics:
    //
    //   DECIBRI_REGEN_VAD_GOLDEN_TTS=1 cargo test-decibri \
    //       vad_golden_output_regression_tts_speech -- --nocapture
    //
    // Copy the printed EXPECTED_GOLDEN_TTS over the constant below, then rerun
    // without the variable to confirm green. Never hand-edit the numbers to make
    // a failing change pass.

    /// Per-window speech probabilities over the TTS-speech fixture. Regenerate
    /// via the documented env-var path; do not hand-edit. See the comment above.
    const EXPECTED_GOLDEN_TTS: &[f32] = &[
        0.0016697943,
        0.0068838596,
        0.008910686,
        0.007857174,
        0.005906582,
        0.0059607327,
        0.0058532655,
        0.0056402683,
        0.005431384,
        0.0051992834,
        0.005048305,
        0.0049027503,
        0.004735887,
        0.0045850873,
        0.0044474304,
        0.004327625,
        0.00422737,
        0.0041455925,
        0.0040753484,
        0.0040133893,
        0.003958285,
        0.00390923,
        0.0038656592,
        0.0038268864,
        0.0037921667,
        0.003760904,
        0.0037325919,
        0.003706932,
        0.0036835968,
        0.0036624074,
        0.0036430657,
        0.0036253333,
        0.0036090612,
        0.0035939813,
        0.059919864,
        0.80586326,
        0.94249177,
        0.97746277,
        0.99230456,
        0.99498165,
        0.99854183,
        0.99970543,
        0.9992554,
        0.99896127,
        0.9885851,
        0.76601565,
        0.99835265,
        0.99989593,
        0.9996809,
        0.99982965,
        0.99980825,
        0.9996754,
        0.9994799,
        0.9999109,
        0.99987936,
        0.9999951,
        0.9999695,
        0.9999645,
        0.9998757,
        0.9999072,
        0.9999266,
        0.99991095,
        0.9998672,
        0.9999199,
        0.9999012,
        0.99972665,
        0.99773854,
        0.99628925,
        0.9985783,
        0.99857473,
        0.9970066,
        0.9971564,
        0.98947036,
        0.9841496,
        0.9995967,
        0.9999602,
        0.99999416,
        0.9999776,
        0.9999503,
        0.9999956,
        0.99997866,
        0.99998665,
        0.9999939,
        0.99999136,
        0.99999785,
        0.99999464,
        0.99999785,
        0.9999994,
        0.99999774,
        0.99999654,
        0.9999993,
        0.99999577,
        0.9999988,
        0.99999905,
        0.99999845,
        0.9999981,
        0.99999815,
        0.9999993,
        0.99999404,
        0.9999974,
        0.99999905,
        0.999999,
        0.99998784,
        0.9999964,
        0.99999744,
        0.99999523,
        0.99999464,
        0.9999922,
        0.99998105,
        0.99993587,
        0.99971354,
        0.9992842,
        0.9995015,
        0.99989575,
        0.99946034,
        0.97081566,
        0.31423837,
        0.043908834,
        0.01832974,
        0.0073020756,
        0.0073402524,
        0.0073531866,
        0.007080108,
        0.0065951943,
        0.0059630275,
        0.0054104924,
        0.004972309,
        0.0046182275,
        0.004334569,
        0.004111409,
        0.0039377213,
        0.003803432,
        0.0037005246,
        0.0036236346,
        0.003565967,
        0.0035216212,
        0.0034860075,
        0.003455013,
        0.0034240484,
        0.0033897161,
        0.003353119,
        0.0033128262,
        0.003267914,
        0.0032192767,
        0.0031673908,
        0.0031124055,
        0.0030547678,
        0.0029948652,
        0.0029331148,
        0.002843678,
        0.0027451217,
        0.0026483536,
        0.0025538504,
        0.05372742,
        0.9938383,
        0.99997073,
        0.99992335,
        0.9994338,
        0.99971783,
        0.9999653,
        0.9999837,
        0.9999424,
        0.9999435,
        0.9999926,
        0.99999577,
        0.9999976,
        0.99999917,
        0.99999785,
        0.99999744,
        0.9999982,
        0.9999572,
        0.9999954,
        0.99999386,
        0.9999978,
        0.9999988,
        0.99996626,
        0.9999441,
        0.9999922,
        0.99998045,
        0.99983156,
        0.9996438,
        0.99954045,
        0.9996016,
        0.99833137,
        0.99988496,
        0.9999797,
        0.9999845,
        0.99999213,
        0.9999089,
        0.9999163,
        0.9999642,
        0.99996233,
        0.99988997,
        0.9998535,
        0.99880433,
        0.66095096,
        0.08390489,
        0.03502187,
        0.017826378,
        0.021352828,
        0.0225479,
        0.021549493,
        0.02064374,
        0.019463807,
        0.018365651,
        0.017573029,
        0.017026126,
        0.016649067,
    ];

    /// Minimal WAV reader for the test fixture: 16 kHz mono 16-bit PCM only.
    /// Test-local (no runtime dependency added to the crate). Returns interleaved
    /// f32 samples in [-1.0, 1.0].
    fn read_golden_wav(path: &std::path::Path) -> Vec<f32> {
        let bytes = std::fs::read(path).expect("read fixture wav");
        assert_eq!(&bytes[0..4], b"RIFF", "fixture must be RIFF/WAVE");
        assert_eq!(&bytes[8..12], b"WAVE", "fixture must be RIFF/WAVE");
        assert_eq!(
            u16::from_le_bytes([bytes[20], bytes[21]]),
            1,
            "fixture must be PCM"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[22], bytes[23]]),
            1,
            "fixture must be mono"
        );
        assert_eq!(
            u32::from_le_bytes([bytes[24], bytes[25], bytes[26], bytes[27]]),
            16000,
            "fixture must be 16 kHz"
        );
        assert_eq!(
            u16::from_le_bytes([bytes[34], bytes[35]]),
            16,
            "fixture must be 16-bit"
        );

        let (mut o, mut data_off, mut data_len) = (12usize, 44usize, bytes.len() - 44);
        while o + 8 <= bytes.len() {
            let id = &bytes[o..o + 4];
            let sz = u32::from_le_bytes([bytes[o + 4], bytes[o + 5], bytes[o + 6], bytes[o + 7]])
                as usize;
            if id == b"data" {
                data_off = o + 8;
                data_len = sz;
                break;
            }
            o += 8 + sz + (sz & 1);
        }
        let mut out = Vec::with_capacity(data_len / 2);
        let mut i = data_off;
        while i + 1 < data_off + data_len {
            out.push(i16::from_le_bytes([bytes[i], bytes[i + 1]]) as f32 / 32768.0);
            i += 2;
        }
        out
    }

    #[test]
    fn vad_golden_output_regression_tts_speech() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("assets")
            .join("vad-golden-tts-speech-16k.wav");
        let samples = read_golden_wav(&path);

        let mut vad = SileroVad::new(default_config()).unwrap();
        let mut actual: Vec<f32> = Vec::new();
        for window in samples.chunks_exact(GOLDEN_WINDOW_SIZE) {
            actual.push(vad.process(window).unwrap().probability);
        }

        // Regeneration path: print a paste-ready array, then panic so it can
        // never silently pass without asserting. See the comment above.
        if std::env::var("DECIBRI_REGEN_VAD_GOLDEN_TTS").is_ok() {
            let mut body = String::new();
            for (i, p) in actual.iter().enumerate() {
                if i % 14 == 0 {
                    body.push_str("\n        ");
                }
                body.push_str(&format!("{p:?}, "));
            }
            println!("    const EXPECTED_GOLDEN_TTS: &[f32] = &[{body}\n    ];");
            panic!(
                "DECIBRI_REGEN_VAD_GOLDEN_TTS is set: copy the printed EXPECTED_GOLDEN_TTS \
                 above into vad.rs, then rerun without the variable to confirm green."
            );
        }

        assert_eq!(
            actual.len(),
            EXPECTED_GOLDEN_TTS.len(),
            "TTS-speech window count changed: regenerate EXPECTED_GOLDEN_TTS (see comment)"
        );
        // Absolute tolerance (not bit-exact): ORT CPU kernels drift slightly
        // across platforms. See the comment above for why 1e-4 is the bound.
        for (i, (got, want)) in actual.iter().zip(EXPECTED_GOLDEN_TTS.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-4,
                "TTS-speech golden mismatch at window {i}: got {got:?}, expected {want:?}. \
                 A mismatch beyond tolerance means the inference path changed behavior; fix \
                 the code, not this number."
            );
        }

        // Efficacy assertions: this fixture is real (TTS-generated) speech, so
        // the corrected VAD must detect it. Many windows above the 0.5 speech
        // threshold plus a clear silence floor. Observed: 123 of 208 windows
        // above 0.5, min ~0.0017. Before the context fix this was 0 above 0.5.
        let above = actual.iter().filter(|&&p| p >= 0.5).count();
        let lo = actual.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(
            above >= 90,
            "TTS speech should produce many above-0.5 windows once the context fix \
             is in place; got {above}"
        );
        assert!(
            lo < 0.05,
            "TTS-speech fixture should include a clear silence floor; min was {lo}"
        );
    }
}
