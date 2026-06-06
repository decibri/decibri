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
    }

    /// Run inference on a single window of exactly `window_size` samples.
    ///
    /// Inference goes through the internal [`OnnxSession`] trait
    /// (`Box<dyn OnnxSession>` field on `Self`) rather than calling
    /// `ort::Session` directly. The trait owns the marshaling between
    /// borrowed input slices and the backend's native tensor types.
    fn infer_window(&mut self, window: &[f32]) -> Result<f32, DecibriError> {
        let input_shape = [1i64, self.window_size as i64];
        let state_shape = [2i64, 1i64, 128i64];
        let sr_shape = [1i64];
        let sr_data = [self.sample_rate as i64];

        let outputs = self.session.run(OnnxInputs {
            items: &[
                (
                    "input",
                    OnnxTensorView {
                        shape: &input_shape,
                        data: OnnxTensorData::F32(window),
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
            OnnxTensorOwned::F32(v) => v[0],
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
            OnnxTensorOwned::F32(v) => self.state.copy_from_slice(v),
            OnnxTensorOwned::I64(_) => {
                return Err(DecibriError::OnnxBackendFailed {
                    backend: "ort",
                    source: "Silero output `stateN` must be f32".into(),
                });
            }
        }

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

    // ── Golden-output regression ───────────────────────────────────────
    //
    // `vad_golden_output_regression` anchors Silero VAD's exact per-window
    // speech probabilities for a fixed deterministic input. Its purpose is to
    // prove that a refactor of the inference path (for example relocating ORT
    // init or adding execution-provider selection) is behavior-identical: the
    // probabilities must not move. It is not a check of Silero's modeling
    // accuracy.
    //
    // On the fixture and what "spans low and high" means here. The fixture
    // (see `golden_fixture`) alternates silent windows with windows of formant-
    // synthesized voiced audio. The silent windows sit at Silero's
    // non-speech floor (~0.0005); the synthesized windows lift the probability
    // several times above that floor (peaks around 0.003), so the trajectory
    // spans a clear low band and a clear elevated band rather than a single
    // flat value, and the two bursts exercise the LSTM across silence-to-onset
    // transitions. The elevated band does not cross the 0.5 speech-decision
    // threshold: Silero v5 is robust against synthetic audio and, as verified
    // empirically while building this test across four distinct synthesis
    // methods (pure tones, additive formants, impulse-excited formants, and
    // the diphthong-with-breath-noise method used here), reserves high
    // confidence for natural speech. There is no committed natural-speech
    // sample in the repo to draw on. To anchor genuine above-threshold speech
    // frames, drop a small 16 kHz mono speech clip into `golden_fixture` and
    // regenerate; the bit-exact mechanism below is unchanged either way.
    //
    // The EXPECTED_GOLDEN values were generated from the pre-seam
    // implementation on this machine via the regeneration path below.
    // Comparison is bit-exact on the f32 bit patterns: same machine, same
    // ORT build, single inference thread (`with_intra_threads(1)` in the
    // session builder) is deterministic across reruns. The embedded literals
    // use Rust's shortest round-tripping float formatting, so parsing them
    // back yields the identical bits.
    //
    // To regenerate after a genuine, reviewed change to the model file, the
    // ORT version, or the fixture, set the environment variable so the test
    // prints a paste-ready array and then panics:
    //
    //   DECIBRI_REGEN_VAD_GOLDEN=1 cargo test-decibri \
    //       vad_golden_output_regression -- --nocapture
    //
    // Copy the printed EXPECTED_GOLDEN over the constant below, then rerun
    // without the variable to confirm green. Never edit the numbers by hand to
    // make a failing refactor pass: a mismatch means the inference path
    // changed behavior.

    /// Number of 512-sample windows (at 16 kHz) in the golden fixture.
    const GOLDEN_WINDOWS: usize = 80;
    /// Silero VAD window size at 16 kHz.
    const GOLDEN_WINDOW_SIZE: usize = 512;

    /// Per-window speech probabilities produced by the current inference path
    /// over [`golden_fixture`]. Regenerate via the documented env-var path; do
    /// not hand-edit. See the module comment above.
    const EXPECTED_GOLDEN: &[f32] = &[
        0.00059220195,
        0.0005399883,
        0.00053575635,
        0.0005354285,
        0.0005353987,
        0.0005353391,
        0.0005353689,
        0.0005353689,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.00051757693,
        0.00051772594,
        0.003079176,
        0.0005182624,
        0.00051757693,
        0.00051757693,
        0.00051766634,
        0.00051757693,
        0.00080385804,
        0.00053048134,
        0.0008057654,
        0.002694577,
        0.002518475,
        0.00051757693,
        0.0020724237,
        0.00051757693,
        0.00051763654,
        0.00051757693,
        0.00051757693,
        0.00051760674,
        0.0021745563,
        0.00051790476,
        0.00051757693,
        0.00051757693,
        0.0030798316,
        0.00051757693,
        0.00051757693,
        0.00051757693,
        0.00051757693,
        0.00051757693,
        0.00051757693,
        0.00051757693,
        0.0005351901,
        0.0005353391,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005353987,
        0.0005181432,
        0.0014440119,
        0.003076166,
        0.0028525293,
        0.00058060884,
        0.0006099045,
        0.0030797422,
        0.0030797422,
        0.0030797422,
        0.0008032918,
        0.0008033514,
        0.0030797422,
        0.0030797422,
        0.00051757693,
        0.00051757693,
        0.0005183816,
    ];

    /// Build the deterministic golden fixture: a 16 kHz, mono f32 waveform of
    /// `GOLDEN_WINDOWS * GOLDEN_WINDOW_SIZE` samples that alternates between
    /// digital silence and formant-synthesized voiced bursts (a glottal impulse
    /// train plus breath noise exciting three resonators that sweep along a
    /// diphthong trajectory, with a pitch glide and syllabic amplitude
    /// modulation). The silent windows sit at Silero's non-speech floor and the
    /// synthesized windows lift the probability several-fold above it, so the
    /// golden sequence spans a low band and a clearly elevated band rather than
    /// a single flat value (see the module comment for why the elevated band
    /// stays below the 0.5 speech threshold). Two separate bursts also exercise
    /// the LSTM state across silence-to-onset transitions. Pure function of the
    /// sample index: no I/O, and the only randomness is a fixed-seed LCG reset
    /// per burst, so it is identical on every rerun on this machine.
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
        for (i, (got, want)) in actual.iter().zip(EXPECTED_GOLDEN.iter()).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "golden mismatch at window {i}: got {got:?}, expected {want:?}. \
                 A mismatch means the inference path changed behavior; fix the \
                 code, not this number."
            );
        }

        // Guard the fixture itself: it must keep spanning a low silence band
        // and a clearly elevated synthesized-voiced band, so a future fixture
        // edit cannot quietly flatten the anchor into a single value. The
        // thresholds bracket the observed separation (floor ~0.0005, voiced
        // peaks ~0.003) with margin; see the module comment for why the
        // elevated band does not reach the 0.5 speech threshold.
        let lo = actual.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = actual.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            lo < 0.001,
            "golden fixture should include low-probability silence windows, min was {lo}"
        );
        assert!(
            hi > 0.002,
            "golden fixture should include clearly elevated voiced windows, max was {hi}"
        );
        assert!(
            hi > lo * 4.0,
            "golden fixture should span a meaningful probability range, got [{lo}, {hi}]"
        );
    }
}
