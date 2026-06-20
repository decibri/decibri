//! Single-channel speech enhancement (denoise) for the capture path.
//!
//! Wraps the bundled FastEnhancer-T ONNX model as a capture [`Stage`]. The model
//! is the waveform-in / waveform-out variant: it bakes the complete spectral path
//! (windowing, the forward and inverse DFT, power compression, and overlap-add)
//! into its graph, so this stage owns no spectral DSP. The host's job is purely
//! to re-block the captured waveform into the model's analysis frames, carry the
//! three streaming caches across calls, and concatenate the returned hops.
//!
//! # Streaming contract (verified against the model and an offline reference)
//!
//! The model takes one analysis window `wav_in[1, 512]` and three caches
//! (`cache_in_0[1, 256]` the overlap-add tail, `cache_in_1`/`cache_in_2`
//! `[1, 16, 20]` the recurrent state), and returns one hop `wav_out[1, 256]` plus
//! the three updated caches. Frames advance by the hop (256), so consecutive
//! windows overlap by 512 - 256 = 256 samples, and the caches must be fed back
//! each call. The stream is left-padded by one (window - hop) = 256-sample
//! half-window of zeros (the [`accumulator`](Denoise::accumulator) is seeded with
//! 256 zeros) so the first analysis window is `[zeros(256), samples(256)]`, which
//! matches how the upstream streaming export initialises its internal buffer. The
//! model therefore introduces 256 samples of algorithmic latency (the output hop
//! lags its input by window - hop); the capture chain accounts for that latency
//! as a tap lead, not here.
//!
//! Driving the model this way was validated end to end: streamed over real speech
//! it reconstructs the clean signal at ~0.96 correlation with exactly 256 samples
//! of latency, the cache feedback is required (correlation collapses without it),
//! and a chunked accumulator reproduces an offline whole-signal reference bit for
//! bit.

#![cfg(feature = "capture")]

use std::path::Path;

use crate::error::DecibriError;
use crate::microphone::DenoiseModel;
use crate::onnx::{
    OnnxInputs, OnnxOutputs, OnnxSession, OnnxSessionBuilder, OnnxTensorData, OnnxTensorOwned,
    OnnxTensorView,
};
use crate::stage::Stage;

/// Analysis window length the model consumes per call (`wav_in[1, 512]`).
const WINDOW: usize = 512;
/// Hop the window advances by, and the length of one output frame
/// (`wav_out[1, 256]`). The window overlaps the previous one by `WINDOW - HOP`.
const HOP: usize = 256;
/// Length of the overlap-add cache `cache_*_0[1, 256]`.
const CACHE0_LEN: usize = 256;
/// Length of each recurrent cache `cache_*_1`/`cache_*_2[1, 16, 20]`, flattened.
const CACHE_REC_LEN: usize = 16 * 20;

/// FastEnhancer-T denoise stage: a framed, stateful capture [`Stage`] that maps a
/// stream of noisy speech samples to enhanced speech samples.
///
/// Length-changing and latency-introducing: each [`process`](Stage::process)
/// emits whole 256-sample hops as enough input accumulates, buffering the
/// remainder, so its output count trails its input by the framing latency until
/// [`flush`](Stage::flush) drains the tail at close. It runs in the `transform`
/// segment after the VAD tap, so that latency surfaces as a bounded tap lead and
/// never reaches the detector.
pub(crate) struct Denoise {
    /// The model behind the shared ONNX session seam. Loaded from a caller-
    /// supplied path; the crate embeds no model bytes.
    session: Box<dyn OnnxSession>,
    /// Pending samples not yet consumed by a frame. Seeded with `WINDOW - HOP`
    /// zeros (the left-pad half-window); each frame consumes `HOP` from the front
    /// and retains the `WINDOW - HOP`-sample overlap for the next window.
    accumulator: Vec<f32>,
    /// The model's overlap-add tail buffer (`cache_*_0`), carried across calls.
    cache0: Vec<f32>,
    /// The model's first recurrent cache (`cache_*_1`), carried across calls.
    cache1: Vec<f32>,
    /// The model's second recurrent cache (`cache_*_2`), carried across calls.
    cache2: Vec<f32>,
}

impl Denoise {
    /// Build the stage, loading the model from `path` through the shared ONNX
    /// session seam (the same path Silero VAD loads through). Initialises ORT
    /// once per process, like [`crate::vad::SileroVad::new`], so a denoise-only
    /// build still works without a VAD.
    ///
    /// # Errors
    /// - [`DecibriError::ModelLoadFailed`] if the model file cannot be opened.
    /// - The ORT init / session-build errors surfaced by the shared seam.
    pub(crate) fn new(
        model: DenoiseModel,
        path: &Path,
        ort_library_path: Option<&Path>,
    ) -> Result<Self, DecibriError> {
        // ORT initialises exactly once per process (first-wins). A VAD in the
        // same process may have done it already; if not, denoise does it here,
        // from `ort_library_path` when the caller supplied one (the bindings pass
        // their bundled dylib path so a denoise-only capture finds ORT without a
        // VAD or an `ORT_DYLIB_PATH` env var). `None` leaves ORT to its own
        // discovery. Under `ort-download-binaries` the path is ignored.
        crate::onnx::init_ort_once(ort_library_path)?;

        let session = match model {
            // One model today; the exhaustive match keeps a future tier from
            // silently sharing this loader without declaring it.
            DenoiseModel::FastEnhancerT => OnnxSessionBuilder::from_file(path)
                .with_intra_threads(1)
                .build()
                .map_err(relabel_load_error)?,
        };

        Ok(Self {
            session,
            // Left-pad the stream by one half-window of zeros so the first
            // analysis window is `[zeros(256), first 256 samples]`.
            accumulator: vec![0.0; WINDOW - HOP],
            cache0: vec![0.0; CACHE0_LEN],
            cache1: vec![0.0; CACHE_REC_LEN],
            cache2: vec![0.0; CACHE_REC_LEN],
        })
    }

    /// Run one analysis window through the model, appending the enhanced hop to
    /// `out` and advancing the three caches. Reads the front `WINDOW` samples of
    /// the accumulator; the caller drains `HOP` afterward.
    fn run_frame(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // Copy the window out so the accumulator can be drained afterward and so
        // the borrow of `self.accumulator` does not overlap the `&mut self.session`
        // call (mirrors SileroVad::infer_window building its own input buffer).
        let window: Vec<f32> = self.accumulator[..WINDOW].to_vec();
        let wav_in_shape = [1i64, WINDOW as i64];
        let cache0_shape = [1i64, CACHE0_LEN as i64];
        let cache_rec_shape = [1i64, 16, 20];

        let outputs = self.session.run(OnnxInputs {
            items: &[
                (
                    "wav_in",
                    OnnxTensorView {
                        shape: &wav_in_shape,
                        data: OnnxTensorData::F32(&window),
                    },
                ),
                (
                    "cache_in_0",
                    OnnxTensorView {
                        shape: &cache0_shape,
                        data: OnnxTensorData::F32(&self.cache0),
                    },
                ),
                (
                    "cache_in_1",
                    OnnxTensorView {
                        shape: &cache_rec_shape,
                        data: OnnxTensorData::F32(&self.cache1),
                    },
                ),
                (
                    "cache_in_2",
                    OnnxTensorView {
                        shape: &cache_rec_shape,
                        data: OnnxTensorData::F32(&self.cache2),
                    },
                ),
            ],
        })?;

        let wav_out = f32_output(&outputs, "wav_out")?;
        if wav_out.len() != HOP {
            return Err(length_error("wav_out", HOP, wav_out.len()));
        }
        out.extend_from_slice(wav_out);

        copy_cache(&outputs, "cache_out_0", &mut self.cache0)?;
        copy_cache(&outputs, "cache_out_1", &mut self.cache1)?;
        copy_cache(&outputs, "cache_out_2", &mut self.cache2)?;
        Ok(())
    }

    /// Emit every whole frame the accumulator currently holds, advancing by `HOP`
    /// each time and retaining the `WINDOW - HOP` overlap for the next window.
    fn drain_ready(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        while self.accumulator.len() >= WINDOW {
            self.run_frame(out)?;
            self.accumulator.drain(..HOP);
        }
        Ok(())
    }
}

impl Stage for Denoise {
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), DecibriError> {
        self.accumulator.extend_from_slice(input);
        self.drain_ready(out)
    }

    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // Pad the stream out by one full window of zeros (the upstream offline
        // recipe right-pads by n_fft) so the leftover real samples and the
        // model's overlap-add tail are pushed out as final hops, then drain.
        let padded_len = self.accumulator.len() + WINDOW;
        self.accumulator.resize(padded_len, 0.0);
        self.drain_ready(out)?;

        // Leave the stage reusable: re-seed the left-pad half-window and zero the
        // streaming caches, so a subsequent stream starts clean.
        self.accumulator.clear();
        self.accumulator.resize(WINDOW - HOP, 0.0);
        self.cache0.fill(0.0);
        self.cache1.fill(0.0);
        self.cache2.fill(0.0);
        Ok(())
    }
}

/// `OrtSession::open` labels every model-file load failure `VadModelLoadFailed`
/// (the shared seam predates denoise). Relabel that one variant to the model-
/// agnostic [`DecibriError::ModelLoadFailed`] so a denoise load failure is not
/// reported as a VAD error; pass every other build error through unchanged.
fn relabel_load_error(e: DecibriError) -> DecibriError {
    match e {
        DecibriError::VadModelLoadFailed { path, source } => {
            DecibriError::ModelLoadFailed { path, source }
        }
        other => other,
    }
}

/// Borrow a named f32 output tensor's data, or a typed error if it is missing or
/// not f32.
fn f32_output<'a>(outputs: &'a OnnxOutputs, name: &'static str) -> Result<&'a [f32], DecibriError> {
    let tensor = outputs
        .get(name)
        .ok_or_else(|| DecibriError::OnnxBackendFailed {
            backend: "ort",
            source: format!("FastEnhancer output `{name}` missing from session run").into(),
        })?;
    match &tensor.data {
        OnnxTensorOwned::F32(v) => Ok(v.as_slice()),
        OnnxTensorOwned::I64(_) => Err(DecibriError::OnnxBackendFailed {
            backend: "ort",
            source: format!("FastEnhancer output `{name}` must be f32").into(),
        }),
    }
}

/// Copy a named cache output into `dst`, checking the length matches the cache
/// it feeds back (a malformed model output surfaces a typed error, not a panic).
fn copy_cache(
    outputs: &OnnxOutputs,
    name: &'static str,
    dst: &mut [f32],
) -> Result<(), DecibriError> {
    let v = f32_output(outputs, name)?;
    if v.len() != dst.len() {
        return Err(length_error(name, dst.len(), v.len()));
    }
    dst.copy_from_slice(v);
    Ok(())
}

/// Typed error for a model output whose length is not what the streaming
/// contract requires.
fn length_error(name: &str, expected: usize, got: usize) -> DecibriError {
    DecibriError::OnnxBackendFailed {
        backend: "ort",
        source: format!("FastEnhancer output `{name}` has {got} elements, expected {expected}")
            .into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn model_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("models")
            .join("fastenhancer_t.onnx")
    }

    /// A short noisy speech-like signal: a voiced tone plus broadband noise.
    fn noisy(n: usize) -> Vec<f32> {
        let mut seed: u64 = 0x1234_5678_9abc_def0;
        (0..n)
            .map(|i| {
                seed = seed
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                let noise = ((seed >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0;
                let t = i as f32 / 16_000.0;
                0.3 * (2.0 * std::f32::consts::PI * 180.0 * t).sin() + 0.05 * noise
            })
            .collect()
    }

    /// The stage loads the bundled model, re-blocks a stream into 256-sample
    /// hops, advances the caches, and emits finite enhanced audio whose count
    /// tracks the input minus the framing latency.
    #[test]
    fn denoise_stage_processes_and_advances_caches() {
        let mut stage = Denoise::new(DenoiseModel::FastEnhancerT, &model_path(), None)
            .expect("bundled FastEnhancer-T model loads");

        // Caches start zeroed and the accumulator carries the left-pad.
        assert_eq!(stage.accumulator.len(), WINDOW - HOP);
        assert!(stage.cache0.iter().all(|&v| v == 0.0));

        let input = noisy(4096);
        let mut out = Vec::new();
        stage.process(&input, &mut out).expect("process runs");

        // Output is whole hops of finite enhanced audio.
        assert!(!out.is_empty(), "enough input forms at least one hop");
        assert_eq!(out.len() % HOP, 0, "output is whole 256-sample hops");
        assert!(
            out.iter().all(|s| s.is_finite()),
            "enhanced output is finite"
        );
        // Latency: the output trails the input by the framing latency, so it is
        // shorter than the input before flush, but within one window of it.
        assert!(out.len() <= input.len());
        assert!(input.len() - out.len() <= WINDOW);

        // The caches advanced (the model wrote streaming state back).
        assert!(
            stage.cache0.iter().any(|&v| v != 0.0),
            "the overlap-add cache advanced"
        );
        assert!(
            stage.cache1.iter().any(|&v| v != 0.0) || stage.cache2.iter().any(|&v| v != 0.0),
            "a recurrent cache advanced"
        );
    }

    /// `flush` drains the model's latency tail: after processing, flushing emits
    /// further finite samples and leaves the stage reusable (caches re-zeroed,
    /// the left-pad re-seeded).
    #[test]
    fn denoise_flush_drains_tail_and_resets() {
        let mut stage = Denoise::new(DenoiseModel::FastEnhancerT, &model_path(), None)
            .expect("bundled FastEnhancer-T model loads");

        let mut out = Vec::new();
        stage.process(&noisy(4096), &mut out).expect("process runs");
        let processed = out.len();

        let mut tail = Vec::new();
        stage.flush(&mut tail).expect("flush runs");
        assert!(!tail.is_empty(), "flush drains the latency tail");
        assert!(tail.iter().all(|s| s.is_finite()), "flushed tail is finite");

        // The full delivered count (process + flush) recovers the input plus the
        // model's one-hop warm-up latency, so no captured audio is dropped.
        assert!(
            processed + tail.len() >= 4096,
            "process + flush deliver at least the whole input"
        );

        // Reusable: caches re-zeroed, accumulator re-seeded with the left-pad.
        assert!(stage.cache0.iter().all(|&v| v == 0.0), "caches re-zeroed");
        assert_eq!(stage.accumulator.len(), WINDOW - HOP, "left-pad re-seeded");
    }

    /// A nonexistent model path surfaces the model-agnostic `ModelLoadFailed`,
    /// not the VAD-named `VadModelLoadFailed`. Gated to the download-binaries
    /// build (the `test-decibri` alias): under `ort-load-dynamic` a load attempt
    /// against a missing ORT dylib takes a different, ORT-init failure path.
    #[cfg(feature = "ort-download-binaries")]
    #[test]
    fn denoise_missing_model_is_model_load_failed() {
        let missing = Path::new(env!("CARGO_MANIFEST_DIR")).join("no-such-model.onnx");
        // Match on the Result (not `expect_err`) because `Denoise` holds a
        // `Box<dyn OnnxSession>` and is not `Debug`.
        let err = match Denoise::new(DenoiseModel::FastEnhancerT, &missing, None) {
            Ok(_) => panic!("a missing model file must fail to load"),
            Err(e) => e,
        };
        assert!(
            matches!(err, DecibriError::ModelLoadFailed { .. }),
            "denoise surfaces ModelLoadFailed, not VadModelLoadFailed: {err:?}"
        );
    }

    // ── Denoise improvement regression over real speech with synthesized noise ──
    //
    // `denoise_improves_noisy_speech_regression` is the permanent regression that
    // proves the shipped denoise stage measurably cleans up noisy speech. It runs
    // known-noisy speech through the real stage (the bundled FastEnhancer-T model,
    // the streaming convention, real ONNX Runtime) and asserts the enhanced output
    // improves three objective measures over the noisy input by a documented
    // margin, so a later change to the stage, the model, or the frame convention
    // that silently weakens enhancement is caught.
    //
    // Clean reference: the committed 16 kHz mono speech fixture
    // `tests/assets/vad-golden-tts-speech-16k.wav`, the same Kokoro TTS clip the
    // VAD regression uses, so no new fixture and no human-voice biometrics enter
    // the repo. Noise is built deterministically in pure Rust (no committed noise
    // files, no RNG nondeterminism, no network): white and pink stationary noise,
    // a non-stationary colored noise as an out-of-distribution stand-in for real
    // environments, and multi-talker babble built by summing shifted copies of the
    // speech itself (real speech as the interferers). Each noise is mixed at fixed
    // input SNRs of 0, 5, and 10 dB; a clean (no-noise) case is included.
    //
    // Three measures, all pure Rust and deterministic, each computed for the noisy
    // input and the enhanced output against the clean reference, the improvement
    // being (enhanced minus noisy):
    //   - SI-SDR, the scale-invariant signal-to-distortion ratio, in dB.
    //   - Segmental SNR, a per-frame SNR clamped and averaged over active frames.
    //   - Frequency-weighted segmental SNR, a perceptual proxy, NOT ITU PESQ.
    // The stage delays its output by one hop (256 samples); the comparison drops
    // that lead so clean and enhanced line up.
    //
    // Floors sit a margin below the minimum this build measures on its own
    // fixtures, per noise category, so the test is a regression backstop and does
    // not flake on numerical noise: stationary and out-of-distribution noise must
    // clear a real improvement floor; babble (competing speech, which a
    // single-channel model cannot separate, a physical limit not a regression)
    // only has to do no harm; the clean case must not be badly damaged (a bounded
    // floor on its absolute score, since the model is not transparent on
    // already-clean input). The measured minimum and the chosen floor are recorded
    // next to each constant below.
    //
    // To re-measure after a reviewed change to the model, the ORT version, the
    // stage, or the noise construction, set the environment variable to print the
    // per-case table and the per-category minimums, then update the floors:
    //
    //   DECIBRI_PRINT_DENOISE_METRICS=1 cargo test-decibri \
    //       denoise_improves_noisy_speech_regression -- --nocapture

    /// Sample rate of the fixture and the denoise model.
    const SR: f64 = 16_000.0;

    /// Minimal WAV reader for the committed fixture: 16 kHz mono 16-bit PCM. Kept
    /// local so this test is self-contained, mirroring the VAD regression's reader.
    fn read_speech_fixture() -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("assets")
            .join("vad-golden-tts-speech-16k.wav");
        let bytes = std::fs::read(&path).expect("read speech fixture wav");
        assert_eq!(&bytes[0..4], b"RIFF", "fixture must be RIFF/WAVE");
        assert_eq!(&bytes[8..12], b"WAVE", "fixture must be RIFF/WAVE");

        let read_u32 = |b: &[u8], at: usize| {
            u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]]) as usize
        };
        let read_u16 = |b: &[u8], at: usize| u16::from_le_bytes([b[at], b[at + 1]]) as usize;

        // Walk the chunks, confirm the format, and return the data as f32.
        let mut pos = 12;
        let mut channels = 0;
        let mut bits = 0;
        let mut rate = 0;
        let mut samples = Vec::new();
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = read_u32(&bytes, pos + 4);
            let body = pos + 8;
            if id == b"fmt " {
                channels = read_u16(&bytes, body + 2);
                rate = read_u32(&bytes, body + 4);
                bits = read_u16(&bytes, body + 14);
            } else if id == b"data" {
                samples = bytes[body..body + size]
                    .chunks_exact(2)
                    .map(|s| i16::from_le_bytes([s[0], s[1]]) as f32 / 32768.0)
                    .collect();
            }
            pos = body + size + (size & 1);
        }
        assert_eq!(channels, 1, "fixture must be mono");
        assert_eq!(bits, 16, "fixture must be 16-bit PCM");
        assert_eq!(rate, SR as usize, "fixture must be 16 kHz");
        assert!(!samples.is_empty(), "fixture must contain samples");
        samples
    }

    /// A small deterministic PRNG (an LCG) so the synthesized noise is identical
    /// on every run and platform, never `rand`. Same constants as the existing
    /// `noisy` helper above.
    struct Lcg(u64);

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            self.0
        }
        /// Uniform in [-1, 1).
        fn uniform(&mut self) -> f64 {
            (self.next_u64() >> 40) as f64 / (1u64 << 23) as f64 - 1.0
        }
        /// Approximate standard normal: the sum of four uniforms (cheap, fully
        /// deterministic, adequate for a noise generator).
        fn gaussian(&mut self) -> f64 {
            (0..4).map(|_| self.uniform()).sum::<f64>() * 0.5
        }
    }

    /// Scale a buffer so its peak magnitude is 1.0 (so the SNR mixer controls the
    /// final level, not the generator's raw amplitude).
    fn peak_normalize(mut v: Vec<f64>) -> Vec<f64> {
        let peak = v.iter().fold(0.0_f64, |m, &x| m.max(x.abs()));
        if peak > 0.0 {
            for x in &mut v {
                *x /= peak;
            }
        }
        v
    }

    /// White Gaussian noise.
    fn white_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        peak_normalize((0..n).map(|_| rng.gaussian()).collect())
    }

    /// Pink noise via Paul Kellet's economy filter on white noise.
    fn pink_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        let (mut b0, mut b1, mut b2, mut b3, mut b4, mut b5, mut b6) =
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        let out = (0..n)
            .map(|_| {
                let w = rng.gaussian();
                b0 = 0.99886 * b0 + w * 0.0555179;
                b1 = 0.99332 * b1 + w * 0.0750759;
                b2 = 0.96900 * b2 + w * 0.1538520;
                b3 = 0.86650 * b3 + w * 0.3104856;
                b4 = 0.55000 * b4 + w * 0.5329522;
                b5 = -0.7616 * b5 - w * 0.0168980;
                let pink = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
                b6 = w * 0.115926;
                pink
            })
            .collect();
        peak_normalize(out)
    }

    /// A non-stationary out-of-distribution stand-in: pink noise under a slow
    /// amplitude envelope, plus a few deterministic broadband bursts, so its
    /// spectrum and level vary over time (unlike the stationary cases).
    fn modulated_noise(n: usize, seed: u64) -> Vec<f64> {
        let base = pink_noise(n, seed);
        let mut out: Vec<f64> = base
            .iter()
            .enumerate()
            .map(|(i, &x)| {
                let t = i as f64 / SR;
                let env = 0.35 + 0.65 * (0.5 - 0.5 * (2.0 * std::f64::consts::PI * 0.7 * t).cos());
                x * env
            })
            .collect();
        // Deterministic broadband bursts at fixed positions.
        let mut rng = Lcg::new(seed ^ 0xB0FF);
        let burst_len = (0.08 * SR) as usize;
        let mut start = (0.5 * SR) as usize;
        while start + burst_len < n {
            for x in out.iter_mut().skip(start).take(burst_len) {
                *x += rng.gaussian() * 0.9;
            }
            start += (1.3 * SR) as usize;
        }
        peak_normalize(out)
    }

    /// Multi-talker babble: the sum of several shifted, level-varied copies of the
    /// clean speech itself, so the interferers are real speech (the hardest case
    /// for a single-channel model).
    fn babble_noise(speech: &[f32], n: usize, seed: u64) -> Vec<f64> {
        let mut rng = Lcg::new(seed);
        let mut out = vec![0.0_f64; n];
        for _ in 0..5 {
            let shift = (rng.next_u64() as usize) % speech.len();
            let gain = 0.5 + rng.uniform().abs();
            for (i, o) in out.iter_mut().enumerate() {
                *o += speech[(i + shift) % speech.len()] as f64 * gain;
            }
        }
        peak_normalize(out)
    }

    /// Mix speech with noise at a target input SNR (dB), scaling the noise to hit
    /// the SNR deterministically and pulling the peak back under full scale if the
    /// sum would clip.
    fn mix_at_snr(speech: &[f32], noise: &[f64], snr_db: f64) -> Vec<f32> {
        let n = speech.len();
        let ps: f64 = speech.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>() / n as f64;
        let pn: f64 = noise.iter().take(n).map(|&x| x * x).sum::<f64>() / n as f64 + 1e-12;
        let gain = (ps / (pn * 10.0_f64.powf(snr_db / 10.0))).sqrt();
        let mut out: Vec<f32> = speech
            .iter()
            .zip(noise.iter())
            .map(|(&s, &x)| (s as f64 + gain * x) as f32)
            .collect();
        let peak = out.iter().fold(0.0_f32, |m, &x| m.max(x.abs()));
        if peak > 0.99 {
            let k = 0.99 / peak;
            for x in &mut out {
                *x *= k;
            }
        }
        out
    }

    /// Run a signal through the real denoise stage (process then flush), exactly as
    /// the capture chain drives it, and return the enhanced output.
    fn run_denoise(model: &Path, input: &[f32]) -> Vec<f32> {
        let mut stage = Denoise::new(DenoiseModel::FastEnhancerT, model, None)
            .expect("bundled FastEnhancer-T model loads");
        let mut out = Vec::new();
        stage.process(input, &mut out).expect("process runs");
        stage.flush(&mut out).expect("flush drains the tail");
        out
    }

    /// Drop the stage's one-hop output latency so the enhanced signal lines up
    /// with the clean reference, and trim both to a common length.
    fn align<'a>(clean: &'a [f32], enhanced: &'a [f32]) -> (&'a [f32], &'a [f32]) {
        let lead = HOP.min(enhanced.len());
        let est = &enhanced[lead..];
        let m = clean.len().min(est.len());
        (&clean[..m], &est[..m])
    }

    /// SI-SDR (dB): scale-invariant, so a benign overall gain does not move it.
    fn si_sdr(reference: &[f32], estimate: &[f32]) -> f64 {
        let n = reference.len().min(estimate.len());
        let rm = reference[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let em = estimate[..n].iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let r: Vec<f64> = reference[..n].iter().map(|&x| x as f64 - rm).collect();
        let e: Vec<f64> = estimate[..n].iter().map(|&x| x as f64 - em).collect();
        let dot_er: f64 = e.iter().zip(&r).map(|(a, b)| a * b).sum();
        let dot_rr: f64 = r.iter().map(|a| a * a).sum::<f64>() + 1e-12;
        let alpha = dot_er / dot_rr;
        let mut proj_e = 0.0;
        let mut noise_e = 0.0;
        for (a, b) in e.iter().zip(&r) {
            let proj = alpha * b;
            proj_e += proj * proj;
            noise_e += (a - proj) * (a - proj);
        }
        10.0 * ((proj_e + 1e-12) / (noise_e + 1e-12)).log10()
    }

    /// Segmental SNR (dB): per-frame SNR over active frames, clamped to a standard
    /// range and averaged.
    fn seg_snr(reference: &[f32], estimate: &[f32]) -> f64 {
        const FRAME: usize = 256;
        let n = reference.len().min(estimate.len());
        let frames = n / FRAME;
        let mut powers = Vec::with_capacity(frames);
        for f in 0..frames {
            let base = f * FRAME;
            let sig: f64 = reference[base..base + FRAME]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum();
            powers.push(sig);
        }
        let max_power = powers.iter().fold(0.0_f64, |m, &x| m.max(x));
        let floor = max_power * 1e-4;
        let mut sum = 0.0;
        let mut count = 0;
        for (f, &power) in powers.iter().enumerate() {
            if power <= floor {
                continue;
            }
            let base = f * FRAME;
            let err: f64 = reference[base..base + FRAME]
                .iter()
                .zip(&estimate[base..base + FRAME])
                .map(|(&a, &b)| {
                    let d = a as f64 - b as f64;
                    d * d
                })
                .sum();
            let snr = 10.0 * ((power + 1e-10) / (err + 1e-10)).log10();
            sum += snr.clamp(-10.0, 35.0);
            count += 1;
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// In-place iterative radix-2 FFT (no external dependency), used only by the
    /// frequency-weighted segmental SNR proxy below.
    fn fft(re: &mut [f64], im: &mut [f64]) {
        let n = re.len();
        let mut j = 0;
        for i in 1..n {
            let mut bit = n >> 1;
            while j & bit != 0 {
                j ^= bit;
                bit >>= 1;
            }
            j |= bit;
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        let mut len = 2;
        while len <= n {
            let ang = -2.0 * std::f64::consts::PI / len as f64;
            let (wr, wi) = (ang.cos(), ang.sin());
            let mut i = 0;
            while i < n {
                let (mut cr, mut ci) = (1.0_f64, 0.0_f64);
                for k in 0..len / 2 {
                    let a = i + k;
                    let b = i + k + len / 2;
                    let tr = re[b] * cr - im[b] * ci;
                    let ti = re[b] * ci + im[b] * cr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                    let ncr = cr * wr - ci * wi;
                    ci = cr * wi + ci * wr;
                    cr = ncr;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// Magnitude spectrum (first half + DC) of a Hann-windowed frame.
    fn frame_magnitudes(frame: &[f32], window: &[f64]) -> Vec<f64> {
        let n = frame.len();
        let mut re: Vec<f64> = frame
            .iter()
            .zip(window)
            .map(|(&x, &w)| x as f64 * w)
            .collect();
        let mut im = vec![0.0_f64; n];
        fft(&mut re, &mut im);
        (0..=n / 2)
            .map(|k| (re[k] * re[k] + im[k] * im[k]).sqrt())
            .collect()
    }

    /// Frequency-weighted segmental SNR (dB), a perceptual proxy (NOT ITU PESQ):
    /// per-frame, per-band clean-to-error SNR, clamped, energy-weighted, and
    /// averaged over active frames. Bands are linear over the spectrum (a
    /// documented simplification, not Bark critical bands).
    fn fw_seg_snr(reference: &[f32], estimate: &[f32]) -> f64 {
        const FRAME: usize = 512;
        const HOP_LEN: usize = 256;
        const BANDS: usize = 24;
        let n = reference.len().min(estimate.len());
        if n < FRAME {
            return 0.0;
        }
        let window: Vec<f64> = (0..FRAME)
            .map(|i| 0.5 - 0.5 * (2.0 * std::f64::consts::PI * i as f64 / FRAME as f64).cos())
            .collect();
        let bins = FRAME / 2 + 1;

        // Frame energies first, to find active frames.
        let mut sum = 0.0;
        let mut count = 0;
        let mut frame_energy = Vec::new();
        let mut pos = 0;
        while pos + FRAME <= n {
            let e: f64 = reference[pos..pos + FRAME]
                .iter()
                .map(|&x| (x as f64) * (x as f64))
                .sum();
            frame_energy.push(e);
            pos += HOP_LEN;
        }
        let max_energy = frame_energy.iter().fold(0.0_f64, |m, &x| m.max(x));
        let energy_floor = max_energy * 1e-4;

        let mut frame_idx = 0;
        pos = 0;
        while pos + FRAME <= n {
            if frame_energy[frame_idx] > energy_floor {
                let cm = frame_magnitudes(&reference[pos..pos + FRAME], &window);
                let em = frame_magnitudes(&estimate[pos..pos + FRAME], &window);
                let mut w_snr = 0.0;
                let mut w_total = 0.0;
                for band in 0..BANDS {
                    let lo = band * bins / BANDS;
                    let hi = ((band + 1) * bins / BANDS).max(lo + 1).min(bins);
                    let mut cb = 0.0;
                    let mut eb = 0.0;
                    for k in lo..hi {
                        cb += cm[k] * cm[k];
                        let d = cm[k] - em[k];
                        eb += d * d;
                    }
                    let snr = 10.0 * ((cb + 1e-10) / (eb + 1e-10)).log10();
                    let weight = cb.powf(0.2);
                    w_snr += weight * snr.clamp(-10.0, 35.0);
                    w_total += weight;
                }
                if w_total > 0.0 {
                    sum += w_snr / w_total;
                    count += 1;
                }
            }
            frame_idx += 1;
            pos += HOP_LEN;
        }
        if count == 0 {
            0.0
        } else {
            sum / count as f64
        }
    }

    /// The three measures of a processed signal against the clean reference.
    struct Scores {
        si_sdr: f64,
        seg_snr: f64,
        fw_seg_snr: f64,
    }

    fn score(clean: &[f32], processed: &[f32]) -> Scores {
        Scores {
            si_sdr: si_sdr(clean, processed),
            seg_snr: seg_snr(clean, processed),
            fw_seg_snr: fw_seg_snr(clean, processed),
        }
    }

    /// Which floor a case is held to.
    #[derive(Clone, Copy, PartialEq)]
    enum Category {
        Stationary,
        OutOfDistribution,
        Babble,
        Clean,
    }

    // Floors, set a margin below the minimum this build measured (printed by
    // DECIBRI_PRINT_DENOISE_METRICS on Windows, ORT 1.24.x via download-binaries).
    // Each floor is an improvement (enhanced minus noisy) in dB, except the clean
    // case which floors the absolute enhanced score (the model is not transparent
    // on clean input, so this catches a regression that mangles clean speech
    // without demanding transparency). Margins absorb cross-platform ORT drift,
    // which is tiny on these frame-averaged measures.
    //
    // Measured minimums across the cases in each category (Windows, ORT 1.24.x,
    // download-binaries), and the floors chosen, each roughly 2 dB below the
    // measured minimum. A real regression (caches not carried, the frame
    // convention off, the model swapped for a worse one) collapses these
    // improvements toward zero or negative across every measure, well past a 2 dB
    // margin, while cross-platform ORT drift on these frame-averaged measures
    // stays far under it.
    //   stationary (white/pink, 0/5/10 dB), improvement (enhanced minus noisy):
    //     SI-SDR  min  7.08  -> floor 5.0  (margin ~2.1)
    //     segSNR  min  5.15  -> floor 3.0  (margin ~2.2)
    //     fwSNR   min  3.14  -> floor 1.5  (margin ~1.6)
    //   out-of-distribution (modulated, 0/5/10 dB), improvement:
    //     SI-SDR  min  6.10  -> floor 4.0  (margin ~2.1)
    //     segSNR  min  2.97  -> floor 1.0  (margin ~2.0)
    //     fwSNR   min  1.86  -> floor 0.0  (must not worsen; margin ~1.9)
    //   babble (0/5/10 dB), no-harm only: a single-channel model cannot separate
    //   competing speech, so it earns a small SI-SDR gain at best (measured min
    //   2.19, always positive) and its spectral proxy can dip negative; the test
    //   asserts only that it does not make SI-SDR worse than the noisy input:
    //     SI-SDR  min  2.19  -> floor 0.0  (enhanced not worse than noisy)
    //   clean (absolute enhanced score, bounded degradation): the model is not
    //   transparent on clean input (measured SI-SDR 7.62), so the test floors the
    //   absolute score rather than asserting transparency, catching a regression
    //   that destroys clean speech:
    //     SI-SDR  7.62  -> floor 5.0  (margin ~2.6)
    const STATIONARY_SI_SDR_FLOOR: f64 = 5.0;
    const STATIONARY_SEG_SNR_FLOOR: f64 = 3.0;
    const STATIONARY_FW_SNR_FLOOR: f64 = 1.5;
    const OOD_SI_SDR_FLOOR: f64 = 4.0;
    const OOD_SEG_SNR_FLOOR: f64 = 1.0;
    const OOD_FW_SNR_FLOOR: f64 = 0.0;
    const BABBLE_SI_SDR_NO_HARM: f64 = 0.0;
    const CLEAN_SI_SDR_FLOOR: f64 = 5.0;

    /// The full set of cases: each clean reference (here one fixture), each noise
    /// type, at each SNR, plus the clean control.
    fn build_cases(clean: &[f32]) -> Vec<(String, Category, Vec<f32>)> {
        let n = clean.len();
        let snrs = [0.0_f64, 5.0, 10.0];
        let mut cases = Vec::new();
        for &snr in &snrs {
            cases.push((
                format!("white {snr:>2.0}dB"),
                Category::Stationary,
                mix_at_snr(clean, &white_noise(n, 0x5EED_0001), snr),
            ));
            cases.push((
                format!("pink {snr:>2.0}dB"),
                Category::Stationary,
                mix_at_snr(clean, &pink_noise(n, 0x5EED_0002), snr),
            ));
            cases.push((
                format!("modulated {snr:>2.0}dB"),
                Category::OutOfDistribution,
                mix_at_snr(clean, &modulated_noise(n, 0x5EED_0003), snr),
            ));
            cases.push((
                format!("babble {snr:>2.0}dB"),
                Category::Babble,
                mix_at_snr(clean, &babble_noise(clean, n, 0x5EED_0004), snr),
            ));
        }
        cases.push(("clean".to_string(), Category::Clean, clean.to_vec()));
        cases
    }

    /// Feed noisy speech through the real denoise stage and assert the enhancement
    /// clears the per-category floors. See the module-level comment above.
    #[test]
    fn denoise_improves_noisy_speech_regression() {
        let model = model_path();
        let clean = read_speech_fixture();
        let cases = build_cases(&clean);
        let print = std::env::var("DECIBRI_PRINT_DENOISE_METRICS").is_ok();

        // Track per-category minimum improvements (and clean absolute scores) so a
        // re-measurement can set the floors a margin below these.
        let mut min_stat = (f64::MAX, f64::MAX, f64::MAX);
        let mut min_ood = (f64::MAX, f64::MAX, f64::MAX);
        let mut min_babble_si = f64::MAX;
        let mut clean_abs = (0.0, 0.0, 0.0);

        for (label, category, noisy_sig) in &cases {
            let enhanced = run_denoise(&model, noisy_sig);
            let (clean_a, enh_a) = align(&clean, &enhanced);
            let out = score(clean_a, enh_a);

            if *category == Category::Clean {
                clean_abs = (out.si_sdr, out.seg_snr, out.fw_seg_snr);
                if print {
                    eprintln!(
                        "{label:14} (absolute)  SI-SDR {:7.2}  segSNR {:7.2}  fwSNR {:7.2}",
                        out.si_sdr, out.seg_snr, out.fw_seg_snr
                    );
                }
                continue;
            }

            // Noisy input scored over the same aligned region as the enhanced one.
            let noisy_a = &noisy_sig[..clean_a.len().min(noisy_sig.len())];
            let in_scores = score(&clean[..noisy_a.len()], noisy_a);
            let d_si = out.si_sdr - in_scores.si_sdr;
            let d_seg = out.seg_snr - in_scores.seg_snr;
            let d_fw = out.fw_seg_snr - in_scores.fw_seg_snr;

            if print {
                eprintln!(
                    "{label:14}  dSI-SDR {d_si:7.2}  dsegSNR {d_seg:7.2}  dfwSNR {d_fw:7.2}  \
                     (in SI-SDR {:6.2} -> out {:6.2})",
                    in_scores.si_sdr, out.si_sdr
                );
            }

            match category {
                Category::Stationary => {
                    min_stat.0 = min_stat.0.min(d_si);
                    min_stat.1 = min_stat.1.min(d_seg);
                    min_stat.2 = min_stat.2.min(d_fw);
                }
                Category::OutOfDistribution => {
                    min_ood.0 = min_ood.0.min(d_si);
                    min_ood.1 = min_ood.1.min(d_seg);
                    min_ood.2 = min_ood.2.min(d_fw);
                }
                Category::Babble => {
                    min_babble_si = min_babble_si.min(d_si);
                }
                Category::Clean => {}
            }

            if !print {
                match category {
                    Category::Stationary => {
                        assert!(
                            d_si >= STATIONARY_SI_SDR_FLOOR
                                && d_seg >= STATIONARY_SEG_SNR_FLOOR
                                && d_fw >= STATIONARY_FW_SNR_FLOOR,
                            "{label}: stationary improvement below floor: \
                             dSI-SDR {d_si:.2} dsegSNR {d_seg:.2} dfwSNR {d_fw:.2}"
                        );
                    }
                    Category::OutOfDistribution => {
                        assert!(
                            d_si >= OOD_SI_SDR_FLOOR
                                && d_seg >= OOD_SEG_SNR_FLOOR
                                && d_fw >= OOD_FW_SNR_FLOOR,
                            "{label}: out-of-distribution improvement below floor: \
                             dSI-SDR {d_si:.2} dsegSNR {d_seg:.2} dfwSNR {d_fw:.2}"
                        );
                    }
                    Category::Babble => {
                        assert!(
                            d_si >= BABBLE_SI_SDR_NO_HARM,
                            "{label}: babble did harm (SI-SDR change {d_si:.2} below \
                             the no-harm floor)"
                        );
                    }
                    Category::Clean => {}
                }
            }
        }

        if print {
            eprintln!("--- per-category minimums (set floors a margin below) ---");
            eprintln!(
                "stationary  dSI-SDR {:.2}  dsegSNR {:.2}  dfwSNR {:.2}",
                min_stat.0, min_stat.1, min_stat.2
            );
            eprintln!(
                "ood         dSI-SDR {:.2}  dsegSNR {:.2}  dfwSNR {:.2}",
                min_ood.0, min_ood.1, min_ood.2
            );
            eprintln!("babble      dSI-SDR {min_babble_si:.2} (no-harm)");
            eprintln!(
                "clean       SI-SDR {:.2}  segSNR {:.2}  fwSNR {:.2} (absolute)",
                clean_abs.0, clean_abs.1, clean_abs.2
            );
            panic!(
                "DECIBRI_PRINT_DENOISE_METRICS is set: copy the minimums into the floor \
                 constants (a margin below) and rerun without the variable"
            );
        }

        // Clean input is not transparent through the model, but it must not be
        // badly damaged: floor its absolute enhanced SI-SDR.
        assert!(
            clean_abs.0 >= CLEAN_SI_SDR_FLOOR,
            "clean input damaged: enhanced SI-SDR {:.2} below floor {CLEAN_SI_SDR_FLOOR:.2}",
            clean_abs.0
        );
    }
}
