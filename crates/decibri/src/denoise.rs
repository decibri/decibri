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
    pub(crate) fn new(model: DenoiseModel, path: &Path) -> Result<Self, DecibriError> {
        // ORT initialises exactly once per process (first-wins). A VAD in the
        // same process may have done it already; if not, denoise does it here.
        crate::onnx::init_ort_once(None)?;

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
        let mut stage = Denoise::new(DenoiseModel::FastEnhancerT, &model_path())
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
        let mut stage = Denoise::new(DenoiseModel::FastEnhancerT, &model_path())
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
        let err = match Denoise::new(DenoiseModel::FastEnhancerT, &missing) {
            Ok(_) => panic!("a missing model file must fail to load"),
            Err(e) => e,
        };
        assert!(
            matches!(err, DecibriError::ModelLoadFailed { .. }),
            "denoise surfaces ModelLoadFailed, not VadModelLoadFailed: {err:?}"
        );
    }
}
