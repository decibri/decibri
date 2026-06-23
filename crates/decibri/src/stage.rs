//! Internal capture stage chain.
//!
//! A [`CaptureStage`] conditions raw device capture into the canonical format the
//! consumer receives, running between the cpal callback's native buffers and the
//! exact-size reblock in [`crate::microphone`]. The chain has two segments. The
//! `normalize` segment holds up to two stages: [`Downmix`], which averages a
//! multichannel device down to mono, then [`ResampleStage`], which converts the
//! device's native sample rate to the requested target rate. Downmix runs first
//! so the resampler receives mono, the format it expects. The optional
//! `transform` segment runs after `normalize`, holding the conditioning a
//! consumer opts into: the same-length [`DcBlocker`] DC-removal step, then the
//! framed, length-changing denoise stage. A `transform` stage need not preserve
//! length (denoise re-blocks and introduces latency), which is why the VAD tap
//! is snapshotted before this segment runs.
//!
//! Each [`Stage`] reads one block of interleaved f32 samples and writes its
//! output, so stages compose by ping-ponging two buffers. At stream close the
//! chain drains each stage's held tail through the same walk, so the resampler's
//! group-delay tail is delivered rather than dropped. The chain is built by
//! [`build_capture_stage`], which returns `None` when no conditioning is needed
//! (an already-mono device already at the target rate with no enhancement
//! enabled), keeping the capture path on its zero-cost direct path.

use std::path::Path;

use decibri_resampler::{PolyphaseResampler, Resampler};

use crate::error::DecibriError;
use crate::microphone::{DenoiseModel, HighpassFilter};
use crate::sample;

/// A capture processing stage: reads one block of interleaved f32 samples and
/// appends its processed output to `out` (the caller clears `out` first).
///
/// `Send` so the boxed stages can live behind the stream's `Mutex` and cross
/// thread boundaries with the `Send + Sync` [`crate::microphone::MicrophoneStream`].
pub(crate) trait Stage: Send {
    /// Process `input`, appending the result to `out`.
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), DecibriError>;

    /// Drain any end-of-stream tail held in the stage's state into `out`.
    ///
    /// Called once at stream close, after the final [`process`](Stage::process).
    /// A stateless stage holds no tail and keeps this default no-op; a stage that
    /// carries state across blocks (the resampler's anti-alias filter holds a
    /// group-delay tail) overrides it to append the held samples, so no captured
    /// audio is dropped at close.
    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        let _ = out;
        Ok(())
    }

    /// The constant algorithmic delay, in samples at this stage's output rate,
    /// that the stage adds between its input and its output.
    ///
    /// Zero by default: a sample-in, sample-out stage (downmix, the DC blocker)
    /// emits each output sample in step with its input and adds no delay. A
    /// stage that holds samples back overrides this. The resampler reports its
    /// anti-alias filter's group delay, and the framed denoise stage reports the
    /// lead its analysis window introduces. The chain sums these across its
    /// conditioning stages.
    fn latency_samples(&self) -> usize {
        0
    }
}

/// Downmix interleaved multichannel audio to mono by averaging channels.
///
/// Reuses [`sample::downmix_to_mono`] so the engine math is byte-identical to the
/// downmix the bindings apply for the VAD feed: no second, divergent variant.
struct Downmix {
    /// The device channel count to average each interleaved frame down from.
    channels: u16,
}

impl Downmix {
    fn new(channels: u16) -> Self {
        Self { channels }
    }
}

impl Stage for Downmix {
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), DecibriError> {
        out.extend(sample::downmix_to_mono(input, self.channels));
        Ok(())
    }
}

/// Resample mono interleaved audio from the device's native rate to the
/// requested target rate, wrapping the owned [`PolyphaseResampler`].
///
/// [`build_capture_stage`] adds this stage only when the native and target
/// rates differ; a device already at the target rate omits it. It is placed
/// after [`Downmix`] in the chain so the resampler receives mono, the format it
/// expects.
struct ResampleStage(PolyphaseResampler);

impl Stage for ResampleStage {
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The resampler appends to `out` and is infallible once constructed
        // (rates are validated at construction); wrap it as `Ok(())`.
        self.0.process(input, out);
        Ok(())
    }

    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // Drain the resampler's group-delay tail (and any partial-frame carry)
        // into `out`. Called once at close; the resampler appends and is
        // infallible, so wrap it as `Ok(())`.
        self.0.flush(out);
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        // Forward the resampler's own group delay. It is already expressed at
        // the output rate, and is zero when the rates match and the resampler is
        // a passthrough.
        self.0.latency_samples()
    }
}

/// A same-length DSP step: filters a buffer of samples in place, producing
/// exactly as many output samples as it received. The adapter [`InPlace`] wraps
/// any `InPlaceDsp` as a [`Stage`].
///
/// `Send` so the wrapped step can live behind the stream's `Mutex`, matching the
/// [`Stage`] bound. `pub(crate)` so a stage authored in another module (the
/// [`crate::gain::LevelControl`] engine) can implement it.
pub(crate) trait InPlaceDsp: Send {
    /// Filter `samples` in place: each element is replaced by its processed
    /// value, the length unchanged.
    fn process_in_place(&mut self, samples: &mut [f32]);
}

/// Adapts an [`InPlaceDsp`] to the [`Stage`] interface. `process` copies the
/// input into `out` then filters it in place, so the output length equals the
/// input length; `flush` keeps the [`Stage`] default no-op, since a same-length
/// DSP holds no end-of-stream tail.
struct InPlace<T: InPlaceDsp>(T);

impl<T: InPlaceDsp> Stage for InPlace<T> {
    fn process(&mut self, input: &[f32], out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The caller clears `out` first, so this is an exact-length copy of the
        // input that the DSP then rewrites in place: one input sample, one output
        // sample.
        out.extend_from_slice(input);
        self.0.process_in_place(out);
        Ok(())
    }
}

/// One-pole DC-blocking high-pass: removes a constant (DC) offset from the signal
/// while leaving the audio band essentially flat. Implements the standard
/// difference equation `y[n] = x[n] - x[n-1] + R*y[n-1]`, with the pole `R` just
/// below 1 so the corner sits close to DC.
///
/// Same-length and sample-by-sample (one input sample yields one output sample),
/// so it is an [`InPlaceDsp`] driven through [`InPlace`]. The filter memory
/// (`x_prev`, `y_prev`) carries across blocks, so the response is continuous at
/// chunk boundaries; it holds no end-of-stream tail, so it keeps the default
/// no-op flush.
struct DcBlocker {
    /// Previous input sample `x[n-1]`, carried across blocks.
    x_prev: f32,
    /// Previous output sample `y[n-1]`, carried across blocks.
    y_prev: f32,
}

impl DcBlocker {
    /// Feedback coefficient `R`. At 0.995 the high-pass corner sits near 13 Hz at
    /// 16 kHz, well below the voice band, and a DC offset settles out within a
    /// few hundred samples.
    const R: f32 = 0.995;

    fn new() -> Self {
        Self {
            x_prev: 0.0,
            y_prev: 0.0,
        }
    }
}

impl InPlaceDsp for DcBlocker {
    fn process_in_place(&mut self, samples: &mut [f32]) {
        for s in samples.iter_mut() {
            let x = *s;
            let y = x - self.x_prev + Self::R * self.y_prev;
            self.x_prev = x;
            self.y_prev = y;
            *s = y;
        }
    }
}

/// A second-order (biquad) Butterworth high-pass: attenuates content below the
/// corner frequency at 12 dB per octave while leaving the passband essentially
/// flat (the maximally-flat Butterworth response, quality factor `1/sqrt(2)`).
/// Removes low-frequency rumble below the voice band, a steeper and
/// higher-cornered filter than the always-near-DC [`DcBlocker`].
///
/// Same-length and sample-by-sample (one input sample yields one output
/// sample), so it is an [`InPlaceDsp`] driven through [`InPlace`], the same
/// wrapper the [`DcBlocker`] uses. It runs as a Direct Form II Transposed
/// section carrying two state samples (`z1`, `z2`) across blocks, so the
/// response is continuous at chunk boundaries; it holds no end-of-stream tail,
/// so it keeps the default no-op flush, and there is no reset path, so a fresh
/// filter (with zero state) is built per stream open. The coefficients are
/// computed once at construction from the corner frequency and sample rate via
/// the RBJ audio-EQ-cookbook high-pass design.
struct Biquad {
    /// Feed-forward (numerator) coefficients, already normalised by `a0`.
    b0: f32,
    b1: f32,
    b2: f32,
    /// Feedback (denominator) coefficients, already normalised by `a0` (so the
    /// stored `a0` is 1 and is not kept).
    a1: f32,
    a2: f32,
    /// Direct Form II Transposed state, carried across blocks. Zero at
    /// construction.
    z1: f32,
    z2: f32,
}

impl Biquad {
    /// Design a second-order Butterworth high-pass at `cutoff_hz` for a stream
    /// at `sample_rate_hz`, via the RBJ cookbook high-pass formulas with the
    /// Butterworth quality factor `Q = 1/sqrt(2)` (the maximally-flat damping).
    /// The cutoff and rate fully determine the coefficients.
    fn highpass(cutoff_hz: f32, sample_rate_hz: f32) -> Self {
        use std::f32::consts::{FRAC_1_SQRT_2, PI};

        // Butterworth (maximally-flat) damping.
        let q = FRAC_1_SQRT_2;
        let w0 = 2.0 * PI * cutoff_hz / sample_rate_hz;
        let cos_w0 = w0.cos();
        let alpha = w0.sin() / (2.0 * q);

        // RBJ cookbook high-pass section, pre-normalisation.
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;

        // Normalise so a0 == 1, then store.
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }
}

impl InPlaceDsp for Biquad {
    fn process_in_place(&mut self, samples: &mut [f32]) {
        // Direct Form II Transposed: one multiply-add chain per sample, two
        // state words carried forward, length preserved.
        for s in samples.iter_mut() {
            let x = *s;
            let y = self.b0 * x + self.z1;
            self.z1 = self.b1 * x - self.a1 * y + self.z2;
            self.z2 = self.b2 * x - self.a2 * y;
            *s = y;
        }
    }
}

/// The capture stage chain: a `normalize` segment (downmix, resample) that
/// conditions a device to the output format, then an optional `transform`
/// segment (DC removal, then denoise) applied to the normalized signal. The
/// `transform` stages need not preserve length: the framed denoise stage
/// re-blocks and introduces latency.
///
/// Invariant: at least one segment is non-empty. [`build_capture_stage`] is the
/// only constructor and returns `None` rather than an empty chain when both
/// segments are empty, so the capture path can skip the chain entirely when no
/// conditioning is needed.
pub(crate) struct CaptureStage {
    /// Channel and rate normalization stages (downmix, then resample), applied
    /// first.
    normalize: Vec<Box<dyn Stage>>,
    /// Optional conditioning stages (DC removal, then denoise), applied after
    /// `normalize`. Not necessarily same-length: the denoise stage re-blocks and
    /// introduces latency, so its output count differs from its input.
    transform: Vec<Box<dyn Stage>>,
    /// Holds the current block as it passes through the chain; [`run`](Self::run)
    /// returns a borrow of this after the last stage.
    work: Vec<f32>,
    /// The ping-pong partner of `work`: each stage reads one and writes the other.
    scratch: Vec<f32>,
    /// Snapshot of the post-`normalize`, pre-`transform` signal from the most
    /// recent [`run`](Self::run) or [`flush`](Self::flush), captured only when
    /// `transform` is non-empty (otherwise the delivered output already is that
    /// signal). Read via [`tap`](Self::tap); the caller uses it to feed a detector
    /// the signal before the transform step.
    tap: Vec<f32>,
}

impl CaptureStage {
    /// Run the chain on one native block, returning the conditioned output.
    ///
    /// Seeds `work` with `input`, then ping-pongs `work` <-> `scratch` first
    /// across the `normalize` stages and then across the `transform` stages,
    /// leaving the final output in `work`. When `transform` is non-empty, snapshots
    /// the post-`normalize`, pre-`transform` `work` into `tap` first, so the caller
    /// can read the signal as it stood before the transform step.
    pub(crate) fn run(&mut self, input: &[f32]) -> Result<&[f32], DecibriError> {
        self.work.clear();
        self.work.extend_from_slice(input);
        Self::run_segment(&mut self.work, &mut self.scratch, &mut self.normalize)?;
        self.capture_tap();
        Self::run_segment(&mut self.work, &mut self.scratch, &mut self.transform)?;
        Ok(&self.work)
    }

    /// Snapshot `work` (the post-`normalize`, pre-`transform` signal) into `tap`,
    /// but only when `transform` is non-empty. With no transform the delivered
    /// output already is this signal, so nothing is captured and `tap` stays empty
    /// (zero overhead on the no-transform path).
    fn capture_tap(&mut self) {
        if !self.transform.is_empty() {
            self.tap.clear();
            self.tap.extend_from_slice(&self.work);
        }
    }

    /// Ping-pong one block through a single segment's stages, leaving the result
    /// in `work`. Each stage reads `work` and writes `scratch`; the two are then
    /// swapped, so after the loop `work` holds the segment's output. An empty
    /// segment leaves `work` untouched.
    fn run_segment(
        work: &mut Vec<f32>,
        scratch: &mut Vec<f32>,
        stages: &mut [Box<dyn Stage>],
    ) -> Result<(), DecibriError> {
        for stage in stages.iter_mut() {
            scratch.clear();
            stage.process(work, scratch)?;
            std::mem::swap(work, scratch);
        }
        Ok(())
    }

    /// Drain the chain's complete end-of-stream tail, appending it to `out`.
    ///
    /// Walks the `normalize` segment then the `transform` segment with a single
    /// carried buffer, so a tail produced by one stage flows through every
    /// downstream stage (including across the segment boundary) exactly as a
    /// normal block would. For the stateless [`Downmix`] this contributes
    /// nothing; for [`ResampleStage`] it yields the resampler's group-delay tail,
    /// which then passes through the `transform` stages. Call once at stream
    /// close, after the last [`run`](Self::run).
    pub(crate) fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // `work` carries the inter-stage buffer. It starts empty because there is
        // no new input at end of stream, only each stage's drained tail; it is
        // carried unbroken from the `normalize` walk into the `transform` walk.
        self.work.clear();
        Self::flush_segment(&mut self.work, &mut self.scratch, &mut self.normalize)?;
        // Capture the post-`normalize` tail (the resampler's group-delay tail)
        // before it passes through `transform`, so the tap stays aligned with the
        // delivered output across the close path too.
        self.capture_tap();
        Self::flush_segment(&mut self.work, &mut self.scratch, &mut self.transform)?;
        out.extend_from_slice(&self.work);
        Ok(())
    }

    /// The post-`normalize`, pre-`transform` signal captured by the most recent
    /// [`run`](Self::run) or [`flush`](Self::flush). Empty when the chain has no
    /// `transform` segment (nothing is captured on that path).
    pub(crate) fn tap(&self) -> &[f32] {
        &self.tap
    }

    /// Whether the chain has a `transform` segment, so the delivered output is the
    /// post-transform signal and the [`tap`](Self::tap) carries the distinct
    /// pre-transform signal.
    pub(crate) fn has_transform(&self) -> bool {
        !self.transform.is_empty()
    }

    /// The summed algorithmic latency, in samples at the output rate, of the
    /// `transform` (conditioning) stages: the amount by which the delivered,
    /// post-conditioning output trails the post-normalize signal the
    /// [`tap`](Self::tap) holds. The `normalize` stages are excluded on purpose:
    /// they run before the tap is taken, so their delay (the resampler's group
    /// delay) reaches the tap and the delivered output alike and does not
    /// separate them.
    pub(crate) fn transform_latency(&self) -> usize {
        self.transform.iter().map(|s| s.latency_samples()).sum()
    }

    /// Drain one segment's tail, carrying `work` across its stages. At each stage
    /// it feeds the upstream-carried tail through (when non-empty), then drains
    /// that stage's own held tail into the same buffer. `work` enters holding the
    /// previous segment's carried tail (empty for the first non-empty stage) and
    /// leaves holding this segment's output.
    fn flush_segment(
        work: &mut Vec<f32>,
        scratch: &mut Vec<f32>,
        stages: &mut [Box<dyn Stage>],
    ) -> Result<(), DecibriError> {
        for stage in stages.iter_mut() {
            scratch.clear();
            if !work.is_empty() {
                stage.process(work, scratch)?;
            }
            stage.flush(scratch)?;
            std::mem::swap(work, scratch);
        }
        Ok(())
    }
}

/// The opt-in conditioning [`build_capture_stage`] applies after normalization,
/// bundled into one argument so that adding a capability is a new field rather
/// than another positional parameter. The fields map one-to-one to the transform
/// stages, listed in chain order.
pub(crate) struct Transforms<'a> {
    /// Remove a constant DC offset (the [`DcBlocker`]).
    pub dc_removal: bool,
    /// Denoise model selector with its model-file path and optional ORT library
    /// path; `None` leaves denoise off. Honoured only with the `denoise` feature.
    pub denoise: Option<(DenoiseModel, &'a Path, Option<&'a Path>)>,
    /// High-pass cutoff selector (the [`Biquad`]); `None` leaves it off.
    pub highpass: Option<HighpassFilter>,
    /// AGC target level in dBFS (the [`crate::gain::LevelControl`] engine); `None`
    /// leaves it off. Honoured only with the `gain` feature.
    pub agc: Option<i8>,
    /// Limiter ceiling in dBFS (the [`crate::gain::Limiter`] stage); `None` leaves
    /// it off. Honoured only with the `gain` feature.
    pub limiter: Option<f32>,
}

/// Build the capture stage chain that normalizes a device to the output format
/// and applies any opt-in enhancement.
///
/// Pushes [`Downmix`] when the device delivers more channels than the output
/// target (averaging down to the target, which is mono here), then
/// [`ResampleStage`] when the device's `native_rate` differs from `target_rate`
/// (converting the captured audio to the requested rate). When `dc_removal` is
/// set, pushes the [`DcBlocker`] into the `transform` segment; when `denoise` is
/// `Some((model, model_path, ort_library_path))` (and the `denoise` feature is
/// compiled in), pushes the framed denoise stage immediately after it, loading
/// the model through the ONNX seam (initialising ORT from `ort_library_path` when
/// supplied). When `highpass` names a cutoff, pushes a [`Biquad`] high-pass
/// immediately after denoise, so the denoise model receives near-full-band
/// input. When `agc` names a target level, pushes the
/// [`crate::gain::LevelControl`] engine immediately after high-pass (when the
/// `gain` feature is compiled in). When `limiter` names a ceiling, pushes the
/// [`crate::gain::Limiter`] stage last, immediately after the level control
/// (also when the `gain` feature is compiled in), so it catches any peak the
/// upstream level control would let through. All transform stages run after
/// `normalize` on the mono signal at the target rate.
/// Returns `Some(chain)` when at least one stage is needed and `None` when no
/// segment has any (a mono device already at the target rate with no enhancement
/// enabled), leaving the capture path on its direct, zero-cost reblock.
///
/// Returns an error when the resampler rejects the rate pair at construction (as
/// above), or when the denoise model fails to load (a
/// [`DecibriError::ModelLoadFailed`]).
pub(crate) fn build_capture_stage(
    device_channels: u16,
    target_channels: u16,
    native_rate: u32,
    target_rate: u32,
    transforms: Transforms<'_>,
) -> Result<Option<CaptureStage>, DecibriError> {
    // Unpack the bundle back into locals so the build below reads one field per
    // stage, in chain order.
    let Transforms {
        dc_removal,
        denoise,
        highpass,
        agc,
        limiter,
    } = transforms;

    let mut normalize: Vec<Box<dyn Stage>> = Vec::new();

    if device_channels > target_channels {
        normalize.push(Box::new(Downmix::new(device_channels)));
    }

    if native_rate != target_rate {
        // Downmix (if any) ran first, so the resampler receives mono.
        // Construction validates the rate pair; the `?` bridges a failure to
        // DecibriError::ResampleConfigInvalid via From<ResamplerError>.
        let resampler = PolyphaseResampler::new(native_rate, target_rate)?;
        normalize.push(Box::new(ResampleStage(resampler)));
    }

    let mut transform: Vec<Box<dyn Stage>> = Vec::new();

    if dc_removal {
        // Runs after `normalize`, on the mono signal at the target rate.
        transform.push(Box::new(InPlace(DcBlocker::new())));
    }

    // Denoise runs immediately AFTER DC removal (chain order: DcRemoval ->
    // Denoise), on the DC-blocked mono signal at the target rate. The order is
    // load-bearing (denoise wants a clean-rate, DC-free input) and is pinned by
    // this push order and `build_orders_denoise_after_dc`. Unlike the same-length
    // DcBlocker, denoise is framed and latency-introducing, so it sits in the
    // transform segment after the VAD tap (see the tap docs in `microphone`).
    #[cfg(feature = "denoise")]
    if let Some((model, path, ort_library_path)) = denoise {
        transform.push(Box::new(crate::denoise::Denoise::new(
            model,
            path,
            ort_library_path,
        )?));
    }
    // Without the `denoise` feature the parameter is accepted but unused.
    #[cfg(not(feature = "denoise"))]
    let _ = denoise;

    // High-pass (user rumble cut) runs immediately AFTER denoise (chain order:
    // Denoise -> HighPass), so the denoise model receives near-full-band input.
    // It is a same-length, sample-in-sample-out biquad, so it wraps via `InPlace`
    // exactly like the DC blocker and adds no latency. The cutoff comes from the
    // named variant, not a magic number here.
    if let Some(filter) = highpass {
        transform.push(Box::new(InPlace(Biquad::highpass(
            filter.cutoff_hz(),
            target_rate as f32,
        ))));
    }

    // Level control (AGC) runs after high-pass, reserving the slot that sits
    // before the limiter in the full chain order. It is a same-length,
    // sample-in-sample-out engine, so it wraps via `InPlace` like the DC blocker
    // and high-pass and adds no latency. Gated on the `gain` feature, like
    // denoise is gated on `denoise`; without the feature the target is accepted
    // but unused.
    #[cfg(feature = "gain")]
    if let Some(target_db) = agc {
        transform.push(Box::new(InPlace(crate::gain::LevelControl::agc(
            target_db,
            target_rate,
        ))));
    }
    #[cfg(not(feature = "gain"))]
    let _ = agc;

    // The limiter runs LAST in the transform tier, immediately after the level
    // control, so it catches any peak the upstream gain would let exceed the
    // ceiling. It is a same-length, sample-in-sample-out stage, so it wraps via
    // `InPlace` like the level control and adds no latency. Gated on the same
    // `gain` feature as the level-control engine (the pair); without the feature
    // the ceiling is accepted but unused. Nothing runs after it.
    #[cfg(feature = "gain")]
    if let Some(ceiling_db) = limiter {
        transform.push(Box::new(InPlace(crate::gain::Limiter::new(
            ceiling_db,
            target_rate,
        ))));
    }
    #[cfg(not(feature = "gain"))]
    let _ = limiter;

    Ok(if normalize.is_empty() && transform.is_empty() {
        None
    } else {
        Some(CaptureStage {
            normalize,
            transform,
            work: Vec::new(),
            scratch: Vec::new(),
            tap: Vec::new(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// With enhancement off (the default), `build_capture_stage` returns `None`
    /// only for a mono device already at the target rate, and `Some` once a
    /// downmix and/or a resample is needed.
    #[test]
    fn build_returns_none_only_for_mono_at_target_rate() {
        let off = false;
        // Mono, native == target: nothing to normalize.
        assert!(
            build_capture_stage(
                1,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: off,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_none(),
            "mono device at the target rate needs no chain"
        );
        // Multichannel, native == target: downmix only.
        assert!(
            build_capture_stage(
                2,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: off,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_some(),
            "stereo device gets a chain (downmix)"
        );
        assert!(
            build_capture_stage(
                6,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: off,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_some(),
            "5.1 device gets a chain (downmix)"
        );
        // Mono, native != target: resample only.
        assert!(
            build_capture_stage(
                1,
                1,
                48_000,
                16_000,
                Transforms {
                    dc_removal: off,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_some(),
            "mono device above the target rate gets a chain (resample)"
        );
        // Multichannel, native != target: downmix then resample.
        assert!(
            build_capture_stage(
                2,
                1,
                48_000,
                16_000,
                Transforms {
                    dc_removal: off,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_some(),
            "stereo device above the target rate gets a chain (downmix + resample)"
        );
    }

    /// Enabling the DC-removal step adds a `transform` stage, so even a mono
    /// device already at the target rate now gets a chain (transform-only, with
    /// an empty `normalize`); `build_capture_stage` returns `None` only when BOTH
    /// segments are empty.
    #[test]
    fn build_with_dc_removal_adds_transform_even_with_empty_normalize() {
        let on = true;

        // Mono at target, enhancement on: a transform-only chain (no normalize).
        let chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: on,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc_removal builds a chain even with nothing to normalize");
        assert!(
            chain.normalize.is_empty(),
            "a mono device at the target rate needs no normalize stage"
        );
        assert_eq!(
            chain.transform.len(),
            1,
            "dc_removal pushes exactly one transform stage"
        );

        // Stereo above target, enhancement on: downmix + resample + DC.
        let full = build_capture_stage(
            2,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: on,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("downmix + resample + DC chain");
        assert_eq!(full.normalize.len(), 2, "downmix then resample");
        assert_eq!(full.transform.len(), 1, "the DC-removal step");
    }

    /// The downmix chain averages interleaved frames to mono, reproducing
    /// `sample::downmix_to_mono` exactly (sample-identity of the downmix). A
    /// device at the target rate adds no resample stage, so the chain is the
    /// pure downmix.
    #[test]
    fn downmix_chain_averages_to_mono() {
        let mut chain = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("stereo -> downmix chain");
        // Two stereo frames: (0.5, 0.3) -> 0.4, (0.4, 0.6) -> 0.5.
        let out = chain.run(&[0.5, 0.3, 0.4, 0.6]).expect("downmix runs");
        assert_eq!(out.len(), 2, "stereo input halves to mono");
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        // Reusing sample::downmix_to_mono guarantees identical math.
        assert_eq!(out, sample::downmix_to_mono(&[0.5, 0.3, 0.4, 0.6], 2));
    }

    /// A non-target native rate yields a resample chain whose output tracks the
    /// rate ratio: 48 kHz -> 16 kHz downsamples the mono input by 1:3 (minus the
    /// filter's startup ramp, since this `process`-only path does not flush).
    #[test]
    fn resample_chain_changes_rate_and_count() {
        let mut chain = build_capture_stage(
            1,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("48k mono -> resample chain");
        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();
        let out = chain.run(&input).expect("resample runs").to_vec();
        // Downsampling 1:3 from 24000 input is at most 8000 output samples and,
        // after the brief warmup, comfortably above a quarter of the input.
        assert!(
            out.len() <= input.len() / 3,
            "downsampling never exceeds the 1:3 count: {} > {}",
            out.len(),
            input.len() / 3
        );
        assert!(
            out.len() > input.len() / 4,
            "downsampled count {} should be near the 1:3 ratio",
            out.len()
        );
    }

    /// The `From<ResamplerError>` bridge maps the resampler's construction error
    /// onto `DecibriError::ResampleConfigInvalid`, carrying the rate pair from
    /// `RatePairUnsupported` and folding the guarded `ZeroSampleRate` defensively.
    #[test]
    fn resampler_error_bridges_to_decibri_error() {
        use decibri_resampler::ResamplerError;
        let mapped: DecibriError = ResamplerError::RatePairUnsupported {
            in_rate: 7,
            out_rate: 9,
        }
        .into();
        assert!(matches!(
            mapped,
            DecibriError::ResampleConfigInvalid {
                in_rate: 7,
                out_rate: 9
            }
        ));
        let zero: DecibriError = ResamplerError::ZeroSampleRate.into();
        assert!(matches!(zero, DecibriError::ResampleConfigInvalid { .. }));
    }

    /// `build_capture_stage` propagates a resampler construction failure as
    /// `ResampleConfigInvalid`. In-range rates always construct, so this uses an
    /// out-of-range native rate whose anti-alias filter exceeds the resampler's
    /// cap to exercise the `?` bridge end to end.
    #[test]
    fn build_capture_stage_surfaces_unsupported_rate_pair() {
        let result = build_capture_stage(
            1,
            1,
            316_800_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        );
        assert!(
            matches!(result, Err(DecibriError::ResampleConfigInvalid { .. })),
            "an enormous native rate exceeds the resampler's filter cap"
        );
    }

    /// `CaptureStage::flush` drains the resampling chain's complete end-of-stream
    /// tail: processing the whole stream through the chain and then flushing once
    /// reproduces, bit for bit, a bare resampler fed the whole stream then
    /// flushed once. The flushed tail is nonzero (the resampler holds a
    /// group-delay tail), it is appended after the process output, and no sample
    /// is lost, reordered, or scaled. This is the no-sample-dropped guarantee for
    /// the resample path at the chain level.
    #[test]
    fn flush_drains_resampler_tail_exactly() {
        use decibri_resampler::{PolyphaseResampler, Resampler};

        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();

        // Ground truth: a bare resampler, whole input then a single flush.
        let mut reference = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut expected = Vec::new();
        reference.process(&input, &mut expected);
        reference.flush(&mut expected);

        // The chain: process the whole input via run(), then flush() the tail.
        let mut chain = build_capture_stage(
            1,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("48k mono -> resample chain");
        let mut got = chain.run(&input).expect("process runs").to_vec();
        let process_only = got.len();
        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush drains the tail");

        assert!(
            !tail.is_empty(),
            "the resampler holds a group-delay tail to drain"
        );
        got.extend_from_slice(&tail);
        assert_eq!(
            got, expected,
            "chain process+flush reproduces the full resampled signal, tail included, bit for bit"
        );
        assert_eq!(
            got.len(),
            process_only + tail.len(),
            "the flushed tail is appended after the process output, nothing reordered"
        );
    }

    /// `Downmix` is stateless, so its trait-default `flush` appends nothing: a
    /// downmix-only chain has no end-of-stream tail and `flush` yields no extra
    /// samples (the unchanged downmix-only path).
    #[test]
    fn downmix_only_flush_is_empty() {
        let mut chain = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("stereo -> downmix chain");
        let _ = chain.run(&[0.5, 0.3, 0.4, 0.6]).expect("downmix runs");
        let mut tail = Vec::new();
        chain
            .flush(&mut tail)
            .expect("flush on a downmix-only chain");
        assert!(
            tail.is_empty(),
            "a stateless downmix chain has no flush tail"
        );
    }

    /// Transform-off no-op: with enhancement off (the default) the `transform`
    /// segment is empty, so a downmix-only chain matches the pre-transform state:
    /// no DC step, and the output is exactly the downmix.
    #[test]
    fn transform_off_leaves_segment_empty_and_output_unchanged() {
        let mut chain = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("stereo -> downmix chain");
        assert!(
            chain.transform.is_empty(),
            "enhancement off leaves the transform segment empty"
        );
        let out = chain.run(&[0.5, 0.3, 0.4, 0.6]).expect("downmix runs");
        assert_eq!(
            out,
            sample::downmix_to_mono(&[0.5, 0.3, 0.4, 0.6], 2),
            "with no transform the output is exactly the downmix"
        );
    }

    /// DC-removal correctness: a constant offset settles to a near-zero mean, the
    /// length is preserved exactly (one output sample per input), and the
    /// response is continuous across chunk boundaries: the filter state carries,
    /// so splitting the input into two chunks yields the same samples as one.
    #[test]
    fn dc_removal_removes_offset_preserves_length_and_is_continuous() {
        let on = true;

        // Mono at the target rate: a transform-only chain (just the DC step).
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: on,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc-only chain");
        let n = 16_000;
        let input = vec![0.5_f32; n];
        let out = chain.run(&input).expect("dc runs").to_vec();

        assert_eq!(
            out.len(),
            n,
            "the DC step preserves the sample count exactly"
        );
        // The constant offset settles out: the mean of the last quarter is ~0.
        let settled = &out[n - n / 4..];
        let mean = settled.iter().sum::<f32>() / settled.len() as f32;
        assert!(
            mean.abs() < 1e-3,
            "a known DC offset yields a near-zero-mean output (mean {mean})"
        );

        // Continuity: the same input split across two chunks (state carried)
        // yields identical output, so there is no per-chunk discontinuity.
        let mut split = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: on,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc-only chain");
        let mut combined = split.run(&input[..8_000]).expect("first half").to_vec();
        combined.extend_from_slice(split.run(&input[8_000..]).expect("second half"));
        assert_eq!(
            combined, out,
            "filter state carries across chunks: split processing matches one-shot"
        );
    }

    /// Two-segment run/flush: a downmix + resample + DC chain runs its stages in
    /// order (downmix, then resample, then DC) and, at close, delivers the
    /// resampler's group-delay tail through the DC step. Concatenating the run
    /// output with the flushed tail reproduces, bit for bit, a reference that
    /// downmixes, resamples (process then flush), then DC-blocks the whole signal
    /// with a single continuous filter. That equality holds only if the chain
    /// applies the stages in that order and carries the DC filter state unbroken
    /// across the run/flush boundary. The resampler's no-sample-dropped flush
    /// guarantee still holds with a transform present.
    #[test]
    fn two_segment_run_and_flush_order_and_tail() {
        let on = true;

        // Stereo 48k input: a sine plus a DC offset on both channels (so the
        // downmix keeps the offset for the DC step to remove).
        let frames = 12_000;
        let mut input = Vec::with_capacity(frames * 2);
        for k in 0..frames {
            let s = (k as f32 * 0.01).sin() + 0.5;
            input.push(s);
            input.push(s);
        }

        // Reference: downmix -> resample(process + flush) -> DC over the whole.
        let mono = sample::downmix_to_mono(&input, 2);
        let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut resampled = Vec::new();
        resampler.process(&mono, &mut resampled);
        resampler.flush(&mut resampled);
        let mut dc = DcBlocker::new();
        let mut expected = resampled.clone();
        dc.process_in_place(&mut expected);

        // The chain: process the whole input via run(), then flush() the tail.
        let mut chain = build_capture_stage(
            2,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: on,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("downmix + resample + DC chain");
        let mut got = chain.run(&input).expect("run").to_vec();
        let process_only = got.len();
        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush");
        assert!(
            !tail.is_empty(),
            "the resampler still holds a group-delay tail, now flushed through the DC step"
        );
        got.extend_from_slice(&tail);

        assert_eq!(
            got, expected,
            "downmix -> resample -> DC order, DC state carried across run/flush, bit for bit"
        );
        assert_eq!(
            got.len(),
            process_only + tail.len(),
            "the flushed tail is appended after the run output, nothing reordered"
        );
    }

    /// With no `transform` segment, `run` captures no tap (the delivered output
    /// already is the post-normalize signal), and `has_transform` is false.
    #[test]
    fn run_skips_tap_when_no_transform() {
        let mut chain = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("stereo -> downmix-only chain");
        assert!(
            !chain.has_transform(),
            "a downmix-only chain has no transform"
        );
        let _ = chain.run(&[0.5, 0.3, 0.4, 0.6]).expect("downmix runs");
        assert!(
            chain.tap().is_empty(),
            "no transform: nothing is captured into the tap"
        );
    }

    /// With a `transform` present, `run` captures the post-normalize,
    /// pre-transform signal into the tap. For a downmix + resample + DC chain the
    /// tap equals the downmix+resample output (process path) and the delivered
    /// output equals the DC blocker applied to that tap, so the tap is genuinely
    /// the pre-transform signal and differs from the delivered output.
    #[test]
    fn run_captures_post_normalize_tap() {
        let on = true;
        let mut chain = build_capture_stage(
            2,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: on,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("downmix + resample + DC chain");
        assert!(chain.has_transform());

        // Stereo 48k with a DC offset on both channels.
        let frames = 6_000;
        let mut input = Vec::with_capacity(frames * 2);
        for k in 0..frames {
            let s = (k as f32 * 0.01).sin() + 0.5;
            input.push(s);
            input.push(s);
        }

        let out = chain.run(&input).expect("run").to_vec();
        let tap = chain.tap().to_vec();

        // Reference post-normalize signal: downmix then resample (process only,
        // no flush in run).
        let mono = sample::downmix_to_mono(&input, 2);
        let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut expected_norm = Vec::new();
        resampler.process(&mono, &mut expected_norm);
        assert_eq!(
            tap, expected_norm,
            "the tap is exactly the post-normalize (pre-transform) signal"
        );

        // The delivered output is the DC blocker applied to the tap.
        let mut dc = DcBlocker::new();
        let mut expected_out = tap.clone();
        dc.process_in_place(&mut expected_out);
        assert_eq!(out, expected_out, "delivered output is DC(tap)");
        assert_ne!(
            tap, out,
            "the DC step changes the signal, so tap != delivered"
        );
        assert_eq!(
            tap.len(),
            out.len(),
            "the DC step preserves length, so aligned"
        );
    }

    /// At close, `flush` captures the post-normalize flush tail (the resampler's
    /// group-delay tail) into the tap before it passes through the transform, so
    /// the tap stays aligned with the delivered tail across the close path.
    #[test]
    fn flush_captures_post_normalize_tail() {
        let on = true;
        let mut chain = build_capture_stage(
            1,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: on,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("resample + DC chain");
        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();
        let _ = chain.run(&input).expect("run");
        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush");
        let tap_tail = chain.tap().to_vec();
        assert!(
            !tap_tail.is_empty(),
            "the resampler holds a group-delay tail, captured into the tap"
        );
        assert_eq!(
            tap_tail.len(),
            tail.len(),
            "the tap tail and delivered tail are aligned (DC preserves length)"
        );
        // The delivered tail is DC(tap_tail), continuing the filter state from run.
        assert_ne!(
            tap_tail, tail,
            "the delivered tail is post-transform, the tap tail is pre-transform"
        );
    }

    /// `CaptureStage` is `Send` so it can live behind the stream's `Mutex` and
    /// keep `MicrophoneStream: Send + Sync`. The resampler is `Send`, so a chain
    /// carrying a `ResampleStage` keeps the bound.
    #[test]
    fn capture_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CaptureStage>();
    }

    /// The DC step is `Send`, so a chain carrying it in the `transform` segment
    /// stays `Send` and can live behind the stream's `Mutex`.
    #[test]
    fn dc_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<InPlace<DcBlocker>>();
    }

    // ── Denoise (framed transform) ──────────────────────────────────────
    //
    // These load the bundled FastEnhancer-T model through the real ONNX seam, so
    // they run under `cargo test-decibri` (download-binaries) and are gated on the
    // `denoise` feature. They exercise denoise through the chain build path and
    // the tap, complementing the stage's own unit tests in `crate::denoise`.

    #[cfg(feature = "denoise")]
    fn denoise_model_path() -> std::path::PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("models")
            .join("fastenhancer_t.onnx")
    }

    /// A deterministic mono voiced signal (a single tone); the content does not
    /// matter for these structural tests, only that frames form.
    #[cfg(feature = "denoise")]
    fn mono_signal(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 180.0 * i as f32 / 16_000.0).sin())
            .collect()
    }

    /// The denoise stage is pushed AFTER the DC-removal step. `dc_removal +
    /// denoise` yields a two-stage transform (DC then denoise), denoise alone
    /// yields one transform stage, and denoise off with nothing else stays `None`
    /// (the byte-identical no-op for a mono device already at the target rate).
    /// The relative order (DC before denoise) is pinned here and by the
    /// construction order in `build_capture_stage`.
    #[cfg(feature = "denoise")]
    #[test]
    fn build_orders_denoise_after_dc() {
        let path = denoise_model_path();
        let model = DenoiseModel::FastEnhancerT;

        let both = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: Some((model, path.as_path(), None)),
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc + denoise builds a transform chain");
        assert_eq!(
            both.transform.len(),
            2,
            "dc_removal then denoise: two transform stages"
        );
        assert!(
            both.normalize.is_empty(),
            "a mono device at the target rate needs no normalize stage"
        );

        let denoise_only = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: Some((model, path.as_path(), None)),
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("denoise alone builds a transform chain");
        assert_eq!(
            denoise_only.transform.len(),
            1,
            "denoise alone: one transform stage"
        );

        assert!(
            build_capture_stage(
                1,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: false,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_none(),
            "denoise off and nothing else: no chain, byte-identical passthrough"
        );
    }

    /// Denoise is a framed, latency-introducing transform (unlike the same-length
    /// DcBlocker): running a block yields finite, hop-quantized enhanced output,
    /// `flush` drains a non-empty latency tail (a same-length transform holds
    /// none), and the total delivered (process output plus flushed tail) exceeds
    /// the input by the warm-up and latency the model adds. This proves denoise
    /// runs end to end through the chain and is length-changing overall.
    #[cfg(feature = "denoise")]
    #[test]
    fn denoise_chain_is_framed_with_latency_tail() {
        let path = denoise_model_path();
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("denoise chain");

        let input = mono_signal(8000);
        let out = chain.run(&input).expect("denoise runs").to_vec();
        assert!(!out.is_empty(), "enough input produces enhanced hops");
        assert!(
            out.iter().all(|s| s.is_finite()),
            "enhanced output is finite"
        );
        assert_eq!(out.len() % 256, 0, "output is whole 256-sample hops");

        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush drains the tail");
        assert!(
            !tail.is_empty(),
            "denoise holds a latency tail to flush at close (a same-length \
             transform would have none)"
        );
        assert!(
            out.len() + tail.len() > input.len(),
            "delivered (process + flush) exceeds the input by the warm-up and \
             latency: denoise is length-changing overall"
        );
    }

    /// With denoise in the transform segment, the pre-transform tap LEADS the
    /// delivered enhanced output. The tap is the post-normalize signal at real-
    /// time rate (equal to the mono input length here); the delivered output is
    /// hop-quantized and trails by the buffered partial frame, so the tap leads by
    /// a positive amount bounded by one hop (256 samples) for this non-hop-aligned
    /// input. The tap is non-empty (the detector is still fed). The length-equal
    /// lockstep holds only for same-length transforms (asserted for the DC case in
    /// `microphone`); denoise breaks it, so the tap leads instead of staying
    /// length-aligned.
    #[cfg(feature = "denoise")]
    #[test]
    fn denoise_tap_leads_delivered_by_bounded_amount() {
        let path = denoise_model_path();
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("denoise chain");
        assert!(
            chain.has_transform(),
            "denoise is a transform, so the tap is active"
        );

        // 8000 is not a multiple of the hop (256), so a partial frame stays
        // buffered and the tap strictly leads the delivered output by that
        // remainder.
        let input = mono_signal(8000);
        let out = chain.run(&input).expect("denoise runs").to_vec();
        let tap = chain.tap().to_vec();

        assert_eq!(
            tap.len(),
            input.len(),
            "the tap is the real-time post-normalize signal (input length here)"
        );
        assert!(
            !tap.is_empty(),
            "the tap is non-empty: the detector is still fed"
        );
        assert!(
            out.len() < tap.len(),
            "the tap leads the delivered enhanced output (delivered trails by the \
             buffered partial frame)"
        );
        assert!(
            tap.len() - out.len() <= 256,
            "the per-call lead is bounded by one hop"
        );
    }

    // ── Latency accounting ──────────────────────────────────────────────
    //
    // Each stage declares its constant algorithmic latency; the chain sums the
    // `transform`-segment declarations. The `normalize` segment (downmix,
    // resample) is excluded because it runs before the tap is taken, so its
    // delay reaches the tap and the delivered output alike.

    /// Without a framed transform stage the chain adds no conditioning latency:
    /// the same-length DC blocker contributes zero, and the resampler's group
    /// delay is a `normalize` delay that does not count toward the transform
    /// latency. Both a DC-only chain and a downmix + resample + DC chain report
    /// zero.
    #[test]
    fn transform_latency_is_zero_without_a_framed_stage() {
        let dc_only = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc-only chain");
        assert_eq!(
            dc_only.transform_latency(),
            0,
            "the DC blocker is same-length, so it adds no latency"
        );

        let downmix_resample_dc = build_capture_stage(
            2,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("downmix + resample + DC chain");
        assert_eq!(
            downmix_resample_dc.transform_latency(),
            0,
            "the resampler's group delay is a normalize delay, excluded from the transform latency"
        );
    }

    /// `ResampleStage` forwards the resampler's own group delay (nonzero for a
    /// rate change, zero for an identity passthrough), but because it lives in
    /// the `normalize` segment it never enters the chain's transform latency. A
    /// resample + DC chain therefore carries a resampler latency on the stage yet
    /// reports zero transform latency.
    #[test]
    fn resample_latency_forwards_but_does_not_enter_transform_latency() {
        // The stage reports exactly the resampler's group delay.
        let expected = PolyphaseResampler::new(48_000, 16_000)
            .unwrap()
            .latency_samples();
        assert!(
            expected > 0,
            "downsampling 48k -> 16k has a nonzero group delay"
        );
        let stage = ResampleStage(PolyphaseResampler::new(48_000, 16_000).unwrap());
        assert_eq!(
            stage.latency_samples(),
            expected,
            "the stage forwards the resampler's group delay unchanged"
        );

        // An identity resampler forwards zero.
        let identity = ResampleStage(PolyphaseResampler::new(16_000, 16_000).unwrap());
        assert_eq!(
            identity.latency_samples(),
            0,
            "an identity resampler adds no delay"
        );

        // In a chain, that normalize-segment latency stays out of the transform
        // latency.
        let chain = build_capture_stage(
            1,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("resample + DC chain");
        assert_eq!(
            chain.transform_latency(),
            0,
            "the resampler's group delay does not enter the transform latency"
        );
    }

    /// A chain with the framed denoise stage in its `transform` segment reports
    /// that stage's algorithmic latency: one analysis half-window, 256 samples at
    /// the 16 kHz target rate. This is the same lead the per-call tap test
    /// observes behaviorally; here it is read straight off the chain.
    #[cfg(feature = "denoise")]
    #[test]
    fn transform_latency_reports_denoise_lead() {
        let path = denoise_model_path();
        let chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("denoise chain");
        assert_eq!(
            chain.transform_latency(),
            256,
            "the framed denoise stage adds one half-window (512 - 256) of latency"
        );
    }

    // ── High-pass (same-length biquad transform) ────────────────────────
    //
    // The high-pass is a second-order Butterworth biquad: a same-length,
    // sample-in-sample-out `InPlace` stage (like the DC blocker) that runs after
    // denoise in the transform segment. These cover its magnitude response, its
    // cross-block state continuity, that it builds no stage when off, its chain
    // placement, and its zero latency.

    /// Run a unit sine of `freq_hz` through a `filter`-only high-pass chain and
    /// return the steady-state magnitude (output RMS over the settled second half
    /// divided by the input RMS). Used to read the filter's response at a single
    /// frequency for any cutoff variant.
    fn highpass_gain_at(freq_hz: f32, filter: HighpassFilter) -> f32 {
        let fs = 16_000.0_f32;
        let n = 16_000usize;
        let input: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / fs).sin())
            .collect();
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: Some(filter),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("high-pass-only chain");
        let out = chain.run(&input).expect("high-pass runs").to_vec();
        assert_eq!(
            out.len(),
            input.len(),
            "the biquad is same-length: one output sample per input sample"
        );
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        rms(&out[n / 2..]) / rms(&input[n / 2..])
    }

    /// Filter correctness: the 80 Hz second-order Butterworth high-pass strongly
    /// attenuates content well below the corner, passes the speech band at near
    /// unity, and sits near the -3 dB point at the corner itself. A DC offset
    /// settles to a near-zero mean. This is the one place the coefficient math can
    /// be wrong, so it is checked at three frequencies plus DC.
    #[test]
    fn highpass_attenuates_below_cutoff_and_preserves_passband() {
        // Well below the 80 Hz corner (two octaves down): a second-order section
        // rolls off at ~12 dB/octave, so 20 Hz is ~24 dB down (gain ~0.06).
        let sub = highpass_gain_at(20.0, HighpassFilter::Hz80);
        assert!(
            sub < 0.15,
            "20 Hz (well below the 80 Hz corner) is strongly attenuated (gain {sub})"
        );

        // The speech band passes essentially untouched.
        let pass = highpass_gain_at(1000.0, HighpassFilter::Hz80);
        assert!(
            (0.95..=1.05).contains(&pass),
            "1 kHz (passband) is preserved at near unity (gain {pass})"
        );

        // At the corner the Butterworth response is the -3 dB point (1/sqrt(2)).
        let corner = highpass_gain_at(80.0, HighpassFilter::Hz80);
        assert!(
            (0.60..=0.80).contains(&corner),
            "80 Hz (the corner) sits near the -3 dB Butterworth point of 0.707 (gain {corner})"
        );

        // A pure DC offset is removed: the high-pass settles its output to a
        // near-zero mean.
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("high-pass-only chain");
        let n = 16_000usize;
        let out = chain
            .run(&vec![0.5_f32; n])
            .expect("high-pass runs")
            .to_vec();
        let settled = &out[n - n / 4..];
        let mean = settled.iter().sum::<f32>() / settled.len() as f32;
        assert!(
            mean.abs() < 1e-3,
            "a DC offset settles to a near-zero-mean output (mean {mean})"
        );
    }

    /// Filter correctness for the 100 Hz cutoff member: the 100 Hz second-order
    /// Butterworth high-pass strongly attenuates content well below its corner,
    /// passes the speech band at near unity, and sits near the -3 dB point at the
    /// 100 Hz corner itself. Mirrors the 80 Hz test, confirming the second
    /// cutoff variant routes its own frequency through the shared biquad design.
    #[test]
    fn highpass_100hz_attenuates_below_cutoff_and_preserves_passband() {
        // Well below the 100 Hz corner (two octaves down): the second-order
        // section rolls off at ~12 dB/octave, so 25 Hz is ~24 dB down (gain
        // ~0.06).
        let sub = highpass_gain_at(25.0, HighpassFilter::Hz100);
        assert!(
            sub < 0.15,
            "25 Hz (well below the 100 Hz corner) is strongly attenuated (gain {sub})"
        );

        // The speech band passes essentially untouched.
        let pass = highpass_gain_at(1000.0, HighpassFilter::Hz100);
        assert!(
            (0.95..=1.05).contains(&pass),
            "1 kHz (passband) is preserved at near unity (gain {pass})"
        );

        // At the corner the Butterworth response is the -3 dB point (1/sqrt(2)).
        let corner = highpass_gain_at(100.0, HighpassFilter::Hz100);
        assert!(
            (0.60..=0.80).contains(&corner),
            "100 Hz (the corner) sits near the -3 dB Butterworth point of 0.707 (gain {corner})"
        );

        // The 100 Hz cut is more aggressive than the 80 Hz cut at the same
        // sub-band frequency: at 80 Hz, the 100 Hz filter (still below its own
        // corner) attenuates more than the 80 Hz filter (at its corner).
        let at_80_with_100 = highpass_gain_at(80.0, HighpassFilter::Hz100);
        let at_80_with_80 = highpass_gain_at(80.0, HighpassFilter::Hz80);
        assert!(
            at_80_with_100 < at_80_with_80,
            "the 100 Hz cutoff attenuates 80 Hz more than the 80 Hz cutoff does \
             ({at_80_with_100} < {at_80_with_80})"
        );
    }

    /// State continuity: the biquad carries its two state words across blocks, so
    /// the same signal processed in one call equals it processed in irregular
    /// chunks, bit for bit. Without the carry the chunk boundaries would show a
    /// discontinuity.
    #[test]
    fn highpass_state_carries_across_blocks() {
        let fs = 16_000.0_f32;
        let n = 4096usize;
        // A 120 Hz tone over a DC offset, so both the filtered tone and the
        // settling DC exercise the state carry.
        let input: Vec<f32> = (0..n)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 120.0 * i as f32 / fs).sin() + 0.3)
            .collect();

        let one_shot = {
            let mut chain = build_capture_stage(
                1,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: false,
                    denoise: None,
                    highpass: Some(HighpassFilter::Hz80),
                    agc: None,
                    limiter: None,
                },
            )
            .unwrap()
            .expect("high-pass-only chain");
            chain.run(&input).expect("one-shot").to_vec()
        };

        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("high-pass-only chain");
        let mut chunked = Vec::with_capacity(n);
        // Irregular block boundaries, including a single-sample block.
        for chunk in [
            &input[..1000],
            &input[1000..1001],
            &input[1001..3333],
            &input[3333..],
        ] {
            chunked.extend_from_slice(chain.run(chunk).expect("chunk"));
        }

        assert_eq!(
            chunked, one_shot,
            "biquad state carries across block boundaries: chunked equals one-shot"
        );
    }

    /// Off is a true no-op: with `highpass` `None` and nothing else, the chain is
    /// not built at all (byte-identical passthrough); a downmix-only chain with
    /// high-pass off has an empty transform segment (no transparent filter); and
    /// turning it on pushes exactly one transform stage.
    #[test]
    fn highpass_off_builds_no_stage() {
        assert!(
            build_capture_stage(
                1,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: false,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_none(),
            "high-pass off and nothing else: no chain, byte-identical passthrough"
        );

        let downmix_only = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("stereo -> downmix chain");
        assert!(
            downmix_only.transform.is_empty(),
            "high-pass off pushes no transform stage, not a transparent filter"
        );

        let hp = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("high-pass-only chain");
        assert!(
            hp.normalize.is_empty(),
            "a mono device at the target rate needs no normalize stage"
        );
        assert_eq!(
            hp.transform.len(),
            1,
            "high-pass on pushes exactly one transform stage"
        );
    }

    /// Chain placement (feature-independent): with both DC removal and high-pass
    /// on, the transform segment is [DcBlocker, Biquad], so high-pass runs after
    /// DC removal. The relative order is pinned by the push order in
    /// `build_capture_stage` (DC, then denoise, then high-pass).
    #[test]
    fn build_orders_highpass_after_dc() {
        let chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc + high-pass chain");
        assert_eq!(
            chain.transform.len(),
            2,
            "dc_removal then high-pass: two transform stages, high-pass last"
        );
    }

    /// High-pass adds no transform latency: it is a same-length biquad, so it
    /// contributes zero to `transform_latency()`, on its own and stacked with the
    /// (also same-length) DC blocker. The latency seam is unchanged by high-pass.
    #[test]
    fn highpass_adds_no_transform_latency() {
        let hp = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("high-pass-only chain");
        assert_eq!(
            hp.transform_latency(),
            0,
            "the high-pass biquad is same-length, so it adds no latency"
        );

        let dc_and_hp = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc + high-pass chain");
        assert_eq!(
            dc_and_hp.transform_latency(),
            0,
            "DC removal and high-pass are both same-length: zero transform latency"
        );
    }

    /// `InPlace<Biquad>` is `Send`, so a chain carrying the high-pass in its
    /// `transform` segment stays `Send` and can live behind the stream's `Mutex`,
    /// matching the DC blocker.
    #[test]
    fn highpass_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<InPlace<Biquad>>();
    }

    /// Chain placement with denoise present: DC removal, then the framed denoise
    /// stage, then high-pass, so the transform segment holds three stages and
    /// high-pass sits AFTER denoise, so the model receives near-full-band input.
    /// The total transform latency is the denoise lead alone (256), which
    /// also proves the same-length high-pass adds none even sitting after the
    /// framed stage. The relative order is pinned here and by the push order in
    /// `build_capture_stage`.
    #[cfg(feature = "denoise")]
    #[test]
    fn build_orders_highpass_after_denoise() {
        let path = denoise_model_path();
        let chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                highpass: Some(HighpassFilter::Hz80),
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc + denoise + high-pass chain");
        assert_eq!(
            chain.transform.len(),
            3,
            "dc_removal, denoise, then high-pass: three transform stages"
        );
        assert_eq!(
            chain.transform_latency(),
            256,
            "high-pass after denoise adds no latency: the 256 is the denoise lead alone"
        );
    }

    // ── Level control (AGC, same-length transform) ──────────────────────
    //
    // AGC is the LevelControl engine driven through `InPlace`, a same-length,
    // sample-in-sample-out stage like the DC blocker and high-pass. These cover
    // its chain placement (after high-pass), that it builds no stage when off,
    // that it adds no latency, and that it conditions delivered audio. The
    // engine's temporal behaviour (cold start, convergence, gating) is covered by
    // the unit tests in `crate::gain`. Gated on the `gain` feature, which owns the
    // engine and the chain push.

    /// Off is a true no-op: with `agc` `None` and nothing else, the chain is not
    /// built at all (byte-identical passthrough); a downmix-only chain with AGC
    /// off has an empty transform segment (no transparent stage); turning it on
    /// pushes exactly one transform stage.
    #[cfg(feature = "gain")]
    #[test]
    fn agc_off_builds_no_stage() {
        assert!(
            build_capture_stage(
                1,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: false,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None,
                }
            )
            .unwrap()
            .is_none(),
            "AGC off and nothing else: no chain, byte-identical passthrough"
        );

        let downmix_only = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("stereo -> downmix chain");
        assert!(
            downmix_only.transform.is_empty(),
            "AGC off pushes no transform stage, not a transparent stage"
        );

        let agc = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: Some(-18),
                limiter: None,
            },
        )
        .unwrap()
        .expect("AGC-only chain");
        assert!(
            agc.normalize.is_empty(),
            "a mono device at the target rate needs no normalize stage"
        );
        assert_eq!(
            agc.transform.len(),
            1,
            "AGC on pushes exactly one transform stage"
        );
    }

    /// Chain placement: with DC removal, high-pass, and AGC all on, the transform
    /// segment is [DcBlocker, Biquad, LevelControl], so AGC runs last, after
    /// high-pass, reserving the slot before the (not-yet-built) limiter. The
    /// relative order is pinned by the push order in `build_capture_stage`.
    #[cfg(feature = "gain")]
    #[test]
    fn build_orders_agc_after_highpass() {
        let chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc + high-pass + AGC chain");
        assert_eq!(
            chain.transform.len(),
            3,
            "dc_removal, high-pass, then AGC: three transform stages, AGC last"
        );
    }

    /// AGC adds no transform latency: it is a same-length, feedback (no
    /// look-ahead) engine, so it contributes zero to `transform_latency()`, on its
    /// own and stacked with the same-length DC blocker and high-pass. The latency
    /// seam and the VAD-tap invariant are unchanged by AGC.
    #[cfg(feature = "gain")]
    #[test]
    fn agc_adds_no_transform_latency() {
        let agc = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: Some(-18),
                limiter: None,
            },
        )
        .unwrap()
        .expect("AGC-only chain");
        assert_eq!(
            agc.transform_latency(),
            0,
            "the level-control engine is feedback with no look-ahead: zero latency"
        );

        let stacked = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                limiter: None,
            },
        )
        .unwrap()
        .expect("dc + high-pass + AGC chain");
        assert_eq!(
            stacked.transform_latency(),
            0,
            "DC, high-pass, and AGC are all same-length: zero transform latency"
        );
    }

    /// AGC conditions delivered audio through the chain: a quiet but present tone
    /// (below the target, above the noise floor) run through an AGC-only chain is
    /// boosted toward the target, so its settled output level exceeds its input
    /// level. This proves the engine is wired into the chain's run path; its
    /// detailed trajectory is covered in `crate::gain`.
    #[cfg(feature = "gain")]
    #[test]
    fn agc_conditions_audio_through_the_chain() {
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: Some(-18),
                limiter: None,
            },
        )
        .unwrap()
        .expect("AGC-only chain");
        // Half a second of a quiet -30 dBFS tone, above the -50 dBFS noise floor.
        let amp = 0.0316_f32 * std::f32::consts::SQRT_2; // -30 dBFS RMS sine
        let input: Vec<f32> = (0..8000)
            .map(|i| amp * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / 16_000.0).sin())
            .collect();
        let out = chain.run(&input).expect("AGC runs").to_vec();
        assert_eq!(out.len(), input.len(), "AGC is same-length");
        let rms = |s: &[f32]| (s.iter().map(|v| v * v).sum::<f32>() / s.len() as f32).sqrt();
        let in_rms = rms(&input[input.len() - 1600..]);
        let out_rms = rms(&out[out.len() - 1600..]);
        assert!(
            out_rms > in_rms * 2.0,
            "AGC boosts the quiet input toward target (in {in_rms}, out {out_rms})"
        );
    }

    /// `InPlace<LevelControl>` is `Send`, so a chain carrying AGC in its
    /// `transform` segment stays `Send` and can live behind the stream's `Mutex`,
    /// matching the DC blocker and high-pass.
    #[cfg(feature = "gain")]
    #[test]
    fn agc_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<InPlace<crate::gain::LevelControl>>();
    }

    // ── Limiter (same-length transform, last in the chain) ──────────────
    //
    // The limiter is the Limiter stage driven through `InPlace`, a same-length,
    // sample-in-sample-out stage like AGC. These cover its chain placement (last,
    // after AGC), that it builds no stage when off, that it adds no latency, that
    // it caps delivered audio at the ceiling end to end, and that it builds
    // standalone (with AGC off). The stage's own guarantees (the absolute ceiling,
    // graceful shaping, release) are covered by the unit tests in `crate::gain`.
    // Gated on the `gain` feature, which owns the stage and the chain push.

    /// Off is a true no-op: with `limiter` `None` and nothing else, the chain is
    /// not built at all (byte-identical passthrough); a downmix-only chain with
    /// the limiter off has an empty transform segment (no transparent stage);
    /// turning it on pushes exactly one transform stage.
    #[cfg(feature = "gain")]
    #[test]
    fn limiter_off_builds_no_stage() {
        assert!(
            build_capture_stage(
                1,
                1,
                16_000,
                16_000,
                Transforms {
                    dc_removal: false,
                    denoise: None,
                    highpass: None,
                    agc: None,
                    limiter: None
                }
            )
            .unwrap()
            .is_none(),
            "limiter off and nothing else: no chain, byte-identical passthrough"
        );

        let downmix_only = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
            },
        )
        .unwrap()
        .expect("stereo -> downmix chain");
        assert!(
            downmix_only.transform.is_empty(),
            "limiter off pushes no transform stage, not a transparent stage"
        );

        let limiter = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: Some(-1.0),
            },
        )
        .unwrap()
        .expect("limiter-only chain");
        assert!(
            limiter.normalize.is_empty(),
            "a mono device at the target rate needs no normalize stage"
        );
        assert_eq!(
            limiter.transform.len(),
            1,
            "limiter on pushes exactly one transform stage, standalone (AGC off)"
        );
    }

    /// Chain placement: with DC removal, high-pass, AGC, and the limiter all on,
    /// the transform segment is [DcBlocker, Biquad, LevelControl, Limiter], so the
    /// limiter runs LAST, after AGC, with nothing after it. The relative order is
    /// pinned by the push order in `build_capture_stage`.
    #[cfg(feature = "gain")]
    #[test]
    fn build_orders_limiter_after_agc() {
        let chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                limiter: Some(-1.0),
            },
        )
        .unwrap()
        .expect("dc + high-pass + AGC + limiter chain");
        assert_eq!(
            chain.transform.len(),
            4,
            "dc_removal, high-pass, AGC, then limiter: four transform stages, limiter last"
        );
    }

    /// The limiter adds no transform latency: it is a same-length, feedback (no
    /// look-ahead) stage, so it contributes zero to `transform_latency()`, on its
    /// own and stacked with the same-length DC blocker, high-pass, and AGC. The
    /// latency seam and the VAD-tap invariant are unchanged by the limiter.
    #[cfg(feature = "gain")]
    #[test]
    fn limiter_adds_no_transform_latency() {
        let limiter = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: Some(-1.0),
            },
        )
        .unwrap()
        .expect("limiter-only chain");
        assert_eq!(
            limiter.transform_latency(),
            0,
            "the limiter is feedback with no look-ahead: zero latency"
        );

        let stacked = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                limiter: Some(-1.0),
            },
        )
        .unwrap()
        .expect("dc + high-pass + AGC + limiter chain");
        assert_eq!(
            stacked.transform_latency(),
            0,
            "DC, high-pass, AGC, and the limiter are all same-length: zero transform latency"
        );
    }

    /// The limiter caps delivered audio at the ceiling end to end through the
    /// chain's run path: a hot signal with instantaneous full-scale transients run
    /// through a limiter-only chain produces no output sample above the linear
    /// ceiling. This is the absolute guarantee observed at the chain level; the
    /// stage's own per-sample proof is in `crate::gain`.
    #[cfg(feature = "gain")]
    #[test]
    fn limiter_caps_audio_through_the_chain() {
        let ceiling_db = -1.0_f32;
        let ceiling = 10.0_f32.powf(ceiling_db / 20.0);
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: false,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: Some(ceiling_db),
            },
        )
        .unwrap()
        .expect("limiter-only chain");
        // A loud sine with a full-scale spike on the first sample and a burst
        // above full scale partway through.
        let mut input: Vec<f32> = (0..8000)
            .map(|i| 0.95 * (2.0 * std::f32::consts::PI * 1_000.0 * i as f32 / 16_000.0).sin())
            .collect();
        input[0] = 1.0;
        input[4000] = 2.0;
        input[4001] = -2.0;
        let out = chain.run(&input).expect("limiter runs").to_vec();
        assert_eq!(out.len(), input.len(), "the limiter is same-length");
        let worst = out.iter().cloned().fold(0.0_f32, |m, s| m.max(s.abs()));
        assert!(
            worst <= ceiling,
            "no delivered sample exceeds the ceiling end to end ({worst} > {ceiling})"
        );
    }

    /// `InPlace<Limiter>` is `Send`, so a chain carrying the limiter in its
    /// `transform` segment stays `Send` and can live behind the stream's `Mutex`,
    /// matching the DC blocker, high-pass, and AGC.
    #[cfg(feature = "gain")]
    #[test]
    fn limiter_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<InPlace<crate::gain::Limiter>>();
    }
}
