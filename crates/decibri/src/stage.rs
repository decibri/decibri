//! Internal capture stage chain.
//!
//! A [`CaptureStage`] conditions raw device capture into the canonical format the
//! consumer receives, running between the cpal callback's native buffers and the
//! exact-size reblock in [`crate::microphone`]. The chain has two segments. The
//! `normalize` segment holds up to two stages: [`Downmix`], which averages a
//! multichannel device down to mono, then [`ResampleStage`], which converts the
//! device's native sample rate to the requested target rate. Downmix runs first
//! so the resampler receives mono, the format it expects. The optional
//! `transform` segment runs after `normalize`, holding the same-length
//! conditioning step a consumer opts into (the [`DcBlocker`] DC-removal step).
//!
//! Each [`Stage`] reads one block of interleaved f32 samples and writes its
//! output, so stages compose by ping-ponging two buffers. At stream close the
//! chain drains each stage's held tail through the same walk, so the resampler's
//! group-delay tail is delivered rather than dropped. The chain is built by
//! [`build_capture_stage`], which returns `None` when no conditioning is needed
//! (an already-mono device already at the target rate with no enhancement
//! enabled), keeping the capture path on its zero-cost direct path.

use decibri_resampler::{PolyphaseResampler, Resampler};

use crate::error::DecibriError;
use crate::microphone::EnhancementConfig;
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
}

/// A same-length DSP step: filters a buffer of samples in place, producing
/// exactly as many output samples as it received. The adapter [`InPlace`] wraps
/// any `InPlaceDsp` as a [`Stage`].
///
/// `Send` so the wrapped step can live behind the stream's `Mutex`, matching the
/// [`Stage`] bound.
trait InPlaceDsp: Send {
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

/// The capture stage chain: a `normalize` segment (downmix, resample) that
/// conditions a device to the output format, then an optional `transform`
/// segment (the DC-removal step) applied to the normalized signal.
///
/// Invariant: at least one segment is non-empty. [`build_capture_stage`] is the
/// only constructor and returns `None` rather than an empty chain when both
/// segments are empty, so the capture path can skip the chain entirely when no
/// conditioning is needed.
pub(crate) struct CaptureStage {
    /// Channel and rate normalization stages (downmix, then resample), applied
    /// first.
    normalize: Vec<Box<dyn Stage>>,
    /// Optional same-length conditioning stages (the DC-removal step), applied
    /// after `normalize`.
    transform: Vec<Box<dyn Stage>>,
    /// Holds the current block as it passes through the chain; [`run`](Self::run)
    /// returns a borrow of this after the last stage.
    work: Vec<f32>,
    /// The ping-pong partner of `work`: each stage reads one and writes the other.
    scratch: Vec<f32>,
}

impl CaptureStage {
    /// Run the chain on one native block, returning the conditioned output.
    ///
    /// Seeds `work` with `input`, then ping-pongs `work` <-> `scratch` first
    /// across the `normalize` stages and then across the `transform` stages,
    /// leaving the final output in `work`.
    pub(crate) fn run(&mut self, input: &[f32]) -> Result<&[f32], DecibriError> {
        self.work.clear();
        self.work.extend_from_slice(input);
        Self::run_segment(&mut self.work, &mut self.scratch, &mut self.normalize)?;
        Self::run_segment(&mut self.work, &mut self.scratch, &mut self.transform)?;
        Ok(&self.work)
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
        Self::flush_segment(&mut self.work, &mut self.scratch, &mut self.transform)?;
        out.extend_from_slice(&self.work);
        Ok(())
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

/// Build the capture stage chain that normalizes a device to the output format
/// and applies any opt-in enhancement.
///
/// Pushes [`Downmix`] when the device delivers more channels than the output
/// target (averaging down to the target, which is mono here), then
/// [`ResampleStage`] when the device's `native_rate` differs from `target_rate`
/// (converting the captured audio to the requested rate). When `enhancement`
/// enables the DC-removal step, pushes the [`DcBlocker`] into the `transform`
/// segment, which runs after `normalize` on the mono signal at the target rate.
/// Returns `Some(chain)` when at least one stage is needed and `None` when no
/// segment has any (a mono device already at the target rate with no enhancement
/// enabled), leaving the capture path on its direct, zero-cost reblock.
///
/// Returns an error when the resampler rejects the rate pair at construction;
/// every in-range configured rate over any device-native rate constructs, so
/// this is reached only for an out-of-range rate (surfaced defensively).
pub(crate) fn build_capture_stage(
    device_channels: u16,
    target_channels: u16,
    native_rate: u32,
    target_rate: u32,
    enhancement: &EnhancementConfig,
) -> Result<Option<CaptureStage>, DecibriError> {
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

    if enhancement.dc_removal {
        // Runs after `normalize`, on the mono signal at the target rate.
        transform.push(Box::new(InPlace(DcBlocker::new())));
    }

    Ok(if normalize.is_empty() && transform.is_empty() {
        None
    } else {
        Some(CaptureStage {
            normalize,
            transform,
            work: Vec::new(),
            scratch: Vec::new(),
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
        let off = EnhancementConfig::default();
        // Mono, native == target: nothing to normalize.
        assert!(
            build_capture_stage(1, 1, 16_000, 16_000, &off)
                .unwrap()
                .is_none(),
            "mono device at the target rate needs no chain"
        );
        // Multichannel, native == target: downmix only.
        assert!(
            build_capture_stage(2, 1, 16_000, 16_000, &off)
                .unwrap()
                .is_some(),
            "stereo device gets a chain (downmix)"
        );
        assert!(
            build_capture_stage(6, 1, 16_000, 16_000, &off)
                .unwrap()
                .is_some(),
            "5.1 device gets a chain (downmix)"
        );
        // Mono, native != target: resample only.
        assert!(
            build_capture_stage(1, 1, 48_000, 16_000, &off)
                .unwrap()
                .is_some(),
            "mono device above the target rate gets a chain (resample)"
        );
        // Multichannel, native != target: downmix then resample.
        assert!(
            build_capture_stage(2, 1, 48_000, 16_000, &off)
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
        let on = EnhancementConfig { dc_removal: true };

        // Mono at target, enhancement on: a transform-only chain (no normalize).
        let chain = build_capture_stage(1, 1, 16_000, 16_000, &on)
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
        let full = build_capture_stage(2, 1, 48_000, 16_000, &on)
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
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000, &EnhancementConfig::default())
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
        let mut chain = build_capture_stage(1, 1, 48_000, 16_000, &EnhancementConfig::default())
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
        let result = build_capture_stage(1, 1, 316_800_000, 16_000, &EnhancementConfig::default());
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
        let mut chain = build_capture_stage(1, 1, 48_000, 16_000, &EnhancementConfig::default())
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
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000, &EnhancementConfig::default())
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
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000, &EnhancementConfig::default())
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
        let on = EnhancementConfig { dc_removal: true };

        // Mono at the target rate: a transform-only chain (just the DC step).
        let mut chain = build_capture_stage(1, 1, 16_000, 16_000, &on)
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
        let mut split = build_capture_stage(1, 1, 16_000, 16_000, &on)
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
        let on = EnhancementConfig { dc_removal: true };

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
        let mut chain = build_capture_stage(2, 1, 48_000, 16_000, &on)
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
}
