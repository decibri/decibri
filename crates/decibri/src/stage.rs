//! Internal capture stage chain.
//!
//! A [`CaptureStage`] conditions raw device capture into the canonical format the
//! consumer receives, running between the cpal callback's native buffers and the
//! exact-size reblock in [`crate::microphone`]. The chain has one segment,
//! `normalize`, holding up to two stages: [`Downmix`], which averages a
//! multichannel device down to mono, then [`ResampleStage`], which converts the
//! device's native sample rate to the requested target rate. Downmix runs first
//! so the resampler receives mono, the format it expects.
//!
//! Each [`Stage`] reads one block of interleaved f32 samples and writes its
//! output, so stages compose by ping-ponging two buffers. At stream close the
//! chain drains each stage's held tail through the same walk, so the resampler's
//! group-delay tail is delivered rather than dropped. The chain is built by
//! [`build_capture_stage`], which returns `None` when no conditioning is needed
//! (an already-mono device already at the target rate), keeping the capture path
//! on its zero-cost direct path.

use decibri_resampler::{PolyphaseResampler, Resampler};

use crate::error::DecibriError;
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

/// The capture stage chain: the `normalize` segment applied to each native
/// capture block before exact-size delivery.
///
/// Invariant: `normalize` is never empty. [`build_capture_stage`] is the only
/// constructor and returns `None` rather than an empty chain, so the capture path
/// can skip the chain entirely when no conditioning is needed.
pub(crate) struct CaptureStage {
    normalize: Vec<Box<dyn Stage>>,
    /// Holds the current block as it passes through the chain; [`run`](Self::run)
    /// returns a borrow of this after the last stage.
    work: Vec<f32>,
    /// The ping-pong partner of `work`: each stage reads one and writes the other.
    scratch: Vec<f32>,
}

impl CaptureStage {
    /// Run the chain on one native block, returning the conditioned output.
    ///
    /// Seeds `work` with `input`, then ping-pongs `work` <-> `scratch` across the
    /// `normalize` stages, leaving the final output in `work`.
    pub(crate) fn run(&mut self, input: &[f32]) -> Result<&[f32], DecibriError> {
        self.work.clear();
        self.work.extend_from_slice(input);
        for stage in &mut self.normalize {
            self.scratch.clear();
            stage.process(&self.work, &mut self.scratch)?;
            std::mem::swap(&mut self.work, &mut self.scratch);
        }
        Ok(&self.work)
    }

    /// Drain the chain's complete end-of-stream tail, appending it to `out`.
    ///
    /// Walks the `normalize` stages in order carrying a buffer: at each stage it
    /// processes the upstream-carried tail (when non-empty) through the stage,
    /// then drains that stage's own held tail into the same buffer, and passes the
    /// combined result to the next stage. So a tail produced by one stage flows
    /// through every downstream stage exactly as a normal block would. For the
    /// stateless [`Downmix`] this contributes nothing; for [`ResampleStage`] it
    /// yields the resampler's group-delay tail. Call once at stream close, after
    /// the last [`run`](Self::run).
    pub(crate) fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // `work` carries the inter-stage buffer. It starts empty because there is
        // no new input at end of stream, only each stage's drained tail.
        self.work.clear();
        for stage in &mut self.normalize {
            self.scratch.clear();
            // Feed the upstream stage's tail through this stage first (skipped for
            // the first stage, whose carry is always empty), then drain this
            // stage's own tail into the same buffer.
            if !self.work.is_empty() {
                stage.process(&self.work, &mut self.scratch)?;
            }
            stage.flush(&mut self.scratch)?;
            std::mem::swap(&mut self.work, &mut self.scratch);
        }
        out.extend_from_slice(&self.work);
        Ok(())
    }
}

/// Build the capture stage chain that normalizes a device to the output format.
///
/// Pushes [`Downmix`] when the device delivers more channels than the output
/// target (averaging down to the target, which is mono here), then
/// [`ResampleStage`] when the device's `native_rate` differs from `target_rate`
/// (converting the captured audio to the requested rate). Returns `Some(chain)`
/// when at least one stage is needed and `None` when neither is (a mono device
/// already at the target rate), leaving the capture path on its direct,
/// zero-cost reblock.
///
/// Returns an error when the resampler rejects the rate pair at construction;
/// every in-range configured rate over any device-native rate constructs, so
/// this is reached only for an out-of-range rate (surfaced defensively).
pub(crate) fn build_capture_stage(
    device_channels: u16,
    target_channels: u16,
    native_rate: u32,
    target_rate: u32,
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

    Ok(if normalize.is_empty() {
        None
    } else {
        Some(CaptureStage {
            normalize,
            work: Vec::new(),
            scratch: Vec::new(),
        })
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_capture_stage` returns `None` only for a mono device already at
    /// the target rate, and `Some` once a downmix and/or a resample is needed.
    #[test]
    fn build_returns_none_only_for_mono_at_target_rate() {
        // Mono, native == target: nothing to normalize.
        assert!(
            build_capture_stage(1, 1, 16_000, 16_000).unwrap().is_none(),
            "mono device at the target rate needs no chain"
        );
        // Multichannel, native == target: downmix only.
        assert!(
            build_capture_stage(2, 1, 16_000, 16_000).unwrap().is_some(),
            "stereo device gets a chain (downmix)"
        );
        assert!(
            build_capture_stage(6, 1, 16_000, 16_000).unwrap().is_some(),
            "5.1 device gets a chain (downmix)"
        );
        // Mono, native != target: resample only.
        assert!(
            build_capture_stage(1, 1, 48_000, 16_000).unwrap().is_some(),
            "mono device above the target rate gets a chain (resample)"
        );
        // Multichannel, native != target: downmix then resample.
        assert!(
            build_capture_stage(2, 1, 48_000, 16_000).unwrap().is_some(),
            "stereo device above the target rate gets a chain (downmix + resample)"
        );
    }

    /// The downmix chain averages interleaved frames to mono, reproducing
    /// `sample::downmix_to_mono` exactly (sample-identity of the downmix). A
    /// device at the target rate adds no resample stage, so the chain is the
    /// pure downmix.
    #[test]
    fn downmix_chain_averages_to_mono() {
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000)
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
        let mut chain = build_capture_stage(1, 1, 48_000, 16_000)
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
        let result = build_capture_stage(1, 1, 316_800_000, 16_000);
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
        let mut chain = build_capture_stage(1, 1, 48_000, 16_000)
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
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000)
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

    /// `CaptureStage` is `Send` so it can live behind the stream's `Mutex` and
    /// keep `MicrophoneStream: Send + Sync`. The resampler is `Send`, so a chain
    /// carrying a `ResampleStage` keeps the bound.
    #[test]
    fn capture_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CaptureStage>();
    }
}
