//! Internal capture stage chain.
//!
//! A [`CaptureStage`] conditions raw device capture into the canonical format the
//! consumer receives, running between the cpal callback's native buffers and the
//! exact-size reblock in [`crate::microphone`]. A1's chain has one segment,
//! `normalize`, which currently holds at most one stage: [`Downmix`], which
//! averages a multichannel device down to mono.
//!
//! Each [`Stage`] reads one block of interleaved f32 samples and writes its
//! output, so stages compose by ping-ponging two buffers. The chain is built by
//! [`build_capture_stage`], which returns `None` when no conditioning is needed
//! (an already-mono device), keeping the capture path on its zero-cost direct
//! path.

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
}

/// Build the capture stage chain for a device-to-output channel transition.
///
/// Returns `Some([Downmix])` when the device delivers more channels than the
/// output target (averaging down to the target, which is mono here), and `None`
/// when no conditioning is needed (the device already matches the target). A
/// `None` chain leaves the capture path on its direct, zero-cost reblock.
pub(crate) fn build_capture_stage(
    device_channels: u16,
    target_channels: u16,
) -> Option<CaptureStage> {
    let mut normalize: Vec<Box<dyn Stage>> = Vec::new();

    if device_channels > target_channels {
        normalize.push(Box::new(Downmix::new(device_channels)));
    }

    if normalize.is_empty() {
        None
    } else {
        Some(CaptureStage {
            normalize,
            work: Vec::new(),
            scratch: Vec::new(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `build_capture_stage` returns `None` for an already-mono device (nothing
    /// to normalize) and `Some` once the device is multichannel.
    #[test]
    fn build_returns_none_for_mono_and_some_for_multichannel() {
        assert!(
            build_capture_stage(1, 1).is_none(),
            "mono device needs no chain"
        );
        assert!(
            build_capture_stage(2, 1).is_some(),
            "stereo device gets a downmix chain"
        );
        assert!(
            build_capture_stage(6, 1).is_some(),
            "5.1 device gets a downmix chain"
        );
    }

    /// The downmix chain averages interleaved frames to mono, reproducing
    /// `sample::downmix_to_mono` exactly (sample-identity of the downmix).
    #[test]
    fn downmix_chain_averages_to_mono() {
        let mut chain = build_capture_stage(2, 1).expect("stereo -> downmix chain");
        // Two stereo frames: (0.5, 0.3) -> 0.4, (0.4, 0.6) -> 0.5.
        let out = chain.run(&[0.5, 0.3, 0.4, 0.6]).expect("downmix runs");
        assert_eq!(out.len(), 2, "stereo input halves to mono");
        assert!((out[0] - 0.4).abs() < 1e-6);
        assert!((out[1] - 0.5).abs() < 1e-6);
        // Reusing sample::downmix_to_mono guarantees identical math.
        assert_eq!(out, sample::downmix_to_mono(&[0.5, 0.3, 0.4, 0.6], 2));
    }

    /// `CaptureStage` is `Send` so it can live behind the stream's `Mutex` and
    /// keep `MicrophoneStream: Send + Sync`.
    #[test]
    fn capture_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<CaptureStage>();
    }
}
