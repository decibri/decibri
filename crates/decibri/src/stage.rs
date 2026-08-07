//! Internal capture stage chain.
//!
//! A [`CaptureStage`] conditions raw device capture into the canonical format the
//! consumer receives, running between the cpal callback's native buffers and the
//! exact-size reblock in [`crate::microphone`]. The chain has two segments. The
//! `normalize` segment holds up to three stages: [`Downmix`], which averages a
//! multichannel device down to mono, then [`ResampleStage`], which converts the
//! device's native sample rate to the requested target rate, then the echo
//! canceller, which removes the echo of caller-supplied far-end audio. Downmix
//! runs first so the resampler receives mono, the format it expects; the echo
//! canceller runs last so it receives mono at the target rate, the only format
//! it accepts. The optional `transform` segment runs after `normalize`, holding
//! the conditioning a consumer opts into: the same-length [`DcBlocker`]
//! DC-removal step, then the framed, length-changing denoise stage. A
//! `transform` stage need not preserve length (denoise re-blocks and introduces
//! latency), which is why the VAD tap is snapshotted before this segment runs.
//!
//! Each [`Stage`] reads one block of interleaved f32 samples and writes its
//! output, so stages compose by ping-ponging two buffers. At stream close the
//! chain drains each stage's held tail through the same walk, so the resampler's
//! group-delay tail is delivered rather than dropped. The chain is built by
//! [`build_capture_stage`], which returns `None` when no conditioning is needed
//! (an already-mono device already at the target rate with no enhancement
//! enabled), keeping the capture path on its zero-cost direct path.
//!
//! # Channel layout
//!
//! A block crossing a stage boundary is INTERLEAVED and carries the channel
//! count it is interleaved at, as a [`Block`]. Inside a stage that reads across
//! channels the samples are PLANAR: one contiguous run per channel,
//! deinterleaved once at the head of the chain, at the position [`Downmix`]
//! occupies, and re-interleaved at delivery. Interleaved at the boundaries and
//! planar inside is the single arrangement this chain uses. A stage that picks
//! the other one at either point is what this convention exists to rule out,
//! because two stages disagreeing about the layout of the buffer between them
//! is not a difference the type system catches.
//!
//! Lengths that count frames are named `frames`; lengths that count interleaved
//! samples are named `samples`. The two are equal at one channel, which is the
//! reason a name is worth carrying rather than inferring.
//!
//! No stage, and no part of the chain walk, names an upper bound on the channel
//! count. The count is a `u16` because that is the width cpal, WAVE and AIFF all
//! carry it in; the only ceiling the chain may enforce is the one a device
//! reports for itself.

#[cfg(feature = "aec")]
use std::collections::VecDeque;
use std::path::Path;
#[cfg(feature = "aec")]
use std::sync::atomic::{AtomicU64, Ordering};
#[cfg(feature = "aec")]
use std::sync::{Arc, Mutex, PoisonError};

#[cfg(feature = "aec")]
use decibri_aec::{Aec, AecConfig, AecMetrics, AecModel, Suppression};
use decibri_resampler::{PolyphaseResampler, Resampler};

use crate::error::DecibriError;
use crate::microphone::{DenoiseModel, HighpassFilter};
use crate::sample;

/// One block of interleaved f32 samples together with the channel count they
/// are interleaved at.
///
/// The descriptor exists so a stage reads its input's channel count from the
/// block rather than from a value it was constructed with or a count it
/// assumes. A stage that requires mono says so against this count (see the
/// `debug_assert_eq!` at the head of each such stage) instead of being handed a
/// bare slice whose layout it cannot check.
///
/// `channels` is a `u16` and carries no upper bound: it is the width cpal's
/// `StreamConfig`, WAVE's `nChannels` and AIFF's `numChannels` all use.
#[derive(Clone, Copy)]
pub(crate) struct Block<'a> {
    /// The interleaved samples, `frames * channels` of them.
    samples: &'a [f32],
    /// The count `samples` is interleaved at. At least 1.
    channels: u16,
}

impl<'a> Block<'a> {
    /// A block of `samples` interleaved at `channels`.
    ///
    /// `channels` is at least 1 (a zero-channel block has no frames to speak
    /// of); the assertion is a floor, not a ceiling, and no maximum is implied.
    pub(crate) fn new(samples: &'a [f32], channels: u16) -> Self {
        debug_assert!(channels >= 1, "a block carries at least one channel");
        Self { samples, channels }
    }

    /// The interleaved samples.
    pub(crate) fn samples(&self) -> &'a [f32] {
        self.samples
    }

    /// The count the samples are interleaved at.
    pub(crate) fn channels(&self) -> u16 {
        self.channels
    }

    /// Frames in the block: interleaved samples divided by the channel count.
    ///
    /// Divides by `channels.max(1)` so a descriptor built with zero (which the
    /// constructor rejects in a debug build) yields a number rather than
    /// dividing by zero in a release one. Any trailing partial frame is
    /// truncated, matching [`sample::downmix_to_mono`].
    pub(crate) fn frames(&self) -> usize {
        self.samples.len() / self.channels.max(1) as usize
    }

    /// Whether the block carries no samples.
    pub(crate) fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Interleaved samples in the block, the length the bare slice had.
    pub(crate) fn len(&self) -> usize {
        self.samples.len()
    }
}

/// A capture processing stage: reads one [`Block`] of interleaved f32 samples
/// and appends its processed output to `out` (the caller clears `out` first).
///
/// `Send` so the boxed stages can live behind the stream's `Mutex` and cross
/// thread boundaries with the `Send + Sync` [`crate::microphone::MicrophoneStream`].
pub(crate) trait Stage: Send {
    /// Process `input`, appending the result to `out`.
    ///
    /// What is appended is interleaved at
    /// [`output_channels`](Stage::output_channels) of the input's count, which
    /// is the same count for every stage that does not change it.
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError>;

    /// The channel count this stage emits, given the count it is handed.
    ///
    /// The identity by default: a stage conditions the channels it receives and
    /// hands the same number on. [`Downmix`] overrides it, being the one stage
    /// in the chain whose output count differs from its input count. The chain
    /// walks this to resolve its own delivered count
    /// ([`CaptureStage::output_channels`]) rather than naming that count, so a
    /// count fixed here rather than derived is a defect the chain can see.
    ///
    /// Takes the input count rather than answering from state, so a stage whose
    /// output count depends on its input can say so without being rebuilt.
    fn output_channels(&self, input_channels: u16) -> u16 {
        input_channels
    }

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

    /// The echo canceller's transport and cancellation metrics, when this stage
    /// holds a canceller.
    ///
    /// `None` for every other stage, so [`CaptureStage::aec_metrics`] finds the
    /// one stage that answers without downcasting a boxed trait object.
    #[cfg(feature = "aec")]
    fn aec_metrics(&self) -> Option<AecMetrics> {
        None
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
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The count the stage was built for and the count the block carries are
        // the same number reaching it by two routes: the chain resolved this
        // stage's position from the device count, and the walk hands it the
        // running count. They can only disagree if one of the two drifted.
        debug_assert_eq!(
            input.channels(),
            self.channels,
            "the downmix was built for {} channels and handed {}",
            self.channels,
            input.channels()
        );
        let frames = input.frames();
        out.extend(sample::downmix_to_mono(input.samples(), self.channels));
        debug_assert_eq!(
            out.len(),
            frames,
            "the downmix emits exactly one sample per input frame"
        );
        Ok(())
    }

    /// Mono, whatever it was handed: this is the stage the chain collapses at.
    fn output_channels(&self, _input_channels: u16) -> u16 {
        1
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
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The wrapped engine is mono by documented contract, and reading an
        // interleaved block as one time series would filter across channels
        // rather than along each. Stated here rather than assumed, so a chain
        // that ever hands this stage more than one channel says so.
        debug_assert_eq!(
            input.channels(),
            1,
            "the resampler requires mono, got {} channels",
            input.channels()
        );
        // The resampler appends to `out`. Its one steady-path error rejects a
        // block that arrives after the flush; the callers stop feeding a flushed
        // chain, so it is propagated rather than assumed away.
        self.0.process(input.samples(), out)?;
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

/// The far-end reference queue: the bounded hand-off between the caller's push
/// and the [`AecStage`] that drains it.
///
/// The caller pushes played audio through
/// [`MicrophoneStream::push_aec_reference`](crate::microphone::MicrophoneStream::push_aec_reference);
/// the stage takes from the front of it at the rate the capture consumes (see
/// [`AecStage::feed_reference`]), leaving the rest queued. Both sides hold an
/// [`Arc`] of the same queue, so the push never reaches through the capture
/// chain's own lock and never waits on the canceller.
///
/// The queue is bounded in samples, sized in seconds at the declared reference
/// rate by the caller (see
/// [`AEC_REFERENCE_BOUND_SECS`](crate::microphone::AEC_REFERENCE_BOUND_SECS)).
/// A push that does not fit is truncated at its NEWEST end and the discarded
/// samples are counted: what remains is a contiguous run in played order that
/// continues exactly where the canceller's far stream left off, and the deficit
/// lands at the newest end rather than as a hole between samples already fed and
/// samples still queued. It matches the capture channel's own contract, which
/// also discards the newest buffer and counts it
/// ([`MicrophoneStream::overrun_count`](crate::microphone::MicrophoneStream::overrun_count)),
/// so a stall that truncates one stream truncates the other at the same point in
/// the timeline instead of moving them apart.
///
/// The samples a truncation discards are lost, but the TIME they occupied is
/// not: the drain keeps the far-end stream level with the capture by supplying
/// silence for every near-end sample the caller did not cover (see
/// [`AecStage::fill_reference_to`]), so a discarded span costs the cancellation
/// of that span alone rather than moving every later sample off its alignment.
///
/// Because the drain is rate-matched, the queue holds a caller's backlog for as
/// long as the capture takes to reach it: a push of a whole utterance stands in
/// the queue and is read out over the seconds that utterance plays. The bound is
/// therefore the amount a caller may run ahead of its own capture, and
/// [`dropped`](Self::dropped) climbing is a caller running further ahead than
/// that.
#[cfg(feature = "aec")]
pub(crate) struct AecReferenceRing {
    /// Samples pushed and not yet drained, in played order. Never longer than
    /// `capacity`. Behind a `Mutex` so both sides reach it under a shared
    /// `&self`; the drain holds the lock only for a copy off the front, so a
    /// push waits on another push at worst and never on the canceller. A deque
    /// because the drain takes a bounded prefix and leaves the rest, which a
    /// `Vec` could only do by moving the remainder down on every block.
    queued: Mutex<VecDeque<f32>>,
    /// Bound on `queued`, in samples at the declared reference rate.
    capacity: usize,
    /// Samples discarded because the queue was full, read via
    /// [`dropped`](Self::dropped).
    dropped: AtomicU64,
    /// Samples of silence the drain supplied in the caller's place, at the
    /// capture target rate, read via [`silence`](Self::silence). Recorded here
    /// rather than on the stage so the accessor reaches it without taking the
    /// capture chain's lock, as [`dropped`](Self::dropped) does.
    silence: AtomicU64,
}

#[cfg(feature = "aec")]
impl AecReferenceRing {
    /// A queue bounded at `capacity` samples.
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            queued: Mutex::new(VecDeque::with_capacity(capacity)),
            capacity,
            dropped: AtomicU64::new(0),
            silence: AtomicU64::new(0),
        }
    }

    /// Append `reference` in played order, discarding whatever does not fit from
    /// the newest end and counting it.
    ///
    /// Never blocks on the canceller and never fails, so a caller may push from
    /// a renderer callback or a socket handler.
    pub(crate) fn push(&self, reference: &[f32]) {
        let taken = {
            let mut queued = self.queued.lock().unwrap_or_else(PoisonError::into_inner);
            let room = self.capacity.saturating_sub(queued.len());
            let taken = reference.len().min(room);
            queued.extend(reference[..taken].iter().copied());
            taken
        };
        let dropped = reference.len() - taken;
        if dropped > 0 {
            self.dropped.fetch_add(dropped as u64, Ordering::Relaxed);
        }
    }

    /// Move up to `max` samples off the FRONT of the queue into `out`, leaving
    /// whatever is behind them queued in played order, and report how many are
    /// still queued afterwards.
    ///
    /// `out` is cleared first. Taking a bounded prefix rather than the whole
    /// queue is what keeps the far-end frontier level with the capture; the
    /// remainder is not discarded, it is read on later blocks. The count that
    /// comes back is what the caller needs to know whether silence would land in
    /// front of reference or after it, taken under the same lock as the drain.
    fn drain_into(&self, out: &mut Vec<f32>, max: usize) -> usize {
        out.clear();
        let mut queued = self.queued.lock().unwrap_or_else(PoisonError::into_inner);
        let take = max.min(queued.len());
        out.extend(queued.drain(..take));
        queued.len()
    }

    /// Samples pushed and not yet taken by the drain. Read by the tests that pin
    /// the queue's accounting; the capture path reads the same figure through
    /// [`drain_into`](Self::drain_into), under the lock it already holds.
    #[cfg(test)]
    pub(crate) fn queued(&self) -> usize {
        self.queued
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .len()
    }

    /// Samples discarded because the queue was full, for the lifetime of the
    /// queue.
    pub(crate) fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    /// Record `count` far-end samples the drain supplied as silence because the
    /// caller had pushed none for them.
    fn record_silence(&self, count: u64) {
        self.silence.fetch_add(count, Ordering::Relaxed);
    }

    /// Samples of silence the drain supplied in the caller's place, at the
    /// capture target rate, for the lifetime of the queue.
    pub(crate) fn silence(&self) -> u64 {
        self.silence.load(Ordering::Relaxed)
    }
}

/// Echo-cancellation settings [`build_capture_stage`] builds the [`AecStage`]
/// from, carried in [`Transforms`].
///
/// `tail_ms` and `suppression` are `None` when the caller named no value, in
/// which case the canceller's own default stands: decibri names neither number
/// itself, so the two cannot drift apart from the values the canceller
/// documents.
#[cfg(feature = "aec")]
pub(crate) struct AecSettings {
    /// The canceller model to run.
    pub model: AecModel,
    /// Adaptive filter tail length in milliseconds; `None` takes the
    /// canceller's default.
    pub tail_ms: Option<u16>,
    /// Residual-suppression policy; `None` takes the canceller's default.
    pub suppression: Option<Suppression>,
    /// The rate the caller declares its pushed reference is at. Converted to the
    /// capture target rate on the drain side when the two differ.
    pub reference_rate: u32,
    /// The queue the caller pushes into and this stage drains.
    pub reference: Arc<AecReferenceRing>,
}

/// Cancel the echo of caller-supplied far-end audio out of the captured signal.
///
/// [`build_capture_stage`] pushes this last in the `normalize` segment, after
/// [`Downmix`] and [`ResampleStage`], so the canceller receives mono at the
/// target rate: the only format it accepts, and the rate its engine is
/// constructed at. Running before the VAD tap is taken means the detector reads
/// the echo-removed signal rather than the raw capture.
///
/// Each [`process`](Stage::process) feeds the reference into the engine before
/// cancelling the near-end block, and feeds it at the rate the capture consumes
/// it: enough to reach the near-end frontier this block ends at, and no further.
/// Whatever the caller pushed beyond that stays queued and is read on later
/// blocks, so nothing is discarded and the far-end frontier never runs ahead of
/// the capture. The engine re-blocks internally, so a call may append fewer or
/// more samples than it consumed and the totals balance after
/// [`flush`](Stage::flush).
///
/// The far-end frontier is what the canceller measures its alignment from, and
/// it measures it once, on the first block it sees. A frontier that has run
/// ahead of the capture puts every later near-end sample past the reference the
/// canceller holds, which its delay search reads as a window it cannot score, so
/// it never locks and never recovers. Feeding at the capture's own rate is what
/// holds the two together: see
/// [`feed_reference`](AecStage::feed_reference).
///
/// The feed then tops the far-end stream up with silence to the near-end
/// frontier, so the two streams advance together whatever the caller does. That
/// is the played signal, not a substitute for it: while nothing is playing the
/// far end IS silence, and a caller who pushes nothing is saying so. Without the
/// top-up a caller who pauses leaves the far-end frontier permanently behind the
/// capture, which the engine can only read as a reference that never arrives.
#[cfg(feature = "aec")]
struct AecStage {
    /// The canceller, constructed at the capture target rate.
    aec: Aec,
    /// The shared queue the caller pushes far-end audio into.
    reference: Arc<AecReferenceRing>,
    /// The rate the caller declares its pushed reference is at. Held so the
    /// per-block take, which is measured at the capture target rate, converts to
    /// a count of queued samples.
    reference_rate: u32,
    /// The capture target rate, which is the rate the canceller runs at.
    target_rate: u32,
    /// Reference samples taken this block, at the declared reference rate.
    /// Retained across blocks so [`AecReferenceRing::drain_into`] copies into a
    /// buffer that has already reached its high-water length.
    drained: Vec<f32>,
    /// Converts the drained reference to the capture target rate. `Some` only
    /// when the declared reference rate differs from the target rate. The
    /// resampler is mono-only and the reference is mono, so it needs no downmix
    /// ahead of it.
    reference_resampler: Option<PolyphaseResampler>,
    /// The drained reference at the target rate, when a resampler runs.
    resampled: Vec<f32>,
    /// Zeros handed to the engine when the caller's reference falls short of the
    /// capture. Reused across blocks; one block's worth is its high-water mark,
    /// because [`fill_reference_to`](Self::fill_reference_to) restores the
    /// far-ahead-of-near invariant every block and so never has more than one
    /// block to make up.
    silence: Vec<f32>,
    /// Far-end samples handed to the engine, at the capture target rate. Held
    /// against `near_seen` to size the silence top-up.
    far_fed: u64,
    /// Near-end samples handed to the engine, at the capture target rate.
    near_seen: u64,
    /// Whether any near-end sample has reached [`process`](Stage::process). A
    /// stage that received none holds no framing carry, so it emits nothing at
    /// [`flush`](Stage::flush) rather than a tail assembled out of nothing.
    received_near: bool,
    /// Whether the caller still has reference queued after the most recent feed.
    /// While it does, the silence top-up is held back: silence laid down in
    /// front of queued reference is a gap rather than a top-up.
    reference_pending: bool,
}

#[cfg(feature = "aec")]
impl AecStage {
    /// Build the stage for a capture at `target_rate`.
    ///
    /// The engine is constructed at `target_rate`, which is the rate the signal
    /// reaching this stage carries. Returns an error when the canceller rejects
    /// the configuration (bridged by `From<AecError>`) or when the reference
    /// rate pair is one the resampler cannot serve.
    fn new(settings: AecSettings, target_rate: u32) -> Result<Self, DecibriError> {
        let AecSettings {
            model,
            tail_ms,
            suppression,
            reference_rate,
            reference,
        } = settings;

        // `AecConfig` is `#[non_exhaustive]`, so it is built from its own default
        // and assigned field by field. Leaving `tail_ms` and `suppression`
        // unassigned when the caller named no value keeps the canceller's
        // defaults as the single source of those two numbers. `delay_hint_ms` is
        // never assigned: the hint is measured from the reference frontier as the
        // caller's own feeding establishes it, not from an absolute platform
        // latency, so no value decibri or its caller could supply would be
        // correct.
        let mut config = AecConfig::default();
        config.sample_rate = target_rate;
        config.model = model;
        if let Some(tail_ms) = tail_ms {
            config.tail_ms = tail_ms;
        }
        if let Some(suppression) = suppression {
            config.suppression = suppression;
        }
        let aec = Aec::new(config)?;

        // Converting on the drain side keeps the caller's push to a copy into the
        // queue, so the thread that produced the audio pays nothing for the
        // conversion.
        let reference_resampler = if reference_rate != target_rate {
            Some(PolyphaseResampler::new(reference_rate, target_rate)?)
        } else {
            None
        };

        Ok(Self {
            aec,
            reference,
            reference_rate,
            target_rate,
            drained: Vec::new(),
            reference_resampler,
            resampled: Vec::new(),
            silence: Vec::new(),
            far_fed: 0,
            near_seen: 0,
            received_near: false,
            reference_pending: false,
        })
    }

    /// Feed the queued reference into the engine, at most `budget` samples of it
    /// measured at the capture target rate, converting from the declared rate
    /// first when the two differ.
    ///
    /// The budget is what makes the feed rate-matched. The canceller anchors its
    /// alignment on the far-end frontier as it stands at the first block it
    /// processes, and thereafter reads the far-end stream at the position that
    /// anchor implies. A frontier that ran ahead of the capture before the
    /// anchor was taken is therefore an offset every later block carries, and
    /// one the canceller's delay search cannot cover past its own ceiling; a
    /// frontier the capture has since overtaken puts the search window past the
    /// newest reference sample, which the search refuses to score at all. Both
    /// end the same way: no lock, no recovery, and no error.
    ///
    /// Handing over at most what the near-end block consumes keeps the frontier
    /// where the capture is, which is where both bounds are widest. What the
    /// caller pushed beyond that is not discarded, only deferred: it stays at
    /// the front of the queue and is read on the blocks that follow, so a caller
    /// that hands over a whole utterance at once still has every sample of it
    /// cancelled against the capture it echoes into.
    fn feed_reference(&mut self, budget: u64) -> Result<(), DecibriError> {
        if budget == 0 {
            // Nothing was asked for, so nothing was looked at: whether reference
            // is queued is unknown, and the conservative answer is that it is.
            // The silence top-up cannot fire on this path anyway, because a
            // budget of zero means the far end has already reached the frontier.
            self.reference_pending = true;
            return Ok(());
        }
        // The budget is a count of target-rate samples; the queue holds the
        // caller's declared rate. Rounded up so a block takes at least what the
        // near end consumes rather than repeatedly falling a fraction short.
        let take = match self.reference_resampler {
            Some(_) => budget
                .saturating_mul(self.reference_rate as u64)
                .div_ceil(self.target_rate as u64),
            None => budget,
        };
        self.reference_pending = self
            .reference
            .drain_into(&mut self.drained, take.min(usize::MAX as u64) as usize)
            > 0;
        if self.drained.is_empty() {
            return Ok(());
        }
        match &mut self.reference_resampler {
            Some(resampler) => {
                self.resampled.clear();
                resampler.process(&self.drained, &mut self.resampled)?;
                self.aec.feed_reference(&self.resampled);
                self.far_fed += self.resampled.len() as u64;
            }
            None => {
                self.aec.feed_reference(&self.drained);
                self.far_fed += self.drained.len() as u64;
            }
        }
        Ok(())
    }

    /// Feed silence until the far-end stream reaches `near_frontier`, the
    /// near-end sample count this block ends at.
    ///
    /// The engine reads one far-end sample for every near-end sample it cancels,
    /// and takes the far-end frontier as the caller's statement of what has
    /// played. A caller who pushes nothing has stated that nothing played, and
    /// the far-end signal for that span is zero, so the engine is handed exactly
    /// that. A caller feeding in step covers every sample itself and nothing is
    /// added.
    ///
    /// It only ever makes up a shortfall, and the shortfall is at most one
    /// block: the invariant this restores holds entering the call, so the far
    /// end can only have fallen behind by the near-end samples added since.
    ///
    /// The caller runs it only with the queue empty. Silence laid down while
    /// reference is still queued would be a gap in the far-end stream rather
    /// than a top-up of it, and every sample after the gap would sit off its
    /// alignment by the width of the gap.
    fn fill_reference_to(&mut self, near_frontier: u64) {
        let fill = near_frontier.saturating_sub(self.far_fed);
        if fill == 0 {
            return;
        }
        let fill = fill as usize;
        if self.silence.len() < fill {
            self.silence.resize(fill, 0.0);
        }
        self.aec.feed_reference(&self.silence[..fill]);
        self.far_fed += fill as u64;
        self.reference.record_silence(fill as u64);
    }
}

#[cfg(feature = "aec")]
impl Stage for AecStage {
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The canceller is mono on both its near and far ends, and its far-end
        // accounting below counts near-end samples as if each were one frame.
        // Stated here rather than assumed, so a chain that ever hands this stage
        // more than one channel says so.
        debug_assert_eq!(
            input.channels(),
            1,
            "the echo canceller requires mono, got {} channels",
            input.channels()
        );
        // The reference goes in first, up to the frontier this block ends at, and
        // the top-up then covers whatever of the block the caller left uncovered.
        // The budget is the distance the far end has left to travel to reach that
        // frontier, so a caller feeding in step hands over its whole block and a
        // caller running ahead hands over one block's worth of its backlog.
        let near_frontier = self.near_seen + input.len() as u64;
        self.feed_reference(near_frontier.saturating_sub(self.far_fed))?;
        if !self.reference_pending {
            self.fill_reference_to(near_frontier);
        }
        self.near_seen = near_frontier;
        if !input.is_empty() {
            self.received_near = true;
        }
        // The engine appends its output in step with the near-end samples that
        // produced it. Its `latency_samples` is a buffering budget, not an index
        // offset, so nothing here shifts the output by it.
        self.aec.process(input.samples(), out)?;
        Ok(())
    }

    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // A stage that never saw near-end audio holds no framing carry, so it
        // emits nothing rather than a tail with no source.
        if !self.received_near {
            return Ok(());
        }
        // Everything still queued, then the reference resampler's own
        // group-delay tail, both before the engine drains: the far end is
        // complete when the near end's carry comes out, and a caller's push is
        // never left unread. The budget the per-block feed is rate-matched by
        // does not apply here, because there is no near-end audio left for the
        // frontier to run ahead OF: the capture has ended.
        self.feed_reference(u64::MAX)?;
        if let Some(resampler) = &mut self.reference_resampler {
            self.resampled.clear();
            resampler.flush(&mut self.resampled);
            if !self.resampled.is_empty() {
                self.aec.feed_reference(&self.resampled);
            }
        }
        self.aec.flush(out)?;
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        // Forward the engine's own constant framing figure. It is the amount by
        // which the emitted count trails the consumed count, which is what this
        // trait reports, and it is summed only across the `transform` segment
        // that this stage is not part of.
        self.aec.latency_samples()
    }

    fn aec_metrics(&self) -> Option<AecMetrics> {
        Some(self.aec.metrics())
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
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // Every `InPlaceDsp` behind this wrapper carries recursive state across
        // samples (the DC blocker's and biquad's filter memory, the level
        // control's and limiter's gain estimate), and a flat pass over an
        // interleaved block would run that state across channels rather than
        // along each. Stated here rather than assumed, so a chain that ever
        // hands this wrapper more than one channel says so.
        debug_assert_eq!(
            input.channels(),
            1,
            "the scalar in-place wrapper requires mono, got {} channels",
            input.channels()
        );
        // The caller clears `out` first, so this is an exact-length copy of the
        // input that the DSP then rewrites in place: one input sample, one output
        // sample.
        out.extend_from_slice(input.samples());
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
    /// The channel count a block enters the chain at: the device's own count.
    /// Every walk seeds its running count from this.
    input_channels: u16,
    /// The channel count after the `normalize` segment, which is the count the
    /// [`tap`](Self::tap) carries and the count the `transform` segment is
    /// entered at. Resolved at construction by walking `normalize`.
    tap_channels: u16,
    /// The channel count the chain delivers. Resolved at construction by walking
    /// both segments' [`Stage::output_channels`], never named as a constant, so
    /// the value the consumer is told tracks the stages actually built.
    output_channels: u16,
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
    /// Seeds `work` with `input` and sanitizes any non-finite sample to silence,
    /// then ping-pongs `work` <-> `scratch` first across the `normalize` stages and
    /// then across the `transform` stages, leaving the final output in `work`. When
    /// `transform` is non-empty, snapshots the post-`normalize`, pre-`transform`
    /// `work` into `tap` first, so the caller can read the signal as it stood before
    /// the transform step.
    ///
    /// The sanitize pass runs before any stage so a non-finite (NaN or infinity)
    /// sample in a glitched capture buffer cannot poison the recursive state the
    /// conditioning stages carry across blocks (the DC blocker's and high-pass
    /// biquad's filter memory, the denoise caches, the level-control estimate).
    /// Without it, one non-finite sample would leave that feedback state non-finite
    /// and corrupt every later sample for the rest of the stream. Replacing each
    /// non-finite sample with `0.0` keeps the state finite and bounds the damage to
    /// the single offending sample. It is exact on finite input (it changes
    /// nothing), and only the conditioned path pays for it: an unconditioned capture
    /// has no chain, so it keeps its zero-cost direct delivery.
    pub(crate) fn run(&mut self, input: &[f32]) -> Result<&[f32], DecibriError> {
        self.run_normalize(input)?;
        self.capture_tap();
        let mut channels = self.tap_channels;
        Self::run_segment(
            &mut self.work,
            &mut self.scratch,
            &mut self.transform,
            &mut channels,
        )?;
        debug_assert_eq!(
            channels, self.output_channels,
            "the chain walk ended at {} channels but the chain declares {}",
            channels, self.output_channels
        );
        Ok(&self.work)
    }

    /// Run one native block through the `normalize` segment alone, returning
    /// the post-`normalize`, pre-`transform` signal.
    ///
    /// Seeds and sanitizes `work` exactly as [`run`](Self::run) does, then
    /// ping-pongs it across the `normalize` stages only. The returned slice is
    /// the signal [`run`](Self::run) snapshots into [`tap`](Self::tap), so a
    /// caller that reads the pre-`transform` signal and never delivers the
    /// conditioned output takes this instead of a full [`run`](Self::run).
    pub(crate) fn run_normalize(&mut self, input: &[f32]) -> Result<&[f32], DecibriError> {
        self.work.clear();
        self.work.extend_from_slice(input);
        for sample in self.work.iter_mut() {
            if !sample.is_finite() {
                *sample = 0.0;
            }
        }
        let mut channels = self.input_channels;
        Self::run_segment(
            &mut self.work,
            &mut self.scratch,
            &mut self.normalize,
            &mut channels,
        )?;
        debug_assert_eq!(
            channels, self.tap_channels,
            "the normalize walk ended at {} channels but the chain declares {}",
            channels, self.tap_channels
        );
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
    ///
    /// `channels` is the running channel count: it enters holding the count
    /// `work` is interleaved at and leaves holding the count the segment's
    /// output is interleaved at, advanced past each stage by that stage's own
    /// [`Stage::output_channels`]. Carrying it is what lets a stage read its
    /// input's layout from the block instead of assuming one.
    fn run_segment(
        work: &mut Vec<f32>,
        scratch: &mut Vec<f32>,
        stages: &mut [Box<dyn Stage>],
        channels: &mut u16,
    ) -> Result<(), DecibriError> {
        for stage in stages.iter_mut() {
            scratch.clear();
            let emitted = stage.output_channels(*channels);
            stage.process(Block::new(work, *channels), scratch)?;
            *channels = emitted;
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
        // `work` is carried unbroken from the `normalize` walk into the
        // `transform` walk.
        self.flush_normalize()?;
        // Capture the post-`normalize` tail (the resampler's group-delay tail)
        // before it passes through `transform`, so the tap stays aligned with the
        // delivered output across the close path too.
        self.capture_tap();
        let mut channels = self.tap_channels;
        Self::flush_segment(
            &mut self.work,
            &mut self.scratch,
            &mut self.transform,
            &mut channels,
        )?;
        debug_assert_eq!(
            channels, self.output_channels,
            "the chain flush ended at {} channels but the chain declares {}",
            channels, self.output_channels
        );
        out.extend_from_slice(&self.work);
        Ok(())
    }

    /// Drain the `normalize` segment's end-of-stream tail alone, returning it.
    ///
    /// The counterpart of [`run_normalize`](Self::run_normalize) on the close
    /// path: the resampler's group-delay tail, before it passes through
    /// `transform`. `work` starts empty because there is no new input at end of
    /// stream, only each stage's drained tail.
    pub(crate) fn flush_normalize(&mut self) -> Result<&[f32], DecibriError> {
        self.work.clear();
        let mut channels = self.input_channels;
        Self::flush_segment(
            &mut self.work,
            &mut self.scratch,
            &mut self.normalize,
            &mut channels,
        )?;
        debug_assert_eq!(
            channels, self.tap_channels,
            "the normalize flush ended at {} channels but the chain declares {}",
            channels, self.tap_channels
        );
        Ok(&self.work)
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

    /// The channel count the chain's delivered output is interleaved at.
    ///
    /// Resolved at construction by walking every stage's
    /// [`Stage::output_channels`] from the device count the chain was built
    /// with, so it reports what the stages actually do rather than a count named
    /// alongside them. A consumer reads this instead of deriving the count from
    /// the configuration a second time.
    pub(crate) fn output_channels(&self) -> u16 {
        self.output_channels
    }

    /// The summed algorithmic latency, in samples at the output rate, of the
    /// `transform` (conditioning) stages: the amount by which the delivered,
    /// post-conditioning output trails the post-normalize signal the
    /// [`tap`](Self::tap) holds. The `normalize` stages are excluded on purpose:
    /// they run before the tap is taken, so their delay (the resampler's group
    /// delay, the echo canceller's framing) reaches the tap and the delivered
    /// output alike and does not separate them.
    pub(crate) fn transform_latency(&self) -> usize {
        self.transform.iter().map(|s| s.latency_samples()).sum()
    }

    /// The echo canceller's metrics, or `None` when the chain has no canceller.
    ///
    /// The canceller is a `normalize` stage, so only that segment is walked.
    #[cfg(feature = "aec")]
    pub(crate) fn aec_metrics(&self) -> Option<AecMetrics> {
        self.normalize.iter().find_map(|stage| stage.aec_metrics())
    }

    /// Drain one segment's tail, carrying `work` across its stages. At each stage
    /// it feeds the upstream-carried tail through (when non-empty), then drains
    /// that stage's own held tail into the same buffer. `work` enters holding the
    /// previous segment's carried tail (empty for the first non-empty stage) and
    /// leaves holding this segment's output.
    ///
    /// `channels` is the running channel count, advanced past each stage exactly
    /// as [`run_segment`](Self::run_segment) advances it. A stage's drained tail
    /// is interleaved at that stage's OUTPUT count, so the count advances once
    /// per stage, after both the carried block and the tail have been written.
    fn flush_segment(
        work: &mut Vec<f32>,
        scratch: &mut Vec<f32>,
        stages: &mut [Box<dyn Stage>],
        channels: &mut u16,
    ) -> Result<(), DecibriError> {
        for stage in stages.iter_mut() {
            scratch.clear();
            let emitted = stage.output_channels(*channels);
            if !work.is_empty() {
                stage.process(Block::new(work, *channels), scratch)?;
            }
            stage.flush(scratch)?;
            *channels = emitted;
            std::mem::swap(work, scratch);
        }
        Ok(())
    }
}

/// The opt-in processing [`build_capture_stage`] applies, bundled into one
/// argument so that adding a capability is a new field rather than another
/// positional parameter. The first five fields map one-to-one to the transform
/// stages, listed in chain order; `aec` maps to the last stage of the
/// `normalize` segment.
///
/// [`Default`] is every field off, the configuration that pushes no optional
/// stage at all, so a caller names only what it enables. A field added here must
/// have an off state that is its type's own default, or the derive stops being
/// correct and the impl has to be written out; `default_is_the_all_off_literal`
/// holds that line.
#[derive(Default)]
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
    /// Echo-cancellation settings with the shared far-end reference queue;
    /// `None` leaves echo cancellation off.
    ///
    /// The one field here whose stage lands in `normalize` rather than
    /// `transform`: it rides in this bundle because that keeps a new capability
    /// a field on one struct instead of a parameter on
    /// [`build_capture_stage`]'s signature.
    #[cfg(feature = "aec")]
    pub aec: Option<AecSettings>,
}

/// Build the capture stage chain that normalizes a device to the output format
/// and applies any opt-in enhancement.
///
/// Pushes [`Downmix`] when the device delivers more channels than the output
/// target (averaging down to the target, which is mono here), then
/// [`ResampleStage`] when the device's `native_rate` differs from `target_rate`
/// (converting the captured audio to the requested rate), then the
/// [`AecStage`] when `aec` names settings (and the `aec` feature is compiled
/// in), last in the `normalize` segment so the canceller receives mono at the
/// target rate and the VAD tap carries the echo-removed signal. When `dc_removal` is
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
/// above), when the denoise model fails to load (a
/// [`DecibriError::ModelLoadFailed`]), or when the echo canceller rejects its
/// configuration (a [`DecibriError::AecSampleRateUnsupported`] for a target rate
/// outside its window, a [`DecibriError::AecConfigInvalid`] otherwise).
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
        #[cfg(feature = "aec")]
        aec,
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

    // Echo cancellation runs LAST in the normalize segment, after the downmix
    // and the resample, so the canceller receives mono at the target rate: the
    // only channel count it accepts, and the rate its engine is constructed at.
    // Nothing runs between it and the VAD tap, so the detector reads the
    // echo-removed signal. The order is pinned by this push position and
    // `build_orders_aec_last_in_normalize`.
    #[cfg(feature = "aec")]
    if let Some(settings) = aec {
        normalize.push(Box::new(AecStage::new(settings, target_rate)?));
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

    // Resolve the channel count each segment ends at by asking the stages that
    // were actually built, in the order they run. Named nowhere: a count written
    // here as a literal would agree with the stages today and stop agreeing the
    // moment a stage that changes the count is added or removed.
    let tap_channels = normalize.iter().fold(device_channels, |channels, stage| {
        stage.output_channels(channels)
    });
    let output_channels = transform.iter().fold(tap_channels, |channels, stage| {
        stage.output_channels(channels)
    });

    Ok(if normalize.is_empty() && transform.is_empty() {
        None
    } else {
        Some(CaptureStage {
            normalize,
            transform,
            input_channels: device_channels,
            tap_channels,
            output_channels,
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
                    ..Default::default()
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
                    ..Default::default()
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
                    ..Default::default()
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
                    ..Default::default()
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
                    ..Default::default()
                }
            )
            .unwrap()
            .is_some(),
            "stereo device above the target rate gets a chain (downmix + resample)"
        );
    }

    /// The chain resolves its channel counts by walking the stages it built,
    /// never by naming them.
    ///
    /// Regression: an `input_channels` or `output_channels` written as a
    /// literal. Every capture configuration decibri accepts today is mono in and
    /// mono out, so a constant `1` in either place agrees with the resolved
    /// value at every point the rest of the suite observes it, and planting one
    /// leaves the whole suite green. The counts below are chosen so a constant
    /// cannot pass: a device count the builder must carry unchanged, and a chain
    /// whose resolved output is not 1.
    ///
    /// The chains are built and inspected, never run. Stages that require mono
    /// assert it, so this pins what the builder RESOLVED rather than what a
    /// multichannel block would do through it.
    ///
    /// The counts also run far past any plausible invented ceiling, up to
    /// `u16::MAX`, so a fixed maximum added to the builder fails here rather
    /// than passing unnoticed for every count anyone happened to test.
    #[test]
    fn chain_channel_counts_are_resolved_by_walking_the_stages() {
        // The device count reaches the chain unchanged, at every count.
        for device_channels in [1u16, 2, 3, 6, 8, 17, 64, 1024, u16::MAX] {
            let chain =
                build_capture_stage(device_channels, 1, 48_000, 16_000, Transforms::default())
                    .expect("no channel count is rejected by the builder")
                    .expect("a rate change always builds a chain");
            assert_eq!(
                chain.input_channels, device_channels,
                "the chain must carry the device count it was given"
            );
        }

        // `Downmix` declares mono, so a chain holding one resolves to 1 whatever
        // went into it.
        for device_channels in [2u16, 6, 1024] {
            let chain =
                build_capture_stage(device_channels, 1, 16_000, 16_000, Transforms::default())
                    .expect("no channel count is rejected by the builder")
                    .expect("a device above the target gets a downmix chain");
            assert_eq!(
                chain.output_channels(),
                1,
                "the downmix declares mono, so the chain resolves to mono"
            );
        }

        // No downmix in the chain: every stage passes the count through, so the
        // chain resolves to a count that is NOT 1. This is the arm a constant
        // cannot pass.
        for channels in [2u16, 5, 32] {
            let chain =
                build_capture_stage(channels, channels, 48_000, 16_000, Transforms::default())
                    .expect("no channel count is rejected by the builder")
                    .expect("a rate change builds a resample chain");
            assert_eq!(
                chain.output_channels(),
                channels,
                "with nothing collapsing, the count passes through unchanged"
            );
            assert_eq!(
                chain.input_channels,
                chain.output_channels(),
                "a chain that collapses nothing delivers what it was handed"
            );
        }
    }

    /// The block descriptor carries the channel count it was given and derives
    /// its frame count from it, rather than either being assumed.
    ///
    /// Regression: a descriptor that drops or caps the count, or a `frames()`
    /// that answers from the sample count alone. The counts run past any
    /// plausible invented ceiling for the same reason as above.
    #[test]
    fn a_block_carries_its_channel_count_and_derives_its_frames() {
        let samples = [0.0f32; 720];
        for channels in [1u16, 2, 3, 6, 8, 16, 45, 720] {
            let block = Block::new(&samples, channels);
            assert_eq!(
                block.channels(),
                channels,
                "the descriptor carries the count it was given"
            );
            assert_eq!(
                block.frames(),
                samples.len() / channels as usize,
                "frames are derived from the carried count"
            );
            assert_eq!(
                block.len(),
                samples.len(),
                "the interleaved sample count is the length the bare slice had"
            );
            assert!(!block.is_empty());
        }
        // A trailing partial frame is truncated, matching the downmix engine.
        let odd = [0.0f32; 7];
        assert_eq!(
            Block::new(&odd, 2).frames(),
            3,
            "a partial frame is dropped"
        );
        assert!(Block::new(&[], 1).is_empty());
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
                ..Default::default()
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
                ..Default::default()
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
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
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
        let mut chain = build_capture_stage(1, 1, 48_000, 16_000, Transforms::default())
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

    /// The `From<ResamplerError>` bridge gives each resampler failure its own
    /// decibri identity through an explicit arm: `RatePairUnsupported`
    /// carries its rate pair into `ResampleConfigInvalid`, the steady-path
    /// `ProcessAfterFlush` becomes `ResampleAfterFlush` rather than a stream
    /// error, and the rate-less `ZeroSampleRate` becomes
    /// `SampleRateOutOfRange` with no invented rate pair. The unknown-error
    /// fallback (`ResampleFailed`) is not constructible here: the pinned
    /// resampler release defines exactly these three variants, so no
    /// unrecognised value exists to feed the bridge. Its message forwarding
    /// is pinned in `error::tests`.
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
        let fed_after_flush: DecibriError = ResamplerError::ProcessAfterFlush.into();
        assert!(matches!(fed_after_flush, DecibriError::ResampleAfterFlush));
        let zero: DecibriError = ResamplerError::ZeroSampleRate.into();
        assert!(matches!(zero, DecibriError::SampleRateOutOfRange));
    }

    /// `build_capture_stage` propagates a resampler construction failure as
    /// `ResampleConfigInvalid`. In-range rates always construct, so this uses an
    /// out-of-range native rate whose anti-alias filter exceeds the resampler's
    /// cap to exercise the `?` bridge end to end.
    #[test]
    fn build_capture_stage_surfaces_unsupported_rate_pair() {
        let result = build_capture_stage(1, 1, 316_800_000, 16_000, Transforms::default());
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
        reference.process(&input, &mut expected).unwrap();
        reference.flush(&mut expected);

        // The chain: process the whole input via run(), then flush() the tail.
        let mut chain = build_capture_stage(1, 1, 48_000, 16_000, Transforms::default())
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

    /// A resampling chain that has processed nothing drains nothing: closing a
    /// stream that captured no audio appends no samples at all, rather than a
    /// tail of the filter's initial state.
    #[test]
    fn flush_without_input_appends_nothing() {
        let mut chain = resampling_chain();
        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush runs unfed");
        assert!(
            tail.is_empty(),
            "an unfed resampling chain drains no samples, got {}",
            tail.len()
        );
    }

    /// Flushing a resampling chain a second time appends nothing: the tail is
    /// drained once and is not re-emitted.
    #[test]
    fn repeated_flush_appends_nothing() {
        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();
        let mut chain = resampling_chain();
        chain.run(&input).expect("process runs");

        let mut first = Vec::new();
        chain
            .flush(&mut first)
            .expect("first flush drains the tail");
        assert!(
            !first.is_empty(),
            "the resampler holds a group-delay tail to drain"
        );

        let mut second = Vec::new();
        chain.flush(&mut second).expect("second flush runs");
        assert!(
            second.is_empty(),
            "a repeated flush appends no samples, got {}",
            second.len()
        );
    }

    /// A 48000 to 16000 mono chain with no conditioning: the rate pair builds a
    /// [`ResampleStage`], which a matched pair would omit.
    fn resampling_chain() -> CaptureStage {
        build_capture_stage(1, 1, 48_000, 16_000, Transforms::default())
            .unwrap()
            .expect("48k mono -> resample chain")
    }

    /// `Downmix` is stateless, so its trait-default `flush` appends nothing: a
    /// downmix-only chain has no end-of-stream tail and `flush` yields no extra
    /// samples (the unchanged downmix-only path).
    #[test]
    fn downmix_only_flush_is_empty() {
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
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
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
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
                ..Default::default()
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
                ..Default::default()
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
        resampler.process(&mono, &mut resampled).unwrap();
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
                ..Default::default()
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
        let mut chain = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
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
                ..Default::default()
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
        resampler.process(&mono, &mut expected_norm).unwrap();
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
                ..Default::default()
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
    #[cfg(any(feature = "denoise", feature = "aec"))]
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
                ..Default::default()
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
                denoise: Some((model, path.as_path(), None)),
                ..Default::default()
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
            build_capture_stage(1, 1, 16_000, 16_000, Transforms::default())
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
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                ..Default::default()
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

    /// A denoise chain that processed nothing drains nothing: closing a stream
    /// that captured no audio appends no samples, rather than the framing padding
    /// pushed through an empty model. The absolute expectation is zero, on the
    /// transform-only chain and on one carrying a `normalize` segment too.
    #[cfg(feature = "denoise")]
    #[test]
    fn unfed_denoise_chain_flush_appends_nothing() {
        let path = denoise_model_path();
        let model = DenoiseModel::FastEnhancerT;

        for (channels, native_rate, dc_removal, label) in [
            (1u16, 16_000u32, false, "denoise only"),
            (2, 48_000, true, "downmix + resample + dc + denoise"),
        ] {
            let mut chain = build_capture_stage(
                channels,
                1,
                native_rate,
                16_000,
                Transforms {
                    dc_removal,
                    denoise: Some((model, path.as_path(), None)),
                    ..Default::default()
                },
            )
            .unwrap()
            .expect("denoise chain");

            let mut tail = Vec::new();
            chain.flush(&mut tail).expect("flush runs unfed");
            assert_eq!(
                tail.len(),
                0,
                "{label}: an unfed chain delivered {} samples it never received",
                tail.len()
            );
        }
    }

    /// A denoise chain fed even a fraction of one analysis frame still delivers
    /// that audio at close. This is the regression the unfed gate must not break:
    /// 100 samples produce no output through `run`, so the whole delivery comes
    /// from the flush.
    #[cfg(feature = "denoise")]
    #[test]
    fn barely_fed_denoise_chain_still_drains_its_tail() {
        let path = denoise_model_path();
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("denoise chain");

        let out = chain.run(&mono_signal(100)).expect("run").to_vec();
        assert_eq!(out.len(), 0, "100 samples do not fill one analysis window");

        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush drains the tail");
        // Left-pad (256) + 100 real + one window of padding (512) = 868 samples,
        // which yields two whole 256-sample hops.
        assert_eq!(tail.len(), 512, "the real tail is delivered in full");
        assert!(
            tail.iter().any(|&s| s != 0.0),
            "the tail carries the audio the chain received"
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
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                ..Default::default()
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
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                ..Default::default()
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
                highpass: Some(filter),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                ..Default::default()
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
                    highpass: Some(HighpassFilter::Hz80),
                    ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                ..Default::default()
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
            build_capture_stage(1, 1, 16_000, 16_000, Transforms::default())
                .unwrap()
                .is_none(),
            "high-pass off and nothing else: no chain, byte-identical passthrough"
        );

        let downmix_only = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
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
                highpass: Some(HighpassFilter::Hz80),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                ..Default::default()
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
                ..Default::default()
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
            build_capture_stage(1, 1, 16_000, 16_000, Transforms::default())
                .unwrap()
                .is_none(),
            "AGC off and nothing else: no chain, byte-identical passthrough"
        );

        let downmix_only = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
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
                agc: Some(-18),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                ..Default::default()
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
                agc: Some(-18),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                ..Default::default()
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
                agc: Some(-18),
                ..Default::default()
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
            build_capture_stage(1, 1, 16_000, 16_000, Transforms::default())
                .unwrap()
                .is_none(),
            "limiter off and nothing else: no chain, byte-identical passthrough"
        );

        let downmix_only = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
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
                limiter: Some(-1.0),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                limiter: Some(-1.0),
                ..Default::default()
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
                limiter: Some(-1.0),
                ..Default::default()
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
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                limiter: Some(-1.0),
                ..Default::default()
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
                limiter: Some(ceiling_db),
                ..Default::default()
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

    // ── Non-finite input guard ──────────────────────────────────────────
    //
    // `run` sanitizes non-finite (NaN/infinity) input to silence before any stage,
    // so a glitched capture sample cannot poison the recursive state the
    // conditioning stages carry across blocks. These pin that the output stays
    // finite and the stream keeps carrying signal after the glitch, and that the
    // guard is exact on finite input.

    /// A non-finite input sample is sanitized at the chain entry before any stage
    /// runs, so it never reaches the recursive state of the conditioning stages.
    /// Every conditioned config delivers only finite output and keeps carrying real
    /// signal after the glitch (the state recovered because it never saw the
    /// non-finite sample). Without the guard the DC blocker, the high-pass biquad,
    /// and the level control would hold the non-finite value in their feedback and
    /// corrupt every later sample for the rest of the stream.
    #[test]
    fn non_finite_input_does_not_poison_the_chain() {
        // A tone above the noise floor, so the level control is active (it would
        // otherwise poison its gain on the non-finite sample rather than freeze).
        // The glitches sit early so a poisoned stream would be corrupt or silent by
        // the time the second half is checked.
        fn run_glitched(
            dc: bool,
            highpass: Option<HighpassFilter>,
            agc: Option<i8>,
            limiter: Option<f32>,
        ) -> Vec<f32> {
            let sr = 16_000u32;
            let mut input: Vec<f32> = (0..8_000)
                .map(|i| 0.3 * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sr as f32).sin())
                .collect();
            input[100] = f32::NAN;
            input[200] = f32::INFINITY;
            input[300] = f32::NEG_INFINITY;
            let mut chain = build_capture_stage(
                1,
                1,
                sr,
                sr,
                Transforms {
                    dc_removal: dc,
                    highpass,
                    agc,
                    limiter,
                    ..Default::default()
                },
            )
            .unwrap()
            .expect("a conditioned config builds a chain");
            let mut out = chain.run(&input).expect("run").to_vec();
            let mut tail = Vec::new();
            chain.flush(&mut tail).expect("flush");
            out.extend_from_slice(&tail);
            out
        }

        fn assert_clean(label: &str, out: &[f32]) {
            assert!(
                out.iter().all(|s| s.is_finite()),
                "{label}: a non-finite input sample leaked into the output"
            );
            // The second half still carries energy: the recursive state recovered
            // rather than collapsing the rest of the stream to silence.
            let energy_after: f32 = out[4_000..].iter().map(|s| s * s).sum();
            assert!(
                energy_after > 0.0,
                "{label}: the stream was silenced after the glitch"
            );
        }

        assert_clean("dc-only", &run_glitched(true, None, None, None));
        assert_clean(
            "highpass-only",
            &run_glitched(false, Some(HighpassFilter::Hz80), None, None),
        );
        #[cfg(feature = "gain")]
        {
            assert_clean(
                "agc-only (no limiter)",
                &run_glitched(false, None, Some(-18), None),
            );
            assert_clean(
                "dc+hp+agc (no limiter)",
                &run_glitched(true, Some(HighpassFilter::Hz80), Some(-18), None),
            );
            assert_clean(
                "full chain",
                &run_glitched(true, Some(HighpassFilter::Hz80), Some(-18), Some(-1.0)),
            );
        }
    }

    /// The non-finite guard is exact on finite input: a full conditioning chain
    /// (DC, high-pass, level control, limiter) reproduces, bit for bit, the same
    /// same-length stages applied directly with no chain, so the guard changes
    /// nothing for conforming audio.
    #[cfg(feature = "gain")]
    #[test]
    fn non_finite_guard_is_exact_on_finite_input() {
        let sr = 16_000u32;
        // A finite signal with a DC offset and a present level, exercising every
        // same-length stage.
        let input: Vec<f32> = (0..4_000)
            .map(|i| 0.2 * (2.0 * std::f32::consts::PI * 200.0 * i as f32 / sr as f32).sin() + 0.05)
            .collect();

        let mut chain = build_capture_stage(
            1,
            1,
            sr,
            sr,
            Transforms {
                dc_removal: true,
                highpass: Some(HighpassFilter::Hz80),
                agc: Some(-18),
                limiter: Some(-1.0),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("conditioned chain");
        let got = chain.run(&input).expect("run").to_vec();

        // The same same-length stages in chain order, applied directly (no guard).
        let mut reference = input.clone();
        DcBlocker::new().process_in_place(&mut reference);
        Biquad::highpass(HighpassFilter::Hz80.cutoff_hz(), sr as f32)
            .process_in_place(&mut reference);
        crate::gain::LevelControl::agc(-18, sr).process_in_place(&mut reference);
        crate::gain::Limiter::new(-1.0, sr).process_in_place(&mut reference);

        assert_eq!(
            got, reference,
            "the guard must not alter finite input (bit-for-bit identity)"
        );
    }

    /// The guard also protects the denoise stage: a non-finite input does not reach
    /// the model caches, so the enhanced output stays finite and the stream is not
    /// silenced after the glitch.
    #[cfg(feature = "denoise")]
    #[test]
    fn non_finite_input_does_not_poison_denoise() {
        let path = denoise_model_path();
        let mut chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("denoise chain");
        let mut input = mono_signal(8_000);
        input[100] = f32::NAN;
        input[200] = f32::INFINITY;
        input[300] = f32::NEG_INFINITY;
        let mut out = chain.run(&input).expect("run").to_vec();
        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush");
        out.extend_from_slice(&tail);
        assert!(
            out.iter().all(|s| s.is_finite()),
            "denoise leaked a non-finite output sample"
        );
        assert!(
            out.iter().map(|s| s * s).sum::<f32>() > 0.0,
            "the denoise stream was silenced after the glitch"
        );
    }

    /// The derived [`Default`] for [`Transforms`] is the all-off configuration,
    /// field for field. Every elided field elsewhere in the crate resolves to a
    /// value pinned here, so the literal below stays exhaustive: a field added to
    /// the struct without a line here fails to compile.
    #[test]
    fn default_is_the_all_off_literal() {
        let all_off = Transforms {
            dc_removal: false,
            denoise: None,
            highpass: None,
            agc: None,
            limiter: None,
            #[cfg(feature = "aec")]
            aec: None,
        };
        let default = Transforms::default();
        assert_eq!(
            default.dc_removal, all_off.dc_removal,
            "the default must not remove DC"
        );
        assert_eq!(
            default.denoise, all_off.denoise,
            "the default must not denoise"
        );
        assert_eq!(
            default.highpass, all_off.highpass,
            "the default must not high-pass"
        );
        assert_eq!(default.agc, all_off.agc, "the default must not apply AGC");
        assert_eq!(
            default.limiter, all_off.limiter,
            "the default must not limit"
        );
        #[cfg(feature = "aec")]
        assert!(
            default.aec.is_none() && all_off.aec.is_none(),
            "the default must not cancel echo"
        );
    }

    // ── Echo cancellation ──────────────────────────────────────────────
    //
    // Every case below builds its chain through `build_capture_stage`, so the
    // stage's position in the chain is part of what is exercised rather than
    // something the test arranges for itself.

    /// Deterministic broadband noise in `[-0.5, 0.5)`. The canceller's delay
    /// search is correlation-based, so it needs far-end material with an
    /// unambiguous peak; sustained periodic material has none. A 64-bit linear
    /// congruential generator, so the sequence is identical on every platform and
    /// every run and a failure is reproducible.
    #[cfg(feature = "aec")]
    fn far_noise(n: usize) -> Vec<f32> {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) as f32 / 2_147_483_648.0) - 0.5
            })
            .collect()
    }

    /// Root-mean-square level of `samples` in dBFS, floored so an exactly silent
    /// buffer is a finite number rather than negative infinity.
    #[cfg(feature = "aec")]
    fn rms_db(samples: &[f32]) -> f32 {
        let mean_square = samples.iter().map(|&s| s * s).sum::<f32>() / samples.len().max(1) as f32;
        10.0 * mean_square.max(1e-20).log10()
    }

    /// Echo-cancellation settings sharing `queue` with the caller side, at the
    /// canceller's own defaults for everything the tests do not vary.
    #[cfg(feature = "aec")]
    fn aec_settings(queue: &Arc<AecReferenceRing>, reference_rate: u32) -> AecSettings {
        AecSettings {
            model: AecModel::default(),
            tail_ms: None,
            suppression: None,
            reference_rate,
            reference: Arc::clone(queue),
        }
    }

    /// A chain with echo cancellation on, sharing `queue`.
    #[cfg(feature = "aec")]
    fn aec_chain(
        device_channels: u16,
        native_rate: u32,
        target_rate: u32,
        dc_removal: bool,
        queue: &Arc<AecReferenceRing>,
        reference_rate: u32,
    ) -> CaptureStage {
        build_capture_stage(
            device_channels,
            1,
            native_rate,
            target_rate,
            Transforms {
                dc_removal,
                aec: Some(aec_settings(queue, reference_rate)),
                ..Default::default()
            },
        )
        .expect("the chain builds")
        .expect("echo cancellation builds a chain")
    }

    /// Run `input` through `chain` in `block`-sized pieces and then flush,
    /// returning everything the chain delivered across the whole stream.
    #[cfg(feature = "aec")]
    fn whole_stream(chain: &mut CaptureStage, input: &[f32], block: usize) -> Vec<f32> {
        let mut out = Vec::new();
        for piece in input.chunks(block) {
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");
        out
    }

    /// The golden TTS recording, decoded through the offline path at the
    /// recording's own rate with every conditioning option off, so the samples
    /// delivered are the recording itself.
    #[cfg(feature = "aec")]
    fn golden_recording() -> Vec<f32> {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("assets")
            .join("vad-golden-tts-speech-16k.wav");
        crate::file::File::open(path, crate::file::FileConfig::default())
            .expect("the golden recording opens")
            .flat_map(|chunk| chunk.expect("the golden recording decodes").data)
            .collect()
    }

    /// Echo cancellation enabled with no reference ever pushed delivers the same
    /// samples, bit for bit, as echo cancellation off. With an all-zero far end
    /// the canceller's estimate is exactly zero, so its error signal is the
    /// near-end input unchanged and its residual suppressor never arms.
    ///
    /// This is the state most streams are in most of the time, so it is the
    /// load-bearing case: it exercises the stage being present, drained every
    /// block, and inert. Regression: a stage that is not inert, or a drain that
    /// feeds the engine something when the caller pushed nothing.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_without_a_reference_is_bit_identical_to_aec_off() {
        // The canceller alone: the chain is the stage and nothing else, so echo
        // cancellation off builds no chain at all and its delivery IS the input.
        let mono = mono_signal(9_600);
        for block in [320, 1_600, 4_801] {
            let queue = Arc::new(AecReferenceRing::new(16_000));
            let mut chain = aec_chain(1, 16_000, 16_000, false, &queue, 16_000);
            assert_eq!(
                whole_stream(&mut chain, &mono, block),
                mono,
                "an unfed canceller must deliver its input unchanged (block {block})"
            );
        }

        // The canceller inside a full chain: a stereo 48 kHz device down to mono
        // 16 kHz with DC removal, so the stage sits between the resampler and the
        // conditioning exactly as it does in production.
        let stereo: Vec<f32> = mono_signal(19_200)
            .iter()
            .enumerate()
            .map(|(i, &s)| if i % 2 == 0 { s } else { s * 0.5 + 0.01 })
            .collect();
        for block in [320, 1_600, 4_800] {
            let queue = Arc::new(AecReferenceRing::new(16_000));
            let mut with = aec_chain(2, 48_000, 16_000, true, &queue, 16_000);
            let mut without = build_capture_stage(
                2,
                1,
                48_000,
                16_000,
                Transforms {
                    dc_removal: true,
                    ..Default::default()
                },
            )
            .expect("the chain builds")
            .expect("downmix and resample build a chain");
            assert_eq!(
                whole_stream(&mut with, &stereo, block),
                whole_stream(&mut without, &stereo, block),
                "an unfed canceller must not perturb a full chain (block {block})"
            );
        }
    }

    /// The same inert-stage invariant on real speech rather than a synthetic
    /// tone: the golden TTS recording through an echo-cancelling chain with no
    /// reference is delivered bit for bit as the chain without one delivers it.
    /// Regression: an invariant that holds on a synthetic signal and not on the
    /// material the pinned detector anchors are measured against.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_without_a_reference_is_bit_identical_on_the_golden_recording() {
        let recording = golden_recording();
        assert!(
            recording.len() > 100_000,
            "the golden recording decoded to {} samples",
            recording.len()
        );

        let queue = Arc::new(AecReferenceRing::new(16_000));
        let mut with = aec_chain(1, 48_000, 16_000, true, &queue, 16_000);
        let mut without = build_capture_stage(
            1,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: true,
                ..Default::default()
            },
        )
        .expect("the chain builds")
        .expect("resample builds a chain");
        assert_eq!(
            whole_stream(&mut with, &recording, 1_600),
            whole_stream(&mut without, &recording, 1_600),
            "an unfed canceller must not perturb real speech"
        );
    }

    /// A far-end stream and its echo in the capture, on one timeline: the echo of
    /// far-end sample `i` sits at near-end sample `i + DELAY`, so a caller that
    /// pushes the reference early is pushing audio the capture has not reached
    /// yet rather than audio from a different signal.
    #[cfg(feature = "aec")]
    fn far_and_its_echo(samples: usize, delay: usize, gain: f32) -> (Vec<f32>, Vec<f32>) {
        let far = far_noise(samples);
        let mut near = vec![0.0f32; far.len()];
        for i in delay..near.len() {
            near[i] = gain * far[i - delay];
        }
        (far, near)
    }

    /// A caller that hands the canceller its far-end audio BEFORE reading its
    /// first capture chunk still locks and cancels, and reaches exactly the
    /// alignment a caller feeding in step reaches.
    ///
    /// This is the opening move of a talk-back application: start the microphone,
    /// play a greeting, then enter the read loop. The whole greeting is pushed
    /// before the first chunk is read. Regression: a feed that hands the canceller
    /// everything queued at the head of the block, which puts the far-end frontier
    /// the alignment is anchored on ahead of the capture by the whole greeting.
    /// Measured against that regression, the two-second case below never locks at
    /// all: `delay_samples` stays `None`, `acquisition_parked` covers all 192,000
    /// samples of the stream, and the echo is delivered at 0.0 dB of reduction for
    /// its whole length. The cliff was at 1340 samples, 84 ms, so a greeting of a
    /// tenth of a second was enough.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_locks_on_a_reference_pushed_before_the_first_capture_block() {
        const RATE: usize = 16_000;
        const DELAY: usize = 400;
        const SECONDS: usize = 12;
        const BLOCK: usize = 320;
        let (far, near) = far_and_its_echo(SECONDS * RATE, DELAY, 0.25);

        // The alignment a caller feeding in step reaches, which the pre-pushing
        // callers below must reach as well.
        let run = |prepush: usize| {
            let queue = Arc::new(AecReferenceRing::new(2 * RATE));
            let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &queue, RATE as u32);
            let mut pushed = prepush;
            queue.push(&far[..pushed]);
            let mut out = Vec::new();
            for (block, piece) in near.chunks(BLOCK).enumerate() {
                let played = ((block + 1) * BLOCK).min(far.len());
                if played > pushed {
                    queue.push(&far[pushed..played]);
                    pushed = played;
                }
                out.extend_from_slice(chain.run(piece).expect("run"));
            }
            chain.flush(&mut out).expect("flush");
            (out, chain.aec_metrics().expect("metrics"), queue)
        };

        let (_, in_step, _) = run(0);
        let aligned = in_step
            .delay_samples
            .expect("a caller feeding in step locks");

        // 32,000 samples is the whole two-second queue bound, and 24 times the
        // 1340-sample cliff the greedy feed died at, so the case is not marginal.
        for prepush in [1_600usize, 8_000, 32_000] {
            let (out, metrics, queue) = run(prepush);
            assert_eq!(
                metrics.delay_samples,
                Some(aligned),
                "a reference pushed before the first block must reach the same \
                 alignment as one fed in step (pre-push {prepush})"
            );
            assert_eq!(
                queue.dropped(),
                0,
                "a pre-push inside the bound discards nothing (pre-push {prepush})"
            );
            assert_eq!(
                metrics.reference_starved, 0,
                "the far end stays level with the capture (pre-push {prepush})"
            );
            assert_eq!(
                metrics.reference_reanchors, 0,
                "the alignment holds for the whole stream (pre-push {prepush})"
            );

            let window = (SECONDS - 1) * RATE..(SECONDS - 1) * RATE + RATE / 2;
            let removed = rms_db(&near[window.clone()]) - rms_db(&out[window]);
            assert!(
                removed >= 30.0,
                "a pre-pushed reference must still cancel (pre-push {prepush}, \
                 removed {removed:.1} dB)"
            );
        }
    }

    /// The far-end frontier the canceller is handed never runs ahead of the
    /// capture, whatever the caller's pushes look like.
    ///
    /// Asserted on the frontier itself rather than inferred from cancellation
    /// succeeding: what has left the queue is what has reached the canceller, so
    /// the samples pushed minus the samples still queued is the frontier, and it
    /// must equal the near-end samples the chain has consumed. Regression: a feed
    /// that empties the queue every block, which is what puts the frontier ahead;
    /// against it the first block below leaves the queue empty and the frontier
    /// 31,680 samples ahead of a capture that has delivered 320.
    #[cfg(feature = "aec")]
    #[test]
    fn the_reference_frontier_never_runs_ahead_of_the_capture() {
        const RATE: usize = 16_000;
        const BLOCK: usize = 320;
        const SECONDS: usize = 8;
        let (far, near) = far_and_its_echo(SECONDS * RATE, 400, 0.25);

        // A whole two seconds handed over before the first read, then one
        // utterance every second, which keeps the queue occupied throughout.
        let queue = Arc::new(AecReferenceRing::new(2 * RATE));
        let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &queue, RATE as u32);
        let mut pushed = 2 * RATE;
        queue.push(&far[..pushed]);

        let mut out = Vec::new();
        for (block, piece) in near.chunks(BLOCK).enumerate() {
            let at = block * BLOCK;
            if at > 0 && at.is_multiple_of(RATE) && pushed < far.len() {
                let end = (pushed + RATE).min(far.len());
                queue.push(&far[pushed..end]);
                pushed = end;
            }
            out.extend_from_slice(chain.run(piece).expect("run"));

            let consumed = at + piece.len();
            let frontier = pushed - queue.queued();
            assert_eq!(
                frontier, consumed,
                "the far-end frontier must sit exactly at the capture's own \
                 frontier (block {block})"
            );
        }
        chain.flush(&mut out).expect("flush");

        // The bound holds from the very first block, which is the one the
        // alignment is anchored on.
        let first = Arc::new(AecReferenceRing::new(2 * RATE));
        let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &first, RATE as u32);
        first.push(&far[..2 * RATE]);
        chain.run(&near[..BLOCK]).expect("run");
        assert_eq!(
            2 * RATE - first.queued(),
            BLOCK,
            "the first block takes one block's worth of reference and no more"
        );
    }

    /// Everything a caller pushes reaches the canceller, in played order, however
    /// far ahead of the capture it was pushed. Nothing is discarded to hold the
    /// frontier back; it is deferred and read out later.
    ///
    /// The echo of the LAST part of a two-second pre-push is measured, so a feed
    /// that held the frontier back by throwing the overflow away would fail here
    /// even though it would pass every test that only measures the steady state.
    /// Regression: a bounded feed that discards rather than defers, which would
    /// make the echo of the discarded span uncancellable.
    #[cfg(feature = "aec")]
    #[test]
    fn a_reference_pushed_ahead_is_read_out_in_full() {
        const RATE: usize = 16_000;
        const BLOCK: usize = 320;
        const SECONDS: usize = 12;
        const PREPUSH: usize = 2 * RATE;
        let (far, near) = far_and_its_echo(SECONDS * RATE, 400, 0.25);

        let queue = Arc::new(AecReferenceRing::new(2 * RATE));
        let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &queue, RATE as u32);
        let mut pushed = PREPUSH;
        queue.push(&far[..pushed]);
        let mut out = Vec::new();
        for (block, piece) in near.chunks(BLOCK).enumerate() {
            let played = ((block + 1) * BLOCK).min(far.len());
            if played > pushed {
                queue.push(&far[pushed..played]);
                pushed = played;
            }
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");

        // Pushed, minus what the bound discarded, minus what is still queued, is
        // what the canceller received. Nothing was discarded and nothing is left,
        // so it received every sample.
        assert_eq!(pushed, far.len(), "the whole far end was pushed");
        assert_eq!(
            queue.dropped(),
            0,
            "a push inside the bound discards nothing"
        );
        assert_eq!(
            queue.queued(),
            0,
            "the feed reads the queue out rather than leaving it behind"
        );

        // Inside the second half of the pre-pushed span, past the lock: this
        // reference was pushed before the capture reached it and the echo of it
        // still cancels.
        let deferred = PREPUSH / 2 + RATE / 4..PREPUSH / 2 + RATE * 3 / 4;
        let removed = rms_db(&near[deferred.clone()]) - rms_db(&out[deferred]);
        assert!(
            removed >= 20.0,
            "the deferred part of the pre-push must still cancel \
             (removed {removed:.1} dB)"
        );
    }

    /// A synthetic echo of the pushed reference is cancelled. The whole stream is
    /// echo, so what the chain delivers IS the residual, and the measurement is
    /// the level the chain removed. Regression: a reference that never reaches the
    /// engine, whether from an unwired drain, a drain ordered after the near-end
    /// block, or a conversion that discards it.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_cancels_a_synthetic_echo_of_its_reference() {
        const RATE: usize = 16_000;
        const DELAY: usize = 400;
        const ECHO_GAIN: f32 = 0.25;
        let far = far_noise(6 * RATE);
        let mut near = vec![0.0f32; far.len()];
        for i in DELAY..near.len() {
            near[i] = ECHO_GAIN * far[i - DELAY];
        }

        let queue = Arc::new(AecReferenceRing::new(RATE));
        let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &queue, RATE as u32);
        // Each block's reference is pushed immediately before the block it echoes
        // into, the interleaving a caller pushing as it plays produces.
        let mut out = Vec::new();
        for (piece, reference) in near.chunks(320).zip(far.chunks(320)) {
            queue.push(reference);
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");
        assert_eq!(
            queue.dropped(),
            0,
            "a reference pushed in step with the capture must not overflow the queue"
        );

        // Measured past the delay search and the filter's convergence, and inside
        // a single stationary region so the measurement does not depend on where
        // the engine's own re-anchor at lock falls.
        let window = 3 * RATE..3 * RATE + RATE / 2;
        let removed = rms_db(&near[window.clone()]) - rms_db(&out[window]);
        assert!(
            removed >= 30.0,
            "the canceller must remove at least 30 dB of the echo (removed {removed:.1} dB)"
        );

        // The alignment it reached is reported, so a chain that cancels by some
        // other route than a locked delay does not pass silently.
        let metrics = chain.aec_metrics().expect("a canceller reports metrics");
        assert!(
            metrics.delay_samples.is_some(),
            "the canceller must report the alignment it locked onto"
        );
    }

    /// The canceller is constructed at the target rate and receives the signal
    /// after the resampler, so a reference declared at the target rate cancels an
    /// echo that was present in the device's native-rate capture. Regression: the
    /// stage drifting ahead of the resampler in the normalize segment, where it
    /// would be handed native-rate audio and cancel nothing.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_runs_after_the_resampler() {
        const TARGET: usize = 16_000;
        const NATIVE: usize = 48_000;
        const DELAY: usize = 1_200;
        const ECHO_GAIN: f32 = 0.25;

        // The reference is at the target rate; the echo it produced is in the
        // device's native-rate capture, so the test upsamples it to build one.
        let far = far_noise(4 * TARGET);
        let mut far_native = Vec::new();
        let mut upsampler =
            PolyphaseResampler::new(TARGET as u32, NATIVE as u32).expect("rate pair");
        upsampler.process(&far, &mut far_native).expect("upsample");
        upsampler.flush(&mut far_native);

        let mut near = vec![0.0f32; far_native.len()];
        for i in DELAY..near.len() {
            near[i] = ECHO_GAIN * far_native[i - DELAY];
        }

        let queue = Arc::new(AecReferenceRing::new(TARGET));
        let mut chain = aec_chain(
            1,
            NATIVE as u32,
            TARGET as u32,
            false,
            &queue,
            TARGET as u32,
        );
        let mut out = Vec::new();
        for (piece, reference) in near.chunks(960).zip(far.chunks(320)) {
            queue.push(reference);
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");

        // The echo as it stands after the resampler, which is what the canceller
        // had to remove: the same chain with echo cancellation off.
        let mut plain =
            build_capture_stage(1, 1, NATIVE as u32, TARGET as u32, Transforms::default())
                .expect("the chain builds")
                .expect("resample builds a chain");
        let echo_at_target = whole_stream(&mut plain, &near, 960);

        let window = 2 * TARGET..3 * TARGET;
        let removed = rms_db(&echo_at_target[window.clone()]) - rms_db(&out[window]);
        assert!(
            removed >= 20.0,
            "a target-rate reference must cancel the resampled echo (removed {removed:.1} dB)"
        );
    }

    /// A reference declared at a rate other than the capture target is converted
    /// by decibri before the canceller sees it, and still cancels. Regression: a
    /// mis-rated reference, which the canceller cannot detect and which cancels
    /// nothing while reporting no error.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_resamples_a_reference_declared_at_another_rate() {
        const TARGET: usize = 16_000;
        const REFERENCE: usize = 48_000;
        const DELAY: usize = 400;
        const ECHO_GAIN: f32 = 0.25;

        // The caller plays at 48 kHz and declares that rate; the echo reaches a
        // 16 kHz capture, so the test downsamples to build one.
        let far = far_noise(4 * REFERENCE);
        let mut far_at_target = Vec::new();
        let mut downsampler =
            PolyphaseResampler::new(REFERENCE as u32, TARGET as u32).expect("rate pair");
        downsampler
            .process(&far, &mut far_at_target)
            .expect("downsample");
        downsampler.flush(&mut far_at_target);

        let mut near = vec![0.0f32; far_at_target.len()];
        for i in DELAY..near.len() {
            near[i] = ECHO_GAIN * far_at_target[i - DELAY];
        }

        let queue = Arc::new(AecReferenceRing::new(REFERENCE));
        let mut chain = aec_chain(
            1,
            TARGET as u32,
            TARGET as u32,
            false,
            &queue,
            REFERENCE as u32,
        );
        let mut out = Vec::new();
        for (piece, reference) in near.chunks(320).zip(far.chunks(960)) {
            queue.push(reference);
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");

        let window = 2 * TARGET..3 * TARGET;
        let removed = rms_db(&near[window.clone()]) - rms_db(&out[window]);
        assert!(
            removed >= 20.0,
            "a reference declared at another rate must be converted and cancel (removed {removed:.1} dB)"
        );
    }

    /// A caller that stops pushing while nothing is playing keeps cancelling
    /// afterwards, at the level it reached before the pause.
    ///
    /// The far end goes silent for three seconds in the middle of the stream and
    /// the caller pushes nothing across it, which is what a synthesis pipeline
    /// that stops between utterances does. Regression: a drain that hands the
    /// engine nothing when the queue is empty, which leaves the far-end frontier
    /// permanently behind the capture. Measured against that regression, the
    /// stream below starves 190,416 of its 256,000 near-end samples and delivers
    /// -0.01 dB after the pause instead of the level asserted here, and it never
    /// recovers for the rest of the call.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_cancels_after_the_caller_stops_pushing() {
        const RATE: usize = 16_000;
        const DELAY: usize = 400;
        const ECHO_GAIN: f32 = 0.25;
        const SECONDS: usize = 12;
        const PAUSE_AT: usize = 4;
        const PAUSE: usize = 3;

        // The far end is silent across the pause either way. The two runs differ
        // only in whether the caller pushes that silence itself or stops pushing
        // and leaves decibri to supply it, which is the whole of the change.
        let mut far = far_noise(SECONDS * RATE);
        for sample in far[PAUSE_AT * RATE..(PAUSE_AT + PAUSE) * RATE].iter_mut() {
            *sample = 0.0;
        }
        let mut near = vec![0.0f32; far.len()];
        for i in DELAY..near.len() {
            near[i] = ECHO_GAIN * far[i - DELAY];
        }

        let run = |push_the_silence: bool| {
            let queue = Arc::new(AecReferenceRing::new(2 * RATE));
            let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &queue, RATE as u32);
            let mut out = Vec::new();
            for (block, piece) in near.chunks(320).enumerate() {
                let at = block * 320;
                let paused = (PAUSE_AT * RATE..(PAUSE_AT + PAUSE) * RATE).contains(&at);
                if push_the_silence || !paused {
                    queue.push(&far[at..at + piece.len()]);
                }
                out.extend_from_slice(chain.run(piece).expect("run"));
            }
            chain.flush(&mut out).expect("flush");
            (out, chain.aec_metrics().expect("metrics"), queue.silence())
        };

        let (pushed, _, pushed_silence) = run(true);
        let (unpushed, metrics, silence) = run(false);

        // The engine is handed the same far-end samples in the same order at the
        // same points in the capture, so the two streams are the same stream.
        assert_eq!(
            unpushed, pushed,
            "a caller that stops pushing must get what a caller pushing the \
             silence itself gets"
        );

        let window = (SECONDS - 1) * RATE..(SECONDS - 1) * RATE + RATE / 2;
        let removed = rms_db(&near[window.clone()]) - rms_db(&unpushed[window]);
        assert!(
            removed >= 30.0,
            "the canceller must still remove at least 30 dB after a pause \
             (removed {removed:.1} dB)"
        );

        // The far end was kept level with the capture across the pause, so the
        // engine found a far-end sample for every near-end sample and needed no
        // re-anchor to recover an alignment it never lost.
        assert_eq!(
            metrics.reference_starved, 0,
            "a paused caller must not starve the canceller"
        );
        assert_eq!(
            metrics.reference_reanchors, 0,
            "a paused caller must not force the canceller to re-anchor"
        );
        assert_eq!(
            silence,
            (PAUSE * RATE) as u64,
            "the silence decibri supplied is exactly the span the caller did not"
        );
        assert_eq!(
            pushed_silence, 0,
            "a caller pushing every block leaves nothing for decibri to supply"
        );
    }

    /// A caller that hands over a whole utterance in a single call, larger than
    /// the queue's previous one-second bound, locks and cancels.
    ///
    /// The push is 1.5 seconds, made once the stream is already running, which is
    /// where a talk-back application's first playback falls. Regression: a bound
    /// that truncates an ordinary utterance push, and an alignment that is built
    /// against a far-end frontier the caller was not asked to keep level.
    /// Measured against that regression this stream never locks at all:
    /// `delay_samples` stays `None` for its whole length.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_locks_on_a_reference_handed_over_a_whole_utterance_at_a_time() {
        const RATE: usize = 16_000;
        const DELAY: usize = 400;
        const ECHO_GAIN: f32 = 0.25;
        const SECONDS: usize = 12;
        // 1.5 s: larger than the one second the queue used to hold, and inside
        // the two seconds it holds now.
        const UTTERANCE: usize = 24_000;

        let mut far = far_noise(SECONDS * RATE);
        // Nothing plays for the first second, so the first push lands on a stream
        // that has already anchored.
        for sample in far[..RATE].iter_mut() {
            *sample = 0.0;
        }
        let mut near = vec![0.0f32; far.len()];
        for i in DELAY..near.len() {
            near[i] = ECHO_GAIN * far[i - DELAY];
        }

        let queue = Arc::new(AecReferenceRing::new(2 * RATE));
        let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &queue, RATE as u32);
        let mut out = Vec::new();
        // Each utterance is handed over in one call the moment the previous one
        // has finished playing, so the far-end frontier and the capture time of
        // the push are the same figure.
        let mut frontier = RATE;
        for (block, piece) in near.chunks(320).enumerate() {
            if block * 320 >= frontier && frontier < far.len() {
                let end = (frontier + UTTERANCE).min(far.len());
                queue.push(&far[frontier..end]);
                frontier = end;
            }
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");

        assert_eq!(
            queue.dropped(),
            0,
            "an utterance-sized push must fit the queue whole"
        );
        let metrics = chain.aec_metrics().expect("a canceller reports metrics");
        assert!(
            metrics.delay_samples.is_some(),
            "a caller that pushes an utterance at a time must still lock"
        );
        assert_eq!(
            metrics.reference_starved, 0,
            "a push the queue held whole starves nothing"
        );

        let window = (SECONDS - 1) * RATE..(SECONDS - 1) * RATE + RATE / 2;
        let removed = rms_db(&near[window.clone()]) - rms_db(&out[window]);
        assert!(
            removed >= 30.0,
            "an utterance-at-a-time caller must still cancel (removed {removed:.1} dB)"
        );
    }

    /// A push larger than the queue can hold costs the cancellation of the span
    /// it discarded and nothing else.
    ///
    /// The discarded samples are counted, and the span they occupied is handed to
    /// the engine as silence, so every later sample keeps the alignment it had.
    /// Regression: a discard that shortens the far-end timeline instead of
    /// blanking part of it, which moves every sample after it and ends
    /// cancellation for the rest of the call rather than for the discarded span.
    #[cfg(feature = "aec")]
    #[test]
    fn an_oversized_reference_push_costs_only_the_span_it_discarded() {
        const RATE: usize = 16_000;
        const DELAY: usize = 400;
        const ECHO_GAIN: f32 = 0.25;
        const SECONDS: usize = 12;
        const CAPACITY: usize = 2 * RATE;
        // Three seconds handed over at once against a two-second queue, so one
        // second of every push is discarded.
        const UTTERANCE: usize = 3 * RATE;

        let mut far = far_noise(SECONDS * RATE);
        for sample in far[..RATE].iter_mut() {
            *sample = 0.0;
        }
        let mut near = vec![0.0f32; far.len()];
        for i in DELAY..near.len() {
            near[i] = ECHO_GAIN * far[i - DELAY];
        }

        let queue = Arc::new(AecReferenceRing::new(CAPACITY));
        let mut chain = aec_chain(1, RATE as u32, RATE as u32, false, &queue, RATE as u32);
        let mut out = Vec::new();
        let mut frontier = RATE;
        for (block, piece) in near.chunks(320).enumerate() {
            if block * 320 >= frontier && frontier < far.len() {
                let end = (frontier + UTTERANCE).min(far.len());
                queue.push(&far[frontier..end]);
                frontier = end;
            }
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");

        // Pushes land at 1 s, 4 s, 7 s and 10 s. The first three hand over three
        // seconds each, of which the queue keeps two, so each discards one
        // second; the fourth has only the stream's last two seconds left to push
        // and fits whole.
        assert_eq!(
            queue.dropped(),
            3 * (UTTERANCE - CAPACITY) as u64,
            "the samples past the bound are counted"
        );
        let metrics = chain.aec_metrics().expect("a canceller reports metrics");
        assert!(
            metrics.delay_samples.is_some(),
            "a discard must not cost the canceller its lock"
        );
        assert_eq!(
            metrics.reference_starved, 0,
            "the discarded span is handed over as silence, so nothing starves"
        );

        // Inside the second push's kept span (4 s to 6 s), converged and with the
        // reference intact.
        let kept = 5 * RATE..5 * RATE + RATE / 2;
        let removed = rms_db(&near[kept.clone()]) - rms_db(&out[kept]);
        assert!(
            removed >= 30.0,
            "the span the queue held must still cancel (removed {removed:.1} dB)"
        );

        // Inside the same push's discarded span (6 s to 7 s), where the engine was
        // handed silence and there is nothing to cancel against. Pinned so the
        // cost of a discard stays bounded to the span it discarded rather than
        // being assumed to be.
        let discarded = 6 * RATE + RATE / 4..6 * RATE + RATE * 3 / 4;
        let unremoved = rms_db(&near[discarded.clone()]) - rms_db(&out[discarded]);
        assert!(
            unremoved.abs() < 3.0,
            "the discarded span passes through (level moved {unremoved:.1} dB)"
        );
    }

    /// A chain whose canceller received no near-end audio emits nothing at flush,
    /// whether or not a reference was pushed. Regression: the phantom tail, where
    /// a stage that never processed a sample assembles one at close out of its own
    /// padding.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_emits_nothing_at_flush_without_near_audio() {
        for pushed in [0usize, 4_000] {
            let queue = Arc::new(AecReferenceRing::new(16_000));
            let mut chain = aec_chain(1, 16_000, 16_000, false, &queue, 16_000);
            if pushed > 0 {
                queue.push(&far_noise(pushed));
            }
            let mut tail = Vec::new();
            chain.flush(&mut tail).expect("flush");
            assert!(
                tail.is_empty(),
                "a canceller with no near-end audio must emit nothing at flush (pushed {pushed})"
            );
        }
    }

    /// The reference queue keeps the oldest contiguous run in played order,
    /// discards from the newest end when it is full, and counts what it discarded.
    /// A bounded drain takes a prefix and leaves the rest in played order.
    /// Regression: unbounded growth on a caller that pushes faster than the
    /// capture consumes, a hole punched between samples already fed to the
    /// canceller and samples still queued, which moves the alignment rather than
    /// shortening the reference, and a bounded drain that takes off the newest
    /// end and so hands the canceller its far end out of order.
    #[cfg(feature = "aec")]
    #[test]
    fn reference_queue_keeps_the_oldest_run_and_counts_what_it_dropped() {
        let queue = AecReferenceRing::new(8);
        queue.push(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert_eq!(queue.dropped(), 0, "a push that fits discards nothing");

        queue.push(&[6.0, 7.0, 8.0, 9.0, 10.0]);
        assert_eq!(
            queue.dropped(),
            2,
            "the samples past the bound are discarded and counted"
        );
        assert_eq!(queue.queued(), 8, "the queue holds what it kept");

        // A bounded take reads the OLDEST samples and leaves the rest queued.
        let mut drained = Vec::new();
        queue.drain_into(&mut drained, 3);
        assert_eq!(
            drained,
            vec![1.0, 2.0, 3.0],
            "a bounded take reads the oldest samples, in played order"
        );
        assert_eq!(queue.queued(), 5, "the remainder stays queued");

        queue.drain_into(&mut drained, 64);
        assert_eq!(
            drained,
            vec![4.0, 5.0, 6.0, 7.0, 8.0],
            "a take larger than the queue reads the rest, still in played order"
        );
        assert_eq!(queue.queued(), 0, "and leaves the queue empty");

        // A drained queue takes its full capacity again, and the count is
        // cumulative across drains.
        queue.push(&[0.5f32; 11]);
        assert_eq!(
            queue.dropped(),
            5,
            "the discard count accumulates across drains"
        );
        drained.clear();
        queue.drain_into(&mut drained, usize::MAX);
        assert_eq!(drained.len(), 8, "the bound holds after a drain");

        // A take of nothing leaves the queue untouched.
        queue.push(&[0.25f32; 4]);
        queue.drain_into(&mut drained, 0);
        assert!(drained.is_empty(), "a take of nothing reads nothing");
        assert_eq!(queue.queued(), 4, "and leaves the queue as it was");
    }

    /// With the canceller in the normalize segment and a same-length conditioning
    /// stage after it, the detector tap and the delivered output stay in lockstep
    /// across the whole stream, including through the canceller's own re-anchor at
    /// delay lock, which shortens the delivered stream once. Regression: the
    /// canceller landing on the far side of the tap, where its re-blocking would
    /// separate the two and the detector's input would stop corresponding to the
    /// audio delivered.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_leaves_the_tap_in_lockstep_with_the_delivered_output() {
        const RATE: usize = 16_000;
        const DELAY: usize = 400;
        let far = far_noise(4 * RATE);
        let mut near = vec![0.0f32; far.len()];
        for i in DELAY..near.len() {
            near[i] = 0.25 * far[i - DELAY];
        }

        let queue = Arc::new(AecReferenceRing::new(RATE));
        // DC removal is same-length, so without the canceller the tap and the
        // delivered output advance one for one; the canceller must not change that.
        let mut chain = aec_chain(1, RATE as u32, RATE as u32, true, &queue, RATE as u32);
        assert!(
            chain.has_transform(),
            "DC removal builds a transform segment, so a tap is captured"
        );

        let mut tapped = 0usize;
        let mut delivered = 0usize;
        for (piece, reference) in near.chunks(320).zip(far.chunks(320)) {
            queue.push(reference);
            delivered += chain.run(piece).expect("run").len();
            tapped += chain.tap().len();
            assert_eq!(
                tapped, delivered,
                "the tap and the delivered output advance together"
            );
        }
        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush");
        assert_eq!(
            tapped + chain.tap().len(),
            delivered + tail.len(),
            "the tap and the delivered output stay in lockstep through close"
        );
    }

    /// Echo cancellation alone leaves the chain with no transform segment, so no
    /// detector tap is allocated and the delivered output already is the
    /// echo-removed signal a detector should read. It also adds nothing to the
    /// transform latency, because it is a normalize stage. Regression: the stage
    /// landing in the transform segment, which would put the tap ahead of the
    /// canceller and feed the detector the echo it was enabled to remove.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_only_chain_leaves_the_tap_inactive_and_adds_no_transform_latency() {
        let queue = Arc::new(AecReferenceRing::new(16_000));
        let mut chain = aec_chain(1, 16_000, 16_000, false, &queue, 16_000);
        assert!(
            !chain.has_transform(),
            "echo cancellation alone builds no transform segment"
        );
        assert_eq!(
            chain.transform_latency(),
            0,
            "a normalize stage adds no transform latency"
        );
        chain.run(&mono_signal(1_600)).expect("run");
        assert!(
            chain.tap().is_empty(),
            "no tap is captured without a transform segment"
        );
    }

    /// The chain reports the canceller's metrics, and only the chain that holds
    /// one. Regression: a metrics accessor that answers from the wrong segment or
    /// silently reports nothing once the stage is present.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_metrics_come_from_the_chain_that_holds_the_canceller() {
        let plain = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
            .expect("the chain builds")
            .expect("downmix builds a chain");
        assert!(
            plain.aec_metrics().is_none(),
            "a chain with no canceller reports no metrics"
        );

        let queue = Arc::new(AecReferenceRing::new(16_000));
        let mut chain = aec_chain(1, 16_000, 16_000, false, &queue, 16_000);
        chain.run(&mono_signal(1_600)).expect("run");
        let metrics = chain.aec_metrics().expect("a canceller reports metrics");
        assert_eq!(
            metrics.delay_samples, None,
            "no reference was pushed, so no alignment was found"
        );
        assert!(
            metrics.acquisition_parked > 0,
            "the far-end read is parked while no reference flows"
        );
    }

    /// An unknown model name is rejected by the canceller's own parse, and the
    /// rejection names the models a caller may select. Regression: a copy of the
    /// model list inside decibri, which is a second place to update when the set
    /// grows and a place for the two to disagree.
    #[cfg(feature = "aec")]
    #[test]
    fn an_unknown_aec_model_name_is_rejected_by_the_cancellers_parse() {
        let err = "tao"
            .parse::<AecModel>()
            .expect_err("an unknown model name must not parse");
        let converted = DecibriError::from(err);
        assert!(
            matches!(converted, DecibriError::AecConfigInvalid { .. }),
            "an unknown model reaches AecConfigInvalid, got {converted:?}"
        );
        let message = converted.to_string();
        assert!(
            message.contains("'tao'") && message.contains("'tau'"),
            "the rejection names the received string and the available models: {message}"
        );

        // Every publicly selectable name parses, so the set decibri delegates to
        // is the set the canceller publishes.
        for name in AecModel::PUBLIC_MODEL_NAMES {
            assert!(
                name.parse::<AecModel>().is_ok(),
                "the published model name {name} must parse"
            );
        }
    }
}
