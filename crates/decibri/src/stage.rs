//! Internal capture stage chain.
//!
//! A [`CaptureStage`] conditions raw device capture into the canonical format the
//! consumer receives, running between the cpal callback's native buffers and the
//! exact-size reblock in [`crate::microphone`]. The chain has two segments. The
//! `normalize` segment holds up to four stages: [`Downmix`], which averages a
//! multichannel device down to mono (or the [`Select`] gather a channel map
//! replaces it with), then [`Deinterleave`] when more than one channel
//! continues past that point, then [`ResampleStage`], which converts the
//! device's native sample rate to the requested target rate with one engine
//! per carried channel, then the echo canceller, which removes the echo of
//! caller-supplied far-end audio. The channel stage runs first so every later
//! stage sees the delivered channel count; the echo
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
//! Audio enters the chain INTERLEAVED, and every signal that leaves it (the
//! delivered output and the detector tap) is interleaved again; in between,
//! the chain runs PLANAR: one contiguous run per channel. A block crossing a
//! stage boundary carries the channel count it is laid out at, as a
//! [`Block`]. The [`Deinterleave`] stage rearranges the block once at the
//! head of the chain, in the position [`Downmix`] occupies or immediately
//! after the channel stage standing there (that stage reads the interleaved
//! device frames), and the block is re-interleaved at each delivery. At one
//! channel the two layouts are the same arrangement, so no rearrangement is
//! built. Interleaved at the boundaries and
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

    /// The echo cancellers' transport and cancellation metrics, one entry per
    /// delivered channel in delivered order, when this stage holds cancellers.
    ///
    /// `None` for every other stage, so [`CaptureStage::aec_metrics`] finds the
    /// one stage that answers without downcasting a boxed trait object. The
    /// returned vector is never empty: a stage that holds cancellers holds at
    /// least one.
    #[cfg(feature = "aec")]
    fn aec_metrics(&self) -> Option<Vec<AecMetrics>> {
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

/// Gather named device channels out of interleaved multichannel audio.
///
/// The map is a list of 0-based device channel indices, one entry per
/// delivered channel: delivered channel `j` of each output frame is device
/// channel `map[j]` of the matching input frame, in the order the map gives.
/// The same shape as CoreAudio AUHAL's channel map
/// (`kAudioOutputUnitProperty_ChannelMap`, an array of device channel indices,
/// one per client channel); NOT miniaudio's `channelMap`, which is a spatial
/// layout. Duplicate entries are permitted (each delivered channel is an
/// independent copy of its source) and order is meaningful.
///
/// Pushed in place of [`Downmix`] when
/// [`crate::microphone::MicrophoneConfig::channel_map`] names a map; the
/// entries are validated against the device's own report before the chain is
/// built, and no fixed maximum is enforced anywhere in this stage.
struct Select {
    /// The device channel count each interleaved input frame carries.
    channels: u16,
    /// The 0-based device channel indices to gather, one per delivered
    /// channel. Every entry is below `channels`; the caller validated that
    /// against the device's report.
    map: Vec<u16>,
}

impl Select {
    fn new(channels: u16, map: &[u16]) -> Self {
        Self {
            channels,
            map: map.to_vec(),
        }
    }
}

impl Stage for Select {
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The count the stage was built for and the count the block carries are
        // the same number reaching it by two routes, exactly as for `Downmix`.
        debug_assert_eq!(
            input.channels(),
            self.channels,
            "the select was built for {} channels and handed {}",
            self.channels,
            input.channels()
        );
        let frames = input.frames();
        out.reserve(frames * self.map.len());
        // `chunks_exact` truncates a trailing partial frame, matching
        // `Block::frames` and `sample::downmix_to_mono`.
        for frame in input.samples().chunks_exact(self.channels as usize) {
            for &index in &self.map {
                out.push(frame[index as usize]);
            }
        }
        debug_assert_eq!(
            out.len(),
            frames * self.map.len(),
            "the select emits exactly one sample per map entry per input frame"
        );
        Ok(())
    }

    /// The map's own length: one delivered channel per entry, whatever the
    /// input count.
    fn output_channels(&self, _input_channels: u16) -> u16 {
        self.map.len() as u16
    }
}

/// Rearrange one interleaved buffer into the planar layout, appended to `dst`:
/// `channels` contiguous runs, one per channel, each run holding that
/// channel's sample from every frame in order. Any trailing partial frame is
/// truncated, matching [`Block::frames`] and [`sample::downmix_to_mono`].
fn deinterleave_into(samples: &[f32], channels: u16, dst: &mut Vec<f32>) {
    let channels = channels.max(1) as usize;
    let frames = samples.len() / channels;
    dst.reserve(frames * channels);
    for channel in 0..channels {
        dst.extend(samples.chunks_exact(channels).map(|frame| frame[channel]));
    }
}

/// Rearrange one planar buffer (as [`deinterleave_into`] lays it out) back
/// into interleaved frames, appended to `dst`. The buffer's length must be a
/// multiple of `channels`: the chain only re-interleaves buffers whose runs it
/// built at equal length.
fn interleave_into(planar: &[f32], channels: u16, dst: &mut Vec<f32>) {
    let channels = channels.max(1) as usize;
    debug_assert!(
        planar.len().is_multiple_of(channels),
        "a planar buffer holds equal-length runs: {} samples cannot split into {} channels",
        planar.len(),
        channels
    );
    let frames = planar.len() / channels;
    dst.reserve(frames * channels);
    for frame in 0..frames {
        dst.extend((0..channels).map(|channel| planar[channel * frames + frame]));
    }
}

/// Splice one planar buffer onto another run by run: run i of `carried`
/// becomes carried's run i followed by `tail`'s run i, so a stage's drained
/// tail continues each channel's own run. With `carried` empty the result is
/// `tail`'s runs unchanged. Both buffers hold `channels` equal-length runs.
fn splice_planar_tail(carried: &mut Vec<f32>, tail: &[f32], channels: u16) {
    let channels = channels.max(1) as usize;
    debug_assert!(
        carried.len().is_multiple_of(channels) && tail.len().is_multiple_of(channels),
        "planar buffers hold equal-length runs: {} and {} samples cannot both split into {} channels",
        carried.len(),
        tail.len(),
        channels
    );
    let carried_frames = carried.len() / channels;
    let tail_frames = tail.len() / channels;
    let mut spliced = Vec::with_capacity(carried.len() + tail.len());
    for channel in 0..channels {
        spliced
            .extend_from_slice(&carried[channel * carried_frames..(channel + 1) * carried_frames]);
        spliced.extend_from_slice(&tail[channel * tail_frames..(channel + 1) * tail_frames]);
    }
    *carried = spliced;
}

/// Rearrange interleaved frames into the planar layout the chain runs inside:
/// one contiguous run per channel. Pushed once at the head of the `normalize`
/// segment when more than one channel continues past the channel stage; a
/// chain carrying one channel builds no rearrangement, the two layouts being
/// the same arrangement there. The count passes through unchanged.
struct Deinterleave;

impl Stage for Deinterleave {
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        deinterleave_into(input.samples(), input.channels(), out);
        Ok(())
    }
}

/// Resample planar audio from the device's native rate to the requested target
/// rate: one owned [`PolyphaseResampler`] per carried channel, each engine
/// processing its channel's contiguous run.
///
/// [`build_capture_stage`] adds this stage only when the native and target
/// rates differ; a device already at the target rate omits it. It is placed
/// after the channel stage (and [`Deinterleave`]) in the chain, so each engine
/// receives one channel's run, the mono stream its documented contract
/// requires. Every engine is constructed from the same rate pair, so the
/// engine choice, the coefficient table and the group delay are identical
/// across channels, and equal-length input runs yield equal-length output
/// runs on every call: the emission cadence is a function of the rate pair
/// and the count of samples fed, never of their values.
struct ResampleStage {
    /// One engine per carried channel, index i processing planar run i. The
    /// length is resolved at chain build time; no fixed maximum exists.
    engines: Vec<PolyphaseResampler>,
}

impl ResampleStage {
    /// Build `channels` engines for the `native_rate` to `target_rate` pair.
    /// Construction validates the rate pair once per engine, identically; the
    /// `?` bridges a failure to `DecibriError::ResampleConfigInvalid` via
    /// `From<ResamplerError>`.
    fn new(native_rate: u32, target_rate: u32, channels: u16) -> Result<Self, DecibriError> {
        let engines = (0..channels.max(1))
            .map(|_| PolyphaseResampler::new(native_rate, target_rate))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { engines })
    }
}

impl Stage for ResampleStage {
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The count the stage was built for and the count the block carries are
        // the same number reaching it by two routes, exactly as for `Downmix`
        // and `Select`. A block at another count would hand at least one engine
        // a run that is not one channel's samples.
        debug_assert_eq!(
            input.channels() as usize,
            self.engines.len(),
            "the resampler was built for {} channels and handed {}",
            self.engines.len(),
            input.channels()
        );
        // Each engine appends its channel's output; equal-length runs in yield
        // equal-length runs out, which is what keeps the planar layout valid
        // downstream. The one steady-path error rejects a block that arrives
        // after the flush; the callers stop feeding a flushed chain, so it is
        // propagated rather than assumed away, and it strikes every engine at
        // the same call or none.
        let frames = input.frames();
        let mut emitted = None;
        for (channel, engine) in self.engines.iter_mut().enumerate() {
            let before = out.len();
            engine.process(
                &input.samples()[channel * frames..(channel + 1) * frames],
                out,
            )?;
            let count = out.len() - before;
            debug_assert!(
                emitted.is_none_or(|e: usize| e == count),
                "identical engines fed equal-length runs emit equal-length runs"
            );
            emitted = Some(count);
        }
        Ok(())
    }

    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // Drain each engine's group-delay tail (and any partial-frame carry)
        // into `out`, in channel order, so the appended tail is planar like
        // every processed block. Called once at close; the engines append and
        // are infallible, so wrap it as `Ok(())`. Identical engines hold
        // identical-length tails, which the assertion states.
        let mut emitted = None;
        for engine in self.engines.iter_mut() {
            let before = out.len();
            engine.flush(out);
            let count = out.len() - before;
            debug_assert!(
                emitted.is_none_or(|e: usize| e == count),
                "identical engines hold equal-length tails"
            );
            emitted = Some(count);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        // Forward one engine's group delay: every engine is constructed from
        // the same rate pair, so the figure is identical across channels and
        // the channels stay aligned. It is already expressed at the output
        // rate, and is zero when the rates match and the engine is a
        // passthrough.
        self.engines
            .first()
            .map_or(0, |engine| engine.latency_samples())
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

    /// The queued samples in played order, front to back. Test support like
    /// [`queued`](Self::queued): the tests that pin what the push leaves in the
    /// queue read the contents through this; the capture path only ever reads
    /// through [`drain_into`](Self::drain_into).
    #[cfg(test)]
    pub(crate) fn queued_samples(&self) -> Vec<f32> {
        self.queued
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .iter()
            .copied()
            .collect()
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
/// the channel stage (and [`Deinterleave`]) and [`ResampleStage`], so the
/// stage receives the delivered channels at the target rate, planar above one
/// channel. It holds one engine per delivered channel, index i cancelling
/// planar run i, exactly as [`ResampleStage`] holds one resampler per carried
/// channel: each engine receives one channel's run, the mono stream its
/// documented contract requires. Running before the VAD tap is taken means the
/// detector reads the echo-removed signal rather than the raw capture.
///
/// Each [`process`](Stage::process) feeds the reference into every engine
/// before cancelling the near-end block, and feeds it at the rate the capture
/// consumes it: enough to reach the near-end frontier this block ends at, and
/// no further. Whatever the caller pushed beyond that stays queued and is read
/// on later blocks, so nothing is discarded and the far-end frontier never
/// runs ahead of the capture. The engine re-blocks internally, so a call may
/// append fewer or more samples than it consumed and the totals balance after
/// [`flush`](Stage::flush).
///
/// The far-end frontier is what each canceller measures its alignment from,
/// and it measures it once, on the first block it sees. A frontier that has
/// run ahead of the capture puts every later near-end sample past the
/// reference the canceller holds, which its delay search reads as a window it
/// cannot score, so it never locks and never recovers. Feeding at the
/// capture's own rate is what holds the two together: see
/// [`feed_reference`](AecStage::feed_reference).
///
/// The feed then tops the far-end stream up with silence to the near-end
/// frontier, so the two streams advance together whatever the caller does. That
/// is the played signal, not a substitute for it: while nothing is playing the
/// far end IS silence, and a caller who pushes nothing is saying so. Without the
/// top-up a caller who pauses leaves the far-end frontier permanently behind the
/// capture, which the engine can only read as a reference that never arrives.
///
/// Above one channel the near-end feed is re-blocked (see the `pending`
/// field): every engine receives whole multiples of its own
/// `latency_samples()` per call, which holds the engines' emitted counts
/// identical however their delay searches behave. At one channel the input is
/// handed to the single engine directly, so the call cadence of a
/// single-channel capture is untouched by the re-blocker's existence.
#[cfg(feature = "aec")]
struct AecStage {
    /// One engine per delivered channel, index i cancelling planar run i, each
    /// constructed at the capture target rate from the same configuration. The
    /// length is resolved at chain build time; no fixed maximum exists.
    /// Nothing is shared between engines, including each engine's internal
    /// far-end history: every engine is fed the same reference and the same
    /// per-call near lengths, and each finds its own channel's echo delay.
    engines: Vec<Aec>,
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
    /// The near-end re-blocker: one carry run per channel, holding the frames
    /// received but not yet handed to the engines. Used only above one
    /// channel; at one channel the input goes to the engine directly and
    /// these stay empty.
    ///
    /// Each engine is fed whole multiples of [`Self::block`] per call, so its
    /// internal partial-block carry is empty at every call boundary. That is
    /// load-bearing: when an engine's delay acquisition promotes a lock it
    /// resets its canceller, and the reset clears the canceller's internal
    /// carry before the current call's samples are pushed, so what a
    /// promotion can discard is `(samples fed) mod block`. Feeding in block
    /// multiples pins that at zero for every engine, which is what keeps N
    /// engines emitting identical counts however their promotions fall. The
    /// reset-before-push ordering is behaviour the canceller crate exhibits,
    /// not a contract it documents; a canceller change that reorders the two
    /// breaks the equal-count property silently, with no error on either
    /// side.
    pending: Vec<Vec<f32>>,
    /// The engines' shared framing granularity, read from
    /// `latency_samples()` at construction: identical engines report an
    /// identical figure. The re-blocker feeds in whole multiples of this.
    block: usize,
}

#[cfg(feature = "aec")]
impl AecStage {
    /// Build the stage for a capture at `target_rate` carrying `channels`
    /// delivered channels: one engine per channel, all from the same
    /// configuration. A count of zero is floored to one, as [`Block::new`]
    /// floors its channel count.
    ///
    /// Every engine is constructed at `target_rate`, which is the rate the
    /// signal reaching this stage carries. Returns an error when the canceller
    /// rejects the configuration (bridged by `From<AecError>`) or when the
    /// reference rate pair is one the resampler cannot serve; construction
    /// validates the configuration once per engine, identically, so a failure
    /// strikes the first engine or none.
    fn new(settings: AecSettings, target_rate: u32, channels: u16) -> Result<Self, DecibriError> {
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
        let channels = channels.max(1);
        let engines = (0..channels)
            .map(|_| Aec::new(config.clone()))
            .collect::<Result<Vec<_>, _>>()?;
        let block = engines[0].latency_samples();

        // Converting on the drain side keeps the caller's push to a copy into the
        // queue, so the thread that produced the audio pays nothing for the
        // conversion. One resampler serves every engine: the reference is mono
        // and is drained once, so the converted block is handed to each engine
        // in turn.
        let reference_resampler = if reference_rate != target_rate {
            Some(PolyphaseResampler::new(reference_rate, target_rate)?)
        } else {
            None
        };

        Ok(Self {
            engines,
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
            pending: (0..channels).map(|_| Vec::new()).collect(),
            block,
        })
    }

    /// Feed the queued reference into every engine, at most `budget` samples of
    /// it measured at the capture target rate, converting from the declared rate
    /// first when the two differ. The queue is drained once and the drained
    /// block handed to each engine in turn, so the caller's single push serves
    /// every channel; `budget` and [`Self::far_fed`] are per-engine counts, the
    /// same for all engines because all are fed identically.
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
                for engine in self.engines.iter_mut() {
                    engine.feed_reference(&self.resampled);
                }
                self.far_fed += self.resampled.len() as u64;
            }
            None => {
                for engine in self.engines.iter_mut() {
                    engine.feed_reference(&self.drained);
                }
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
        for engine in self.engines.iter_mut() {
            engine.feed_reference(&self.silence[..fill]);
        }
        self.far_fed += fill as u64;
        self.reference.record_silence(fill as u64);
    }
}

#[cfg(feature = "aec")]
impl Stage for AecStage {
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The count the stage was built for and the count the block carries are
        // the same number reaching it by two routes, exactly as for `Downmix`
        // and `Select`. A block at another count would hand at least one engine
        // a run that is not one channel's samples.
        debug_assert_eq!(
            input.channels() as usize,
            self.engines.len(),
            "the echo canceller was built for {} channels and handed {}",
            self.engines.len(),
            input.channels()
        );
        if self.engines.len() == 1 {
            // One channel: the input goes to the engine directly, with no
            // re-blocking, so the engine sees exactly the call cadence the
            // native buffers produce. The engine drives its reference feeding
            // and re-anchor evaluation off the lengths of these calls, which
            // is why the single-channel path bypasses the re-blocker rather
            // than passing through it with a window of one.
            //
            // The reference goes in first, up to the frontier this block ends
            // at, and the top-up then covers whatever of the block the caller
            // left uncovered. The budget is the distance the far end has left
            // to travel to reach that frontier, so a caller feeding in step
            // hands over its whole block and a caller running ahead hands
            // over one block's worth of its backlog.
            let near_frontier = self.near_seen + input.len() as u64;
            self.feed_reference(near_frontier.saturating_sub(self.far_fed))?;
            if !self.reference_pending {
                self.fill_reference_to(near_frontier);
            }
            self.near_seen = near_frontier;
            if !input.is_empty() {
                self.received_near = true;
            }
            // The engine appends its output in step with the near-end samples
            // that produced it. Its `latency_samples` is a buffering budget,
            // not an index offset, so nothing here shifts the output by it.
            self.engines[0].process(input.samples(), out)?;
            return Ok(());
        }
        // Above one channel the input is planar: one run per channel, appended
        // to that channel's carry. The engines are then fed the largest whole
        // multiple of `block` the carry holds, every channel the same count,
        // and the remainder stays carried. See the `pending` field for why the
        // block-multiple feed is load-bearing and not a convenience.
        let frames = input.frames();
        if frames > 0 {
            self.received_near = true;
        }
        for (channel, pend) in self.pending.iter_mut().enumerate() {
            pend.extend_from_slice(&input.samples()[channel * frames..(channel + 1) * frames]);
        }
        let held = self.pending[0].len();
        let feed = held - (held % self.block);
        if feed == 0 {
            // No whole block yet: nothing reaches the engines, the near
            // frontier does not advance, and no reference is drained for it.
            return Ok(());
        }
        // The reference accounting is per engine and identical across engines:
        // one drain covers all of them (see `feed_reference`), and the frames
        // fed this call are what the frontier advances by.
        let near_frontier = self.near_seen + feed as u64;
        self.feed_reference(near_frontier.saturating_sub(self.far_fed))?;
        if !self.reference_pending {
            self.fill_reference_to(near_frontier);
        }
        self.near_seen = near_frontier;
        // Each engine consumes `feed` samples of its own channel and, having
        // been fed a whole multiple of its block from an empty internal carry,
        // appends exactly `feed` samples: equal-length runs in, equal-length
        // runs out, which is what keeps the planar layout valid downstream.
        let mut emitted = None;
        for (channel, engine) in self.engines.iter_mut().enumerate() {
            let before = out.len();
            engine.process(&self.pending[channel][..feed], out)?;
            let count = out.len() - before;
            debug_assert!(
                emitted.is_none_or(|e: usize| e == count),
                "identical engines fed equal block-multiple runs emit equal-length runs"
            );
            emitted = Some(count);
        }
        for pend in self.pending.iter_mut() {
            pend.drain(..feed);
        }
        Ok(())
    }

    fn flush(&mut self, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // A stage that never saw near-end audio holds no framing carry, so it
        // emits nothing rather than a tail with no source.
        if !self.received_near {
            return Ok(());
        }
        // The re-blocker's held remainder goes into the engines first, exactly
        // as a steady-state block would: rate-matched reference, then the near
        // samples, so the engines' own frontier accounting sees an ordinary
        // call. Shorter than one block, each engine banks its run as internal
        // carry and appends nothing; the carry comes out in the drain below.
        let remainder = self.pending.first().map_or(0, Vec::len);
        if remainder > 0 {
            let near_frontier = self.near_seen + remainder as u64;
            self.feed_reference(near_frontier.saturating_sub(self.far_fed))?;
            if !self.reference_pending {
                self.fill_reference_to(near_frontier);
            }
            self.near_seen = near_frontier;
            for (channel, engine) in self.engines.iter_mut().enumerate() {
                engine.process(&self.pending[channel], out)?;
            }
            for pend in self.pending.iter_mut() {
                pend.clear();
            }
        }
        // Everything still queued, then the reference resampler's own
        // group-delay tail, both before the engines drain: the far end is
        // complete when the near end's carry comes out, and a caller's push is
        // never left unread. The budget the per-block feed is rate-matched by
        // does not apply here, because there is no near-end audio left for the
        // frontier to run ahead OF: the capture has ended.
        self.feed_reference(u64::MAX)?;
        if let Some(resampler) = &mut self.reference_resampler {
            self.resampled.clear();
            resampler.flush(&mut self.resampled);
            if !self.resampled.is_empty() {
                for engine in self.engines.iter_mut() {
                    engine.feed_reference(&self.resampled);
                }
            }
        }
        // Drain each engine's end-of-stream carry, in channel order, so the
        // appended tail is planar like every processed block. Identical
        // engines fed identically hold identical-length tails, which the
        // assertion states.
        let mut emitted = None;
        for engine in self.engines.iter_mut() {
            let before = out.len();
            engine.flush(out)?;
            let count = out.len() - before;
            debug_assert!(
                emitted.is_none_or(|e: usize| e == count),
                "identical engines hold identical-length tails"
            );
            emitted = Some(count);
        }
        Ok(())
    }

    fn latency_samples(&self) -> usize {
        // Forward one engine's constant framing figure: every engine is
        // constructed from the same configuration, so the figure is identical
        // across channels. It is the amount by which the emitted count trails
        // the consumed count, which is what this trait reports, and it is
        // summed only across the `transform` segment that this stage is not
        // part of.
        self.engines
            .first()
            .map_or(0, |engine| engine.latency_samples())
    }

    fn aec_metrics(&self) -> Option<Vec<AecMetrics>> {
        Some(self.engines.iter().map(Aec::metrics).collect())
    }
}

/// A same-length DSP step: filters a buffer of samples in place, producing
/// exactly as many output samples as it received. The adapter [`PerChannel`]
/// wraps any `InPlaceDsp` as a [`Stage`], one instance per channel.
///
/// `Send` so the wrapped step can live behind the stream's `Mutex`, matching the
/// [`Stage`] bound. `pub(crate)` so a stage authored in another module (the
/// [`crate::gain::LevelControl`] engine) can implement it.
pub(crate) trait InPlaceDsp: Send {
    /// Filter `samples` in place: each element is replaced by its processed
    /// value, the length unchanged.
    fn process_in_place(&mut self, samples: &mut [f32]);
}

/// Adapts an [`InPlaceDsp`] to the [`Stage`] interface over any channel count:
/// one wrapped instance per channel, each filtering its own planar run in
/// place, so the recursive state every `InPlaceDsp` carries (the DC blocker's
/// and biquad's filter memory) advances along one channel and never across
/// two. The output length equals the input length; `flush` keeps the
/// [`Stage`] default no-op, since a same-length DSP holds no end-of-stream
/// tail.
struct PerChannel<T: InPlaceDsp> {
    /// One instance per channel, index i filtering planar run i. The length is
    /// resolved at chain build time; no fixed maximum exists.
    instances: Vec<T>,
}

impl<T: InPlaceDsp> PerChannel<T> {
    /// One `instance()` per channel. A count of zero is floored to one, as
    /// [`Block::new`] floors its channel count.
    fn new(channels: u16, instance: impl Fn() -> T) -> Self {
        Self {
            instances: (0..channels.max(1)).map(|_| instance()).collect(),
        }
    }
}

impl<T: InPlaceDsp> Stage for PerChannel<T> {
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        // The count the wrapper was built for and the count the block carries
        // are the same number reaching it by two routes, exactly as for
        // `Downmix` and `Select`. A block at another count would hand at least
        // one instance a run that is not one channel's samples, filtering its
        // recursive state across a channel boundary.
        debug_assert_eq!(
            input.channels() as usize,
            self.instances.len(),
            "the per-channel wrapper was built for {} channels and handed {}",
            self.instances.len(),
            input.channels()
        );
        let frames = input.frames();
        debug_assert_eq!(
            input.len(),
            frames * self.instances.len(),
            "a planar block holds equal-length runs"
        );
        // The caller clears `out` first, so this is an exact-length copy of the
        // input that each instance then rewrites in place over its own run: one
        // input sample, one output sample.
        out.extend_from_slice(input.samples());
        for (channel, instance) in self.instances.iter_mut().enumerate() {
            instance.process_in_place(&mut out[channel * frames..(channel + 1) * frames]);
        }
        Ok(())
    }
}

/// A same-length DSP step whose detection runs across channels: filters a
/// planar buffer in place, reading each frame across every channel and
/// applying one shared gain to all of them, the length unchanged. The adapter
/// [`Linked`] wraps any `LinkedDsp` as a [`Stage`].
///
/// `Send` for the same reason as [`InPlaceDsp`]; `pub(crate)` so the engines
/// authored in [`crate::gain`] can implement it.
pub(crate) trait LinkedDsp: Send {
    /// Filter `planar` in place: `channels` contiguous equal-length runs, one
    /// detector reading per frame across the channels, one gain applied to
    /// every channel of that frame.
    fn process_planar(&mut self, planar: &mut [f32], channels: u16);
}

/// Adapts a [`LinkedDsp`] to the [`Stage`] interface. `process` copies the
/// input into `out` then filters it in place at the block's own channel
/// count, so the output length equals the input length; `flush` keeps the
/// [`Stage`] default no-op. The wrapped engine holds one detector and one
/// gain whatever the count, so no per-instance count is stored; the chain's
/// resolved-count assertions in the walks hold the count steady.
struct Linked<T: LinkedDsp>(T);

impl<T: LinkedDsp> Stage for Linked<T> {
    fn process(&mut self, input: Block<'_>, out: &mut Vec<f32>) -> Result<(), DecibriError> {
        out.extend_from_slice(input.samples());
        self.0.process_planar(out, input.channels());
        Ok(())
    }
}

/// One-pole DC-blocking high-pass: removes a constant (DC) offset from the signal
/// while leaving the audio band essentially flat. Implements the standard
/// difference equation `y[n] = x[n] - x[n-1] + R*y[n-1]`, with the pole `R` just
/// below 1 so the corner sits close to DC.
///
/// Same-length and sample-by-sample (one input sample yields one output sample),
/// so it is an [`InPlaceDsp`] driven through [`PerChannel`], one instance per
/// carried channel. The filter memory
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
/// sample), so it is an [`InPlaceDsp`] driven through [`PerChannel`], the same
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
        self.run_normalize_planar(input)?;
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
        self.reinterleave_work(self.output_channels);
        Ok(&self.work)
    }

    /// Run one native block through the `normalize` segment alone, returning
    /// the post-`normalize`, pre-`transform` signal, interleaved.
    ///
    /// Seeds and sanitizes `work` exactly as [`run`](Self::run) does, then
    /// ping-pongs it across the `normalize` stages only. The returned slice is
    /// the signal [`run`](Self::run) snapshots into [`tap`](Self::tap), so a
    /// caller that reads the pre-`transform` signal and never delivers the
    /// conditioned output takes this instead of a full [`run`](Self::run).
    pub(crate) fn run_normalize(&mut self, input: &[f32]) -> Result<&[f32], DecibriError> {
        self.run_normalize_planar(input)?;
        self.reinterleave_work(self.tap_channels);
        Ok(&self.work)
    }

    /// The shared body of [`run`](Self::run) and
    /// [`run_normalize`](Self::run_normalize): seed and sanitize `work`, then
    /// walk the `normalize` stages, leaving `work` in the chain's internal
    /// layout at [`tap_channels`](Self::tap_channels).
    fn run_normalize_planar(&mut self, input: &[f32]) -> Result<(), DecibriError> {
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
        Ok(())
    }

    /// Convert `work` from the chain's internal planar layout back to
    /// interleaved frames at `channels`, via `scratch`. A no-op at one
    /// channel, where the two layouts are the same arrangement. Every signal
    /// that leaves the chain passes through this: the delivered output, the
    /// post-`normalize` return, and (through [`capture_tap`](Self::capture_tap))
    /// the detector tap.
    fn reinterleave_work(&mut self, channels: u16) {
        if channels > 1 {
            self.scratch.clear();
            interleave_into(&self.work, channels, &mut self.scratch);
            std::mem::swap(&mut self.work, &mut self.scratch);
        }
    }

    /// Snapshot `work` (the post-`normalize`, pre-`transform` signal) into `tap`,
    /// but only when `transform` is non-empty. With no transform the delivered
    /// output already is this signal, so nothing is captured and `tap` stays empty
    /// (zero overhead on the no-transform path). The tap crosses the chain
    /// boundary to the detector, so it is interleaved like every delivery;
    /// `MicrophoneStream::detector_feed` collapses it as interleaved frames at
    /// [`tap_channels`](Self::tap_channels).
    fn capture_tap(&mut self) {
        if !self.transform.is_empty() {
            self.tap.clear();
            if self.tap_channels > 1 {
                interleave_into(&self.work, self.tap_channels, &mut self.tap);
            } else {
                self.tap.extend_from_slice(&self.work);
            }
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
        self.flush_normalize_planar()?;
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
        self.reinterleave_work(self.output_channels);
        out.extend_from_slice(&self.work);
        Ok(())
    }

    /// Drain the `normalize` segment's end-of-stream tail alone, returning it
    /// interleaved.
    ///
    /// The counterpart of [`run_normalize`](Self::run_normalize) on the close
    /// path: the resampler's group-delay tail, before it passes through
    /// `transform`.
    pub(crate) fn flush_normalize(&mut self) -> Result<&[f32], DecibriError> {
        self.flush_normalize_planar()?;
        self.reinterleave_work(self.tap_channels);
        Ok(&self.work)
    }

    /// The shared body of [`flush`](Self::flush) and
    /// [`flush_normalize`](Self::flush_normalize): walk the `normalize`
    /// stages' tails, leaving `work` in the chain's internal layout at
    /// [`tap_channels`](Self::tap_channels). `work` starts empty because there
    /// is no new input at end of stream, only each stage's drained tail.
    fn flush_normalize_planar(&mut self) -> Result<(), DecibriError> {
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

    /// The channel count the [`tap`](Self::tap) signal is interleaved at: the
    /// count the `normalize` segment ends at.
    ///
    /// Resolved at construction by walking the `normalize` stages'
    /// [`Stage::output_channels`] from the device count the chain was built
    /// with, so it reports what the stages actually do rather than a count
    /// named alongside them. A consumer reading the tap collapses or
    /// interprets it at this count.
    pub(crate) fn tap_channels(&self) -> u16 {
        self.tap_channels
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

    /// The echo cancellers' metrics, one entry per delivered channel in
    /// delivered order, or `None` when the chain has no canceller. The vector
    /// is never empty: a chain with cancellation holds at least one engine.
    ///
    /// The canceller is a `normalize` stage, so only that segment is walked.
    #[cfg(feature = "aec")]
    pub(crate) fn aec_metrics(&self) -> Option<Vec<AecMetrics>> {
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
    /// is laid out at that stage's OUTPUT count, so the count advances once
    /// per stage, after both the carried block and the tail have been written.
    ///
    /// At one emitted channel a stage's own tail is appended directly after the
    /// processed carry, which continues the one run. Above one, the tail
    /// continues EACH channel's run, so the two planar buffers are spliced run
    /// by run instead of appended whole; an appended whole tail would land
    /// channel 0's tail inside channel 0's neighbour and shift every later run.
    /// The splice allocates, which the close path (this walk's only caller)
    /// pays once per stream.
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
            if emitted > 1 {
                let mut tail = Vec::new();
                stage.flush(&mut tail)?;
                if !tail.is_empty() {
                    splice_planar_tail(scratch, &tail, emitted);
                }
            } else {
                stage.flush(scratch)?;
            }
            *channels = emitted;
            std::mem::swap(work, scratch);
        }
        Ok(())
    }
}

/// The opt-in processing [`build_capture_stage`] applies, bundled into one
/// argument so that adding a capability is a new field rather than another
/// positional parameter. `dc_removal` through `limiter` map one-to-one to the
/// transform stages, listed in chain order; `channel_map` maps to the first
/// stage of the `normalize` segment and `aec` to its last.
///
/// [`Default`] is every field off, the configuration that pushes no optional
/// stage at all, so a caller names only what it enables. A field added here must
/// have an off state that is its type's own default, or the derive stops being
/// correct and the impl has to be written out; `default_is_the_all_off_literal`
/// holds that line.
#[derive(Default)]
pub(crate) struct Transforms<'a> {
    /// Gather the named device channels (the [`Select`] stage) instead of
    /// averaging them; `None` (the default) keeps the documented average of
    /// every opened channel (the [`Downmix`] stage). One 0-based device
    /// channel index per delivered channel, so the length equals
    /// `target_channels`; the caller validated every entry against the
    /// device's own report.
    pub channel_map: Option<&'a [u16]>,
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
    /// Its stage lands last in `normalize` rather than in `transform`, as
    /// `channel_map`'s lands first there: both ride in this bundle because
    /// that keeps a new capability a field on one struct instead of a
    /// parameter on [`build_capture_stage`]'s signature.
    #[cfg(feature = "aec")]
    pub aec: Option<AecSettings>,
}

/// Build the capture stage chain that normalizes a device to the output format
/// and applies any opt-in enhancement.
///
/// Pushes [`Select`] when `channel_map` names a map (gathering the named
/// device channels, in the order given), otherwise [`Downmix`] when the device
/// delivers more channels than the output target (averaging down to the
/// target, which is mono here), then [`Deinterleave`] when more than one
/// channel continues past that point (rearranging the block into the planar
/// layout the rest of the chain runs in), then
/// [`ResampleStage`] when the device's `native_rate` differs from `target_rate`
/// (converting the captured audio to the requested rate, one engine per
/// carried channel), then the
/// [`AecStage`] when `aec` names settings (and the `aec` feature is compiled
/// in), last in the `normalize` segment so the cancellers receive the
/// delivered channels at the target rate (one engine per channel, each
/// reading its own planar run) and the VAD tap carries the echo-removed
/// signal. When `dc_removal` is
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
/// `normalize` on the delivered channels at the target rate: the two filters
/// through [`PerChannel`] with one instance per channel, the denoise stage
/// with one cache set per channel, and the two gain stages through [`Linked`]
/// with one detector across the channels and one gain applied to all.
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
        channel_map,
        dc_removal,
        denoise,
        highpass,
        agc,
        limiter,
        #[cfg(feature = "aec")]
        aec,
    } = transforms;

    let mut normalize: Vec<Box<dyn Stage>> = Vec::new();

    // A named map gathers; without one, more device channels than the target
    // collapse to the documented average. The map carries one entry per
    // delivered channel, so its length is the target count by contract; the
    // capture path validated that (with the entry range) against the resolved
    // device before building the chain.
    match channel_map {
        Some(map) => {
            debug_assert_eq!(
                map.len(),
                target_channels as usize,
                "the channel map carries one entry per delivered channel"
            );
            normalize.push(Box::new(Select::new(device_channels, map)));
        }
        None if device_channels > target_channels => {
            normalize.push(Box::new(Downmix::new(device_channels)));
        }
        None => {}
    }

    // The count carried past the channel stage, resolved by walking what was
    // actually built, exactly as the segment counts below are. When it is
    // above one, the block is rearranged into the planar layout here, once,
    // at the head of the chain; at one channel the two layouts coincide and
    // no stage is built.
    let chain_channels = normalize.iter().fold(device_channels, |channels, stage| {
        stage.output_channels(channels)
    });
    if chain_channels > 1 {
        normalize.push(Box::new(Deinterleave));
    }

    if native_rate != target_rate {
        // The channel stage (Select or Downmix, if any) ran first, so each
        // engine receives one channel's run.
        // Construction validates the rate pair; the `?` bridges a failure to
        // DecibriError::ResampleConfigInvalid via From<ResamplerError>.
        normalize.push(Box::new(ResampleStage::new(
            native_rate,
            target_rate,
            chain_channels,
        )?));
    }

    // Echo cancellation runs LAST in the normalize segment, after the channel
    // stage and the resample, so the stage receives the delivered channels at
    // the target rate, planar above one channel: one engine per channel, each
    // reading its own run, at the rate the engines are constructed at.
    // Nothing runs between it and the VAD tap, so the detector reads the
    // echo-removed signal. The order is pinned by this push position and
    // `build_ends_normalize_at_the_canceller`.
    #[cfg(feature = "aec")]
    if let Some(settings) = aec {
        normalize.push(Box::new(AecStage::new(
            settings,
            target_rate,
            chain_channels,
        )?));
    }

    // Resolve the channel count the `normalize` segment ends at by asking the
    // stages that were actually built, in the order they run. Named nowhere: a
    // count written here as a literal would agree with the stages today and
    // stop agreeing the moment a stage that changes the count is added or
    // removed. The transform stages below are built at this count.
    let tap_channels = normalize.iter().fold(device_channels, |channels, stage| {
        stage.output_channels(channels)
    });

    let mut transform: Vec<Box<dyn Stage>> = Vec::new();

    if dc_removal {
        // Runs after `normalize`, on the delivered channels at the target
        // rate: one filter instance per channel, each over its own run.
        transform.push(Box::new(PerChannel::new(tap_channels, DcBlocker::new)));
    }

    // Denoise runs immediately AFTER DC removal (chain order: DcRemoval ->
    // Denoise), on the DC-blocked signal at the target rate. The order is
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
            tap_channels,
        )?));
    }
    // Without the `denoise` feature the parameter is accepted but unused.
    #[cfg(not(feature = "denoise"))]
    let _ = denoise;

    // High-pass (user rumble cut) runs immediately AFTER denoise (chain order:
    // Denoise -> HighPass), so the denoise model receives near-full-band input.
    // It is a same-length, sample-in-sample-out biquad, so it wraps via
    // `PerChannel` exactly like the DC blocker and adds no latency. The cutoff
    // comes from the named variant, not a magic number here.
    if let Some(filter) = highpass {
        transform.push(Box::new(PerChannel::new(tap_channels, || {
            Biquad::highpass(filter.cutoff_hz(), target_rate as f32)
        })));
    }

    // Level control (AGC) runs after high-pass, reserving the slot that sits
    // before the limiter in the full chain order. It is a same-length engine
    // whose detection runs linked across the channels (one detector, one gain
    // applied to all, so the inter-channel balance is preserved), so it wraps
    // via `Linked` and adds no latency. Gated on the `gain` feature, like
    // denoise is gated on `denoise`; without the feature the target is accepted
    // but unused.
    #[cfg(feature = "gain")]
    if let Some(target_db) = agc {
        transform.push(Box::new(Linked(crate::gain::LevelControl::agc(
            target_db,
            target_rate,
        ))));
    }
    #[cfg(not(feature = "gain"))]
    let _ = agc;

    // The limiter runs LAST in the transform tier, immediately after the level
    // control, so it catches any peak the upstream gain would let exceed the
    // ceiling. It is a same-length stage detecting linked across the channels
    // like the level control, so it wraps via `Linked` and adds no latency.
    // Gated on the same `gain` feature as the level-control engine (the pair);
    // without the feature the ceiling is accepted but unused. Nothing runs
    // after it.
    #[cfg(feature = "gain")]
    if let Some(ceiling_db) = limiter {
        transform.push(Box::new(Linked(crate::gain::Limiter::new(
            ceiling_db,
            target_rate,
        ))));
    }
    #[cfg(not(feature = "gain"))]
    let _ = limiter;

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

    // ── Channel selection (the Select gather stage) ────────────────────

    /// A map of `[0]` and a map of `[1]` over a stereo source with different
    /// content per channel return exactly those channels, sample for sample.
    ///
    /// Regression: a gather taking the wrong index, or an off-by-one in the
    /// interleaved stride, compiles and produces plausible audio; this pins
    /// the delivered samples to the named device channel exactly.
    #[test]
    fn select_gathers_exactly_the_named_device_channel() {
        // Two channels with unmistakably different content: channel 0 counts
        // up from 1.0, channel 1 counts down from -1.0.
        let interleaved = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0];
        for (map, expected) in [
            (vec![0_u16], vec![1.0_f32, 2.0, 3.0, 4.0]),
            (vec![1_u16], vec![-1.0_f32, -2.0, -3.0, -4.0]),
        ] {
            let mut chain = build_capture_stage(
                2,
                1,
                16_000,
                16_000,
                Transforms {
                    channel_map: Some(&map),
                    ..Default::default()
                },
            )
            .unwrap()
            .expect("a mapped stereo source builds a select chain");
            assert_eq!(
                chain.output_channels(),
                1,
                "the select chain delivers the map's one channel"
            );
            let out = chain.run(&interleaved).expect("select runs");
            assert_eq!(out, expected, "map {map:?} returns exactly that channel");
        }
    }

    /// With no map, a multichannel source still produces the documented
    /// average, byte-identical to `sample::downmix_to_mono`: the map's absence
    /// leaves the collapse exactly as it was.
    #[test]
    fn no_map_keeps_the_average_byte_identical() {
        let interleaved = [0.9, -0.3, 0.0, 0.6, 0.0, -0.6, -0.5, 0.5, 0.5];
        let mut chain = build_capture_stage(3, 1, 16_000, 16_000, Transforms::default())
            .unwrap()
            .expect("a 3-channel source builds a downmix chain");
        let out = chain.run(&interleaved).expect("downmix runs").to_vec();
        assert_eq!(
            out,
            sample::downmix_to_mono(&interleaved, 3),
            "the no-map collapse is the documented average, byte for byte"
        );
    }

    /// The gather takes entries in map order at its general width, and a
    /// duplicated entry is an independent copy of its source channel.
    #[test]
    fn select_gathers_in_map_order_at_general_width() {
        // One 3-channel frame (7.0, 8.0, 9.0), gathered as (channel 2,
        // channel 0), then as (channel 1, channel 1).
        let interleaved = [7.0, 8.0, 9.0, 70.0, 80.0, 90.0];
        let mut reorder = build_capture_stage(
            3,
            2,
            16_000,
            16_000,
            Transforms {
                channel_map: Some(&[2, 0]),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a reordering gather builds a chain");
        assert_eq!(reorder.output_channels(), 2, "two delivered channels");
        let out = reorder.run(&interleaved).expect("gather runs");
        assert_eq!(out, &[9.0, 7.0, 90.0, 70.0], "entries land in map order");

        let mut duplicate = build_capture_stage(
            3,
            2,
            16_000,
            16_000,
            Transforms {
                channel_map: Some(&[1, 1]),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a duplicating gather builds a chain");
        let out = duplicate.run(&interleaved).expect("gather runs");
        assert_eq!(
            out,
            &[8.0, 8.0, 80.0, 80.0],
            "a duplicated entry copies its source channel independently"
        );
    }

    /// The gather accepts an index bounded only by the device's own report:
    /// a device reporting 60000 channels accepts index 59999.
    ///
    /// The negative control for the no-fixed-maximum rule: a constant, an
    /// array-sized state, or a `1..=N` bound added anywhere on this path
    /// later fails loudly here.
    #[test]
    fn select_accepts_an_index_bounded_only_by_the_device_report() {
        const DEVICE_CHANNELS: u16 = 60_000;
        let map = [DEVICE_CHANNELS - 1];
        let mut chain = build_capture_stage(
            DEVICE_CHANNELS,
            1,
            16_000,
            16_000,
            Transforms {
                channel_map: Some(&map),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a top-index gather builds a chain");
        // Two frames whose last channel carries a marker value.
        let mut interleaved = vec![0.0_f32; DEVICE_CHANNELS as usize * 2];
        interleaved[DEVICE_CHANNELS as usize - 1] = 0.25;
        interleaved[DEVICE_CHANNELS as usize * 2 - 1] = 0.75;
        let out = chain.run(&interleaved).expect("gather runs");
        assert_eq!(
            out,
            &[0.25, 0.75],
            "the top index of a 60000-channel report is served"
        );
    }

    /// A map composes with the resampler: the gather runs first in
    /// `normalize`, so the resampler receives the selected mono channel, and
    /// the chain's output equals resampling that channel alone.
    #[test]
    fn select_then_resample_matches_resampling_the_selected_channel() {
        let frames = 24_000;
        let left: Vec<f32> = (0..frames).map(|n| (n as f32 * 0.01).sin()).collect();
        let right: Vec<f32> = (0..frames).map(|n| (n as f32 * 0.02).cos()).collect();
        let interleaved: Vec<f32> = left
            .iter()
            .zip(&right)
            .flat_map(|(&l, &r)| [l, r])
            .collect();

        let mut chain = build_capture_stage(
            2,
            1,
            48_000,
            16_000,
            Transforms {
                channel_map: Some(&[1]),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("select + resample chain");
        let mut out = chain.run(&interleaved).expect("chain runs").to_vec();
        chain.flush(&mut out).expect("chain flushes");

        let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut expected = Vec::new();
        resampler.process(&right, &mut expected).unwrap();
        resampler.flush(&mut expected);
        assert_eq!(
            out, expected,
            "the chain equals resampling the selected channel alone"
        );
    }

    // ── The planar carriage (multichannel through the chain) ────────────
    //
    // No production configuration reaches these paths while the accepted
    // channel count is 1, so these tests are the only thing exercising them:
    // the planar rearrangement pair, the per-channel and linked wrappers, the
    // per-channel resampler engines, and the flush splice. Each drives the
    // machinery directly or through `build_capture_stage` with a target above
    // one.

    /// The rearrangement pair is exact: deinterleave then interleave returns
    /// the original buffer bit for bit, at counts that do not divide common
    /// block sizes and at `u16::MAX`. A trailing partial frame is truncated by
    /// the deinterleave, matching `Block::frames` and the downmix.
    #[test]
    fn deinterleave_and_interleave_round_trip_exactly() {
        for (channels, frames) in [(2u16, 53usize), (3, 53), (5, 7), (7, 160), (48, 3)] {
            let interleaved: Vec<f32> = (0..frames)
                .flat_map(|frame| {
                    (0..channels).map(move |channel| channel as f32 + frame as f32 * 0.001)
                })
                .collect();
            let mut planar = Vec::new();
            deinterleave_into(&interleaved, channels, &mut planar);
            assert_eq!(planar.len(), interleaved.len());
            // Run i holds channel i's samples in frame order.
            for channel in 0..channels as usize {
                for frame in 0..frames {
                    assert_eq!(
                        planar[channel * frames + frame],
                        interleaved[frame * channels as usize + channel],
                        "channel {channel} frame {frame} lands in its run"
                    );
                }
            }
            let mut back = Vec::new();
            interleave_into(&planar, channels, &mut back);
            assert_eq!(
                back, interleaved,
                "the round trip is exact at {channels} channels"
            );
        }

        // The count is bounded only by its type: the pair round-trips at
        // `u16::MAX`. A fixed maximum added to either function fails here.
        let channels = u16::MAX;
        let interleaved: Vec<f32> = (0..2usize)
            .flat_map(|frame| (0..channels).map(move |channel| channel as f32 + frame as f32 * 0.5))
            .collect();
        let mut planar = Vec::new();
        deinterleave_into(&interleaved, channels, &mut planar);
        let mut back = Vec::new();
        interleave_into(&planar, channels, &mut back);
        assert_eq!(
            back, interleaved,
            "the round trip is exact at u16::MAX channels"
        );

        // A trailing partial frame is dropped, not misread as a whole one.
        let mut truncated = Vec::new();
        deinterleave_into(&[1.0, 2.0, 3.0, 4.0, 5.0], 2, &mut truncated);
        assert_eq!(
            truncated,
            &[1.0, 3.0, 2.0, 4.0],
            "the partial frame is truncated, matching Block::frames"
        );
    }

    /// `PerChannel` applies the wrapped stage to every channel independently:
    /// each channel's output equals a fresh single-channel engine run over
    /// that channel alone, bit for bit, across block boundaries. The channels
    /// carry unmistakably different content (a sine, exact silence, a
    /// constant offset), so a stride or offset error that mixes recursive
    /// state across channels breaks the equality rather than passing
    /// plausibly.
    #[test]
    fn per_channel_runs_the_wrapped_stage_independently() {
        let frames = 480;
        let sine: Vec<f32> = (0..2 * frames).map(|n| (n as f32 * 0.05).sin()).collect();
        let silence = vec![0.0f32; 2 * frames];
        let offset = vec![0.5f32; 2 * frames];

        // Two consecutive planar blocks, so the per-channel state must carry
        // across the boundary along each channel.
        let mut stage = PerChannel::new(3, DcBlocker::new);
        let mut delivered = Vec::new();
        for range in [0..frames, frames..2 * frames] {
            let planar: Vec<f32> = sine[range.clone()]
                .iter()
                .chain(&silence[range.clone()])
                .chain(&offset[range.clone()])
                .copied()
                .collect();
            let mut out = Vec::new();
            stage
                .process(Block::new(&planar, 3), &mut out)
                .expect("the wrapper processes a planar block");
            assert_eq!(out.len(), planar.len(), "same-length per block");
            delivered.push(out);
        }

        // Reference: one fresh engine per channel over that channel alone.
        for (channel, input) in [(0usize, &sine), (1, &silence), (2, &offset)] {
            let mut engine = DcBlocker::new();
            let mut expected = input.to_vec();
            engine.process_in_place(&mut expected);
            for (block_index, range) in [0..frames, frames..2 * frames].into_iter().enumerate() {
                let run = &delivered[block_index][channel * frames..(channel + 1) * frames];
                assert_eq!(
                    run, &expected[range],
                    "channel {channel} equals its own engine, bit for bit (block {block_index})"
                );
            }
        }
    }

    /// A silent channel stays exactly silent through a full multichannel
    /// chain (rearrangement, per-channel resampling, per-channel filters,
    /// flush splice), and the sounding channel is bit-identical to the mono
    /// chain over its signal alone. This is the per-channel independence
    /// property at chain scope: a stride bug leaks the sine into the silence
    /// or the recursive state across the boundary, and either breaks an exact
    /// assertion here.
    #[test]
    fn a_silent_channel_stays_silent_through_the_chain() {
        let frames = 24_000;
        let sine: Vec<f32> = (0..frames).map(|n| (n as f32 * 0.01).sin()).collect();
        let transforms = || Transforms {
            dc_removal: true,
            highpass: Some(HighpassFilter::Hz80),
            ..Default::default()
        };

        // Stereo device, stereo target: no channel stage, so the chain is
        // Deinterleave, the two resampler engines, then the per-channel
        // filters, with the flush splicing each engine's tail onto its own
        // run.
        let interleaved: Vec<f32> = sine.iter().flat_map(|&s| [s, 0.0]).collect();
        let mut chain = build_capture_stage(2, 2, 48_000, 16_000, transforms())
            .unwrap()
            .expect("a rate change builds a chain");
        assert_eq!(chain.output_channels(), 2);
        let mut delivered = chain.run(&interleaved).expect("the chain runs").to_vec();
        chain.flush(&mut delivered).expect("the chain flushes");

        // The mono reference: the identical configuration over the sine alone.
        let mut mono = build_capture_stage(1, 1, 48_000, 16_000, transforms())
            .unwrap()
            .expect("the mono chain builds");
        let mut expected = mono.run(&sine).expect("the mono chain runs").to_vec();
        mono.flush(&mut expected).expect("the mono chain flushes");

        assert_eq!(
            delivered.len(),
            2 * expected.len(),
            "two delivered channels, one frame per mono sample"
        );
        for (frame, &value) in expected.iter().enumerate() {
            assert_eq!(
                delivered[2 * frame],
                value,
                "frame {frame}: the sounding channel equals the mono chain bit for bit"
            );
            assert_eq!(
                delivered[2 * frame + 1],
                0.0,
                "frame {frame}: the silent channel stays exactly silent"
            );
        }
    }

    /// The chain resolves K device channels to a smaller delivered count
    /// through the gather: each delivered channel equals resampling the
    /// mapped device channel alone, bit for bit, through run and flush.
    #[test]
    fn the_chain_resolves_k_in_to_target_out_through_the_walk() {
        let frames = 24_000;
        let a: Vec<f32> = (0..frames).map(|n| (n as f32 * 0.01).sin()).collect();
        let b: Vec<f32> = (0..frames).map(|n| (n as f32 * 0.02).cos()).collect();
        // Four device channels; channel 3 carries `a`, channel 0 carries `b`,
        // the middle two carry distinct decoys.
        let interleaved: Vec<f32> = (0..frames)
            .flat_map(|n| [b[n], 0.25, -0.75, a[n]])
            .collect();

        let map = [3u16, 0];
        let mut chain = build_capture_stage(
            4,
            2,
            48_000,
            16_000,
            Transforms {
                channel_map: Some(&map),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a mapped multichannel gather builds a chain");
        assert_eq!(
            chain.output_channels(),
            2,
            "the map's two delivered channels"
        );
        let mut delivered = chain.run(&interleaved).expect("the chain runs").to_vec();
        chain.flush(&mut delivered).expect("the chain flushes");

        for (channel, source) in [(0usize, &a), (1, &b)] {
            let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();
            let mut expected = Vec::new();
            resampler.process(source, &mut expected).unwrap();
            resampler.flush(&mut expected);
            assert_eq!(delivered.len(), 2 * expected.len());
            for (frame, &value) in expected.iter().enumerate() {
                assert_eq!(
                    delivered[2 * frame + channel],
                    value,
                    "delivered channel {channel} equals resampling its mapped device channel"
                );
            }
        }
    }

    /// The tap is interleaved at the post-`normalize` count: with a transform
    /// present and nothing in `normalize` but the rearrangement, the tap
    /// equals the original interleaved input bit for bit, which is the layout
    /// `detector_feed` collapses at `tap_channels`.
    #[test]
    fn the_tap_is_interleaved_at_the_post_normalize_count() {
        let interleaved = [0.1f32, -0.4, 0.2, -0.3, 0.3, -0.2, 0.4, -0.1];
        let mut chain = build_capture_stage(
            2,
            2,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a transform builds a chain");
        assert_eq!(chain.tap_channels(), 2);
        chain.run(&interleaved).expect("the chain runs");
        assert_eq!(
            chain.tap(),
            &interleaved,
            "the tap crosses the boundary interleaved, so the pre-transform \
             snapshot equals the input here"
        );
        assert_eq!(
            sample::downmix_to_mono(chain.tap(), 2),
            &[
                (0.1 + -0.4) / 2.0,
                (0.2 + -0.3) / 2.0,
                (0.3 + -0.2) / 2.0,
                (0.4 + -0.1) / 2.0
            ],
            "the tap collapses as interleaved frames"
        );
    }

    /// The gain tier the builder wires detects across channels: driving a
    /// stereo chain with one channel at exactly half the other's amplitude
    /// delivers outputs whose ratio is still exactly one half, sample for
    /// sample, while the level control materially raises the quiet signal.
    /// Detection per channel would drive both channels toward the same
    /// target, erasing the ratio, so this pins the linked wiring.
    #[cfg(feature = "gain")]
    #[test]
    fn the_chain_gain_tier_preserves_inter_channel_balance() {
        let frames = 16_000;
        // A quiet but present tone (above the noise floor, below the target).
        let loud: Vec<f32> = (0..frames)
            .map(|n| 0.02 * (n as f32 * 0.09).sin())
            .collect();
        let interleaved: Vec<f32> = loud.iter().flat_map(|&s| [s, 0.5 * s]).collect();

        let mut chain = build_capture_stage(
            2,
            2,
            16_000,
            16_000,
            Transforms {
                agc: Some(-18),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a gain chain builds");
        let delivered = chain.run(&interleaved).expect("the chain runs").to_vec();

        for frame in 0..frames {
            assert_eq!(
                delivered[2 * frame + 1],
                0.5 * delivered[2 * frame],
                "frame {frame}: one shared gain keeps the exact inter-channel ratio"
            );
        }
        // Not vacuous: the level control moved the quiet signal toward the
        // target rather than leaving the gain at unity.
        let tail = &delivered[delivered.len() - 4000..];
        let input_tail = &interleaved[interleaved.len() - 4000..];
        let rms = |s: &[f32]| (s.iter().map(|x| x * x).sum::<f32>() / s.len() as f32).sqrt();
        assert!(
            rms(tail) > 2.0 * rms(input_tail),
            "the level control raised the quiet signal materially"
        );
    }

    /// The mono chain is byte-identical to composing the engines directly:
    /// the delivered output of a DC + high-pass + AGC + limiter chain equals
    /// running the four unchanged engines in sequence over the same blocks.
    /// The wrappers the builder installs are exactly transparent at one
    /// channel, which is this change set's safety property on the reachable
    /// path.
    #[cfg(feature = "gain")]
    #[test]
    fn the_mono_chain_is_byte_identical_to_the_primitive_composition() {
        let input: Vec<f32> = (0..12_000)
            .map(|n| 0.3 * (n as f32 * 0.03).sin() + 0.05)
            .collect();
        let blocks = [
            &input[..1000],
            &input[1000..1001],
            &input[1001..7321],
            &input[7321..],
        ];

        let mut chain = build_capture_stage(
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
        .expect("a transform chain builds");
        let mut delivered = Vec::new();
        for block in blocks {
            delivered.extend_from_slice(chain.run(block).expect("the chain runs"));
        }
        chain.flush(&mut delivered).expect("the chain flushes");

        // The reference composition: the same four engines, driven directly.
        let mut dc = DcBlocker::new();
        let mut highpass = Biquad::highpass(HighpassFilter::Hz80.cutoff_hz(), 16_000.0);
        let mut agc = crate::gain::LevelControl::agc(-18, 16_000);
        let mut limiter = crate::gain::Limiter::new(-1.0, 16_000);
        let mut expected = Vec::new();
        for block in blocks {
            let mut buf = block.to_vec();
            dc.process_in_place(&mut buf);
            highpass.process_in_place(&mut buf);
            agc.process_in_place(&mut buf);
            limiter.process_in_place(&mut buf);
            expected.extend_from_slice(&buf);
        }

        assert_eq!(
            delivered, expected,
            "the chain equals the engine composition, bit for bit"
        );
    }

    /// Every piece of the planar carriage serves `u16::MAX` channels: the
    /// negative control for the no-fixed-maximum rule on the paths this
    /// change adds. A constant, an array-sized state, or a `1..=N` bound
    /// added to the rearrangement, the per-channel wrapper, or the builder
    /// fails loudly here.
    #[test]
    fn the_planar_carriage_serves_u16_max_channels() {
        const CHANNELS: u16 = u16::MAX;
        let frames = 2usize;
        // Channel c carries [c, c + 0.25]: distinct per channel and per frame.
        let interleaved: Vec<f32> = (0..frames)
            .flat_map(|frame| (0..CHANNELS).map(move |c| c as f32 + frame as f32 * 0.25))
            .collect();

        let mut chain = build_capture_stage(
            CHANNELS,
            CHANNELS,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a transform chain builds at the type's full width");
        assert_eq!(chain.output_channels(), CHANNELS);
        let delivered = chain.run(&interleaved).expect("the chain runs").to_vec();

        // Each channel equals its own fresh engine, at the top of the range
        // as at the bottom.
        for channel in [0u16, 1, 32_767, CHANNELS - 2, CHANNELS - 1] {
            let mut engine = DcBlocker::new();
            let mut expected = vec![channel as f32, channel as f32 + 0.25];
            engine.process_in_place(&mut expected);
            for frame in 0..frames {
                assert_eq!(
                    delivered[frame * CHANNELS as usize + channel as usize],
                    expected[frame],
                    "channel {channel} frame {frame} is served at full width"
                );
            }
        }
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
        assert_send::<PerChannel<DcBlocker>>();
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
        let stage = ResampleStage::new(48_000, 16_000, 1).unwrap();
        assert_eq!(
            stage.latency_samples(),
            expected,
            "the stage forwards the resampler's group delay unchanged"
        );

        // An identity resampler forwards zero.
        let identity = ResampleStage::new(16_000, 16_000, 1).unwrap();
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
    // sample-in-sample-out `PerChannel` stage (like the DC blocker) that runs after
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

    /// `PerChannel<Biquad>` is `Send`, so a chain carrying the high-pass in its
    /// `transform` segment stays `Send` and can live behind the stream's `Mutex`,
    /// matching the DC blocker.
    #[test]
    fn highpass_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<PerChannel<Biquad>>();
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
    // AGC is the LevelControl engine driven through `Linked`, a same-length
    // stage whose detection runs across the delivered channels. These cover
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

    /// `Linked<LevelControl>` is `Send`, so a chain carrying AGC in its
    /// `transform` segment stays `Send` and can live behind the stream's `Mutex`,
    /// matching the DC blocker and high-pass.
    #[cfg(feature = "gain")]
    #[test]
    fn agc_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Linked<crate::gain::LevelControl>>();
    }

    // ── Limiter (same-length transform, last in the chain) ──────────────
    //
    // The limiter is the Limiter stage driven through `Linked`, a same-length
    // stage detecting across the delivered channels like AGC. These cover its chain placement (last,
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

    /// `Linked<Limiter>` is `Send`, so a chain carrying the limiter in its
    /// `transform` segment stays `Send` and can live behind the stream's `Mutex`,
    /// matching the DC blocker, high-pass, and AGC.
    #[cfg(feature = "gain")]
    #[test]
    fn limiter_stage_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<Linked<crate::gain::Limiter>>();
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
            channel_map: None,
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
            default.channel_map, all_off.channel_map,
            "the default must not gather channels"
        );
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
            (out, chain.aec_metrics().expect("metrics").remove(0), queue)
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
        let metrics = chain
            .aec_metrics()
            .expect("a canceller reports metrics")
            .remove(0);
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

    /// The canceller holds the LAST slot in the normalize segment, after the
    /// downmix and the resample, so nothing runs between it and the tap and
    /// the tap carries the echo-removed signal. A full normalize segment pins
    /// the position through the stage that answers metrics: only the last
    /// slot's stage is the canceller. Regression: a stage added to the
    /// segment's tail, which would push the canceller off the end and hand the
    /// tap a signal the canceller has not seen.
    #[cfg(feature = "aec")]
    #[test]
    fn build_ends_normalize_at_the_canceller() {
        let queue = Arc::new(AecReferenceRing::new(16_000));
        let chain = aec_chain(2, 48_000, 16_000, false, &queue, 16_000);
        assert_eq!(
            chain.normalize.len(),
            3,
            "downmix, resample, canceller: three normalize stages"
        );
        assert!(
            chain
                .normalize
                .last()
                .expect("a non-empty normalize segment")
                .aec_metrics()
                .is_some(),
            "the canceller holds the last normalize slot"
        );
        assert!(
            chain.normalize[..chain.normalize.len() - 1]
                .iter()
                .all(|stage| stage.aec_metrics().is_none()),
            "no stage before the last is the canceller"
        );
        assert!(
            chain.transform.is_empty(),
            "with conditioning off the chain ends at the normalize segment"
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
            (
                out,
                chain.aec_metrics().expect("metrics").remove(0),
                queue.silence(),
            )
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
        let metrics = chain
            .aec_metrics()
            .expect("a canceller reports metrics")
            .remove(0);
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
        let metrics = chain
            .aec_metrics()
            .expect("a canceller reports metrics")
            .remove(0);
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
        let metrics = chain
            .aec_metrics()
            .expect("a canceller reports metrics")
            .remove(0);
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

    /// A chain delivering `channels` channels with echo cancellation on and no
    /// other stage: the device count equals the delivered count at one rate,
    /// so the normalize segment is the rearrangement and the cancellers alone.
    #[cfg(feature = "aec")]
    fn aec_multichannel_chain(
        channels: u16,
        rate: u32,
        queue: &Arc<AecReferenceRing>,
        reference_rate: u32,
    ) -> CaptureStage {
        build_capture_stage(
            channels,
            channels,
            rate,
            rate,
            Transforms {
                aec: Some(aec_settings(queue, reference_rate)),
                ..Default::default()
            },
        )
        .expect("the chain builds")
        .expect("echo cancellation builds a chain")
    }

    /// Interleave equal-length per-channel signals into frames.
    #[cfg(feature = "aec")]
    fn interleave_frames(channels: &[Vec<f32>]) -> Vec<f32> {
        let frames = channels[0].len();
        let mut out = Vec::with_capacity(frames * channels.len());
        for frame in 0..frames {
            for channel in channels {
                out.push(channel[frame]);
            }
        }
        out
    }

    /// Split interleaved frames back into per-channel runs.
    #[cfg(feature = "aec")]
    fn split_frames(interleaved: &[f32], channels: usize) -> Vec<Vec<f32>> {
        let mut out = vec![Vec::with_capacity(interleaved.len() / channels); channels];
        for frame in interleaved.chunks_exact(channels) {
            for (channel, &sample) in frame.iter().enumerate() {
                out[channel].push(sample);
            }
        }
        out
    }

    /// Above one channel the near-end feed is re-blocked to whole multiples of
    /// the engines' 256-sample framing; at one channel it is not. Pinned on
    /// the far-end frontier: the silence top-up covers exactly the frames
    /// handed to the engines, so the counter reads back what each call fed.
    ///
    /// Regression, above one channel: a feed that hands the engines a partial
    /// block, which a later delay promotion would truncate differently per
    /// channel and shear the channels apart silently. Regression, at one
    /// channel: the re-blocker engaging there, which would change the call
    /// cadence the single-channel engine has always seen and move its
    /// reference feeding and re-anchor evaluation without changing its
    /// latency.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_re_blocks_above_one_channel_and_only_there() {
        // Two channels, 320 frames per call: not a block multiple, so the
        // re-blocker holds a growing remainder and feeds 256, 256, 256, 512.
        let queue = Arc::new(AecReferenceRing::new(16_000));
        let mut chain = aec_multichannel_chain(2, 16_000, &queue, 16_000);
        let signal = mono_signal(4 * 320);
        let stereo = interleave_frames(&[signal.clone(), signal.clone()]);
        let mut delivered = Vec::new();
        for (call, piece) in stereo.chunks(2 * 320).enumerate() {
            let out_len = chain.run(piece).expect("run").len();
            delivered.push(out_len);
            let expected_fed: u64 = [256u64, 512, 768, 1280][call];
            assert_eq!(
                queue.silence(),
                expected_fed,
                "call {call} advances the far-end frontier by whole blocks"
            );
        }
        assert_eq!(
            delivered,
            vec![2 * 256, 2 * 256, 2 * 256, 2 * 512],
            "each call delivers both channels at the block-multiple count it fed"
        );
        let mut tail = Vec::new();
        chain.flush(&mut tail).expect("flush");
        assert_eq!(
            delivered.iter().sum::<usize>() + tail.len(),
            stereo.len(),
            "the totals balance after the flush"
        );

        // The same signal at block-multiple calls passes straight through,
        // nothing held: 512, 512 and the trailing 256 frames each deliver in
        // full on the call that fed them.
        let queue = Arc::new(AecReferenceRing::new(16_000));
        let mut chain = aec_multichannel_chain(2, 16_000, &queue, 16_000);
        for (call, piece) in stereo.chunks(2 * 512).enumerate() {
            assert_eq!(
                chain.run(piece).expect("run").len(),
                piece.len(),
                "a block-multiple call passes straight through (call {call})"
            );
        }
        assert_eq!(queue.silence(), 1280, "nothing is held back at a multiple");

        // One channel, the same 320-sample calls: the frontier advances by the
        // full input length every call, not by a block multiple, because the
        // input goes to the engine directly and the engine holds its own
        // carry. The emitted counts still land on block boundaries, which is
        // the engine's internal framing at work, not the re-blocker's.
        let queue = Arc::new(AecReferenceRing::new(16_000));
        let mut chain = aec_chain(1, 16_000, 16_000, false, &queue, 16_000);
        let mut delivered = Vec::new();
        for (call, piece) in signal.chunks(320).enumerate() {
            delivered.push(chain.run(piece).expect("run").len());
            let expected_fed = 320 * (call as u64 + 1);
            assert_eq!(
                queue.silence(),
                expected_fed,
                "call {call} advances the mono frontier by the input length"
            );
        }
        assert_eq!(
            delivered,
            vec![256, 256, 256, 512],
            "the mono emission cadence is the engine's own framing"
        );
    }

    /// Two delivered channels carrying the same signal against the same
    /// reference deliver the same samples, bit for bit, across lock and
    /// promotion. The engines are identical, are fed identical runs in
    /// identical calls, and share the drained reference, so the first
    /// diverging sample would mark the exact call where the per-channel state
    /// separated. This is the sharpest observable form of the lockstep
    /// property: the per-call equal-count assertions run inside the stage in
    /// debug builds, and this pins the delivered bytes on top of them.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_identical_channels_deliver_identical_samples() {
        const RATE: usize = 16_000;
        const SECONDS: usize = 8;
        const DELAY: usize = 400;
        const ECHO_GAIN: f32 = 0.25;

        let mut far = far_noise(SECONDS * RATE);
        for sample in far[..RATE / 2].iter_mut() {
            *sample = 0.0;
        }
        let mut near = vec![0.0f32; far.len()];
        for i in DELAY..near.len() {
            near[i] = ECHO_GAIN * far[i - DELAY];
        }

        let queue = Arc::new(AecReferenceRing::new(2 * RATE));
        let mut chain = aec_multichannel_chain(2, RATE as u32, &queue, RATE as u32);
        let stereo = interleave_frames(&[near.clone(), near.clone()]);
        let mut out = Vec::new();
        for (call, piece) in stereo.chunks(2 * 320).enumerate() {
            let start = call * 320;
            queue.push(&far[start..(start + 320).min(far.len())]);
            out.extend_from_slice(chain.run(piece).expect("run"));
        }
        chain.flush(&mut out).expect("flush");

        let channels = split_frames(&out, 2);
        assert_eq!(
            channels[0], channels[1],
            "identical inputs through identical engines deliver identical bytes"
        );

        // Guard against the pair passing because nothing was cancelled: the
        // echo is removed from both channels, so the equality above compared
        // working cancellers rather than two passthroughs.
        let window = (SECONDS - 1) * RATE..(SECONDS - 1) * RATE + RATE / 2;
        let removed = rms_db(&near[window.clone()]) - rms_db(&channels[0][window]);
        assert!(
            removed >= 30.0,
            "both channels must cancel (removed {removed:.1} dB)"
        );
    }

    /// Channels with different echo delays each lock their own alignment, stay
    /// in lockstep through a mid-stream change of both echo paths, and are
    /// both cancelled. The delivered counts are asserted per call (both
    /// channels, whole blocks) and in total, and the per-channel metrics
    /// report two distinct locked alignments about the geometry's own spacing
    /// apart.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_channels_with_different_delays_lock_apart_and_stay_in_lockstep() {
        const RATE: usize = 16_000;
        const SECONDS: usize = 10;
        const SWITCH: usize = 5 * RATE;
        const DELAY_A: usize = 400;
        const DELAY_B: usize = 900;
        const SHIFT: usize = 800;
        const ECHO_GAIN: f32 = 0.25;

        let mut far = far_noise(SECONDS * RATE);
        for sample in far[..RATE / 2].iter_mut() {
            *sample = 0.0;
        }
        let echo = |delay_before: usize, delay_after: usize| -> Vec<f32> {
            let mut near = vec![0.0f32; far.len()];
            for i in 0..near.len() {
                let delay = if i < SWITCH {
                    delay_before
                } else {
                    delay_after
                };
                if i >= delay {
                    near[i] = ECHO_GAIN * far[i - delay];
                }
            }
            near
        };
        let near_a = echo(DELAY_A, DELAY_A + SHIFT);
        let near_b = echo(DELAY_B, DELAY_B + SHIFT);

        let queue = Arc::new(AecReferenceRing::new(2 * RATE));
        let mut chain = aec_multichannel_chain(2, RATE as u32, &queue, RATE as u32);
        let stereo = interleave_frames(&[near_a.clone(), near_b.clone()]);
        let mut out = Vec::new();
        for (call, piece) in stereo.chunks(2 * 320).enumerate() {
            let start = call * 320;
            queue.push(&far[start..(start + 320).min(far.len())]);
            let before = out.len();
            out.extend_from_slice(chain.run(piece).expect("run"));
            let emitted = out.len() - before;
            assert!(
                emitted.is_multiple_of(2) && (emitted / 2).is_multiple_of(256),
                "call {call} delivers both channels in whole blocks, got {emitted}"
            );
        }
        chain.flush(&mut out).expect("flush");
        assert_eq!(
            out.len(),
            stereo.len(),
            "every input frame is delivered on both channels after the flush"
        );

        // Both channels are cancelled, before the echo paths move and after
        // both cancellers have re-converged on the moved paths.
        let channels = split_frames(&out, 2);
        for window in [4 * RATE..4 * RATE + RATE / 2, 9 * RATE..9 * RATE + RATE / 2] {
            for (index, (channel, near)) in channels.iter().zip([&near_a, &near_b]).enumerate() {
                let removed = rms_db(&near[window.clone()]) - rms_db(&channel[window.clone()]);
                assert!(
                    removed >= 20.0,
                    "channel {index} must cancel in {window:?} (removed {removed:.1} dB)"
                );
            }
        }

        // One report per delivered channel, each holding its own alignment:
        // the two offsets sit about the delay spacing apart, so the engines
        // demonstrably searched their own channels rather than sharing one
        // decision.
        let metrics = chain.aec_metrics().expect("a canceller reports metrics");
        assert_eq!(metrics.len(), 2, "one report per delivered channel");
        let offset_a = metrics[0].delay_samples.expect("channel 0 locks");
        let offset_b = metrics[1].delay_samples.expect("channel 1 locks");
        let spacing = offset_b as i64 - offset_a as i64;
        assert!(
            (400..=600).contains(&spacing),
            "the alignments sit the echo spacing apart, got {offset_a} and {offset_b}"
        );
    }

    /// The metrics vector carries exactly one entry per delivered channel, at
    /// one channel and well past any plausible machine budget: the count is a
    /// property of the chain that was built, and no capacity ceiling exists
    /// anywhere between the configuration and the engines. A maximum added
    /// later fails here loudly.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_builds_and_reports_one_engine_per_delivered_channel() {
        let queue = Arc::new(AecReferenceRing::new(16_000));
        let chain = aec_chain(1, 16_000, 16_000, false, &queue, 16_000);
        assert_eq!(
            chain.aec_metrics().expect("metrics").len(),
            1,
            "one channel, one engine"
        );

        for channels in [2u16, 8] {
            let queue = Arc::new(AecReferenceRing::new(16_000));
            let mut chain = aec_multichannel_chain(channels, 16_000, &queue, 16_000);
            assert_eq!(
                chain.aec_metrics().expect("metrics").len(),
                channels as usize,
                "{channels} channels, {channels} engines"
            );
            // One re-blocked call runs every engine, so the count above is
            // backed by engines that process, not placeholders.
            let frames = 320usize;
            let block: Vec<f32> = mono_signal(frames * channels as usize);
            assert_eq!(
                chain.run(&block).expect("run").len(),
                256 * channels as usize,
                "every engine emits its channel's whole blocks"
            );
        }
    }
}
