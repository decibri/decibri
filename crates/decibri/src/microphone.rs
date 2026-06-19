//! Microphone capture: open an input device and pull [`AudioChunk`]s from it.
//!
//! Build a [`MicrophoneConfig`], construct a [`Microphone`], then
//! [`start`](Microphone::start) it to obtain a [`MicrophoneStream`]. Read audio
//! with [`next_chunk`](MicrophoneStream::next_chunk) (blocking, with a timeout)
//! or [`try_next_chunk`](MicrophoneStream::try_next_chunk) (non-blocking). Both
//! take a requested sample count and deliver exactly that many interleaved
//! samples per chunk, re-blocking the device's native capture buffers on the
//! consumer side. The final chunk at stream close may be shorter, carrying the
//! remaining tail (no captured sample is dropped). The playback counterpart is
//! [`crate::speaker`].

#[cfg(feature = "capture")]
use std::collections::VecDeque;
#[cfg(feature = "capture")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "capture")]
use std::sync::{Arc, Mutex, PoisonError};
#[cfg(feature = "capture")]
use std::time::{Duration, Instant};

#[cfg(feature = "capture")]
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError};

#[cfg(feature = "capture")]
use crate::backend::{
    AudioBackend, BackendDevice, BackendStream, CpalBackend, InputDataCallback,
    StreamErrorCallback, StreamParams,
};
use crate::device::DeviceSelector;
use crate::error::DecibriError;
#[cfg(feature = "capture")]
use crate::stage::{build_capture_stage, CaptureStage};

/// Opt-in capture enhancement settings.
///
/// Carried on [`MicrophoneConfig::enhancement`]. Every field defaults to off, so
/// the capture path is unchanged unless a consumer opts in. `#[non_exhaustive]`:
/// construct it with [`EnhancementConfig::default`] and assign the fields you
/// want, so adding an enhancement later stays backward compatible.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct EnhancementConfig {
    /// Remove a constant (DC) offset from captured audio with a one-pole
    /// DC-blocking high-pass, applied after the channel and rate normalization.
    /// Default: false (off).
    pub dc_removal: bool,
}

/// Configuration for a microphone capture session.
///
/// `#[non_exhaustive]`: construct it with [`MicrophoneConfig::default`] and then
/// assign the public fields you need. Direct struct-literal construction from
/// another crate is intentionally not supported, so adding a field later stays
/// backward compatible.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MicrophoneConfig {
    /// Sample rate in Hz. Range: 1000–384000. Default: 16000.
    pub sample_rate: u32,
    /// Number of input channels. Range: 1–32. Default: 1.
    pub channels: u16,
    /// Frames per audio callback buffer. Range: 64–65536. Default: 1600.
    pub frames_per_buffer: u32,
    /// Device selection. Default: system default input.
    pub device: DeviceSelector,
    /// Opt-in capture enhancement. Default: every enhancement off, so the
    /// capture path is unchanged.
    pub enhancement: EnhancementConfig,
}

impl Default for MicrophoneConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            frames_per_buffer: 1600,
            device: DeviceSelector::Default,
            enhancement: EnhancementConfig::default(),
        }
    }
}

impl MicrophoneConfig {
    /// Validate the configuration: sample rate, channel count, and buffer size
    /// must fall within the supported ranges.
    pub fn validate(&self) -> Result<(), DecibriError> {
        if !(1000..=384000).contains(&self.sample_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
        }
        if !(1..=32).contains(&self.channels) {
            return Err(DecibriError::ChannelsOutOfRange);
        }
        if !(64..=65536).contains(&self.frames_per_buffer) {
            return Err(DecibriError::FramesPerBufferOutOfRange);
        }
        Ok(())
    }
}

/// A chunk of captured audio data.
///
/// `#[non_exhaustive]`: produced by the capture path and read field by field by
/// consumers. Sealing it keeps future metadata additions backward compatible.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AudioChunk {
    /// Interleaved f32 samples in range [-1.0, 1.0].
    pub data: Vec<f32>,
    /// Sample rate of this chunk.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

/// Bound on the capture channel. A stalled consumer cannot grow memory without
/// limit: once this many `AudioChunk`s are queued, the realtime callback drops
/// new chunks (counting them, see [`MicrophoneStream::overrun_count`]) rather
/// than blocking the audio thread or allocating without bound.
///
/// This is a memory bound, not a fixed-duration guarantee. One queued item is
/// one cpal callback buffer, whose duration is backend-dependent: on WASAPI cpal
/// ignores `BufferSize::Fixed` and delivers the driver period (often ~10 ms, not
/// `frames_per_buffer`), so 64 items can be anywhere from well under a second to
/// several seconds of audio. It is sized so a consumer that keeps pace never
/// drops; only a genuine stall does.
#[cfg(feature = "capture")]
const CAPTURE_CHANNEL_CAPACITY: usize = 64;

/// Memory bound for the pre-transform VAD tap, in seconds of audio at the target
/// rate. The tap accumulates the post-normalize signal in lockstep with the
/// delivered output and is drained by [`MicrophoneStream::vad_input`]. A consumer
/// that enables an enhancement step but never drains the tap would otherwise grow
/// memory without limit; beyond this many seconds the oldest tapped samples are
/// dropped. Sized far above the in-flight reblock depth, so an actively draining
/// VAD never reaches it and the tap stays aligned with the delivered output.
#[cfg(feature = "capture")]
const VAD_TAP_BOUND_SECS: usize = 2;

/// An open capture stream you pull [`AudioChunk`]s from.
///
/// Obtained from [`Microphone::start`]. Read audio with
/// [`next_chunk`](Self::next_chunk) (blocking, with a timeout) or
/// [`try_next_chunk`](Self::try_next_chunk) (non-blocking). Call
/// [`stop`](Self::stop) to end capture and release the device; dropping the
/// handle also releases it. The type is `Send + Sync`.
#[cfg(feature = "capture")]
pub struct MicrophoneStream {
    /// Owns the platform audio stream for the lifetime of this handle.
    ///
    /// Held behind [`BackendStream`](crate::backend::BackendStream), which keeps
    /// the stream behind a `Mutex<Option<...>>` so [`stop`](Self::stop) can drop
    /// it under `&self` to release the device while this type stays
    /// `Send + Sync` (the bindings require `Sync` for their `#[pyclass]` and
    /// `py.detach`). Dropping this field also releases the device. The test-only
    /// `tests::test_stream()` helper builds it with `BackendStream::empty()`
    /// (no real device); production stores the opened stream.
    _stream: BackendStream,
    receiver: Receiver<AudioChunk>,
    running: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
    // Last device/driver error reported by the cpal error callback while
    // streaming, retrievable via [`take_last_error`](Self::take_last_error).
    // Lets a consumer that sees a closed stream distinguish a driver failure
    // from an explicit `stop()`. The cpal callback writes it; consumers drain
    // it. Uncontended in practice (written at most once, on failure).
    last_error: Arc<Mutex<Option<DecibriError>>>,
    // Count of capture buffers dropped because the channel was full (a stalled
    // consumer). Incremented by the realtime callback, read via
    // [`overrun_count`](Self::overrun_count). Bounds memory by dropping rather
    // than queuing without limit.
    overruns: Arc<AtomicU64>,
    // Consumer-side re-block buffer: a FIFO of interleaved `f32` samples pulled
    // from `receiver`, drained in fixed `samples`-sized blocks so
    // [`next_chunk`](Self::next_chunk) / [`try_next_chunk`](Self::try_next_chunk)
    // deliver exactly the requested size regardless of the device's native
    // buffer size. Lives behind a `Mutex` (kept off the realtime callback, which
    // only `try_send`s native buffers) so the type stays `Send + Sync` and
    // concurrent consumers serialize. `Mutex<VecDeque<f32>>` is `Send + Sync`
    // because `VecDeque<f32>` is `Send`.
    reblock_buffer: Mutex<VecDeque<f32>>,
    // The capture stage chain, applied to each native block before it lands in
    // `reblock_buffer`. `None` when no conditioning is needed (an already-mono
    // device), keeping the drain on the direct, zero-cost A0b path. When `Some`,
    // the chain runs behind its own `Mutex` so the stage buffers mutate under a
    // shared `&self` while the type stays `Send + Sync` (a `CaptureStage` is
    // `Send`, so `Mutex<CaptureStage>` is `Send + Sync`).
    capture_stage: Option<Mutex<CaptureStage>>,
    // One-time guard for the close-path chain flush. The chain's stages carry
    // conditioning state between blocks (the resampler holds its anti-alias
    // filter's group-delay tail); at close that held tail is drained once into
    // `reblock_buffer` so it is delivered rather than dropped. Set the first time
    // close is detected with a chain present, under the `reblock_buffer` lock, so
    // the drain runs exactly once. `AtomicBool` keeps the type `Send + Sync`.
    chain_flushed: AtomicBool,
    // Pre-transform (post-normalize) side channel for the VAD feed. `Some` only
    // when the chain has a `transform` segment, so the delivered output (which is
    // post-transform) differs from the signal a detector should read. Populated in
    // lockstep with `reblock_buffer` during `ingest`/`flush_chain` (same sample
    // count, the transform preserves length) and drained by
    // [`vad_input`](Self::vad_input). `None` when there is no transform: the
    // delivered output already is the post-normalize signal, so the tap is unused
    // and `vad_input` returns `None` with zero overhead. `Mutex<VecDeque<f32>>` is
    // `Send + Sync`, so the type stays `Send + Sync`.
    vad_tap: Option<Mutex<VecDeque<f32>>>,
    // Memory bound (in samples) for `vad_tap`, computed from the target rate at
    // construction (see `VAD_TAP_BOUND_SECS`). Oldest tapped samples are dropped
    // beyond this so a consumer that enables an enhancement step but never drains
    // `vad_input` cannot grow memory without limit. Never reached while a VAD
    // actively drains the tap, so it does not perturb alignment.
    vad_tap_cap: usize,
}

#[cfg(feature = "capture")]
impl MicrophoneStream {
    /// Direct access to the underlying `crossbeam_channel::Receiver`.
    ///
    /// Intended for **in-process Rust consumers** and for bindings (like the
    /// decibri Node.js addon) that integrate the channel into their own
    /// drain pump or event loop.
    ///
    /// FFI bindings targeting languages without native `crossbeam_channel`
    /// support, such as Python and the eventual mobile platforms, should
    /// prefer [`try_next_chunk`](Self::try_next_chunk) and
    /// [`next_chunk`](Self::next_chunk): they expose the same data with a
    /// three-state return (`Some` / `None` / `Err(MicrophoneStreamClosed)`)
    /// that maps cleanly across a language boundary.
    pub fn receiver(&self) -> &Receiver<AudioChunk> {
        &self.receiver
    }

    /// Attempt to read exactly `samples` interleaved samples without blocking.
    ///
    /// `samples` is the requested block size in interleaved `f32` samples
    /// (frames times channels). The device's native capture buffers are
    /// re-blocked on the consumer side, so every returned chunk holds exactly
    /// `samples` samples. `samples` should be non-zero and, for frame
    /// alignment, a multiple of the channel count.
    ///
    /// # Returns
    /// - `Ok(Some(chunk))`: a full block of exactly `samples` samples was
    ///   available and has been dequeued.
    /// - `Ok(None)`: the stream is open and running, but fewer than `samples`
    ///   samples are buffered. Try again shortly, or call
    ///   [`next_chunk`](Self::next_chunk) to block until a full block arrives.
    /// - `Err(DecibriError::MicrophoneStreamClosed)`: the stream is closed
    ///   (either by explicit [`stop`](Self::stop) or by an audio-driver error
    ///   reported via the cpal error callback) and the buffer is now empty (any
    ///   final tail was already delivered as a short chunk). No further chunks
    ///   will ever be available.
    ///
    /// Buffered samples that form one or more full blocks are delivered first.
    /// Once the stream closes with fewer than `samples` remaining, the final
    /// partial block (1..`samples` samples) is delivered as one short chunk, and
    /// the closed signal is returned on the next call once the buffer is empty.
    /// No captured sample is dropped: every chunk is exactly `samples` long
    /// except the final chunk at close, which carries the remaining tail.
    ///
    /// # Thread safety
    /// May be called from any thread. Takes the re-block buffer mutex, so
    /// concurrent callers serialize on it; a non-blocking
    /// `crossbeam_channel::try_recv` drains the native buffers into it.
    ///
    /// # Stability
    /// Part of decibri's stable FFI-consumer API surface, alongside
    /// [`next_chunk`](Self::next_chunk).
    pub fn try_next_chunk(&self, samples: usize) -> Result<Option<AudioChunk>, DecibriError> {
        let mut buf = self
            .reblock_buffer
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        // Pull every immediately-available native buffer into the re-block
        // buffer without blocking.
        let mut disconnected = false;
        loop {
            match self.receiver.try_recv() {
                Ok(chunk) => self.ingest(&mut buf, chunk)?,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if buf.len() >= samples {
            // A full block is ready; deliver buffered data first, even when the
            // stream has since closed.
            Ok(Some(self.take_block(&mut buf, samples)))
        } else if disconnected || !self.is_open() {
            // Closed with fewer than a full block left. Drain the chain's
            // end-of-stream tail (once) into the buffer first, then deliver the
            // remaining samples as full blocks plus one final short chunk.
            self.flush_chain(&mut buf)?;
            self.take_block_or_closed(&mut buf, samples)
        } else {
            // Open, but not yet a full block. Try again shortly.
            Ok(None)
        }
    }

    /// Read exactly `samples` interleaved samples, blocking the calling thread
    /// until a full block arrives, the stream closes, or `timeout` elapses.
    ///
    /// `samples` is the requested block size in interleaved `f32` samples
    /// (frames times channels). The device's native capture buffers are
    /// re-blocked on the consumer side, so a returned chunk holds exactly
    /// `samples` samples. `samples` should be non-zero and, for frame
    /// alignment, a multiple of the channel count.
    ///
    /// # Arguments
    /// - `timeout = None`: block indefinitely until a full block arrives or the
    ///   stream closes.
    /// - `timeout = Some(dur)`: block at most `dur`; return `Ok(None)` if the
    ///   deadline passes before a full block accumulates. The partial block is
    ///   retained for the next call.
    ///
    /// # Returns
    /// - `Ok(Some(chunk))`: a full block of exactly `samples` samples was
    ///   received within the deadline.
    /// - `Ok(None)`: `timeout` elapsed before a full block accumulated. The
    ///   stream is still open and the partial stays buffered.
    /// - `Err(DecibriError::MicrophoneStreamClosed)`: the stream closed and the
    ///   buffer is now empty. Any full blocks buffered at the time of close are
    ///   delivered first, then the final partial block (1..`samples` samples) as
    ///   one short chunk; this error is only returned once the buffer is empty.
    ///   No captured sample is dropped.
    ///
    /// # Thread safety
    /// May be called from any thread. Blocks only the calling thread; other
    /// threads can call [`stop`](Self::stop) concurrently to unblock this call
    /// within approximately 20 ms. Holds the re-block buffer mutex for the
    /// duration of the call, so concurrent reads serialize.
    ///
    /// Implementation note: this method polls both the channel and
    /// [`is_open`](Self::is_open) at a short interval. An explicit
    /// [`stop`](Self::stop) disconnects the channel (it drops the cpal `Stream`,
    /// and with it the sender), which wakes a blocked wait promptly; the poll
    /// additionally covers a driver-error stop, which flips
    /// [`is_open`](Self::is_open) without dropping the stream.
    ///
    /// # Stability
    /// Part of decibri's stable FFI-consumer API surface.
    pub fn next_chunk(
        &self,
        samples: usize,
        timeout: Option<Duration>,
    ) -> Result<Option<AudioChunk>, DecibriError> {
        // Poll both the channel and `is_open` at this cadence so concurrent
        // `stop()` calls unblock a waiter within one interval. 20 ms is well
        // below a typical audio frame period (100 ms at 16 kHz / 1600
        // frames) so the extra wakeups cost negligible CPU.
        const POLL_INTERVAL: Duration = Duration::from_millis(20);

        let deadline = timeout.map(|t| Instant::now() + t);
        let mut buf = self
            .reblock_buffer
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        loop {
            // Fast path: a full block is already buffered.
            if buf.len() >= samples {
                return Ok(Some(self.take_block(&mut buf, samples)));
            }

            let wait = match deadline {
                Some(dl) => {
                    let now = Instant::now();
                    if now >= dl {
                        // Deadline reached. Absorb any last-moment arrivals,
                        // then deliver a full block if one is now ready, else
                        // `None` (the partial stays buffered for the next call).
                        self.drain_available(&mut buf)?;
                        if buf.len() >= samples {
                            return Ok(Some(self.take_block(&mut buf, samples)));
                        }
                        return Ok(None);
                    }
                    std::cmp::min(dl - now, POLL_INTERVAL)
                }
                None => POLL_INTERVAL,
            };

            match self.receiver.recv_timeout(wait) {
                Ok(chunk) => {
                    self.ingest(&mut buf, chunk)?;
                    // Loop: re-check whether a full block is ready.
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Re-check whether `stop()` or a driver error fired while we
                    // waited (a driver-error close flips `is_open` without
                    // disconnecting the channel).
                    if !self.is_open() {
                        self.drain_available(&mut buf)?;
                        self.flush_chain(&mut buf)?;
                        return self.take_block_or_closed(&mut buf, samples);
                    }
                    // Stream still alive; loop with whatever deadline remains.
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Channel closed and drained. Drain the chain's end-of-stream
                    // tail (once) into the buffer, then deliver any remaining full
                    // blocks and the final short tail before reporting closed.
                    self.flush_chain(&mut buf)?;
                    return self.take_block_or_closed(&mut buf, samples);
                }
            }
        }
    }

    /// Drain exactly `samples` interleaved samples off the front of the re-block
    /// buffer into a fresh [`AudioChunk`], stamping it with this stream's sample
    /// rate and channel count. The caller guarantees `buf.len() >= samples`.
    /// Sample values and order are preserved; only the chunk boundary changes.
    fn take_block(&self, buf: &mut VecDeque<f32>, samples: usize) -> AudioChunk {
        AudioChunk {
            data: buf.drain(..samples).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Close-path delivery. Drains a chunk off a closing/closed stream: a full
    /// `samples`-block if one remains, then any final tail (1..`samples`
    /// samples) as one short chunk, and finally `Err(MicrophoneStreamClosed)`
    /// once the buffer is empty. No captured sample is dropped at close.
    fn take_block_or_closed(
        &self,
        buf: &mut VecDeque<f32>,
        samples: usize,
    ) -> Result<Option<AudioChunk>, DecibriError> {
        if buf.len() >= samples {
            Ok(Some(self.take_block(buf, samples)))
        } else if buf.is_empty() {
            Err(DecibriError::MicrophoneStreamClosed)
        } else {
            // Final partial block: deliver the remaining 1..`samples` samples.
            let remaining = buf.len();
            Ok(Some(self.take_block(buf, remaining)))
        }
    }

    /// Move every immediately-available native buffer from the channel into the
    /// re-block buffer without blocking. Stops on an empty or disconnected
    /// channel; the caller inspects `buf.len()` and the close state afterwards.
    fn drain_available(&self, buf: &mut VecDeque<f32>) -> Result<(), DecibriError> {
        while let Ok(chunk) = self.receiver.try_recv() {
            self.ingest(buf, chunk)?;
        }
        Ok(())
    }

    /// Feed one native capture block into the re-block buffer, running the
    /// capture stage chain first when one is present.
    ///
    /// `None` chain: the block's samples are appended directly, byte-identical to
    /// the A0b path, with no lock, allocation, or copy beyond the existing
    /// reblock. `Some` chain: the chain runs behind its `Mutex` and its
    /// conditioned output is appended instead.
    ///
    /// When the chain has a transform, the post-normalize, pre-transform tap is
    /// captured into the VAD side channel in lockstep with the delivered output,
    /// so [`vad_input`](Self::vad_input) can hand a detector the signal before the
    /// enhancement step.
    fn ingest(&self, buf: &mut VecDeque<f32>, chunk: AudioChunk) -> Result<(), DecibriError> {
        match &self.capture_stage {
            None => {
                buf.extend(chunk.data);
                Ok(())
            }
            Some(stage) => {
                let mut stage = stage.lock().unwrap_or_else(PoisonError::into_inner);
                let out = stage.run(&chunk.data)?;
                buf.extend(out.iter().copied());
                // Copy the pre-transform tap out, release the stage lock, then
                // append it to the VAD buffer so the two locks are never nested.
                let tap = stage.has_transform().then(|| stage.tap().to_vec());
                drop(stage);
                if let Some(tap) = tap {
                    self.push_vad_tap(tap);
                }
                Ok(())
            }
        }
    }

    /// Append post-normalize samples to the VAD tap, dropping the oldest beyond
    /// the [`VAD_TAP_BOUND_SECS`] memory bound. A no-op when the tap is inactive
    /// (no transform). The bound only trips when a transform is enabled but the
    /// tap is not being drained, so it never perturbs an actively draining VAD.
    fn push_vad_tap(&self, samples: Vec<f32>) {
        if let Some(tap) = &self.vad_tap {
            let mut tap = tap.lock().unwrap_or_else(PoisonError::into_inner);
            tap.extend(samples);
            if tap.len() > self.vad_tap_cap {
                let excess = tap.len() - self.vad_tap_cap;
                tap.drain(..excess);
            }
        }
    }

    /// Drain the capture chain's end-of-stream tail into the re-block buffer,
    /// exactly once, when the stream closes.
    ///
    /// The chain's stages carry conditioning state between blocks (the resampler
    /// holds its anti-alias filter's group-delay tail). At close this drains that
    /// held tail through the chain and appends it to `buf`, so the existing
    /// reblock delivers it as part of the final chunk(s) rather than dropping it.
    /// Runs at most once, guarded by `chain_flushed`; a stream with no chain
    /// (`None`) drains nothing and the direct path is unchanged. The caller holds
    /// the `reblock_buffer` lock, which orders this drain against every other
    /// buffer access and makes the guard's check-and-set effectively atomic.
    ///
    /// Called after the last native block has been drained through
    /// [`ingest`](Self::ingest) and before the close-time delivery, so the tail
    /// is appended after all processed output and reblocked normally.
    fn flush_chain(&self, buf: &mut VecDeque<f32>) -> Result<(), DecibriError> {
        if let Some(stage) = &self.capture_stage {
            if !self.chain_flushed.swap(true, Ordering::Relaxed) {
                let mut stage = stage.lock().unwrap_or_else(PoisonError::into_inner);
                let mut tail = Vec::new();
                stage.flush(&mut tail)?;
                buf.extend(tail);
                // Capture the post-normalize flush tail into the VAD tap too, so
                // the tap does not desync from the delivered output at close: the
                // resampler's group-delay tail is part of the post-normalize
                // signal the detector should read.
                let tap = stage.has_transform().then(|| stage.tap().to_vec());
                drop(stage);
                if let Some(tap) = tap {
                    self.push_vad_tap(tap);
                }
            }
        }
        Ok(())
    }

    /// Check whether the stream is still actively capturing.
    pub fn is_open(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// The sample rate (Hz) this stream was opened with.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The channel count this stream was opened with.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// The pre-transform (post-normalize) samples for the VAD feed, drained in
    /// lockstep with [`next_chunk`](Self::next_chunk).
    ///
    /// Binding-internal plumbing: a binding that runs voice-activity detection
    /// calls this right after a `next_chunk` / `try_next_chunk` delivery, passing
    /// the delivered chunk's length, and feeds the returned samples to the
    /// detector instead of the delivered chunk. This makes the detector read the
    /// signal BEFORE the opt-in enhancement step, so enabling enhancement does not
    /// change detection. Not part of the stable FFI-consumer surface.
    ///
    /// # Returns
    /// - `Some(v)`: the chain has an enhancement step, so the delivered output is
    ///   the post-enhancement signal; `v` holds up to `samples` post-normalize
    ///   samples, the pre-enhancement counterpart of the just-delivered block.
    ///   Sample-aligned with that block because the enhancement step preserves
    ///   length.
    /// - `None`: the chain has no enhancement step, so the delivered chunk already
    ///   is the post-normalize signal; feed the detector the delivered chunk
    ///   exactly as before, with no allocation or copy.
    ///
    /// # Thread safety
    /// Takes only the tap mutex (never the reblock or chain locks), so it composes
    /// with a concurrent [`next_chunk`](Self::next_chunk). Sample-alignment relies
    /// on the same consumer calling it once per delivered block, as the binding
    /// pump does.
    pub fn vad_input(&self, samples: usize) -> Option<Vec<f32>> {
        let tap = self.vad_tap.as_ref()?;
        let mut tap = tap.lock().unwrap_or_else(PoisonError::into_inner);
        let take = samples.min(tap.len());
        Some(tap.drain(..take).collect())
    }

    /// Take the last device/driver error reported while streaming, if any.
    ///
    /// When the cpal error callback fires (device unplug, driver failure) it
    /// records a typed [`DecibriError::DeviceFailed`] and closes the stream.
    /// Reading it here drains it (returns `None` afterwards), letting a
    /// consumer that observes a closed stream tell a driver failure apart from
    /// an explicit [`stop`](Self::stop).
    pub fn take_last_error(&self) -> Option<DecibriError> {
        self.last_error.lock().ok().and_then(|mut slot| slot.take())
    }

    /// Total number of capture buffers dropped because the channel was full
    /// (a consumer that could not keep up). Stays 0 while the consumer keeps
    /// pace; a rising count means audio is being dropped to bound memory.
    pub fn overrun_count(&self) -> u64 {
        self.overruns.load(Ordering::Relaxed)
    }

    /// Stop capturing audio and release the device.
    ///
    /// Flips the running flag, then drops the held stream, so the OS releases
    /// the input device (and the mic-in-use indicator clears) at once rather
    /// than only when this handle is dropped. Dropping the stream also
    /// disconnects the capture channel; chunks already buffered remain readable
    /// via [`try_next_chunk`](Self::try_next_chunk) or
    /// [`next_chunk`](Self::next_chunk) until the buffer is drained, after which
    /// those methods return `Err(MicrophoneStreamClosed)`.
    ///
    /// May be called from any thread; wakes a concurrent
    /// [`next_chunk`](Self::next_chunk) waiter promptly (the channel disconnect
    /// unblocks it). Releasing the device blocks briefly while the audio thread
    /// tears down.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        // Drop the held stream to release the OS device now, not only on drop.
        // Poison-tolerant and idempotent (see `BackendStream::stop`).
        self._stream.stop();
    }
}

/// An input device you capture audio from.
///
/// Build one from a [`MicrophoneConfig`], then [`start`](Self::start) it to open
/// the device and obtain a [`MicrophoneStream`] of [`AudioChunk`]s. The playback
/// counterpart is [`Speaker`](crate::Speaker).
///
/// ```no_run
/// use decibri::{Microphone, MicrophoneConfig};
///
/// let mic = Microphone::new(MicrophoneConfig::default())?;
/// let stream = mic.start()?;
/// # Ok::<(), decibri::DecibriError>(())
/// ```
#[cfg(feature = "capture")]
pub struct Microphone {
    config: MicrophoneConfig,
    device: BackendDevice,
}

#[cfg(feature = "capture")]
impl Microphone {
    /// Create a microphone: validates the [`MicrophoneConfig`] and resolves the
    /// selected input device. Does not open the stream; call [`start`](Self::start)
    /// for that.
    pub fn new(config: MicrophoneConfig) -> Result<Self, DecibriError> {
        config.validate()?;
        let device = CpalBackend.resolve_input_device(&config.device)?;
        Ok(Self { config, device })
    }

    /// List the available input devices.
    pub fn devices() -> Result<Vec<crate::device::MicrophoneInfo>, DecibriError> {
        crate::device::input_devices()
    }

    /// Start capturing audio. Returns a stream handle with a receiver for audio chunks.
    pub fn start(&self) -> Result<MicrophoneStream, DecibriError> {
        let (sender, receiver): (Sender<AudioChunk>, Receiver<AudioChunk>) =
            crossbeam_channel::bounded(CAPTURE_CHANNEL_CAPACITY);

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        // The requested rate is the TARGET the consumer receives. The device is
        // opened at its native rate, settled here from its default supported
        // format, and the capture chain resamples native -> target so delivery
        // is at exactly the requested rate.
        let target_rate = self.config.sample_rate;
        let native_rate = CpalBackend.native_input_rate(&self.device)?;
        let channels = self.config.channels;
        let frames_per_buffer = self.config.frames_per_buffer;

        // Build the normalize chain for this device. The target is mono (1
        // channel) at the requested rate: `build_capture_stage` adds a downmix
        // for a multichannel device and a resample when the native rate differs,
        // and returns `None` only for a mono device already at the target rate.
        // The stream reports the OUTPUT channel count (mono when the chain
        // downmixes, which is what the exact-size `samples` math is counted in)
        // and the target rate.
        let target_channels: u16 = 1;
        let capture_stage = build_capture_stage(
            channels,
            target_channels,
            native_rate,
            target_rate,
            &self.config.enhancement,
        )?;
        let output_channels = if channels > target_channels {
            target_channels
        } else {
            channels
        };

        let err_running = running.clone();
        let last_error = Arc::new(Mutex::new(None));
        let err_last_error = last_error.clone();
        let overruns = Arc::new(AtomicU64::new(0));
        let overruns_cb = overruns.clone();

        // Realtime data callback: capture, copy, non-blocking send. Identical
        // work as before; the seam wraps only stream construction.
        let on_data: InputDataCallback = Box::new(move |data: &[f32]| {
            if !running_clone.load(Ordering::Relaxed) {
                return;
            }
            // The native chunk carries the device's capture format (its native
            // rate and channel count); the consumer-side chain normalizes it.
            let chunk = AudioChunk {
                data: data.to_vec(),
                sample_rate: native_rate,
                channels,
            };
            // Non-blocking send: the realtime audio thread must never block. On
            // a full channel (a stalled consumer) drop this chunk and count it
            // rather than growing memory without bound; a disconnected receiver
            // also just discards.
            match sender.try_send(chunk) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    overruns_cb.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Disconnected(_)) => {}
            }
        });

        // Error callback: record the typed cause and mark the stream closed so a
        // consumer that sees the stream close can distinguish a driver failure
        // from stop(). The backend builds the typed `DeviceFailed`.
        let on_error: StreamErrorCallback = Box::new(move |err: DecibriError| {
            eprintln!("{err}");
            if let Ok(mut slot) = err_last_error.lock() {
                *slot = Some(err);
            }
            err_running.store(false, Ordering::Relaxed);
        });

        // Open the device at its native rate; the capture chain resamples to the
        // target. A device already at the target rate has no resample stage.
        let params = StreamParams {
            channels,
            sample_rate: native_rate,
            frames_per_buffer: Some(frames_per_buffer),
        };
        let _stream = CpalBackend.open_input_stream(&self.device, &params, on_data, on_error)?;

        // The VAD tap is active only when the chain has a transform segment, so
        // the delivered (post-transform) output differs from the pre-transform
        // signal a detector should read. With no transform the delivered output
        // already is that signal, so no tap is allocated.
        let vad_tap = match &capture_stage {
            Some(stage) if stage.has_transform() => Some(Mutex::new(VecDeque::new())),
            _ => None,
        };
        let vad_tap_cap = target_rate as usize * VAD_TAP_BOUND_SECS;

        Ok(MicrophoneStream {
            _stream,
            receiver,
            running,
            // Consumers receive the target rate; `take_block` stamps it on every
            // delivered chunk.
            sample_rate: target_rate,
            channels: output_channels,
            last_error,
            overruns,
            reblock_buffer: Mutex::new(VecDeque::new()),
            capture_stage: capture_stage.map(Mutex::new),
            chain_flushed: AtomicBool::new(false),
            vad_tap,
            vad_tap_cap,
        })
    }
}

#[cfg(all(test, feature = "capture"))]
mod tests {
    use super::*;
    use std::thread;

    /// Construct a synthetic `MicrophoneStream` with no underlying cpal device,
    /// the given stage chain, and the given output channel count. Returns the
    /// stream plus test-side handles to inject native chunks and flip running.
    fn test_stream_with(
        capture_stage: Option<CaptureStage>,
        channels: u16,
    ) -> (MicrophoneStream, Sender<AudioChunk>, Arc<AtomicBool>) {
        let (sender, receiver) = crossbeam_channel::unbounded::<AudioChunk>();
        let running = Arc::new(AtomicBool::new(true));
        let vad_tap = match &capture_stage {
            Some(stage) if stage.has_transform() => Some(Mutex::new(VecDeque::new())),
            _ => None,
        };
        let stream = MicrophoneStream {
            _stream: BackendStream::empty(),
            receiver,
            running: running.clone(),
            sample_rate: 16000,
            channels,
            last_error: Arc::new(Mutex::new(None)),
            overruns: Arc::new(AtomicU64::new(0)),
            reblock_buffer: Mutex::new(VecDeque::new()),
            capture_stage: capture_stage.map(Mutex::new),
            vad_tap,
            vad_tap_cap: 16000 * VAD_TAP_BOUND_SECS,
            chain_flushed: AtomicBool::new(false),
        };
        (stream, sender, running)
    }

    /// Mono stream with no stage chain (the `None`, A0b-identical path).
    fn test_stream() -> (MicrophoneStream, Sender<AudioChunk>, Arc<AtomicBool>) {
        test_stream_with(None, 1)
    }

    fn make_chunk(first_sample: f32) -> AudioChunk {
        AudioChunk {
            data: vec![first_sample],
            sample_rate: 16000,
            channels: 1,
        }
    }

    /// A native chunk carrying an arbitrary run of interleaved samples, for the
    /// re-blocking tests (native buffers are variable-size).
    fn make_native_chunk(data: Vec<f32>) -> AudioChunk {
        AudioChunk {
            data,
            sample_rate: 16000,
            channels: 1,
        }
    }

    #[test]
    fn test_try_next_chunk_returns_none_when_empty() {
        let (stream, _sender, _running) = test_stream();
        let result = stream.try_next_chunk(1).unwrap();
        assert!(
            result.is_none(),
            "try_next_chunk on empty open stream should return Ok(None)"
        );
    }

    #[test]
    fn test_try_next_chunk_returns_chunk_when_available() {
        let (stream, sender, _running) = test_stream();
        sender.send(make_chunk(0.42)).unwrap();

        let result = stream.try_next_chunk(1).unwrap();
        let chunk = result.expect("should have received the injected chunk");
        assert_eq!(chunk.data, vec![0.42]);
    }

    #[test]
    fn test_try_next_chunk_returns_err_when_closed() {
        let (stream, sender, running) = test_stream();
        drop(sender); // simulate cpal stream dropping the sender
        running.store(false, Ordering::Relaxed);

        let err = stream.try_next_chunk(1).unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    #[test]
    fn test_next_chunk_blocks_until_chunk_arrives() {
        let (stream, sender, _running) = test_stream();

        let producer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            sender.send(make_chunk(0.77)).unwrap();
        });

        let result = stream.next_chunk(1, None).unwrap();
        let chunk = result.expect("should have received the eventually-pushed chunk");
        assert_eq!(chunk.data, vec![0.77]);

        producer.join().unwrap();
    }

    #[test]
    fn test_next_chunk_timeout_returns_none() {
        let (stream, _sender, _running) = test_stream();
        let start = Instant::now();
        let result = stream
            .next_chunk(1, Some(Duration::from_millis(50)))
            .unwrap();
        let elapsed = start.elapsed();

        assert!(
            result.is_none(),
            "next_chunk with timeout and no arrivals should return Ok(None)"
        );
        // Lower bound: at least the requested timeout.
        assert!(
            elapsed >= Duration::from_millis(40),
            "next_chunk returned too early: {elapsed:?}"
        );
    }

    #[test]
    fn test_next_chunk_flushes_buffered_before_closed_err() {
        let (stream, sender, running) = test_stream();
        sender.send(make_chunk(1.0)).unwrap();
        sender.send(make_chunk(2.0)).unwrap();
        drop(sender); // simulate cpal stream drop
        running.store(false, Ordering::Relaxed);

        // First two calls drain the buffer, third reports closed.
        let c1 = stream
            .next_chunk(1, Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(c1.unwrap().data, vec![1.0]);

        let c2 = stream
            .next_chunk(1, Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(c2.unwrap().data, vec![2.0]);

        let err = stream
            .next_chunk(1, Some(Duration::from_millis(100)))
            .unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    #[test]
    fn test_next_chunk_returns_closed_within_polling_interval_after_stop() {
        let (stream, _sender, running) = test_stream();

        let r = running.clone();
        let stopper = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            r.store(false, Ordering::Relaxed);
        });

        // next_chunk with no timeout should wake up on the next 20 ms poll
        // after stop() flips the running flag. Give a generous 250 ms ceiling
        // to absorb scheduler jitter on loaded CI runners.
        let start = Instant::now();
        let err = stream.next_chunk(1, None).unwrap_err();
        let elapsed = start.elapsed();

        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
        assert!(
            elapsed < Duration::from_millis(250),
            "next_chunk took too long to detect stop(): {elapsed:?}"
        );

        stopper.join().unwrap();
    }

    /// Re-blocking delivers full blocks of exactly the requested size, in order,
    /// then a final short chunk carrying the tail, with no sample lost,
    /// reordered, or altered. Irregular native buffers carrying the values 0..14
    /// (not a multiple of the block size 4) are re-blocked; the concatenation of
    /// the delivered chunks equals the FULL input stream, tail included.
    #[test]
    fn test_next_chunk_delivers_exact_blocks_then_final_tail() {
        let (stream, sender, running) = test_stream();
        sender
            .send(make_native_chunk(vec![0.0, 1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        sender.send(make_native_chunk(vec![5.0, 6.0])).unwrap();
        sender
            .send(make_native_chunk(vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            ]))
            .unwrap();
        drop(sender); // no more native buffers will arrive
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(chunk)) => chunks.push(chunk.data),
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Every block is exactly `samples` long except the final tail (14 % 4 == 2).
        let (last, full) = chunks.split_last().expect("at least one chunk delivered");
        for block in full {
            assert_eq!(
                block.len(),
                samples,
                "non-final blocks are exactly `samples` long"
            );
        }
        assert_eq!(last.len(), 2, "the final chunk carries the 2-sample tail");

        // Sample identity: the full input stream is reconstructed, nothing dropped.
        let collected: Vec<f32> = chunks.into_iter().flatten().collect();
        assert_eq!(
            collected,
            (0..14).map(|n| n as f32).collect::<Vec<f32>>(),
            "re-blocking preserves every sample value and order, including the tail"
        );
    }

    /// A single native buffer larger than the requested block size is split
    /// into successive full blocks across calls, the leftover carried forward;
    /// the final sub-block remainder is delivered as a short chunk at close.
    #[test]
    fn test_next_chunk_splits_large_native_buffer_into_blocks() {
        let (stream, sender, running) = test_stream();
        sender
            .send(make_native_chunk((0..10).map(|n| n as f32).collect()))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 3;
        let b1 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("first block");
        let b2 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("second block");
        let b3 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("third block");
        assert_eq!(b1.data, vec![0.0, 1.0, 2.0]);
        assert_eq!(b2.data, vec![3.0, 4.0, 5.0]);
        assert_eq!(b3.data, vec![6.0, 7.0, 8.0]);

        // 10 % 3 == 1 leftover sample: delivered as a final short chunk, then closed.
        let tail = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("final tail");
        assert_eq!(tail.data, vec![9.0]);
        let err = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    /// `try_next_chunk` returns `Ok(None)` while fewer than a full block are
    /// buffered on an open stream, then `Ok(Some)` once enough native samples
    /// accumulate; the block is exactly `samples` long.
    #[test]
    fn test_try_next_chunk_short_returns_none_then_full_block() {
        let (stream, sender, _running) = test_stream();
        let samples = 4;

        sender.send(make_native_chunk(vec![10.0, 11.0])).unwrap();
        assert!(
            stream.try_next_chunk(samples).unwrap().is_none(),
            "fewer than `samples` on an open stream returns Ok(None)"
        );

        sender.send(make_native_chunk(vec![12.0, 13.0])).unwrap();
        let chunk = stream
            .try_next_chunk(samples)
            .unwrap()
            .expect("a full block is now available");
        assert_eq!(chunk.data, vec![10.0, 11.0, 12.0, 13.0]);

        // The buffer is empty again.
        assert!(stream.try_next_chunk(samples).unwrap().is_none());
    }

    /// The final partial block (fewer than `samples` samples at close) is
    /// delivered as one short chunk, then the stream reports closed.
    #[test]
    fn test_final_partial_tail_delivered_on_close() {
        let (stream, sender, running) = test_stream();
        sender
            .send(make_native_chunk(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let c1 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("first full block");
        assert_eq!(c1.data, vec![0.0, 1.0, 2.0, 3.0]);

        // The remaining 2 samples are delivered as a final short chunk.
        let tail = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("final tail");
        assert_eq!(tail.data, vec![4.0, 5.0]);

        // Now the buffer is empty: the stream reports closed.
        let err = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    /// `try_next_chunk` delivers the final tail as a short chunk when the stream
    /// closes with a partial buffered, then reports closed.
    #[test]
    fn test_try_next_chunk_delivers_tail_on_close() {
        let (stream, sender, running) = test_stream();
        sender.send(make_native_chunk(vec![1.0, 2.0])).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let tail = stream
            .try_next_chunk(samples)
            .unwrap()
            .expect("final tail delivered");
        assert_eq!(tail.data, vec![1.0, 2.0]);

        let err = stream.try_next_chunk(samples).unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    /// Empty-chain no-op (the `None` path): with no stage chain (a mono device),
    /// `next_chunk` delivers the raw reblocked samples byte-identically to the
    /// A0b path, exact size and final-tail behaviour included.
    #[test]
    fn test_empty_chain_is_byte_identical_noop() {
        let (stream, sender, running) = test_stream(); // mono, capture_stage = None
        assert!(stream.capture_stage.is_none(), "a mono stream has no chain");
        sender
            .send(make_native_chunk(vec![0.0, 1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        sender
            .send(make_native_chunk(vec![5.0, 6.0, 7.0, 8.0]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let mut collected: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => collected.extend_from_slice(&c.data),
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        // None path == A0b reblock: 9 samples -> two blocks of 4 then a 1-sample
        // tail, every value passed through untransformed and in order.
        assert_eq!(collected, (0..9).map(|n| n as f32).collect::<Vec<f32>>());
    }

    /// Cost no-op (the `None` path): a mono device builds no chain, so the drain
    /// stays on the direct reblock. The `None` arm of `ingest` is a plain
    /// `buf.extend(chunk.data)` with no chain lock, allocation, or copy,
    /// structurally identical to A0b.
    #[test]
    fn test_mono_device_builds_no_chain() {
        assert!(
            build_capture_stage(1, 1, 16000, 16000, &EnhancementConfig::default())
                .unwrap()
                .is_none(),
            "a mono device at the target rate needs no normalize chain"
        );
        let (stream, _sender, _running) = test_stream();
        assert!(
            stream.capture_stage.is_none(),
            "no CaptureStage is allocated for a mono stream"
        );
    }

    /// Auto-normalize (the `Some([Downmix])` path): a multichannel source is
    /// downmixed to correct mono, exact size holds on the mono output, the
    /// downmix is sample-identical to `sample::downmix_to_mono`, and the final
    /// tail is delivered.
    #[test]
    fn test_downmix_chain_yields_correct_mono() {
        let stage = build_capture_stage(2, 1, 16000, 16000, &EnhancementConfig::default()).unwrap();
        assert!(stage.is_some(), "a stereo device gets a downmix chain");
        let (stream, sender, running) = test_stream_with(stage, 1); // output is mono

        // Stereo native chunks (interleaved L,R). Frame means:
        //   [0.5,0.3]->0.4 [0.4,0.6]->0.5 ; [0.0,0.2]->0.1 [0.8,0.4]->0.6 [0.1,0.1]->0.1
        sender
            .send(make_native_chunk(vec![0.5, 0.3, 0.4, 0.6]))
            .unwrap();
        sender
            .send(make_native_chunk(vec![0.0, 0.2, 0.8, 0.4, 0.1, 0.1]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 2; // 2 MONO samples per block
        let mut collected: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.channels, 1, "the normalized output is mono");
                    collected.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Mono stream [0.4,0.5,0.1,0.6,0.1] -> blocks [0.4,0.5] [0.1,0.6], tail [0.1].
        let expected = [0.4, 0.5, 0.1, 0.6, 0.1];
        assert_eq!(
            collected.len(),
            expected.len(),
            "nothing dropped beyond the final-tail rule"
        );
        for (got, want) in collected.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    /// Resample-in-capture (the `Some([ResampleStage])` path): a mono device
    /// above the target rate has its audio resampled to the target inside
    /// `ingest`, exact-size delivery holds on the resampled output, every
    /// delivered chunk reports the target rate and mono, and the total sample
    /// count tracks the 1:3 rate ratio (48 kHz -> 16 kHz).
    #[test]
    fn test_resample_chain_delivers_exact_blocks_at_target_rate() {
        let stage = build_capture_stage(1, 1, 48_000, 16_000, &EnhancementConfig::default())
            .unwrap()
            .expect("48k mono -> resample chain");
        // test_stream_with stamps the stream at 16 kHz (the target), mono.
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();
        sender.send(make_native_chunk(input.clone())).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320; // 20 ms at 16 kHz
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.channels, 1, "resampled output is mono");
                    assert_eq!(c.sample_rate, 16_000, "chunks report the target rate");
                    blocks.push(c.data);
                }
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        let (last, full) = blocks.split_last().expect("at least one block delivered");
        for block in full {
            assert_eq!(
                block.len(),
                samples,
                "non-final blocks are exactly `samples` long on the resampled output"
            );
        }
        assert!(
            !last.is_empty() && last.len() <= samples,
            "the final tail carries 1..=`samples` resampled samples"
        );

        let total: usize = blocks.iter().map(|b| b.len()).sum();
        // A 1:3 downsample of the 24000-sample input yields roughly 8000 output
        // samples, plus the resampler's group-delay tail now drained at close, so
        // the total sits just above input/3 and well below half the input.
        assert!(
            total > input.len() / 4 && total < input.len() / 2,
            "resampled total {total} tracks the 1:3 ratio of {} input",
            input.len()
        );
    }

    /// Resample close path, the no-sample-dropped proof: a known signal fed
    /// through the resampling capture chain and read to close delivers the
    /// COMPLETE resampled signal, group-delay tail included. The concatenation of
    /// every delivered chunk (steady full blocks plus the final short tail)
    /// equals a bare resampler fed the whole input then flushed once, bit for
    /// bit: no sample lost, reordered, requantized, or scaled. The flushed tail
    /// arrives as part of the final chunk(s) through the normal reblock (full
    /// blocks then one final partial), not a separate post-close emission.
    /// Bit-equality also proves the flush ran exactly once: a missing flush would
    /// drop the tail and a double flush would append extra samples, either of
    /// which breaks the equality.
    #[test]
    fn test_resample_close_delivers_full_signal_with_flushed_tail() {
        use decibri_resampler::{PolyphaseResampler, Resampler};

        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();

        // Ground truth: a bare resampler fed the whole input, then one flush.
        let mut reference = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut expected = Vec::new();
        reference.process(&input, &mut expected);
        reference.flush(&mut expected);

        // The process-only count (no flush) shows the tail is a real contribution
        // that the resample path dropped before this drain existed.
        let mut process_only = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut process_out = Vec::new();
        process_only.process(&input, &mut process_out);

        let stage = build_capture_stage(1, 1, 48_000, 16_000, &EnhancementConfig::default())
            .unwrap()
            .expect("48k mono -> resample chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);
        sender.send(make_native_chunk(input)).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320; // 20 ms at 16 kHz
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.sample_rate, 16_000, "chunks report the target rate");
                    assert_eq!(c.channels, 1, "resampled output is mono");
                    blocks.push(c.data);
                }
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Delivered as full blocks then one final partial: the tail rides the
        // normal reblock, not a separate emission.
        let (last, full) = blocks.split_last().expect("at least one block delivered");
        for block in full {
            assert_eq!(
                block.len(),
                samples,
                "non-final blocks are exactly `samples` long, tail included"
            );
        }
        assert!(
            !last.is_empty() && last.len() <= samples,
            "the final chunk carries 1..=`samples` resampled samples"
        );

        // No sample dropped, none added: the delivered stream equals the full
        // resampled signal (steady output plus the flushed group-delay tail).
        let delivered: Vec<f32> = blocks.into_iter().flatten().collect();
        assert_eq!(
            delivered, expected,
            "the resample close path delivers the complete resampled signal, tail included"
        );
        assert!(
            expected.len() > process_out.len(),
            "the flushed tail adds the samples the process-only path dropped"
        );

        // Idempotent close: further reads stay closed with no resurrected audio.
        for _ in 0..3 {
            let err = stream
                .next_chunk(samples, Some(Duration::from_millis(20)))
                .unwrap_err();
            assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
        }
    }

    /// Enhancement-on, end to end through `ingest` (the `Some([transform])`
    /// path): with `dc_removal` enabled on a mono device already at the target
    /// rate, the chain is transform-only (no downmix, no resample), so a constant
    /// DC offset fed through the stream is delivered with the same sample count
    /// (the DC step preserves length and holds no tail) and a settled mean near
    /// zero (the offset is removed). This proves the transform segment is applied
    /// to delivered audio through the normal capture path.
    #[test]
    fn test_enhancement_on_removes_dc_end_to_end() {
        let enhancement = EnhancementConfig { dc_removal: true };
        let stage = build_capture_stage(1, 1, 16_000, 16_000, &enhancement)
            .unwrap()
            .expect("dc_removal builds a transform-only chain for a mono device");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        // A constant 0.5 offset: pure DC, no audio content.
        let n = 16_000;
        let input = vec![0.5_f32; n];
        sender.send(make_native_chunk(input)).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320;
        let mut collected: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.channels, 1, "the enhanced output stays mono");
                    assert_eq!(c.sample_rate, 16_000, "no resample, the rate is unchanged");
                    collected.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        assert_eq!(
            collected.len(),
            n,
            "the DC step preserves the sample count exactly (no resample, no tail)"
        );
        // After the filter settles, the constant offset is gone: the mean of the
        // last quarter is essentially zero.
        let settled = &collected[n - n / 4..];
        let mean = settled.iter().sum::<f32>() / settled.len() as f32;
        assert!(
            mean.abs() < 1e-3,
            "the DC offset is removed end to end (settled mean {mean})"
        );
    }

    /// VAD tap inactive (enhancement off): a chain with no transform (here a
    /// downmix-only chain) leaves the tap unallocated, so `vad_input` is `None`
    /// throughout and a binding feeds VAD the delivered chunk exactly as before.
    #[test]
    fn test_vad_input_none_when_no_transform() {
        let stage = build_capture_stage(2, 1, 16_000, 16_000, &EnhancementConfig::default())
            .unwrap()
            .expect("stereo -> downmix-only chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);
        assert!(
            stream.vad_input(4).is_none(),
            "no transform: the tap is inactive and vad_input returns None"
        );

        sender
            .send(make_native_chunk(vec![0.5, 0.3, 0.4, 0.6]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);
        loop {
            match stream.next_chunk(2, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => assert!(
                    stream.vad_input(c.data.len()).is_none(),
                    "vad_input stays None throughout when no transform is present"
                ),
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
    }

    /// A mono device builds no chain at all, so the tap is inactive and
    /// `vad_input` is `None` (the binding uses the delivered chunk).
    #[test]
    fn test_vad_input_none_for_no_chain() {
        let (stream, _sender, _running) = test_stream();
        assert!(stream.capture_stage.is_none(), "a mono stream has no chain");
        assert!(
            stream.vad_input(4).is_none(),
            "no chain: vad_input returns None"
        );
    }

    /// Enhancement-on tap correctness: with DC removal active, `vad_input` returns
    /// the post-normalize (pre-DC-removal) signal, NOT the post-transform delivered
    /// output. A constant DC offset proves it: the delivered output has the offset
    /// removed, but the VAD feed still carries it, so VAD reads the pre-transform
    /// signal. The tap and the delivered output stay aligned (same sample count).
    #[test]
    fn test_vad_input_returns_pre_transform_signal() {
        let enhancement = EnhancementConfig { dc_removal: true };
        let stage = build_capture_stage(1, 1, 16_000, 16_000, &enhancement)
            .unwrap()
            .expect("dc-only chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        let n = 16_000;
        let input = vec![0.5_f32; n];
        sender.send(make_native_chunk(input)).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320;
        let mut delivered: Vec<f32> = Vec::new();
        let mut vad_feed: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    let pre = stream
                        .vad_input(c.data.len())
                        .expect("tap active when a transform is present");
                    vad_feed.extend_from_slice(&pre);
                    delivered.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        assert_eq!(delivered.len(), n, "no resample, length preserved");
        assert_eq!(
            vad_feed.len(),
            n,
            "the VAD feed is aligned with the delivered output"
        );
        // The VAD feed carries the DC offset (pre-transform).
        assert!(
            vad_feed.iter().all(|&s| (s - 0.5).abs() < 1e-6),
            "the VAD feed is the exact post-normalize (pre-DC) signal, offset intact"
        );
        // The delivered output has the offset removed (post-transform).
        let settled = &delivered[n - n / 4..];
        let d_mean = settled.iter().sum::<f32>() / settled.len() as f32;
        assert!(
            d_mean.abs() < 1e-3,
            "the delivered output has the DC offset removed (post-transform)"
        );
    }

    /// The tap stays aligned with the delivered output through the resampler's
    /// flushed group-delay tail at close: the VAD feed equals the post-normalize
    /// signal (resample process + flush) bit for bit and matches the delivered
    /// count, so the close path does not desync the tap.
    #[test]
    fn test_vad_input_aligned_through_resampler_flush_tail() {
        use decibri_resampler::{PolyphaseResampler, Resampler};

        let enhancement = EnhancementConfig { dc_removal: true };
        let stage = build_capture_stage(1, 1, 48_000, 16_000, &enhancement)
            .unwrap()
            .expect("resample + DC chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        let input: Vec<f32> = (0..24_000).map(|k| (k as f32 * 0.01).sin() + 0.5).collect();
        sender.send(make_native_chunk(input.clone())).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320;
        let mut delivered: Vec<f32> = Vec::new();
        let mut vad_feed: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    let pre = stream.vad_input(c.data.len()).expect("tap active");
                    vad_feed.extend_from_slice(&pre);
                    delivered.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Ground truth post-normalize signal: the resampler over the whole input,
        // process then flush (no DC removal).
        let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut expected_norm = Vec::new();
        resampler.process(&input, &mut expected_norm);
        resampler.flush(&mut expected_norm);

        assert_eq!(
            vad_feed, expected_norm,
            "the VAD feed is the post-normalize signal incl. the resampler flush tail"
        );
        assert_eq!(
            vad_feed.len(),
            delivered.len(),
            "tap and delivered stay aligned through the flushed tail"
        );
        assert_ne!(
            vad_feed, delivered,
            "the delivered output is post-transform (DC removed), so it differs from the feed"
        );
    }

    /// Compile-time assertion that `Arc<Mutex<MicrophoneStream>>` is `Send + Sync`,
    /// which requires `MicrophoneStream: Send`. This is the wrapping strategy the
    /// bindings document for consumers needing shared access from multiple
    /// threads.
    ///
    /// If `MicrophoneStream` ever becomes `!Send` (for example by adding an `Rc<_>`
    /// or `RefCell<_>` field), this test fails to compile, catching the
    /// regression at build time rather than at a binding wrap call site.
    #[test]
    fn test_arc_mutex_microphone_stream_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<std::sync::Mutex<MicrophoneStream>>>();
    }

    /// `Arc<Mutex<MicrophoneStream>>` exercised from two threads racing for
    /// the lock: each thread reads one injected chunk, the Mutex
    /// serializes access, and together they consume exactly the chunks
    /// injected (no duplicates, no losses, no deadlock).
    ///
    /// `Barrier` forces both threads to attempt the lock acquisition
    /// simultaneously so the test exercises contention rather than
    /// sequential non-overlapping access.
    #[test]
    fn test_arc_mutex_microphone_stream_serializes_two_threads() {
        use std::sync::{Barrier, Mutex};

        let (stream, sender, _running) = test_stream();
        sender.send(make_chunk(1.0)).unwrap();
        sender.send(make_chunk(2.0)).unwrap();

        let shared = Arc::new(Mutex::new(stream));
        let barrier = Arc::new(Barrier::new(2));

        let s1 = shared.clone();
        let b1 = barrier.clone();
        let t1 = thread::spawn(move || {
            b1.wait();
            let guard = s1.lock().unwrap();
            guard.try_next_chunk(1).unwrap()
        });

        let s2 = shared.clone();
        let b2 = barrier.clone();
        let t2 = thread::spawn(move || {
            b2.wait();
            let guard = s2.lock().unwrap();
            guard.try_next_chunk(1).unwrap()
        });

        let c1 = t1.join().unwrap().expect("thread 1 must receive a chunk");
        let c2 = t2.join().unwrap().expect("thread 2 must receive a chunk");

        let mut vals = [c1.data[0], c2.data[0]];
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            vals,
            [1.0, 2.0],
            "both chunks must be consumed exactly once with no duplicates or losses"
        );
    }

    /// Compile-time guard that `MicrophoneStream` is `Send + Sync`. The
    /// `_stream` field is a `Mutex` to keep this bound; a future change to
    /// `Cell`/`RefCell` would make the type `!Sync` and fail this test in the
    /// core crate, rather than only surfacing as a `decibri-python` build break
    /// (pyo3's `#[pyclass]` and `py.detach` both require `Sync`).
    #[test]
    fn test_microphone_stream_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MicrophoneStream>();
    }

    /// `stop()` takes the held stream out from under the mutex (releasing the
    /// device in production) and is idempotent. The test seam holds no real
    /// device, so this asserts the slot is empty after stop and that a second
    /// stop does not panic; the real `Some -> None` drop is exercised by the
    /// binding suites and integration capture.
    #[test]
    fn test_stop_empties_stream_and_is_idempotent() {
        let (stream, _sender, _running) = test_stream();
        stream.stop();
        assert!(!stream.is_open(), "stop() clears the running flag");
        assert!(
            !stream._stream.is_active(),
            "stop() leaves the stream slot empty (device released)"
        );
        stream.stop(); // idempotent: an already-empty slot must not panic
    }
}
