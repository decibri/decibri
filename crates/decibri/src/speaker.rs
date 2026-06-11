//! Speaker playback: open an output device and push `f32` samples to it.
//!
//! Build a [`SpeakerConfig`], construct a [`Speaker`], then
//! [`start`](Speaker::start) it to obtain a [`SpeakerStream`]. Push audio with
//! [`send`](SpeakerStream::send), block until the queue empties with
//! [`drain`](SpeakerStream::drain), or end immediately with
//! [`stop`](SpeakerStream::stop). The capture counterpart is
//! [`crate::microphone`].

#[cfg(feature = "playback")]
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
#[cfg(feature = "playback")]
use std::sync::{Arc, Mutex};

#[cfg(feature = "playback")]
use cpal::traits::{DeviceTrait, StreamTrait};
#[cfg(feature = "playback")]
use crossbeam_channel::{Receiver, Sender};

use crate::device::DeviceSelector;
use crate::error::DecibriError;

/// Configuration for audio output.
#[derive(Debug, Clone)]
pub struct SpeakerConfig {
    /// Sample rate in Hz. Range: 1000–384000. Default: 16000.
    pub sample_rate: u32,
    /// Number of output channels. Range: 1–32. Default: 1.
    pub channels: u16,
    /// Device selection. Default: system default output.
    pub device: DeviceSelector,
}

impl Default for SpeakerConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            device: DeviceSelector::Default,
        }
    }
}

impl SpeakerConfig {
    /// Validate the configuration: sample rate and channel count must fall
    /// within the supported ranges.
    pub fn validate(&self) -> Result<(), DecibriError> {
        if !(1000..=384000).contains(&self.sample_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
        }
        if !(1..=32).contains(&self.channels) {
            return Err(DecibriError::ChannelsOutOfRange);
        }
        Ok(())
    }
}

/// Fill one cpal output buffer from the playback queue.
///
/// Extracted from the output callback so the playback logic can be unit-tested
/// without a cpal device. When the stream is stopped (`running` is false) it
/// discards anything still queued and emits silence, counting any drain
/// sentinels it drops so a blocked [`SpeakerStream::drain`] still returns.
/// Otherwise it plays the accumulator and then the channel, stashing any
/// overflow back into `accum`, and bumps `sentinels_played` for each drain
/// sentinel (an empty vec) it consumes.
#[cfg(feature = "playback")]
fn render_output(
    data: &mut [f32],
    accum: &mut Vec<f32>,
    receiver: &Receiver<Vec<f32>>,
    running: &AtomicBool,
    sentinels_played: &AtomicUsize,
) {
    if !running.load(Ordering::Relaxed) {
        // Stopped: drop everything queued and play silence. Count discarded
        // sentinels so a concurrently blocked drain() is released.
        accum.clear();
        while let Ok(chunk) = receiver.try_recv() {
            if chunk.is_empty() {
                sentinels_played.fetch_add(1, Ordering::Relaxed);
            }
        }
        data.fill(0.0);
        return;
    }

    let mut written = 0;

    // First, play any leftover from the accumulator.
    if !accum.is_empty() {
        let take = accum.len().min(data.len());
        data[..take].copy_from_slice(&accum[..take]);
        accum.drain(..take);
        written += take;
    }

    // Then pull from the channel until the buffer is full.
    while written < data.len() {
        match receiver.try_recv() {
            Ok(samples) => {
                if samples.is_empty() {
                    // Drain sentinel: a batch boundary. Record it so the matching
                    // drain() returns, then keep filling from any audio queued
                    // after it so playback stays gapless across drains.
                    sentinels_played.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let need = data.len() - written;
                if samples.len() <= need {
                    data[written..written + samples.len()].copy_from_slice(&samples);
                    written += samples.len();
                } else {
                    // More samples than fit: fill the buffer, stash the rest.
                    data[written..].copy_from_slice(&samples[..need]);
                    accum.extend_from_slice(&samples[need..]);
                    written = data.len();
                }
            }
            Err(_) => {
                // Nothing queued: fill the remainder with silence.
                data[written..].fill(0.0);
                return;
            }
        }
    }
}

/// Queue samples for playback. Shared by [`SpeakerStream::send`] and
/// [`SpeakerSink::send`]. An empty buffer is a no-op. A stopped stream (or a
/// disconnected channel) returns [`DecibriError::SpeakerStreamClosed`].
#[cfg(feature = "playback")]
fn send_impl(
    sender: &Sender<Vec<f32>>,
    running: &AtomicBool,
    samples: Vec<f32>,
) -> Result<(), DecibriError> {
    if samples.is_empty() {
        return Ok(());
    }
    if !running.load(Ordering::Relaxed) {
        return Err(DecibriError::SpeakerStreamClosed);
    }
    sender
        .send(samples)
        .map_err(|_| DecibriError::SpeakerStreamClosed)
}

/// Block until the audio queued before this call has played. Shared by
/// [`SpeakerStream::drain`] and [`SpeakerSink::drain`]. Reserves the next
/// sentinel ticket, sends the sentinel, then waits until the output callback has
/// consumed that many sentinels. Repeatable and non-terminal: it never ends the
/// stream. Returns early if the stream has been stopped or its channel closed.
#[cfg(feature = "playback")]
fn drain_impl(
    sender: &Sender<Vec<f32>>,
    running: &AtomicBool,
    sentinel_seq: &AtomicUsize,
    sentinels_played: &AtomicUsize,
) {
    let target = sentinel_seq.fetch_add(1, Ordering::Relaxed) + 1;
    if sender.send(Vec::new()).is_err() {
        // Channel closed: nothing remains to play.
        return;
    }
    while sentinels_played.load(Ordering::Relaxed) < target {
        if !running.load(Ordering::Relaxed) {
            // Stopped: queued audio is discarded, so there is nothing to await.
            return;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}

/// An open playback stream you push `f32` samples to.
///
/// Obtained from [`Speaker::start`]. Queue samples with [`send`](Self::send),
/// which applies backpressure when the internal buffer is full. Call
/// [`drain`](Self::drain) to block until everything queued has played, or
/// [`stop`](Self::stop) to end immediately, discard what remains, and release
/// the device. Dropping the stream also releases the device. The type is
/// `Send + Sync`.
#[cfg(feature = "playback")]
pub struct SpeakerStream {
    // `Mutex<Option<...>>` (not `Cell`/`RefCell`) so `stop(&self)` can `take()`
    // and drop the stream under `&self`, releasing the device, while keeping
    // this type `Send + Sync`. The Python binding requires `Sync`: its
    // `#[pyclass]` (`assert_pyclass_send_sync`) and `py.detach` both need the
    // stream type to be `Sync`, and `cpal::Stream` is itself `Send + Sync` on
    // every backend. `Cell`/`RefCell` are `!Sync` and would break
    // `decibri-python`. The lock is uncontended: only `stop()` and `drop` ever
    // touch this field; the cpal callback owns the channel receiver, not this.
    // The `Option` also lets the test seam build a stream with no real device
    // (`Mutex::new(None)`); production stores `Mutex::new(Some(stream))`.
    _stream: Mutex<Option<cpal::Stream>>,
    sender: Sender<Vec<f32>>,
    running: Arc<AtomicBool>,
    // Drain handshake. `drain()` reserves the next `sentinel_seq` ticket and
    // sends an empty-vec sentinel; the output callback bumps `sentinels_played`
    // each time it consumes one. A drain returns once `sentinels_played` reaches
    // its ticket, so sequential drains each wait for their own audio.
    sentinel_seq: Arc<AtomicUsize>,
    sentinels_played: Arc<AtomicUsize>,
    // Last device/driver error reported by the cpal output error callback
    // while streaming, retrievable via [`take_last_error`](Self::take_last_error).
    // Lets a consumer that sees a closed stream distinguish a driver failure
    // from an explicit `stop()`. Written at most once, on failure.
    last_error: Arc<Mutex<Option<DecibriError>>>,
}

#[cfg(feature = "playback")]
impl SpeakerStream {
    /// Send f32 samples for playback.
    ///
    /// Samples are queued into an internal bounded channel feeding the cpal
    /// output callback. If the channel is full, this method **blocks** the
    /// calling thread until the callback consumes enough samples to make
    /// room, so that backpressure propagates naturally to the producer.
    ///
    /// Empty `samples` vectors are treated as a no-op and return `Ok(())`
    /// without touching the channel.
    ///
    /// # Returns
    /// - `Ok(())`: samples accepted into the playback queue (or buffer was
    ///   empty).
    /// - `Err(DecibriError::SpeakerStreamClosed)`: the output stream has
    ///   been stopped or dropped; no further samples can be played.
    ///
    /// # Thread safety
    /// `&self`-taking: may be called concurrently from multiple threads, but
    /// the interleaving of their samples in the playback stream is not
    /// defined. For deterministic output, serialize calls from a single
    /// producer thread.
    ///
    /// # Stability
    /// Part of decibri's stable FFI-consumer API surface. Signature and
    /// three-state semantics (ok / empty-noop / closed) are guaranteed
    /// stable across 4.x; any change will be a breaking version bump.
    pub fn send(&self, samples: Vec<f32>) -> Result<(), DecibriError> {
        send_impl(&self.sender, &self.running, samples)
    }

    /// Block until everything queued at call time has played. Repeatable: the
    /// stream stays open afterward, so further [`send`](Self::send) calls keep
    /// working and a later `drain` waits for its own audio. Does not end the
    /// stream. Returns early if the stream has been stopped.
    pub fn drain(&self) {
        drain_impl(
            &self.sender,
            &self.running,
            &self.sentinel_seq,
            &self.sentinels_played,
        );
    }

    /// Check whether the stream is still actively playing.
    pub fn is_playing(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Take the last device/driver error reported while streaming, if any.
    ///
    /// When the cpal output error callback fires (device unplug, driver
    /// failure) it records a typed [`DecibriError::DeviceFailed`] and stops
    /// playback. Reading it here drains it (returns `None` afterwards), letting
    /// a consumer that observes a closed stream tell a driver failure apart
    /// from an explicit [`stop`](Self::stop).
    pub fn take_last_error(&self) -> Option<DecibriError> {
        self.last_error.lock().ok().and_then(|mut slot| slot.take())
    }

    /// Immediate stop, releasing the device. The output goes silent at once and
    /// all queued audio is discarded; [`is_playing`](Self::is_playing) is false
    /// immediately and later non-empty [`send`](Self::send) calls (including via
    /// a surviving [`SpeakerSink`]) return [`DecibriError::SpeakerStreamClosed`].
    /// The held `cpal::Stream` is dropped here, so the OS releases the output
    /// device on stop rather than only when the handle is dropped; this blocks
    /// briefly while the audio thread tears down.
    pub fn stop(&self) {
        // Flag first: a surviving sink's send/drain key off this and short-circuit.
        self.running.store(false, Ordering::Relaxed);
        // Then drop the cpal stream to release the OS device now, not only on
        // drop. Poison-tolerant: if a teardown elsewhere panicked while holding
        // the lock, still clear the slot rather than panicking in stop().
        if let Ok(mut guard) = self._stream.lock() {
            *guard = None;
        }
    }

    /// A cloneable, `Send` handle to this stream's sample channel and drain
    /// state. Use it to push samples or wait for drain from another thread (for
    /// example a worker pool) without holding the `SpeakerStream` itself. Because
    /// the handle clones only the lock-free channel sender and atomic flags, its
    /// off-thread `send` and `drain` never block a holder of the stream that
    /// needs `stop` or `is_playing`.
    ///
    /// The handle shares the same crossbeam channel and atomic drain flags as
    /// the stream, so [`SpeakerSink::send`] and [`SpeakerSink::drain`] behave
    /// exactly like their [`SpeakerStream`] counterparts. The stream must stay
    /// alive while a sink is in use: dropping the `SpeakerStream` releases the
    /// device, after which a `send` on a surviving sink returns
    /// [`DecibriError::SpeakerStreamClosed`] rather than playing.
    pub fn sink(&self) -> SpeakerSink {
        SpeakerSink {
            sender: self.sender.clone(),
            running: self.running.clone(),
            sentinel_seq: self.sentinel_seq.clone(),
            sentinels_played: self.sentinels_played.clone(),
        }
    }
}

/// A cloneable, `Send` handle to a running [`SpeakerStream`]'s sample channel
/// and drain state.
///
/// Obtained from [`SpeakerStream::sink`]. The `SpeakerStream` itself is `Send`,
/// so it can be held wherever its owner likes; a `SpeakerSink` is a cheaper,
/// lock-free companion that clones only the thread-safe pieces (the crossbeam
/// channel sender and the atomic flags) and is `Send + Sync + Clone`. Several
/// threads can push samples or wait for drain through sinks at once, and because
/// a sink never touches the audio stream handle, its `send` and `drain` cannot
/// block whoever holds the `SpeakerStream` and needs `stop` or `is_playing`.
#[cfg(feature = "playback")]
#[derive(Clone)]
pub struct SpeakerSink {
    sender: Sender<Vec<f32>>,
    running: Arc<AtomicBool>,
    sentinel_seq: Arc<AtomicUsize>,
    sentinels_played: Arc<AtomicUsize>,
}

#[cfg(feature = "playback")]
impl SpeakerSink {
    /// Send `f32` samples for playback. Identical semantics to
    /// [`SpeakerStream::send`]: blocks when the internal buffer is full
    /// (backpressure), treats an empty vector as a no-op, and returns
    /// [`DecibriError::SpeakerStreamClosed`] once the stream has stopped.
    pub fn send(&self, samples: Vec<f32>) -> Result<(), DecibriError> {
        send_impl(&self.sender, &self.running, samples)
    }

    /// Block until everything queued at call time has played. Identical
    /// semantics to [`SpeakerStream::drain`]: repeatable, leaves the stream
    /// open, and returns early if the stream has been stopped.
    pub fn drain(&self) {
        drain_impl(
            &self.sender,
            &self.running,
            &self.sentinel_seq,
            &self.sentinels_played,
        );
    }
}

/// An output device you play audio to.
///
/// Build one from a [`SpeakerConfig`], then [`start`](Self::start) it to open
/// the device and obtain a [`SpeakerStream`] you push `f32` samples to. The
/// capture counterpart is [`Microphone`](crate::Microphone).
///
/// ```no_run
/// use decibri::{Speaker, SpeakerConfig};
///
/// let speaker = Speaker::new(SpeakerConfig::default())?;
/// let stream = speaker.start()?;
/// stream.send(vec![0.0_f32; 16_000])?; // one second of silence at 16 kHz mono
/// stream.drain();
/// # Ok::<(), decibri::DecibriError>(())
/// ```
#[cfg(feature = "playback")]
pub struct Speaker {
    config: SpeakerConfig,
    device: cpal::Device,
}

#[cfg(feature = "playback")]
impl Speaker {
    /// Create a speaker: validates the [`SpeakerConfig`] and resolves the
    /// selected output device. Does not open the stream; call
    /// [`start`](Self::start) for that.
    pub fn new(config: SpeakerConfig) -> Result<Self, DecibriError> {
        config.validate()?;
        let device = crate::device::resolve_output_device(&config.device)?;
        Ok(Self { config, device })
    }

    /// List the available output devices.
    pub fn devices() -> Result<Vec<crate::device::SpeakerInfo>, DecibriError> {
        crate::device::output_devices()
    }

    /// Start the output stream. Returns a handle for sending samples.
    pub fn start(&self) -> Result<SpeakerStream, DecibriError> {
        let (sender, receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) =
            crossbeam_channel::bounded(32);

        let running = Arc::new(AtomicBool::new(true));
        let sentinel_seq = Arc::new(AtomicUsize::new(0));
        let sentinels_played = Arc::new(AtomicUsize::new(0));
        let running_cb = running.clone();
        let sentinels_played_cb = sentinels_played.clone();
        let running_clone = running.clone();
        let last_error = Arc::new(Mutex::new(None));
        let err_last_error = last_error.clone();

        let sample_rate = self.config.sample_rate;
        let channels = self.config.channels;

        let stream_config = cpal::StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Default,
        };

        // Internal accumulator for the cpal callback.
        // cpal requests variable-sized buffers; we feed from the channel.
        let mut accum: Vec<f32> = Vec::new();

        let stream = self
            .device
            .build_output_stream(
                &stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    render_output(
                        data,
                        &mut accum,
                        &receiver,
                        &running_cb,
                        &sentinels_played_cb,
                    );
                },
                move |err| {
                    // Record a typed cause so a consumer that sees the stream
                    // close can distinguish a driver failure from stop().
                    let typed = DecibriError::DeviceFailed(err.to_string());
                    eprintln!("{typed}");
                    if let Ok(mut slot) = err_last_error.lock() {
                        *slot = Some(typed);
                    }
                    running_clone.store(false, Ordering::Relaxed);
                },
                None,
            )
            .map_err(|e| DecibriError::StreamOpenFailed(e.to_string()))?;

        stream
            .play()
            .map_err(|e| DecibriError::StreamStartFailed(e.to_string()))?;

        Ok(SpeakerStream {
            _stream: Mutex::new(Some(stream)),
            sender,
            running,
            sentinel_seq,
            sentinels_played,
            last_error,
        })
    }
}

#[cfg(all(test, feature = "playback"))]
mod tests {
    use super::*;
    use std::thread;
    use std::time::{Duration, Instant};

    /// Build a synthetic `SpeakerStream` with no cpal device, returning the
    /// stream plus the receiving end of its channel and a clone of the
    /// played-sentinel counter, so a test can drive the output callback by hand.
    fn test_stream() -> (SpeakerStream, Receiver<Vec<f32>>, Arc<AtomicUsize>) {
        let (sender, receiver) = crossbeam_channel::bounded::<Vec<f32>>(32);
        let sentinels_played = Arc::new(AtomicUsize::new(0));
        let stream = SpeakerStream {
            _stream: Mutex::new(None),
            sender,
            running: Arc::new(AtomicBool::new(true)),
            sentinel_seq: Arc::new(AtomicUsize::new(0)),
            sentinels_played: sentinels_played.clone(),
            last_error: Arc::new(Mutex::new(None)),
        };
        (stream, receiver, sentinels_played)
    }

    /// Run a single output-callback cycle against the synthetic stream and
    /// return the buffer it produced.
    fn render_once(
        stream: &SpeakerStream,
        accum: &mut Vec<f32>,
        receiver: &Receiver<Vec<f32>>,
        len: usize,
    ) -> Vec<f32> {
        let mut data = vec![0.0_f32; len];
        render_output(
            &mut data,
            accum,
            receiver,
            &stream.running,
            &stream.sentinels_played,
        );
        data
    }

    /// Render callback cycles until `sentinels_played` reaches `target`, with a
    /// generous ceiling so a logic error fails the test instead of hanging it.
    fn consume_until(
        stream: &SpeakerStream,
        accum: &mut Vec<f32>,
        receiver: &Receiver<Vec<f32>>,
        played: &AtomicUsize,
        target: usize,
    ) {
        let mut data = vec![0.0_f32; 8];
        let deadline = Instant::now() + Duration::from_secs(2);
        while played.load(Ordering::Relaxed) < target {
            render_output(
                &mut data,
                accum,
                receiver,
                &stream.running,
                &stream.sentinels_played,
            );
            assert!(
                Instant::now() <= deadline,
                "timed out waiting for {target} sentinel(s)"
            );
            thread::sleep(Duration::from_millis(1));
        }
    }

    #[test]
    fn test_render_plays_queued_audio_then_silence() {
        let (stream, receiver, _played) = test_stream();
        stream.send(vec![0.5_f32, 0.5, 0.5]).unwrap();

        let mut accum = Vec::new();
        let data = render_once(&stream, &mut accum, &receiver, 5);

        assert_eq!(&data[..3], &[0.5, 0.5, 0.5], "queued audio is played");
        assert_eq!(&data[3..], &[0.0, 0.0], "the remainder is silence");
    }

    #[test]
    fn test_render_stashes_overflow_into_accumulator() {
        let (stream, receiver, _played) = test_stream();
        stream.send(vec![1.0_f32; 6]).unwrap();

        let mut accum = Vec::new();
        // Buffer smaller than the queued chunk: 4 play now, 2 are stashed.
        let first = render_once(&stream, &mut accum, &receiver, 4);
        assert_eq!(first, vec![1.0; 4]);
        assert_eq!(
            accum,
            vec![1.0; 2],
            "overflow is stashed for the next cycle"
        );

        // The next cycle plays the stash, then silence.
        let second = render_once(&stream, &mut accum, &receiver, 4);
        assert_eq!(&second[..2], &[1.0, 1.0]);
        assert_eq!(&second[2..], &[0.0, 0.0]);
        assert!(accum.is_empty());
    }

    #[test]
    fn test_stop_discards_queued_audio() {
        let (stream, receiver, _played) = test_stream();
        stream.send(vec![0.5_f32; 50]).unwrap();
        stream.send(vec![0.25_f32; 50]).unwrap();
        stream.stop();

        let mut accum = Vec::new();
        let data = render_once(&stream, &mut accum, &receiver, 64);

        assert!(
            data.iter().all(|&s| s == 0.0),
            "a stopped stream must emit only silence"
        );
        assert!(
            receiver.try_recv().is_err(),
            "stop() must discard queued audio, leaving the channel empty"
        );
        assert!(accum.is_empty(), "the accumulator must be cleared on stop");
    }

    #[test]
    fn test_stop_discards_when_channel_full() {
        let (stream, receiver, _played) = test_stream();
        // Fill the bounded(32) channel to capacity.
        for _ in 0..32 {
            stream.send(vec![1.0_f32; 8]).unwrap();
        }
        stream.stop();

        let mut accum = Vec::new();
        let data = render_once(&stream, &mut accum, &receiver, 16);

        assert!(
            data.iter().all(|&s| s == 0.0),
            "a full channel must still go silent after stop (regression: stop() \
             used to no-op when the channel was full)"
        );
        assert!(
            receiver.try_recv().is_err(),
            "a full channel must be fully discarded after stop"
        );
    }

    #[test]
    fn test_is_playing_false_immediately_after_stop() {
        let (stream, _receiver, _played) = test_stream();
        assert!(stream.is_playing(), "a fresh stream reports playing");
        stream.stop();
        assert!(
            !stream.is_playing(),
            "is_playing() must be false the instant stop() returns"
        );
    }

    #[test]
    fn test_send_after_stop_returns_closed() {
        let (stream, _receiver, _played) = test_stream();
        stream.stop();

        let err = stream.send(vec![1.0_f32; 4]).unwrap_err();
        assert!(
            matches!(err, DecibriError::SpeakerStreamClosed),
            "a non-empty send() after stop() must fail with SpeakerStreamClosed"
        );
        assert!(
            stream.send(Vec::new()).is_ok(),
            "an empty send is a documented no-op even after stop"
        );
    }

    #[test]
    fn test_second_drain_waits_for_its_own_audio() {
        let (stream, receiver, played) = test_stream();
        let mut accum = Vec::new();

        // Round 1: queue audio, drain on a worker, consume here.
        stream.send(vec![1.0_f32; 4]).unwrap();
        let sink1 = stream.sink();
        let h1 = thread::spawn(move || sink1.drain());
        consume_until(&stream, &mut accum, &receiver, &played, 1);
        h1.join().unwrap();

        // Round 2: with the old one-shot flag this drain returned instantly.
        // Prove it now blocks until round-2 audio is actually consumed.
        stream.send(vec![2.0_f32; 4]).unwrap();
        let sink2 = stream.sink();
        let done = Arc::new(AtomicBool::new(false));
        let done_writer = done.clone();
        let h2 = thread::spawn(move || {
            sink2.drain();
            done_writer.store(true, Ordering::Relaxed);
        });

        // Give the second drain a chance to (wrongly) finish early.
        thread::sleep(Duration::from_millis(60));
        assert!(
            !done.load(Ordering::Relaxed),
            "second drain must not return before its audio is consumed \
             (stale-sentinel regression)"
        );

        consume_until(&stream, &mut accum, &receiver, &played, 2);
        h2.join().unwrap();
        assert!(
            done.load(Ordering::Relaxed),
            "second drain returns once its audio is consumed"
        );
    }

    #[test]
    fn test_send_after_drain_succeeds_and_stream_stays_open() {
        let (stream, receiver, played) = test_stream();
        let mut accum = Vec::new();

        stream.send(vec![1.0_f32; 4]).unwrap();
        let sink = stream.sink();
        let h = thread::spawn(move || sink.drain());
        consume_until(&stream, &mut accum, &receiver, &played, 1);
        h.join().unwrap();

        assert!(stream.is_playing(), "drain() must not end the stream");
        assert!(
            stream.send(vec![2.0_f32; 4]).is_ok(),
            "send() after drain() must still succeed"
        );
    }

    #[test]
    fn test_sink_drain_is_repeatable() {
        let (stream, receiver, played) = test_stream();
        let mut accum = Vec::new();

        for round in 1..=2usize {
            let sink = stream.sink();
            sink.send(vec![round as f32; 4]).unwrap();
            let h = thread::spawn(move || sink.drain());
            consume_until(&stream, &mut accum, &receiver, &played, round);
            h.join().unwrap();
        }
        assert_eq!(
            played.load(Ordering::Relaxed),
            2,
            "each drain through the sink waits for its own audio"
        );
    }

    /// Compile-time guard that `SpeakerStream` is `Send + Sync`. The `_stream`
    /// field is a `Mutex` to keep this bound; a future change to `Cell`/`RefCell`
    /// would make the type `!Sync` and fail this test in the core crate, rather
    /// than only surfacing as a `decibri-python` build break (pyo3's
    /// `#[pyclass]` and `py.detach` both require `Sync`).
    #[test]
    fn test_speaker_stream_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SpeakerStream>();
    }

    /// `stop()` takes the held stream out from under the mutex (releasing the
    /// device in production) and is idempotent. The test seam holds no real
    /// device, so this asserts the slot is empty after stop and that a second
    /// stop does not panic; the real `Some -> None` drop is exercised by the
    /// binding suites and integration playback.
    #[test]
    fn test_stop_empties_stream_and_is_idempotent() {
        let (stream, _receiver, _played) = test_stream();
        stream.stop();
        assert!(!stream.is_playing(), "stop() clears the running flag");
        assert!(
            stream._stream.lock().unwrap().is_none(),
            "stop() leaves the stream slot empty (device released)"
        );
        stream.stop(); // idempotent: an already-empty slot must not panic
    }
}
