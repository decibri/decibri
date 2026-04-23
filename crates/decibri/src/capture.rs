#[cfg(feature = "capture")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "capture")]
use std::sync::Arc;
#[cfg(feature = "capture")]
use std::time::{Duration, Instant};

#[cfg(feature = "capture")]
use cpal::traits::{DeviceTrait, StreamTrait};
#[cfg(feature = "capture")]
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError};

use crate::device::DeviceSelector;
use crate::error::DecibriError;

/// Configuration for audio capture.
#[derive(Debug, Clone)]
pub struct CaptureConfig {
    /// Sample rate in Hz. Range: 1000–384000. Default: 16000.
    pub sample_rate: u32,
    /// Number of input channels. Range: 1–32. Default: 1.
    pub channels: u16,
    /// Frames per audio callback buffer. Range: 64–65536. Default: 1600.
    pub frames_per_buffer: u32,
    /// Device selection. Default: system default input.
    pub device: DeviceSelector,
}

impl Default for CaptureConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            frames_per_buffer: 1600,
            device: DeviceSelector::Default,
        }
    }
}

impl CaptureConfig {
    /// Validate configuration values match the JS API constraints.
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
#[derive(Debug, Clone)]
pub struct AudioChunk {
    /// Interleaved f32 samples in range [-1.0, 1.0].
    pub data: Vec<f32>,
    /// Sample rate of this chunk.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

/// Handle to an active audio capture session.
#[cfg(feature = "capture")]
pub struct CaptureStream {
    /// Holds the cpal stream alive for the lifetime of this handle. Wrapped
    /// in `Option` purely so the test-only `tests::test_stream()` helper can
    /// construct a `CaptureStream` without a real audio device. Production
    /// code always stores `Some(stream)`; drop semantics are identical (the
    /// `Option`'s drop runs the inner `cpal::Stream`'s drop).
    _stream: Option<cpal::Stream>,
    receiver: Receiver<AudioChunk>,
    running: Arc<AtomicBool>,
    #[allow(dead_code)]
    sample_rate: u32,
    #[allow(dead_code)]
    channels: u16,
}

#[cfg(feature = "capture")]
impl CaptureStream {
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
    /// three-state return (`Some` / `None` / `Err(CaptureStreamClosed)`)
    /// that maps cleanly across a language boundary.
    pub fn receiver(&self) -> &Receiver<AudioChunk> {
        &self.receiver
    }

    /// Attempt to read the next audio chunk without blocking.
    ///
    /// # Returns
    /// - `Ok(Some(chunk))` — a chunk was immediately available and has been
    ///   dequeued.
    /// - `Ok(None)` — the stream is open and running, but no chunk is
    ///   currently buffered. Try again shortly, or call
    ///   [`next_chunk`](Self::next_chunk) to block.
    /// - `Err(DecibriError::CaptureStreamClosed)` — the stream is closed
    ///   (either by explicit [`stop`](Self::stop) or by an audio-driver
    ///   error reported via the cpal error callback). No further chunks
    ///   will ever be available.
    ///
    /// Any chunks that were buffered before the stream closed are delivered
    /// first; the closed signal is only returned once the buffer is empty.
    ///
    /// # Thread safety
    /// May be called from any thread. Internally uses a lock-free
    /// `crossbeam_channel::try_recv` and an atomic load.
    ///
    /// # Stability
    /// Part of decibri's stable FFI-consumer API surface, alongside
    /// [`next_chunk`](Self::next_chunk). Guaranteed signature-stable across
    /// 3.x; any change will be a breaking version bump.
    pub fn try_next_chunk(&self) -> Result<Option<AudioChunk>, DecibriError> {
        match self.receiver.try_recv() {
            Ok(chunk) => Ok(Some(chunk)),
            Err(TryRecvError::Empty) => {
                if self.is_open() {
                    Ok(None)
                } else {
                    Err(DecibriError::CaptureStreamClosed)
                }
            }
            Err(TryRecvError::Disconnected) => Err(DecibriError::CaptureStreamClosed),
        }
    }

    /// Read the next audio chunk, blocking the calling thread until one
    /// arrives, the stream closes, or `timeout` elapses.
    ///
    /// # Arguments
    /// - `timeout = None` — block indefinitely until a chunk arrives or the
    ///   stream closes.
    /// - `timeout = Some(dur)` — block at most `dur`; return `Ok(None)` if
    ///   the deadline passes with no chunk.
    ///
    /// # Returns
    /// - `Ok(Some(chunk))` — chunk received within the deadline.
    /// - `Ok(None)` — `timeout` elapsed without a chunk arriving. The stream
    ///   is still open.
    /// - `Err(DecibriError::CaptureStreamClosed)` — stream closed before a
    ///   chunk could be delivered. Any chunks that were buffered at the time
    ///   of close are delivered first; this error is only returned once the
    ///   buffer is empty.
    ///
    /// # Thread safety
    /// May be called from any thread. Blocks only the calling thread; other
    /// threads can call [`stop`](Self::stop) concurrently to unblock this
    /// call within approximately 20 ms.
    ///
    /// Implementation note: `stop()` does not disconnect the underlying
    /// channel (the cpal `Stream` still holds the sender until the
    /// `CaptureStream` is dropped), so this method internally polls both
    /// the channel and [`is_open`](Self::is_open) at a short interval
    /// rather than relying on channel disconnection alone to wake up.
    ///
    /// # Stability
    /// Part of decibri's stable FFI-consumer API surface.
    pub fn next_chunk(
        &self,
        timeout: Option<Duration>,
    ) -> Result<Option<AudioChunk>, DecibriError> {
        // Poll both the channel and `is_open` at this cadence so concurrent
        // `stop()` calls unblock a waiter within one interval. 20 ms is well
        // below a typical audio frame period (100 ms at 16 kHz / 1600
        // frames) so the extra wakeups cost negligible CPU.
        const POLL_INTERVAL: Duration = Duration::from_millis(20);

        let deadline = timeout.map(|t| Instant::now() + t);

        loop {
            let wait = match deadline {
                Some(dl) => {
                    let now = Instant::now();
                    if now >= dl {
                        // Timeout elapsed without a chunk. Drain any
                        // last-moment arrival before reporting None.
                        return Ok(self.receiver.try_recv().ok());
                    }
                    std::cmp::min(dl - now, POLL_INTERVAL)
                }
                None => POLL_INTERVAL,
            };

            match self.receiver.recv_timeout(wait) {
                Ok(chunk) => return Ok(Some(chunk)),
                Err(RecvTimeoutError::Timeout) => {
                    // Re-check whether `stop()` fired while we waited.
                    if !self.is_open() {
                        // Drain any buffered chunk (stop() does not flush).
                        return match self.receiver.try_recv() {
                            Ok(chunk) => Ok(Some(chunk)),
                            Err(_) => Err(DecibriError::CaptureStreamClosed),
                        };
                    }
                    // Stream still alive; loop with whatever deadline remains.
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(DecibriError::CaptureStreamClosed);
                }
            }
        }
    }

    /// Check whether the stream is still actively capturing.
    pub fn is_open(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop capturing audio.
    ///
    /// Signals the capture callback to stop producing chunks. Already-buffered
    /// chunks remain readable via [`try_next_chunk`](Self::try_next_chunk) or
    /// [`next_chunk`](Self::next_chunk) until the buffer is drained, at which
    /// point those methods return `Err(CaptureStreamClosed)`.
    ///
    /// May be called from any thread; wakes a concurrent
    /// [`next_chunk`](Self::next_chunk) waiter within approximately 20 ms.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
    }
}

/// Audio capture engine.
#[cfg(feature = "capture")]
pub struct AudioCapture {
    config: CaptureConfig,
    device: cpal::Device,
}

#[cfg(feature = "capture")]
impl AudioCapture {
    /// Create a new capture instance. Validates config and resolves the device.
    pub fn new(config: CaptureConfig) -> Result<Self, DecibriError> {
        config.validate()?;
        let device = crate::device::resolve_device(&config.device)?;
        Ok(Self { config, device })
    }

    /// Start capturing audio. Returns a stream handle with a receiver for audio chunks.
    pub fn start(&self) -> Result<CaptureStream, DecibriError> {
        let (sender, receiver): (Sender<AudioChunk>, Receiver<AudioChunk>) =
            crossbeam_channel::unbounded();

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        let sample_rate = self.config.sample_rate;
        let channels = self.config.channels;
        let frames_per_buffer = self.config.frames_per_buffer;

        let stream_config = cpal::StreamConfig {
            channels,
            sample_rate,
            buffer_size: cpal::BufferSize::Fixed(frames_per_buffer),
        };

        let err_running = running.clone();

        let stream = self
            .device
            .build_input_stream(
                &stream_config,
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    if !running_clone.load(Ordering::Relaxed) {
                        return;
                    }
                    let chunk = AudioChunk {
                        data: data.to_vec(),
                        sample_rate,
                        channels,
                    };
                    // If the receiver is dropped, just discard.
                    let _ = sender.send(chunk);
                },
                move |err| {
                    eprintln!("decibri: audio stream error: {err}");
                    err_running.store(false, Ordering::Relaxed);
                },
                None, // No timeout
            )
            .map_err(|e| DecibriError::StreamOpenFailed(e.to_string()))?;

        stream
            .play()
            .map_err(|e| DecibriError::StreamStartFailed(e.to_string()))?;

        Ok(CaptureStream {
            _stream: Some(stream),
            receiver,
            running,
            sample_rate,
            channels,
        })
    }
}

#[cfg(all(test, feature = "capture"))]
mod tests {
    use super::*;
    use std::thread;

    /// Construct a synthetic `CaptureStream` with no underlying cpal device
    /// for unit-testing the channel-reading methods. Returns the stream
    /// plus test-side handles to inject chunks and flip the running flag.
    fn test_stream() -> (CaptureStream, Sender<AudioChunk>, Arc<AtomicBool>) {
        let (sender, receiver) = crossbeam_channel::unbounded::<AudioChunk>();
        let running = Arc::new(AtomicBool::new(true));
        let stream = CaptureStream {
            _stream: None,
            receiver,
            running: running.clone(),
            sample_rate: 16000,
            channels: 1,
        };
        (stream, sender, running)
    }

    fn make_chunk(first_sample: f32) -> AudioChunk {
        AudioChunk {
            data: vec![first_sample],
            sample_rate: 16000,
            channels: 1,
        }
    }

    #[test]
    fn test_try_next_chunk_returns_none_when_empty() {
        let (stream, _sender, _running) = test_stream();
        let result = stream.try_next_chunk().unwrap();
        assert!(
            result.is_none(),
            "try_next_chunk on empty open stream should return Ok(None)"
        );
    }

    #[test]
    fn test_try_next_chunk_returns_chunk_when_available() {
        let (stream, sender, _running) = test_stream();
        sender.send(make_chunk(0.42)).unwrap();

        let result = stream.try_next_chunk().unwrap();
        let chunk = result.expect("should have received the injected chunk");
        assert_eq!(chunk.data, vec![0.42]);
    }

    #[test]
    fn test_try_next_chunk_returns_err_when_closed() {
        let (stream, sender, running) = test_stream();
        drop(sender); // simulate cpal stream dropping the sender
        running.store(false, Ordering::Relaxed);

        let err = stream.try_next_chunk().unwrap_err();
        assert!(matches!(err, DecibriError::CaptureStreamClosed));
    }

    #[test]
    fn test_next_chunk_blocks_until_chunk_arrives() {
        let (stream, sender, _running) = test_stream();

        let producer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            sender.send(make_chunk(0.77)).unwrap();
        });

        let result = stream.next_chunk(None).unwrap();
        let chunk = result.expect("should have received the eventually-pushed chunk");
        assert_eq!(chunk.data, vec![0.77]);

        producer.join().unwrap();
    }

    #[test]
    fn test_next_chunk_timeout_returns_none() {
        let (stream, _sender, _running) = test_stream();
        let start = Instant::now();
        let result = stream.next_chunk(Some(Duration::from_millis(50))).unwrap();
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
        let c1 = stream.next_chunk(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(c1.unwrap().data, vec![1.0]);

        let c2 = stream.next_chunk(Some(Duration::from_millis(100))).unwrap();
        assert_eq!(c2.unwrap().data, vec![2.0]);

        let err = stream
            .next_chunk(Some(Duration::from_millis(100)))
            .unwrap_err();
        assert!(matches!(err, DecibriError::CaptureStreamClosed));
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
        let err = stream.next_chunk(None).unwrap_err();
        let elapsed = start.elapsed();

        assert!(matches!(err, DecibriError::CaptureStreamClosed));
        assert!(
            elapsed < Duration::from_millis(250),
            "next_chunk took too long to detect stop(): {elapsed:?}"
        );

        stopper.join().unwrap();
    }

    /// Compile-time assertion that `Arc<Mutex<CaptureStream>>` is `Send + Sync`,
    /// which requires `CaptureStream: Send`. This is the wrapping strategy PyO3
    /// will document for Python consumers needing shared access from multiple
    /// threads.
    ///
    /// If `CaptureStream` ever becomes `!Send` (for example by adding an `Rc<_>`
    /// or `RefCell<_>` field), this test fails to compile, catching the
    /// regression at build time rather than at a PyO3 wrap call site.
    #[test]
    fn test_arc_mutex_capture_stream_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<std::sync::Mutex<CaptureStream>>>();
    }

    /// `Arc<Mutex<CaptureStream>>` exercised from two threads racing for
    /// the lock: each thread reads one injected chunk, the Mutex
    /// serializes access, and together they consume exactly the chunks
    /// injected (no duplicates, no losses, no deadlock).
    ///
    /// `Barrier` forces both threads to attempt the lock acquisition
    /// simultaneously so the test exercises contention rather than
    /// sequential non-overlapping access.
    #[test]
    fn test_arc_mutex_capture_stream_serializes_two_threads() {
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
            guard.try_next_chunk().unwrap()
        });

        let s2 = shared.clone();
        let b2 = barrier.clone();
        let t2 = thread::spawn(move || {
            b2.wait();
            let guard = s2.lock().unwrap();
            guard.try_next_chunk().unwrap()
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
}
