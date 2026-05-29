//! Speaker playback: open an output device and push `f32` samples to it.
//!
//! Build a [`SpeakerConfig`], construct a [`Speaker`], then
//! [`start`](Speaker::start) it to obtain a [`SpeakerStream`]. Push audio with
//! [`send`](SpeakerStream::send), block until the queue empties with
//! [`drain`](SpeakerStream::drain), or end immediately with
//! [`stop`](SpeakerStream::stop). The capture counterpart is
//! [`crate::microphone`].

#[cfg(feature = "playback")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "playback")]
use std::sync::Arc;

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

/// An open playback stream you push `f32` samples to.
///
/// Obtained from [`Speaker::start`]. Queue samples with [`send`](Self::send),
/// which applies backpressure when the internal buffer is full. Call
/// [`drain`](Self::drain) to block until everything queued has played, or
/// [`stop`](Self::stop) to end immediately and discard what remains. Dropping
/// the stream releases the device. The type is `!Sync`: keep it on one thread
/// or wrap it in a mutex.
#[cfg(feature = "playback")]
pub struct SpeakerStream {
    _stream: cpal::Stream,
    sender: Sender<Vec<f32>>,
    running: Arc<AtomicBool>,
    drain_complete: Arc<AtomicBool>,
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
        if samples.is_empty() {
            return Ok(());
        }
        self.sender
            .send(samples)
            .map_err(|_| DecibriError::SpeakerStreamClosed)
    }

    /// Graceful drain: sends a sentinel (empty vec), then blocks until the
    /// cpal callback has played all remaining samples and received the sentinel.
    pub fn drain(&self) {
        // Send empty vec as sentinel
        let _ = self.sender.send(Vec::new());
        // Poll until cpal callback sets drain_complete
        while !self.drain_complete.load(Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        self.running.store(false, Ordering::Relaxed);
    }

    /// Check whether the stream is still actively playing.
    pub fn is_playing(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Immediate stop. Discards remaining samples in the channel.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        // Drop all pending samples by draining the channel
        while self.sender.try_send(Vec::new()).is_ok() {}
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

    /// Resolve a [`DeviceSelector`] to a concrete output device.
    pub fn resolve_device(selector: &DeviceSelector) -> Result<cpal::Device, DecibriError> {
        crate::device::resolve_output_device(selector)
    }

    /// Start the output stream. Returns a handle for sending samples.
    pub fn start(&self) -> Result<SpeakerStream, DecibriError> {
        let (sender, receiver): (Sender<Vec<f32>>, Receiver<Vec<f32>>) =
            crossbeam_channel::bounded(32);

        let running = Arc::new(AtomicBool::new(true));
        let drain_complete = Arc::new(AtomicBool::new(false));
        let drain_complete_clone = drain_complete.clone();
        let running_clone = running.clone();

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
                    let mut written = 0;

                    // First, drain any leftover from the accumulator
                    if !accum.is_empty() {
                        let take = accum.len().min(data.len());
                        data[..take].copy_from_slice(&accum[..take]);
                        accum.drain(..take);
                        written += take;
                    }

                    // Then pull from the channel until we've filled the buffer
                    while written < data.len() {
                        match receiver.try_recv() {
                            Ok(samples) => {
                                if samples.is_empty() {
                                    // Sentinel: all data has been sent. Fill rest with silence.
                                    for sample in &mut data[written..] {
                                        *sample = 0.0;
                                    }
                                    drain_complete_clone.store(true, Ordering::Relaxed);
                                    return;
                                }
                                let need = data.len() - written;
                                if samples.len() <= need {
                                    data[written..written + samples.len()]
                                        .copy_from_slice(&samples);
                                    written += samples.len();
                                } else {
                                    // More samples than needed: fill buffer, stash remainder
                                    data[written..].copy_from_slice(&samples[..need]);
                                    accum.extend_from_slice(&samples[need..]);
                                    written = data.len();
                                }
                            }
                            Err(_) => {
                                // No data available: fill remaining with silence
                                for sample in &mut data[written..] {
                                    *sample = 0.0;
                                }
                                return;
                            }
                        }
                    }
                },
                move |err| {
                    eprintln!("decibri: audio output error: {err}");
                    running_clone.store(false, Ordering::Relaxed);
                },
                None,
            )
            .map_err(|e| DecibriError::StreamOpenFailed(e.to_string()))?;

        stream
            .play()
            .map_err(|e| DecibriError::StreamStartFailed(e.to_string()))?;

        Ok(SpeakerStream {
            _stream: stream,
            sender,
            running,
            drain_complete,
        })
    }
}
