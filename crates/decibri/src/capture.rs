#[cfg(feature = "capture")]
use std::sync::atomic::{AtomicBool, Ordering};
#[cfg(feature = "capture")]
use std::sync::Arc;

#[cfg(feature = "capture")]
use cpal::traits::{DeviceTrait, StreamTrait};
#[cfg(feature = "capture")]
use crossbeam_channel::{Receiver, Sender};

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
    _stream: cpal::Stream,
    receiver: Receiver<AudioChunk>,
    running: Arc<AtomicBool>,
    #[allow(dead_code)]
    sample_rate: u32,
    #[allow(dead_code)]
    channels: u16,
}

#[cfg(feature = "capture")]
impl CaptureStream {
    /// Get a reference to the receiver for reading audio chunks.
    pub fn receiver(&self) -> &Receiver<AudioChunk> {
        &self.receiver
    }

    /// Check whether the stream is still actively capturing.
    pub fn is_open(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// Stop capturing audio.
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
            sample_rate: sample_rate,
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
            _stream: stream,
            receiver,
            running,
            sample_rate,
            channels,
        })
    }
}
