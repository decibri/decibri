use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi_derive::napi;

use decibri::capture::{AudioCapture, CaptureConfig, CaptureStream};
use decibri::device::{self, DeviceSelector};
use decibri::output::{AudioOutput, OutputConfig, OutputStream};
use decibri::sample;

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SampleFormat {
    Int16,
    Float32,
}

/// Options passed from JS constructor.
#[napi(object)]
#[derive(Default)]
pub struct DecibriOptions {
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub frames_per_buffer: Option<u32>,
    pub format: Option<String>,
    pub device: Option<serde_json::Value>,
}

/// Native bridge class exposed to Node.js via napi-rs.
#[napi]
pub struct DecibriBridge {
    capture: Option<AudioCapture>,
    stream: Option<CaptureStream>,
    format: SampleFormat,
    frames_per_buffer: u32,
    channels: u16,
    running: Arc<AtomicBool>,
    pump_handle: Option<thread::JoinHandle<()>>,
}

#[napi]
impl DecibriBridge {
    #[napi(constructor)]
    pub fn new(options: Option<DecibriOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();

        let sample_rate = opts.sample_rate.unwrap_or(16000);
        let channels = opts.channels.unwrap_or(1) as u16;
        let frames_per_buffer = opts.frames_per_buffer.unwrap_or(1600);

        let format_str = opts.format.as_deref().unwrap_or("int16");
        let format = match format_str {
            "int16" => SampleFormat::Int16,
            "float32" => SampleFormat::Float32,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "format must be 'int16' or 'float32'",
                ))
            }
        };

        let device = resolve_device_option(&opts.device)?;

        let config = CaptureConfig {
            sample_rate,
            channels,
            frames_per_buffer,
            device,
        };

        let capture = AudioCapture::new(config).map_err(to_napi_error)?;

        Ok(Self {
            capture: Some(capture),
            stream: None,
            format,
            frames_per_buffer,
            channels,
            running: Arc::new(AtomicBool::new(false)),
            pump_handle: None,
        })
    }

    /// Start capturing audio. The callback receives `(err, chunk)` for each buffer.
    #[napi(
        ts_args_type = "callback: (err: Error | null, chunk: Buffer) => void"
    )]
    pub fn start(&mut self, callback: Function<Buffer, ()>) -> Result<()> {
        if self.running.load(Ordering::Relaxed) {
            return Err(Error::new(
                Status::GenericFailure,
                "Decibri is already running. Call stop() first.",
            ));
        }

        let capture = self.capture.as_ref().ok_or_else(|| {
            Error::new(Status::GenericFailure, "Capture not initialized")
        })?;

        let stream = capture.start().map_err(to_napi_error)?;
        let receiver = stream.receiver().clone();

        self.running.store(true, Ordering::Relaxed);
        let running = self.running.clone();
        let format = self.format;
        let target_samples = self.frames_per_buffer as usize * self.channels as usize;

        // Create a threadsafe function to call back into JS from the pump thread.
        let tsfn = callback.build_threadsafe_function()
            .callee_handled::<true>()
            .build()?;

        let handle = thread::spawn(move || {
            // Frame-exact buffering: accumulate f32 samples from cpal and only
            // send to JS when we have exactly frames_per_buffer * channels samples.
            // WASAPI ignores BufferSize::Fixed and delivers its own chunk sizes,
            // so this ensures the API contract (exact chunk size) is honoured.
            let mut accum: Vec<f32> = Vec::with_capacity(target_samples);

            while running.load(Ordering::Relaxed) {
                match receiver.recv() {
                    Ok(chunk) => {
                        accum.extend_from_slice(&chunk.data);

                        // Drain full frames from the accumulator
                        while accum.len() >= target_samples {
                            let frame: Vec<f32> = accum.drain(..target_samples).collect();
                            let bytes = match format {
                                SampleFormat::Int16 => sample::f32_to_i16_le_bytes(&frame),
                                SampleFormat::Float32 => sample::f32_to_f32_le_bytes(&frame),
                            };
                            tsfn.call(Ok(Buffer::from(bytes)), ThreadsafeFunctionCallMode::NonBlocking);
                        }
                    }
                    Err(_) => {
                        // Channel closed — capture stopped.
                        // Discard partial buffer (matches C++ PortAudio behaviour).
                        break;
                    }
                }
            }
        });

        self.stream = Some(stream);
        self.pump_handle = Some(handle);

        Ok(())
    }

    /// Stop capturing audio.
    #[napi]
    pub fn stop(&mut self) {
        self.running.store(false, Ordering::Relaxed);

        if let Some(stream) = self.stream.as_ref() {
            stream.stop();
        }
        self.stream = None;

        if let Some(handle) = self.pump_handle.take() {
            let _ = handle.join();
        }
    }

    /// Whether the microphone is currently capturing.
    #[napi(getter)]
    pub fn is_open(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// List all available audio input devices.
    #[napi]
    pub fn devices() -> Result<Vec<DeviceInfoJs>> {
        let devices = device::enumerate_input_devices().map_err(to_napi_error)?;
        Ok(devices
            .into_iter()
            .map(|d| DeviceInfoJs {
                index: d.index as u32,
                name: d.name,
                max_input_channels: d.max_input_channels as u32,
                default_sample_rate: d.default_sample_rate,
                is_default: d.is_default,
            })
            .collect())
    }

    /// Version information.
    #[napi]
    pub fn version() -> VersionInfoJs {
        VersionInfoJs {
            decibri: env!("CARGO_PKG_VERSION").to_string(),
            portaudio: "cpal 0.17".to_string(),
        }
    }
}

/// Device info returned to JS.
#[napi(object)]
pub struct DeviceInfoJs {
    pub index: u32,
    pub name: String,
    pub max_input_channels: u32,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// Version info returned to JS.
#[napi(object)]
pub struct VersionInfoJs {
    pub decibri: String,
    pub portaudio: String,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn resolve_device_option(device: &Option<serde_json::Value>) -> Result<DeviceSelector> {
    match device {
        None | Some(serde_json::Value::Null) => Ok(DeviceSelector::Default),
        Some(serde_json::Value::Number(n)) => {
            let idx = n.as_u64().ok_or_else(|| {
                Error::new(Status::InvalidArg, "device index must be a non-negative integer")
            })? as usize;
            Ok(DeviceSelector::Index(idx))
        }
        Some(serde_json::Value::String(s)) => Ok(DeviceSelector::Name(s.clone())),
        _ => Err(Error::new(
            Status::InvalidArg,
            "device must be a number (index) or string (name)",
        )),
    }
}

fn to_napi_error(e: decibri::error::DecibriError) -> Error {
    use decibri::error::DecibriError::*;
    let status = match &e {
        SampleRateOutOfRange | ChannelsOutOfRange | FramesPerBufferOutOfRange
        | DeviceIndexOutOfRange => Status::InvalidArg,
        InvalidFormat | DeviceNotFound(_) | MultipleDevicesMatch { .. } => Status::InvalidArg,
        _ => Status::GenericFailure,
    };
    Error::new(status, e.to_string())
}

// ═══════════════════════════════════════════════════════════════════════════════
// DecibriOutputBridge — audio playback
// ═══════════════════════════════════════════════════════════════════════════════

/// Options passed from JS constructor for output.
#[napi(object)]
#[derive(Default)]
pub struct DecibriOutputOptions {
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub format: Option<String>,
    pub device: Option<serde_json::Value>,
}

/// Output device info returned to JS.
#[napi(object)]
pub struct OutputDeviceInfoJs {
    pub index: u32,
    pub name: String,
    pub max_output_channels: u32,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// Native bridge class for audio output, exposed to Node.js via napi-rs.
#[napi]
pub struct DecibriOutputBridge {
    output: Option<AudioOutput>,
    stream: Option<OutputStream>,
    format: SampleFormat,
}

#[napi]
impl DecibriOutputBridge {
    #[napi(constructor)]
    pub fn new(options: Option<DecibriOutputOptions>) -> Result<Self> {
        let opts = options.unwrap_or_default();

        let sample_rate = opts.sample_rate.unwrap_or(16000);
        let channels = opts.channels.unwrap_or(1) as u16;

        let format_str = opts.format.as_deref().unwrap_or("int16");
        let format = match format_str {
            "int16" => SampleFormat::Int16,
            "float32" => SampleFormat::Float32,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "format must be 'int16' or 'float32'",
                ))
            }
        };

        let device = resolve_device_option(&opts.device)?;

        let config = OutputConfig {
            sample_rate,
            channels,
            device,
        };

        let output = AudioOutput::new(config).map_err(to_napi_error)?;

        Ok(Self {
            output: Some(output),
            stream: None,
            format,
        })
    }

    /// Write PCM data for playback. Starts the output stream on first call.
    /// Empty buffers are a no-op.
    #[napi]
    pub fn write(&mut self, buffer: Buffer) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Start stream on first write
        if self.stream.is_none() {
            let output = self.output.as_ref().ok_or_else(|| {
                Error::new(Status::GenericFailure, "Output not initialized")
            })?;
            let stream = output.start().map_err(to_napi_error)?;
            self.stream = Some(stream);
        }

        let samples = match self.format {
            SampleFormat::Int16 => sample::i16_le_bytes_to_f32(&buffer),
            SampleFormat::Float32 => sample::f32_le_bytes_to_f32(&buffer),
        };

        self.stream
            .as_ref()
            .unwrap()
            .send(samples)
            .map_err(to_napi_error)
    }

    /// Graceful drain: blocks until all queued samples have been played.
    #[napi]
    pub fn drain(&self) {
        if let Some(stream) = self.stream.as_ref() {
            stream.drain();
        }
    }

    /// Immediate stop. Discards remaining samples.
    #[napi]
    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.as_ref() {
            stream.stop();
        }
        self.stream = None;
    }

    /// Whether audio is currently being output.
    #[napi(getter)]
    pub fn is_playing(&self) -> bool {
        self.stream
            .as_ref()
            .map_or(false, |s| s.is_playing())
    }

    /// List all available audio output devices.
    #[napi]
    pub fn devices() -> Result<Vec<OutputDeviceInfoJs>> {
        let devices = device::enumerate_output_devices().map_err(to_napi_error)?;
        Ok(devices
            .into_iter()
            .map(|d| OutputDeviceInfoJs {
                index: d.index as u32,
                name: d.name,
                max_output_channels: d.max_output_channels as u32,
                default_sample_rate: d.default_sample_rate,
                is_default: d.is_default,
            })
            .collect())
    }

    /// Version information.
    #[napi]
    pub fn version() -> VersionInfoJs {
        VersionInfoJs {
            decibri: env!("CARGO_PKG_VERSION").to_string(),
            portaudio: "cpal 0.17".to_string(),
        }
    }
}
