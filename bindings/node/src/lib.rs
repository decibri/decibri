use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi::{Env, Task};
use napi_derive::napi;

use decibri::device::{self, DeviceSelector};
use decibri::microphone::{Microphone, MicrophoneConfig, MicrophoneStream};
use decibri::sample;
use decibri::speaker::{Speaker, SpeakerConfig, SpeakerSink, SpeakerStream};
use decibri::vad::{SileroVad, VadConfig};
use decibri::CPAL_VERSION;

/// Audio sample format.
#[derive(Debug, Clone, Copy, PartialEq)]
enum SampleFormat {
    Int16,
    Float32,
}

/// Options passed from JS constructor.
///
/// Note on `ort_library_path` (exposed to JS as `ortLibraryPath`): this is an
/// internal plumbing field used by the JS wrapper in
/// `npm/decibri/src/decibri.js`. The wrapper auto-resolves the bundled ORT
/// dylib path via `require.resolve('@decibri/decibri-<platform>')` and passes
/// the absolute path through to here so Rust's `ort::init_from` can load the
/// correct library for the user's platform.
///
/// The field is hidden from the auto-generated TypeScript definitions via
/// `#[napi(skip_typescript)]` so TS/JS consumers do not see `ortLibraryPath`
/// on the public options surface. It is still marshaled at runtime, so the
/// JS wrapper can pass it through. Rationale: adding a typed public option
/// later is non-breaking; removing one is breaking, so we default to not
/// exposing it until a concrete use case arises.
///
/// Users who need to override ORT resolution (e.g. point at a system ORT, or
/// A/B between bundled and custom builds) should set the `ORT_DYLIB_PATH`
/// environment variable before Node starts. That's the supported public
/// mechanism.
///
/// If this field is `None` (the default when not injected by the wrapper),
/// Rust calls `ort::init()` which honours `ORT_DYLIB_PATH`.
#[napi(object)]
#[derive(Default)]
pub struct DecibriOptions {
    pub sample_rate: Option<u32>,
    pub channels: Option<u32>,
    pub frames_per_buffer: Option<u32>,
    pub format: Option<String>,
    pub device: Option<serde_json::Value>,
    pub vad_mode: Option<String>,
    pub model_path: Option<String>,
    #[napi(skip_typescript)]
    pub ort_library_path: Option<String>,
}

/// Native bridge class exposed to Node.js via napi-rs.
#[napi]
pub struct DecibriBridge {
    capture: Option<Microphone>,
    stream: Option<MicrophoneStream>,
    format: SampleFormat,
    frames_per_buffer: u32,
    channels: u16,
    running: Arc<AtomicBool>,
    pump_handle: Option<thread::JoinHandle<()>>,
    vad: Option<SileroVad>,
    vad_probability: Arc<AtomicU32>,
}

/// The `Send` pieces of a microphone bridge produced by the open work: the
/// resolved device wrapped in a `Microphone` and the (optionally) loaded Silero
/// model. Holding these lets the blocking open work run off the JS thread in an
/// `AsyncTask` and the bridge be assembled back on the JS thread in `resolve`.
/// Every field here is `Send`, so the whole struct can cross the libuv boundary:
/// `Microphone` owns only config plus a `cpal::Device` (Send + Sync), and
/// `SileroVad` already crosses threads in the capture pump. The `cpal::Stream`
/// is not created here; it is opened later in `start`.
pub struct MicrophoneParts {
    capture: Microphone,
    format: SampleFormat,
    frames_per_buffer: u32,
    channels: u16,
    vad: Option<SileroVad>,
}

/// Resolve the device and load the Silero model. This is the blocking open work
/// shared by the synchronous constructor and the async `openAsync` task. It is
/// safe to run off the JS thread: it touches no napi/JS state and produces only
/// `Send` values. The Silero model load is the slow part (roughly 100 to 500 ms
/// on a cold cache); device resolution is fast.
fn build_microphone_parts(options: Option<DecibriOptions>) -> Result<MicrophoneParts> {
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

    let config = MicrophoneConfig {
        sample_rate,
        channels,
        frames_per_buffer,
        device,
    };

    let capture = Microphone::new(config).map_err(to_napi_error)?;

    // Silero VAD: load model if vadMode is 'silero'
    let vad_mode = opts.vad_mode.as_deref().unwrap_or("energy");
    let vad = if vad_mode == "silero" {
        let model_path = opts.model_path.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "modelPath is required when vadMode is 'silero'",
            )
        })?;
        let vad_config = VadConfig {
            model_path: std::path::PathBuf::from(model_path),
            sample_rate,
            threshold: 0.5, // JS wrapper controls the actual threshold
            // Populated by the JS wrapper from require.resolve on the
            // resolved platform package. When `None`, Rust falls through
            // to `ort::init()` which honours the `ORT_DYLIB_PATH` env var.
            ort_library_path: opts
                .ort_library_path
                .as_deref()
                .map(std::path::PathBuf::from),
        };
        Some(SileroVad::new(vad_config).map_err(to_napi_error)?)
    } else {
        None
    };

    Ok(MicrophoneParts {
        capture,
        format,
        frames_per_buffer,
        channels,
        vad,
    })
}

impl MicrophoneParts {
    /// Assemble the bridge from the loaded parts. No blocking work: just struct
    /// assembly with fresh runtime state. Runs on the JS thread (either inside
    /// the synchronous constructor or in the async task's `resolve`).
    fn into_bridge(self) -> DecibriBridge {
        DecibriBridge {
            capture: Some(self.capture),
            stream: None,
            format: self.format,
            frames_per_buffer: self.frames_per_buffer,
            channels: self.channels,
            running: Arc::new(AtomicBool::new(false)),
            pump_handle: None,
            vad: self.vad,
            vad_probability: Arc::new(AtomicU32::new(0)),
        }
    }
}

/// Background task that performs the blocking microphone open work (device
/// resolution and the Silero model load) on the libuv thread pool, off the JS
/// event loop, then resolves to a constructed bridge on the JS thread. A failed
/// open propagates through the default `reject`, rejecting the Promise with the
/// matching error.
pub struct OpenMicrophoneTask {
    options: Option<DecibriOptions>,
}

impl Task for OpenMicrophoneTask {
    type Output = MicrophoneParts;
    type JsValue = DecibriBridge;

    fn compute(&mut self) -> Result<Self::Output> {
        build_microphone_parts(self.options.take())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output.into_bridge())
    }
}

#[napi]
impl DecibriBridge {
    #[napi(constructor)]
    pub fn new(options: Option<DecibriOptions>) -> Result<Self> {
        Ok(build_microphone_parts(options)?.into_bridge())
    }

    /// Construct a microphone bridge without blocking the JS event loop. The
    /// device resolution and Silero model load run on the libuv thread pool;
    /// the returned Promise resolves to a fully constructed bridge, or rejects
    /// with the matching error. The synchronous `new` remains available and
    /// unchanged.
    #[napi]
    pub fn open_async(options: Option<DecibriOptions>) -> AsyncTask<OpenMicrophoneTask> {
        AsyncTask::new(OpenMicrophoneTask { options })
    }

    /// Start capturing audio. The callback receives `(err, chunk)` for each buffer.
    #[napi(ts_args_type = "callback: (err: Error | null, chunk: Buffer) => void")]
    pub fn start(&mut self, callback: Function<Buffer, ()>) -> Result<()> {
        if self.running.load(Ordering::Relaxed) {
            return Err(Error::new(
                Status::GenericFailure,
                "audio stream is already running. Call stop() first.",
            ));
        }

        let capture = self
            .capture
            .as_ref()
            .ok_or_else(|| Error::new(Status::GenericFailure, "Capture not initialized"))?;

        let stream = capture.start().map_err(to_napi_error)?;
        let receiver = stream.receiver().clone();

        self.running.store(true, Ordering::Relaxed);
        let running = self.running.clone();
        let format = self.format;
        let target_samples = self.frames_per_buffer as usize * self.channels as usize;
        let vad_probability = self.vad_probability.clone();

        // Move VAD into the pump thread (SileroVad is Send)
        let mut vad = self.vad.take();

        // Create a threadsafe function to call back into JS from the pump thread.
        let tsfn = callback
            .build_threadsafe_function()
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

                            // Run Silero VAD on the f32 frame (before format conversion)
                            if let Some(ref mut v) = vad {
                                if let Ok(result) = v.process(&frame) {
                                    vad_probability
                                        .store(result.probability.to_bits(), Ordering::Relaxed);
                                }
                            }

                            let bytes = match format {
                                SampleFormat::Int16 => sample::f32_to_i16_le_bytes(&frame),
                                SampleFormat::Float32 => sample::f32_to_f32_le_bytes(&frame),
                            };
                            tsfn.call(
                                Ok(Buffer::from(bytes)),
                                ThreadsafeFunctionCallMode::NonBlocking,
                            );
                        }
                    }
                    Err(_) => {
                        // Channel closed. Capture stopped.
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

    /// Latest VAD speech probability (0.0 to 1.0). Updated by pump thread.
    /// Returns 0.0 if VAD is not active.
    #[napi(getter)]
    pub fn vad_probability(&self) -> f64 {
        f32::from_bits(self.vad_probability.load(Ordering::Relaxed)) as f64
    }

    /// List all available audio input devices.
    #[napi]
    pub fn devices() -> Result<Vec<DeviceInfoJs>> {
        let devices = device::input_devices().map_err(to_napi_error)?;
        Ok(devices
            .into_iter()
            .map(|d| DeviceInfoJs {
                index: d.index as u32,
                name: d.name,
                id: d.id,
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
            audio_backend: format!("cpal {}", CPAL_VERSION),
        }
    }
}

/// Device info returned to JS.
#[napi(object)]
pub struct DeviceInfoJs {
    pub index: u32,
    pub name: String,
    /// Stable per-host device ID suitable for `device: { id: ... }` selection.
    /// WASAPI endpoint ID on Windows, CoreAudio UID on macOS, ALSA pcm_id on
    /// Linux. Empty string if cpal cannot produce a stable ID for this device.
    pub id: String,
    pub max_input_channels: u32,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// Version info returned to JS.
#[napi(object)]
pub struct VersionInfoJs {
    pub decibri: String,
    pub audio_backend: String,
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn resolve_device_option(device: &Option<serde_json::Value>) -> Result<DeviceSelector> {
    match device {
        None | Some(serde_json::Value::Null) => Ok(DeviceSelector::Default),
        Some(serde_json::Value::Number(n)) => {
            let idx = n.as_u64().ok_or_else(|| {
                Error::new(
                    Status::InvalidArg,
                    "device index must be a non-negative integer",
                )
            })? as usize;
            Ok(DeviceSelector::Index(idx))
        }
        Some(serde_json::Value::String(s)) => Ok(DeviceSelector::Name(s.clone())),
        Some(serde_json::Value::Object(map)) => match map.get("id") {
            Some(serde_json::Value::String(s)) => Ok(DeviceSelector::Id(s.clone())),
            Some(_) => Err(Error::new(Status::InvalidArg, "device.id must be a string")),
            None => Err(Error::new(
                Status::InvalidArg,
                "device object must have an 'id' string property",
            )),
        },
        _ => Err(Error::new(
            Status::InvalidArg,
            "device must be a number (index), string (name), or object with 'id' property",
        )),
    }
}

fn to_napi_error(e: decibri::error::DecibriError) -> Error {
    use decibri::error::DecibriError::*;
    // Every variant is listed explicitly so the grep check in our release
    // procedure can confirm coverage. A trailing `_ =>` arm is required
    // because `DecibriError` is `#[non_exhaustive]`; it catches future
    // variants added upstream and maps them to the safe default.
    let status = match &e {
        // Config validation (bad caller input)
        SampleRateOutOfRange
        | ChannelsOutOfRange
        | FramesPerBufferOutOfRange
        | InvalidFormat
        | VadSampleRateUnsupported(_)
        | VadThresholdOutOfRange(_) => Status::InvalidArg,

        // Device selection (bad caller input)
        MicrophoneNotFound(_)
        | SpeakerNotFound(_)
        | MultipleDevicesMatch { .. }
        | DeviceIndexOutOfRange
        | NotAnInputDevice => Status::InvalidArg,

        // Runtime / environmental failures
        NoMicrophoneFound
        | NoSpeakerFound
        | DeviceEnumerationFailed(_)
        | AlreadyRunning
        | StreamOpenFailed(_)
        | StreamStartFailed(_)
        | PermissionDenied
        | MicrophoneStreamClosed
        | SpeakerStreamClosed
        | OrtInitFailed { .. }
        | OrtLoadFailed { .. }
        | OrtPathInvalid { .. }
        | OrtSessionBuildFailed(_)
        | OrtThreadsConfigFailed(_)
        | VadModelLoadFailed { .. }
        | OrtInferenceFailed(_)
        | OrtTensorCreateFailed { .. }
        | OrtTensorExtractFailed { .. } => Status::GenericFailure,

        // `DecibriError` is `#[non_exhaustive]`. A new variant added upstream
        // without a corresponding arm here falls through to `GenericFailure`
        // at runtime rather than failing to compile. Variant coverage was
        // compiler-enforced during the 3.2.0 refactor by temporarily removing
        // both `#[non_exhaustive]` and this arm; both have been restored.
        _ => Status::GenericFailure,
    };
    Error::new(status, e.to_string())
}

// ═══════════════════════════════════════════════════════════════════════════════
// DecibriOutputBridge: audio playback
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
    /// Stable per-host device ID. See `DeviceInfoJs.id` for format and
    /// fallback semantics; identical rules for output devices.
    pub id: String,
    pub max_output_channels: u32,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// Native bridge class for audio output, exposed to Node.js via napi-rs.
#[napi]
pub struct DecibriOutputBridge {
    output: Option<Speaker>,
    stream: Option<SpeakerStream>,
    /// A `Send` handle to the live stream's channel and drain state, captured
    /// when the `stream` is created. The async write/drain tasks clone this to do
    /// the blocking work off the JS thread without holding the `stream`, which the
    /// bridge keeps on the JS thread. `None` until the stream is first created.
    sink: Option<SpeakerSink>,
    format: SampleFormat,
}

/// The `Send` pieces of a speaker bridge produced by the open work. Unlike the
/// microphone, there is no model to load: the only open work is device
/// resolution, so `Speaker.open()` exists chiefly for API parity with
/// `Microphone.open()` (and the Python `AsyncSpeaker.open()`). `Speaker` owns
/// only config plus a `cpal::Device` (Send + Sync), so this struct is `Send`;
/// the `cpal::Stream` is opened lazily on the first write, not here.
pub struct SpeakerParts {
    output: Speaker,
    format: SampleFormat,
}

/// Resolve the output device. Shared by the synchronous constructor and the
/// async `openAsync` task; safe to run off the JS thread and produces only
/// `Send` values.
fn build_speaker_parts(options: Option<DecibriOutputOptions>) -> Result<SpeakerParts> {
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

    let config = SpeakerConfig {
        sample_rate,
        channels,
        device,
    };

    let output = Speaker::new(config).map_err(to_napi_error)?;

    Ok(SpeakerParts { output, format })
}

impl SpeakerParts {
    /// Assemble the bridge from the resolved parts. Runs on the JS thread.
    fn into_bridge(self) -> DecibriOutputBridge {
        DecibriOutputBridge {
            output: Some(self.output),
            stream: None,
            sink: None,
            format: self.format,
        }
    }
}

/// Background task that performs the blocking channel `send` (write under
/// backpressure) off the JS event loop. It holds only `Send` primitives: a clone
/// of the stream's crossbeam `Sender` (wrapped in a `SpeakerSink`) and the
/// already-converted samples. The `cpal::Stream` is not touched here; because the
/// sink is lock-free, the offloaded `send` cannot block the JS thread's `stop` or
/// `is_playing`. A closed stream rejects the Promise with `SpeakerStreamClosed`.
/// An empty task (no sink) is a no-op for empty writes.
pub struct WriteTask {
    sink: Option<SpeakerSink>,
    samples: Option<Vec<f32>>,
}

impl Task for WriteTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        if let (Some(sink), Some(samples)) = (self.sink.as_ref(), self.samples.take()) {
            sink.send(samples).map_err(to_napi_error)?;
        }
        Ok(())
    }

    fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> {
        Ok(())
    }
}

/// Background task that performs the drain wait (the poll loop that blocks until
/// the cpal callback has played everything queued) off the JS event loop. It
/// holds only a `Send` `SpeakerSink` clone and never touches the `cpal::Stream`;
/// because the sink is lock-free, a long drain never blocks the JS thread's
/// `stop` or `is_playing`. With no stream yet created the sink is `None` and the
/// drain is a no-op that resolves immediately, matching the synchronous `drain`.
pub struct DrainTask {
    sink: Option<SpeakerSink>,
}

impl Task for DrainTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<Self::Output> {
        if let Some(sink) = self.sink.as_ref() {
            sink.drain();
        }
        Ok(())
    }

    fn resolve(&mut self, _env: Env, _output: Self::Output) -> Result<Self::JsValue> {
        Ok(())
    }
}

/// Background task that performs the speaker open work (device resolution) on
/// the libuv thread pool, then resolves to a constructed bridge on the JS
/// thread. A failed open rejects the Promise with the matching error.
pub struct OpenSpeakerTask {
    options: Option<DecibriOutputOptions>,
}

impl Task for OpenSpeakerTask {
    type Output = SpeakerParts;
    type JsValue = DecibriOutputBridge;

    fn compute(&mut self) -> Result<Self::Output> {
        build_speaker_parts(self.options.take())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output.into_bridge())
    }
}

#[napi]
impl DecibriOutputBridge {
    #[napi(constructor)]
    pub fn new(options: Option<DecibriOutputOptions>) -> Result<Self> {
        Ok(build_speaker_parts(options)?.into_bridge())
    }

    /// Construct a speaker bridge without blocking the JS event loop. The device
    /// resolution runs on the libuv thread pool; the returned Promise resolves
    /// to a constructed bridge, or rejects with the matching error. The
    /// synchronous `new` remains available and unchanged.
    #[napi]
    pub fn open_async(options: Option<DecibriOutputOptions>) -> AsyncTask<OpenSpeakerTask> {
        AsyncTask::new(OpenSpeakerTask { options })
    }

    /// Open the output stream on first use, capturing the stream (kept on the JS
    /// thread by the bridge) and a `Send` sink handle (used by the async tasks).
    /// Idempotent: a no-op once the stream exists. Always runs on the JS thread.
    fn ensure_stream(&mut self) -> Result<()> {
        if self.stream.is_none() {
            let output = self
                .output
                .as_ref()
                .ok_or_else(|| Error::new(Status::GenericFailure, "Output not initialized"))?;
            let stream = output.start().map_err(to_napi_error)?;
            self.sink = Some(stream.sink());
            self.stream = Some(stream);
        }
        Ok(())
    }

    /// Write PCM data for playback. Starts the output stream on first call.
    /// Empty buffers are a no-op.
    #[napi]
    pub fn write(&mut self, buffer: Buffer) -> Result<()> {
        if buffer.is_empty() {
            return Ok(());
        }

        // Start stream on first write
        self.ensure_stream()?;

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

    /// Non-blocking write: convert the samples and start the stream on the JS
    /// thread (a fast device open, same as the synchronous first write), then
    /// perform the blocking channel `send` (which stalls under backpressure when
    /// the queue is full) on the libuv thread pool. The returned Promise
    /// resolves when the samples are queued, or rejects with the matching error.
    /// Empty buffers resolve immediately. The synchronous `write` is unchanged.
    #[napi]
    pub fn write_async(&mut self, buffer: Buffer) -> Result<AsyncTask<WriteTask>> {
        if buffer.is_empty() {
            return Ok(AsyncTask::new(WriteTask {
                sink: None,
                samples: None,
            }));
        }

        // Stream creation stays on the JS thread; only the lock-free send is offloaded.
        self.ensure_stream()?;

        let samples = match self.format {
            SampleFormat::Int16 => sample::i16_le_bytes_to_f32(&buffer),
            SampleFormat::Float32 => sample::f32_le_bytes_to_f32(&buffer),
        };

        Ok(AsyncTask::new(WriteTask {
            sink: self.sink.clone(),
            samples: Some(samples),
        }))
    }

    /// Graceful drain: blocks until all queued samples have been played.
    #[napi]
    pub fn drain(&self) {
        if let Some(stream) = self.stream.as_ref() {
            stream.drain();
        }
    }

    /// Non-blocking drain: the poll loop that waits for the cpal callback to play
    /// everything queued runs on the libuv thread pool instead of the event loop.
    /// The returned Promise resolves when the buffer has drained. With no stream
    /// yet created it resolves immediately. The synchronous `drain` is unchanged.
    #[napi]
    pub fn drain_async(&self) -> AsyncTask<DrainTask> {
        AsyncTask::new(DrainTask {
            sink: self.sink.clone(),
        })
    }

    /// Immediate stop. Discards remaining samples.
    #[napi]
    pub fn stop(&mut self) {
        if let Some(stream) = self.stream.as_ref() {
            stream.stop();
        }
        self.stream = None;
        self.sink = None;
    }

    /// Whether audio is currently being output.
    #[napi(getter)]
    pub fn is_playing(&self) -> bool {
        self.stream.as_ref().is_some_and(|s| s.is_playing())
    }

    /// List all available audio output devices.
    #[napi]
    pub fn devices() -> Result<Vec<OutputDeviceInfoJs>> {
        let devices = device::output_devices().map_err(to_napi_error)?;
        Ok(devices
            .into_iter()
            .map(|d| OutputDeviceInfoJs {
                index: d.index as u32,
                name: d.name,
                id: d.id,
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
            audio_backend: format!("cpal {}", CPAL_VERSION),
        }
    }
}
