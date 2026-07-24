use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use napi::bindgen_prelude::*;
use napi::threadsafe_function::ThreadsafeFunctionCallMode;
use napi::{Env, Task};
use napi_derive::napi;

use decibri::device::{self, DeviceSelector};
use decibri::file::{File as CoreFile, FileConfig, VadReport as CoreVadReport};
use decibri::microphone::{
    DenoiseModel, HighpassFilter, Microphone, MicrophoneConfig, MicrophoneStream,
};
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
    /// Capture DC-removal toggle. When `true`, removes a constant (DC) offset
    /// from the captured audio with a one-pole DC-blocking high-pass, applied
    /// first in the transform chain (before denoise). Absent or `false` leaves
    /// it off (the default), a byte-identical no-op. Pure DSP: no bundled file
    /// and no model path, like `highpass`.
    pub dc_removal: Option<bool>,
    /// Capture denoise model selector. The only accepted value is
    /// `'fastenhancer-t'`; absent leaves denoise off. The JS wrapper resolves
    /// the bundled model file and passes its path through `denoise_model_path`.
    pub denoise: Option<String>,
    /// Filesystem path to the denoise model's ONNX file, resolved by the JS
    /// wrapper from its bundled copy. Required when `denoise` names a model.
    /// Internal plumbing (the user passes only `denoise`), so it is hidden from
    /// the generated TypeScript like `ort_library_path`.
    #[napi(skip_typescript)]
    pub denoise_model_path: Option<String>,
    #[napi(skip_typescript)]
    pub ort_library_path: Option<String>,
    /// Capture high-pass filter cutoff in Hz. The accepted values are `80` (an
    /// 80 Hz second-order Butterworth high-pass) and `100` (a 100 Hz one);
    /// absent leaves the high-pass off. Pure DSP: no bundled file and no model
    /// path, unlike `denoise`.
    pub highpass: Option<i32>,
    /// Capture AGC target level in dBFS: an integer in `-40..=-3` (typical -18);
    /// absent leaves AGC off. Drives the captured level toward the target. Pure
    /// DSP: no bundled file and no model path, like `highpass`.
    pub agc: Option<i32>,
    /// Capture limiter ceiling in dBFS (sample-peak): a number in `-3.0..=0.0`
    /// (typical -1.0); absent leaves the limiter off. Holds the captured signal
    /// at or below the ceiling, catching a peak the AGC would let through. Pure
    /// DSP: no bundled file and no model path, like `agc`.
    pub limiter: Option<f64>,
}

/// Native bridge class exposed to Node.js via napi-rs.
#[napi]
pub struct DecibriBridge {
    capture: Option<Microphone>,
    stream: Option<Arc<MicrophoneStream>>,
    format: SampleFormat,
    frames_per_buffer: u32,
    running: Arc<AtomicBool>,
    pump_handle: Option<thread::JoinHandle<Option<SileroVad>>>,
    vad: Option<SileroVad>,
    // Energy VAD active (vadMode == "energy"). Mutually exclusive with `vad`
    // (Silero): the pump computes the energy score (RMS of the pre-enhancement
    // tap) into `vad_probability` instead of a Silero probability. A `Copy`
    // flag, so the pump captures it by value (no hand-back like `vad`).
    energy_vad: bool,
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
    vad: Option<SileroVad>,
    energy_vad: bool,
}

/// Resolve the device and load the Silero model. This is the blocking open work
/// shared by the synchronous constructor and the async `openAsync` task. It is
/// safe to run off the JS thread: it touches no napi/JS state and produces only
/// `Send` values. The Silero model load is the slow part (roughly 100 to 500 ms
/// on a cold cache); device resolution is fast.
fn build_microphone_parts(options: Option<DecibriOptions>) -> Result<MicrophoneParts> {
    let opts = options.unwrap_or_default();

    let sample_rate = opts.sample_rate.unwrap_or(16000);
    let channels: u16 = opts
        .channels
        .unwrap_or(1)
        .try_into()
        .map_err(|_| Error::new(Status::InvalidArg, "invalid channel count".to_string()))?;
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

    // `MicrophoneConfig` is `#[non_exhaustive]`: default-construct then assign
    // the public fields rather than using a struct literal.
    let mut config = MicrophoneConfig::default();
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.frames_per_buffer = frames_per_buffer;
    config.device = device;

    // DC removal: a same-length one-pole DC-blocking high-pass, the first
    // transform stage (before denoise). Absent or `false` leaves it off (the
    // default), a byte-identical no-op. A plain bool toggle, so there is no
    // range check, only the forward to the core config field.
    config.dc_removal = opts.dc_removal.unwrap_or(false);

    // Denoise: map the closed-set model name to the core selector and attach
    // the bundled model path the JS wrapper resolved. Absent leaves denoise off
    // and the capture path unchanged. The model is loaded later, in start(), so
    // an unknown name fails here but a load failure surfaces from start().
    if let Some(name) = opts.denoise.as_deref() {
        let model = match name {
            "fastenhancer-t" => DenoiseModel::FastEnhancerT,
            other => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("denoise must be 'fastenhancer-t', got '{other}'"),
                ))
            }
        };
        let model_path = opts.denoise_model_path.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "denoiseModelPath is required when denoise is set",
            )
        })?;
        config.denoise = Some(model);
        config.denoise_model_path = Some(std::path::PathBuf::from(model_path));
        // The denoise stage loads through the ONNX seam, so hand the core the
        // bundled ORT dylib path the JS wrapper resolved (the same path the VAD
        // uses). Without it a denoise-only capture could not find the bundled
        // runtime on a default install.
        config.ort_library_path = opts
            .ort_library_path
            .as_deref()
            .map(std::path::PathBuf::from);
    }

    // High-pass: map the closed-set cutoff in Hz to the core selector. Absent
    // leaves the high-pass off and the capture path unchanged. Pure DSP, so
    // there is no model file or ORT to resolve, only the closed-set check (the
    // JS wrapper performs the same check and raises the RangeError; this is the
    // native backstop).
    if let Some(hz) = opts.highpass {
        let filter = match hz {
            80 => HighpassFilter::Hz80,
            100 => HighpassFilter::Hz100,
            other => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("highpass must be one of: 80, 100; got {other}"),
                ))
            }
        };
        config.highpass = Some(filter);
    }

    // AGC: a dBFS target level threaded to the core level-control engine. The JS
    // wrapper performs the user-facing range check (a RangeError); this is the
    // native backstop, range-checking the JS number before narrowing it to the
    // core's i8 (the core also guards in validate()).
    if let Some(target) = opts.agc {
        if !(-40..=-3).contains(&target) {
            return Err(Error::new(
                Status::InvalidArg,
                "agc target level must be between -40 and -3",
            ));
        }
        config.agc = Some(target as i8);
    }

    // Limiter: a sample-peak ceiling in dBFS threaded to the core limiter stage.
    // The JS wrapper performs the user-facing range check (a RangeError); this is
    // the native backstop, range-checking the JS number before narrowing it to the
    // core's f32 (the core also guards in validate()).
    if let Some(ceiling) = opts.limiter {
        if !(-3.0..=0.0).contains(&ceiling) {
            return Err(Error::new(
                Status::InvalidArg,
                "limiter ceiling must be between -3.0 and 0.0",
            ));
        }
        config.limiter = Some(ceiling as f32);
    }

    let capture = Microphone::new(config).map_err(to_napi_error)?;

    // VAD mode selector. The JS wrapper passes 'silero' or 'energy' only when VAD
    // is enabled, and omits it otherwise, so an absent value means VAD is off.
    // 'silero' loads the Silero model below; 'energy' runs the wrapper-free RMS
    // path in the pump (no model, no native object), keyed by `energy_vad`.
    let vad_mode = opts.vad_mode.as_deref();
    let energy_vad = vad_mode == Some("energy");
    let vad = if vad_mode == Some("silero") {
        let model_path = opts.model_path.ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "modelPath is required when vadMode is 'silero'",
            )
        })?;
        // `VadConfig` is `#[non_exhaustive]`: default-construct then assign.
        let mut vad_config = VadConfig::default();
        vad_config.model_path = std::path::PathBuf::from(model_path);
        vad_config.sample_rate = sample_rate;
        // JS wrapper controls the actual threshold.
        vad_config.threshold = 0.5;
        // Populated by the JS wrapper from require.resolve on the resolved
        // platform package. When `None`, Rust falls through to `ort::init()`
        // which honours the `ORT_DYLIB_PATH` env var.
        vad_config.ort_library_path = opts
            .ort_library_path
            .as_deref()
            .map(std::path::PathBuf::from);
        Some(SileroVad::new(vad_config).map_err(to_napi_error)?)
    } else {
        None
    };

    Ok(MicrophoneParts {
        capture,
        format,
        frames_per_buffer,
        vad,
        energy_vad,
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
            running: Arc::new(AtomicBool::new(false)),
            pump_handle: None,
            vad: self.vad,
            energy_vad: self.energy_vad,
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

        // Reclaim the VAD from a previous pump that finished without stop()
        // (for example after a device error ended capture), so this start runs
        // with VAD again. After a clean stop() this take() yields None.
        if let Some(handle) = self.pump_handle.take() {
            self.vad = handle.join().ok().flatten();
        }
        // Start each session from a clean VAD probability rather than a value
        // left over from a previous session.
        self.vad_probability
            .store(0.0f32.to_bits(), Ordering::Relaxed);

        // Keep the core stream behind an `Arc` so the pump thread can pull
        // re-blocked chunks and poll its `is_open` / `take_last_error` while the
        // bridge still owns it.
        let stream = Arc::new(capture.start().map_err(to_napi_error)?);
        let pump_stream = stream.clone();

        self.running.store(true, Ordering::Relaxed);
        let running = self.running.clone();
        let format = self.format;
        // After the engine's normalize chain the output is mono, so the block
        // size and the VAD downmix trigger are counted in OUTPUT channels (read
        // from the stream), not the device channels. The binding-side downmix
        // becomes a no-op once the engine has already delivered mono.
        let output_channels = stream.channels();
        let target_samples = self.frames_per_buffer as usize * output_channels as usize;
        let channels = output_channels;
        let vad_probability = self.vad_probability.clone();

        // Move VAD into the pump thread (SileroVad is Send); it is handed back
        // when the pump exits so a later start() can reuse it. `energy_vad` is a
        // Copy flag, captured by value (nothing to hand back).
        let mut vad = self.vad.take();
        let energy_vad = self.energy_vad;

        // Create a threadsafe function to call back into JS from the pump thread.
        let tsfn = callback
            .build_threadsafe_function()
            .callee_handled::<true>()
            .build()?;

        let handle = thread::spawn(move || {
            // The core re-blocks the device's native buffers to exactly
            // `target_samples` (frames_per_buffer * channels), so each delivered
            // chunk is one full frame (the final chunk at close may be shorter,
            // carrying the remaining tail): no binding-side accumulation. A timed
            // read (matching the core's ~20 ms wakeup) keeps the loop responsive.
            // The loop runs until the core stream closes: an explicit stop() or a
            // device error surfaces as Err, but next_chunk drains and forwards all
            // buffered audio (full blocks then the final short tail) first, so no
            // captured frame is lost.
            const POLL: Duration = Duration::from_millis(20);

            loop {
                match pump_stream.next_chunk(target_samples, Some(POLL)) {
                    Ok(Some(chunk)) => {
                        let frame = chunk.data;

                        // Run VAD on the f32 frame (before format conversion).
                        // Both modes read the signal BEFORE the opt-in
                        // enhancement step: when the tap is active, drain it;
                        // otherwise the delivered frame already is that signal.
                        // Reading the pre-enhancement signal means enabling an
                        // enhancement step does not change detection, the same
                        // guarantee for Silero and energy alike. Called once per
                        // delivered chunk so the tap drains in lockstep. VAD
                        // models a single channel, so downmix interleaved
                        // multichannel frames to mono first: feeding interleaved
                        // samples makes the score read consecutive channels as
                        // successive mono samples. The interleaved `frame` is
                        // untouched and is still what gets emitted to JS below.
                        if vad.is_some() || energy_vad {
                            let pre_enh = pump_stream.vad_input(frame.len());
                            let vad_frame: &[f32] = pre_enh.as_deref().unwrap_or(&frame);
                            let downmixed;
                            let vad_input: &[f32] = if channels > 1 {
                                downmixed = sample::downmix_to_mono(vad_frame, channels);
                                &downmixed
                            } else {
                                vad_frame
                            };
                            if let Some(ref mut v) = vad {
                                if let Ok(result) = v.process(vad_input) {
                                    vad_probability
                                        .store(result.probability.to_bits(), Ordering::Relaxed);
                                }
                            } else {
                                // Energy mode: the score is the RMS of the
                                // pre-enhancement signal, on the same [0, 1]
                                // scale the energy threshold compares against.
                                let score = sample::rms(vad_input);
                                vad_probability.store(score.to_bits(), Ordering::Relaxed);
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
                    Ok(None) => {
                        // Timeout with no full frame yet: keep waiting.
                    }
                    Err(_) => {
                        // The core stream closed (an explicit stop() or a
                        // device/driver error) and all buffered audio has been
                        // forwarded. Forward the typed cause to JS when one was
                        // recorded so the callback raises 'error' rather than
                        // silently stalling; a clean stop records none.
                        if let Some(err) = pump_stream.take_last_error() {
                            tsfn.call(
                                Err(to_napi_error(err)),
                                ThreadsafeFunctionCallMode::NonBlocking,
                            );
                        }
                        running.store(false, Ordering::Relaxed);
                        break;
                    }
                }
            }

            // Hand the VAD back to the bridge (it was moved in via take()).
            vad
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
            // Reclaim the VAD the pump hands back so a later start() keeps VAD
            // (start() moved it into the pump via take()).
            self.vad = handle.join().ok().flatten();
        }

        // Clear the probability so a restarted bridge does not read a stale
        // pre-stop score before its first new inference.
        self.vad_probability
            .store(0.0f32.to_bits(), Ordering::Relaxed);
    }

    /// Whether the microphone is currently capturing.
    #[napi(getter)]
    pub fn is_open(&self) -> bool {
        // Reflect the core stream's real state: a device/driver error closes
        // the core stream, so `isOpen` reports `false` rather than the bridge's
        // own flag staying `true` until the pump notices.
        self.stream.as_ref().is_some_and(|s| s.is_open())
    }

    /// Latest VAD speech probability (0.0 to 1.0). Updated by pump thread.
    /// Returns 0.0 if VAD is not active.
    #[napi(getter)]
    pub fn vad_probability(&self) -> f64 {
        f32::from_bits(self.vad_probability.load(Ordering::Relaxed)) as f64
    }

    /// Number of capture buffers dropped because the consumer could not keep
    /// pace (the core stream's overrun counter). Returns 0 while the consumer
    /// keeps up or when no stream is active. Returned as f64 (an exact JS
    /// number for any realistic count) to match the `vadProbability` getter.
    #[napi(getter)]
    pub fn overrun_count(&self) -> f64 {
        self.stream.as_ref().map_or(0, |s| s.overrun_count()) as f64
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
        | MultichannelNotSupported
        | FramesPerBufferOutOfRange
        | AgcTargetOutOfRange
        | LimiterCeilingOutOfRange
        | InvalidFormat
        | VadSampleRateUnsupported(_)
        | VadThresholdOutOfRange(_)
        | WavInvalid { .. }
        | VadNotConfigured => Status::InvalidArg,

        // Offline file read failure (I/O)
        FileReadFailed { .. } => Status::GenericFailure,

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
        | FileEngaged
        | DeviceFailed { .. }
        | OrtInitFailed { .. }
        | OrtLoadFailed { .. }
        | OrtPathInvalid { .. }
        | OrtSessionBuildFailed(_)
        | OrtThreadsConfigFailed(_)
        | VadModelLoadFailed { .. }
        | ModelLoadFailed { .. }
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
    let channels: u16 = opts
        .channels
        .unwrap_or(1)
        .try_into()
        .map_err(|_| Error::new(Status::InvalidArg, "invalid channel count".to_string()))?;

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

    // `SpeakerConfig` is `#[non_exhaustive]`: default-construct then assign.
    let mut config = SpeakerConfig::default();
    config.sample_rate = sample_rate;
    config.channels = channels;
    config.device = device;

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

// ---------------------------------------------------------------------------
// FileHandle: offline source class.
//
// Wraps the core offline File: constructed from a WAV path or in-memory
// samples, pulled for conditioned chunks (readChunk), or consumed by a
// whole-recording analysis (analyze). Mirrors DecibriBridge's split of
// responsibilities: the native side computes the per-chunk VAD score on the
// pre-conditioning feed and exposes it via `vadProbability`; the JS wrapper
// applies the threshold + holdoff policy, in FILE time.
// ---------------------------------------------------------------------------

/// Options passed from the JS `File` wrapper. The conditioning fields mirror
/// `DecibriOptions` exactly; the live-capture-only fields (device, channels,
/// framesPerBuffer) do not apply to an offline source. `vadThreshold` and
/// `vadHoldoffMs` are internal plumbing (the user passes them on the `vad`
/// config object; the wrapper resolves them), hidden from the generated
/// TypeScript like `ortLibraryPath`.
#[napi(object)]
#[derive(Default)]
pub struct FileOptions {
    pub sample_rate: Option<u32>,
    pub format: Option<String>,
    pub vad_mode: Option<String>,
    #[napi(skip_typescript)]
    pub vad_threshold: Option<f64>,
    #[napi(skip_typescript)]
    pub vad_holdoff_ms: Option<u32>,
    pub model_path: Option<String>,
    pub dc_removal: Option<bool>,
    pub denoise: Option<String>,
    #[napi(skip_typescript)]
    pub denoise_model_path: Option<String>,
    #[napi(skip_typescript)]
    pub ort_library_path: Option<String>,
    pub highpass: Option<i32>,
    pub agc: Option<i32>,
    pub limiter: Option<f64>,
}

/// One scored voice-activity window of a recording: `start` / `end` in
/// seconds of file time, the speech probability, and the raw threshold test.
#[napi(object)]
pub struct VadWindow {
    pub start: f64,
    pub end: f64,
    pub vad_score: f64,
    pub is_speech: bool,
}

/// One merged speech region of a recording, in seconds of file time.
#[napi(object)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
}

/// The whole-recording analysis `File.analyze()` resolves to: per-window
/// scores and merged speech segments, in file order.
#[napi(object)]
pub struct VadReport {
    pub scores: Vec<VadWindow>,
    pub segments: Vec<Segment>,
}

fn convert_report(report: CoreVadReport) -> VadReport {
    VadReport {
        scores: report
            .scores
            .iter()
            .map(|w| VadWindow {
                start: w.start,
                end: w.end,
                vad_score: f64::from(w.probability),
                is_speech: w.is_speech,
            })
            .collect(),
        segments: report
            .segments
            .iter()
            .map(|s| Segment {
                start: s.start,
                end: s.end,
            })
            .collect(),
    }
}

/// Build the core [`FileConfig`] from the JS options, mapping each selector
/// exactly as `build_microphone_parts` maps the same names for live capture.
fn build_file_config(opts: &FileOptions) -> Result<FileConfig> {
    let sample_rate = opts.sample_rate.unwrap_or(16000);

    // `FileConfig` is `#[non_exhaustive]`: default-construct then assign the
    // public fields rather than using a struct literal.
    let mut config = FileConfig::default();
    config.sample_rate = sample_rate;
    config.dc_removal = opts.dc_removal.unwrap_or(false);

    if let Some(name) = opts.denoise.as_deref() {
        let model = match name {
            "fastenhancer-t" => DenoiseModel::FastEnhancerT,
            other => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("denoise must be 'fastenhancer-t', got '{other}'"),
                ))
            }
        };
        let model_path = opts.denoise_model_path.as_deref().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "denoiseModelPath is required when denoise is set",
            )
        })?;
        config.denoise = Some(model);
        config.denoise_model_path = Some(std::path::PathBuf::from(model_path));
        config.ort_library_path = opts
            .ort_library_path
            .as_deref()
            .map(std::path::PathBuf::from);
    }

    if let Some(hz) = opts.highpass {
        let filter = match hz {
            80 => HighpassFilter::Hz80,
            100 => HighpassFilter::Hz100,
            other => {
                return Err(Error::new(
                    Status::InvalidArg,
                    format!("highpass must be one of: 80, 100; got {other}"),
                ))
            }
        };
        config.highpass = Some(filter);
    }

    if let Some(target) = opts.agc {
        if !(-40..=-3).contains(&target) {
            return Err(Error::new(
                Status::InvalidArg,
                "agc target level must be between -40 and -3",
            ));
        }
        config.agc = Some(target as i8);
    }

    if let Some(ceiling) = opts.limiter {
        if !(-3.0..=0.0).contains(&ceiling) {
            return Err(Error::new(
                Status::InvalidArg,
                "limiter ceiling must be between -3.0 and 0.0",
            ));
        }
        config.limiter = Some(ceiling as f32);
    }

    // Whole-recording analysis runs the core's own detector; wire it only for
    // the silero mode (energy mode has no whole-recording analysis and its
    // per-chunk score needs no model).
    if opts.vad_mode.as_deref() == Some("silero") {
        let model_path = opts.model_path.as_deref().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "modelPath is required when vadMode is 'silero'",
            )
        })?;
        // `VadConfig` is `#[non_exhaustive]`: default-construct then assign.
        let mut vad_config = VadConfig::default();
        vad_config.model_path = std::path::PathBuf::from(model_path);
        vad_config.threshold = opts.vad_threshold.unwrap_or(0.5) as f32;
        vad_config.ort_library_path = opts
            .ort_library_path
            .as_deref()
            .map(std::path::PathBuf::from);
        config.vad = Some(vad_config);
        config.vad_holdoff_ms = opts.vad_holdoff_ms.unwrap_or(300);
    }

    Ok(config)
}

/// The `Send` pieces of an opened offline source, assembled into a
/// [`FileHandle`] on the JS thread (by the synchronous factory or in the
/// async open task's `resolve`).
pub struct FileParts {
    file: CoreFile,
    format: SampleFormat,
    vad_enabled: bool,
    energy_vad: bool,
    vad_model_path: Option<String>,
    vad_threshold: f32,
    ort_library_path: Option<String>,
    sample_rate: u32,
}

/// Open a WAV path as the core offline source: the blocking work (file read,
/// WAV parse, chain construction) shared by the synchronous factory and the
/// async open task. Touches no napi/JS state, so it can run off the JS thread.
fn build_file_parts(path: &str, options: Option<&FileOptions>) -> Result<FileParts> {
    let default_opts = FileOptions::default();
    let opts = options.unwrap_or(&default_opts);
    let format = match opts.format.as_deref().unwrap_or("int16") {
        "int16" => SampleFormat::Int16,
        "float32" => SampleFormat::Float32,
        _ => {
            return Err(Error::new(
                Status::InvalidArg,
                "format must be 'int16' or 'float32'",
            ))
        }
    };
    let config = build_file_config(opts)?;
    let file = CoreFile::open(path, config).map_err(to_napi_error)?;
    Ok(FileParts {
        file,
        format,
        vad_enabled: opts.vad_mode.is_some(),
        energy_vad: opts.vad_mode.as_deref() == Some("energy"),
        vad_model_path: opts.model_path.clone(),
        vad_threshold: opts.vad_threshold.unwrap_or(0.5) as f32,
        ort_library_path: opts.ort_library_path.clone(),
        sample_rate: opts.sample_rate.unwrap_or(16000),
    })
}

impl FileParts {
    fn into_handle(self) -> FileHandle {
        FileHandle {
            inner: Some(self.file),
            consumed: false,
            engaged: false,
            format: self.format,
            vad_enabled: self.vad_enabled,
            energy_vad: self.energy_vad,
            vad: None,
            vad_model_path: self.vad_model_path,
            vad_threshold: self.vad_threshold,
            ort_library_path: self.ort_library_path,
            vad_probability: 0.0,
            sample_rate: self.sample_rate,
        }
    }
}

/// Background task that performs the blocking file open work (disk read, WAV
/// parse, chain construction) on the libuv thread pool, off the JS event
/// loop, then resolves to a constructed handle on the JS thread.
pub struct OpenFileTask {
    path: String,
    options: Option<FileOptions>,
}

impl Task for OpenFileTask {
    type Output = FileParts;
    type JsValue = FileHandle;

    fn compute(&mut self) -> Result<Self::Output> {
        build_file_parts(&self.path, self.options.as_ref())
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(output.into_handle())
    }
}

/// Background task that consumes the source with the core's whole-recording
/// analysis on the libuv thread pool. A `File` whose source is already gone
/// (analyzed, or closed) rejects rather than silently re-reading; a `File`
/// that was read instead is refused earlier, by the engagement check.
pub struct AnalyzeFileTask {
    file: Option<CoreFile>,
}

impl Task for AnalyzeFileTask {
    type Output = CoreVadReport;
    type JsValue = VadReport;

    fn compute(&mut self) -> Result<Self::Output> {
        let file = self.file.take().ok_or_else(|| {
            Error::new(
                Status::InvalidArg,
                "File already consumed; construct a new File for another pass",
            )
        })?;
        file.analyze().map_err(to_napi_error)
    }

    fn resolve(&mut self, _env: Env, output: Self::Output) -> Result<Self::JsValue> {
        Ok(convert_report(output))
    }
}

/// Native offline-source handle exposed to Node.js via napi-rs. The public
/// `File` Readable lives in the JS wrapper; consumers construct that, not
/// this handle, directly.
#[napi]
pub struct FileHandle {
    /// The core offline source. `None` once consumed by analysis, closed, or
    /// fully read: a File is a single pass.
    inner: Option<CoreFile>,
    /// Set only when `analyze()` takes the source. Separates the analyzed
    /// state, where a later read fails loud, from natural exhaustion and
    /// `close()`, where a later read ends quietly.
    consumed: bool,
    /// Set once `read_chunk` has driven the source. Analysis is refused from
    /// then on, so a report can never describe only the unread remainder.
    engaged: bool,
    format: SampleFormat,
    vad_enabled: bool,
    energy_vad: bool,
    /// Lazily constructed per-chunk Silero detector (silero mode only), so an
    /// analysis-only File does not load the model twice.
    vad: Option<SileroVad>,
    vad_model_path: Option<String>,
    vad_threshold: f32,
    ort_library_path: Option<String>,
    /// Most recent per-chunk score, exposed via the `vadProbability` getter
    /// exactly as DecibriBridge exposes its own.
    vad_probability: f32,
    sample_rate: u32,
}

#[napi]
impl FileHandle {
    /// Open a WAV path as an offline source, synchronously (blocks on disk
    /// I/O; the JS wrapper's async `File.open` uses `openAsync` instead).
    #[napi(factory)]
    pub fn open(path: String, options: Option<FileOptions>) -> Result<FileHandle> {
        Ok(build_file_parts(&path, options.as_ref())?.into_handle())
    }

    /// Open a WAV path without blocking the JS event loop: the disk read,
    /// WAV parse, and chain construction run on the libuv thread pool.
    #[napi]
    pub fn open_async(path: String, options: Option<FileOptions>) -> AsyncTask<OpenFileTask> {
        AsyncTask::new(OpenFileTask { path, options })
    }

    /// Wrap in-memory samples as an offline source. `samples` are mono f32 in
    /// [-1.0, 1.0]; `inputRate` is their native rate (raw samples carry no
    /// header). No I/O, so construction is synchronous.
    #[napi(factory)]
    pub fn buffer(
        samples: Float32Array,
        input_rate: u32,
        options: Option<FileOptions>,
    ) -> Result<FileHandle> {
        let default_opts = FileOptions::default();
        let opts = options.as_ref().unwrap_or(&default_opts);
        let format = match opts.format.as_deref().unwrap_or("int16") {
            "int16" => SampleFormat::Int16,
            "float32" => SampleFormat::Float32,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "format must be 'int16' or 'float32'",
                ))
            }
        };
        let config = build_file_config(opts)?;
        let file = CoreFile::buffer(samples.to_vec(), input_rate, config).map_err(to_napi_error)?;
        Ok(FileParts {
            file,
            format,
            vad_enabled: opts.vad_mode.is_some(),
            energy_vad: opts.vad_mode.as_deref() == Some("energy"),
            vad_model_path: opts.model_path.clone(),
            vad_threshold: opts.vad_threshold.unwrap_or(0.5) as f32,
            ort_library_path: opts.ort_library_path.clone(),
            sample_rate: opts.sample_rate.unwrap_or(16000),
        }
        .into_handle())
    }

    /// Pull the next conditioned chunk, advancing the per-chunk VAD score on
    /// the pre-conditioning feed. Returns `null` once the source is fully
    /// delivered (after the end-of-stream tail) or already consumed.
    #[napi]
    pub fn read_chunk(&mut self) -> Result<Option<Buffer>> {
        // A File analyzed then iterated fails loud rather than yielding an
        // empty stream indistinguishable from a silent recording. Exhaustion
        // and close() leave `consumed` false and still read as ended.
        if self.consumed {
            return Err(Error::new(
                Status::InvalidArg,
                "File already consumed; construct a new File for another pass",
            ));
        }
        let Some(file) = self.inner.as_mut() else {
            return Ok(None);
        };
        // The cursor is about to move: analysis of this File is refused from
        // here on. Set before the pull, so a pull that delivers nothing still
        // counts.
        self.engaged = true;
        let chunk = match file.next() {
            None => {
                // Fully delivered: drop the source so held memory is released
                // as soon as the pass ends.
                self.inner = None;
                return Ok(None);
            }
            Some(Err(e)) => return Err(to_napi_error(e)),
            Some(Ok(chunk)) => chunk,
        };

        if self.vad_enabled {
            // The core hands out the pre-conditioning feed when it is
            // maintained (VAD configured, or a transform makes the delivered
            // output differ); on `None` the delivered chunk already is that
            // signal, the same fallback the live pump applies.
            let feed = file.vad_input();
            if self.energy_vad {
                let frame: &[f32] = match feed.as_deref() {
                    Some(f) => f,
                    None => &chunk.data,
                };
                if !frame.is_empty() {
                    self.vad_probability = sample::rms(frame);
                }
            } else if let Some(feed) = feed {
                if !feed.is_empty() {
                    if self.vad.is_none() {
                        let model_path = self.vad_model_path.as_deref().ok_or_else(|| {
                            Error::new(
                                Status::InvalidArg,
                                "modelPath is required when vadMode is 'silero'",
                            )
                        })?;
                        // `VadConfig` is `#[non_exhaustive]`: default-construct
                        // then assign. The rate the core runs detection at
                        // already accounts for the internal feed resample.
                        let mut vad_config = VadConfig::default();
                        vad_config.model_path = std::path::PathBuf::from(model_path);
                        vad_config.sample_rate = file.vad_rate().unwrap_or(16000);
                        vad_config.threshold = self.vad_threshold;
                        vad_config.ort_library_path = self
                            .ort_library_path
                            .as_deref()
                            .map(std::path::PathBuf::from);
                        self.vad = Some(SileroVad::new(vad_config).map_err(to_napi_error)?);
                    }
                    if let Some(vad) = self.vad.as_mut() {
                        let result = vad.process(&feed).map_err(to_napi_error)?;
                        self.vad_probability = result.probability;
                    }
                }
            }
        }

        let bytes = match self.format {
            SampleFormat::Int16 => sample::f32_to_i16_le_bytes(&chunk.data),
            SampleFormat::Float32 => sample::f32_to_f32_le_bytes(&chunk.data),
        };
        Ok(Some(bytes.into()))
    }

    /// Consume the source with the core's whole-recording analysis, off the
    /// JS event loop. Resolves to the `VadReport`; a `File` built without VAD
    /// rejects with the core's typed error, never a silently constructed
    /// detector.
    ///
    /// Everything checkable is checked before the source is taken, so a
    /// rejected call leaves the File exactly as it was and the caller can fix
    /// the configuration and retry.
    #[napi]
    pub fn analyze(&mut self) -> Result<AsyncTask<AnalyzeFileTask>> {
        if self.engaged {
            return Err(to_napi_error(decibri::error::DecibriError::FileEngaged));
        }
        // The detector configuration is read off the source itself, so this
        // check cannot drift from the one the core applies.
        if self.inner.as_ref().is_some_and(|f| f.vad_rate().is_none()) {
            return Err(to_napi_error(
                decibri::error::DecibriError::VadNotConfigured,
            ));
        }
        let file = self.inner.take();
        // The source is taken: a later iteration reads as consumed, not as an
        // empty recording, whatever the analysis resolves to.
        if file.is_some() {
            self.consumed = true;
        }
        Ok(AsyncTask::new(AnalyzeFileTask { file }))
    }

    /// Release the source. Idempotent; a closed File reads as ended.
    #[napi]
    pub fn close(&mut self) {
        self.inner = None;
    }

    /// Most recent per-chunk VAD score (0.0 to 1.0), computed on the
    /// pre-conditioning feed. 0.0 before the first chunk or with VAD off.
    #[napi(getter)]
    pub fn vad_probability(&self) -> f64 {
        f64::from(self.vad_probability)
    }

    /// The target output rate every delivered chunk carries.
    #[napi(getter)]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}
