use thiserror::Error;

/// Errors produced by decibri operations.
///
/// This enum is `#[non_exhaustive]`: consumers pattern-matching on it must
/// include a `_ =>` catch-all arm so future variant additions are not
/// source-breaking.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecibriError {
    // ── Config validation ──────────────────────────────────────────────
    #[error("sampleRate must be between 1000 and 384000")]
    SampleRateOutOfRange,

    #[error("channels must be between 1 and 32")]
    ChannelsOutOfRange,

    #[error("framesPerBuffer must be between 64 and 65536")]
    FramesPerBufferOutOfRange,

    #[error("format must be 'int16' or 'float32'")]
    InvalidFormat,

    // ── Device errors ──────────────────────────────────────────────────
    #[error("No audio input device found matching \"{0}\"")]
    DeviceNotFound(String),

    #[error("Multiple devices match \"{name}\":\n{matches}")]
    MultipleDevicesMatch { name: String, matches: String },

    #[error("device index out of range. Call Decibri.devices() to list available devices")]
    DeviceIndexOutOfRange,

    #[error("No microphone found. Check system audio input settings.")]
    NoMicrophoneFound,

    #[error("No audio output device found. Check system audio settings.")]
    NoOutputDeviceFound,

    #[error("Selected device is not a valid input device.")]
    NotAnInputDevice,

    #[error("Failed to enumerate devices: {0}")]
    DeviceEnumerationFailed(String),

    // ── Stream errors ──────────────────────────────────────────────────
    #[error("Decibri is already running. Call stop() first.")]
    AlreadyRunning,

    #[error("Failed to open audio stream: {0}")]
    StreamOpenFailed(String),

    #[error("Failed to start audio stream: {0}")]
    StreamStartFailed(String),

    #[error("Microphone permission denied. Enable in System Preferences > Security & Privacy.")]
    PermissionDenied,

    /// Capture stream has closed (stopped explicitly or by driver error).
    /// Returned from `CaptureStream::try_next_chunk` / `next_chunk` when the
    /// underlying channel has disconnected.
    #[error("Capture stream is closed")]
    CaptureStreamClosed,

    /// Output stream has closed. Returned from `OutputStream::send` when the
    /// underlying channel has disconnected.
    #[error("Output stream closed")]
    OutputStreamClosed,

    // ── VAD config validation ──────────────────────────────────────────
    /// Payload carries the offending sample rate; not formatted into the
    /// Display string to preserve the 3.1.x message text.
    #[error("Silero VAD only supports sample rates 8000 and 16000")]
    VadSampleRateUnsupported(u32),

    /// Payload carries the offending threshold; not formatted into the
    /// Display string to preserve the 3.1.x message text.
    #[error("VAD threshold must be between 0.0 and 1.0")]
    VadThresholdOutOfRange(f32),

    // ── ORT and VAD model errors ───────────────────────────────────────
    #[error(
        "decibri: failed to initialize ONNX Runtime: {message}. \
         Either pass ort_library_path in VadConfig, set ORT_DYLIB_PATH to \
         point to a valid ONNX Runtime library, or enable the \
         `ort-download-binaries` feature for zero-config builds."
    )]
    OrtInitFailed { message: String },

    #[error(
        "decibri: failed to load ONNX Runtime from {path}: {message}. \
         If ORT_DYLIB_PATH is set, verify it points to a valid ONNX Runtime \
         library for your platform. Otherwise the bundled ORT may be missing \
         from your platform package. Try reinstalling decibri."
    )]
    OrtLoadFailed { path: String, message: String },

    #[error("Failed to create ort session builder: {0}")]
    OrtSessionBuildFailed(String),

    #[error("Failed to set ort threads: {0}")]
    OrtThreadsConfigFailed(String),

    #[error("Failed to load Silero VAD model from {path}: {message}")]
    VadModelLoadFailed { path: String, message: String },

    #[error("Silero VAD inference failed: {0}")]
    OrtInferenceFailed(String),

    #[error("Failed to create {kind} tensor: {message}")]
    OrtTensorCreateFailed { kind: &'static str, message: String },

    #[error("Failed to extract {kind} tensor: {message}")]
    OrtTensorExtractFailed { kind: &'static str, message: String },
}
