use std::path::PathBuf;

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
    //
    // These variants carry a structured `ort::Error` via `#[source]` so
    // downstream consumers can walk `error.source()` to the underlying
    // ORT error for programmatic handling. Paths are `PathBuf` (not
    // `String`) so consumers retain path semantics without re-parsing.
    #[error(
        "decibri: failed to initialize ONNX Runtime: {source}. \
         Either pass ort_library_path in VadConfig, set ORT_DYLIB_PATH to \
         point to a valid ONNX Runtime library, or enable the \
         `ort-download-binaries` feature for zero-config builds."
    )]
    OrtInitFailed {
        #[source]
        source: ort::Error,
    },

    #[error(
        "decibri: failed to load ONNX Runtime from {}: {source}. \
         If ORT_DYLIB_PATH is set, verify it points to a valid ONNX Runtime \
         library for your platform. Otherwise the bundled ORT may be missing \
         from your platform package. Try reinstalling decibri.",
        path.display()
    )]
    OrtLoadFailed {
        path: PathBuf,
        #[source]
        source: ort::Error,
    },

    /// The `ort_library_path` in `VadConfig` failed a pre-check before it
    /// could be handed to `ort::init_from`. Used only by the Windows
    /// hang-prevention pre-check in `init_ort_once`.
    ///
    /// Intentionally does *not* carry an `ort::Error` source, because
    /// constructing an `ort::Error` under `ort-load-dynamic` calls into the
    /// ORT C API (`ortsys![CreateStatus]`) — which is exactly the hang this
    /// pre-check is designed to prevent. Keeping this variant string-only
    /// means the pre-check never touches ORT symbols.
    ///
    /// Display message matches [`Self::OrtLoadFailed`] so Node, Python, and
    /// future FFI consumers see the same actionable hint regardless of which
    /// failure path was taken.
    #[error(
        "decibri: failed to load ONNX Runtime from {}: {reason}. \
         If ORT_DYLIB_PATH is set, verify it points to a valid ONNX Runtime \
         library for your platform. Otherwise the bundled ORT may be missing \
         from your platform package. Try reinstalling decibri.",
        path.display()
    )]
    OrtPathInvalid { path: PathBuf, reason: &'static str },

    #[error("Failed to create ort session builder: {0}")]
    OrtSessionBuildFailed(#[source] ort::Error),

    #[error("Failed to set ort threads: {0}")]
    OrtThreadsConfigFailed(#[source] ort::Error),

    #[error("Failed to load Silero VAD model from {}: {source}", path.display())]
    VadModelLoadFailed {
        path: PathBuf,
        #[source]
        source: ort::Error,
    },

    #[error("Silero VAD inference failed: {0}")]
    OrtInferenceFailed(#[source] ort::Error),

    #[error("Failed to create {kind} tensor: {source}")]
    OrtTensorCreateFailed {
        kind: &'static str,
        #[source]
        source: ort::Error,
    },

    #[error("Failed to extract {kind} tensor: {source}")]
    OrtTensorExtractFailed {
        kind: &'static str,
        #[source]
        source: ort::Error,
    },
}

impl DecibriError {
    /// Returns true if this error represents a failure to use a specific
    /// ORT library path.
    ///
    /// Consumers handling "the `ort_library_path` is wrong" logic should
    /// match on this rather than enumerating [`Self::OrtLoadFailed`] and
    /// [`Self::OrtPathInvalid`] separately:
    ///
    /// - [`Self::OrtLoadFailed`] fires when ORT tried to load the path and
    ///   failed (e.g. wrong ORT version, corrupted dylib).
    /// - [`Self::OrtPathInvalid`] fires when decibri's filesystem pre-check
    ///   rejected the path before ORT saw it (nonexistent, directory, etc).
    ///
    /// Both represent the same user-facing failure mode: "this path cannot
    /// be used to load ORT." The split is a mechanical necessity (see the
    /// rustdoc on [`Self::OrtPathInvalid`]), not a categorization users
    /// need to care about.
    pub fn is_ort_path_error(&self) -> bool {
        matches!(
            self,
            Self::OrtLoadFailed { .. } | Self::OrtPathInvalid { .. }
        )
    }
}
