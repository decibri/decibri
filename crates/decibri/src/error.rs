use std::path::PathBuf;

use thiserror::Error;

// Per-platform guidance for `PermissionDenied`. Attribute-gated so each
// platform's binary embeds only its own hint string. The attribute-gated
// pattern matches the crate's existing conditional-compilation idiom: 52
// `#[cfg(...)]` attribute sites across the crate (all feature-predicates),
// 0 `cfg!()` macro sites. The `not(any(...))` fallback covers BSDs and
// any future target-os values uncategorized at compile time.

#[cfg(target_os = "macos")]
const PERMISSION_HINT: &str = "Enable in System Preferences > Security & Privacy > Microphone.";

#[cfg(target_os = "windows")]
const PERMISSION_HINT: &str = "Enable in Settings > Privacy & Security > Microphone.";

#[cfg(target_os = "linux")]
const PERMISSION_HINT: &str =
    "Check your distribution's audio permissions (PulseAudio / PipeWire).";

#[cfg(not(any(target_os = "macos", target_os = "windows", target_os = "linux")))]
const PERMISSION_HINT: &str = "Check your system audio permissions.";

/// Errors produced by decibri operations.
///
/// This enum is `#[non_exhaustive]`: consumers pattern-matching on it must
/// include a `_ =>` catch-all arm so future variant additions are not
/// source-breaking.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum DecibriError {
    // ── Config validation ──────────────────────────────────────────────
    #[error("sample rate must be between 1000 and 384000")]
    SampleRateOutOfRange,

    #[error("channels must be between 1 and 32")]
    ChannelsOutOfRange,

    #[error("frames per buffer must be between 64 and 65536")]
    FramesPerBufferOutOfRange,

    #[error("format must be 'int16' or 'float32'")]
    InvalidFormat,

    // ── Device errors ──────────────────────────────────────────────────
    #[error("No audio input device found matching \"{0}\"")]
    DeviceNotFound(String),

    /// No output device matched a `Name` or `Id` selector.
    ///
    /// The split from [`Self::DeviceNotFound`] exists because the display
    /// message needs to say "output" when the lookup was against output
    /// devices, not "input". Issued by [`crate::device::resolve_output_device`]
    /// via the internal direction-specific `not_found_error` hook.
    #[error("No audio output device found matching \"{0}\"")]
    OutputDeviceNotFound(String),

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
    #[error("audio stream is already running. Call stop() first.")]
    AlreadyRunning,

    #[error("Failed to open audio stream: {0}")]
    StreamOpenFailed(String),

    #[error("Failed to start audio stream: {0}")]
    StreamStartFailed(String),

    #[error("Microphone permission denied. {}", PERMISSION_HINT)]
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
         Either pass ort_library_path when constructing the VAD, set \
         ORT_DYLIB_PATH to point to a valid ONNX Runtime library, or \
         enable the `ort-download-binaries` feature for zero-config builds."
    )]
    OrtInitFailed {
        #[source]
        source: ort::Error,
    },

    #[error(
        "decibri: failed to load ONNX Runtime from {}: {source}. \
         If ORT_DYLIB_PATH is set, verify it points to a valid ONNX Runtime \
         library for your platform. Otherwise the bundled ONNX Runtime may \
         be missing from your installation. Try reinstalling decibri.",
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
    /// ORT C API (`ortsys![CreateStatus]`), which is exactly the hang this
    /// pre-check is designed to prevent. Keeping this variant string-only
    /// means the pre-check never touches ORT symbols.
    ///
    /// Display message matches [`Self::OrtLoadFailed`] so Node, Python, and
    /// future FFI consumers see the same actionable hint regardless of which
    /// failure path was taken.
    #[error(
        "decibri: failed to load ONNX Runtime from {}: {reason}. \
         If ORT_DYLIB_PATH is set, verify it points to a valid ONNX Runtime \
         library for your platform. Otherwise the bundled ONNX Runtime may \
         be missing from your installation. Try reinstalling decibri.",
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

    /// Raised when ONNX Runtime is used in a child process after being
    /// initialized in the parent (Phase 9 Item C7).
    ///
    /// Python's `fork` start method on Linux duplicates the parent's
    /// memory into the child, but ONNX Runtime's internal state
    /// (allocators, thread pools, model graph state) is not safe to
    /// share across forked processes. Without proactive detection, a
    /// child that uses Silero VAD inherited via fork would silently
    /// produce wrong probabilities, segfault, or hang.
    ///
    /// `init_pid` is the pid that first initialized ORT (typically the
    /// parent that loaded `Microphone(vad="silero")` before fork);
    /// `current_pid` is the pid that called the inference path and
    /// triggered the detection.
    ///
    /// Remediation is in the user's hands: switch to `spawn` start
    /// method or construct `Microphone(vad="silero")` inside the child
    /// rather than the parent. The Display message embeds both options.
    #[error(
        "ONNX Runtime was initialized in pid {init_pid} but is being used in pid {current_pid}. \
         Python's fork() start method is incompatible with ORT initialization. Use \
         multiprocessing.set_start_method('spawn') or construct Microphone(vad='silero') \
         inside each child process."
    )]
    ForkAfterOrtInit { init_pid: u32, current_pid: u32 },

    /// Forward-compat catch-all for non-ORT ONNX backends (CoreML, TFLite,
    /// etc.) once the workspace split lands at 4.0.
    ///
    /// Not emitted by ORT-backed code paths in 3.x: ORT failures use the 8
    /// preceding variants (`OrtSessionBuildFailed`, `OrtThreadsConfigFailed`,
    /// `VadModelLoadFailed`, `OrtInferenceFailed`, `OrtTensorCreateFailed`,
    /// `OrtTensorExtractFailed`, plus `OrtInitFailed` and `OrtLoadFailed`)
    /// which carry an `ort::Error` source directly.
    ///
    /// Permitted by `#[non_exhaustive]` on this enum (see line 31). Existing
    /// `is_ort_path_error` returns false on this variant: `OnnxBackendFailed`
    /// is not an ORT path-loading failure.
    ///
    /// `backend` identifies which non-ORT backend produced the error
    /// (e.g. `"coreml"`, `"tflite"`); `source` carries the backend-native
    /// error boxed for trait-object compatibility.
    #[error("ONNX backend error from {backend}: {source}")]
    OnnxBackendFailed {
        backend: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
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
