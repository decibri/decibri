#[cfg(any(feature = "vad", feature = "denoise"))]
use std::path::PathBuf;

use thiserror::Error;

// Per-platform guidance for `PermissionDenied`. Attribute-gated so each
// platform's binary embeds only its own hint string. The attribute-gated
// pattern matches the crate's existing conditional-compilation idiom: 52
// `#[cfg(...)]` attribute sites across the crate (all feature-predicates),
// 0 `cfg!()` macro sites. The `not(any(...))` fallback covers BSDs and
// any future target-os values uncategorized at compile time.

#[cfg(target_os = "macos")]
const PERMISSION_HINT: &str = "Enable in System Settings > Privacy & Security > Microphone.";

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

    /// The `agc` target level fell outside the supported dBFS range.
    ///
    /// `agc` is `Option<i8>` on [`crate::microphone::MicrophoneConfig`], so a
    /// representably-invalid target (a value outside `-40..=-3`) can reach the
    /// core directly from a Rust consumer that bypasses the bindings. This is the
    /// load-bearing backstop: [`crate::microphone::MicrophoneConfig::validate`]
    /// returns it rather than clamping, matching how `sample_rate` is range
    /// checked. Static message to keep the text stable.
    #[error("agc target level must be between -40 and -3")]
    AgcTargetOutOfRange,

    /// The `limiter` ceiling fell outside the supported dBFS range.
    ///
    /// `limiter` is `Option<f32>` on [`crate::microphone::MicrophoneConfig`], so
    /// a representably-invalid ceiling (a value outside `-3.0..=0.0`) can reach
    /// the core directly from a Rust consumer that bypasses the bindings. This is
    /// the load-bearing backstop: [`crate::microphone::MicrophoneConfig::validate`]
    /// returns it rather than clamping, matching how `agc` is range checked. A
    /// distinct variant from [`Self::AgcTargetOutOfRange`] because the parameter,
    /// range, and message differ. Static message to keep the text stable.
    #[error("limiter ceiling must be between -3.0 and 0.0")]
    LimiterCeilingOutOfRange,

    #[error("format must be 'int16' or 'float32'")]
    InvalidFormat,

    // ── Device errors ──────────────────────────────────────────────────
    #[error("No microphone found matching \"{0}\"")]
    MicrophoneNotFound(String),

    /// No output device matched a `Name` or `Id` selector.
    ///
    /// Distinct from [`Self::MicrophoneNotFound`] so the display message names
    /// the speaker when the lookup was against output devices. Issued when a
    /// `Name` or `Id` selector matches no output device.
    #[error("No speaker found matching \"{0}\"")]
    SpeakerNotFound(String),

    #[error("Multiple devices match \"{name}\":\n{matches}")]
    MultipleDevicesMatch { name: String, matches: String },

    #[error("device index out of range. Call devices() to list available devices")]
    DeviceIndexOutOfRange,

    #[error("No microphone found. Check system audio input settings.")]
    NoMicrophoneFound,

    #[error("No speaker found. Check system audio settings.")]
    NoSpeakerFound,

    #[error("Selected device is not a valid microphone.")]
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

    /// Microphone stream has closed (stopped explicitly or by driver error).
    /// Returned from `MicrophoneStream::try_next_chunk` / `next_chunk` when the
    /// underlying channel has disconnected.
    #[error("Microphone stream is closed")]
    MicrophoneStreamClosed,

    /// Speaker stream has closed. Returned from `SpeakerStream::send` when the
    /// underlying channel has disconnected.
    #[error("Speaker stream is closed")]
    SpeakerStreamClosed,

    /// An active audio stream failed at the device or driver level while
    /// running (device unplugged, driver reset, exclusive-mode preemption).
    ///
    /// Distinct from [`Self::StreamOpenFailed`] / [`Self::StreamStartFailed`],
    /// which fire at open/start time: this is reported by the cpal error
    /// callback during streaming. The underlying `cpal::StreamError` is carried
    /// boxed as a `#[source]` (the same pattern as [`Self::OnnxBackendFailed`]),
    /// so a consumer can walk `error.source()` (and downcast to
    /// `cpal::StreamError`) to distinguish a cause such as device-not-available
    /// from a backend-specific driver error, with no cpal type appearing in this
    /// enum's public signature. After it fires the stream is treated as closed;
    /// a consumer that sees `MicrophoneStreamClosed` / `SpeakerStreamClosed` can
    /// call `take_last_error()` on the stream to retrieve this cause. Additive
    /// variant permitted by `#[non_exhaustive]`.
    #[error("decibri: audio device error: {source}")]
    DeviceFailed {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    // ── Resample config validation ─────────────────────────────────────
    /// The capture resampler could not be constructed for the negotiated
    /// device-native to requested-target sample rate conversion.
    ///
    /// The payload carries the offending rates for programmatic inspection;
    /// they are not formatted into the Display string, so the message text
    /// stays stable (matching [`Self::VadSampleRateUnsupported`]). Unreachable
    /// for a configured rate in the validated `1000..=384000` range over any
    /// device-native rate (every such pair is far below the resampler's filter
    /// cap); surfaced defensively. Additive variant permitted by
    /// `#[non_exhaustive]`.
    #[cfg(feature = "capture")]
    #[error("the requested sample rate conversion is not supported by the resampler")]
    ResampleConfigInvalid { in_rate: u32, out_rate: u32 },

    // ── VAD config validation ──────────────────────────────────────────
    /// Payload carries the offending sample rate; not formatted into the
    /// Display string to keep the message text stable.
    #[error("Silero VAD only supports sample rates 8000 and 16000")]
    VadSampleRateUnsupported(u32),

    /// Payload carries the offending threshold; not formatted into the
    /// Display string to keep the message text stable.
    #[error("VAD threshold must be between 0.0 and 1.0")]
    VadThresholdOutOfRange(f32),

    // ── ORT and VAD model errors ───────────────────────────────────────
    //
    // These variants carry a structured `ort::Error` via `#[source]` so
    // downstream consumers can walk `error.source()` to the underlying
    // ORT error for programmatic handling. Paths are `PathBuf` (not
    // `String`) so consumers retain path semantics without re-parsing.
    #[cfg(any(feature = "vad", feature = "denoise"))]
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

    #[cfg(any(feature = "vad", feature = "denoise"))]
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
    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error(
        "decibri: failed to load ONNX Runtime from {}: {reason}. \
         If ORT_DYLIB_PATH is set, verify it points to a valid ONNX Runtime \
         library for your platform. Otherwise the bundled ONNX Runtime may \
         be missing from your installation. Try reinstalling decibri.",
        path.display()
    )]
    OrtPathInvalid { path: PathBuf, reason: &'static str },

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to create ort session builder: {0}")]
    OrtSessionBuildFailed(#[source] ort::Error),

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to set ort threads: {0}")]
    OrtThreadsConfigFailed(#[source] ort::Error),

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to load Silero VAD model from {}: {source}", path.display())]
    VadModelLoadFailed {
        path: PathBuf,
        #[source]
        source: ort::Error,
    },

    /// A bundled model file failed to load through the shared ONNX session seam.
    ///
    /// Model-agnostic counterpart to [`Self::VadModelLoadFailed`]: the capture
    /// denoise stage surfaces this when its model file cannot be opened, so a
    /// denoise load failure is not reported as a VAD error. Carries the
    /// offending path and the underlying `ort::Error` via `#[source]`, matching
    /// the other model-load variants. Additive variant permitted by
    /// `#[non_exhaustive]`.
    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to load model from {}: {source}", path.display())]
    ModelLoadFailed {
        path: PathBuf,
        #[source]
        source: ort::Error,
    },

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Silero VAD inference failed: {0}")]
    OrtInferenceFailed(#[source] ort::Error),

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to create {kind} tensor: {source}")]
    OrtTensorCreateFailed {
        kind: &'static str,
        #[source]
        source: ort::Error,
    },

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to extract {kind} tensor: {source}")]
    OrtTensorExtractFailed {
        kind: &'static str,
        #[source]
        source: ort::Error,
    },

    /// Raised when ONNX Runtime is used in a child process after being
    /// initialized in the parent.
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

    /// Reserved catch-all for non-ORT ONNX backends.
    ///
    /// Not emitted by ORT-backed code paths: ORT failures use the eight
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

/// Bridge the capture resampler's construction error into the central error.
///
/// Both `ResamplerError` variants are construction-time; the resampler's steady
/// `process` path is infallible, so there is no per-chunk error to map.
/// `RatePairUnsupported` carries the offending rates through; `ZeroSampleRate`
/// (guarded: the configured target is validated nonzero and the device-native
/// rate is nonzero) and any future variant route to the same error defensively.
#[cfg(feature = "capture")]
impl From<decibri_resampler::ResamplerError> for DecibriError {
    fn from(e: decibri_resampler::ResamplerError) -> Self {
        use decibri_resampler::ResamplerError;
        match e {
            ResamplerError::RatePairUnsupported { in_rate, out_rate } => {
                DecibriError::ResampleConfigInvalid { in_rate, out_rate }
            }
            _ => DecibriError::ResampleConfigInvalid {
                in_rate: 0,
                out_rate: 0,
            },
        }
    }
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
        #[cfg(any(feature = "vad", feature = "denoise"))]
        {
            matches!(
                self,
                Self::OrtLoadFailed { .. } | Self::OrtPathInvalid { .. }
            )
        }
        #[cfg(not(any(feature = "vad", feature = "denoise")))]
        {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error as _;

    #[test]
    fn device_failed_carries_structured_source() {
        let inner = std::io::Error::other("device gone");
        let err = DecibriError::DeviceFailed {
            source: Box::new(inner),
        };
        // Display renders the source's message in the frozen style.
        assert_eq!(err.to_string(), "decibri: audio device error: device gone");
        // The cause is structured and walkable, not merely stringified.
        assert!(err.source().is_some());
    }

    /// The macOS permission hint uses the modern System Settings wording
    /// ("System Settings > Privacy & Security", not the pre-Ventura "System
    /// Preferences > Security & Privacy"). Frozen public API text.
    #[cfg(target_os = "macos")]
    #[test]
    fn macos_permission_hint_uses_modern_wording() {
        assert_eq!(
            PERMISSION_HINT,
            "Enable in System Settings > Privacy & Security > Microphone."
        );
    }

    /// The `PermissionDenied` Display message keeps its frozen leading prefix
    /// regardless of platform, so binding-layer prefix classification stays
    /// stable independent of the per-platform hint body.
    #[test]
    fn permission_denied_message_prefix_is_frozen() {
        let msg = DecibriError::PermissionDenied.to_string();
        assert!(
            msg.starts_with("Microphone permission denied. "),
            "frozen prefix must not drift: {msg}"
        );
    }
}
