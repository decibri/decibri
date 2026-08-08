#[cfg(any(feature = "vad", feature = "denoise", feature = "capture"))]
use std::path::PathBuf;

use thiserror::Error;

// Per-platform guidance for `PermissionDenied`. Attribute-gated so each
// platform's binary embeds only its own hint string. The attribute-gated
// pattern matches the crate's existing conditional-compilation idiom: 203
// `#[cfg(...)]` attribute sites across the crate, 0 `cfg!()` macro sites.
// The `not(any(...))` fallback covers BSDs and any future target-os values
// uncategorized at compile time.

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

    /// A configuration requested a channel count below one.
    ///
    /// Reached from both surfaces by a zero channel count:
    /// [`crate::microphone::MicrophoneConfig::validate`] and
    /// [`crate::speaker::SpeakerConfig::validate`] each return it for `channels
    /// == 0`. Neither surface has an upper bound in its own validation: capture
    /// is mono only and reports [`Self::MultichannelNotSupported`] above one,
    /// and playback leaves the maximum to the device, which answers when the
    /// stream is opened ([`Self::SpeakerChannelsUnsupported`]). The message
    /// names the floor only, because no upper bound is enforced anywhere. It is
    /// an exact-match string that downstream consumers branch on, so it is
    /// frozen at the text below.
    #[error("channels must be at least 1")]
    ChannelsOutOfRange,

    /// A microphone capture configuration requested more than one channel.
    ///
    /// Microphone capture is mono only:
    /// [`crate::microphone::MicrophoneConfig::validate`] rejects `channels > 1`
    /// rather than silently downmixing it to mono. The `channels` field is
    /// retained, so honouring `channels > 1` later (by delivering true
    /// interleaved multichannel) stays an additive change: the accepted set
    /// widens from `{1}` outward, breaking no caller. A dedicated variant
    /// rather than reusing [`Self::ChannelsOutOfRange`] reads more
    /// intentionally and signals the mono-only constraint explicitly; a zero
    /// channel count remains [`Self::ChannelsOutOfRange`]. The speaker path is
    /// unaffected: output may be multichannel, and the count it accepts is the
    /// device's to state, reported through
    /// [`Self::SpeakerChannelsUnsupported`]. Static message to keep the text
    /// stable.
    #[error("multichannel capture is not supported; channels must be 1 (mono)")]
    MultichannelNotSupported,

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

    /// The `compression` level for a FLAC save fell outside the encoder's
    /// 0 to 8 range.
    ///
    /// `compression` is `Option<u8>` on `SaveOptions`, so a representably
    /// invalid level can reach the core directly from a Rust consumer that
    /// bypasses the bindings. This is the load-bearing backstop: `File::save`
    /// returns it rather than clamping, matching how `agc` and `limiter` are
    /// range checked. Checked before the writer runs, so the writer's own
    /// rejection of the same level is unreachable. Static message to keep the
    /// text stable. Additive variant permitted by `#[non_exhaustive]`.
    #[error("flac compression level must be between 0 and 8")]
    FlacCompressionOutOfRange,

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

    /// An output stream could not be opened at the requested channel count, and
    /// the count is above what the device reports supporting.
    ///
    /// `requested` is the count [`crate::speaker::SpeakerConfig`] carried;
    /// `available` is the device's own figure, the same one
    /// [`crate::device::SpeakerInfo::max_output_channels`] reports; `reason` is
    /// the platform's message, formatted into the Display string so its text
    /// reaches the consumer (matching [`Self::AecConfigInvalid`]; the leading
    /// `the output device does not support` clause is the stable part).
    ///
    /// Narrower than [`Self::StreamOpenFailed`], which stays the report for an
    /// output open that failed for any other reason, including one at or below
    /// the device's figure. A device serves more channels than it reports
    /// whenever the host mixes and converts for it, so the figure is what the
    /// device states rather than a bound decibri enforces: nothing is refused
    /// ahead of the device. Additive variant permitted by `#[non_exhaustive]`.
    #[error("the output device does not support {requested} output channels; it reports {available}: {reason}")]
    SpeakerChannelsUnsupported {
        requested: u16,
        available: u16,
        reason: String,
    },

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

    // ── Resample errors ────────────────────────────────────────────────
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

    /// The resample chain was fed audio after it was flushed.
    ///
    /// Raised on both the capture path and the `File` detector-feed path,
    /// which share the resample bridge. Both paths stop feeding a flushed
    /// chain, so this is surfaced defensively. Static message to keep the
    /// text stable. Deliberately not feature-gated, so the identity table's
    /// coverage stays unconditional. Additive variant permitted by
    /// `#[non_exhaustive]`.
    #[error("the resample chain was fed after it was flushed")]
    ResampleAfterFlush,

    /// The resampler reported an error decibri does not recognise.
    ///
    /// `reason` is the resampler's own message, formatted into the Display
    /// string so the upstream text reaches the consumer (matching
    /// [`Self::AecConfigInvalid`]; the leading `resampler error:` prefix is
    /// the stable part). Every error the pinned resampler release defines maps to
    /// its own variant, so this is reached only by an error added in a later
    /// resampler release; surfaced defensively. Deliberately not
    /// feature-gated, so the identity table's coverage stays unconditional.
    /// Additive variant permitted by `#[non_exhaustive]`.
    #[error("resampler error: {reason}")]
    ResampleFailed { reason: String },

    // ── VAD config validation ──────────────────────────────────────────
    /// Payload carries the offending sample rate; not formatted into the
    /// Display string to keep the message text stable.
    #[error("Silero VAD only supports sample rates 8000 and 16000")]
    VadSampleRateUnsupported(u32),

    /// Payload carries the offending threshold; not formatted into the
    /// Display string to keep the message text stable.
    #[error("VAD threshold must be between 0.0 and 1.0")]
    VadThresholdOutOfRange(f32),

    // ── Echo-cancellation config validation ────────────────────────────
    /// Echo cancellation was requested with a capture sample rate outside the
    /// window the canceller supports.
    ///
    /// decibri's own accepted range (`1000..=384000`) is wider at both ends
    /// than the canceller's `8000..=48000`, so this rejects a rate that is
    /// valid for a capture without echo cancellation. Payload carries the
    /// offending rate; not formatted into the Display string to keep the
    /// message text stable (matching [`Self::VadSampleRateUnsupported`]).
    /// Deliberately not feature-gated, so the identity table's coverage stays
    /// unconditional. Additive variant permitted by `#[non_exhaustive]`.
    #[error("echo cancellation only supports sample rates 8000 to 48000")]
    AecSampleRateUnsupported(u32),

    /// The echo canceller rejected its configuration at construction.
    ///
    /// `reason` is the canceller's own message, formatted into the Display
    /// string so the upstream text reaches the consumer (matching
    /// [`Self::ResampleFailed`]; the leading `echo canceller configuration
    /// error:` prefix is the stable part). It carries the canceller's
    /// filter-tail, echo-delay and search-delay window rejections, its
    /// unknown-model rejection, and any error a later canceller release adds.
    /// [`crate::microphone::MicrophoneConfig::validate`] also returns it, with
    /// a decibri-authored reason behind the same prefix, for a declared
    /// reference channel count of 0. Deliberately not feature-gated, so the
    /// identity table's coverage stays unconditional. Additive variant
    /// permitted by `#[non_exhaustive]`.
    #[error("echo canceller configuration error: {reason}")]
    AecConfigInvalid { reason: String },

    // ── Offline file source errors ─────────────────────────────────────
    /// An offline audio file could not be read from disk.
    ///
    /// Carries the offending path and the underlying I/O failure boxed via
    /// `#[source]`, so a consumer can distinguish a missing file from a
    /// permission failure by walking `error.source()`. Additive variant
    /// permitted by `#[non_exhaustive]`.
    #[cfg(feature = "capture")]
    #[error("Failed to read audio file {}: {source}", path.display())]
    FileReadFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// An encoded audio file could not be written to disk.
    ///
    /// The write-side twin of [`Self::FileReadFailed`]: carries the offending
    /// path and the underlying I/O failure boxed via `#[source]`, so a
    /// consumer can distinguish a missing directory from a permission failure
    /// or a full disk by walking `error.source()`. Reported by `File::save`
    /// after the audio is encoded, so the encoded bytes were sound and the
    /// failure is the filesystem's. Additive variant permitted by
    /// `#[non_exhaustive]`.
    #[cfg(feature = "capture")]
    #[error("Failed to write audio file {}: {source}", path.display())]
    FileWriteFailed {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    /// An offline audio file is in a format decibri cannot decode.
    ///
    /// Covers a container that was not recognised, a container naming a codec
    /// the reader does not carry, a codec at a sample width it does not carry,
    /// and a channel layout it cannot decode. `reason` is the reader's own
    /// message, which names the tag, four-CC or width in question, formatted
    /// into the Display string so that text reaches the consumer (matching
    /// [`Self::AecConfigInvalid`]; the leading `unsupported audio format:`
    /// prefix is the stable part).
    ///
    /// Also where a reader failure decibri has no arm for lands, so a reader
    /// release that adds one reports what decibri can be sure of rather than
    /// naming a cause it cannot know. Deliberately not feature-gated, so the
    /// identity table's coverage stays unconditional. Additive variant
    /// permitted by `#[non_exhaustive]`.
    #[error("unsupported audio format: {reason}")]
    AudioFormatUnsupported { reason: String },

    /// An offline audio file is structurally wrong.
    ///
    /// The container was identified and parsed up to the point where the bytes
    /// were not what the format requires there. `reason` is the reader's own
    /// message, which names the byte offset and what was expected at it,
    /// formatted into the Display string (the leading `malformed audio file:`
    /// prefix is the stable part). Deliberately not feature-gated, so the
    /// identity table's coverage stays unconditional. Additive variant
    /// permitted by `#[non_exhaustive]`.
    #[error("malformed audio file: {reason}")]
    AudioFileMalformed { reason: String },

    /// An offline audio file ends before the audio it declares.
    ///
    /// Raised for a file shorter than its own headers state, including one
    /// whose declared payload length is not a whole number of frames.
    /// `reason` is the reader's own message, which names what was needed
    /// against what was available, formatted into the Display string (the
    /// leading `truncated audio file:` prefix is the stable part).
    /// Deliberately not feature-gated, so the identity table's coverage stays
    /// unconditional. Additive variant permitted by `#[non_exhaustive]`.
    #[error("truncated audio file: {reason}")]
    AudioFileTruncated { reason: String },

    /// Whole-recording analysis was requested on a source built without a
    /// voice-activity detection configuration.
    ///
    /// Analysis never constructs a default detector silently; the caller
    /// opts in at construction. Static message to keep the text stable.
    /// Additive variant permitted by `#[non_exhaustive]`.
    #[error("analysis requires VAD; construct the File with a vad configuration")]
    VadNotConfigured,

    /// Whole-recording analysis was requested on a source whose iteration has
    /// already advanced the read cursor.
    ///
    /// Analysis reports window and segment times from the start of the
    /// recording, so it runs only on a source that is still at its start.
    /// Static message to keep the text stable. Additive variant permitted by
    /// `#[non_exhaustive]`.
    #[error("File iteration has begun; construct a new File to analyze the whole recording")]
    FileEngaged,

    // ── ORT and VAD model errors ───────────────────────────────────────
    //
    // These variants carry the underlying ORT failure boxed as a
    // `Box<dyn std::error::Error + Send + Sync>` via `#[source]` (the same
    // pattern as `DeviceFailed` keeps cpal out of its signature), so
    // downstream consumers can walk `error.source()` to the cause for
    // programmatic handling with no concrete `ort::Error` type appearing in
    // this enum's public signature. Paths are `PathBuf` (not `String`) so
    // consumers retain path semantics without re-parsing.
    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error(
        "decibri: failed to initialize ONNX Runtime: {source}. \
         Either pass ort_library_path when constructing the VAD, set \
         ORT_DYLIB_PATH to point to a valid ONNX Runtime library, or \
         enable the `ort-download-binaries` feature for zero-config builds."
    )]
    OrtInitFailed {
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
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
        source: Box<dyn std::error::Error + Send + Sync>,
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
    OrtSessionBuildFailed(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to set ort threads: {0}")]
    OrtThreadsConfigFailed(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to load Silero VAD model from {}: {source}", path.display())]
    VadModelLoadFailed {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// A bundled model file failed to load through the shared ONNX session seam.
    ///
    /// Model-agnostic counterpart to [`Self::VadModelLoadFailed`]: the capture
    /// denoise stage surfaces this when its model file cannot be opened, so a
    /// denoise load failure is not reported as a VAD error. Carries the
    /// offending path and the underlying error boxed via `#[source]`, matching
    /// the other model-load variants. Additive variant permitted by
    /// `#[non_exhaustive]`.
    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to load model from {}: {source}", path.display())]
    ModelLoadFailed {
        path: PathBuf,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Silero VAD inference failed: {0}")]
    OrtInferenceFailed(#[source] Box<dyn std::error::Error + Send + Sync>),

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to create {kind} tensor: {source}")]
    OrtTensorCreateFailed {
        kind: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[error("Failed to extract {kind} tensor: {source}")]
    OrtTensorExtractFailed {
        kind: &'static str,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
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
    /// which carry the underlying ORT failure boxed via `#[source]`.
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

// ── Error identity ─────────────────────────────────────────────────────────
//
// One entry per variant drives three things: the variant's own name, its
// stable code, and one representative instance for the catalog. The generated
// `code` and `variant_name` matches carry no catch-all arm, so a variant added
// to the enum without an entry here fails to compile inside this crate.
// `#[non_exhaustive]` suppresses exhaustiveness checking for downstream crates
// only; a mapping written outside this crate gets no such protection.

/// A representative `#[source]` payload for a catalog instance.
fn sample_source() -> Box<dyn std::error::Error + Send + Sync> {
    Box::new(std::io::Error::other("sample"))
}

macro_rules! error_identity {
    ($(
        $( #[$attr:meta] )*
        $pattern:pat => $name:literal, $code:literal, $sample:expr ;
    )*) => {
        impl DecibriError {
            /// The variant's own name.
            ///
            /// The Python binding's exception classes are named identically,
            /// so this is the name of the class a Python consumer catches.
            pub fn variant_name(&self) -> &'static str {
                match self {
                    $( $( #[$attr] )* $pattern => $name, )*
                }
            }

            /// Stable machine-readable identity for this failure, shared by
            /// every binding.
            ///
            /// SCREAMING_SNAKE of the variant name, with one exception:
            /// [`Self::OrtPathInvalid`] reports `ORT_LOAD_FAILED`, the same
            /// code as [`Self::OrtLoadFailed`], because the two are one
            /// user-facing failure (see [`Self::is_ort_path_error`]).
            ///
            /// These values are a frozen surface. Node exposes them as
            /// `err.code`; Python exposes the same identity as the exception
            /// class name.
            pub fn code(&self) -> &'static str {
                match self {
                    $( $( #[$attr] )* $pattern => $code, )*
                }
            }
        }

        // An array literal cannot carry a per-element `cfg`, so the entries
        // are pushed one at a time into a caller-owned vector.
        fn fill_error_catalog(catalog: &mut Vec<DecibriError>) {
            $( $( #[$attr] )* catalog.push($sample); )*
        }

        /// One representative instance of every variant compiled into this
        /// build, for binding-side coverage tests.
        ///
        /// Test support, not a consumer API: the instances carry placeholder
        /// payloads and the order is the declaration order of the enum.
        #[doc(hidden)]
        pub fn error_catalog() -> Vec<DecibriError> {
            let mut catalog = Vec::new();
            fill_error_catalog(&mut catalog);
            catalog
        }
    };
}

error_identity! {
    DecibriError::SampleRateOutOfRange => "SampleRateOutOfRange", "SAMPLE_RATE_OUT_OF_RANGE",
        DecibriError::SampleRateOutOfRange;
    DecibriError::ChannelsOutOfRange => "ChannelsOutOfRange", "CHANNELS_OUT_OF_RANGE",
        DecibriError::ChannelsOutOfRange;
    DecibriError::MultichannelNotSupported => "MultichannelNotSupported", "MULTICHANNEL_NOT_SUPPORTED",
        DecibriError::MultichannelNotSupported;
    DecibriError::FramesPerBufferOutOfRange => "FramesPerBufferOutOfRange", "FRAMES_PER_BUFFER_OUT_OF_RANGE",
        DecibriError::FramesPerBufferOutOfRange;
    DecibriError::AgcTargetOutOfRange => "AgcTargetOutOfRange", "AGC_TARGET_OUT_OF_RANGE",
        DecibriError::AgcTargetOutOfRange;
    DecibriError::LimiterCeilingOutOfRange => "LimiterCeilingOutOfRange", "LIMITER_CEILING_OUT_OF_RANGE",
        DecibriError::LimiterCeilingOutOfRange;
    DecibriError::FlacCompressionOutOfRange => "FlacCompressionOutOfRange", "FLAC_COMPRESSION_OUT_OF_RANGE",
        DecibriError::FlacCompressionOutOfRange;
    DecibriError::InvalidFormat => "InvalidFormat", "INVALID_FORMAT",
        DecibriError::InvalidFormat;

    DecibriError::MicrophoneNotFound(_) => "MicrophoneNotFound", "MICROPHONE_NOT_FOUND",
        DecibriError::MicrophoneNotFound("sample".to_string());
    DecibriError::SpeakerNotFound(_) => "SpeakerNotFound", "SPEAKER_NOT_FOUND",
        DecibriError::SpeakerNotFound("sample".to_string());
    DecibriError::MultipleDevicesMatch { .. } => "MultipleDevicesMatch", "MULTIPLE_DEVICES_MATCH",
        DecibriError::MultipleDevicesMatch {
            name: "sample".to_string(),
            matches: "sample".to_string(),
        };
    DecibriError::DeviceIndexOutOfRange => "DeviceIndexOutOfRange", "DEVICE_INDEX_OUT_OF_RANGE",
        DecibriError::DeviceIndexOutOfRange;
    DecibriError::NoMicrophoneFound => "NoMicrophoneFound", "NO_MICROPHONE_FOUND",
        DecibriError::NoMicrophoneFound;
    DecibriError::NoSpeakerFound => "NoSpeakerFound", "NO_SPEAKER_FOUND",
        DecibriError::NoSpeakerFound;
    DecibriError::NotAnInputDevice => "NotAnInputDevice", "NOT_AN_INPUT_DEVICE",
        DecibriError::NotAnInputDevice;
    DecibriError::DeviceEnumerationFailed(_) => "DeviceEnumerationFailed", "DEVICE_ENUMERATION_FAILED",
        DecibriError::DeviceEnumerationFailed("sample".to_string());

    DecibriError::AlreadyRunning => "AlreadyRunning", "ALREADY_RUNNING",
        DecibriError::AlreadyRunning;
    DecibriError::StreamOpenFailed(_) => "StreamOpenFailed", "STREAM_OPEN_FAILED",
        DecibriError::StreamOpenFailed("sample".to_string());
    DecibriError::StreamStartFailed(_) => "StreamStartFailed", "STREAM_START_FAILED",
        DecibriError::StreamStartFailed("sample".to_string());
    DecibriError::SpeakerChannelsUnsupported { .. } => "SpeakerChannelsUnsupported", "SPEAKER_CHANNELS_UNSUPPORTED",
        DecibriError::SpeakerChannelsUnsupported {
            requested: 8,
            available: 2,
            reason: "sample".to_string(),
        };
    DecibriError::PermissionDenied => "PermissionDenied", "PERMISSION_DENIED",
        DecibriError::PermissionDenied;
    DecibriError::MicrophoneStreamClosed => "MicrophoneStreamClosed", "MICROPHONE_STREAM_CLOSED",
        DecibriError::MicrophoneStreamClosed;
    DecibriError::SpeakerStreamClosed => "SpeakerStreamClosed", "SPEAKER_STREAM_CLOSED",
        DecibriError::SpeakerStreamClosed;
    DecibriError::DeviceFailed { .. } => "DeviceFailed", "DEVICE_FAILED",
        DecibriError::DeviceFailed { source: sample_source() };

    #[cfg(feature = "capture")]
    DecibriError::ResampleConfigInvalid { .. } => "ResampleConfigInvalid", "RESAMPLE_CONFIG_INVALID",
        DecibriError::ResampleConfigInvalid { in_rate: 44100, out_rate: 16000 };
    DecibriError::ResampleAfterFlush => "ResampleAfterFlush", "RESAMPLE_AFTER_FLUSH",
        DecibriError::ResampleAfterFlush;
    DecibriError::ResampleFailed { .. } => "ResampleFailed", "RESAMPLE_FAILED",
        DecibriError::ResampleFailed { reason: "sample".to_string() };

    DecibriError::VadSampleRateUnsupported(_) => "VadSampleRateUnsupported", "VAD_SAMPLE_RATE_UNSUPPORTED",
        DecibriError::VadSampleRateUnsupported(44100);
    DecibriError::VadThresholdOutOfRange(_) => "VadThresholdOutOfRange", "VAD_THRESHOLD_OUT_OF_RANGE",
        DecibriError::VadThresholdOutOfRange(1.5);

    DecibriError::AecSampleRateUnsupported(_) => "AecSampleRateUnsupported", "AEC_SAMPLE_RATE_UNSUPPORTED",
        DecibriError::AecSampleRateUnsupported(96000);
    DecibriError::AecConfigInvalid { .. } => "AecConfigInvalid", "AEC_CONFIG_INVALID",
        DecibriError::AecConfigInvalid { reason: "sample".to_string() };

    #[cfg(feature = "capture")]
    DecibriError::FileReadFailed { .. } => "FileReadFailed", "FILE_READ_FAILED",
        DecibriError::FileReadFailed {
            path: PathBuf::from("sample.wav"),
            source: std::io::Error::other("sample"),
        };
    #[cfg(feature = "capture")]
    DecibriError::FileWriteFailed { .. } => "FileWriteFailed", "FILE_WRITE_FAILED",
        DecibriError::FileWriteFailed {
            path: PathBuf::from("sample.wav"),
            source: std::io::Error::other("sample"),
        };
    DecibriError::AudioFormatUnsupported { .. } => "AudioFormatUnsupported", "AUDIO_FORMAT_UNSUPPORTED",
        DecibriError::AudioFormatUnsupported { reason: "sample".to_string() };
    DecibriError::AudioFileMalformed { .. } => "AudioFileMalformed", "AUDIO_FILE_MALFORMED",
        DecibriError::AudioFileMalformed { reason: "sample".to_string() };
    DecibriError::AudioFileTruncated { .. } => "AudioFileTruncated", "AUDIO_FILE_TRUNCATED",
        DecibriError::AudioFileTruncated { reason: "sample".to_string() };

    DecibriError::VadNotConfigured => "VadNotConfigured", "VAD_NOT_CONFIGURED",
        DecibriError::VadNotConfigured;
    DecibriError::FileEngaged => "FileEngaged", "FILE_ENGAGED",
        DecibriError::FileEngaged;

    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtInitFailed { .. } => "OrtInitFailed", "ORT_INIT_FAILED",
        DecibriError::OrtInitFailed { source: sample_source() };
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtLoadFailed { .. } => "OrtLoadFailed", "ORT_LOAD_FAILED",
        DecibriError::OrtLoadFailed {
            path: PathBuf::from("sample"),
            source: sample_source(),
        };
    // Shares ORT_LOAD_FAILED with OrtLoadFailed: one user-facing failure, one
    // identity. The only code shared by two variants.
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtPathInvalid { .. } => "OrtPathInvalid", "ORT_LOAD_FAILED",
        DecibriError::OrtPathInvalid {
            path: PathBuf::from("sample"),
            reason: "sample",
        };
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtSessionBuildFailed(_) => "OrtSessionBuildFailed", "ORT_SESSION_BUILD_FAILED",
        DecibriError::OrtSessionBuildFailed(sample_source());
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtThreadsConfigFailed(_) => "OrtThreadsConfigFailed", "ORT_THREADS_CONFIG_FAILED",
        DecibriError::OrtThreadsConfigFailed(sample_source());
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::VadModelLoadFailed { .. } => "VadModelLoadFailed", "VAD_MODEL_LOAD_FAILED",
        DecibriError::VadModelLoadFailed {
            path: PathBuf::from("sample"),
            source: sample_source(),
        };
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::ModelLoadFailed { .. } => "ModelLoadFailed", "MODEL_LOAD_FAILED",
        DecibriError::ModelLoadFailed {
            path: PathBuf::from("sample"),
            source: sample_source(),
        };
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtInferenceFailed(_) => "OrtInferenceFailed", "ORT_INFERENCE_FAILED",
        DecibriError::OrtInferenceFailed(sample_source());
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtTensorCreateFailed { .. } => "OrtTensorCreateFailed", "ORT_TENSOR_CREATE_FAILED",
        DecibriError::OrtTensorCreateFailed {
            kind: "sample",
            source: sample_source(),
        };
    #[cfg(any(feature = "vad", feature = "denoise"))]
    DecibriError::OrtTensorExtractFailed { .. } => "OrtTensorExtractFailed", "ORT_TENSOR_EXTRACT_FAILED",
        DecibriError::OrtTensorExtractFailed {
            kind: "sample",
            source: sample_source(),
        };

    DecibriError::ForkAfterOrtInit { .. } => "ForkAfterOrtInit", "FORK_AFTER_ORT_INIT",
        DecibriError::ForkAfterOrtInit { init_pid: 1, current_pid: 2 };
    DecibriError::OnnxBackendFailed { .. } => "OnnxBackendFailed", "ONNX_BACKEND_FAILED",
        DecibriError::OnnxBackendFailed {
            backend: "sample",
            source: sample_source(),
        };
}

/// Bridge the capture resampler's errors into the central error.
///
/// Every error the pinned resampler release defines has its own arm.
/// `RatePairUnsupported` carries the offending rates through.
/// `ProcessAfterFlush` is the one steady-path variant and maps to
/// [`DecibriError::ResampleAfterFlush`], which names audio fed into a flushed
/// chain on the capture and `File` paths alike; both read paths short-circuit
/// on the flushed latch and `File` stops advancing once flushed, so no
/// decibri path reaches it. `ZeroSampleRate` is a rate outside decibri's
/// validated range, so it maps to [`DecibriError::SampleRateOutOfRange`].
/// Every rate the `File` path feeds the resampler is validated into
/// `1000..=384000`, as is the capture target, but the device-native rate is
/// not, so a backend reporting zero reaches this arm.
///
/// The catch-all is reached only by an error added in a later resampler
/// release. It maps to [`DecibriError::ResampleFailed`], which forwards the
/// resampler's own message rather than naming a cause decibri cannot know.
/// A later rate-carrying variant reports its rates in that forwarded
/// message.
#[cfg(feature = "capture")]
impl From<decibri_resampler::ResamplerError> for DecibriError {
    fn from(e: decibri_resampler::ResamplerError) -> Self {
        use decibri_resampler::ResamplerError;
        match e {
            ResamplerError::RatePairUnsupported { in_rate, out_rate } => {
                DecibriError::ResampleConfigInvalid { in_rate, out_rate }
            }
            ResamplerError::ProcessAfterFlush => DecibriError::ResampleAfterFlush,
            ResamplerError::ZeroSampleRate => DecibriError::SampleRateOutOfRange,
            _ => DecibriError::ResampleFailed {
                reason: e.to_string(),
            },
        }
    }
}

/// Bridge the offline file reader's errors into the central error.
///
/// The reader's eight variants group into three by what the caller can do about
/// them. The four `Unsupported*` variants say decibri cannot decode this file at
/// all, whatever is wrong with it, so they map to
/// [`DecibriError::AudioFormatUnsupported`]. `Malformed` says the bytes at a
/// known position were wrong, and `Truncated` says the file ended early; those
/// are separate variants because a caller re-encodes for one and re-fetches for
/// the other. Every forwarding arm carries the reader's own message, which names
/// the tag, offset or counts decibri would otherwise have to describe generically.
///
/// `Resample` delegates to the resampler's own conversion above, so a rate
/// failure surfacing through the reader carries the identity it carries
/// everywhere else. `ContainerCodecMismatch` and `Resample` are both unreachable
/// through the reader's entry point; they have arms because [`decibri_decode::DecodeError`]
/// is `#[non_exhaustive]`.
///
/// The catch-all lands on [`DecibriError::AudioFormatUnsupported`]: a variant
/// decibri has no arm for is one its reader release does not know, which is a
/// statement about decibri rather than about the caller's file.
#[cfg(feature = "capture")]
impl From<decibri_decode::DecodeError> for DecibriError {
    fn from(e: decibri_decode::DecodeError) -> Self {
        use decibri_decode::DecodeError;
        // Rendered before the match so every forwarding arm carries the
        // upstream text unchanged.
        let reason = e.to_string();
        match e {
            DecodeError::UnsupportedContainer { .. }
            | DecodeError::UnsupportedCodec { .. }
            | DecodeError::UnsupportedSampleFormat { .. }
            | DecodeError::UnsupportedChannelLayout { .. }
            | DecodeError::ContainerCodecMismatch { .. } => {
                DecibriError::AudioFormatUnsupported { reason }
            }
            DecodeError::Malformed { .. } => DecibriError::AudioFileMalformed { reason },
            DecodeError::Truncated { .. } => DecibriError::AudioFileTruncated { reason },
            DecodeError::Resample(source) => DecibriError::from(source),
            _ => DecibriError::AudioFormatUnsupported { reason },
        }
    }
}

/// Bridge the capture echo canceller's errors into the central error.
///
/// Every error the pinned canceller release defines has its own arm.
/// `SampleRateOutOfRange` is the one a decibri configuration can reach through
/// its own validated range, so it maps to
/// [`DecibriError::AecSampleRateUnsupported`], which carries the offending rate
/// through and keeps the same message a caller sees from
/// [`crate::microphone::MicrophoneConfig::validate`].
///
/// The remaining four are construction-time configuration rejections whose own
/// messages name the window enforced (`TailOutOfRange`,
/// `EchoDelayOutOfRange`, `SearchDelayOutOfRange`) or the string that named no
/// model (`UnknownModel`). They map to [`DecibriError::AecConfigInvalid`],
/// which forwards that message: it is more precise than any text decibri could
/// author, and for the unknown model it is the only text that can list the
/// models a caller may select.
///
/// The catch-all is reached only by an error added in a later canceller
/// release. It maps to the same variant on the same terms, forwarding the
/// canceller's own message rather than naming a cause decibri cannot know.
#[cfg(feature = "aec")]
impl From<decibri_aec::AecError> for DecibriError {
    fn from(e: decibri_aec::AecError) -> Self {
        use decibri_aec::AecError;
        // Rendered before the match so every forwarding arm carries the
        // upstream text unchanged.
        let reason = e.to_string();
        match e {
            AecError::SampleRateOutOfRange { requested } => {
                DecibriError::AecSampleRateUnsupported(requested)
            }
            AecError::TailOutOfRange { .. } => DecibriError::AecConfigInvalid { reason },
            AecError::EchoDelayOutOfRange { .. } => DecibriError::AecConfigInvalid { reason },
            AecError::SearchDelayOutOfRange { .. } => DecibriError::AecConfigInvalid { reason },
            AecError::UnknownModel { .. } => DecibriError::AecConfigInvalid { reason },
            _ => DecibriError::AecConfigInvalid { reason },
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

    /// SCREAMING_SNAKE of the variant name, computed the way the naming rule
    /// states it, so a hand-typed code that drifts from its variant fails here.
    fn screaming_snake(name: &str) -> String {
        let mut out = String::new();
        for (i, ch) in name.char_indices() {
            if i > 0 && ch.is_ascii_uppercase() {
                out.push('_');
            }
            out.push(ch.to_ascii_uppercase());
        }
        out
    }

    /// Every code is SCREAMING_SNAKE of its own variant name, with the single
    /// documented collapse.
    #[test]
    fn codes_follow_the_naming_rule() {
        for err in error_catalog() {
            let name = err.variant_name();
            let expected = if name == "OrtPathInvalid" {
                "ORT_LOAD_FAILED".to_string()
            } else {
                screaming_snake(name)
            };
            assert_eq!(err.code(), expected, "code drifted for {name}");
        }
    }

    /// The catalog carries each variant once, so a binding-side coverage test
    /// walking it sees no variant twice and no variant missing.
    #[test]
    fn catalog_names_are_unique() {
        let catalog = error_catalog();
        let mut names: Vec<&str> = catalog.iter().map(|e| e.variant_name()).collect();
        let total = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), total, "catalog carries a variant twice");
    }

    /// One code is shared by exactly two variants: the documented
    /// `OrtPathInvalid` collapse onto `ORT_LOAD_FAILED`. Every other code
    /// belongs to one variant.
    #[cfg(any(feature = "vad", feature = "denoise"))]
    #[test]
    fn only_the_ort_path_pair_shares_a_code() {
        let catalog = error_catalog();
        let shared: Vec<&str> = catalog
            .iter()
            .filter(|e| {
                catalog
                    .iter()
                    .filter(|other| other.code() == e.code())
                    .count()
                    > 1
            })
            .map(|e| e.variant_name())
            .collect();
        assert_eq!(shared, ["OrtLoadFailed", "OrtPathInvalid"]);
    }

    /// Every catalog instance renders a non-empty Display string, which is what
    /// the binding-side coverage tests classify on.
    #[test]
    fn catalog_instances_render() {
        for err in error_catalog() {
            assert!(
                !err.to_string().is_empty(),
                "empty Display for {}",
                err.variant_name()
            );
        }
    }

    /// The `FileEngaged` Display message is matched by prefix in the Node
    /// binding to assign the error its stable code, so the text is frozen.
    #[test]
    fn file_engaged_message_is_frozen() {
        assert_eq!(
            DecibriError::FileEngaged.to_string(),
            "File iteration has begun; construct a new File to analyze the whole recording"
        );
    }

    /// The `FlacCompressionOutOfRange` Display message is matched by prefix
    /// in the Node binding to classify the error, so the text is frozen.
    #[test]
    fn flac_compression_out_of_range_message_is_frozen() {
        assert_eq!(
            DecibriError::FlacCompressionOutOfRange.to_string(),
            "flac compression level must be between 0 and 8"
        );
    }

    /// The `FileWriteFailed` Display message keeps its frozen leading prefix,
    /// which the Node binding matches to assign the error its stable code.
    #[cfg(feature = "capture")]
    #[test]
    fn file_write_failed_message_prefix_is_frozen() {
        let err = DecibriError::FileWriteFailed {
            path: PathBuf::from("sample.wav"),
            source: std::io::Error::other("disk full"),
        };
        let msg = err.to_string();
        assert!(
            msg.starts_with("Failed to write audio file "),
            "frozen prefix must not drift: {msg}"
        );
        assert!(err.source().is_some(), "the I/O cause is walkable");
    }

    /// The `SpeakerChannelsUnsupported` Display message keeps its frozen
    /// leading clause, which the Node binding matches to assign the error its
    /// stable code, and names both counts plus the platform's own text.
    /// Regression: a reworded clause silently demotes the error to the
    /// unclassified bucket in the Node binding.
    #[test]
    fn speaker_channels_unsupported_message_names_both_counts() {
        let err = DecibriError::SpeakerChannelsUnsupported {
            requested: 124,
            available: 2,
            reason: "the device said no".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "the output device does not support 124 output channels; it reports 2: \
             the device said no"
        );
    }

    /// The `ChannelsOutOfRange` Display message is a frozen exact-match string
    /// that downstream consumers branch on, and it names the floor only.
    /// Regression: a ceiling named in the text that no validation enforces.
    #[test]
    fn channels_out_of_range_message_is_frozen() {
        assert_eq!(
            DecibriError::ChannelsOutOfRange.to_string(),
            "channels must be at least 1"
        );
    }

    /// `ResampleFailed` forwards the upstream text: the reason is formatted
    /// into the Display string behind the stable `resampler error:` prefix
    /// the Node binding classifies on.
    #[test]
    fn resample_failed_forwards_the_upstream_text() {
        let err = DecibriError::ResampleFailed {
            reason: "sample upstream text".to_string(),
        };
        assert_eq!(err.to_string(), "resampler error: sample upstream text");
    }

    /// Every variant the pinned echo canceller release defines reaches the variant
    /// intended for it, and every forwarding arm carries the canceller's own
    /// message through unchanged. Regression: a variant routed to the wrong
    /// decibri variant, and a decibri-authored message replacing text that names
    /// the window enforced or the models a caller may select.
    #[cfg(feature = "aec")]
    #[test]
    fn every_aec_error_reaches_its_intended_variant() {
        use decibri_aec::AecError;

        let rate = DecibriError::from(AecError::SampleRateOutOfRange { requested: 96_000 });
        assert!(
            matches!(rate, DecibriError::AecSampleRateUnsupported(96_000)),
            "an out-of-window rate carries the rate through, got {rate:?}"
        );
        assert_eq!(
            rate.to_string(),
            "echo cancellation only supports sample rates 8000 to 48000",
            "the rate variant keeps decibri's own stable message"
        );

        let forwarding = [
            AecError::TailOutOfRange { requested_ms: 4 },
            AecError::EchoDelayOutOfRange { requested_ms: 4 },
            AecError::SearchDelayOutOfRange {
                requested_ms: 4,
                fine_window_ms: 250,
            },
            AecError::UnknownModel {
                requested: "tao".to_string(),
            },
        ];
        for err in forwarding {
            let upstream = err.to_string();
            let converted = DecibriError::from(err);
            assert!(
                matches!(converted, DecibriError::AecConfigInvalid { .. }),
                "a configuration rejection reaches AecConfigInvalid, got {converted:?}"
            );
            assert_eq!(
                converted.to_string(),
                format!("echo canceller configuration error: {upstream}"),
                "the canceller's own message is forwarded unchanged"
            );
        }
    }
}
