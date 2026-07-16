//! Cross-platform audio capture, playback, and voice-activity detection.
//!
//! decibri records from a microphone, plays audio to a speaker, and runs Silero
//! voice-activity detection through one API on Windows, macOS, and Linux. It is
//! the Rust core that also powers decibri's Python and Node.js packages.
//!
//! # Orientation
//!
//! There are two primary types: [`Microphone`] for input and [`Speaker`] for
//! output. Both follow the same shape: build a configuration, construct the
//! object, then call `start()` to open the device and obtain a stream.
//!
//! - A [`Microphone`] yields a [`MicrophoneStream`] you pull [`AudioChunk`]s
//!   from with [`next_chunk`](MicrophoneStream::next_chunk).
//! - A [`Speaker`] yields a [`SpeakerStream`] you push `f32` samples to with
//!   [`send`](SpeakerStream::send).
//!
//! Voice-activity detection runs through [`SileroVad`]: construct it once, then
//! call [`process`](SileroVad::process) on each chunk's samples.
//!
//! List devices with [`input_devices`] / [`output_devices`] (or
//! [`Microphone::devices`] / [`Speaker::devices`]) and pick one with a
//! [`DeviceSelector`]. Every fallible operation returns [`DecibriError`].
//!
//! # Feature flags
//!
//! | Flag                    | Default | Purpose                                      |
//! |-------------------------|---------|----------------------------------------------|
//! | `capture`               | on      | Microphone input stream support              |
//! | `playback`              | on      | Speaker output stream support                |
//! | `vad`                   | on      | Silero voice-activity detection (pulls `ort`)|
//! | `ort-load-dynamic`      | on      | ONNX Runtime loaded at runtime from a path   |
//! | `ort-download-binaries` | off     | ONNX Runtime downloaded at build time        |
//!
//! `ort-load-dynamic` and `ort-download-binaries` are mutually exclusive;
//! selecting both is a compile error. Disabling `vad` drops the ONNX Runtime
//! dependency entirely.
//!
//! Execution-provider features (`coreml`, `cuda`, `directml`, `rocm`) are off by
//! default and opt in to GPU acceleration on specific platforms. See
//! [`docs/features.md`](https://github.com/decibri/decibri/blob/main/docs/features.md)
//! for the full reference.
//!
//! # Example: capture and detect speech
//!
//! [`MicrophoneStream::next_chunk`](microphone::MicrophoneStream::next_chunk)
//! returns a three-state `Result`: `Ok(Some(chunk))` with data, `Ok(None)` on a
//! timeout while the stream is still open, and
//! `Err(DecibriError::MicrophoneStreamClosed)` once the stream ends.
//!
//! ```no_run
//! use std::time::Duration;
//! use decibri::{Microphone, MicrophoneConfig, SileroVad, VadConfig, DecibriError};
//!
//! let microphone = Microphone::new(MicrophoneConfig::default())?;
//! let stream = microphone.start()?;
//! let mut vad = SileroVad::new(VadConfig::default())?;
//!
//! // 1600 interleaved samples = one 100 ms block of mono 16 kHz audio
//! // (frames_per_buffer * channels for the default config).
//! loop {
//!     match stream.next_chunk(1600, Some(Duration::from_millis(100))) {
//!         Ok(Some(chunk)) => {
//!             let result = vad.process(&chunk.data)?;
//!             if result.is_speech {
//!                 println!("speech @ p={:.2}", result.probability);
//!             }
//!         }
//!         Ok(None) => continue,
//!         Err(DecibriError::MicrophoneStreamClosed) => break,
//!         Err(e) => return Err(e.into()),
//!     }
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! # Process-global ORT initialization
//!
//! ONNX Runtime initializes exactly once per process. The first successful
//! [`SileroVad::new`](vad::SileroVad::new) call wins; later constructions
//! with a different [`vad::VadConfig::ort_library_path`] silently reuse the
//! first path. See [`vad::VadConfig`] for details.
//!
//! # ORT error construction has FFI side effects
//!
//! Under `ort-load-dynamic`, constructing any [`ort::Error`] calls into the
//! ORT C API (`ortsys![CreateStatus]`), which in turn triggers ORT dylib
//! loading. If the dylib path is invalid (missing file, a directory, or a
//! wrong-arch binary), this load can hang indefinitely on some hosts
//! (reproduced on Windows with pyke/ort 2.0.0-rc.12 and onnxruntime 1.24.4).
//!
//! For authors writing FFI bindings on top of decibri: when you need to
//! surface a "this ORT path is wrong" failure from a pre-load check, use
//! [`DecibriError::OrtPathInvalid`]
//! (carries `path: PathBuf` and `reason: &'static str`, never touches ORT
//! symbols) rather than reconstructing an `ort::Error`. decibri's internal
//! `init_ort_once` follows this pattern: it performs a filesystem-level
//! `Path::is_file()` check first and returns
//! [`OrtPathInvalid`](error::DecibriError::OrtPathInvalid) on rejection,
//! only handing the path to `ort::init_from` once the check passes.
//!
//! This constraint is why [`DecibriError`] splits
//! "ORT path failure" across two variants:
//! [`OrtLoadFailed`](error::DecibriError::OrtLoadFailed) carries an
//! `ort::Error` source (reached after `ort::init_from` genuinely failed,
//! by which point ORT is loaded and constructing errors is safe), and
//! [`OrtPathInvalid`](error::DecibriError::OrtPathInvalid) does not
//! (reached before any ORT symbol is touched). The
//! [`is_ort_path_error`](error::DecibriError::is_ort_path_error) helper
//! groups both so consumers don't need to match the variant distinction.
//!
//! # Fork safety
//!
//! Once ORT has been initialized in a process (via the first
//! [`SileroVad::new`](vad::SileroVad::new) call), that process's ORT
//! state survives across `fork()`. ORT is not designed to have its
//! runtime state cloned into a forked child: internal threads, memory
//! pools, and device handles belonging to the parent may misbehave in
//! the child.
//!
//! Python consumers using
//! [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html)
//! should call `multiprocessing.set_start_method('spawn', force=True)`
//! before importing `decibri`. This gives each child its own fresh ORT
//! state rather than inheriting the parent's.
//!
//! Node.js consumers using `child_process` or `worker_threads` are
//! unaffected; those mechanisms do not share ORT state across workers.
//!
//! # Thread safety
//!
//! All public types are `Send` and suitable for cross-thread handoff.
//! [`microphone::MicrophoneStream`] and [`speaker::SpeakerStream`] are also
//! `Sync` (they hold their live platform audio stream behind an internal
//! mutex), so they can be shared across threads, for example via an `Arc`,
//! without external synchronization.
//!
//! # Stability
//!
//! The following stream surface is stable and changes signature only with a
//! breaking version bump:
//!
//! - [`microphone::MicrophoneStream`][]:
//!   [`try_next_chunk`](microphone::MicrophoneStream::try_next_chunk),
//!   [`next_chunk`](microphone::MicrophoneStream::next_chunk),
//!   [`is_open`](microphone::MicrophoneStream::is_open),
//!   [`stop`](microphone::MicrophoneStream::stop).
//! - [`speaker::SpeakerStream`][]:
//!   [`send`](speaker::SpeakerStream::send),
//!   [`drain`](speaker::SpeakerStream::drain),
//!   [`is_playing`](speaker::SpeakerStream::is_playing),
//!   [`stop`](speaker::SpeakerStream::stop).

#[cfg(all(feature = "ort-load-dynamic", feature = "ort-download-binaries"))]
compile_error!(
    "features `ort-load-dynamic` and `ort-download-binaries` are mutually exclusive; \
     select exactly one. The default feature set enables `ort-load-dynamic`; \
     if you want `ort-download-binaries`, disable default features first."
);

pub mod error;
pub mod sample;

// Internal ONNX session abstraction. The trait is `pub(crate)` and not part
// of the public API.
//
// Gated on the presence of an ONNX consumer: `vad` (the Silero VAD) or the
// capture denoise stage (`capture` + `denoise`, which runs FastEnhancer-T
// through this seam). Denoise is a capture-path stage, so a `denoise` build
// without `capture` has no consumer and the module is left out rather than
// compiled dead. Either consumer pulls the `ort` backend via its feature.
#[cfg(any(feature = "vad", all(feature = "capture", feature = "denoise")))]
mod onnx;

/// Resolved cpal version (major.minor) extracted at build time from the
/// crate's `Cargo.toml`.
///
/// Populated by `build.rs` parsing the cpal entry from the crate's
/// `Cargo.toml` (with workspace fallback for `{ workspace = true }`
/// inherits) and truncating to major.minor. Used by binding
/// layers (Node, Python) as the value of the `portaudio` field in their
/// version-info responses:
///
/// ```text
/// portaudio: format!("cpal {}", decibri::CPAL_VERSION)
/// // -> "cpal 0.17"
/// ```
///
/// Single source of truth: bumping cpal in the workspace `Cargo.toml`
/// re-triggers the build script (via `cargo:rerun-if-changed=Cargo.toml`),
/// which updates this constant, which updates every binding that reads it.
/// No manual sync required across crates.
///
/// Format: `"0.17"`. Bindings prefix with `"cpal "` when reporting to users.
/// If future work wants finer granularity (e.g. full `"0.17.3"`), update
/// `truncate_to_major_minor` in `build.rs`.
///
/// Always present, regardless of which decibri features are enabled. The
/// string is emitted unconditionally by the build script; Rust consumers
/// who build without cpal features can ignore this constant or leave it
/// unused without cost.
pub const CPAL_VERSION: &str = env!("DECIBRI_CPAL_VERSION");

#[cfg(any(feature = "capture", feature = "playback"))]
pub mod device;

// Internal audio-backend seam: the one place the crate reaches the platform
// audio library. Not part of the public API.
#[cfg(any(feature = "capture", feature = "playback"))]
mod backend;

// Internal capture stage chain: the normalize segment applied between the raw
// device buffers and the exact-size reblock. Not part of the public API.
#[cfg(feature = "capture")]
mod stage;

#[cfg(feature = "capture")]
pub mod microphone;

#[cfg(feature = "capture")]
pub mod file;

#[cfg(feature = "playback")]
pub mod speaker;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "denoise")]
pub mod denoise;

#[cfg(feature = "gain")]
pub mod gain;

// Crate-root re-exports so consumers can write `use decibri::Microphone`
// rather than reaching through the module path.
#[cfg(feature = "capture")]
pub use microphone::{
    AudioChunk, DenoiseModel, HighpassFilter, Microphone, MicrophoneConfig, MicrophoneStream,
};

#[cfg(feature = "capture")]
pub use file::{File, FileConfig};

#[cfg(all(feature = "capture", feature = "vad"))]
pub use file::{Segment, VadReport, VadWindow};

#[cfg(feature = "playback")]
pub use speaker::{Speaker, SpeakerConfig, SpeakerSink, SpeakerStream};

#[cfg(any(feature = "capture", feature = "playback"))]
pub use device::{input_devices, output_devices, DeviceSelector, MicrophoneInfo, SpeakerInfo};

#[cfg(feature = "vad")]
pub use vad::{SileroVad, VadConfig, VadResult};

pub use error::DecibriError;
