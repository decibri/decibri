//! Cross-platform audio capture, playback, and voice-activity detection.
//!
//! `decibri` is the Rust core behind the `decibri` npm package. It provides:
//!
//! - **Audio capture** ([`capture`]): microphone input with frame-exact
//!   buffering, built on cpal.
//! - **Audio playback** ([`output`]): speaker output with graceful drain
//!   and immediate-stop semantics.
//! - **Voice activity detection** ([`vad`]): Silero VAD v5 inference via
//!   ONNX Runtime, per-call stateful.
//! - **Device enumeration** ([`device`]): list and select audio devices
//!   by index, case-insensitive name substring, or stable per-host ID.
//!
//! # Feature flags
//!
//! | Flag                    | Default | Purpose                                      |
//! |-------------------------|---------|----------------------------------------------|
//! | `capture`               | on      | Microphone input stream support              |
//! | `output`                | on      | Speaker output stream support                |
//! | `vad`                   | on      | Silero VAD ONNX inference                    |
//! | `denoise`               | on      | Reserved (stub)                              |
//! | `gain`                  | on      | Reserved (stub)                              |
//! | `ort-load-dynamic`      | on      | ORT loaded at runtime from a user path       |
//! | `ort-download-binaries` | off     | ORT downloaded at build time, embedded       |
//!
//! `ort-load-dynamic` and `ort-download-binaries` are mutually exclusive;
//! selecting both is a compile error.
//!
//! See [`docs/features.md`](https://github.com/decibri/decibri/blob/main/docs/features.md)
//! for a comprehensive reference including execution-provider features,
//! tradeoff analysis, and guidance for FFI binding authors.
//!
//! # Example: capture and run VAD
//!
//! [`CaptureStream::next_chunk`](capture::CaptureStream::next_chunk) is the
//! recommended FFI-ready interface for reading audio chunks, returning a
//! three-state `Result`: `Ok(Some(chunk))` / `Ok(None)` (timeout, stream
//! still open) / `Err(DecibriError::CaptureStreamClosed)`.
//!
//! ```ignore
//! use std::time::Duration;
//! use decibri::capture::{AudioCapture, CaptureConfig};
//! use decibri::error::DecibriError;
//! use decibri::vad::{SileroVad, VadConfig};
//!
//! let capture = AudioCapture::new(CaptureConfig::default())?;
//! let stream  = capture.start()?;
//! let mut vad = SileroVad::new(VadConfig::default())?;
//!
//! loop {
//!     match stream.next_chunk(Some(Duration::from_millis(100))) {
//!         Ok(Some(chunk)) => {
//!             let result = vad.process(&chunk.data)?;
//!             if result.is_speech {
//!                 println!("speech @ p={:.2}", result.probability);
//!             }
//!         }
//!         Ok(None) => continue,
//!         Err(DecibriError::CaptureStreamClosed) => break,
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
//! [`DecibriError::OrtPathInvalid`](error::DecibriError::OrtPathInvalid)
//! (carries `path: PathBuf` and `reason: &'static str`, never touches ORT
//! symbols) rather than reconstructing an `ort::Error`. decibri's internal
//! `init_ort_once` follows this pattern: it performs a filesystem-level
//! `Path::is_file()` check first and returns
//! [`OrtPathInvalid`](error::DecibriError::OrtPathInvalid) on rejection,
//! only handing the path to `ort::init_from` once the check passes.
//!
//! This constraint is why [`DecibriError`](error::DecibriError) splits
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
//! [`capture::CaptureStream`] and [`output::OutputStream`] are `!Sync`
//! because they hold a `cpal::Stream` internally; wrap them in a mutex or
//! move them into a dedicated thread for shared access.
//!
//! # Stability
//!
//! Within 3.x, the following FFI-consumer surface is declared stable and
//! will not change signature without a breaking version bump:
//!
//! - [`capture::CaptureStream`][]:
//!   [`try_next_chunk`](capture::CaptureStream::try_next_chunk),
//!   [`next_chunk`](capture::CaptureStream::next_chunk),
//!   [`is_open`](capture::CaptureStream::is_open),
//!   [`stop`](capture::CaptureStream::stop).
//! - [`output::OutputStream`][]:
//!   [`send`](output::OutputStream::send),
//!   [`drain`](output::OutputStream::drain),
//!   [`is_playing`](output::OutputStream::is_playing),
//!   [`stop`](output::OutputStream::stop).

#[cfg(all(feature = "ort-load-dynamic", feature = "ort-download-binaries"))]
compile_error!(
    "features `ort-load-dynamic` and `ort-download-binaries` are mutually exclusive; \
     select exactly one. The default feature set enables `ort-load-dynamic`; \
     if you want `ort-download-binaries`, disable default features first."
);

pub mod error;
pub mod sample;

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

#[cfg(any(feature = "capture", feature = "output"))]
pub mod device;

#[cfg(feature = "capture")]
pub mod capture;

#[cfg(feature = "output")]
pub mod output;

#[cfg(feature = "vad")]
pub mod vad;

#[cfg(feature = "denoise")]
pub mod denoise;

#[cfg(feature = "gain")]
pub mod gain;
