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
//!   by index or case-insensitive name substring.
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
//! - [`capture::CaptureStream`][] —
//!   [`try_next_chunk`](capture::CaptureStream::try_next_chunk),
//!   [`next_chunk`](capture::CaptureStream::next_chunk),
//!   [`is_open`](capture::CaptureStream::is_open),
//!   [`stop`](capture::CaptureStream::stop).
//! - [`output::OutputStream`][] —
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
