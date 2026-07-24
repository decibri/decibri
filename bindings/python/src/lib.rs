//! Python bindings for decibri.
//!
//! This crate exposes decibri's audio capture, audio output, and Silero VAD
//! surfaces to Python via PyO3. The binding is a thin adapter layer: policy
//! lives in the Python wrapper (`python/decibri/_classes.py`); this crate
//! handles only the Rust-Python boundary, error mapping, and GIL release on
//! blocking calls.
//!
//! Error mapping covers all 29 `DecibriError` variants and surfaces them as
//! 31 distinct Python exception classes (1 base + 19 direct subclasses + the
//! ORT family of 8 + the OrtPathError catch-target + 2 path-specific ORT
//! subclasses). The Python class hierarchy lives in `decibri.exceptions`;
//! this crate constructs and raises instances via `pyo3::create_exception!`.
//!
//! Format handling is binding-layer only. The decibri Rust core has no
//! `SampleFormat` enum; it operates on `f32` internally. This binding accepts
//! `"int16"` and `"float32"` strings from Python and uses
//! `decibri::sample::*` helpers for byte-level conversion at the boundary.
//!
//! VAD architecture: `vad_mode` selects the detector the bridge runs on the
//! pre-enhancement signal (Silero inference for "silero", an energy RMS for
//! "energy"), exposed as a scalar via `vad_probability`; `vad_holdoff` is
//! stored as inert state. The wrapper layer (`_classes.py`) implements the
//! holdoff and threshold policy on top of that scalar, the same way for both
//! modes.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use std::collections::HashMap;

use tokio::sync::Mutex;

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyModule, PyType};

use pyo3_async_runtimes::tokio::future_into_py;

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyArrayMethods};

use decibri::device::{
    input_devices, output_devices, DeviceSelector, MicrophoneInfo as CoreDeviceInfo,
    SpeakerInfo as CoreOutputDeviceInfo,
};
use decibri::error::DecibriError as CoreDecibriError;
use decibri::file::{File as CoreFile, FileConfig};
use decibri::microphone::{
    AudioChunk, DenoiseModel, HighpassFilter, Microphone, MicrophoneConfig, MicrophoneStream,
};
use decibri::sample::{
    downmix_to_mono, f32_le_bytes_to_f32, f32_to_f32_le_bytes, f32_to_i16_le_bytes,
    i16_le_bytes_to_f32, rms,
};
use decibri::speaker::{Speaker, SpeakerConfig, SpeakerStream};
use decibri::vad::{SileroVad, VadConfig};
use decibri::CPAL_VERSION;

// ---------------------------------------------------------------------------
// Compile-time assertion: all four bridge pyclasses are Send + 'static.
//
// The 'static bound is what `pyo3_async_runtimes::tokio::future_into_py`
// requires when capturing the bridge into an async closure. Asserting at
// compile time means a future change that reintroduces a !Send field, or
// adds a non-static reference, fails the build at the point of regression
// rather than at async integration time, and makes the regression's blast
// radius obvious from the error message (it points at the new field).
//
// Coverage:
//   - MicrophoneBridge / SpeakerBridge: the sync pyclasses; their
//     Send + 'static property comes from not being `unsendable`.
//   - AsyncMicrophoneBridge: holds an Arc<MicrophoneBridge> directly (the
//     sync mic bridge is internally thread-safe, so no outer Mutex is
//     needed); AsyncSpeakerBridge holds an Arc<tokio::sync::Mutex<SpeakerBridge>>.
//     Arc<T> is Send + Sync when T is Send + Sync, so both async wrappers
//     inherit the Send + 'static property automatically.
//
// If this assertion fails to compile, look for a recently added field
// on any of the four pyclasses whose type is not `Send + 'static`: raw
// `cpal::Stream` (use a sendable handle instead), `Rc<_>` / `RefCell<_>`
// (use `Arc<_>` / `Mutex<_>`), `&'a T` for any non-static lifetime
// (extract owned data instead).
//
// Sendability of the sync bridges is empirically backed: the Rust core's
// `MicrophoneStream` is asserted `Send + Sync` by the
// `test_microphone_stream_is_send_and_sync` compile-time guard in
// `crates/decibri/src/microphone.rs` (unconditional, runs on every CI
// platform), and the documented `Send` contract for all public
// capture/output types lives in `crates/decibri/src/lib.rs`.
// ---------------------------------------------------------------------------

const _: () = {
    const fn assert_send_static<T: Send + 'static>() {}
    let _ = assert_send_static::<MicrophoneBridge>;
    let _ = assert_send_static::<SpeakerBridge>;
    let _ = assert_send_static::<AsyncMicrophoneBridge>;
    let _ = assert_send_static::<AsyncSpeakerBridge>;
};

// ---------------------------------------------------------------------------
// Exception class lookup.
//
// The class hierarchy is defined once in pure Python at
// decibri.exceptions. The `__init__` overrides on OrtLoadFailed,
// OrtPathInvalid, and VadModelLoadFailed unpack tuple args into named
// `path` and `reason` attributes per CPython OSError convention.
//
// to_py_err raises instances of those pure-Python classes via
// PyErr::from_type, so consumers catching decibri.exceptions.<X>
// (or the re-exported decibri.<X>) catch what Rust raises. Class
// objects are imported once on first use and cached in EXCEPTION_CLASSES.
// ---------------------------------------------------------------------------

static EXCEPTION_CLASSES: PyOnceLock<HashMap<&'static str, Py<PyType>>> = PyOnceLock::new();

const EXCEPTION_NAMES: &[&str] = &[
    "DecibriError",
    "AlreadyRunning",
    "MicrophoneStreamClosed",
    "DeviceFailed",
    "OnnxBackendFailed",
    "ChannelsOutOfRange",
    "MultichannelNotSupported",
    "DeviceEnumerationFailed",
    "DeviceIndexOutOfRange",
    "MicrophoneNotFound",
    "ForkAfterOrtInit",
    "FramesPerBufferOutOfRange",
    "InvalidFormat",
    "MultipleDevicesMatch",
    "NoMicrophoneFound",
    "NoSpeakerFound",
    "NotAnInputDevice",
    "SpeakerNotFound",
    "SpeakerStreamClosed",
    "PermissionDenied",
    "SampleRateOutOfRange",
    "AgcTargetOutOfRange",
    "LimiterCeilingOutOfRange",
    "StreamOpenFailed",
    "StreamStartFailed",
    "VadSampleRateUnsupported",
    "VadThresholdOutOfRange",
    "FileReadFailed",
    "FileConsumed",
    "WavInvalid",
    "VadNotConfigured",
    "VadModelLoadFailed",
    "ModelLoadFailed",
    "OrtError",
    "OrtInitFailed",
    "OrtSessionBuildFailed",
    "OrtThreadsConfigFailed",
    "OrtTensorCreateFailed",
    "OrtTensorExtractFailed",
    "OrtInferenceFailed",
    "OrtPathError",
    "OrtLoadFailed",
    "OrtPathInvalid",
];

/// Construct a PyErr for the named pure-Python exception class with a single
/// message argument. Used at sites where the binding raises an exception
/// directly (not via to_py_err's CoreDecibriError mapping).
fn raise_named(py: Python<'_>, name: &'static str, msg: impl Into<String>) -> PyErr {
    match exception_class(py, name) {
        Ok(cls) => PyErr::from_type(cls, (msg.into(),)),
        Err(lookup_err) => lookup_err,
    }
}

fn exception_class<'py>(py: Python<'py>, name: &'static str) -> PyResult<Bound<'py, PyType>> {
    let map = EXCEPTION_CLASSES.get_or_try_init(py, || -> PyResult<_> {
        let module = py.import("decibri.exceptions")?;
        let mut map: HashMap<&'static str, Py<PyType>> =
            HashMap::with_capacity(EXCEPTION_NAMES.len());
        for &n in EXCEPTION_NAMES {
            let cls = module.getattr(n)?.cast_into::<PyType>().map_err(|e| {
                PyRuntimeError::new_err(format!("decibri.exceptions.{n} is not a class: {e}"))
            })?;
            map.insert(n, cls.unbind());
        }
        Ok(map)
    })?;

    let cls = map.get(name).ok_or_else(|| {
        PyRuntimeError::new_err(format!(
            "internal error: exception class {name} not registered"
        ))
    })?;
    Ok(cls.bind(py).clone())
}

// ---------------------------------------------------------------------------
// Error mapping: DecibriError -> PyErr.
//
// Covers all 37 variants explicitly. Catch-all `_ =>` arm at end is required
// because DecibriError is #[non_exhaustive].
//
// For variants with a `path` field (OrtLoadFailed, OrtPathInvalid,
// VadModelLoadFailed, ModelLoadFailed), the mapper passes a tuple as the args
// so the Python __init__ override can store named attributes.
// ---------------------------------------------------------------------------

type ExceptionRaiser = Box<dyn FnOnce(Bound<'_, PyType>) -> PyErr>;

fn to_py_err(py: Python<'_>, err: CoreDecibriError) -> PyErr {
    let msg = err.to_string();

    // Determine which exception class to raise plus the args tuple.
    // The pure-Python __init__ overrides on OrtLoadFailed, OrtPathInvalid,
    // and VadModelLoadFailed unpack multi-element tuples into named
    // attributes; other classes accept a single message string.
    let (name, raise): (&'static str, ExceptionRaiser) = match err {
        // Unit variants (13).
        CoreDecibriError::SampleRateOutOfRange => (
            "SampleRateOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::ChannelsOutOfRange => (
            "ChannelsOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::MultichannelNotSupported => (
            "MultichannelNotSupported",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::FramesPerBufferOutOfRange => (
            "FramesPerBufferOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::AgcTargetOutOfRange => (
            "AgcTargetOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::LimiterCeilingOutOfRange => (
            "LimiterCeilingOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::InvalidFormat => (
            "InvalidFormat",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::DeviceIndexOutOfRange => (
            "DeviceIndexOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::NoMicrophoneFound => (
            "NoMicrophoneFound",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::NoSpeakerFound => (
            "NoSpeakerFound",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::NotAnInputDevice => (
            "NotAnInputDevice",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::AlreadyRunning => (
            "AlreadyRunning",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::PermissionDenied => (
            "PermissionDenied",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::MicrophoneStreamClosed => (
            "MicrophoneStreamClosed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::SpeakerStreamClosed => (
            "SpeakerStreamClosed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // Tuple variants with String (5).
        CoreDecibriError::MicrophoneNotFound(_) => (
            "MicrophoneNotFound",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::SpeakerNotFound(_) => (
            "SpeakerNotFound",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::DeviceEnumerationFailed(_) => (
            "DeviceEnumerationFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::StreamOpenFailed(_) => (
            "StreamOpenFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::StreamStartFailed(_) => (
            "StreamStartFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // Tuple variants with primitive (2).
        CoreDecibriError::VadSampleRateUnsupported(_) => (
            "VadSampleRateUnsupported",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::VadThresholdOutOfRange(_) => (
            "VadThresholdOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // Offline file source variants (3).
        CoreDecibriError::FileReadFailed { .. } => (
            "FileReadFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::WavInvalid { .. } => (
            "WavInvalid",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::VadNotConfigured => (
            "VadNotConfigured",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // Tuple variants with #[source] ort::Error (3).
        CoreDecibriError::OrtSessionBuildFailed(_) => (
            "OrtSessionBuildFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::OrtThreadsConfigFailed(_) => (
            "OrtThreadsConfigFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::OrtInferenceFailed(_) => (
            "OrtInferenceFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // Struct variants without path (4).
        CoreDecibriError::MultipleDevicesMatch { .. } => (
            "MultipleDevicesMatch",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::OrtInitFailed { .. } => (
            "OrtInitFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::OrtTensorCreateFailed { .. } => (
            "OrtTensorCreateFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::OrtTensorExtractFailed { .. } => (
            "OrtTensorExtractFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // Struct variants with path attribute (4). Multi-element args
        // tuples that the pure-Python __init__ unpacks into named attributes.
        CoreDecibriError::OrtLoadFailed { ref path, .. } => {
            let path_str = path_to_string(path);
            (
                "OrtLoadFailed",
                Box::new(move |cls| PyErr::from_type(cls, (msg, path_str))),
            )
        }
        CoreDecibriError::OrtPathInvalid { ref path, reason } => {
            let path_str = path_to_string(path);
            let reason_str = reason.to_string();
            (
                "OrtPathInvalid",
                Box::new(move |cls| PyErr::from_type(cls, (msg, path_str, reason_str))),
            )
        }
        CoreDecibriError::VadModelLoadFailed { ref path, .. } => {
            let path_str = path_to_string(path);
            (
                "VadModelLoadFailed",
                Box::new(move |cls| PyErr::from_type(cls, (msg, path_str))),
            )
        }
        // Model-agnostic counterpart to VadModelLoadFailed: the capture denoise
        // stage's model file failed to load. Same path-bearing shape.
        CoreDecibriError::ModelLoadFailed { ref path, .. } => {
            let path_str = path_to_string(path);
            (
                "ModelLoadFailed",
                Box::new(move |cls| PyErr::from_type(cls, (msg, path_str))),
            )
        }

        // ForkAfterOrtInit. Direct DecibriError subclass
        // (not under OrtError); message-only constructor like the unit
        // variants. The pid pair is embedded in the Display message; no
        // separate `init_pid` / `current_pid` Python attributes are
        // exposed in 0.1.0.
        CoreDecibriError::ForkAfterOrtInit { .. } => (
            "ForkAfterOrtInit",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // DeviceFailed: a runtime device/driver failure during streaming.
        // Direct DecibriError subclass like the other stream-level variants
        // (StreamOpenFailed, StreamStartFailed), not under DeviceError (which
        // is the device-enumeration/selection family). Message-only.
        CoreDecibriError::DeviceFailed { .. } => (
            "DeviceFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // OnnxBackendFailed: a non-ORT ONNX backend failure. Direct
        // DecibriError subclass (not under OrtError, which is the ORT-specific
        // family), mirroring ForkAfterOrtInit's deliberate placement.
        // Message-only.
        CoreDecibriError::OnnxBackendFailed { .. } => (
            "OnnxBackendFailed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // #[non_exhaustive] catch-all. Fall through to the base class.
        _ => (
            "DecibriError",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
    };

    match exception_class(py, name) {
        Ok(cls) => raise(cls),
        // Class lookup itself failed (decibri.exceptions module missing or
        // class missing). Surface the lookup error so the failure mode is
        // visible rather than silently masking the original error.
        Err(lookup_err) => lookup_err,
    }
}

#[inline]
fn path_to_string(path: &std::path::Path) -> String {
    path.to_string_lossy().into_owned()
}

// ---------------------------------------------------------------------------
// Format parsing helper.
//
// Accepts only "int16" and "float32". Returns DecibriError::InvalidFormat
// (a unit variant) for any other string. The to_py_err mapper translates
// that to the InvalidFormat Python exception with the canonical Display
// message ("format must be 'int16' or 'float32'").
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BindingSampleFormat {
    Int16,
    Float32,
}

fn parse_sample_format(s: &str) -> Result<BindingSampleFormat, CoreDecibriError> {
    match s {
        "int16" => Ok(BindingSampleFormat::Int16),
        "float32" => Ok(BindingSampleFormat::Float32),
        _ => Err(CoreDecibriError::InvalidFormat),
    }
}

// ---------------------------------------------------------------------------
// VersionInfo: surfaces decibri Rust core version, cpal version, and binding
// version via three string fields. Constructed by MicrophoneBridge::version().
// ---------------------------------------------------------------------------

#[pyclass(module = "decibri._decibri", frozen)]
struct VersionInfo {
    #[pyo3(get)]
    decibri: String,
    #[pyo3(get)]
    audio_backend: String,
    #[pyo3(get)]
    binding: String,
}

#[pymethods]
impl VersionInfo {
    fn __repr__(&self) -> String {
        format!(
            "VersionInfo(decibri='{}', audio_backend='{}', binding='{}')",
            self.decibri, self.audio_backend, self.binding
        )
    }
}

fn build_version_info() -> VersionInfo {
    VersionInfo {
        decibri: env!("CARGO_PKG_VERSION").to_string(),
        audio_backend: format!("cpal {}", CPAL_VERSION),
        binding: "0.7.1".to_string(),
    }
}

// ---------------------------------------------------------------------------
// MicrophoneInfo / SpeakerInfo: read-only Python wrappers around the core
// device structs. The Rust structs are named DeviceInfo / OutputDeviceInfo
// but are exposed to Python under the microphone/speaker names via the
// pyclass `name` attribute. Both core structs are #[non_exhaustive], so the
// binding's pyclasses are the stable Python-facing surface.
// ---------------------------------------------------------------------------

#[pyclass(name = "MicrophoneInfo", module = "decibri._decibri", frozen)]
struct DeviceInfo {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    max_input_channels: u16,
    #[pyo3(get)]
    default_sample_rate: u32,
    #[pyo3(get)]
    is_default: bool,
}

#[pymethods]
impl DeviceInfo {
    fn __repr__(&self) -> String {
        format!(
            "MicrophoneInfo(index={}, name='{}', id='{}', max_input_channels={}, default_sample_rate={}, is_default={})",
            self.index, self.name, self.id, self.max_input_channels, self.default_sample_rate, self.is_default
        )
    }
}

impl From<CoreDeviceInfo> for DeviceInfo {
    fn from(d: CoreDeviceInfo) -> Self {
        DeviceInfo {
            index: d.index,
            name: d.name,
            id: d.id,
            max_input_channels: d.max_input_channels,
            default_sample_rate: d.default_sample_rate,
            is_default: d.is_default,
        }
    }
}

#[pyclass(name = "SpeakerInfo", module = "decibri._decibri", frozen)]
struct OutputDeviceInfo {
    #[pyo3(get)]
    index: usize,
    #[pyo3(get)]
    name: String,
    #[pyo3(get)]
    id: String,
    #[pyo3(get)]
    max_output_channels: u16,
    #[pyo3(get)]
    default_sample_rate: u32,
    #[pyo3(get)]
    is_default: bool,
}

#[pymethods]
impl OutputDeviceInfo {
    fn __repr__(&self) -> String {
        format!(
            "SpeakerInfo(index={}, name='{}', id='{}', max_output_channels={}, default_sample_rate={}, is_default={})",
            self.index, self.name, self.id, self.max_output_channels, self.default_sample_rate, self.is_default
        )
    }
}

impl From<CoreOutputDeviceInfo> for OutputDeviceInfo {
    fn from(d: CoreOutputDeviceInfo) -> Self {
        OutputDeviceInfo {
            index: d.index,
            name: d.name,
            id: d.id,
            max_output_channels: d.max_output_channels,
            default_sample_rate: d.default_sample_rate,
            is_default: d.is_default,
        }
    }
}

// ---------------------------------------------------------------------------
// Device selector helper: translates Python-side device parameter into the
// core's DeviceSelector enum. Accepts None (Default), int (Index), or str
// (Name). The string form does not attempt id/name disambiguation; the core
// resolves both.
// ---------------------------------------------------------------------------

fn build_device_selector(device: Option<&Bound<'_, PyAny>>) -> PyResult<DeviceSelector> {
    let Some(d) = device else {
        return Ok(DeviceSelector::Default);
    };
    if d.is_none() {
        return Ok(DeviceSelector::Default);
    }
    if let Ok(idx) = d.extract::<usize>() {
        return Ok(DeviceSelector::Index(idx));
    }
    if let Ok(s) = d.extract::<String>() {
        return Ok(DeviceSelector::Name(s));
    }
    Err(PyValueError::new_err("device must be None, int, or str"))
}

// ---------------------------------------------------------------------------
// Helpers: encode an audio chunk's Vec<f32> into either a Python
// bytes object (numpy=False, default) or a numpy.ndarray (numpy=True).
// Used by both the sync MicrophoneBridge::read and the async
// AsyncMicrophoneBridge::read paths; the async path constructs the Python
// object outside spawn_blocking because Bound<'py, ...> is GIL-bound.
//
// Memory model: rust-numpy 0.28.0's `into_pyarray` performs a memcpy at
// the boundary
// (the Vec is consumed; data is copied into a freshly-allocated NumPy
// buffer). For typical audio chunks (~1-3 KB) the cost is nanoseconds.
// ---------------------------------------------------------------------------

/// Encode a chunk's f32 sample data into a Python bytes object using the
/// existing decibri::sample byte-level conversion helpers. Default path
/// (numpy=False); preserves the wire format byte-for-byte.
fn encode_chunk_bytes<'py>(
    py: Python<'py>,
    data: &[f32],
    format: BindingSampleFormat,
) -> Bound<'py, PyAny> {
    let bytes = match format {
        BindingSampleFormat::Int16 => f32_to_i16_le_bytes(data),
        BindingSampleFormat::Float32 => f32_to_f32_le_bytes(data),
    };
    PyBytes::new(py, &bytes).into_any()
}

/// Encode a chunk's f32 sample data into a numpy.ndarray. The dtype
/// matches the configured format (int16 -> np.int16, float32 ->
/// np.float32). The shape matches the channel count: 1-D `(N,)` for
/// mono, 2-D `(N, channels)` for multi-channel (interleaved cpal data
/// reshaped via `Array2::from_shape_vec`).
fn encode_chunk_numpy<'py>(
    py: Python<'py>,
    data: Vec<f32>,
    format: BindingSampleFormat,
    channels: u16,
) -> PyResult<Bound<'py, PyAny>> {
    let channels = channels as usize;
    match format {
        BindingSampleFormat::Int16 => {
            // Per-sample f32 -> i16 conversion matches f32_to_i16_le_bytes
            // semantics: clamp into [i16::MIN, i16::MAX] after scaling by
            // 32768.0. Avoids the bytes round-trip for the numpy path.
            let i16_vec: Vec<i16> = data
                .iter()
                .map(|&s| (s * 32768.0).clamp(i16::MIN as f32, i16::MAX as f32) as i16)
                .collect();
            if channels == 1 {
                Ok(i16_vec.into_pyarray(py).into_any())
            } else {
                let n = i16_vec.len() / channels;
                let arr2 = Array2::from_shape_vec((n, channels), i16_vec)
                    .map_err(|e| PyRuntimeError::new_err(format!("ndarray shape error: {e}")))?;
                Ok(arr2.into_pyarray(py).into_any())
            }
        }
        BindingSampleFormat::Float32 => {
            if channels == 1 {
                Ok(data.into_pyarray(py).into_any())
            } else {
                let n = data.len() / channels;
                let arr2 = Array2::from_shape_vec((n, channels), data)
                    .map_err(|e| PyRuntimeError::new_err(format!("ndarray shape error: {e}")))?;
                Ok(arr2.into_pyarray(py).into_any())
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Empirical smoke: pyo3-async-runtimes integration probe.
//
// pyo3-async-runtimes' docs assume a binary entry point with
// #[pyo3_async_runtimes::tokio::main]. Microphone is a cdylib (Python extension
// module) with no main() to attribute, so we rely on the crate's lazy-init
// path. This pyfunction is the persistent regression test that the lazy init
// works in our context. If
// a future pyo3-async-runtimes upgrade breaks the runtime init or
// cancellation propagation primitives, the matching tests in
// tests/test_async_smoke.py fail first and surface the regression at the
// build-pipeline level rather than at AsyncMicrophone integration time.
//
// Returns a Python awaitable that resolves to 42 after a 50 ms tokio sleep.
// Underscored name marks it as internal; not part of the public API.
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "_async_smoke")]
fn async_smoke(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    future_into_py(py, async {
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        Ok(42_i64)
    })
}

// ---------------------------------------------------------------------------
// Empirical smoke: rust-numpy 0.28.0 integration probe.
//
// Verifies that:
//   1. rust-numpy 0.28.0 builds and runs cleanly in our extension-module
//      context with PyO3 0.28.x.
//   2. `Vec<i16>::into_pyarray(py)` produces a numpy.ndarray with the
//      expected dtype, shape, and values.
//   3. The returned ndarray's `flags.owndata == True`, proving the
//      ownership-transfer claim
//      (no memcpy at the Rust-Python boundary; NumPy takes ownership
//      of the Rust Vec's heap allocation directly).
//
// API choice: rust-numpy 0.28.0's `IntoPyArray::into_pyarray(self, py)`
// trait method consumes the Vec and transfers ownership to NumPy. This
// is the recommended ownership-transfer path. Older rust-numpy versions
// used `PyArray::from_vec_bound`; the trait-based approach in 0.28.0 is
// the same operation under a cleaner API.
//
// This pyfunction stays in the codebase as persistent regression coverage:
// if a future
// rust-numpy upgrade breaks `from_vec` / `into_pyarray` semantics or
// pyo3 0.28 compat, the tests in tests/test_numpy_smoke.py fail first
// and surface the regression at the build-pipeline level.
//
// Returns a 1-D numpy.ndarray of dtype np.int16 with values [0, 1, 2, 3, 4].
// Underscored name marks it as internal; not part of the public API.
// ---------------------------------------------------------------------------

#[pyfunction]
#[pyo3(name = "_numpy_smoke")]
fn numpy_smoke(py: Python<'_>) -> Bound<'_, PyArray1<i16>> {
    let vec: Vec<i16> = vec![0, 1, 2, 3, 4];
    vec.into_pyarray(py)
}

// ---------------------------------------------------------------------------
// MicrophoneBridge: stateful capture pyclass.
//
// This pyclass is `Send + 'static` and may cross thread boundaries. The
// compile-time assertion at the top of this module enforces the property;
// do not add a field whose type lacks `Send + 'static` without first
// removing the assertion (which would also block the async work).
// Sendability comes from the Rust core's `MicrophoneStream` already being
// `Send` (see `crates/decibri/src/lib.rs` for the contract and the
// `test_microphone_stream_is_send_and_sync` guard in
// `crates/decibri/src/microphone.rs` for the unconditional compile-time
// assertion). The bridge holds the stream directly; no pump thread or
// channel proxy is needed.
//
// Owns:
//   - capture_config: MicrophoneConfig (built from constructor args, retained
//     for diagnostic introspection)
//   - format: BindingSampleFormat (parsed from constructor 'format' arg)
//   - vad_holdoff_ms: u32 (inert; wrapper layer reads this if it wants to)
//   - active: Mutex<Option<ActiveCapture>> (the open Microphone plus an
//     Arc<MicrophoneStream>; None until start(), cleared by stop())
//   - vad: Mutex<Option<SileroVad>> (None unless constructor's vad=True AND
//     vad_mode=="silero"; per VAD Option (b)+(h) decision)
//   - energy_vad: bool (vad=True AND vad_mode=="energy"; mutually exclusive
//     with `vad`. read() computes the energy RMS on the pre-enhancement tap)
//   - last_vad_probability: AtomicU32 (most recent score as f32 bits: the
//     Silero probability in silero mode or the energy RMS in energy mode;
//     updated on each read() that runs a detector)
//
// The mutable state lives behind interior-mutability primitives so read() and
// stop() are both `&self`. read() takes only a transient lock to clone the
// Arc<MicrophoneStream>, never holding it across the blocking next_chunk, so a
// concurrent stop() can take the same lock, call the core stream's stop()
// (which wakes a parked next_chunk in ~20ms), and clear the state. This is the
// shape the Node binding already uses, and it is what lets stop() safely
// interrupt a read() that is in flight on another thread (sync) or task
// (async) instead of panicking on a borrow or deadlocking on a lock.
// ---------------------------------------------------------------------------

/// Lock a `std::sync::Mutex`, recovering the guard if a previous holder
/// panicked while holding it. The critical sections here are short and
/// panic-free, so poisoning is not expected; recovering rather than
/// propagating keeps a stray panic from turning every later read()/stop()
/// into a poison panic. Mirrors the poison tolerance in the core's stop().
fn lock_recover<T>(m: &StdMutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

/// The live capture state: the open `Microphone` (kept alive so the device
/// stays open for the stream's lifetime) and a shared handle to its
/// `MicrophoneStream`. The stream is an `Arc` so stop() can hold a handle and
/// call the core stop() out of band of a read() parked in next_chunk on
/// another thread.
struct ActiveCapture {
    _capture: Microphone,
    stream: Arc<MicrophoneStream>,
}

#[pyclass(module = "decibri._decibri")]
struct MicrophoneBridge {
    capture_config: MicrophoneConfig,
    format: BindingSampleFormat,
    vad_holdoff_ms: u32,
    active: StdMutex<Option<ActiveCapture>>,
    vad: StdMutex<Option<SileroVad>>,
    // Energy VAD active (vad=True and vad_mode=="energy"). Mutually exclusive
    // with `vad` (Silero): read() computes the energy score (RMS of the
    // pre-enhancement tap) into `last_vad_probability` instead of a Silero
    // probability. Set once at construction; the read path never mutates it.
    energy_vad: bool,
    last_vad_probability: AtomicU32,
    /// When true, `read()` returns a numpy.ndarray instead of
    /// a bytes object. Constructor flag; per-instance commitment for
    /// the iterator's lifetime (the user picks one return type at
    /// construction). Output bridges do NOT have this flag; they
    /// duck-type per-call.
    numpy: bool,
}

// Internal helpers on MicrophoneBridge; not exposed to Python.
// Used by the public `read` pymethod (same module) and by
// `AsyncMicrophoneBridge::read` (which calls these on the shared inner
// bridge inside spawn_blocking to extract owned Vec data + metadata,
// then constructs the Python object outside the spawn_blocking
// boundary because `Bound<'py, ...>` is GIL-bound).
impl MicrophoneBridge {
    /// Read the next audio chunk, run VAD inference if configured, and
    /// return the raw `Vec<f32>` data plus the output channel count. No Python
    /// object construction; returning the raw data lets the caller decide
    /// between PyBytes and PyArray encoding without re-running the cpal read or
    /// VAD pass, and the channel count drives the ndarray shape.
    fn read_raw(
        &self,
        py: Python<'_>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<(Vec<f32>, u16)>> {
        let timeout = timeout_ms.map(Duration::from_millis);

        // Clone a handle to the core stream under a tiny lock, then release the
        // lock BEFORE the blocking read. Holding only an Arc clone (not the
        // `active` lock) across next_chunk is what lets a concurrent stop()
        // take the lock and interrupt a parked read.
        let stream = {
            let guard = lock_recover(&self.active);
            match guard.as_ref() {
                Some(active) => Arc::clone(&active.stream),
                None => {
                    return Err(raise_named(
                        py,
                        "MicrophoneStreamClosed",
                        "capture is not running",
                    ))
                }
            }
        };

        // Requested block size in interleaved OUTPUT samples: frames_per_buffer
        // times the stream's output channel count. The engine's normalize chain
        // makes a multichannel device mono, so this is counted in output (not
        // device) channels; the core re-blocks to exactly this size on every
        // platform.
        let output_channels = stream.channels();
        let samples = self.capture_config.frames_per_buffer as usize * output_channels as usize;

        // Release GIL for the blocking next_chunk call. The `active` lock is
        // not held here, so stop() can proceed concurrently and wake us.
        let result: Result<Option<AudioChunk>, CoreDecibriError> =
            py.detach(|| stream.next_chunk(samples, timeout));

        let Some(chunk) = result.map_err(|e| to_py_err(py, e))? else {
            return Ok(None);
        };

        // Run VAD on the signal BEFORE the opt-in enhancement step. Only read()
        // touches `vad`, so this lock never contends with stop(). Drain the
        // pre-enhancement tap once (when a transform is active); otherwise the
        // delivered chunk already is that signal. Reading the pre-enhancement
        // signal means enabling enhancement does not change detection in either
        // mode. Called once per delivered chunk so the tap drains in lockstep.
        // `chunk.data` stays untouched and is what the caller receives below.
        {
            let mut vad_guard = lock_recover(&self.vad);
            let silero = vad_guard.as_mut();
            if self.energy_vad || silero.is_some() {
                let pre_enh = stream.vad_input(chunk.data.len());
                let vad_frame: &[f32] = pre_enh.as_deref().unwrap_or(&chunk.data);
                // VAD models a single channel. Downmix interleaved multichannel
                // audio to mono first so consecutive channels are not misread as
                // successive mono samples.
                let downmixed;
                let vad_input: &[f32] = if chunk.channels > 1 {
                    downmixed = downmix_to_mono(vad_frame, chunk.channels);
                    &downmixed
                } else {
                    vad_frame
                };
                if let Some(vad) = silero {
                    let result = py
                        .detach(|| vad.process(vad_input))
                        .map_err(|e| to_py_err(py, e))?;
                    self.last_vad_probability
                        .store(result.probability.to_bits(), Ordering::Relaxed);
                } else {
                    // Energy mode: the score is the RMS of the pre-enhancement
                    // signal, on the same [0, 1] scale the energy threshold
                    // compares against.
                    let score = rms(vad_input);
                    self.last_vad_probability
                        .store(score.to_bits(), Ordering::Relaxed);
                }
            }
        }

        Ok(Some((chunk.data, chunk.channels)))
    }

    /// numpy-mode flag accessor for the async wrapper's encoding step.
    fn numpy_mode(&self) -> bool {
        self.numpy
    }

    /// Configured sample format accessor for the async wrapper.
    fn binding_format(&self) -> BindingSampleFormat {
        self.format
    }
}

#[pymethods]
impl MicrophoneBridge {
    #[new]
    #[pyo3(signature = (
        sample_rate,
        channels,
        frames_per_buffer,
        format,
        device = None,
        vad = false,
        vad_threshold = 0.5_f32,
        vad_mode = "silero".to_string(),
        vad_holdoff = 0_u32,
        model_path = None,
        numpy = false,
        ort_library_path = None,
        denoise = None,
        denoise_model_path = None,
        highpass = None,
        agc = None,
        limiter = None,
        dc_removal = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        sample_rate: u32,
        channels: u16,
        frames_per_buffer: u32,
        format: String,
        device: Option<Bound<'_, PyAny>>,
        vad: bool,
        vad_threshold: f32,
        vad_mode: String,
        vad_holdoff: u32,
        model_path: Option<PathBuf>,
        numpy: bool,
        ort_library_path: Option<PathBuf>,
        denoise: Option<String>,
        denoise_model_path: Option<PathBuf>,
        highpass: Option<i64>,
        agc: Option<i8>,
        limiter: Option<f32>,
        dc_removal: bool,
    ) -> PyResult<Self> {
        let parsed_format = parse_sample_format(&format).map_err(|e| to_py_err(py, e))?;
        let device_selector = build_device_selector(device.as_ref())?;

        // `MicrophoneConfig` is `#[non_exhaustive]`: default-construct then
        // assign the public fields rather than using a struct literal.
        let mut capture_config = MicrophoneConfig::default();
        capture_config.sample_rate = sample_rate;
        capture_config.channels = channels;
        capture_config.frames_per_buffer = frames_per_buffer;
        capture_config.device = device_selector;

        // DC removal: a same-length one-pole DC-blocking high-pass, the first
        // transform stage (before denoise). `false` (the default) leaves it off,
        // a byte-identical no-op. A plain bool toggle, so there is no range check,
        // only the forward to the core config field.
        capture_config.dc_removal = dc_removal;

        // Denoise: map the closed-set model name to the core selector and attach
        // the bundled model path the wrapper resolved. Absent leaves denoise off
        // and the capture path unchanged. The model loads later, in start(), so
        // an unknown name fails here but a load failure surfaces from start().
        if let Some(name) = denoise.as_deref() {
            let model = match name {
                "fastenhancer-t" => DenoiseModel::FastEnhancerT,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "denoise must be 'fastenhancer-t', got '{other}'"
                    )))
                }
            };
            let mp = denoise_model_path.ok_or_else(|| {
                PyValueError::new_err("denoise_model_path is required when denoise is set")
            })?;
            capture_config.denoise = Some(model);
            capture_config.denoise_model_path = Some(mp);
            // The denoise stage loads through the ONNX seam, so hand the core the
            // bundled ORT dylib path the wrapper resolved (the same path the VAD
            // uses). Cloned because the VAD construction below also consumes it.
            // Without it a denoise-only capture could not find the bundled
            // runtime on a default install.
            capture_config.ort_library_path = ort_library_path.clone();
        }

        // High-pass: map the closed-set cutoff in Hz to the core selector.
        // Absent leaves the high-pass off and the capture path unchanged. Pure
        // DSP, so there is no model file or ORT to resolve, only the closed-set
        // check (the wrapper performs the same check and raises ValueError; this
        // is the native backstop).
        if let Some(hz) = highpass {
            let filter = match hz {
                80 => HighpassFilter::Hz80,
                100 => HighpassFilter::Hz100,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "highpass must be one of: 80, 100; got {other}"
                    )))
                }
            };
            capture_config.highpass = Some(filter);
        }

        // AGC: thread the dBFS target to the core level-control engine. The
        // wrapper performs the user-facing range check (a ValueError); the core
        // guards the same range in validate() (the load-bearing backstop, raised
        // as AgcTargetOutOfRange), so no inline check is needed here.
        capture_config.agc = agc;

        // Limiter: thread the dBFS ceiling to the core limiter stage. The wrapper
        // performs the user-facing range check (a ValueError); the core guards the
        // same range in validate() (the load-bearing backstop, raised as
        // LimiterCeilingOutOfRange), so no inline check is needed here.
        capture_config.limiter = limiter;

        // VAD construction gate (Option (b)+(h)). The bridge constructs
        // SileroVad only if vad=True AND vad_mode=="silero". The vad_mode
        // string is otherwise inert at this layer; the wrapper validates and
        // dispatches mode policy.
        let vad_instance = if vad && vad_mode == "silero" {
            let mp = model_path.ok_or_else(|| {
                PyValueError::new_err("model_path is required when vad=True and vad_mode='silero'")
            })?;
            // `VadConfig` is `#[non_exhaustive]`: default-construct then assign.
            let mut vad_config = VadConfig::default();
            vad_config.model_path = mp;
            vad_config.sample_rate = sample_rate;
            vad_config.threshold = vad_threshold;
            vad_config.ort_library_path = ort_library_path;
            Some(SileroVad::new(vad_config).map_err(|e| to_py_err(py, e))?)
        } else {
            None
        };

        // Energy VAD: enabled when vad=True and the mode is "energy" (mutually
        // exclusive with the Silero branch above, which constructs vad_instance).
        // The read path computes its RMS score on the pre-enhancement tap.
        let energy_vad = vad && vad_mode == "energy";

        Ok(MicrophoneBridge {
            capture_config,
            format: parsed_format,
            vad_holdoff_ms: vad_holdoff,
            active: StdMutex::new(None),
            vad: StdMutex::new(vad_instance),
            energy_vad,
            last_vad_probability: AtomicU32::new(0),
            numpy,
        })
    }

    fn start(&self, py: Python<'_>) -> PyResult<()> {
        let mut active = lock_recover(&self.active);
        if active.is_some() {
            return Err(raise_named(
                py,
                "AlreadyRunning",
                "capture is already running",
            ));
        }
        let capture = Microphone::new(self.capture_config.clone()).map_err(|e| to_py_err(py, e))?;
        let stream = capture.start().map_err(|e| to_py_err(py, e))?;
        *active = Some(ActiveCapture {
            _capture: capture,
            stream: Arc::new(stream),
        });
        Ok(())
    }

    fn stop(&self) -> PyResult<()> {
        // Take the live capture out under a tiny lock, then signal the core
        // stop OUTSIDE the lock. The core MicrophoneStream::stop() flips the
        // running flag and drops the cpal stream, disconnecting the capture
        // channel and waking any next_chunk parked in a concurrent read()
        // within ~20ms. A parked read holds only an Arc clone of the stream
        // (not the `active` lock), so taking the Option here never blocks on
        // it. Dropping `active` then releases the device.
        let active = lock_recover(&self.active).take();
        if let Some(active) = active {
            active.stream.stop();
        }
        Ok(())
    }

    fn close(&self) -> PyResult<()> {
        // Literal alias for stop, mirroring SpeakerBridge::close.
        // Bridge-level symmetry for users constructing MicrophoneBridge
        // directly (advanced use; surface in __all__).
        self.stop()
    }

    #[pyo3(signature = (timeout_ms = None))]
    fn read<'py>(
        &self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Bound<'py, PyAny>>> {
        let Some((data, channels)) = self.read_raw(py, timeout_ms)? else {
            return Ok(None);
        };
        // Branch on the constructor-set numpy flag. Default
        // (numpy=False) returns PyBytes (wire format); numpy=True
        // returns a numpy.ndarray with dtype matching format and shape
        // matching the OUTPUT channel count (mono after the normalize chain).
        // Both arms are returned as Bound<PyAny> so the union is expressed at
        // the Python boundary.
        if self.numpy {
            Ok(Some(encode_chunk_numpy(py, data, self.format, channels)?))
        } else {
            Ok(Some(encode_chunk_bytes(py, &data, self.format)))
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(slf: PyRef<'_, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Block indefinitely on next_chunk by passing None timeout. If the
        // stream returns None (closed), raise StopIteration via PyO3's
        // None-as-StopIteration convention by returning the appropriate err.
        // Return type is Bound<PyAny> rather than PyBytes because the
        // numpy=True flag makes the iterator yield ndarrays in numpy mode.
        let result = slf.read(py, None)?;
        match result {
            Some(b) => Ok(b),
            None => Err(pyo3::exceptions::PyStopIteration::new_err(())),
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyResult<PyRef<'_, Self>> {
        let py = slf.py();
        slf.start(py)?;
        Ok(slf)
    }

    fn __exit__(
        &self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.stop()?;
        Ok(false)
    }

    #[getter]
    fn is_open(&self) -> bool {
        lock_recover(&self.active).is_some()
    }

    #[getter]
    fn vad_probability(&self) -> f32 {
        f32::from_bits(self.last_vad_probability.load(Ordering::Relaxed))
    }

    /// Returns inert vad_holdoff_ms set at construction; wrapper reads this
    /// to apply policy. Pure storage; bridge does no holdoff logic.
    #[getter]
    fn vad_holdoff_ms(&self) -> u32 {
        self.vad_holdoff_ms
    }

    /// Number of capture buffers dropped because the consumer could not keep
    /// pace (the core stream's overrun counter). Returns 0 while the consumer
    /// keeps up or when no stream is active.
    #[getter]
    fn overrun_count(&self) -> u64 {
        match lock_recover(&self.active).as_ref() {
            Some(active) => active.stream.overrun_count(),
            None => 0,
        }
    }

    #[staticmethod]
    fn devices(py: Python<'_>) -> PyResult<Vec<DeviceInfo>> {
        let core_devices = py.detach(input_devices).map_err(|e| to_py_err(py, e))?;
        Ok(core_devices.into_iter().map(DeviceInfo::from).collect())
    }

    #[staticmethod]
    fn version() -> VersionInfo {
        build_version_info()
    }
}

// ---------------------------------------------------------------------------
// SpeakerBridge: stateful output pyclass.
//
// This pyclass is `Send + 'static` and may cross thread boundaries; the
// compile-time assertion at the top of this module enforces the property.
// Sendability comes from the Rust core's `SpeakerStream` being `Send` per
// the documented contract at `crates/decibri/src/lib.rs`. There
// is no analogous compile-time assertion in `speaker.rs` (microphone.rs has
// one at line 481); CI on the four-platform matrix is the empirical gate.
// Do not add a field whose type lacks `Send + 'static` without first
// removing the module-level assertion.
//
// Owns:
//   - output_config: SpeakerConfig (no frames_per_buffer)
//   - format: BindingSampleFormat
//   - output: Option<Speaker>
//   - stream: Option<SpeakerStream>
// ---------------------------------------------------------------------------

#[pyclass(module = "decibri._decibri")]
struct SpeakerBridge {
    output_config: SpeakerConfig,
    format: BindingSampleFormat,
    output: Option<Speaker>,
    stream: Option<SpeakerStream>,
}

// Internal helpers for SpeakerBridge. Not exposed to
// Python. Each `write_*_samples` helper takes an owned slice or vec
// of typed samples (i16 or f32), converts to the f32 vec the cpal
// stream expects, and pushes through the GIL-released send. The
// public `write` pymethod dispatches on the input PyAny type and
// calls the appropriate helper.
impl SpeakerBridge {
    fn require_format(
        &self,
        expected: BindingSampleFormat,
        ndarray_dtype_name: &str,
        ndarray_shape_name: &str,
    ) -> PyResult<()> {
        if self.format == expected {
            Ok(())
        } else {
            let configured = match self.format {
                BindingSampleFormat::Int16 => "int16",
                BindingSampleFormat::Float32 => "float32",
            };
            Err(PyTypeError::new_err(format!(
                "format='{configured}' configured but {ndarray_shape_name} ndarray \
                 dtype is {ndarray_dtype_name}; convert with arr.astype(np.{configured}) \
                 or construct Speaker with format='{ndarray_dtype_name}'"
            )))
        }
    }

    fn write_bytes_internal(&mut self, py: Python<'_>, samples: &[u8]) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| raise_named(py, "SpeakerStreamClosed", "output is not running"))?;
        let owned_samples: Vec<u8> = samples.to_vec();
        let samples_f32 = match self.format {
            BindingSampleFormat::Int16 => py.detach(|| i16_le_bytes_to_f32(&owned_samples)),
            BindingSampleFormat::Float32 => py.detach(|| f32_le_bytes_to_f32(&owned_samples)),
        };
        py.detach(|| stream.send(samples_f32))
            .map_err(|e| to_py_err(py, e))?;
        Ok(())
    }

    fn write_int16_samples(&mut self, py: Python<'_>, samples: &[i16]) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| raise_named(py, "SpeakerStreamClosed", "output is not running"))?;
        // Per-sample i16 -> f32 conversion mirrors i16_le_bytes_to_f32:
        // divide by 32768.0 to map [i16::MIN, i16::MAX] into approximately
        // [-1.0, 1.0). Avoids the bytes round-trip for the numpy path.
        let owned: Vec<i16> = samples.to_vec();
        let samples_f32: Vec<f32> =
            py.detach(|| owned.iter().map(|&s| s as f32 / 32768.0).collect());
        py.detach(|| stream.send(samples_f32))
            .map_err(|e| to_py_err(py, e))?;
        Ok(())
    }

    fn write_float32_samples(&mut self, py: Python<'_>, samples: &[f32]) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| raise_named(py, "SpeakerStreamClosed", "output is not running"))?;
        let samples_f32: Vec<f32> = samples.to_vec();
        py.detach(|| stream.send(samples_f32))
            .map_err(|e| to_py_err(py, e))?;
        Ok(())
    }
}

#[pymethods]
impl SpeakerBridge {
    #[new]
    #[pyo3(signature = (sample_rate, channels, format, device = None))]
    fn new(
        py: Python<'_>,
        sample_rate: u32,
        channels: u16,
        format: String,
        device: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let parsed_format = parse_sample_format(&format).map_err(|e| to_py_err(py, e))?;
        let device_selector = build_device_selector(device.as_ref())?;

        // `SpeakerConfig` is `#[non_exhaustive]`: default-construct then assign.
        let mut output_config = SpeakerConfig::default();
        output_config.sample_rate = sample_rate;
        output_config.channels = channels;
        output_config.device = device_selector;

        Ok(SpeakerBridge {
            output_config,
            format: parsed_format,
            output: None,
            stream: None,
        })
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.stream.is_some() {
            return Err(raise_named(
                py,
                "AlreadyRunning",
                "output is already running",
            ));
        }
        let output = Speaker::new(self.output_config.clone()).map_err(|e| to_py_err(py, e))?;
        let stream = output.start().map_err(|e| to_py_err(py, e))?;
        self.output = Some(output);
        self.stream = Some(stream);
        Ok(())
    }

    fn write(&mut self, py: Python<'_>, samples: &Bound<'_, PyAny>) -> PyResult<()> {
        // Duck-typed dispatch. Accept Python `bytes` (the wire
        // format; cheap path, hit by existing users), 1-D and 2-D
        // numpy.ndarray with dtype matching the configured format, and
        // raise TypeError on anything else with a clear message. Output
        // uses per-call dispatch (no constructor numpy flag) because each
        // write is independent and stateless.

        // Cheap path: bytes (existing wire format).
        if let Ok(byte_slice) = samples.extract::<&[u8]>() {
            return self.write_bytes_internal(py, byte_slice);
        }

        // numpy.ndarray paths. Order: PyArray1<i16>, PyArray1<f32>,
        // PyArray2<i16>, PyArray2<f32>. dtype must match the configured
        // BindingSampleFormat. Multi-channel ndarrays are flattened from
        // shape (N, channels) to interleaved Vec<f32> for cpal.
        if let Ok(arr) = samples.cast::<PyArray1<i16>>() {
            self.require_format(BindingSampleFormat::Int16, "int16", "1-D")?;
            let readonly = arr.readonly();
            let slice = readonly
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("ndarray must be C-contiguous: {e}")))?;
            return self.write_int16_samples(py, slice);
        }

        if let Ok(arr) = samples.cast::<PyArray1<f32>>() {
            self.require_format(BindingSampleFormat::Float32, "float32", "1-D")?;
            let readonly = arr.readonly();
            let slice = readonly
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("ndarray must be C-contiguous: {e}")))?;
            return self.write_float32_samples(py, slice);
        }

        if let Ok(arr) = samples.cast::<PyArray2<i16>>() {
            self.require_format(BindingSampleFormat::Int16, "int16", "2-D")?;
            let readonly = arr.readonly();
            let slice = readonly.as_slice().map_err(|e| {
                PyValueError::new_err(format!("2-D ndarray must be C-contiguous: {e}"))
            })?;
            return self.write_int16_samples(py, slice);
        }

        if let Ok(arr) = samples.cast::<PyArray2<f32>>() {
            self.require_format(BindingSampleFormat::Float32, "float32", "2-D")?;
            let readonly = arr.readonly();
            let slice = readonly.as_slice().map_err(|e| {
                PyValueError::new_err(format!("2-D ndarray must be C-contiguous: {e}"))
            })?;
            return self.write_float32_samples(py, slice);
        }

        Err(PyTypeError::new_err(
            "samples must be bytes or numpy.ndarray with dtype matching the \
             configured format (int16 or float32); 1-D for mono or 2-D \
             (N, channels) for multi-channel",
        ))
    }

    fn drain(&mut self, py: Python<'_>) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| raise_named(py, "SpeakerStreamClosed", "output is not running"))?;
        py.detach(|| stream.drain());
        Ok(())
    }

    fn stop(&mut self) -> PyResult<()> {
        self.stream = None;
        self.output = None;
        Ok(())
    }

    fn close(&mut self) -> PyResult<()> {
        self.stop()
    }

    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        let py = slf.py();
        slf.start(py)?;
        Ok(slf)
    }

    fn __exit__(
        &mut self,
        _exc_type: Option<&Bound<'_, PyAny>>,
        _exc_value: Option<&Bound<'_, PyAny>>,
        _traceback: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.stop()?;
        Ok(false)
    }

    #[getter]
    fn is_playing(&self) -> bool {
        self.stream.is_some()
    }

    #[staticmethod]
    fn devices(py: Python<'_>) -> PyResult<Vec<OutputDeviceInfo>> {
        let core_devices = py.detach(output_devices).map_err(|e| to_py_err(py, e))?;
        Ok(core_devices
            .into_iter()
            .map(OutputDeviceInfo::from)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// AsyncMicrophoneBridge: async wrapper around MicrophoneBridge.
//
// Holds an `Arc<MicrophoneBridge>` (the sync mic bridge is internally
// thread-safe, so no outer Mutex is needed) and exposes the same blocking
// methods as the sync bridge, but async. Each blocking method dispatches the
// underlying sync work to a Tokio worker thread via
// `tokio::task::spawn_blocking`, keeping the runtime thread responsive.
//
// The compile-time `Send + 'static` assertion at the top of this module is
// extended to cover this pyclass; PyO3 + pyo3-async-runtimes capture into
// `future_into_py` requires it.
//
// Construction is sync (Python class instantiation is always sync). The
// constructor's parameter list mirrors `MicrophoneBridge::new` exactly; users
// instantiate with the same kwargs and accept that ORT model loading
// happens during construction (potentially slow). The post-construction
// async surface (`start`, `stop`, `read`, etc.) is what the
// `AsyncMicrophone` Python wrapper proxies through.
//
// stop() no longer serializes behind an in-flight read(): the sync bridge's
// read() and stop() are both `&self` and share the core stream by `Arc`, so a
// concurrent stop() calls the core MicrophoneStream::stop() (waking a parked
// next_chunk in ~20ms) without contending on any bridge-wide lock.
// spawn_blocking does not cooperatively cancel OS threads; cancellation
// surfaces immediately to Python while the spawned thread runs to completion.
// For `read`, the thread returns once the current chunk arrives or stop()
// closes the stream; for `drain`, the thread waits for the cpal output to
// flush (potentially seconds). The Python coroutine's `CancelledError`
// arrives immediately regardless.
//
// Iterator (`__iter__` / `__next__`) and context manager (`__enter__` /
// `__exit__`) protocols are deliberately NOT implemented on this Rust
// pyclass; those are sync-only protocols. The `AsyncMicrophone` Python
// wrapper layers `__aiter__` / `__anext__` and `__aenter__` / `__aexit__`
// on top of the async `read` / `start` / `stop` methods exposed here.
// ---------------------------------------------------------------------------

#[pyclass(module = "decibri._decibri")]
pub(crate) struct AsyncMicrophoneBridge {
    inner: Arc<MicrophoneBridge>,
    // Lock-free mirror of the bridge's open state. The public is_open getter
    // is awaitable (returns a future), which makes it unusable from a
    // synchronous Python property. The wrapper class needs a sync property to
    // keep API symmetry with sync Microphone.is_open; this AtomicBool is its
    // source of truth. Updated by start() and stop() after the inner mutation
    // succeeds; read by the is_open_sync getter without locking.
    is_open_atomic: Arc<AtomicBool>,
}

#[pymethods]
impl AsyncMicrophoneBridge {
    #[new]
    #[pyo3(signature = (
        sample_rate,
        channels,
        frames_per_buffer,
        format,
        device = None,
        vad = false,
        vad_threshold = 0.5_f32,
        vad_mode = "silero".to_string(),
        vad_holdoff = 0_u32,
        model_path = None,
        numpy = false,
        ort_library_path = None,
        denoise = None,
        denoise_model_path = None,
        highpass = None,
        agc = None,
        limiter = None,
        dc_removal = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        sample_rate: u32,
        channels: u16,
        frames_per_buffer: u32,
        format: String,
        device: Option<Bound<'_, PyAny>>,
        vad: bool,
        vad_threshold: f32,
        vad_mode: String,
        vad_holdoff: u32,
        model_path: Option<PathBuf>,
        numpy: bool,
        ort_library_path: Option<PathBuf>,
        denoise: Option<String>,
        denoise_model_path: Option<PathBuf>,
        highpass: Option<i64>,
        agc: Option<i8>,
        limiter: Option<f32>,
        dc_removal: bool,
    ) -> PyResult<Self> {
        let inner = MicrophoneBridge::new(
            py,
            sample_rate,
            channels,
            frames_per_buffer,
            format,
            device,
            vad,
            vad_threshold,
            vad_mode,
            vad_holdoff,
            model_path,
            numpy,
            ort_library_path,
            denoise,
            denoise_model_path,
            highpass,
            agc,
            limiter,
            dc_removal,
        )?;
        Ok(AsyncMicrophoneBridge {
            inner: Arc::new(inner),
            is_open_atomic: Arc::new(AtomicBool::new(false)),
        })
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let is_open_atomic = Arc::clone(&self.is_open_atomic);
        future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                Python::attach(|py| inner.start(py))?;
                // Only set the atomic AFTER the inner
                // start() succeeds. If start() returns Err, the ? operator
                // returns early and the atomic stays false.
                is_open_atomic.store(true, Ordering::Release);
                Ok(())
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }

    fn stop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let is_open_atomic = Arc::clone(&self.is_open_atomic);
        future_into_py(py, async move {
            // Run on a blocking worker: stop() calls the core stream stop(),
            // which blocks briefly while the audio thread tears down. It needs
            // no bridge-wide lock, so it proceeds even while a read() is parked
            // in next_chunk, waking that read within ~20ms.
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                inner.stop()?;
                // Clear the atomic after the inner
                // stop() succeeds. inner.stop() itself is infallible, but
                // mirroring the pattern from start() keeps the symmetry.
                is_open_atomic.store(false, Ordering::Release);
                Ok(())
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // Close is a literal alias for stop, mirroring
        // AsyncSpeakerBridge::close. Bridge-level symmetry for users
        // constructing AsyncMicrophoneBridge directly.
        self.stop(py)
    }

    #[pyo3(signature = (timeout_ms = None))]
    fn read<'py>(&self, py: Python<'py>, timeout_ms: Option<u64>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move {
            // Extract Vec<f32> + metadata inside spawn_blocking,
            // construct the Python object (PyBytes or PyArray) outside.
            // Bound<'py, ...> is GIL-bound and not Send + 'static, so
            // it cannot cross the spawn_blocking boundary. Vec<f32>,
            // u16, bool, and BindingSampleFormat are all Send + 'static.
            //
            // read_raw is `&self` and holds no bridge-wide lock across the
            // blocking next_chunk, so a concurrent stop() interrupts it.
            let extracted: Option<AsyncReadOutput> =
                tokio::task::spawn_blocking(move || -> PyResult<Option<AsyncReadOutput>> {
                    Python::attach(|py| {
                        let opt = inner.read_raw(py, timeout_ms)?;
                        Ok(opt.map(|(data, channels)| {
                            (data, inner.binding_format(), channels, inner.numpy_mode())
                        }))
                    })
                })
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))??;
            Python::attach(|py| -> PyResult<Option<Py<PyAny>>> {
                let Some((data, format, channels, numpy)) = extracted else {
                    return Ok(None);
                };
                let bound = if numpy {
                    encode_chunk_numpy(py, data, format, channels)?
                } else {
                    encode_chunk_bytes(py, &data, format)
                };
                Ok(Some(bound.unbind()))
            })
        })
    }

    #[getter]
    fn is_open<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move {
            // Non-blocking getter: the bridge's is_open() takes only a tiny
            // lock on the capture state. No spawn_blocking needed.
            Ok(inner.is_open())
        })
    }

    #[getter]
    fn is_open_sync(&self) -> bool {
        // Lock-free sync getter for the wrapper's
        // synchronous AsyncMicrophone.is_open property. Reads the atomic
        // mirror updated by start() and stop(); no Mutex acquisition,
        // no awaiting, safe to call from any thread without a runtime.
        self.is_open_atomic.load(Ordering::Acquire)
    }

    #[getter]
    fn vad_probability<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move { Ok(inner.vad_probability()) })
    }

    #[getter]
    fn vad_holdoff_ms<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move { Ok(inner.vad_holdoff_ms()) })
    }

    #[staticmethod]
    fn devices(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        future_into_py(py, async move {
            tokio::task::spawn_blocking(|| -> PyResult<Vec<DeviceInfo>> {
                Python::attach(MicrophoneBridge::devices)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }

    #[staticmethod]
    fn version(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        // version() is non-blocking (reads compile-time constants); no
        // spawn_blocking, just wrap into a resolved future.
        future_into_py(py, async move { Ok(MicrophoneBridge::version()) })
    }
}

// ---------------------------------------------------------------------------
// AsyncSpeakerBridge: async wrapper around SpeakerBridge.
//
// Same architecture as AsyncMicrophoneBridge:
// `Arc<tokio::sync::Mutex<SpeakerBridge>>` with each blocking
// method dispatched via spawn_blocking.
//
// The cancellation behaviour applies most visibly to `drain`: the cpal
// output buffer can hold multiple seconds of pending audio; spawn_blocking
// does not abort the
// drain on Python cancellation, so the audio finishes playing even after
// the Python coroutine is cancelled. The Python coroutine sees
// CancelledError immediately; the audio finishes asynchronously to the
// caller's logic. This is documented in the AsyncSpeaker.drain docstring
// in the Python wrapper.
// ---------------------------------------------------------------------------

/// Owned bundle of (data, format, channels, numpy_flag) extracted from
/// a `MicrophoneBridge` for the async read path. All fields are
/// `Send + 'static` so they cross the `spawn_blocking` boundary.
type AsyncReadOutput = (Vec<f32>, BindingSampleFormat, u16, bool);

/// Owned input data for an async write call. Extracted from the input
/// `Bound<'_, PyAny>` while the GIL is held; the resulting variants are
/// all `Send + 'static` so they can cross the `spawn_blocking` boundary.
/// On the worker thread, the `dispatch` method re-acquires the GIL
/// (via `Python::attach`) and calls the appropriate sync helper on the
/// inner `SpeakerBridge`.
///
/// dtype validation against the configured format happens twice: once
/// at extraction time (via `require_format`) for the ndarray paths, and
/// again inside the sync helpers (defense-in-depth; cheap). The bytes
/// path skips the dtype check because bytes carry no dtype.
enum AsyncWriteData {
    Bytes(Vec<u8>),
    Int16Samples(Vec<i16>),
    Float32Samples(Vec<f32>),
}

impl AsyncWriteData {
    fn extract(samples: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(byte_slice) = samples.extract::<&[u8]>() {
            return Ok(AsyncWriteData::Bytes(byte_slice.to_vec()));
        }
        if let Ok(arr) = samples.cast::<PyArray1<i16>>() {
            let readonly = arr.readonly();
            let slice = readonly
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("ndarray must be C-contiguous: {e}")))?;
            return Ok(AsyncWriteData::Int16Samples(slice.to_vec()));
        }
        if let Ok(arr) = samples.cast::<PyArray1<f32>>() {
            let readonly = arr.readonly();
            let slice = readonly
                .as_slice()
                .map_err(|e| PyValueError::new_err(format!("ndarray must be C-contiguous: {e}")))?;
            return Ok(AsyncWriteData::Float32Samples(slice.to_vec()));
        }
        if let Ok(arr) = samples.cast::<PyArray2<i16>>() {
            let readonly = arr.readonly();
            let slice = readonly.as_slice().map_err(|e| {
                PyValueError::new_err(format!("2-D ndarray must be C-contiguous: {e}"))
            })?;
            return Ok(AsyncWriteData::Int16Samples(slice.to_vec()));
        }
        if let Ok(arr) = samples.cast::<PyArray2<f32>>() {
            let readonly = arr.readonly();
            let slice = readonly.as_slice().map_err(|e| {
                PyValueError::new_err(format!("2-D ndarray must be C-contiguous: {e}"))
            })?;
            return Ok(AsyncWriteData::Float32Samples(slice.to_vec()));
        }
        Err(PyTypeError::new_err(
            "samples must be bytes or numpy.ndarray with dtype int16 or float32; \
             1-D for mono or 2-D (N, channels) for multi-channel",
        ))
    }

    fn dispatch(self, py: Python<'_>, bridge: &mut SpeakerBridge) -> PyResult<()> {
        match self {
            AsyncWriteData::Bytes(v) => bridge.write_bytes_internal(py, &v),
            AsyncWriteData::Int16Samples(v) => {
                bridge.require_format(BindingSampleFormat::Int16, "int16", "1-D or 2-D")?;
                bridge.write_int16_samples(py, &v)
            }
            AsyncWriteData::Float32Samples(v) => {
                bridge.require_format(BindingSampleFormat::Float32, "float32", "1-D or 2-D")?;
                bridge.write_float32_samples(py, &v)
            }
        }
    }
}

#[pyclass(module = "decibri._decibri")]
pub(crate) struct AsyncSpeakerBridge {
    inner: Arc<Mutex<SpeakerBridge>>,
    // Lock-free mirror of inner.stream
    // state. Same shape as AsyncMicrophoneBridge.is_open_atomic; same
    // motivation (sync wrapper property needs lock-free truth).
    is_playing_atomic: Arc<AtomicBool>,
}

#[pymethods]
impl AsyncSpeakerBridge {
    #[new]
    #[pyo3(signature = (sample_rate, channels, format, device = None))]
    fn new(
        py: Python<'_>,
        sample_rate: u32,
        channels: u16,
        format: String,
        device: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let inner = SpeakerBridge::new(py, sample_rate, channels, format, device)?;
        Ok(AsyncSpeakerBridge {
            inner: Arc::new(Mutex::new(inner)),
            is_playing_atomic: Arc::new(AtomicBool::new(false)),
        })
    }

    fn start<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let is_playing_atomic = Arc::clone(&self.is_playing_atomic);
        future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                let mut bridge = inner.blocking_lock();
                Python::attach(|py| bridge.start(py))?;
                // Set atomic only after
                // inner.start() succeeds. ? returns early on Err.
                is_playing_atomic.store(true, Ordering::Release);
                Ok(())
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }

    fn write<'py>(
        &self,
        py: Python<'py>,
        samples: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyAny>> {
        // Duck-typed dispatch on input type. Extract owned Send
        // + 'static data while the GIL is held (PyReadonlyArray and
        // Bound<PyAny> are GIL-bound and cannot cross spawn_blocking),
        // then cross the boundary with the AsyncWriteData enum and
        // dispatch against the inner sync bridge inside spawn_blocking.
        let owned = AsyncWriteData::extract(samples)?;
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                let mut bridge = inner.blocking_lock();
                Python::attach(|py| owned.dispatch(py, &mut bridge))
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }

    fn drain<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                let mut bridge = inner.blocking_lock();
                Python::attach(|py| bridge.drain(py))
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }

    fn stop<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        let is_playing_atomic = Arc::clone(&self.is_playing_atomic);
        future_into_py(py, async move {
            tokio::task::spawn_blocking(move || -> PyResult<()> {
                let mut bridge = inner.blocking_lock();
                bridge.stop()?;
                // Clear atomic after
                // inner.stop() succeeds (infallible today; mirroring
                // pattern keeps symmetry with start()).
                is_playing_atomic.store(false, Ordering::Release);
                Ok(())
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }

    fn close<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        // close is documented in the sync bridge as an alias for stop.
        self.stop(py)
    }

    #[getter]
    fn is_playing<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let inner = Arc::clone(&self.inner);
        future_into_py(py, async move {
            let bridge = inner.lock().await;
            Ok(bridge.is_playing())
        })
    }

    #[getter]
    fn is_playing_sync(&self) -> bool {
        // Lock-free sync getter for the
        // wrapper's synchronous AsyncSpeaker.is_playing property. Same
        // pattern as AsyncMicrophoneBridge.is_open_sync.
        self.is_playing_atomic.load(Ordering::Acquire)
    }

    #[staticmethod]
    fn devices(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
        future_into_py(py, async move {
            tokio::task::spawn_blocking(|| -> PyResult<Vec<OutputDeviceInfo>> {
                Python::attach(SpeakerBridge::devices)
            })
            .await
            .map_err(|e| PyRuntimeError::new_err(format!("spawn_blocking failed: {e}")))?
        })
    }
}

// ---------------------------------------------------------------------------
// FileBridge: offline source pyclass.
//
// Wraps the core offline File: constructed from a WAV path or in-memory
// samples, iterated for conditioned chunks (read), or consumed by a
// whole-recording analysis (analyze). Mirrors MicrophoneBridge's split of
// responsibilities: the bridge computes the per-chunk VAD score natively on
// the pre-conditioning feed and exposes it via `vad_probability`; the Python
// wrapper applies the threshold + holdoff policy, in FILE time.
//
// The per-chunk detector is constructed lazily on the first read, so an
// analysis-only File does not load the Silero model twice (analysis drives
// the core's own detector).
// ---------------------------------------------------------------------------

#[pyclass]
struct FileBridge {
    /// The core offline source. `None` once consumed by analysis, closed, or
    /// fully iterated: a File is a single pass.
    inner: StdMutex<Option<CoreFile>>,
    /// Set only when `analyze()` takes the source. Separates the analyzed
    /// state, where a later read fails loud, from natural exhaustion and
    /// `close()`, where a later read ends quietly.
    consumed: AtomicBool,
    format: BindingSampleFormat,
    numpy: bool,
    vad_enabled: bool,
    /// Energy mode: the per-chunk score is the RMS of the pre-conditioning
    /// feed, no model. Mutually exclusive with the Silero detector below.
    energy_vad: bool,
    /// Lazily constructed per-chunk Silero detector (silero mode only).
    vad: StdMutex<Option<SileroVad>>,
    vad_model_path: Option<PathBuf>,
    vad_threshold: f32,
    ort_library_path: Option<PathBuf>,
    /// Most recent per-chunk score as f32 bits, exposed via the
    /// `vad_probability` getter exactly as MicrophoneBridge exposes its own.
    last_vad_probability: AtomicU32,
    sample_rate: u32,
}

/// Build the core [`FileConfig`] from the bridge keyword arguments, mapping
/// each selector exactly as `MicrophoneBridge::new` maps the same names.
#[allow(clippy::too_many_arguments)]
fn build_file_config(
    sample_rate: u32,
    vad: bool,
    vad_threshold: f32,
    vad_mode: &str,
    vad_holdoff: u32,
    model_path: Option<&PathBuf>,
    ort_library_path: Option<&PathBuf>,
    denoise: Option<&str>,
    denoise_model_path: Option<PathBuf>,
    highpass: Option<i64>,
    agc: Option<i8>,
    limiter: Option<f32>,
    dc_removal: bool,
) -> PyResult<FileConfig> {
    // `FileConfig` is `#[non_exhaustive]`: default-construct then assign the
    // public fields rather than using a struct literal.
    let mut config = FileConfig::default();
    config.sample_rate = sample_rate;
    config.dc_removal = dc_removal;

    if let Some(name) = denoise {
        let model = match name {
            "fastenhancer-t" => DenoiseModel::FastEnhancerT,
            other => {
                return Err(PyValueError::new_err(format!(
                    "denoise must be 'fastenhancer-t', got '{other}'"
                )))
            }
        };
        let mp = denoise_model_path.ok_or_else(|| {
            PyValueError::new_err("denoise_model_path is required when denoise is set")
        })?;
        config.denoise = Some(model);
        config.denoise_model_path = Some(mp);
        config.ort_library_path = ort_library_path.cloned();
    }

    if let Some(hz) = highpass {
        let filter = match hz {
            80 => HighpassFilter::Hz80,
            100 => HighpassFilter::Hz100,
            other => {
                return Err(PyValueError::new_err(format!(
                    "highpass must be one of: 80, 100; got {other}"
                )))
            }
        };
        config.highpass = Some(filter);
    }

    config.agc = agc;
    config.limiter = limiter;

    // Whole-recording analysis runs the core's own detector; wire the
    // detector configuration only for the silero mode (energy mode has no
    // whole-recording analysis and its per-chunk score needs no model).
    if vad && vad_mode == "silero" {
        let mp = model_path.ok_or_else(|| {
            PyValueError::new_err("model_path is required when vad=True and vad_mode='silero'")
        })?;
        // `VadConfig` is `#[non_exhaustive]`: default-construct then assign.
        let mut vad_config = VadConfig::default();
        vad_config.model_path = mp.clone();
        vad_config.threshold = vad_threshold;
        vad_config.ort_library_path = ort_library_path.cloned();
        config.vad = Some(vad_config);
        config.vad_holdoff_ms = vad_holdoff;
    }

    Ok(config)
}

/// Extract the in-memory samples argument for `FileBridge::buffer`: a list of
/// floats extracts directly; the wrapper transports numpy arrays as raw f32
/// little-endian bytes to avoid a per-sample boundary cost.
fn extract_buffer_samples(samples: &Bound<'_, PyAny>) -> PyResult<Vec<f32>> {
    if let Ok(bytes) = samples.cast::<PyBytes>() {
        return Ok(f32_le_bytes_to_f32(bytes.as_bytes()));
    }
    if let Ok(vec) = samples.extract::<Vec<f32>>() {
        return Ok(vec);
    }
    Err(PyTypeError::new_err(
        "samples must be a list of floats or f32 little-endian bytes",
    ))
}

#[pymethods]
impl FileBridge {
    /// Open a WAV path as an offline source.
    #[staticmethod]
    #[pyo3(signature = (
        path,
        sample_rate = 16000,
        format = "int16".to_string(),
        vad = false,
        vad_threshold = 0.5_f32,
        vad_mode = "silero".to_string(),
        vad_holdoff = 300_u32,
        model_path = None,
        numpy = false,
        ort_library_path = None,
        denoise = None,
        denoise_model_path = None,
        highpass = None,
        agc = None,
        limiter = None,
        dc_removal = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn open(
        py: Python<'_>,
        path: PathBuf,
        sample_rate: u32,
        format: String,
        vad: bool,
        vad_threshold: f32,
        vad_mode: String,
        vad_holdoff: u32,
        model_path: Option<PathBuf>,
        numpy: bool,
        ort_library_path: Option<PathBuf>,
        denoise: Option<String>,
        denoise_model_path: Option<PathBuf>,
        highpass: Option<i64>,
        agc: Option<i8>,
        limiter: Option<f32>,
        dc_removal: bool,
    ) -> PyResult<Self> {
        let parsed_format = parse_sample_format(&format).map_err(|e| to_py_err(py, e))?;
        let config = build_file_config(
            sample_rate,
            vad,
            vad_threshold,
            &vad_mode,
            vad_holdoff,
            model_path.as_ref(),
            ort_library_path.as_ref(),
            denoise.as_deref(),
            denoise_model_path,
            highpass,
            agc,
            limiter,
            dc_removal,
        )?;
        let file = CoreFile::open(&path, config).map_err(|e| to_py_err(py, e))?;
        Ok(Self::from_core(
            file,
            parsed_format,
            numpy,
            vad,
            &vad_mode,
            vad_threshold,
            model_path,
            ort_library_path,
            sample_rate,
        ))
    }

    /// Wrap in-memory samples as an offline source. `samples` is a list of
    /// floats or raw f32 little-endian bytes; `input_rate` is their native
    /// rate (raw samples carry no header).
    #[staticmethod]
    #[pyo3(signature = (
        samples,
        input_rate,
        sample_rate = 16000,
        format = "int16".to_string(),
        vad = false,
        vad_threshold = 0.5_f32,
        vad_mode = "silero".to_string(),
        vad_holdoff = 300_u32,
        model_path = None,
        numpy = false,
        ort_library_path = None,
        denoise = None,
        denoise_model_path = None,
        highpass = None,
        agc = None,
        limiter = None,
        dc_removal = false,
    ))]
    #[allow(clippy::too_many_arguments)]
    fn buffer(
        py: Python<'_>,
        samples: Bound<'_, PyAny>,
        input_rate: u32,
        sample_rate: u32,
        format: String,
        vad: bool,
        vad_threshold: f32,
        vad_mode: String,
        vad_holdoff: u32,
        model_path: Option<PathBuf>,
        numpy: bool,
        ort_library_path: Option<PathBuf>,
        denoise: Option<String>,
        denoise_model_path: Option<PathBuf>,
        highpass: Option<i64>,
        agc: Option<i8>,
        limiter: Option<f32>,
        dc_removal: bool,
    ) -> PyResult<Self> {
        let parsed_format = parse_sample_format(&format).map_err(|e| to_py_err(py, e))?;
        let data = extract_buffer_samples(&samples)?;
        let config = build_file_config(
            sample_rate,
            vad,
            vad_threshold,
            &vad_mode,
            vad_holdoff,
            model_path.as_ref(),
            ort_library_path.as_ref(),
            denoise.as_deref(),
            denoise_model_path,
            highpass,
            agc,
            limiter,
            dc_removal,
        )?;
        let file = CoreFile::buffer(data, input_rate, config).map_err(|e| to_py_err(py, e))?;
        Ok(Self::from_core(
            file,
            parsed_format,
            numpy,
            vad,
            &vad_mode,
            vad_threshold,
            model_path,
            ort_library_path,
            sample_rate,
        ))
    }

    /// Read the next conditioned chunk, advancing the per-chunk VAD score on
    /// the pre-conditioning feed. Returns `None` once the source is fully
    /// delivered (after the end-of-stream tail) or already consumed.
    fn read<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        // A File analyzed then iterated fails loud rather than yielding an
        // empty stream indistinguishable from a silent recording. Exhaustion
        // and close() leave `consumed` false and still read as ended.
        if self.consumed.load(Ordering::Relaxed) {
            return Err(raise_named(
                py,
                "FileConsumed",
                "File already consumed; construct a new File for another pass",
            ));
        }
        let (data, channels) = {
            let mut guard = lock_recover(&self.inner);
            let Some(file) = guard.as_mut() else {
                return Ok(None);
            };
            let next = py.detach(|| file.next());
            let chunk = match next {
                None => {
                    // Fully delivered: drop the source so held memory is
                    // released as soon as the pass ends.
                    *guard = None;
                    return Ok(None);
                }
                Some(Err(e)) => return Err(to_py_err(py, e)),
                Some(Ok(chunk)) => chunk,
            };

            if self.vad_enabled {
                // The core hands out the pre-conditioning feed when it is
                // maintained (VAD configured, or a transform makes the
                // delivered output differ); on `None` the delivered chunk
                // already is that signal, the same fallback the live pump
                // applies.
                let feed = file.vad_input();
                if self.energy_vad {
                    // Energy mode: the score is the RMS of the
                    // pre-conditioning signal, on the same [0, 1] scale the
                    // energy threshold compares against.
                    let frame: &[f32] = match feed.as_deref() {
                        Some(f) => f,
                        None => &chunk.data,
                    };
                    if !frame.is_empty() {
                        let score = rms(frame);
                        self.last_vad_probability
                            .store(score.to_bits(), Ordering::Relaxed);
                    }
                } else if let Some(feed) = feed {
                    // Silero mode always configures the core detector feed.
                    if !feed.is_empty() {
                        let mut vad_guard = lock_recover(&self.vad);
                        if vad_guard.is_none() {
                            *vad_guard = Some(self.build_detector(py, file.vad_rate())?);
                        }
                        if let Some(vad) = vad_guard.as_mut() {
                            let result = py
                                .detach(|| vad.process(&feed))
                                .map_err(|e| to_py_err(py, e))?;
                            self.last_vad_probability
                                .store(result.probability.to_bits(), Ordering::Relaxed);
                        }
                    }
                }
            }
            (chunk.data, chunk.channels)
        };

        let obj = if self.numpy {
            encode_chunk_numpy(py, data, self.format, channels)?
        } else {
            encode_chunk_bytes(py, &data, self.format)
        };
        Ok(Some(obj))
    }

    /// Consume the source with the core's whole-recording analysis. Returns
    /// the per-window scores as `(start, end, probability, is_speech)` tuples
    /// and the merged segments as `(start, end)` tuples; the wrapper shapes
    /// them into the public report types.
    #[allow(clippy::type_complexity)]
    fn analyze(&self, py: Python<'_>) -> PyResult<(Vec<(f64, f64, f32, bool)>, Vec<(f64, f64)>)> {
        let file = lock_recover(&self.inner).take();
        let Some(file) = file else {
            return Err(raise_named(
                py,
                "FileConsumed",
                "File already consumed; construct a new File for another pass",
            ));
        };
        // The source is taken: a later iteration reads as consumed, not as an
        // empty recording, whatever the analysis below returns.
        self.consumed.store(true, Ordering::Relaxed);
        let report = py.detach(|| file.analyze()).map_err(|e| to_py_err(py, e))?;
        Ok((
            report
                .scores
                .iter()
                .map(|w| (w.start, w.end, w.probability, w.is_speech))
                .collect(),
            report.segments.iter().map(|s| (s.start, s.end)).collect(),
        ))
    }

    /// Release the source. Idempotent; a closed File reads as ended.
    fn close(&self) {
        *lock_recover(&self.inner) = None;
    }

    /// Most recent per-chunk VAD score (0.0 to 1.0), computed on the
    /// pre-conditioning feed. 0.0 before the first chunk or with VAD off.
    #[getter]
    fn vad_probability(&self) -> f32 {
        f32::from_bits(self.last_vad_probability.load(Ordering::Relaxed))
    }

    /// The target output rate every delivered chunk carries.
    #[getter]
    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

impl FileBridge {
    /// Shared constructor tail for the two source forms.
    #[allow(clippy::too_many_arguments)]
    fn from_core(
        file: CoreFile,
        format: BindingSampleFormat,
        numpy: bool,
        vad: bool,
        vad_mode: &str,
        vad_threshold: f32,
        model_path: Option<PathBuf>,
        ort_library_path: Option<PathBuf>,
        sample_rate: u32,
    ) -> Self {
        Self {
            inner: StdMutex::new(Some(file)),
            consumed: AtomicBool::new(false),
            format,
            numpy,
            vad_enabled: vad,
            energy_vad: vad && vad_mode == "energy",
            vad: StdMutex::new(None),
            vad_model_path: model_path,
            vad_threshold,
            ort_library_path,
            last_vad_probability: AtomicU32::new(0),
            sample_rate,
        }
    }

    /// Construct the lazily-built per-chunk Silero detector at the rate the
    /// core File runs detection at (which already accounts for the internal
    /// feed resample).
    fn build_detector(&self, py: Python<'_>, vad_rate: Option<u32>) -> PyResult<SileroVad> {
        let mp = self.vad_model_path.clone().ok_or_else(|| {
            PyValueError::new_err("model_path is required when vad=True and vad_mode='silero'")
        })?;
        // `VadConfig` is `#[non_exhaustive]`: default-construct then assign.
        let mut vad_config = VadConfig::default();
        vad_config.model_path = mp;
        vad_config.sample_rate = vad_rate.unwrap_or(16000);
        vad_config.threshold = self.vad_threshold;
        vad_config.ort_library_path = self.ort_library_path.clone();
        SileroVad::new(vad_config).map_err(|e| to_py_err(py, e))
    }
}

// ---------------------------------------------------------------------------
// Module entry point.
//
// Exception classes are not registered as module attributes here. They live
// in pure Python at decibri.exceptions; to_py_err looks them up via
// exception_class() at first use and caches the lookup.
// ---------------------------------------------------------------------------

#[pymodule]
fn _decibri(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VersionInfo>()?;
    m.add_class::<DeviceInfo>()?;
    m.add_class::<OutputDeviceInfo>()?;
    m.add_class::<MicrophoneBridge>()?;
    m.add_class::<SpeakerBridge>()?;
    m.add_class::<FileBridge>()?;

    // Async pyclasses. The Python wrappers (AsyncMicrophone,
    // AsyncSpeaker) live in python/decibri/_async_classes.py and
    // proxy to these via Arc<tokio::sync::Mutex<sync_bridge>>.
    m.add_class::<AsyncMicrophoneBridge>()?;
    m.add_class::<AsyncSpeakerBridge>()?;

    // Empirical smoke; see comment block above the
    // async_smoke fn declaration for rationale.
    m.add_function(wrap_pyfunction!(async_smoke, m)?)?;

    // Empirical smoke; see comment block above the
    // numpy_smoke fn declaration for rationale.
    m.add_function(wrap_pyfunction!(numpy_smoke, m)?)?;

    Ok(())
}
