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
//! VAD architecture: the `vad_mode` and `vad_holdoff` parameters accepted by
//! `DecibriBridge.__init__` are stored as inert state on the bridge or used
//! only as construction gates; the wrapper layer (`_classes.py`) implements
//! mode/holdoff/threshold policy on top of the raw Silero probability that
//! the bridge exposes via `vad_probability`.

use std::path::PathBuf;
use std::time::Duration;

use std::collections::HashMap;

use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyBytes, PyModule, PyType};

use decibri::capture::{AudioCapture, AudioChunk, CaptureConfig, CaptureStream};
use decibri::device::{
    enumerate_input_devices, enumerate_output_devices, DeviceInfo as CoreDeviceInfo,
    DeviceSelector, OutputDeviceInfo as CoreOutputDeviceInfo,
};
use decibri::error::DecibriError as CoreDecibriError;
use decibri::output::{AudioOutput, OutputConfig, OutputStream};
use decibri::sample::{
    f32_le_bytes_to_f32, f32_to_f32_le_bytes, f32_to_i16_le_bytes, i16_le_bytes_to_f32,
};
use decibri::vad::{SileroVad, VadConfig};
use decibri::CPAL_VERSION;

// ---------------------------------------------------------------------------
// Exception class lookup.
//
// The 31-class hierarchy is defined once in pure Python at
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
    "CaptureStreamClosed",
    "ChannelsOutOfRange",
    "DeviceEnumerationFailed",
    "DeviceIndexOutOfRange",
    "DeviceNotFound",
    "FramesPerBufferOutOfRange",
    "InvalidFormat",
    "MultipleDevicesMatch",
    "NoMicrophoneFound",
    "NoOutputDeviceFound",
    "NotAnInputDevice",
    "OutputDeviceNotFound",
    "OutputStreamClosed",
    "PermissionDenied",
    "SampleRateOutOfRange",
    "StreamOpenFailed",
    "StreamStartFailed",
    "VadSampleRateUnsupported",
    "VadThresholdOutOfRange",
    "VadModelLoadFailed",
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
// Covers all 29 variants explicitly. Catch-all `_ =>` arm at end is required
// because DecibriError is #[non_exhaustive] (per F10).
//
// For variants with a `path` field (OrtLoadFailed, OrtPathInvalid,
// VadModelLoadFailed), the mapper passes a tuple as the args so the Python
// __init__ override can store named attributes per F11.
// ---------------------------------------------------------------------------

type ExceptionRaiser = Box<dyn FnOnce(Bound<'_, PyType>) -> PyErr>;

fn to_py_err(py: Python<'_>, err: CoreDecibriError) -> PyErr {
    let msg = err.to_string();

    // Determine which exception class to raise plus the args tuple.
    // The pure-Python __init__ overrides on OrtLoadFailed, OrtPathInvalid,
    // and VadModelLoadFailed unpack multi-element tuples into named
    // attributes; other classes accept a single message string.
    let (name, raise): (&'static str, ExceptionRaiser) = match err {
        // Unit variants (12).
        CoreDecibriError::SampleRateOutOfRange => (
            "SampleRateOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::ChannelsOutOfRange => (
            "ChannelsOutOfRange",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::FramesPerBufferOutOfRange => (
            "FramesPerBufferOutOfRange",
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
        CoreDecibriError::NoOutputDeviceFound => (
            "NoOutputDeviceFound",
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
        CoreDecibriError::CaptureStreamClosed => (
            "CaptureStreamClosed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::OutputStreamClosed => (
            "OutputStreamClosed",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),

        // Tuple variants with String (5).
        CoreDecibriError::DeviceNotFound(_) => (
            "DeviceNotFound",
            Box::new(move |cls| PyErr::from_type(cls, (msg,))),
        ),
        CoreDecibriError::OutputDeviceNotFound(_) => (
            "OutputDeviceNotFound",
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

        // Struct variants with path attribute (3). Multi-element args
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

        // #[non_exhaustive] catch-all per F10. Fall through to the base class.
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
// (unit variant per F9) for any other string. The to_py_err mapper translates
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
// version via three string fields. Constructed by DecibriBridge::version().
// ---------------------------------------------------------------------------

#[pyclass(module = "decibri._decibri", frozen)]
struct VersionInfo {
    #[pyo3(get)]
    decibri: String,
    #[pyo3(get)]
    portaudio: String,
    #[pyo3(get)]
    binding: String,
}

#[pymethods]
impl VersionInfo {
    fn __repr__(&self) -> String {
        format!(
            "VersionInfo(decibri='{}', portaudio='{}', binding='{}')",
            self.decibri, self.portaudio, self.binding
        )
    }
}

fn build_version_info() -> VersionInfo {
    VersionInfo {
        decibri: env!("CARGO_PKG_VERSION").to_string(),
        portaudio: format!("cpal {}", CPAL_VERSION),
        binding: "0.1.0a1".to_string(),
    }
}

// ---------------------------------------------------------------------------
// DeviceInfo / OutputDeviceInfo: read-only Python wrappers around the core
// device structs. Both are #[non_exhaustive] in the Rust core, so the
// binding's pyclasses are the stable Python-facing surface.
// ---------------------------------------------------------------------------

#[pyclass(module = "decibri._decibri", frozen)]
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
            "DeviceInfo(index={}, name='{}', id='{}', max_input_channels={}, default_sample_rate={}, is_default={})",
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

#[pyclass(module = "decibri._decibri", frozen)]
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
            "OutputDeviceInfo(index={}, name='{}', id='{}', max_output_channels={}, default_sample_rate={}, is_default={})",
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
// DecibriBridge: stateful capture pyclass.
//
// Owns:
//   - capture_config: CaptureConfig (built from constructor args, retained
//     for diagnostic introspection)
//   - format: BindingSampleFormat (parsed from constructor 'format' arg)
//   - vad_holdoff_ms: u32 (inert; wrapper layer reads this if it wants to)
//   - capture: Option<AudioCapture> (None until start())
//   - stream: Option<CaptureStream> (None until start())
//   - vad: Option<SileroVad> (None unless constructor's vad=True AND
//     vad_mode=="silero"; per VAD Option (b)+(h) decision)
//   - last_vad_probability: f32 (most recent Silero raw probability; updated
//     on each read() that runs Silero inference)
// ---------------------------------------------------------------------------

#[pyclass(module = "decibri._decibri", unsendable)]
struct DecibriBridge {
    capture_config: CaptureConfig,
    format: BindingSampleFormat,
    vad_holdoff_ms: u32,
    capture: Option<AudioCapture>,
    stream: Option<CaptureStream>,
    vad: Option<SileroVad>,
    last_vad_probability: f32,
}

#[pymethods]
impl DecibriBridge {
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
    ) -> PyResult<Self> {
        if numpy {
            return Err(PyNotImplementedError::new_err(
                "numpy=True is not supported in 0.1.0a1; use bytes and decode in Python",
            ));
        }

        let parsed_format = parse_sample_format(&format).map_err(|e| to_py_err(py, e))?;
        let device_selector = build_device_selector(device.as_ref())?;

        let capture_config = CaptureConfig {
            sample_rate,
            channels,
            frames_per_buffer,
            device: device_selector,
        };

        // VAD construction gate (Option (b)+(h)). The bridge constructs
        // SileroVad only if vad=True AND vad_mode=="silero". The vad_mode
        // string is otherwise inert at this layer; the wrapper validates and
        // dispatches mode policy.
        let vad_instance = if vad && vad_mode == "silero" {
            let mp = model_path.ok_or_else(|| {
                PyValueError::new_err("model_path is required when vad=True and vad_mode='silero'")
            })?;
            let vad_config = VadConfig {
                model_path: mp,
                sample_rate,
                threshold: vad_threshold,
                ort_library_path,
            };
            Some(SileroVad::new(vad_config).map_err(|e| to_py_err(py, e))?)
        } else {
            None
        };

        Ok(DecibriBridge {
            capture_config,
            format: parsed_format,
            vad_holdoff_ms: vad_holdoff,
            capture: None,
            stream: None,
            vad: vad_instance,
            last_vad_probability: 0.0,
        })
    }

    fn start(&mut self, py: Python<'_>) -> PyResult<()> {
        if self.stream.is_some() {
            return Err(raise_named(
                py,
                "AlreadyRunning",
                "capture is already running",
            ));
        }
        let capture =
            AudioCapture::new(self.capture_config.clone()).map_err(|e| to_py_err(py, e))?;
        let stream = capture.start().map_err(|e| to_py_err(py, e))?;
        self.capture = Some(capture);
        self.stream = Some(stream);
        Ok(())
    }

    fn stop(&mut self) -> PyResult<()> {
        // Drop the stream first, then the capture handle. Both are infallible
        // on drop; the explicit Option swap clarifies intent.
        self.stream = None;
        self.capture = None;
        Ok(())
    }

    #[pyo3(signature = (timeout_ms = None))]
    fn read<'py>(
        &mut self,
        py: Python<'py>,
        timeout_ms: Option<u64>,
    ) -> PyResult<Option<Bound<'py, PyBytes>>> {
        let timeout = timeout_ms.map(Duration::from_millis);
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| raise_named(py, "CaptureStreamClosed", "capture is not running"))?;

        // Release GIL for the blocking next_chunk call. Run VAD inference
        // inside the same allow_threads region to avoid GIL ping-pong.
        let result: Result<Option<AudioChunk>, CoreDecibriError> = py.detach(|| {
            let chunk = stream.next_chunk(timeout)?;
            Ok(chunk)
        });

        let Some(chunk) = result.map_err(|e| to_py_err(py, e))? else {
            return Ok(None);
        };

        // Run Silero VAD inference if configured. This is also blocking-ish
        // (ONNX inference); release GIL.
        if let Some(vad) = self.vad.as_mut() {
            let result = py
                .detach(|| vad.process(&chunk.data))
                .map_err(|e| to_py_err(py, e))?;
            self.last_vad_probability = result.probability;
        }

        // Encode to bytes per format selected at construction time.
        let bytes = match self.format {
            BindingSampleFormat::Int16 => f32_to_i16_le_bytes(&chunk.data),
            BindingSampleFormat::Float32 => f32_to_f32_le_bytes(&chunk.data),
        };

        Ok(Some(PyBytes::new(py, &bytes)))
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(
        mut slf: PyRefMut<'_, Self>,
        py: Python<'py>,
    ) -> PyResult<Bound<'py, PyBytes>> {
        // Block indefinitely on next_chunk by passing None timeout. If the
        // stream returns None (closed), raise StopIteration via PyO3's
        // None-as-StopIteration convention by returning the appropriate err.
        let bytes = slf.read(py, None)?;
        match bytes {
            Some(b) => Ok(b),
            None => Err(pyo3::exceptions::PyStopIteration::new_err(())),
        }
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
    fn is_open(&self) -> bool {
        self.stream.is_some()
    }

    #[getter]
    fn vad_probability(&self) -> f32 {
        self.last_vad_probability
    }

    /// Returns inert vad_holdoff_ms set at construction; wrapper reads this
    /// to apply policy. Pure storage; bridge does no holdoff logic.
    #[getter]
    fn vad_holdoff_ms(&self) -> u32 {
        self.vad_holdoff_ms
    }

    #[staticmethod]
    fn devices(py: Python<'_>) -> PyResult<Vec<DeviceInfo>> {
        let core_devices = py
            .detach(enumerate_input_devices)
            .map_err(|e| to_py_err(py, e))?;
        Ok(core_devices.into_iter().map(DeviceInfo::from).collect())
    }

    #[staticmethod]
    fn version() -> VersionInfo {
        build_version_info()
    }
}

// ---------------------------------------------------------------------------
// DecibriOutputBridge: stateful output pyclass.
//
// Owns:
//   - output_config: OutputConfig (no frames_per_buffer per F2)
//   - format: BindingSampleFormat
//   - output: Option<AudioOutput>
//   - stream: Option<OutputStream>
// ---------------------------------------------------------------------------

#[pyclass(module = "decibri._decibri", unsendable)]
struct DecibriOutputBridge {
    output_config: OutputConfig,
    format: BindingSampleFormat,
    output: Option<AudioOutput>,
    stream: Option<OutputStream>,
}

#[pymethods]
impl DecibriOutputBridge {
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

        let output_config = OutputConfig {
            sample_rate,
            channels,
            device: device_selector,
        };

        Ok(DecibriOutputBridge {
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
        let output = AudioOutput::new(self.output_config.clone()).map_err(|e| to_py_err(py, e))?;
        let stream = output.start().map_err(|e| to_py_err(py, e))?;
        self.output = Some(output);
        self.stream = Some(stream);
        Ok(())
    }

    fn write(&mut self, py: Python<'_>, samples: &[u8]) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| raise_named(py, "OutputStreamClosed", "output is not running"))?;

        // Materialize a Vec<u8> from the Python-borrowed slice before entering
        // allow_threads. PyO3's contract requires no Python-borrowed data
        // inside the GIL-released closure. One allocation per write call;
        // correctness over micro-optimization.
        let owned_samples: Vec<u8> = samples.to_vec();

        let samples_f32 = match self.format {
            BindingSampleFormat::Int16 => py.detach(|| i16_le_bytes_to_f32(&owned_samples)),
            BindingSampleFormat::Float32 => py.detach(|| f32_le_bytes_to_f32(&owned_samples)),
        };

        py.detach(|| stream.send(samples_f32))
            .map_err(|e| to_py_err(py, e))?;
        Ok(())
    }

    fn drain(&mut self, py: Python<'_>) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| raise_named(py, "OutputStreamClosed", "output is not running"))?;
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
        let core_devices = py
            .detach(enumerate_output_devices)
            .map_err(|e| to_py_err(py, e))?;
        Ok(core_devices
            .into_iter()
            .map(OutputDeviceInfo::from)
            .collect())
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
    m.add_class::<DecibriBridge>()?;
    m.add_class::<DecibriOutputBridge>()?;

    Ok(())
}
