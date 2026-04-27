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

use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyNotImplementedError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

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
// Exception class declarations (31 classes total).
//
// Hierarchy (mirrors python/decibri/exceptions.py from Commit 1):
//   DecibriError (base, subclass of Exception)
//     |- 20 direct subclasses (config, device, stream, format, lifecycle)
//     |- OrtError                            [catch-target]
//          |- OrtInitFailed
//          |- OrtSessionBuildFailed
//          |- OrtThreadsConfigFailed
//          |- VadModelLoadFailed             [carries path attribute]
//          |- OrtTensorCreateFailed
//          |- OrtTensorExtractFailed
//          |- OrtInferenceFailed
//          |- OrtPathError                   [catch-target, no instances]
//                |- OrtLoadFailed            [carries path attribute]
//                |- OrtPathInvalid           [carries path + reason]
// ---------------------------------------------------------------------------

create_exception!(_decibri, DecibriError, PyException);

// 20 direct subclasses of DecibriError (alphabetical within logical group).
create_exception!(_decibri, AlreadyRunning, DecibriError);
create_exception!(_decibri, CaptureStreamClosed, DecibriError);
create_exception!(_decibri, ChannelsOutOfRange, DecibriError);
create_exception!(_decibri, DeviceEnumerationFailed, DecibriError);
create_exception!(_decibri, DeviceIndexOutOfRange, DecibriError);
create_exception!(_decibri, DeviceNotFound, DecibriError);
create_exception!(_decibri, FramesPerBufferOutOfRange, DecibriError);
create_exception!(_decibri, InvalidFormat, DecibriError);
create_exception!(_decibri, MultipleDevicesMatch, DecibriError);
create_exception!(_decibri, NoMicrophoneFound, DecibriError);
create_exception!(_decibri, NoOutputDeviceFound, DecibriError);
create_exception!(_decibri, NotAnInputDevice, DecibriError);
create_exception!(_decibri, OutputDeviceNotFound, DecibriError);
create_exception!(_decibri, OutputStreamClosed, DecibriError);
create_exception!(_decibri, PermissionDenied, DecibriError);
create_exception!(_decibri, SampleRateOutOfRange, DecibriError);
create_exception!(_decibri, StreamOpenFailed, DecibriError);
create_exception!(_decibri, StreamStartFailed, DecibriError);
create_exception!(_decibri, VadSampleRateUnsupported, DecibriError);
create_exception!(_decibri, VadThresholdOutOfRange, DecibriError);

// ORT family. VadModelLoadFailed is an OrtError subclass per Commit 1's
// Python hierarchy (since the failure path runs through ORT's session loader).
create_exception!(_decibri, OrtError, DecibriError);
create_exception!(_decibri, OrtInitFailed, OrtError);
create_exception!(_decibri, OrtSessionBuildFailed, OrtError);
create_exception!(_decibri, OrtThreadsConfigFailed, OrtError);
create_exception!(_decibri, VadModelLoadFailed, OrtError);
create_exception!(_decibri, OrtTensorCreateFailed, OrtError);
create_exception!(_decibri, OrtTensorExtractFailed, OrtError);
create_exception!(_decibri, OrtInferenceFailed, OrtError);

// OrtPathError catch-target plus its two path-bearing leaves.
create_exception!(_decibri, OrtPathError, OrtError);
create_exception!(_decibri, OrtLoadFailed, OrtPathError);
create_exception!(_decibri, OrtPathInvalid, OrtPathError);

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

fn to_py_err(err: CoreDecibriError) -> PyErr {
    let msg = err.to_string();

    match err {
        // Unit variants (12).
        CoreDecibriError::SampleRateOutOfRange => SampleRateOutOfRange::new_err(msg),
        CoreDecibriError::ChannelsOutOfRange => ChannelsOutOfRange::new_err(msg),
        CoreDecibriError::FramesPerBufferOutOfRange => FramesPerBufferOutOfRange::new_err(msg),
        CoreDecibriError::InvalidFormat => InvalidFormat::new_err(msg),
        CoreDecibriError::DeviceIndexOutOfRange => DeviceIndexOutOfRange::new_err(msg),
        CoreDecibriError::NoMicrophoneFound => NoMicrophoneFound::new_err(msg),
        CoreDecibriError::NoOutputDeviceFound => NoOutputDeviceFound::new_err(msg),
        CoreDecibriError::NotAnInputDevice => NotAnInputDevice::new_err(msg),
        CoreDecibriError::AlreadyRunning => AlreadyRunning::new_err(msg),
        CoreDecibriError::PermissionDenied => PermissionDenied::new_err(msg),
        CoreDecibriError::CaptureStreamClosed => CaptureStreamClosed::new_err(msg),
        CoreDecibriError::OutputStreamClosed => OutputStreamClosed::new_err(msg),

        // Tuple variants with String (5). Destructure but use generated msg
        // (which already includes the inner string per Display impl).
        CoreDecibriError::DeviceNotFound(_) => DeviceNotFound::new_err(msg),
        CoreDecibriError::OutputDeviceNotFound(_) => OutputDeviceNotFound::new_err(msg),
        CoreDecibriError::DeviceEnumerationFailed(_) => DeviceEnumerationFailed::new_err(msg),
        CoreDecibriError::StreamOpenFailed(_) => StreamOpenFailed::new_err(msg),
        CoreDecibriError::StreamStartFailed(_) => StreamStartFailed::new_err(msg),

        // Tuple variants with primitive (2).
        CoreDecibriError::VadSampleRateUnsupported(_) => VadSampleRateUnsupported::new_err(msg),
        CoreDecibriError::VadThresholdOutOfRange(_) => VadThresholdOutOfRange::new_err(msg),

        // Tuple variants with #[source] ort::Error (3).
        CoreDecibriError::OrtSessionBuildFailed(_) => OrtSessionBuildFailed::new_err(msg),
        CoreDecibriError::OrtThreadsConfigFailed(_) => OrtThreadsConfigFailed::new_err(msg),
        CoreDecibriError::OrtInferenceFailed(_) => OrtInferenceFailed::new_err(msg),

        // Struct variants (7).
        CoreDecibriError::MultipleDevicesMatch { .. } => MultipleDevicesMatch::new_err(msg),
        CoreDecibriError::OrtInitFailed { .. } => OrtInitFailed::new_err(msg),
        CoreDecibriError::OrtTensorCreateFailed { .. } => OrtTensorCreateFailed::new_err(msg),
        CoreDecibriError::OrtTensorExtractFailed { .. } => OrtTensorExtractFailed::new_err(msg),

        // Struct variants with path attribute (3 of the 7). Pass tuples so
        // the Python __init__ overrides can populate named attributes.
        CoreDecibriError::OrtLoadFailed { ref path, .. } => {
            OrtLoadFailed::new_err((msg.clone(), path_to_string(path)))
        }
        CoreDecibriError::OrtPathInvalid { ref path, reason } => {
            OrtPathInvalid::new_err((msg.clone(), path_to_string(path), reason.to_string()))
        }
        CoreDecibriError::VadModelLoadFailed { ref path, .. } => {
            VadModelLoadFailed::new_err((msg.clone(), path_to_string(path)))
        }

        // #[non_exhaustive] catch-all per F10.
        _ => DecibriError::new_err(msg),
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

        let parsed_format = parse_sample_format(&format).map_err(to_py_err)?;
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
            Some(SileroVad::new(vad_config).map_err(to_py_err)?)
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

    fn start(&mut self) -> PyResult<()> {
        if self.stream.is_some() {
            return Err(AlreadyRunning::new_err("capture is already running"));
        }
        let capture = AudioCapture::new(self.capture_config.clone()).map_err(to_py_err)?;
        let stream = capture.start().map_err(to_py_err)?;
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
            .ok_or_else(|| CaptureStreamClosed::new_err("capture is not running"))?;

        // Release GIL for the blocking next_chunk call. Run VAD inference
        // inside the same allow_threads region to avoid GIL ping-pong.
        let result: Result<Option<AudioChunk>, CoreDecibriError> = py.detach(|| {
            let chunk = stream.next_chunk(timeout)?;
            Ok(chunk)
        });

        let Some(chunk) = result.map_err(to_py_err)? else {
            return Ok(None);
        };

        // Run Silero VAD inference if configured. This is also blocking-ish
        // (ONNX inference); release GIL.
        if let Some(vad) = self.vad.as_mut() {
            let result = py.detach(|| vad.process(&chunk.data)).map_err(to_py_err)?;
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
        slf.start()?;
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
        let core_devices = py.detach(enumerate_input_devices).map_err(to_py_err)?;
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
        sample_rate: u32,
        channels: u16,
        format: String,
        device: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let parsed_format = parse_sample_format(&format).map_err(to_py_err)?;
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

    fn start(&mut self) -> PyResult<()> {
        if self.stream.is_some() {
            return Err(AlreadyRunning::new_err("output is already running"));
        }
        let output = AudioOutput::new(self.output_config.clone()).map_err(to_py_err)?;
        let stream = output.start().map_err(to_py_err)?;
        self.output = Some(output);
        self.stream = Some(stream);
        Ok(())
    }

    fn write(&mut self, py: Python<'_>, samples: &[u8]) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| OutputStreamClosed::new_err("output is not running"))?;

        // Materialize a Vec<u8> from the Python-borrowed slice before entering
        // allow_threads. PyO3's contract requires no Python-borrowed data
        // inside the GIL-released closure. One allocation per write call;
        // correctness over micro-optimization.
        let owned_samples: Vec<u8> = samples.to_vec();

        let samples_f32 = match self.format {
            BindingSampleFormat::Int16 => py.detach(|| i16_le_bytes_to_f32(&owned_samples)),
            BindingSampleFormat::Float32 => py.detach(|| f32_le_bytes_to_f32(&owned_samples)),
        };

        py.detach(|| stream.send(samples_f32)).map_err(to_py_err)?;
        Ok(())
    }

    fn drain(&mut self, py: Python<'_>) -> PyResult<()> {
        let stream = self
            .stream
            .as_mut()
            .ok_or_else(|| OutputStreamClosed::new_err("output is not running"))?;
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
        slf.start()?;
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
        let core_devices = py.detach(enumerate_output_devices).map_err(to_py_err)?;
        Ok(core_devices
            .into_iter()
            .map(OutputDeviceInfo::from)
            .collect())
    }
}

// ---------------------------------------------------------------------------
// Exception registration: bind all 31 classes as module attributes so they
// are importable as `decibri._decibri.<ClassName>`. The pure-Python
// `decibri.exceptions` module re-exports these under the public names.
// ---------------------------------------------------------------------------

fn register_exceptions(py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("DecibriError", py.get_type::<DecibriError>())?;

    m.add("AlreadyRunning", py.get_type::<AlreadyRunning>())?;
    m.add("CaptureStreamClosed", py.get_type::<CaptureStreamClosed>())?;
    m.add("ChannelsOutOfRange", py.get_type::<ChannelsOutOfRange>())?;
    m.add(
        "DeviceEnumerationFailed",
        py.get_type::<DeviceEnumerationFailed>(),
    )?;
    m.add(
        "DeviceIndexOutOfRange",
        py.get_type::<DeviceIndexOutOfRange>(),
    )?;
    m.add("DeviceNotFound", py.get_type::<DeviceNotFound>())?;
    m.add(
        "FramesPerBufferOutOfRange",
        py.get_type::<FramesPerBufferOutOfRange>(),
    )?;
    m.add("InvalidFormat", py.get_type::<InvalidFormat>())?;
    m.add(
        "MultipleDevicesMatch",
        py.get_type::<MultipleDevicesMatch>(),
    )?;
    m.add("NoMicrophoneFound", py.get_type::<NoMicrophoneFound>())?;
    m.add("NoOutputDeviceFound", py.get_type::<NoOutputDeviceFound>())?;
    m.add("NotAnInputDevice", py.get_type::<NotAnInputDevice>())?;
    m.add(
        "OutputDeviceNotFound",
        py.get_type::<OutputDeviceNotFound>(),
    )?;
    m.add("OutputStreamClosed", py.get_type::<OutputStreamClosed>())?;
    m.add("PermissionDenied", py.get_type::<PermissionDenied>())?;
    m.add(
        "SampleRateOutOfRange",
        py.get_type::<SampleRateOutOfRange>(),
    )?;
    m.add("StreamOpenFailed", py.get_type::<StreamOpenFailed>())?;
    m.add("StreamStartFailed", py.get_type::<StreamStartFailed>())?;
    m.add(
        "VadSampleRateUnsupported",
        py.get_type::<VadSampleRateUnsupported>(),
    )?;
    m.add(
        "VadThresholdOutOfRange",
        py.get_type::<VadThresholdOutOfRange>(),
    )?;

    m.add("VadModelLoadFailed", py.get_type::<VadModelLoadFailed>())?;

    m.add("OrtError", py.get_type::<OrtError>())?;
    m.add("OrtInitFailed", py.get_type::<OrtInitFailed>())?;
    m.add(
        "OrtSessionBuildFailed",
        py.get_type::<OrtSessionBuildFailed>(),
    )?;
    m.add(
        "OrtThreadsConfigFailed",
        py.get_type::<OrtThreadsConfigFailed>(),
    )?;
    m.add(
        "OrtTensorCreateFailed",
        py.get_type::<OrtTensorCreateFailed>(),
    )?;
    m.add(
        "OrtTensorExtractFailed",
        py.get_type::<OrtTensorExtractFailed>(),
    )?;
    m.add("OrtInferenceFailed", py.get_type::<OrtInferenceFailed>())?;

    m.add("OrtPathError", py.get_type::<OrtPathError>())?;
    m.add("OrtLoadFailed", py.get_type::<OrtLoadFailed>())?;
    m.add("OrtPathInvalid", py.get_type::<OrtPathInvalid>())?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Module entry point.
// ---------------------------------------------------------------------------

#[pymodule]
fn _decibri(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();

    m.add_class::<VersionInfo>()?;
    m.add_class::<DeviceInfo>()?;
    m.add_class::<OutputDeviceInfo>()?;
    m.add_class::<DecibriBridge>()?;
    m.add_class::<DecibriOutputBridge>()?;

    register_exceptions(py, m)?;

    Ok(())
}
