//! Internal audio-backend seam.
//!
//! [`AudioBackend`] is the single place the crate reaches the platform audio
//! library. Device enumeration, selector resolution, and stream opening all go
//! through this trait, and [`CpalBackend`] is its sole implementation: the only
//! code in the crate that names a platform-audio (cpal) type. Containing the
//! platform library behind one trait keeps a backend version change a
//! single-file, single-implementation swap, and keeps the rest of the crate in
//! decibri-owned types ([`DecibriError`], [`DeviceSelector`], [`MicrophoneInfo`],
//! [`SpeakerInfo`], and the small support types defined below).

use std::sync::Mutex;

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::device::{compute_is_default, ComputedRow, DeviceSelector, MicrophoneInfo, SpeakerInfo};
use crate::error::DecibriError;

// ── Decibri-owned support types ────────────────────────────────────────────

/// A resolved platform audio device: an opaque, decibri-owned handle around the
/// backend's native device so the platform type does not cross the seam.
pub(crate) struct BackendDevice {
    inner: cpal::Device,
}

/// A live platform audio stream, owned so that dropping it (on
/// [`stop`](Self::stop) or on drop) releases the OS device.
///
/// The native stream is held behind a `Mutex<Option<...>>` so
/// [`stop`](Self::stop) can take and drop it under a shared `&self` while the
/// type stays `Send + Sync` (the bindings require `Sync`: a `#[pyclass]` and
/// `py.detach` both need it). The lock is uncontended: only `stop()` and drop
/// ever touch it. The `Option` also lets the test seam build a handle with no
/// real stream.
pub(crate) struct BackendStream {
    inner: Mutex<Option<cpal::Stream>>,
}

impl BackendStream {
    /// Drop the held stream now, releasing the OS device, rather than only on
    /// drop. Idempotent and poison-tolerant: a poisoned lock still clears the
    /// slot rather than panicking.
    pub(crate) fn stop(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            *guard = None;
        }
    }

    /// Whether the stream is still held (not yet dropped by [`stop`](Self::stop)).
    #[cfg(test)]
    pub(crate) fn is_active(&self) -> bool {
        self.inner.lock().map(|g| g.is_some()).unwrap_or(false)
    }

    /// A handle holding no native stream, for unit tests that exercise the
    /// stream-reading paths without opening a real device.
    #[cfg(test)]
    pub(crate) fn empty() -> Self {
        Self {
            inner: Mutex::new(None),
        }
    }
}

/// Parameters for opening a stream: the decibri-owned mirror of the fields the
/// open path sets on the platform stream config.
pub(crate) struct StreamParams {
    pub channels: u16,
    pub sample_rate: u32,
    /// Fixed buffer size in frames, or `None` for the backend default. Capture
    /// pins a fixed size; playback uses the backend default.
    pub frames_per_buffer: Option<u32>,
}

/// Realtime capture callback: invoked with one interleaved `f32` input buffer.
#[cfg(feature = "capture")]
pub(crate) type InputDataCallback = Box<dyn FnMut(&[f32]) + Send + 'static>;

/// Realtime render callback: invoked to fill one interleaved `f32` output buffer.
#[cfg(feature = "playback")]
pub(crate) type OutputDataCallback = Box<dyn FnMut(&mut [f32]) + Send + 'static>;

/// Stream error callback: invoked with a typed [`DecibriError::DeviceFailed`]
/// when the device or driver fails while streaming.
pub(crate) type StreamErrorCallback = Box<dyn FnMut(DecibriError) + Send + 'static>;

// ── The seam ───────────────────────────────────────────────────────────────

/// The crate's audio-backend seam. The implementor owns every interaction with
/// the platform audio library; callers see only decibri-owned types.
pub(crate) trait AudioBackend: Send + Sync {
    /// List the available input devices.
    fn input_devices(&self) -> Result<Vec<MicrophoneInfo>, DecibriError>;

    /// List the available output devices.
    fn output_devices(&self) -> Result<Vec<SpeakerInfo>, DecibriError>;

    /// Resolve a selector to an input device handle.
    #[cfg(feature = "capture")]
    fn resolve_input_device(
        &self,
        selector: &DeviceSelector,
    ) -> Result<BackendDevice, DecibriError>;

    /// Report the device's native input sample rate: the rate of its default
    /// supported format, which the capture stream is opened at before the
    /// capture chain resamples to the requested target rate.
    #[cfg(feature = "capture")]
    fn native_input_rate(&self, device: &BackendDevice) -> Result<u32, DecibriError>;

    /// Resolve a selector to an output device handle.
    #[cfg(feature = "playback")]
    fn resolve_output_device(
        &self,
        selector: &DeviceSelector,
    ) -> Result<BackendDevice, DecibriError>;

    /// Open and start an input stream on `device`, driving `on_data` with each
    /// captured buffer and `on_error` with a typed device failure.
    #[cfg(feature = "capture")]
    fn open_input_stream(
        &self,
        device: &BackendDevice,
        params: &StreamParams,
        on_data: InputDataCallback,
        on_error: StreamErrorCallback,
    ) -> Result<BackendStream, DecibriError>;

    /// Open and start an output stream on `device`, driving `on_data` to fill
    /// each playback buffer and `on_error` with a typed device failure.
    #[cfg(feature = "playback")]
    fn open_output_stream(
        &self,
        device: &BackendDevice,
        params: &StreamParams,
        on_data: OutputDataCallback,
        on_error: StreamErrorCallback,
    ) -> Result<BackendStream, DecibriError>;
}

/// The cpal-backed [`AudioBackend`]: the crate's sole backend implementation and
/// the only place a cpal type is named.
pub(crate) struct CpalBackend;

impl AudioBackend for CpalBackend {
    fn input_devices(&self) -> Result<Vec<MicrophoneInfo>, DecibriError> {
        enumerate_devices::<Input>()
    }

    fn output_devices(&self) -> Result<Vec<SpeakerInfo>, DecibriError> {
        enumerate_devices::<Output>()
    }

    #[cfg(feature = "capture")]
    fn resolve_input_device(
        &self,
        selector: &DeviceSelector,
    ) -> Result<BackendDevice, DecibriError> {
        Ok(BackendDevice {
            inner: resolve_device_generic::<Input>(selector)?,
        })
    }

    #[cfg(feature = "capture")]
    fn native_input_rate(&self, device: &BackendDevice) -> Result<u32, DecibriError> {
        // The default input config is the device's native supported format, so
        // opening at this rate is always accepted; the capture chain resamples
        // it to the requested target. A query failure maps to the same
        // enumeration error the rest of the seam uses.
        device
            .inner
            .default_input_config()
            .map(|config| config.sample_rate())
            .map_err(|e| DecibriError::DeviceEnumerationFailed(e.to_string()))
    }

    #[cfg(feature = "playback")]
    fn resolve_output_device(
        &self,
        selector: &DeviceSelector,
    ) -> Result<BackendDevice, DecibriError> {
        Ok(BackendDevice {
            inner: resolve_device_generic::<Output>(selector)?,
        })
    }

    #[cfg(feature = "capture")]
    fn open_input_stream(
        &self,
        device: &BackendDevice,
        params: &StreamParams,
        mut on_data: InputDataCallback,
        mut on_error: StreamErrorCallback,
    ) -> Result<BackendStream, DecibriError> {
        let stream = device
            .inner
            .build_input_stream(
                &stream_config(params),
                move |data: &[f32], _: &cpal::InputCallbackInfo| {
                    on_data(data);
                },
                move |err| {
                    // Convert the platform error to the typed cause at the
                    // boundary, exactly as before; the decibri callback records
                    // it and marks the stream closed.
                    on_error(DecibriError::DeviceFailed {
                        source: Box::new(err),
                    });
                },
                None,
            )
            .map_err(|e| DecibriError::StreamOpenFailed(e.to_string()))?;

        stream
            .play()
            .map_err(|e| DecibriError::StreamStartFailed(e.to_string()))?;

        Ok(BackendStream {
            inner: Mutex::new(Some(stream)),
        })
    }

    #[cfg(feature = "playback")]
    fn open_output_stream(
        &self,
        device: &BackendDevice,
        params: &StreamParams,
        mut on_data: OutputDataCallback,
        mut on_error: StreamErrorCallback,
    ) -> Result<BackendStream, DecibriError> {
        let stream = device
            .inner
            .build_output_stream(
                &stream_config(params),
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    on_data(data);
                },
                move |err| {
                    on_error(DecibriError::DeviceFailed {
                        source: Box::new(err),
                    });
                },
                None,
            )
            .map_err(|e| DecibriError::StreamOpenFailed(e.to_string()))?;

        stream
            .play()
            .map_err(|e| DecibriError::StreamStartFailed(e.to_string()))?;

        Ok(BackendStream {
            inner: Mutex::new(Some(stream)),
        })
    }
}

/// Build a cpal stream config from decibri-owned [`StreamParams`].
fn stream_config(params: &StreamParams) -> cpal::StreamConfig {
    cpal::StreamConfig {
        channels: params.channels,
        sample_rate: params.sample_rate,
        buffer_size: match params.frames_per_buffer {
            Some(frames) => cpal::BufferSize::Fixed(frames),
            None => cpal::BufferSize::Default,
        },
    }
}

// ── cpal device enumeration and resolution ─────────────────────────────────
//
// Direction-generic over the input vs output differences in cpal's enumeration
// and resolution APIs, so one implementation of `enumerate_devices` and
// `resolve_device_generic` covers both. The pure `is_default` matching and the
// `ComputedRow` row type are cpal-free and live in `crate::device`.

/// Internal trait abstracting the input vs output differences in cpal's device
/// enumeration and resolution APIs. Each `impl` is a small table of function
/// pointers into cpal.
trait DeviceDirection {
    type Info;

    fn default_device(host: &cpal::Host) -> Option<cpal::Device>;
    fn list_devices(host: &cpal::Host) -> Result<Vec<cpal::Device>, DecibriError>;
    /// Returns `(channels, sample_rate)`; `(0, 0)` on config failure (matches
    /// the historical behaviour of surfacing an "unknown" device rather than
    /// propagating the error).
    fn default_config(device: &cpal::Device) -> (u16, u32);
    fn build_info(row: ComputedRow) -> Self::Info;
    fn no_device_error() -> DecibriError;
    /// Direction-specific "not found" error for a `Name` / `Id` lookup miss.
    /// Input returns [`DecibriError::MicrophoneNotFound`]; Output returns
    /// [`DecibriError::SpeakerNotFound`].
    fn not_found_error(query: String) -> DecibriError;
}

struct Input;

struct Output;

impl DeviceDirection for Input {
    type Info = MicrophoneInfo;

    fn default_device(host: &cpal::Host) -> Option<cpal::Device> {
        host.default_input_device()
    }

    fn list_devices(host: &cpal::Host) -> Result<Vec<cpal::Device>, DecibriError> {
        host.input_devices()
            .map(|it| it.collect())
            .map_err(|e| DecibriError::DeviceEnumerationFailed(e.to_string()))
    }

    fn default_config(device: &cpal::Device) -> (u16, u32) {
        match device.default_input_config() {
            Ok(config) => (config.channels(), config.sample_rate()),
            Err(_) => (0, 0),
        }
    }

    fn build_info(row: ComputedRow) -> Self::Info {
        MicrophoneInfo {
            index: row.index,
            name: row.name,
            id: row.id,
            max_input_channels: row.channels,
            default_sample_rate: row.sample_rate,
            is_default: row.is_default,
        }
    }

    fn no_device_error() -> DecibriError {
        DecibriError::NoMicrophoneFound
    }

    fn not_found_error(query: String) -> DecibriError {
        DecibriError::MicrophoneNotFound(query)
    }
}

impl DeviceDirection for Output {
    type Info = SpeakerInfo;

    fn default_device(host: &cpal::Host) -> Option<cpal::Device> {
        host.default_output_device()
    }

    fn list_devices(host: &cpal::Host) -> Result<Vec<cpal::Device>, DecibriError> {
        host.output_devices()
            .map(|it| it.collect())
            .map_err(|e| DecibriError::DeviceEnumerationFailed(e.to_string()))
    }

    fn default_config(device: &cpal::Device) -> (u16, u32) {
        match device.default_output_config() {
            Ok(config) => (config.channels(), config.sample_rate()),
            Err(_) => (0, 0),
        }
    }

    fn build_info(row: ComputedRow) -> Self::Info {
        SpeakerInfo {
            index: row.index,
            name: row.name,
            id: row.id,
            max_output_channels: row.channels,
            default_sample_rate: row.sample_rate,
            is_default: row.is_default,
        }
    }

    fn no_device_error() -> DecibriError {
        DecibriError::NoSpeakerFound
    }

    fn not_found_error(query: String) -> DecibriError {
        DecibriError::SpeakerNotFound(query)
    }
}

/// Shared implementation behind [`AudioBackend::input_devices`] /
/// [`AudioBackend::output_devices`]. Direction-generic via [`DeviceDirection`].
fn enumerate_devices<D: DeviceDirection>() -> Result<Vec<D::Info>, DecibriError> {
    let host = cpal::default_host();

    // Stable per-host id (WASAPI endpoint ID / CoreAudio UID / ALSA pcm_id)
    // for the OS default device. `.id()` is fallible on rare host backends;
    // treat a failure as "no default" rather than propagating.
    let default_id = D::default_device(&host).and_then(|d| d.id().ok());

    let devices = D::list_devices(&host)?;

    let rows: Vec<(Option<cpal::DeviceId>, String, u16, u32)> = devices
        .into_iter()
        .enumerate()
        .map(|(index, device)| {
            let id = device.id().ok();
            let name = device
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| format!("Unknown Device {index}"));
            let (channels, sample_rate) = D::default_config(&device);
            (id, name, channels, sample_rate)
        })
        .collect();

    Ok(compute_is_default(rows, default_id)
        .into_iter()
        .map(D::build_info)
        .collect())
}

/// Shared implementation behind [`AudioBackend::resolve_input_device`] /
/// [`AudioBackend::resolve_output_device`]. Direction-generic via
/// [`DeviceDirection`].
fn resolve_device_generic<D: DeviceDirection>(
    selector: &DeviceSelector,
) -> Result<cpal::Device, DecibriError> {
    let host = cpal::default_host();

    match selector {
        DeviceSelector::Default => D::default_device(&host).ok_or_else(D::no_device_error),

        DeviceSelector::Index(idx) => D::list_devices(&host)?
            .into_iter()
            .nth(*idx)
            .ok_or(DecibriError::DeviceIndexOutOfRange),

        DeviceSelector::Name(query) => {
            let query_lower = query.to_lowercase();
            let devices = D::list_devices(&host)?;

            let mut matches: Vec<(usize, String, cpal::Device)> = Vec::new();
            for (index, device) in devices.into_iter().enumerate() {
                let name = device
                    .description()
                    .map(|d| d.name().to_string())
                    .unwrap_or_default();
                if name.to_lowercase().contains(&query_lower) {
                    matches.push((index, name, device));
                }
            }

            match matches.len() {
                0 => Err(D::not_found_error(query.clone())),
                1 => Ok(matches.into_iter().next().unwrap().2),
                _ => {
                    let match_list = matches
                        .iter()
                        .map(|(idx, name, _)| format!("  [{idx}] {name}"))
                        .collect::<Vec<_>>()
                        .join("\n");
                    Err(DecibriError::MultipleDevicesMatch {
                        name: query.clone(),
                        matches: format!(
                            "{match_list}\nUse a more specific name or pass the device index directly."
                        ),
                    })
                }
            }
        }

        DeviceSelector::Id(query) => {
            // Exact match on cpal::DeviceId's `Display` representation. Device
            // IDs are unique per host, so at most one device matches. A device
            // whose `id()` call fails (rare) is silently skipped; the scenario
            // is treated as "not found", matching `enumerate_devices`, which
            // assigns such a device an empty `id` string that no query matches.
            let devices = D::list_devices(&host)?;
            for device in devices {
                if let Ok(id) = device.id() {
                    if id.to_string() == *query {
                        return Ok(device);
                    }
                }
            }
            Err(D::not_found_error(query.clone()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cpal backend satisfies the [`AudioBackend`] seam contract, including
    /// its `Send + Sync` supertrait bound (the bindings depend on it).
    #[test]
    fn cpal_backend_satisfies_the_seam() {
        fn assert_backend<T: AudioBackend>() {}
        fn assert_send_sync<T: Send + Sync>() {}
        assert_backend::<CpalBackend>();
        assert_send_sync::<CpalBackend>();
    }

    /// The stream handle is `Send + Sync` so the stream types that hold it stay
    /// `Send + Sync` (the bindings require `Sync`).
    #[test]
    fn backend_stream_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BackendStream>();
    }

    /// `stop()` drops the held stream and is idempotent. The test seam holds no
    /// real device, so this asserts the slot is empty after stop and that a
    /// second stop does not panic; the real `Some -> None` drop (device release)
    /// is exercised by the binding suites and integration capture/playback.
    #[test]
    fn backend_stream_stop_empties_and_is_idempotent() {
        let stream = BackendStream::empty();
        assert!(!stream.is_active(), "an empty handle holds no stream");
        stream.stop();
        assert!(!stream.is_active(), "stop() leaves the slot empty");
        stream.stop(); // idempotent: an already-empty slot must not panic
    }

    /// Enumeration through the seam reaches cpal and returns without panicking
    /// on any host, including headless CI runners with no audio devices (the
    /// result may be an empty list or an enumeration error, both acceptable).
    #[test]
    fn backend_enumeration_runs() {
        let _ = CpalBackend.input_devices();
        let _ = CpalBackend.output_devices();
    }

    /// The direction-specific `not_found_error` produces the correct
    /// `DecibriError` variant and display string for each direction: input
    /// lookups return `MicrophoneNotFound`, output lookups `SpeakerNotFound`.
    #[test]
    fn not_found_error_produces_direction_correct_variant() {
        let input_err = Input::not_found_error("nonexistent-input".to_string());
        let output_err = Output::not_found_error("nonexistent-output".to_string());

        assert!(
            matches!(input_err, DecibriError::MicrophoneNotFound(ref s) if s == "nonexistent-input"),
            "Input direction must return DecibriError::MicrophoneNotFound"
        );
        assert!(
            matches!(output_err, DecibriError::SpeakerNotFound(ref s) if s == "nonexistent-output"),
            "Output direction must return DecibriError::SpeakerNotFound"
        );

        assert_eq!(
            input_err.to_string(),
            "No microphone found matching \"nonexistent-input\"",
            "Input variant's Display must name the microphone"
        );
        assert_eq!(
            output_err.to_string(),
            "No speaker found matching \"nonexistent-output\"",
            "Output variant's Display must name the speaker"
        );
    }
}
