#[cfg(any(feature = "capture", feature = "output"))]
use cpal::traits::{DeviceTrait, HostTrait};

use crate::error::DecibriError;

/// Information about an audio input device.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    /// Stable per-host device ID, suitable for passing to
    /// [`DeviceSelector::Id`] for selection that survives across
    /// enumerations.
    ///
    /// The string is cpal's `DeviceId` [`Display`] output:
    /// - Windows (WASAPI): endpoint ID (e.g. `{0.0.1.00000000}.{...}`)
    /// - macOS (CoreAudio): device UID
    /// - Linux (ALSA): PCM identifier
    ///
    /// Empty string if cpal could not produce a stable ID for this
    /// device (rare; some host backends cannot assign IDs to every
    /// enumerated device). A device with an empty `id` cannot be
    /// selected by [`DeviceSelector::Id`] and must be selected via
    /// [`DeviceSelector::Index`] or [`DeviceSelector::Name`] instead.
    ///
    /// [`Display`]: std::fmt::Display
    pub id: String,
    pub max_input_channels: u16,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// Information about an audio output device.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OutputDeviceInfo {
    pub index: usize,
    pub name: String,
    /// Stable per-host device ID. See [`DeviceInfo::id`] for format and
    /// fallback semantics; the rules are identical for output devices.
    pub id: String,
    pub max_output_channels: u16,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// How to select an audio device.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DeviceSelector {
    /// Use the system default device.
    Default,
    /// Select by device index.
    Index(usize),
    /// Select by case-insensitive name substring match.
    Name(String),
    /// Select by stable per-host device ID.
    ///
    /// The ID is the string produced by `cpal::DeviceId`'s [`Display`] impl
    /// (WASAPI endpoint ID on Windows, CoreAudio UID on macOS, ALSA pcm_id
    /// on Linux). Matching is exact string equality, not substring.
    ///
    /// Obtain an ID from [`DeviceInfo::id`] / [`OutputDeviceInfo::id`]
    /// returned by [`enumerate_input_devices`] / [`enumerate_output_devices`].
    ///
    /// Prefer `Id` over [`Self::Name`] when you need stable device selection
    /// across enumerations: display names can shift across OS versions or
    /// when other devices are plugged in, but per-host IDs don't.
    ///
    /// [`Display`]: std::fmt::Display
    Id(String),
}

/// Internal: enumerated device row with `is_default` already resolved.
/// Used as the output of the pure `compute_is_default` helper so both input
/// and output enumeration can share the same device-identity matching logic
/// (and so the logic is testable without mocking `cpal::Host`).
#[cfg(any(feature = "capture", feature = "output"))]
struct ComputedRow {
    index: usize,
    name: String,
    /// Stable per-host device ID as a string, or empty if the device has no
    /// cpal-assignable ID. Forwarded to `DeviceInfo.id` / `OutputDeviceInfo.id`.
    id: String,
    channels: u16,
    sample_rate: u32,
    is_default: bool,
}

/// Pure helper for Issue #14 fix.
///
/// Given a list of `(id, name, channels, sample_rate)` tuples representing
/// enumerated devices and an optional default-device id, produce one
/// `ComputedRow` per input with `is_default` resolved by identity comparison
/// (NOT by name-equality. See [#14](https://github.com/decibri/decibri/issues/14) for context.)
///
/// Semantics:
/// - A row's `id` of `None` (caller couldn't fetch a cpal `DeviceId` for that
///   device) is treated as "not default".
/// - A `default_id` of `None` (no default device reported by the host) means
///   no row is flagged default.
/// - Otherwise, a row is default iff its id equals `default_id`. With a stable
///   per-host id (WASAPI endpoint ID, CoreAudio UID, ALSA pcm_id) exactly one
///   row matches even when multiple devices share a display name.
///
/// Generic over `Id: PartialEq` so unit tests can use `String` ids and verify
/// the matching logic without depending on a live cpal host.
#[cfg(any(feature = "capture", feature = "output"))]
fn compute_is_default<Id: PartialEq + ToString>(
    rows: Vec<(Option<Id>, String, u16, u32)>,
    default_id: Option<Id>,
) -> Vec<ComputedRow> {
    rows.into_iter()
        .enumerate()
        .map(|(index, (id, name, channels, sample_rate))| {
            let is_default = match (id.as_ref(), default_id.as_ref()) {
                (Some(row_id), Some(default)) => row_id == default,
                _ => false,
            };
            // `cpal::DeviceId` implements `Display`, so `.to_string()` gives
            // a stable per-host identifier. `None` (device has no assignable
            // id on this host) becomes an empty string: the enumeration still
            // includes the device but it cannot be selected by `DeviceSelector::Id`.
            let id = id.map(|x| x.to_string()).unwrap_or_default();
            ComputedRow {
                index,
                name,
                id,
                channels,
                sample_rate,
                is_default,
            }
        })
        .collect()
}

/// Internal trait abstracting the input vs output differences in cpal's
/// device enumeration and resolution APIs. Allows one generic implementation
/// of `enumerate_devices` and `resolve_device` to cover both directions.
///
/// Not part of the public API; lives here only to deduplicate the per-direction
/// code paths. Each `impl` is a small table of function pointers into cpal.
#[cfg(any(feature = "capture", feature = "output"))]
trait DeviceDirection {
    type Info;

    fn default_device(host: &cpal::Host) -> Option<cpal::Device>;
    fn list_devices(host: &cpal::Host) -> Result<Vec<cpal::Device>, DecibriError>;
    /// Returns `(channels, sample_rate)`; `(0, 0)` on config failure (matches
    /// the pre-consolidation behaviour of surfacing an "unknown" device
    /// rather than propagating the error).
    fn default_config(device: &cpal::Device) -> (u16, u32);
    fn build_info(row: ComputedRow) -> Self::Info;
    fn no_device_error() -> DecibriError;
    /// Direction-specific "not found" error for a `Name` / `Id` lookup miss.
    /// Input returns [`DecibriError::DeviceNotFound`]; Output returns
    /// [`DecibriError::OutputDeviceNotFound`]. Keeping this on the trait
    /// avoids hardcoding the input-oriented variant in the shared
    /// `resolve_device_generic` body.
    fn not_found_error(query: String) -> DecibriError;
}

#[cfg(any(feature = "capture", feature = "output"))]
struct Input;

#[cfg(any(feature = "capture", feature = "output"))]
struct Output;

#[cfg(any(feature = "capture", feature = "output"))]
impl DeviceDirection for Input {
    type Info = DeviceInfo;

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
        DeviceInfo {
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
        DecibriError::DeviceNotFound(query)
    }
}

#[cfg(any(feature = "capture", feature = "output"))]
impl DeviceDirection for Output {
    type Info = OutputDeviceInfo;

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
        OutputDeviceInfo {
            index: row.index,
            name: row.name,
            id: row.id,
            max_output_channels: row.channels,
            default_sample_rate: row.sample_rate,
            is_default: row.is_default,
        }
    }

    fn no_device_error() -> DecibriError {
        DecibriError::NoOutputDeviceFound
    }

    fn not_found_error(query: String) -> DecibriError {
        DecibriError::OutputDeviceNotFound(query)
    }
}

/// Shared implementation for `enumerate_input_devices` / `enumerate_output_devices`.
/// Direction-generic via the `DeviceDirection` trait.
#[cfg(any(feature = "capture", feature = "output"))]
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

/// Shared implementation for `resolve_device` / `resolve_output_device`.
#[cfg(any(feature = "capture", feature = "output"))]
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
            // Exact match on cpal::DeviceId's `Display` representation.
            // Device IDs are unique per host, so at most one device matches.
            // A device whose `id()` call fails (rare; some hosts cannot
            // produce a stable id for every enumerated device) is silently
            // skipped; the scenario is treated as "not found" rather than
            // surfacing a cpal-level error. This matches `enumerate_devices`,
            // which assigns such a device an empty `id` string that no query
            // can match.
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

/// Enumerate all available audio input devices.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn enumerate_input_devices() -> Result<Vec<DeviceInfo>, DecibriError> {
    enumerate_devices::<Input>()
}

/// Enumerate all available audio output devices.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn enumerate_output_devices() -> Result<Vec<OutputDeviceInfo>, DecibriError> {
    enumerate_devices::<Output>()
}

/// Resolve a device selector to a cpal Device.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn resolve_device(selector: &DeviceSelector) -> Result<cpal::Device, DecibriError> {
    resolve_device_generic::<Input>(selector)
}

/// Resolve an output device selector to a cpal Device.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn resolve_output_device(selector: &DeviceSelector) -> Result<cpal::Device, DecibriError> {
    resolve_device_generic::<Output>(selector)
}

#[cfg(all(test, any(feature = "capture", feature = "output")))]
mod tests {
    use super::*;

    // Synthetic id type for testing. Any `PartialEq` works because
    // `compute_is_default` is generic over the id type. Using String keeps the
    // tests readable and avoids any dependency on a live cpal host.
    fn row(id: Option<&str>, name: &str) -> (Option<String>, String, u16, u32) {
        (id.map(String::from), name.to_string(), 2, 48_000)
    }

    /// Issue #14: two devices with the same display name but different per-host
    /// IDs; only the device whose id matches the default is flagged.
    #[test]
    fn test_is_default_two_devices_same_name_different_ids() {
        let rows = vec![
            row(Some("usb-mic-A"), "Microphone"),
            row(Some("usb-mic-B"), "Microphone"),
        ];
        let result = compute_is_default(rows, Some("usb-mic-B".to_string()));

        assert_eq!(result.len(), 2);
        assert!(
            !result[0].is_default,
            "first duplicate-named mic must NOT be flagged when default is the second"
        );
        assert!(
            result[1].is_default,
            "second duplicate-named mic (matching default id) must be flagged"
        );
        // Both rows keep their shared display name. The bug was never about
        // renaming devices, only about misattribution of is_default.
        assert_eq!(result[0].name, "Microphone");
        assert_eq!(result[1].name, "Microphone");
    }

    /// Host reports no default device (rare but possible on headless Linux /
    /// CI runners with no audio devices): no row is flagged.
    #[test]
    fn test_is_default_no_default_reported() {
        let rows = vec![row(Some("mic-A"), "Mic A"), row(Some("mic-B"), "Mic B")];
        let result = compute_is_default::<String>(rows, None);

        assert_eq!(result.len(), 2);
        assert!(
            result.iter().all(|r| !r.is_default),
            "no row may be flagged when host reports no default"
        );
    }

    /// A device whose `id()` call failed (represented as `None` in the helper
    /// input) is never flagged default, even if the host reports some default.
    #[test]
    fn test_is_default_row_with_failed_id() {
        let rows = vec![
            row(None, "Mystery device with unavailable id"),
            row(Some("mic-B"), "Mic B"),
        ];
        let result = compute_is_default(rows, Some("mic-B".to_string()));

        assert_eq!(result.len(), 2);
        assert!(
            !result[0].is_default,
            "row with id() == None must never be flagged default"
        );
        assert!(
            result[1].is_default,
            "row whose id matches the default must be flagged"
        );
    }

    /// Empty device list produces an empty result, regardless of what the
    /// host reports as the default.
    #[test]
    fn test_is_default_empty_device_list() {
        let rows: Vec<(Option<String>, String, u16, u32)> = vec![];
        let result_no_default = compute_is_default::<String>(rows.clone(), None);
        let result_with_default = compute_is_default(rows, Some("phantom-mic".to_string()));

        assert!(result_no_default.is_empty());
        assert!(result_with_default.is_empty());
    }

    /// The `id` field on `ComputedRow` is populated from the input `Option<Id>`
    /// via `ToString`, with `None` becoming an empty string. This field flows
    /// through to `DeviceInfo.id` / `OutputDeviceInfo.id` and backs
    /// `DeviceSelector::Id` selection.
    #[test]
    fn test_id_is_propagated_to_computed_row() {
        let rows = vec![
            row(Some("mic-with-id"), "Microphone A"),
            row(None, "Microphone without id"),
        ];
        let result = compute_is_default(rows, Some("mic-with-id".to_string()));

        assert_eq!(result.len(), 2);
        assert_eq!(
            result[0].id, "mic-with-id",
            "Some(id) must be preserved as a non-empty string"
        );
        assert_eq!(
            result[1].id, "",
            "None id must map to an empty string so the device is unreachable via DeviceSelector::Id"
        );
    }

    /// Verifies that the direction-specific `not_found_error` trait method
    /// produces the correct `DecibriError` variant and display string for
    /// each direction. Input lookups must say "input" in the message;
    /// output lookups must say "output". This catches the regression that
    /// motivated adding `OutputDeviceNotFound`: before the split, both
    /// directions produced `DeviceNotFound` whose hardcoded message said
    /// "input" regardless of whether the lookup was against input or
    /// output devices.
    #[test]
    fn test_not_found_error_produces_direction_correct_variant() {
        let input_err = Input::not_found_error("nonexistent-input".to_string());
        let output_err = Output::not_found_error("nonexistent-output".to_string());

        assert!(
            matches!(input_err, DecibriError::DeviceNotFound(ref s) if s == "nonexistent-input"),
            "Input direction must return DecibriError::DeviceNotFound"
        );
        assert!(
            matches!(output_err, DecibriError::OutputDeviceNotFound(ref s) if s == "nonexistent-output"),
            "Output direction must return DecibriError::OutputDeviceNotFound"
        );

        assert_eq!(
            input_err.to_string(),
            "No audio input device found matching \"nonexistent-input\"",
            "Input variant's Display must say \"input\""
        );
        assert_eq!(
            output_err.to_string(),
            "No audio output device found matching \"nonexistent-output\"",
            "Output variant's Display must say \"output\""
        );
    }
}
