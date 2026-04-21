#[cfg(any(feature = "capture", feature = "output"))]
use cpal::traits::{DeviceTrait, HostTrait};

use crate::error::DecibriError;

/// Information about an audio input device.
#[derive(Debug, Clone)]
pub struct DeviceInfo {
    pub index: usize,
    pub name: String,
    pub max_input_channels: u16,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// Information about an audio output device.
#[derive(Debug, Clone)]
pub struct OutputDeviceInfo {
    pub index: usize,
    pub name: String,
    pub max_output_channels: u16,
    pub default_sample_rate: u32,
    pub is_default: bool,
}

/// How to select an audio device.
#[derive(Debug, Clone)]
pub enum DeviceSelector {
    /// Use the system default device.
    Default,
    /// Select by device index.
    Index(usize),
    /// Select by case-insensitive name substring match.
    Name(String),
}

/// Internal: enumerated device row with `is_default` already resolved.
/// Used as the output of the pure `compute_is_default` helper so both input
/// and output enumeration can share the same device-identity matching logic
/// (and so the logic is testable without mocking `cpal::Host`).
#[cfg(any(feature = "capture", feature = "output"))]
struct ComputedRow {
    index: usize,
    name: String,
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
fn compute_is_default<Id: PartialEq>(
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
            ComputedRow {
                index,
                name,
                channels,
                sample_rate,
                is_default,
            }
        })
        .collect()
}

/// Enumerate all available audio input devices.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn enumerate_input_devices() -> Result<Vec<DeviceInfo>, DecibriError> {
    let host = cpal::default_host();

    // Stable per-host id (WASAPI endpoint ID / CoreAudio UID / ALSA pcm_id)
    // for the OS default device. `.id()` is fallible on rare host backends;
    // treat a failure as "no default" rather than propagating.
    let default_id = host.default_input_device().and_then(|d| d.id().ok());

    let devices = host
        .input_devices()
        .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?;

    let rows: Vec<(Option<cpal::DeviceId>, String, u16, u32)> = devices
        .enumerate()
        .map(|(index, device)| {
            let id = device.id().ok();
            let name = device
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| format!("Unknown Device {index}"));
            let (channels, sample_rate) = match device.default_input_config() {
                Ok(config) => (config.channels(), config.sample_rate()),
                Err(_) => (0, 0),
            };
            (id, name, channels, sample_rate)
        })
        .collect();

    Ok(compute_is_default(rows, default_id)
        .into_iter()
        .map(|r| DeviceInfo {
            index: r.index,
            name: r.name,
            max_input_channels: r.channels,
            default_sample_rate: r.sample_rate,
            is_default: r.is_default,
        })
        .collect())
}

/// Resolve a device selector to a cpal Device.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn resolve_device(selector: &DeviceSelector) -> Result<cpal::Device, DecibriError> {
    let host = cpal::default_host();

    match selector {
        DeviceSelector::Default => host
            .default_input_device()
            .ok_or(DecibriError::NoMicrophoneFound),

        DeviceSelector::Index(idx) => {
            let devices: Vec<_> = host
                .input_devices()
                .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?
                .collect();

            devices
                .into_iter()
                .nth(*idx)
                .ok_or(DecibriError::DeviceIndexOutOfRange)
        }

        DeviceSelector::Name(query) => {
            let query_lower = query.to_lowercase();
            let devices = host
                .input_devices()
                .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?;

            let mut matches: Vec<(usize, String, cpal::Device)> = Vec::new();
            for (index, device) in devices.enumerate() {
                let name = device
                    .description()
                    .map(|d| d.name().to_string())
                    .unwrap_or_default();
                if name.to_lowercase().contains(&query_lower) {
                    matches.push((index, name, device));
                }
            }

            match matches.len() {
                0 => Err(DecibriError::DeviceNotFound(query.clone())),
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
    }
}

/// Enumerate all available audio output devices.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn enumerate_output_devices() -> Result<Vec<OutputDeviceInfo>, DecibriError> {
    let host = cpal::default_host();

    // Stable per-host id for the OS default output device; see
    // `enumerate_input_devices` for rationale (same Issue #14 fix pattern).
    let default_id = host.default_output_device().and_then(|d| d.id().ok());

    let devices = host
        .output_devices()
        .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?;

    let rows: Vec<(Option<cpal::DeviceId>, String, u16, u32)> = devices
        .enumerate()
        .map(|(index, device)| {
            let id = device.id().ok();
            let name = device
                .description()
                .map(|d| d.name().to_string())
                .unwrap_or_else(|_| format!("Unknown Device {index}"));
            let (channels, sample_rate) = match device.default_output_config() {
                Ok(config) => (config.channels(), config.sample_rate()),
                Err(_) => (0, 0),
            };
            (id, name, channels, sample_rate)
        })
        .collect();

    Ok(compute_is_default(rows, default_id)
        .into_iter()
        .map(|r| OutputDeviceInfo {
            index: r.index,
            name: r.name,
            max_output_channels: r.channels,
            default_sample_rate: r.sample_rate,
            is_default: r.is_default,
        })
        .collect())
}

/// Resolve an output device selector to a cpal Device.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn resolve_output_device(selector: &DeviceSelector) -> Result<cpal::Device, DecibriError> {
    let host = cpal::default_host();

    match selector {
        DeviceSelector::Default => host
            .default_output_device()
            .ok_or(DecibriError::NoOutputDeviceFound),

        DeviceSelector::Index(idx) => {
            let devices: Vec<_> = host
                .output_devices()
                .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?
                .collect();

            devices
                .into_iter()
                .nth(*idx)
                .ok_or(DecibriError::DeviceIndexOutOfRange)
        }

        DeviceSelector::Name(query) => {
            let query_lower = query.to_lowercase();
            let devices = host
                .output_devices()
                .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?;

            let mut matches: Vec<(usize, String, cpal::Device)> = Vec::new();
            for (index, device) in devices.enumerate() {
                let name = device
                    .description()
                    .map(|d| d.name().to_string())
                    .unwrap_or_default();
                if name.to_lowercase().contains(&query_lower) {
                    matches.push((index, name, device));
                }
            }

            match matches.len() {
                0 => Err(DecibriError::DeviceNotFound(query.clone())),
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
    }
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
}
