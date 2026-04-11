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

/// Enumerate all available audio input devices.
#[cfg(any(feature = "capture", feature = "output"))]
pub fn enumerate_input_devices() -> Result<Vec<DeviceInfo>, DecibriError> {
    let host = cpal::default_host();

    let default_device_name = host
        .default_input_device()
        .and_then(|d| d.description().ok())
        .map(|desc| desc.name().to_string());

    let devices = host
        .input_devices()
        .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?;

    let mut result = Vec::new();
    for (index, device) in devices.enumerate() {
        let name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| format!("Unknown Device {index}"));

        let default_config = device.default_input_config();
        let (max_channels, default_rate) = match default_config {
            Ok(config) => (config.channels(), config.sample_rate()),
            Err(_) => (0, 0),
        };

        let is_default = default_device_name.as_ref() == Some(&name);

        result.push(DeviceInfo {
            index,
            name,
            max_input_channels: max_channels,
            default_sample_rate: default_rate,
            is_default,
        });
    }

    Ok(result)
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

    let default_device_name = host
        .default_output_device()
        .and_then(|d| d.description().ok())
        .map(|desc| desc.name().to_string());

    let devices = host
        .output_devices()
        .map_err(|e| DecibriError::Other(format!("Failed to enumerate devices: {e}")))?;

    let mut result = Vec::new();
    for (index, device) in devices.enumerate() {
        let name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| format!("Unknown Device {index}"));

        let default_config = device.default_output_config();
        let (max_channels, default_rate) = match default_config {
            Ok(config) => (config.channels(), config.sample_rate()),
            Err(_) => (0, 0),
        };

        let is_default = default_device_name.as_ref() == Some(&name);

        result.push(OutputDeviceInfo {
            index,
            name,
            max_output_channels: max_channels,
            default_sample_rate: default_rate,
            is_default,
        });
    }

    Ok(result)
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
