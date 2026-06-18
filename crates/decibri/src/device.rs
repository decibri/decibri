//! Audio device discovery and selection.
//!
//! [`input_devices`] and [`output_devices`] list the available microphones and
//! speakers as [`MicrophoneInfo`] / [`SpeakerInfo`]. A [`DeviceSelector`] picks
//! one by system default, index, case-insensitive name substring, or stable
//! per-host id, and is the value [`crate::MicrophoneConfig`] and
//! [`crate::SpeakerConfig`] carry in their `device` field.
//!
//! The platform enumeration and resolution themselves run behind
//! [`crate::backend::AudioBackend`]; this module owns the public device types
//! and the pure `is_default` matching, and routes listing through the seam.

use crate::error::DecibriError;

/// Information about a microphone (audio input) device.
///
/// Returned by [`input_devices`] and
/// [`Microphone::devices`](crate::Microphone::devices). For speakers, see
/// [`SpeakerInfo`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MicrophoneInfo {
    pub index: usize,
    pub name: String,
    /// Stable per-host device ID, suitable for passing to
    /// [`DeviceSelector::Id`] for selection that survives across
    /// enumerations.
    ///
    /// The string is the platform device identifier:
    /// - Windows (WASAPI): endpoint ID (e.g. `{0.0.1.00000000}.{...}`)
    /// - macOS (CoreAudio): device UID
    /// - Linux (ALSA): PCM identifier
    ///
    /// Empty string if the platform could not produce a stable ID for this
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

/// Information about a speaker (audio output) device.
///
/// Returned by [`output_devices`] and
/// [`Speaker::devices`](crate::Speaker::devices). For microphones, see
/// [`MicrophoneInfo`].
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SpeakerInfo {
    pub index: usize,
    pub name: String,
    /// Stable per-host device ID. See [`MicrophoneInfo::id`] for format and
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
    /// Obtain an ID from [`MicrophoneInfo::id`] / [`SpeakerInfo::id`]
    /// returned by [`input_devices`] / [`output_devices`].
    ///
    /// Prefer `Id` over [`Self::Name`] when you need stable device selection
    /// across enumerations: display names can shift across OS versions or
    /// when other devices are plugged in, but per-host IDs don't.
    ///
    /// [`Display`]: std::fmt::Display
    Id(String),
}

/// Internal: enumerated device row with `is_default` already resolved.
/// Used as the output of the pure [`compute_is_default`] helper so both input
/// and output enumeration can share the same device-identity matching logic
/// (and so the logic is testable without mocking the host).
#[cfg(any(feature = "capture", feature = "playback"))]
pub(crate) struct ComputedRow {
    pub(crate) index: usize,
    pub(crate) name: String,
    /// Stable per-host device ID as a string, or empty if the device has no
    /// host-assignable ID. Forwarded to `MicrophoneInfo.id` / `SpeakerInfo.id`.
    pub(crate) id: String,
    pub(crate) channels: u16,
    pub(crate) sample_rate: u32,
    pub(crate) is_default: bool,
}

/// Pure helper for Issue #14 fix.
///
/// Given a list of `(id, name, channels, sample_rate)` tuples representing
/// enumerated devices and an optional default-device id, produce one
/// `ComputedRow` per input with `is_default` resolved by identity comparison
/// (NOT by name-equality. See [#14](https://github.com/decibri/decibri/issues/14) for context.)
///
/// Semantics:
/// - A row's `id` of `None` (caller couldn't fetch a host `DeviceId` for that
///   device) is treated as "not default".
/// - A `default_id` of `None` (no default device reported by the host) means
///   no row is flagged default.
/// - Otherwise, a row is default iff its id equals `default_id`. With a stable
///   per-host id (WASAPI endpoint ID, CoreAudio UID, ALSA pcm_id) exactly one
///   row matches even when multiple devices share a display name.
///
/// Generic over `Id: PartialEq` so unit tests can use `String` ids and verify
/// the matching logic without depending on a live host.
#[cfg(any(feature = "capture", feature = "playback"))]
pub(crate) fn compute_is_default<Id: PartialEq + ToString>(
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

/// List the available microphone (input) devices.
///
/// Also available as [`Microphone::devices`](crate::Microphone::devices).
#[cfg(any(feature = "capture", feature = "playback"))]
pub fn input_devices() -> Result<Vec<MicrophoneInfo>, DecibriError> {
    use crate::backend::AudioBackend;
    crate::backend::CpalBackend.input_devices()
}

/// List the available speaker (output) devices.
///
/// Also available as [`Speaker::devices`](crate::Speaker::devices).
#[cfg(any(feature = "capture", feature = "playback"))]
pub fn output_devices() -> Result<Vec<SpeakerInfo>, DecibriError> {
    use crate::backend::AudioBackend;
    crate::backend::CpalBackend.output_devices()
}

#[cfg(all(test, any(feature = "capture", feature = "playback")))]
mod tests {
    use super::*;

    // Synthetic id type for testing. Any `PartialEq` works because
    // `compute_is_default` is generic over the id type. Using String keeps the
    // tests readable and avoids any dependency on a live host.
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
    /// through to `MicrophoneInfo.id` / `SpeakerInfo.id` and backs
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
}
