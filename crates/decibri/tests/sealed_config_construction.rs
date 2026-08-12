//! Public-surface stability: the config structs are `#[non_exhaustive]`.
//!
//! From another crate a `#[non_exhaustive]` struct can no longer be built with
//! a struct literal (or a `..default()` functional update), but its public
//! fields can still be assigned after a `default()` construction. This test
//! lives in the `tests/` integration target (a separate crate from `decibri`)
//! so it exercises that cross-crate restriction, which the in-crate unit tests
//! cannot see (struct literals are always allowed within the defining crate).
//!
//! Each test default-constructs a config, mutates its public fields, and
//! asserts the mutation took effect, pinning the supported post-seal
//! construction pattern for external Rust consumers.

#[cfg(feature = "capture")]
#[test]
fn microphone_config_default_then_mutate() {
    use decibri::microphone::MicrophoneConfig;

    let mut config = MicrophoneConfig::default();
    config.sample_rate = 48_000;
    config.channels = 2;
    config.frames_per_buffer = 480;

    assert_eq!(config.sample_rate, 48_000);
    assert_eq!(config.channels, 2);
    assert_eq!(config.frames_per_buffer, 480);
    assert!(config.validate().is_ok());
}

#[cfg(feature = "playback")]
#[test]
fn speaker_config_default_then_mutate() {
    use decibri::speaker::SpeakerConfig;

    let mut config = SpeakerConfig::default();
    config.sample_rate = 44_100;
    config.channels = 2;

    assert_eq!(config.sample_rate, 44_100);
    assert_eq!(config.channels, 2);
    assert!(config.validate().is_ok());
}

#[cfg(feature = "vad")]
#[test]
fn vad_config_default_then_mutate() {
    use decibri::vad::VadConfig;

    let mut config = VadConfig::default();
    config.sample_rate = 8_000;
    config.threshold = 0.7;

    assert_eq!(config.sample_rate, 8_000);
    assert!((config.threshold - 0.7).abs() < f32::EPSILON);
    // 8 kHz is a supported VAD rate; validate returns the 256-sample window.
    assert_eq!(config.validate().unwrap(), 256);
}
