use thiserror::Error;

/// Errors produced by decibri operations.
#[derive(Debug, Error)]
pub enum DecibriError {
    #[error("sampleRate must be between 1000 and 384000")]
    SampleRateOutOfRange,

    #[error("channels must be between 1 and 32")]
    ChannelsOutOfRange,

    #[error("framesPerBuffer must be between 64 and 65536")]
    FramesPerBufferOutOfRange,

    #[error("format must be 'int16' or 'float32'")]
    InvalidFormat,

    #[error("No audio input device found matching \"{0}\"")]
    DeviceNotFound(String),

    #[error("Multiple devices match \"{name}\":\n{matches}")]
    MultipleDevicesMatch { name: String, matches: String },

    #[error("device index out of range. Call Decibri.devices() to list available devices")]
    DeviceIndexOutOfRange,

    #[error("No microphone found. Check system audio input settings.")]
    NoMicrophoneFound,

    #[error("No audio output device found. Check system audio settings.")]
    NoOutputDeviceFound,

    #[error("Selected device is not a valid input device.")]
    NotAnInputDevice,

    #[error("Decibri is already running. Call stop() first.")]
    AlreadyRunning,

    #[error("Failed to open audio stream: {0}")]
    StreamOpenFailed(String),

    #[error("Failed to start audio stream: {0}")]
    StreamStartFailed(String),

    #[error("Microphone permission denied. Enable in System Preferences > Security & Privacy.")]
    PermissionDenied,

    #[error("{0}")]
    Other(String),
}
