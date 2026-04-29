"""decibri Python bindings.

Cross-platform audio capture, output, and voice activity detection.

Phase 2 sync surface plus Phase 5 async surface (AsyncDecibri,
AsyncDecibriOutput). NumPy zero-copy support is held for Phase 6.
"""

from decibri._async_classes import AsyncDecibri, AsyncDecibriOutput
from decibri._classes import Decibri, DecibriOutput
from decibri._decibri import (
    DecibriBridge,
    DecibriOutputBridge,
    DeviceInfo,
    OutputDeviceInfo,
    VersionInfo,
)
from decibri.exceptions import (
    AlreadyRunning,
    CaptureStreamClosed,
    ChannelsOutOfRange,
    DecibriError,
    DeviceEnumerationFailed,
    DeviceIndexOutOfRange,
    DeviceNotFound,
    FramesPerBufferOutOfRange,
    InvalidFormat,
    MultipleDevicesMatch,
    NoMicrophoneFound,
    NoOutputDeviceFound,
    NotAnInputDevice,
    OrtError,
    OrtInferenceFailed,
    OrtInitFailed,
    OrtLoadFailed,
    OrtPathError,
    OrtPathInvalid,
    OrtSessionBuildFailed,
    OrtTensorCreateFailed,
    OrtTensorExtractFailed,
    OrtThreadsConfigFailed,
    OutputDeviceNotFound,
    OutputStreamClosed,
    PermissionDenied,
    SampleRateOutOfRange,
    StreamOpenFailed,
    StreamStartFailed,
    VadModelLoadFailed,
    VadSampleRateUnsupported,
    VadThresholdOutOfRange,
)

__version__ = "0.1.0a1"

__all__ = [
    # Public Python wrapper classes (sync)
    "Decibri",
    "DecibriOutput",
    # Public Python wrapper classes (async; Phase 5)
    "AsyncDecibri",
    "AsyncDecibriOutput",
    # Internal pyclasses (advanced use)
    "DecibriBridge",
    "DecibriOutputBridge",
    # Value types
    "DeviceInfo",
    "OutputDeviceInfo",
    "VersionInfo",
    # Exception hierarchy (32 classes)
    "AlreadyRunning",
    "CaptureStreamClosed",
    "ChannelsOutOfRange",
    "DecibriError",
    "DeviceEnumerationFailed",
    "DeviceIndexOutOfRange",
    "DeviceNotFound",
    "FramesPerBufferOutOfRange",
    "InvalidFormat",
    "MultipleDevicesMatch",
    "NoMicrophoneFound",
    "NoOutputDeviceFound",
    "NotAnInputDevice",
    "OrtError",
    "OrtInferenceFailed",
    "OrtInitFailed",
    "OrtLoadFailed",
    "OrtPathError",
    "OrtPathInvalid",
    "OrtSessionBuildFailed",
    "OrtTensorCreateFailed",
    "OrtTensorExtractFailed",
    "OrtThreadsConfigFailed",
    "OutputDeviceNotFound",
    "OutputStreamClosed",
    "PermissionDenied",
    "SampleRateOutOfRange",
    "StreamOpenFailed",
    "StreamStartFailed",
    "VadModelLoadFailed",
    "VadSampleRateUnsupported",
    "VadThresholdOutOfRange",
]
