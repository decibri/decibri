"""decibri Python bindings.

Cross-platform audio capture, output, and voice activity detection.

Public surface: ``Microphone`` and ``Speaker`` (sync), ``AsyncMicrophone``
and ``AsyncSpeaker`` (async). Lowercase factory functions
(``decibri.microphone()``, ``decibri.speaker()``,
``decibri.async_microphone()``, ``decibri.async_speaker()``) provide
sounddevice-style call-site convenience over the same classes.
"""

from typing import Any

from decibri._async_classes import AsyncMicrophone, AsyncSpeaker
from decibri._classes import Microphone, Speaker
from decibri._decibri import (
    DeviceInfo,
    MicrophoneBridge,
    OutputDeviceInfo,
    SpeakerBridge,
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


def microphone(**kwargs: Any) -> Microphone:
    """Create a ``Microphone`` instance.

    Module-level convenience for ``decibri.Microphone(**kwargs)``;
    mirrors the sounddevice ``sd.rec()`` ergonomic. Both forms are
    supported and equivalent.
    """
    return Microphone(**kwargs)


def speaker(**kwargs: Any) -> Speaker:
    """Create a ``Speaker`` instance.

    Module-level convenience for ``decibri.Speaker(**kwargs)``.
    """
    return Speaker(**kwargs)


def async_microphone(**kwargs: Any) -> AsyncMicrophone:
    """Create an ``AsyncMicrophone`` instance.

    Module-level convenience for ``decibri.AsyncMicrophone(**kwargs)``.
    """
    return AsyncMicrophone(**kwargs)


def async_speaker(**kwargs: Any) -> AsyncSpeaker:
    """Create an ``AsyncSpeaker`` instance.

    Module-level convenience for ``decibri.AsyncSpeaker(**kwargs)``.
    """
    return AsyncSpeaker(**kwargs)


def devices() -> list[DeviceInfo]:
    """List available audio input devices.

    Module-level convenience for ``Microphone.devices()``. Returns input
    devices only; for output devices use ``decibri.output_devices()``.
    To list both, concatenate the two.
    """
    return Microphone.devices()


def output_devices() -> list[OutputDeviceInfo]:
    """List available audio output devices.

    Module-level convenience for ``Speaker.devices()``.
    """
    return Speaker.devices()


def version() -> VersionInfo:
    """Return version info for decibri, the audio backend, and the binding.

    Module-level convenience for ``Microphone.version()``.
    """
    return Microphone.version()


__all__ = [
    # Public Python wrapper classes (sync)
    "Microphone",
    "Speaker",
    # Public Python wrapper classes (async; Phase 5)
    "AsyncMicrophone",
    "AsyncSpeaker",
    # Lowercase factory functions (Phase 7.5; sounddevice-style)
    "microphone",
    "speaker",
    "async_microphone",
    "async_speaker",
    # Module-level convenience functions (Phase 7.5)
    "devices",
    "output_devices",
    "version",
    # Internal pyclasses (advanced use)
    "MicrophoneBridge",
    "SpeakerBridge",
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
