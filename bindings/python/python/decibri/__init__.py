"""decibri Python bindings.

Cross-platform audio capture, output, and voice activity detection.

Public surface: ``Microphone`` and ``Speaker`` (sync), ``AsyncMicrophone``
and ``AsyncSpeaker`` (async). Lowercase factory functions
(``decibri.microphone()``, ``decibri.speaker()``,
``decibri.async_microphone()``, ``decibri.async_speaker()``) provide
sounddevice-style call-site convenience over the same classes.
"""

from typing import Any as _Any

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


def microphone(**kwargs: _Any) -> Microphone:
    """Create a ``Microphone`` instance.

    Module-level convenience for ``decibri.Microphone(**kwargs)``;
    mirrors the sounddevice ``sd.rec()`` ergonomic. Both forms are
    supported and equivalent.
    """
    return Microphone(**kwargs)


def speaker(**kwargs: _Any) -> Speaker:
    """Create a ``Speaker`` instance.

    Module-level convenience for ``decibri.Speaker(**kwargs)``.
    """
    return Speaker(**kwargs)


def async_microphone(**kwargs: _Any) -> AsyncMicrophone:
    """Create an ``AsyncMicrophone`` instance.

    Module-level convenience for ``decibri.AsyncMicrophone(**kwargs)``.
    """
    return AsyncMicrophone(**kwargs)


def async_speaker(**kwargs: _Any) -> AsyncSpeaker:
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


def record_to_file(
    path: str,
    duration_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
    device: int | str | None = None,
) -> None:
    """Record microphone audio to a 16-bit PCM WAV file for ``duration_seconds``.

    Highest-leverage convenience helper for the Quickstart path. Wraps
    ``Microphone`` plus the stdlib ``wave`` module so users can capture
    audio without writing a chunk loop.

    Parameters
    ----------
    path : str
        Output WAV file path. Parent directory must exist; the file is
        created or overwritten.
    duration_seconds : float
        Approximate recording duration. Actual duration is rounded up to
        the nearest ``frames_per_buffer`` chunk (1600 frames = 100ms at
        16kHz by default), so the file is at most ~100ms longer than
        requested.
    sample_rate : int, optional
        Capture sample rate in Hz. Default 16000.
    channels : int, optional
        Number of input channels. Default 1 (mono).
    device : int | str | None, optional
        Input device selector. ``None`` (default) uses the system
        default input. Pass an integer index from ``devices()`` or a
        substring of the device name.

    Notes
    -----
    Always writes 16-bit PCM (``dtype='int16'``); WAV's most portable
    format. For float32 capture or other formats use ``Microphone``
    directly.
    """
    import wave

    bytes_per_sample = 2  # int16
    frames_per_buffer = 1600  # 100ms at 16kHz; cross-binding default
    chunks_needed = max(
        1, int((duration_seconds * sample_rate + frames_per_buffer - 1) // frames_per_buffer)
    )

    with wave.open(path, "wb") as wav, Microphone(
        sample_rate=sample_rate,
        channels=channels,
        frames_per_buffer=frames_per_buffer,
        dtype="int16",
        device=device,
    ) as mic:
        wav.setnchannels(channels)
        wav.setsampwidth(bytes_per_sample)
        wav.setframerate(sample_rate)

        for _ in range(chunks_needed):
            chunk = mic.read(timeout_ms=2000)
            if chunk is None:
                break
            # int16 mode + numpy=False guarantees bytes from read().
            assert isinstance(chunk, bytes)
            wav.writeframes(chunk)


async def async_record_to_file(
    path: str,
    duration_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
    device: int | str | None = None,
) -> None:
    """Async parallel of ``record_to_file``.

    Records microphone audio to a 16-bit PCM WAV file using
    ``AsyncMicrophone``. Same parameters and semantics as
    ``record_to_file``; await this from an asyncio context.
    """
    import wave

    bytes_per_sample = 2  # int16
    frames_per_buffer = 1600
    chunks_needed = max(
        1, int((duration_seconds * sample_rate + frames_per_buffer - 1) // frames_per_buffer)
    )

    mic = AsyncMicrophone(
        sample_rate=sample_rate,
        channels=channels,
        frames_per_buffer=frames_per_buffer,
        dtype="int16",
        device=device,
    )
    with wave.open(path, "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(bytes_per_sample)
        wav.setframerate(sample_rate)
        async with mic as d:
            for _ in range(chunks_needed):
                chunk = await d.read(timeout_ms=2000)
                if chunk is None:
                    break
                assert isinstance(chunk, bytes)
                wav.writeframes(chunk)


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
    # File convenience functions (Phase 7.6 Item C5)
    "record_to_file",
    "async_record_to_file",
    # Internal pyclasses (advanced use, accessible via attribute lookup
    # but not in __all__).
    # Value types
    "DeviceInfo",
    "OutputDeviceInfo",
    "VersionInfo",
    # Exception hierarchy entry points (Phase 7.6 Item C3, Shape C2):
    # only the catch-target roots are surfaced in __all__ to keep
    # `from decibri import *` legible. The full 32-class hierarchy
    # remains importable via top-level attribute lookup
    # (``decibri.<Class>``) AND via the explicit submodule
    # (``decibri.exceptions.<Class>``); see decibri.exceptions for the
    # complete list. The three names below cover catch-anything,
    # catch-any-ORT, and catch-any-ORT-path-error respectively.
    "DecibriError",
    "OrtError",
    "OrtPathError",
]
