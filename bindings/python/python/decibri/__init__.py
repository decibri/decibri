"""decibri Python bindings.

Cross-platform audio capture, output, and voice activity detection.

Public surface: ``Microphone`` and ``Speaker`` (sync), ``AsyncMicrophone``
and ``AsyncSpeaker`` (async). Construct directly via the class
constructors (``decibri.Microphone(...)`` / ``decibri.Speaker(...)``).
"""

from decibri._async_classes import AsyncMicrophone, AsyncSpeaker
from decibri._classes import Chunk, Microphone, Speaker
from decibri._decibri import (
    MicrophoneInfo,
    SpeakerInfo,
    VersionInfo,
)
from decibri.exceptions import (
    AlreadyRunning,
    ChannelsOutOfRange,
    DecibriError,
    DeviceEnumerationFailed,
    DeviceError,
    DeviceIndexOutOfRange,
    ForkAfterOrtInit,
    FramesPerBufferOutOfRange,
    InvalidFormat,
    MicrophoneNotFound,
    MicrophoneStreamClosed,
    MultipleDevicesMatch,
    NoMicrophoneFound,
    NoSpeakerFound,
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
    PermissionDenied,
    SampleRateOutOfRange,
    SpeakerNotFound,
    SpeakerStreamClosed,
    StreamOpenFailed,
    StreamStartFailed,
    VadModelLoadFailed,
    VadSampleRateUnsupported,
    VadThresholdOutOfRange,
)

__version__ = "0.4.2"


def input_devices() -> list[MicrophoneInfo]:
    """List available audio input devices.

    Module-level convenience for ``Microphone.devices()``. Returns
    input devices only; for output devices use
    ``decibri.output_devices()``. To list both, concatenate the two.
    """
    return Microphone.devices()


def output_devices() -> list[SpeakerInfo]:
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
        Approximate recording duration. The loop accumulates audio until
        the captured frame count meets or exceeds the requested amount,
        so the file is at most one chunk longer than requested.
    sample_rate : int, optional
        Capture sample rate in Hz. Default 16000.
    channels : int, optional
        Number of input channels. Default 1 (mono).
    device : int | str | None, optional
        Input device selector. ``None`` (default) uses the system
        default input. Pass an integer index from ``input_devices()``
        or a substring of the device name.

    Notes
    -----
    Always writes 16-bit PCM (``dtype='int16'``); WAV's most portable
    format. For float32 capture or other formats use ``Microphone``
    directly.
    """
    import wave

    bytes_per_sample = 2  # int16
    frames_per_buffer = 1600  # 100ms at 16kHz; cross-binding default
    total_frames_needed = max(1, int(duration_seconds * sample_rate))

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

        # Frame-count termination, not chunk-count: the cpal callback's
        # actual buffer size depends on the platform audio backend (on
        # Windows WASAPI it is typically the device period, not the
        # requested frames_per_buffer), so a chunk-count loop captures
        # the wrong duration on backends that ignore the buffer hint.
        total_frames_written = 0
        while total_frames_written < total_frames_needed:
            chunk = mic.read()
            if chunk is None:
                break
            # int16 mode + numpy=False guarantees bytes from read().
            assert isinstance(chunk, bytes)
            wav.writeframes(chunk)
            total_frames_written += len(chunk) // (bytes_per_sample * channels)


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
    total_frames_needed = max(1, int(duration_seconds * sample_rate))

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
            # Frame-count termination, not chunk-count: see record_to_file
            # for the rationale (cpal callback size is platform-dependent).
            total_frames_written = 0
            while total_frames_written < total_frames_needed:
                chunk = await d.read()
                if chunk is None:
                    break
                assert isinstance(chunk, bytes)
                wav.writeframes(chunk)
                total_frames_written += len(chunk) // (bytes_per_sample * channels)


__all__ = [
    # Public Python wrapper classes (sync)
    "Microphone",
    "Speaker",
    # Public Python wrapper classes (async)
    "AsyncMicrophone",
    "AsyncSpeaker",
    # Module-level convenience functions
    "input_devices",
    "output_devices",
    "version",
    # File convenience functions
    "record_to_file",
    "async_record_to_file",
    # Value types
    "MicrophoneInfo",
    "SpeakerInfo",
    "VersionInfo",
    # Audio chunk with metadata
    "Chunk",
    # Exception hierarchy entry points:
    # only the catch-target roots plus ForkAfterOrtInit are surfaced in
    # __all__ to keep `from decibri import *` legible. The full
    # 32+-class hierarchy remains importable via top-level attribute
    # lookup (``decibri.<Class>``) AND via the explicit submodule
    # (``decibri.exceptions.<Class>``); see decibri.exceptions for the
    # complete list. The four catch-target names cover catch-anything,
    # catch-any-device-error, catch-any-ORT, and catch-any-ORT-path-error.
    # ForkAfterOrtInit is also surfaced because it is the rare
    # decibri-specific exception that users are likely to want to catch
    # by name to distinguish "must use spawn start method" from other
    # DecibriError causes.
    "DecibriError",
    "DeviceError",
    "ForkAfterOrtInit",
    "OrtError",
    "OrtPathError",
]
