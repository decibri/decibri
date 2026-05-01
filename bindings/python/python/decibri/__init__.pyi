"""Type stubs for the decibri Python package."""

from pathlib import Path

from decibri._async_classes import AsyncMicrophone as AsyncMicrophone
from decibri._async_classes import AsyncSpeaker as AsyncSpeaker
from decibri._classes import Microphone as Microphone
from decibri._classes import Speaker as Speaker
from decibri._decibri import DeviceInfo as DeviceInfo
from decibri._decibri import MicrophoneBridge as MicrophoneBridge
from decibri._decibri import OutputDeviceInfo as OutputDeviceInfo
from decibri._decibri import SpeakerBridge as SpeakerBridge
from decibri._decibri import VersionInfo as VersionInfo
from decibri.exceptions import AlreadyRunning as AlreadyRunning
from decibri.exceptions import CaptureStreamClosed as CaptureStreamClosed
from decibri.exceptions import ChannelsOutOfRange as ChannelsOutOfRange
from decibri.exceptions import DecibriError as DecibriError
from decibri.exceptions import DeviceEnumerationFailed as DeviceEnumerationFailed
from decibri.exceptions import DeviceIndexOutOfRange as DeviceIndexOutOfRange
from decibri.exceptions import DeviceNotFound as DeviceNotFound
from decibri.exceptions import FramesPerBufferOutOfRange as FramesPerBufferOutOfRange
from decibri.exceptions import InvalidFormat as InvalidFormat
from decibri.exceptions import MultipleDevicesMatch as MultipleDevicesMatch
from decibri.exceptions import NoMicrophoneFound as NoMicrophoneFound
from decibri.exceptions import NoOutputDeviceFound as NoOutputDeviceFound
from decibri.exceptions import NotAnInputDevice as NotAnInputDevice
from decibri.exceptions import OrtError as OrtError
from decibri.exceptions import OrtInferenceFailed as OrtInferenceFailed
from decibri.exceptions import OrtInitFailed as OrtInitFailed
from decibri.exceptions import OrtLoadFailed as OrtLoadFailed
from decibri.exceptions import OrtPathError as OrtPathError
from decibri.exceptions import OrtPathInvalid as OrtPathInvalid
from decibri.exceptions import OrtSessionBuildFailed as OrtSessionBuildFailed
from decibri.exceptions import OrtTensorCreateFailed as OrtTensorCreateFailed
from decibri.exceptions import OrtTensorExtractFailed as OrtTensorExtractFailed
from decibri.exceptions import OrtThreadsConfigFailed as OrtThreadsConfigFailed
from decibri.exceptions import OutputDeviceNotFound as OutputDeviceNotFound
from decibri.exceptions import OutputStreamClosed as OutputStreamClosed
from decibri.exceptions import PermissionDenied as PermissionDenied
from decibri.exceptions import SampleRateOutOfRange as SampleRateOutOfRange
from decibri.exceptions import StreamOpenFailed as StreamOpenFailed
from decibri.exceptions import StreamStartFailed as StreamStartFailed
from decibri.exceptions import VadModelLoadFailed as VadModelLoadFailed
from decibri.exceptions import VadSampleRateUnsupported as VadSampleRateUnsupported
from decibri.exceptions import VadThresholdOutOfRange as VadThresholdOutOfRange

__version__: str
__all__: list[str]

def microphone(
    sample_rate: int = 16000,
    channels: int = 1,
    frames_per_buffer: int = 1600,
    dtype: str = "int16",
    device: int | str | None = None,
    vad: bool | str = False,
    vad_threshold: float | None = None,
    vad_holdoff_ms: int = 300,
    model_path: str | Path | None = None,
    numpy: bool = False,
    ort_library_path: str | Path | None = None,
) -> Microphone: ...
def speaker(
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
    device: int | str | None = None,
) -> Speaker: ...
def async_microphone(
    sample_rate: int = 16000,
    channels: int = 1,
    frames_per_buffer: int = 1600,
    dtype: str = "int16",
    device: int | str | None = None,
    vad: bool | str = False,
    vad_threshold: float | None = None,
    vad_holdoff_ms: int = 300,
    model_path: str | Path | None = None,
    numpy: bool = False,
    ort_library_path: str | Path | None = None,
) -> AsyncMicrophone: ...
def async_speaker(
    sample_rate: int = 16000,
    channels: int = 1,
    dtype: str = "int16",
    device: int | str | None = None,
) -> AsyncSpeaker: ...
def devices() -> list[DeviceInfo]: ...
def output_devices() -> list[OutputDeviceInfo]: ...
def version() -> VersionInfo: ...
def record_to_file(
    path: str,
    duration_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
    device: int | str | None = None,
) -> None: ...
async def async_record_to_file(
    path: str,
    duration_seconds: float,
    sample_rate: int = 16000,
    channels: int = 1,
    device: int | str | None = None,
) -> None: ...
