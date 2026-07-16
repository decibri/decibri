"""Type stubs for the decibri Python package."""

from decibri._async_classes import AsyncFile as AsyncFile
from decibri._async_classes import AsyncMicrophone as AsyncMicrophone
from decibri._async_classes import AsyncSpeaker as AsyncSpeaker
from decibri._classes import Chunk as Chunk
from decibri._classes import File as File
from decibri._classes import Microphone as Microphone
from decibri._classes import Segment as Segment
from decibri._classes import Speaker as Speaker
from decibri._classes import VadReport as VadReport
from decibri._classes import VadWindow as VadWindow
from decibri._decibri import MicrophoneInfo as MicrophoneInfo
from decibri._decibri import SpeakerInfo as SpeakerInfo
from decibri._decibri import VersionInfo as VersionInfo
from decibri.exceptions import AgcTargetOutOfRange as AgcTargetOutOfRange
from decibri.exceptions import AlreadyRunning as AlreadyRunning
from decibri.exceptions import MicrophoneStreamClosed as MicrophoneStreamClosed
from decibri.exceptions import ChannelsOutOfRange as ChannelsOutOfRange
from decibri.exceptions import MultichannelNotSupported as MultichannelNotSupported
from decibri.exceptions import DecibriError as DecibriError
from decibri.exceptions import DeviceEnumerationFailed as DeviceEnumerationFailed
from decibri.exceptions import DeviceError as DeviceError
from decibri.exceptions import DeviceIndexOutOfRange as DeviceIndexOutOfRange
from decibri.exceptions import MicrophoneNotFound as MicrophoneNotFound
from decibri.exceptions import ModelLoadFailed as ModelLoadFailed
from decibri.exceptions import ForkAfterOrtInit as ForkAfterOrtInit
from decibri.exceptions import FramesPerBufferOutOfRange as FramesPerBufferOutOfRange
from decibri.exceptions import InvalidFormat as InvalidFormat
from decibri.exceptions import LimiterCeilingOutOfRange as LimiterCeilingOutOfRange
from decibri.exceptions import MultipleDevicesMatch as MultipleDevicesMatch
from decibri.exceptions import NoMicrophoneFound as NoMicrophoneFound
from decibri.exceptions import NoSpeakerFound as NoSpeakerFound
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
from decibri.exceptions import SpeakerNotFound as SpeakerNotFound
from decibri.exceptions import SpeakerStreamClosed as SpeakerStreamClosed
from decibri.exceptions import PermissionDenied as PermissionDenied
from decibri.exceptions import SampleRateOutOfRange as SampleRateOutOfRange
from decibri.exceptions import StreamOpenFailed as StreamOpenFailed
from decibri.exceptions import StreamStartFailed as StreamStartFailed
from decibri.exceptions import FileReadFailed as FileReadFailed
from decibri.exceptions import VadModelLoadFailed as VadModelLoadFailed
from decibri.exceptions import VadNotConfigured as VadNotConfigured
from decibri.exceptions import VadSampleRateUnsupported as VadSampleRateUnsupported
from decibri.exceptions import VadThresholdOutOfRange as VadThresholdOutOfRange
from decibri.exceptions import WavInvalid as WavInvalid

__version__: str
__all__: list[str]

def input_devices() -> list[MicrophoneInfo]: ...
def output_devices() -> list[SpeakerInfo]: ...
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
