"""Type stubs for the compiled ``decibri._decibri`` Rust extension module.

Matches the Rust surface in ``bindings/python/src/lib.rs``. Kept alongside
the compiled ``.pyd`` / ``.so`` so mypy and IDEs see types without loading
the extension. Update this file when the Rust surface changes.

Internal module. The bridge classes
(``MicrophoneBridge``, ``SpeakerBridge``, ``FileBridge``) are accessible
only via ``decibri._decibri.<X>``;
they are NOT re-exported on the top-level ``decibri`` module and are NOT
part of the public 0.1.0 API. Consumers should construct the wrapper
classes (``decibri.Microphone``, ``decibri.Speaker``,
``decibri.AsyncMicrophone``, ``decibri.AsyncSpeaker``) directly. The
bridges remain importable for advanced users but carry no API stability
guarantee across versions.

Exception classes are NOT re-exported on this module. They live exclusively
at ``decibri.exceptions``; ``to_py_err`` in the Rust binding raises instances
of those pure-Python classes via ``PyErr::from_type``. Consumers should
import exceptions from ``decibri`` or directly from ``decibri.exceptions``.

Wrapper-only naming translations:
    The public Python wrappers translate three kwargs at the boundary
    while the bridge keeps the cross-binding-historical names:
        wrapper ``dtype=``       -> bridge ``format=``
        wrapper ``vad_holdoff_ms=`` -> bridge ``vad_holdoff=``
        wrapper ``as_ndarray=``  -> bridge ``numpy=``
    Direct bridge consumers (advanced use) continue to use the
    bridge-level names. The wrapper-side ``vad_score`` property is a
    mode-aware view of the bridge's ``vad_probability``; the bridge
    name is unchanged.
"""

from pathlib import Path
from typing import Any, TYPE_CHECKING, Union

from decibri._classes import Device

if TYPE_CHECKING:
    import numpy as np

    # read returns bytes (default) or ndarray (numpy=True);
    # write accepts either. The runtime numpy dependency is optional.
    SampleData = Union[bytes, "np.ndarray[Any, Any]"]
else:
    SampleData = bytes

__all__ = [
    "FileBridge",
    "MicrophoneBridge",
    "SpeakerBridge",
    "MicrophoneInfo",
    "SpeakerInfo",
    "VersionInfo",
]


class FileBridge:
    """Internal offline-source bridge. Public ``File`` wrapper lives in
    ``decibri._classes``; consumers should construct ``File``, not
    ``FileBridge``, directly.

    The bridge runs the per-chunk detector on the pre-conditioning feed
    and exposes the score via ``vad_probability``; threshold and holdoff
    policy live in the wrapper layer, measured in file time. Whole-recording
    analysis consumes the source and returns raw score and segment tuples
    the wrapper shapes into the public report types.
    """

    @staticmethod
    def open(
        path: str | Path,
        sample_rate: int = 16000,
        channels: int = 1,
        channel_map: list[int] | None = None,
        format: str = "int16",
        vad: bool = False,
        vad_threshold: float = 0.5,
        vad_mode: str = "silero",
        vad_holdoff: int = 300,
        model_path: str | Path | None = None,
        numpy: bool = False,
        ort_library_path: str | Path | None = None,
        denoise: str | None = None,
        denoise_model_path: str | Path | None = None,
        highpass: int | None = None,
        agc: int | None = None,
        limiter: float | None = None,
        dc_removal: bool = False,
        detector_source: int | None = None,
    ) -> FileBridge: ...
    @staticmethod
    def buffer(
        samples: list[float] | bytes,
        input_rate: int,
        input_channels: int = 1,
        sample_rate: int = 16000,
        channels: int = 1,
        channel_map: list[int] | None = None,
        format: str = "int16",
        vad: bool = False,
        vad_threshold: float = 0.5,
        vad_mode: str = "silero",
        vad_holdoff: int = 300,
        model_path: str | Path | None = None,
        numpy: bool = False,
        ort_library_path: str | Path | None = None,
        denoise: str | None = None,
        denoise_model_path: str | Path | None = None,
        highpass: int | None = None,
        agc: int | None = None,
        limiter: float | None = None,
        dc_removal: bool = False,
        detector_source: int | None = None,
    ) -> FileBridge: ...
    def read(self) -> SampleData | None: ...
    def check_not_engaged(self) -> None: ...
    def analyze(
        self,
    ) -> tuple[
        list[tuple[float, float, float, bool]],
        list[tuple[float, float]],
    ]: ...
    def save(
        self,
        path: str | Path,
        format: str | None = None,
        compression: int | None = None,
    ) -> tuple[int, int]: ...
    def close(self) -> None: ...
    @property
    def vad_probability(self) -> float: ...
    @property
    def sample_rate(self) -> int: ...
    @property
    def input_rate(self) -> int: ...
    @property
    def channels(self) -> int: ...


class VersionInfo:
    """Version information for the decibri core and its audio runtime.

    The ``decibri`` field is the Rust core version (``CARGO_PKG_VERSION``
    at build time), not the Python package version. The ``binding`` field
    is the Python package version. The ``audio_backend`` field is the cpal
    crate version, prefixed with ``"cpal "`` to identify the backend.
    """

    @property
    def decibri(self) -> str: ...
    @property
    def audio_backend(self) -> str: ...
    @property
    def binding(self) -> str: ...
    def __repr__(self) -> str: ...


class MicrophoneInfo:
    """Audio input device metadata. Returned by ``MicrophoneBridge.devices()``."""

    @property
    def index(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def id(self) -> str: ...
    @property
    def max_input_channels(self) -> int: ...
    @property
    def default_sample_rate(self) -> int: ...
    @property
    def is_default(self) -> bool: ...
    def __repr__(self) -> str: ...


class SpeakerInfo:
    """Audio output device metadata. Returned by ``SpeakerBridge.devices()``."""

    @property
    def index(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def id(self) -> str: ...
    @property
    def max_output_channels(self) -> int: ...
    @property
    def default_sample_rate(self) -> int: ...
    @property
    def is_default(self) -> bool: ...
    def __repr__(self) -> str: ...


class MicrophoneBridge:
    """Internal capture bridge. Public ``Microphone`` wrapper lives in
    ``decibri._classes``; consumers should construct ``Microphone``, not
    ``MicrophoneBridge``, directly.

    The bridge runs the detector selected by ``vad_mode`` on the
    pre-enhancement signal (Silero inference for ``"silero"``, an energy
    RMS for ``"energy"``) and exposes the score via ``vad_probability``.
    Threshold and holdoff policy live in the wrapper layer; the bridge
    stores ``vad_holdoff_ms`` as inert state.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        frames_per_buffer: int,
        format: str,
        device: int | str | Device | None = None,
        vad: bool = False,
        vad_threshold: float = 0.5,
        vad_mode: str = "silero",
        vad_holdoff: int = 0,
        model_path: str | Path | None = None,
        numpy: bool = False,
        ort_library_path: str | Path | None = None,
        denoise: str | None = None,
        denoise_model_path: str | Path | None = None,
        highpass: int | None = None,
        agc: int | None = None,
        limiter: float | None = None,
        dc_removal: bool = False,
        aec: str | None = None,
        aec_tail_ms: int | None = None,
        aec_suppression: str | None = None,
        aec_reference_sample_rate: int | None = None,
        aec_reference_channels: int | None = None,
        channel_map: list[int] | None = None,
        detector_source: int | None = None,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def read(self, timeout_ms: int | None = None) -> SampleData | None: ...
    def push_aec_reference(self, samples: SampleData) -> None: ...
    def aec_metrics(
        self,
    ) -> (
        tuple[
            int | None,
            float,
            bool,
            int,
            int,
            int,
            int,
            int,
            list[tuple[int | None, float, bool, int, int, int]],
        ]
        | None
    ): ...
    def __iter__(self) -> MicrophoneBridge: ...
    def __next__(self) -> bytes: ...
    def __enter__(self) -> MicrophoneBridge: ...
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool: ...
    @property
    def is_open(self) -> bool: ...
    @property
    def vad_probability(self) -> float: ...
    @property
    def vad_holdoff_ms(self) -> int: ...
    @property
    def overrun_count(self) -> int: ...
    @staticmethod
    def devices() -> list[MicrophoneInfo]: ...
    @staticmethod
    def version() -> VersionInfo: ...


class SpeakerBridge:
    """Internal output bridge. Public ``Speaker`` wrapper lives in
    ``decibri._classes``; consumers should construct ``Speaker``,
    not ``SpeakerBridge``, directly.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        format: str,
        device: int | str | Device | None = None,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def close(self) -> None: ...
    def write(self, samples: SampleData) -> None: ...
    def drain(self) -> None: ...
    def __enter__(self) -> SpeakerBridge: ...
    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> bool: ...
    @property
    def is_playing(self) -> bool: ...
    @property
    def underrun_count(self) -> int: ...
    @staticmethod
    def devices() -> list[SpeakerInfo]: ...
