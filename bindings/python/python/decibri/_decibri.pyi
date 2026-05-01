"""Type stubs for the compiled ``decibri._decibri`` Rust extension module.

Matches the Rust surface in ``bindings/python/src/lib.rs``. Kept alongside
the compiled ``.pyd`` / ``.so`` so mypy and IDEs see types without loading
the extension. Update this file when the Rust surface changes.

Exception classes are NOT re-exported on this module. They live exclusively
at ``decibri.exceptions``; ``to_py_err`` in the Rust binding raises instances
of those pure-Python classes via ``PyErr::from_type``. Consumers should
import exceptions from ``decibri`` (after Commit 6 re-export) or directly
from ``decibri.exceptions``.
"""

from pathlib import Path
from typing import Awaitable, Coroutine, Any, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import numpy as np

    # Phase 6: read returns bytes (default) or ndarray (numpy=True);
    # write accepts either. The runtime numpy dependency is optional.
    SampleData = Union[bytes, "np.ndarray[Any, Any]"]
else:
    SampleData = bytes

__all__ = [
    "AsyncMicrophoneBridge",
    "AsyncSpeakerBridge",
    "MicrophoneBridge",
    "SpeakerBridge",
    "DeviceInfo",
    "OutputDeviceInfo",
    "VersionInfo",
]


class VersionInfo:
    """Version information for the decibri core and its audio runtime.

    The ``decibri`` field is the Rust core version (``CARGO_PKG_VERSION``
    at build time), not the Python wheel version. The ``binding`` field
    is the Python wheel version. The ``portaudio`` field is the cpal
    crate version, prefixed with ``"cpal "`` for compatibility with the
    Node binding's identical field.
    """

    @property
    def decibri(self) -> str: ...
    @property
    def portaudio(self) -> str: ...
    @property
    def binding(self) -> str: ...
    def __repr__(self) -> str: ...


class DeviceInfo:
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


class OutputDeviceInfo:
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

    The bridge is a thin inference primitive. VAD policy (threshold
    application, holdoff state machine, mode dispatch) lives in the
    wrapper layer; the bridge exposes the raw Silero probability via
    ``vad_probability`` and stores ``vad_holdoff_ms`` as inert state.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        frames_per_buffer: int,
        format: str,
        device: int | str | None = None,
        vad: bool = False,
        vad_threshold: float = 0.5,
        vad_mode: str = "silero",
        vad_holdoff: int = 0,
        model_path: str | Path | None = None,
        numpy: bool = False,
        ort_library_path: str | Path | None = None,
    ) -> None: ...
    def start(self) -> None: ...
    def stop(self) -> None: ...
    def read(self, timeout_ms: int | None = None) -> SampleData | None: ...
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
    @staticmethod
    def devices() -> list[DeviceInfo]: ...
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
        device: int | str | None = None,
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
    @staticmethod
    def devices() -> list[OutputDeviceInfo]: ...


class AsyncMicrophoneBridge:
    """Async wrapper around ``MicrophoneBridge`` (Phase 5).

    Internal pyclass. Public ``AsyncMicrophone`` wrapper lives in
    ``decibri._async_classes``; consumers should construct
    ``AsyncMicrophone``, not ``AsyncMicrophoneBridge``, directly. Each method
    that maps to a blocking sync method dispatches via
    ``tokio::task::spawn_blocking`` and returns a Python awaitable.

    Constructor signature matches ``MicrophoneBridge`` exactly. Construction
    itself is sync (Python class instantiation is always sync) and may
    block on ORT model load when ``vad=True and vad_mode='silero'``.

    The non-blocking getters (``is_open``, ``vad_probability``,
    ``vad_holdoff_ms``) are exposed as awaitables here because the Rust
    binding implements them via ``future_into_py`` for uniformity. The
    public ``AsyncMicrophone`` wrapper provides synchronous-property access
    backed by Python-side state tracking; consumers should prefer that
    surface.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        frames_per_buffer: int,
        format: str,
        device: int | str | None = None,
        vad: bool = False,
        vad_threshold: float = 0.5,
        vad_mode: str = "silero",
        vad_holdoff: int = 0,
        model_path: str | Path | None = None,
        numpy: bool = False,
        ort_library_path: str | Path | None = None,
    ) -> None: ...
    def start(self) -> Coroutine[Any, Any, None]: ...
    def stop(self) -> Coroutine[Any, Any, None]: ...
    def read(
        self, timeout_ms: int | None = None
    ) -> Coroutine[Any, Any, SampleData | None]: ...
    @property
    def is_open(self) -> Awaitable[bool]: ...
    @property
    def vad_probability(self) -> Awaitable[float]: ...
    @property
    def vad_holdoff_ms(self) -> Awaitable[int]: ...
    @staticmethod
    def devices() -> Coroutine[Any, Any, list[DeviceInfo]]: ...
    @staticmethod
    def version() -> Coroutine[Any, Any, VersionInfo]: ...


class AsyncSpeakerBridge:
    """Async wrapper around ``SpeakerBridge`` (Phase 5).

    Internal pyclass. Public ``AsyncSpeaker`` wrapper lives in
    ``decibri._async_classes``; consumers should construct
    ``AsyncSpeaker``, not ``AsyncSpeakerBridge``, directly.
    """

    def __init__(
        self,
        sample_rate: int,
        channels: int,
        format: str,
        device: int | str | None = None,
    ) -> None: ...
    def start(self) -> Coroutine[Any, Any, None]: ...
    def stop(self) -> Coroutine[Any, Any, None]: ...
    def close(self) -> Coroutine[Any, Any, None]: ...
    def write(self, samples: SampleData) -> Coroutine[Any, Any, None]: ...
    def drain(self) -> Coroutine[Any, Any, None]: ...
    @property
    def is_playing(self) -> Awaitable[bool]: ...
    @staticmethod
    def devices() -> Coroutine[Any, Any, list[OutputDeviceInfo]]: ...
