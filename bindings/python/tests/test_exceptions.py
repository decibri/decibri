"""Exception hierarchy tests.

Covers every exception class shipped in the public ``decibri`` namespace:
the DecibriError base, its direct subclasses, and the DeviceError, OrtError
and OrtPathError intermediates with their subclasses. The class counts are
enforced by the assertions below (``test_class_count`` and the catch-target
tests), so this docstring states the hierarchy's shape and leaves the sizes
to them.

Verifies:
- Each class is reachable, raisable, and catchable by its own type.
- The catch-target hierarchy: ``except DeviceError`` catches every
  device-related instance variant; ``except OrtError`` catches every
  ORT-family instance variant; ``except OrtPathError`` catches the
  path-specific variants; ``except DecibriError`` catches everything.
- Path-bearing exceptions (OrtLoadFailed, OrtPathInvalid, VadModelLoadFailed,
  ModelLoadFailed) expose .path (and .reason for OrtPathInvalid) as named
  attributes per CPython OSError convention.
- Single source of truth: the binding raises
  instances of the pure-Python exception classes, not Rust-side duplicates.
"""

import re
from pathlib import Path

import decibri
from decibri import (
    AecConfigInvalid,
    AecSampleRateUnsupported,
    AgcTargetOutOfRange,
    AlreadyRunning,
    BlockSizeNotFrameAligned,
    MicrophoneStreamClosed,
    ChannelMapLengthMismatch,
    ChannelMapOutOfRange,
    ChannelsOutOfRange,
    ChannelSelectionAmbiguous,
    MicrophoneChannelsUnsupported,
    DecibriError,
    DetectorSourceOutOfRange,
    DeviceEnumerationFailed,
    DeviceError,
    DeviceFailed,
    DeviceIndexOutOfRange,
    FileConsumed,
    FileEngaged,
    FileReadFailed,
    FileWriteFailed,
    FlacCompressionOutOfRange,
    ForkAfterOrtInit,
    MicrophoneNotFound,
    FramesPerBufferOutOfRange,
    InvalidFormat,
    LimiterCeilingOutOfRange,
    ModelLoadFailed,
    MultipleDevicesMatch,
    NoMicrophoneFound,
    NoSpeakerFound,
    NotAnInputDevice,
    OnnxBackendFailed,
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
    ResampleAfterFlush,
    ResampleConfigInvalid,
    ResampleFailed,
    SpeakerChannelsUnsupported,
    SpeakerNotFound,
    SpeakerStreamClosed,
    SampleRateOutOfRange,
    StreamOpenFailed,
    StreamStartFailed,
    VadModelLoadFailed,
    VadNotConfigured,
    VadSampleRateUnsupported,
    VadThresholdOutOfRange,
    AudioFormatUnsupported,
    AudioFileMalformed,
    AudioFileTruncated,
    FileChannelsUnsupported,
    FileChannelSelectionAmbiguous,
    FileChannelMapOutOfRange,
)
import pytest


# ---------------------------------------------------------------------------
# All 64 classes are reachable and inherit from Exception via DecibriError.
# ---------------------------------------------------------------------------


ALL_DECIBRI_ERROR_CLASSES = (
    DecibriError,
    # 41 direct DecibriError subclasses (non-device, non-ORT). DeviceFailed
    # is a runtime device/driver failure (distinct from the DeviceError
    # enumeration/selection family); OnnxBackendFailed is the non-ORT ONNX
    # backend catch-all (distinct from the OrtError family); FileConsumed and
    # FileEngaged are the File single-pass lifecycle errors.
    AlreadyRunning,
    MicrophoneStreamClosed,
    ChannelMapLengthMismatch,
    ChannelMapOutOfRange,
    ChannelsOutOfRange,
    ChannelSelectionAmbiguous,
    MicrophoneChannelsUnsupported,
    BlockSizeNotFrameAligned,
    FileChannelsUnsupported,
    FileChannelSelectionAmbiguous,
    FileChannelMapOutOfRange,
    FramesPerBufferOutOfRange,
    AgcTargetOutOfRange,
    LimiterCeilingOutOfRange,
    DetectorSourceOutOfRange,
    FlacCompressionOutOfRange,
    InvalidFormat,
    SpeakerStreamClosed,
    PermissionDenied,
    SampleRateOutOfRange,
    StreamOpenFailed,
    StreamStartFailed,
    SpeakerChannelsUnsupported,
    VadSampleRateUnsupported,
    VadThresholdOutOfRange,
    AecSampleRateUnsupported,
    AecConfigInvalid,
    ResampleConfigInvalid,
    ResampleAfterFlush,
    ResampleFailed,
    DeviceFailed,
    OnnxBackendFailed,
    FileConsumed,
    FileEngaged,
    FileReadFailed,
    FileWriteFailed,
    AudioFormatUnsupported,
    AudioFileMalformed,
    AudioFileTruncated,
    VadNotConfigured,
    ForkAfterOrtInit,
    # DeviceError intermediate + 8 direct subclasses
    DeviceError,
    DeviceEnumerationFailed,
    DeviceIndexOutOfRange,
    MicrophoneNotFound,
    MultipleDevicesMatch,
    NoMicrophoneFound,
    NoSpeakerFound,
    NotAnInputDevice,
    SpeakerNotFound,
    # OrtError + 8 direct + OrtPathError + 2 direct
    OrtError,
    OrtInitFailed,
    OrtSessionBuildFailed,
    OrtThreadsConfigFailed,
    VadModelLoadFailed,
    ModelLoadFailed,
    OrtInferenceFailed,
    OrtTensorCreateFailed,
    OrtTensorExtractFailed,
    OrtPathError,
    OrtLoadFailed,
    OrtPathInvalid,
)


def test_class_count() -> None:
    # 63 total: 1 base + 41 direct + DeviceError + 8 device + OrtError
    # + 8 ORT direct + OrtPathError + 2 path.
    assert len(ALL_DECIBRI_ERROR_CLASSES) == 63


def test_all_inherit_from_decibri_error() -> None:
    for cls in ALL_DECIBRI_ERROR_CLASSES:
        assert issubclass(cls, DecibriError), f"{cls.__name__} does not inherit DecibriError"


def test_decibri_error_inherits_exception() -> None:
    assert issubclass(DecibriError, Exception)


# ---------------------------------------------------------------------------
# Hierarchy structure: catch-target intermediates.
# ---------------------------------------------------------------------------


ORT_FAMILY_INSTANCE_CLASSES = (
    OrtInitFailed,
    OrtSessionBuildFailed,
    OrtThreadsConfigFailed,
    VadModelLoadFailed,
    ModelLoadFailed,
    OrtInferenceFailed,
    OrtTensorCreateFailed,
    OrtTensorExtractFailed,
    OrtLoadFailed,
    OrtPathInvalid,
)


ORT_PATH_INSTANCE_CLASSES = (
    OrtLoadFailed,
    OrtPathInvalid,
)


def test_ort_error_catches_all_ten_variants() -> None:
    """except OrtError catches all 10 ORT-family instance classes."""
    assert len(ORT_FAMILY_INSTANCE_CLASSES) == 10
    for cls in ORT_FAMILY_INSTANCE_CLASSES:
        assert issubclass(cls, OrtError), f"{cls.__name__} is not an OrtError subclass"


def test_ort_path_error_catches_exactly_two_variants() -> None:
    """except OrtPathError catches the 2 path-specific instance classes."""
    assert len(ORT_PATH_INSTANCE_CLASSES) == 2
    for cls in ORT_PATH_INSTANCE_CLASSES:
        assert issubclass(cls, OrtPathError), f"{cls.__name__} is not an OrtPathError subclass"


def test_ort_init_failed_is_not_path_error() -> None:
    """OrtInitFailed has no path; it is NOT under OrtPathError."""
    assert not issubclass(OrtInitFailed, OrtPathError)
    assert issubclass(OrtInitFailed, OrtError)


# ---------------------------------------------------------------------------
# DeviceError catch-target.
#
# DeviceError is the parent of all 8 device-related exception classes.
# Symmetric with OrtError; existing catches via DecibriError are preserved.
# ---------------------------------------------------------------------------


DEVICE_ERROR_INSTANCE_CLASSES = (
    MicrophoneNotFound,
    SpeakerNotFound,
    MultipleDevicesMatch,
    DeviceIndexOutOfRange,
    NoMicrophoneFound,
    NoSpeakerFound,
    NotAnInputDevice,
    DeviceEnumerationFailed,
)


def test_device_error_catches_all_eight_variants() -> None:
    """except DeviceError catches all 8 device-related instance classes."""
    assert len(DEVICE_ERROR_INSTANCE_CLASSES) == 8
    for cls in DEVICE_ERROR_INSTANCE_CLASSES:
        assert issubclass(cls, DeviceError), f"{cls.__name__} is not a DeviceError subclass"


def test_device_error_subclasses_still_catchable_as_decibri_error() -> None:
    """The DeviceError reparenting preserves the DecibriError parent chain."""
    for cls in DEVICE_ERROR_INSTANCE_CLASSES:
        assert issubclass(cls, DecibriError), (
            f"{cls.__name__} should still be catchable as DecibriError "
            f"after reparenting"
        )


def test_device_error_inherits_from_decibri_error() -> None:
    """DeviceError itself is a DecibriError subclass."""
    assert issubclass(DeviceError, DecibriError)


def test_device_error_is_not_ort_error() -> None:
    """DeviceError is independent of OrtError; no overlap."""
    assert not issubclass(DeviceError, OrtError)
    assert not issubclass(OrtError, DeviceError)


# ---------------------------------------------------------------------------
# DeviceFailed and OnnxBackendFailed: dedicated, distinct error types.
#
# Both are direct DecibriError subclasses (matching how the Rust core sections
# them: DeviceFailed is a runtime "Stream error" beside StreamOpenFailed /
# StreamStartFailed, not a device-enumeration DeviceError; OnnxBackendFailed
# is the reserved non-ORT backend catch-all, not an OrtError). Before this
# change both mapped to the generic DecibriError base; they are now catchable
# as their own types.
# ---------------------------------------------------------------------------


def test_device_failed_is_dedicated_direct_subclass() -> None:
    assert issubclass(DeviceFailed, DecibriError)
    # NOT under the device-enumeration DeviceError family, nor OrtError.
    assert not issubclass(DeviceFailed, DeviceError)
    assert not issubclass(DeviceFailed, OrtError)
    # Catchable by its own type.
    with pytest.raises(DeviceFailed):
        raise DeviceFailed("device gone")


def test_onnx_backend_failed_is_dedicated_direct_subclass() -> None:
    assert issubclass(OnnxBackendFailed, DecibriError)
    # NOT under OrtError (it is the non-ORT backend catch-all), nor DeviceError.
    assert not issubclass(OnnxBackendFailed, OrtError)
    assert not issubclass(OnnxBackendFailed, DeviceError)
    with pytest.raises(OnnxBackendFailed):
        raise OnnxBackendFailed("backend boom")


def test_device_failed_is_not_a_stream_closed_subclass() -> None:
    """A device failure is reported as ``DeviceFailed``, not as a closed stream.

    Pins the catch-target consequence: code that catches
    ``MicrophoneStreamClosed`` or ``SpeakerStreamClosed`` to handle a
    disconnect no longer catches one, and has to catch ``DecibriError`` (or
    ``DeviceFailed``) instead.
    """
    assert not issubclass(DeviceFailed, MicrophoneStreamClosed)
    assert not issubclass(DeviceFailed, SpeakerStreamClosed)
    # The reverse also holds: a deliberate stop is not a device failure.
    assert not issubclass(MicrophoneStreamClosed, DeviceFailed)
    assert not issubclass(SpeakerStreamClosed, DeviceFailed)
    # Both remain reachable from the one catch-root a consumer can rely on.
    for cls in (DeviceFailed, MicrophoneStreamClosed, SpeakerStreamClosed):
        with pytest.raises(DecibriError):
            raise cls("boom")


def test_new_error_types_importable_from_top_level() -> None:
    """Both are reachable via attribute lookup on the package (not just the
    exceptions submodule), even though they are not in ``__all__``."""
    assert decibri.DeviceFailed is DeviceFailed
    assert decibri.OnnxBackendFailed is OnnxBackendFailed


# ---------------------------------------------------------------------------
# Path-bearing exceptions expose .path (and .reason for OrtPathInvalid).
# These three classes have __init__ overrides per CPython OSError convention.
# ---------------------------------------------------------------------------


def test_ort_load_failed_exposes_path() -> None:
    e = OrtLoadFailed("cannot load ORT", "/some/path/onnxruntime.dll")
    assert e.path == "/some/path/onnxruntime.dll"
    assert str(e) == "cannot load ORT"


def test_ort_path_invalid_exposes_path_and_reason() -> None:
    e = OrtPathInvalid("path is bad", "/bad/path", "not a regular file")
    assert e.path == "/bad/path"
    assert e.reason == "not a regular file"
    assert str(e) == "path is bad"


def test_vad_model_load_failed_exposes_path() -> None:
    e = VadModelLoadFailed("model load failed", "/some/silero.onnx")
    assert e.path == "/some/silero.onnx"
    assert str(e) == "model load failed"


def test_model_load_failed_exposes_path() -> None:
    e = ModelLoadFailed("model load failed", "/some/fastenhancer_t.onnx")
    assert e.path == "/some/fastenhancer_t.onnx"
    assert str(e) == "model load failed"


def test_model_load_failed_is_ort_error_not_vad() -> None:
    """ModelLoadFailed is an OrtError but distinct from VadModelLoadFailed."""
    assert issubclass(ModelLoadFailed, OrtError)
    assert issubclass(ModelLoadFailed, DecibriError)
    assert ModelLoadFailed is not VadModelLoadFailed
    assert not issubclass(ModelLoadFailed, OrtPathError)


# ---------------------------------------------------------------------------
# Each class can be raised and caught by its own type.
# ---------------------------------------------------------------------------


def test_each_class_raises_and_catches_itself() -> None:
    """Construction + raise + catch round-trip works for every class.

    Path-bearing classes need their full __init__ signatures; others accept
    a single message arg per CPython BaseException.__init__.
    """
    for cls in ALL_DECIBRI_ERROR_CLASSES:
        if cls is OrtLoadFailed or cls is VadModelLoadFailed or cls is ModelLoadFailed:
            instance = cls("test msg", "/test/path")
        elif cls is OrtPathInvalid:
            instance = cls("test msg", "/test/path", "test reason")
        else:
            instance = cls("test msg")
        with pytest.raises(cls):
            raise instance


# ---------------------------------------------------------------------------
# Single-source-of-truth check.
#
# If the Rust binding created its own exception classes via
# create_exception!, a consumer catching decibri.exceptions.<X> would NOT
# catch instances raised by the Rust mapper. Instead, to_py_err routes
# through PyErr::from_type using the pure-Python classes from
# decibri.exceptions, so the Rust binding raises pure-Python instances.
#
# Empirical proof: trigger an error from the binding (parse_sample_format
# rejects "bogus"); assert the caught exception is an instance of the
# pure-Python class re-exported from decibri.exceptions.
# ---------------------------------------------------------------------------


def test_binding_raises_pure_python_classes() -> None:
    """Triggering a binding error catches it via the pure-Python class."""
    from decibri import _decibri
    from decibri import exceptions as py_exc

    with pytest.raises(py_exc.InvalidFormat) as exc_info:
        _decibri.MicrophoneBridge(
            sample_rate=16000,
            channels=1,
            frames_per_buffer=512,
            format="bogus_format_string",
        )
    e = exc_info.value
    assert isinstance(e, py_exc.InvalidFormat)
    assert isinstance(e, py_exc.DecibriError)
    # Same class object as re-exported via decibri.<X>
    assert type(e) is decibri.InvalidFormat
    assert type(e) is py_exc.InvalidFormat


# ---------------------------------------------------------------------------
# EXCEPTION_NAMES registry parity.
#
# The Rust binding's EXCEPTION_NAMES registry (bindings/python/src/lib.rs)
# names every class the binding raises; exception_class() resolves each name
# against decibri.exceptions at first use. A misspelled or missing entry only
# fails at raise time, deep inside whichever call happens to hit it first, so
# this test parses the registry out of the Rust source and resolves every
# entry up front. Mirrors the source-parsing shape of the stub parity test in
# test_smoke.py. Runs only where the Rust source tree is present (the main
# pytest suite); the wheel install-test gate does not include this file.
# ---------------------------------------------------------------------------


_BINDING_LIB_RS = Path(__file__).resolve().parent.parent / "src" / "lib.rs"


def test_exception_names_registry_resolves_in_module() -> None:
    """Every EXCEPTION_NAMES entry names a DecibriError class in decibri.exceptions."""
    from decibri import exceptions as py_exc

    source = _BINDING_LIB_RS.read_text(encoding="utf-8")
    registry = re.search(
        r"const EXCEPTION_NAMES: &\[&str\] = &\[(.*?)\];", source, re.DOTALL
    )
    assert registry is not None, "EXCEPTION_NAMES not found in src/lib.rs"
    names = re.findall(r'"([A-Za-z]+)"', registry.group(1))
    # Parse sanity: the registry is never empty and always contains the base.
    assert names, "no entries parsed from EXCEPTION_NAMES"
    assert "DecibriError" in names, "base class missing from parsed EXCEPTION_NAMES"

    for name in names:
        cls = getattr(py_exc, name, None)
        assert cls is not None, (
            f"EXCEPTION_NAMES entry {name!r} has no matching class in decibri.exceptions"
        )
        assert isinstance(cls, type) and issubclass(cls, py_exc.DecibriError), (
            f"decibri.exceptions.{name} is not a DecibriError subclass"
        )


# ---------------------------------------------------------------------------
# Core variant parity.
#
# to_py_err picks the class by the core's DecibriError::variant_name(), so a
# core variant with no class of that name falls back to the base class and
# stops being distinguishable. The core's identity table is exhaustive with no
# catch-all arm, so a new variant fails the Rust build until it is listed
# there; these tests are what carry that guarantee across into Python. Runs
# only where the Rust source tree is present (the main pytest suite); the wheel
# install-test gate does not include this file.
# ---------------------------------------------------------------------------


_CORE_ERROR_RS = (
    Path(__file__).resolve().parents[3] / "crates" / "decibri" / "src" / "error.rs"
)


def _core_variant_names() -> list[str]:
    source = _CORE_ERROR_RS.read_text(encoding="utf-8")
    table = re.search(r"error_identity! \{(.*?)\n\}\n", source, re.DOTALL)
    assert table is not None, "error_identity! table not found in crates/decibri/src/error.rs"
    names = re.findall(r'=>\s*"([A-Za-z]+)",\s*"[A-Z0-9_]+"', table.group(1))
    # Parse sanity: the table is never empty and always carries a known variant.
    assert names, "no variants parsed from the core identity table"
    assert "PermissionDenied" in names, "known variant missing from the parsed table"
    # The count is pinned deliberately and must be updated when a core variant
    # is added or removed.
    assert len(names) == 58, (
        f"parsed {len(names)} variants from the core identity table, expected 58;"
        " the count is pinned deliberately and must be updated when a core"
        " variant is added or removed"
    )
    return names


def test_core_variants_have_exception_classes() -> None:
    """Every core DecibriError variant names a class in decibri.exceptions."""
    from decibri import exceptions as py_exc

    for name in _core_variant_names():
        cls = getattr(py_exc, name, None)
        assert cls is not None, (
            f"core variant {name!r} has no matching class in decibri.exceptions"
        )
        assert isinstance(cls, type) and issubclass(cls, py_exc.DecibriError), (
            f"decibri.exceptions.{name} is not a DecibriError subclass"
        )
        assert cls is not py_exc.DecibriError, (
            f"core variant {name!r} resolves to the DecibriError base class,"
            " not a class of its own"
        )


def test_core_variants_are_in_the_registry() -> None:
    """Every core variant is registered, so none falls back to the base class."""
    source = _BINDING_LIB_RS.read_text(encoding="utf-8")
    registry = re.search(
        r"const EXCEPTION_NAMES: &\[&str\] = &\[(.*?)\];", source, re.DOTALL
    )
    assert registry is not None, "EXCEPTION_NAMES not found in src/lib.rs"
    registered = set(re.findall(r'"([A-Za-z]+)"', registry.group(1)))

    missing = [name for name in _core_variant_names() if name not in registered]
    assert not missing, f"core variants absent from EXCEPTION_NAMES: {missing}"
