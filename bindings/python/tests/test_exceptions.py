"""Phase 2 exception hierarchy tests.

Covers all 32 exception classes shipped in the public ``decibri`` namespace:
1 base (DecibriError) + 11 direct subclasses + DeviceError intermediate
+ 8 direct DeviceError subclasses + OrtError intermediate + 7 direct
OrtError subclasses + OrtPathError intermediate + 2 direct OrtPathError
subclasses.

Verifies:
- Each class is reachable, raisable, and catchable by its own type.
- The catch-target hierarchy: ``except DeviceError`` catches all 8
  device-related instance variants; ``except OrtError`` catches all 9
  ORT-family instance variants; ``except OrtPathError`` catches the 2
  path-specific variants; ``except DecibriError`` catches everything.
- Path-bearing exceptions (OrtLoadFailed, OrtPathInvalid, VadModelLoadFailed)
  expose .path (and .reason for OrtPathInvalid) as named attributes per
  CPython OSError convention.
- Single source of truth (Commit 5 routing fix): the binding raises
  instances of the pure-Python exception classes, not Rust-side duplicates.
"""

import decibri
from decibri import (
    AlreadyRunning,
    CaptureStreamClosed,
    ChannelsOutOfRange,
    DecibriError,
    DeviceEnumerationFailed,
    DeviceError,
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
import pytest


# ---------------------------------------------------------------------------
# All 32 classes are reachable and inherit from Exception via DecibriError.
# ---------------------------------------------------------------------------


ALL_DECIBRI_ERROR_CLASSES = (
    DecibriError,
    # 11 direct DecibriError subclasses (non-device, non-ORT)
    AlreadyRunning,
    CaptureStreamClosed,
    ChannelsOutOfRange,
    FramesPerBufferOutOfRange,
    InvalidFormat,
    OutputStreamClosed,
    PermissionDenied,
    SampleRateOutOfRange,
    StreamOpenFailed,
    StreamStartFailed,
    VadSampleRateUnsupported,
    VadThresholdOutOfRange,
    # DeviceError intermediate + 8 direct subclasses (Phase 7.7 Item B7)
    DeviceError,
    DeviceEnumerationFailed,
    DeviceIndexOutOfRange,
    DeviceNotFound,
    MultipleDevicesMatch,
    NoMicrophoneFound,
    NoOutputDeviceFound,
    NotAnInputDevice,
    OutputDeviceNotFound,
    # OrtError + 7 direct + OrtPathError + 2 direct
    OrtError,
    OrtInitFailed,
    OrtSessionBuildFailed,
    OrtThreadsConfigFailed,
    VadModelLoadFailed,
    OrtInferenceFailed,
    OrtTensorCreateFailed,
    OrtTensorExtractFailed,
    OrtPathError,
    OrtLoadFailed,
    OrtPathInvalid,
)


def test_class_count() -> None:
    # Phase 7.7 Item B7 added DeviceError intermediate, taking the total
    # from 32 to 33: 1 base + 11 direct + DeviceError + 8 device + OrtError
    # + 7 ORT direct + OrtPathError + 2 path. 29 instance + 4 catch-target
    # intermediates (DecibriError, DeviceError, OrtError, OrtPathError).
    assert len(ALL_DECIBRI_ERROR_CLASSES) == 33


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


def test_ort_error_catches_all_nine_variants() -> None:
    """except OrtError catches all 9 ORT-family instance classes."""
    assert len(ORT_FAMILY_INSTANCE_CLASSES) == 9
    for cls in ORT_FAMILY_INSTANCE_CLASSES:
        assert issubclass(cls, OrtError), f"{cls.__name__} is not an OrtError subclass"


def test_ort_path_error_catches_exactly_two_variants() -> None:
    """except OrtPathError catches the 2 path-specific instance classes."""
    assert len(ORT_PATH_INSTANCE_CLASSES) == 2
    for cls in ORT_PATH_INSTANCE_CLASSES:
        assert issubclass(cls, OrtPathError), f"{cls.__name__} is not an OrtPathError subclass"


def test_ort_init_failed_is_not_path_error() -> None:
    """OrtInitFailed has no path; it is NOT under OrtPathError. Per Q1 revision."""
    assert not issubclass(OrtInitFailed, OrtPathError)
    assert issubclass(OrtInitFailed, OrtError)


# ---------------------------------------------------------------------------
# Phase 7.7 Item B7: DeviceError catch-target.
#
# DeviceError is the parent of all 8 device-related exception classes.
# Symmetric with OrtError; existing catches via DecibriError are preserved.
# ---------------------------------------------------------------------------


DEVICE_ERROR_INSTANCE_CLASSES = (
    DeviceNotFound,
    OutputDeviceNotFound,
    MultipleDevicesMatch,
    DeviceIndexOutOfRange,
    NoMicrophoneFound,
    NoOutputDeviceFound,
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
            f"after Phase 7.7 reparenting"
        )


def test_device_error_inherits_from_decibri_error() -> None:
    """DeviceError itself is a DecibriError subclass."""
    assert issubclass(DeviceError, DecibriError)


def test_device_error_is_not_ort_error() -> None:
    """DeviceError is independent of OrtError; no overlap."""
    assert not issubclass(DeviceError, OrtError)
    assert not issubclass(OrtError, DeviceError)


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


# ---------------------------------------------------------------------------
# Each class can be raised and caught by its own type.
# ---------------------------------------------------------------------------


def test_each_class_raises_and_catches_itself() -> None:
    """Construction + raise + catch round-trip works for every class.

    Path-bearing classes need their full __init__ signatures; others accept
    a single message arg per CPython BaseException.__init__.
    """
    for cls in ALL_DECIBRI_ERROR_CLASSES:
        if cls is OrtLoadFailed or cls is VadModelLoadFailed:
            instance = cls("test msg", "/test/path")
        elif cls is OrtPathInvalid:
            instance = cls("test msg", "/test/path", "test reason")
        else:
            instance = cls("test msg")
        with pytest.raises(cls):
            raise instance


# ---------------------------------------------------------------------------
# Single-source-of-truth check (Commit 5 routing fix).
#
# Before Commit 5: the Rust binding created its own exception classes via
# create_exception!, so a consumer catching decibri.exceptions.<X> would NOT
# catch instances raised by the Rust mapper. Commit 5 routed to_py_err
# through PyErr::from_type using the pure-Python classes from
# decibri.exceptions, so the Rust binding now raises pure-Python instances.
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
