"""decibri exception hierarchy.

This module is the public home for the decibri exception hierarchy. All
classes are also re-exported at ``decibri.<X>`` for convenience; users
may import from either path.

29 instance classes (one per Rust DecibriError variant) plus 3 intermediate
parent classes (DeviceError, OrtError, OrtPathError) for catch ergonomics,
totaling 32 class definitions. Single-inheritance hierarchy per CPython
convention.

Hierarchy:
    DecibriError
    + 11 direct subclasses (config + runtime errors that don't involve
      device enumeration or ORT)
    + DeviceError (intermediate; no instances; catches device-related)
        + 8 direct device subclasses (DeviceNotFound, OutputDeviceNotFound,
          MultipleDevicesMatch, DeviceIndexOutOfRange, NoMicrophoneFound,
          NoOutputDeviceFound, NotAnInputDevice, DeviceEnumerationFailed)
    + OrtError (intermediate; no instances; catches all ORT-related)
        + 7 direct ORT subclasses (init, session, threads, inference, tensors)
        + OrtPathError (intermediate; no instances; catches path-specific)
            + OrtLoadFailed (has path field; ORT failed to load)
            + OrtPathInvalid (has path field; pre-check rejected)

Design rationale: see Q1 (revised 2026-04-26) in the Phase 2 design
decisions reference doc. Phase 7.7 Item B7 added DeviceError for symmetry
with OrtError and OrtPathError. Intermediate parents (DeviceError,
OrtError, OrtPathError) are catch-targets only; they have no instances
themselves. The Rust core's DecibriError variants map directly to the 29
instance classes via the to_py_err mapper in bindings/python/src/lib.rs.
"""


class DecibriError(Exception):
    """Base class for all decibri-raised exceptions.

    All exceptions raised by the decibri Python binding inherit from this
    class. Catch DecibriError to handle any decibri error generically;
    catch specific subclasses for fine-grained handling.
    """


# Direct DecibriError subclasses that are not device-related and not ORT-related


class SampleRateOutOfRange(DecibriError):
    """Raised when sample_rate is outside the supported range."""


class ChannelsOutOfRange(DecibriError):
    """Raised when channels is outside the supported range."""


class FramesPerBufferOutOfRange(DecibriError):
    """Raised when frames_per_buffer is outside the supported range."""


class InvalidFormat(DecibriError):
    """Raised when format string is not recognized."""


# DeviceError intermediate parent (Phase 7.7 Item B7)


class DeviceError(DecibriError):
    """Base class for all device-enumeration and selection errors.

    Catches: DeviceNotFound, OutputDeviceNotFound, MultipleDevicesMatch,
    DeviceIndexOutOfRange, NoMicrophoneFound, NoOutputDeviceFound,
    NotAnInputDevice, DeviceEnumerationFailed. 8 instance classes total.

    Use this when you want to catch any device-selection failure with a
    single except clause: bad device name, ambiguous match, missing
    hardware, host-API enumeration error. All eight remain catchable as
    DecibriError (the parent chain is preserved).

    This class has no instances of its own; it is a catch-target only.
    """


# Direct DeviceError subclasses (8 instance classes)


class DeviceNotFound(DeviceError):
    """Raised when the named input device cannot be found on the system."""


class OutputDeviceNotFound(DeviceError):
    """Raised when the named output device cannot be found on the system."""


class MultipleDevicesMatch(DeviceError):
    """Raised when device name is ambiguous; suggests using device ID."""


class DeviceIndexOutOfRange(DeviceError):
    """Raised when device index is out of range for the host."""


class NoMicrophoneFound(DeviceError):
    """Raised when no input device is available on the system."""


class NoOutputDeviceFound(DeviceError):
    """Raised when no output device is available on the system."""


class NotAnInputDevice(DeviceError):
    """Raised when a device matched but is not an input device."""


class DeviceEnumerationFailed(DeviceError):
    """Raised when the underlying audio system fails to enumerate devices."""


class AlreadyRunning(DecibriError):
    """Raised when start() is called on an instance already running."""


class StreamOpenFailed(DecibriError):
    """Raised when the audio stream fails to open."""


class StreamStartFailed(DecibriError):
    """Raised when the audio stream fails to start after opening."""


class PermissionDenied(DecibriError):
    """Raised when the OS denies microphone access.

    Platform-specific guidance is included in the error message; consult
    the message text for actionable next steps.
    """


class CaptureStreamClosed(DecibriError):
    """Raised when reading from a closed capture stream."""


class OutputStreamClosed(DecibriError):
    """Raised when writing to a closed output stream."""


class VadSampleRateUnsupported(DecibriError):
    """Raised when VAD is enabled with an unsupported sample rate."""


class VadThresholdOutOfRange(DecibriError):
    """Raised when vad_threshold is not in the [0.0, 1.0] range."""


# OrtError intermediate parent


class OrtError(DecibriError):
    """Base class for all ORT-related errors.

    Catches: OrtInitFailed, OrtSessionBuildFailed, OrtThreadsConfigFailed,
    VadModelLoadFailed, OrtInferenceFailed, OrtTensorCreateFailed,
    OrtTensorExtractFailed, OrtLoadFailed, OrtPathInvalid (the last two
    via OrtPathError intermediate). 9 instance classes total.

    Use this when you want to catch any ORT setup or runtime failure with
    a single except clause. For path-specific failures only (where the
    error has a `path` attribute), catch OrtPathError instead.

    This class has no instances of its own; it is a catch-target only.
    """


# Direct OrtError subclasses (7 instance classes)


class OrtInitFailed(OrtError):
    """Raised when ORT initialization itself fails (no path was supplied).

    Fires when init_ort_once(None) is called and ort::init().commit()
    fails. Has no `path` attribute (no path was supplied to fail on).

    The error message provides remediation guidance: pass ort_library_path,
    set ORT_DYLIB_PATH environment variable, or enable the
    ort-download-binaries feature for zero-config builds.
    """


class OrtSessionBuildFailed(OrtError):
    """Raised when building an ORT inference session fails."""


class OrtThreadsConfigFailed(OrtError):
    """Raised when configuring ORT thread pools fails."""


class VadModelLoadFailed(OrtError):
    """Raised when loading the Silero VAD ONNX model fails."""

    def __init__(self, msg: str, path: str) -> None:
        super().__init__(msg)
        self.path = path


class OrtInferenceFailed(OrtError):
    """Raised when ORT inference produces an error."""


class OrtTensorCreateFailed(OrtError):
    """Raised when creating an ORT input tensor fails."""


class OrtTensorExtractFailed(OrtError):
    """Raised when extracting values from an ORT output tensor fails."""


# OrtPathError intermediate parent


class OrtPathError(OrtError):
    """Base class for ORT errors involving a specific library path.

    Catches: OrtLoadFailed, OrtPathInvalid. 2 instance classes total.
    Both have a `path` attribute. Does not catch OrtInitFailed (which
    has no path; fires when no path was supplied to ORT).

    Use this when you want path-retry logic: if the user-supplied path
    fails, catch OrtPathError and try a different path. The `path`
    attribute on the caught exception identifies which path failed.

    This class has no instances of its own; it is a catch-target only.
    """


# OrtPathError subclasses (2 instance classes)


class OrtLoadFailed(OrtPathError):
    """Raised when ORT tried to load a specific library path and failed.

    The path passed decibri's filesystem pre-check, but
    ort::init_from(path) rejected it (e.g. wrong ORT version, corrupted
    dylib, ABI mismatch).

    The `path` attribute on the exception identifies which path failed.
    """

    def __init__(self, msg: str, path: str) -> None:
        super().__init__(msg)
        self.path = path


class OrtPathInvalid(OrtPathError):
    """Raised when a library path failed decibri's pre-check before ORT saw it.

    The pre-check exists to prevent a hang on Windows when ORT is asked
    to load an invalid path under ort-load-dynamic. Fires for nonexistent
    files, directory paths, etc. Does not carry an underlying ort::Error
    source (constructing one would call into the ORT C API that the
    pre-check is designed to avoid).

    The `path` attribute on the exception identifies which path failed.
    """

    def __init__(self, msg: str, path: str, reason: str) -> None:
        super().__init__(msg)
        self.path = path
        self.reason = reason


# Phase 9 Item C7: fork-safety detection (Linux). Direct DecibriError
# subclass, NOT under OrtError, because this is a usage error (user
# initialized ORT in parent then forked) rather than an ORT-internal
# error. See LD-9-9. Catch via `except DecibriError` continues to work.


class ForkAfterOrtInit(DecibriError):
    """Raised when ONNX Runtime is used in a child process after being
    initialized in the parent.

    Python's default ``fork`` start method on Linux duplicates the
    parent's memory into the child, but ONNX Runtime's internal state is
    not safe to share across forked processes. Using a SileroVad-enabled
    Microphone in a forked child produces silent wrong answers or
    segfaults; decibri detects the pid mismatch at the start of every
    Silero inference call and raises this exception instead.

    Remediation:
        - Set ``multiprocessing.set_start_method('spawn')`` before
          spawning workers, OR
        - Construct ``Microphone(vad='silero')`` inside each child
          process, never in the parent before fork.

    See ``docs/ecosystem/multiprocessing.md`` for verified examples.
    """
