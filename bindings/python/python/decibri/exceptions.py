"""decibri exception hierarchy.

This module is the public home for the decibri exception hierarchy. All
classes are also re-exported at ``decibri.<X>`` for convenience; users
may import from either path.

decibri raises a DecibriError subclass whenever the core has a name for the
failure, and a Python built-in when the failure is a wrapper-level argument
shape the core never sees. An out-of-range agc target raises
AgcTargetOutOfRange; a malformed vad, denoise or highpass value raises
ValueError.

60 instance classes plus 3 intermediate parent classes (DeviceError,
OrtError, OrtPathError) for catch ergonomics, totaling 63 class
definitions beneath DecibriError. Single-inheritance hierarchy per
CPython convention.

Hierarchy:
    DecibriError
    + 42 direct subclasses (config + runtime errors that don't involve
      device enumeration or ORT, including DeviceFailed and OnnxBackendFailed)
    + DeviceError (intermediate; no instances; catches device-related)
        + 8 direct device subclasses (MicrophoneNotFound, SpeakerNotFound,
          MultipleDevicesMatch, DeviceIndexOutOfRange, NoMicrophoneFound,
          NoSpeakerFound, NotAnInputDevice, DeviceEnumerationFailed)
    + OrtError (intermediate; no instances; catches all ORT-related)
        + 8 direct ORT subclasses (init, session, threads, models,
          inference, tensors)
        + OrtPathError (intermediate; no instances; catches path-specific)
            + OrtLoadFailed (has path field; ORT failed to load)
            + OrtPathInvalid (has path field; pre-check rejected)

DeviceError exists for catch symmetry with OrtError and OrtPathError.
Intermediate parents (DeviceError, OrtError, OrtPathError) are
catch-targets only; they have no instances themselves. The Rust core's
DecibriError variants map directly to the instance classes via the
to_py_err mapper in bindings/python/src/lib.rs.
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


class AgcTargetOutOfRange(DecibriError):
    """Raised when the agc target level is outside the supported dBFS range."""


class LimiterCeilingOutOfRange(DecibriError):
    """Raised when the limiter ceiling is outside the supported dBFS range."""


class DetectorSourceOutOfRange(DecibriError):
    """Raised when the detector source names a delivered channel at or above
    the configured delivered channel count."""


class FlacCompressionOutOfRange(DecibriError):
    """Raised when the compression level for a FLAC save is outside 0 to 8."""


class InvalidFormat(DecibriError):
    """Raised when format string is not recognized."""


# DeviceError intermediate parent


class DeviceError(DecibriError):
    """Base class for all device-enumeration and selection errors.

    Catches: MicrophoneNotFound, SpeakerNotFound, MultipleDevicesMatch,
    DeviceIndexOutOfRange, NoMicrophoneFound, NoSpeakerFound,
    NotAnInputDevice, DeviceEnumerationFailed. 8 instance classes total.

    Use this when you want to catch any device-selection failure with a
    single except clause: bad device name, ambiguous match, missing
    hardware, host-API enumeration error. All eight remain catchable as
    DecibriError (the parent chain is preserved).

    This class has no instances of its own; it is a catch-target only.
    """


# Direct DeviceError subclasses (8 instance classes)


class MicrophoneNotFound(DeviceError):
    """Raised when the named microphone cannot be found on the system."""


class SpeakerNotFound(DeviceError):
    """Raised when the named speaker cannot be found on the system."""


class MultipleDevicesMatch(DeviceError):
    """Raised when device name is ambiguous; suggests using device ID."""


class DeviceIndexOutOfRange(DeviceError):
    """Raised when device index is out of range for the host."""


class NoMicrophoneFound(DeviceError):
    """Raised when no input device is available on the system."""


class NoSpeakerFound(DeviceError):
    """Raised when no speaker is available on the system."""


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


class SpeakerChannelsUnsupported(DecibriError):
    """Raised when an output device cannot serve the requested channel count.

    ``Speaker`` bounds ``channels`` below only: how many output channels can be
    carried is the device's answer, so the count is offered to the device when
    the stream opens. A device that refuses a count above the figure it reports
    raises this; the message names the count asked for, the figure the device
    reports (the same one ``SpeakerInfo.max_output_channels`` carries) and the
    platform's own text. An open that fails for any other reason stays
    ``StreamOpenFailed``.
    """


class ChannelMapOutOfRange(DecibriError):
    """Raised when a capture channel map names a device channel that does not exist.

    ``channel_map`` entries are 0-based device channel indices, checked
    against the resolved device's own channel count when the stream starts
    (only the device can say how many channels it has; no fixed maximum
    exists). The message names the offending entry and the count the device
    reports.
    """


class ChannelMapLengthMismatch(DecibriError):
    """Raised when a capture channel map's length differs from ``channels``.

    ``channel_map`` carries one entry per delivered channel, so its length
    must equal the ``channels`` count. The message names both figures.
    """


class MicrophoneChannelsUnsupported(DecibriError):
    """Raised when a capture ``channels`` count exceeds the device's own.

    decibri delivers device channels, so it cannot manufacture one the
    device does not have. Checked against the resolved device at
    ``start()``; the message names the count asked for and the count the
    device reports. A ``channel_map`` lifts this, because a map may repeat
    a channel: ``channels=4`` with ``channel_map=[0, 0, 0, 0]`` on a mono
    device is accepted.
    """


class ChannelSelectionAmbiguous(DecibriError):
    """Raised when a capture asks for a strict subset of the device's channels.

    ``channels=1`` delivers the average of every device channel and a
    count equal to the device's own delivers them all, in device order.
    A count between the two does not say which channels it means, so
    ``channel_map`` names them rather than decibri choosing. Checked
    against the resolved device at ``start()``; the message names the
    count asked for and the count the device reports.
    """


class BlockSizeNotFrameAligned(DecibriError):
    """Raised when a read asks for a block that is not a whole number of frames.

    Block sizes count interleaved samples, so above one channel the size
    must be a multiple of ``channels``. A size that is not would cut a
    frame at the chunk boundary and rotate the channels of every chunk
    after it, which nothing in the delivered audio would reveal. Cannot
    be raised by a mono capture.
    """


class FileChannelsUnsupported(DecibriError):
    """Raised when a file-source ``channels`` count exceeds the source's own.

    decibri delivers source channels, so it cannot manufacture one the
    source does not have. Checked at construction, where the source's
    count is first known (the container's header, or ``input_channels``
    for a buffer source); the message names the count asked for and the
    count the source has. The offline counterpart of
    ``MicrophoneChannelsUnsupported``. A ``channel_map`` lifts this,
    because a map may repeat a channel.
    """


class FileChannelSelectionAmbiguous(DecibriError):
    """Raised when a file source asks for a strict subset of its channels.

    ``channels=1`` delivers the average of every source channel and a
    count equal to the source's own delivers them all, in source order.
    A count between the two does not say which channels it means, so
    ``channel_map`` names them rather than decibri choosing. Checked at
    construction, where the source's count is first known; the message
    names both counts. The offline counterpart of
    ``ChannelSelectionAmbiguous``.
    """


class FileChannelMapOutOfRange(DecibriError):
    """Raised when a file-source channel map names a channel the source lacks.

    Every ``channel_map`` entry names a 0-based source channel, so each
    must be below the source's own count. Checked at construction, where
    the source's count is first known; the message names the offending
    entry and the count the source has. The offline counterpart of
    ``ChannelMapOutOfRange``.
    """


class PermissionDenied(DecibriError):
    """Raised when the OS denies microphone access.

    Platform-specific guidance is included in the error message; consult
    the message text for actionable next steps.
    """


class MicrophoneStreamClosed(DecibriError):
    """Raised when reading from a closed microphone stream."""


class SpeakerStreamClosed(DecibriError):
    """Raised when writing to a closed speaker stream."""


class DeviceFailed(DecibriError):
    """Raised when an active stream fails at the device or driver level.

    Fires when a running microphone or speaker stream is interrupted by a
    device or driver fault (device unplugged, driver reset, exclusive-mode
    preemption), as opposed to the device-enumeration and selection failures
    grouped under DeviceError. A direct DecibriError subclass, alongside the
    other runtime stream errors (StreamOpenFailed, StreamStartFailed); catch
    via DecibriError to handle it generically.
    """


class VadSampleRateUnsupported(DecibriError):
    """Raised when VAD is enabled with an unsupported sample rate."""


class VadThresholdOutOfRange(DecibriError):
    """Raised when vad_threshold is not in the [0.0, 1.0] range."""


class AecSampleRateUnsupported(DecibriError):
    """Raised when echo cancellation is enabled with an unsupported sample rate.

    The canceller supports 8000 to 48000 Hz, narrower than the range sample_rate
    otherwise accepts, so a rate that is valid for a plain capture is rejected
    once echo cancellation is on.
    """


class AecMultichannelUnsupported(DecibriError):
    """Raised when echo cancellation is combined with more than one channel.

    decibri runs the canceller on one near-end channel. Fed an interleaved
    multichannel capture it never acquires its delay, returns the audio
    unchanged and reports no fault, and a discarded internal carry that is
    not a whole number of frames rotates the channel identities of
    everything after it, at any count above one. Neither is visible in the
    delivered audio, so the combination is refused at construction.
    Cancellation on one channel of an array is unaffected: ``channels=1``
    with a ``channel_map`` naming that channel is accepted. The message
    names the offending count and both accepted ways forward.
    """


class AecConfigInvalid(DecibriError):
    """Raised when the echo canceller rejects its configuration.

    Carries the canceller's own message, which names the window it enforces or
    the string that named no model.
    """


class ResampleConfigInvalid(DecibriError):
    """Raised when a capture sample rate conversion is unsupported.

    Defensive: every rate pair the configuration validator accepts is within
    the resampler's range, so this is not reachable from the Python surface.
    It exists so the core's variant has a class here like every other.
    """


class ResampleAfterFlush(DecibriError):
    """Raised when the resample chain is fed audio after it was flushed.

    Defensive: the capture and File paths stop feeding a chain once it is
    flushed, so this is not reachable from the Python surface. It exists so
    the core's variant has a class here like every other.
    """


class ResampleFailed(DecibriError):
    """Raised when the resampler reports an error decibri does not recognise.

    The message forwards the resampler's own text. Defensive: every error the
    pinned resampler release defines maps to its own class, so this is not
    reachable from the Python surface. It exists so the core's variant has a
    class here like every other.
    """


class FileReadFailed(DecibriError):
    """Raised when an offline audio file cannot be read from disk."""


class FileWriteFailed(DecibriError):
    """Raised when an encoded audio file cannot be written to disk.

    The write-side twin of FileReadFailed: reported by ``File.save`` when
    the filesystem refuses the write (a missing directory, a permission
    failure, a full disk). The audio was encoded before the failure, so the
    fault is the destination's, not the recording's.
    """


class AudioFormatUnsupported(DecibriError):
    """Raised when an offline audio file is in a format decibri cannot decode.

    Covers a container that was not recognised, a container naming a codec the
    reader does not carry, a codec at a sample width it does not carry, and a
    channel layout it cannot decode. The message names the specific tag,
    four-CC or width.
    """


class AudioFileMalformed(DecibriError):
    """Raised when an offline audio file is structurally wrong.

    The container was identified and parsed up to the point where the bytes
    were not what the format requires there. The message names the byte offset
    and what was expected at it.
    """


class AudioFileTruncated(DecibriError):
    """Raised when an offline audio file ends before the audio it declares.

    Covers a file shorter than its own headers state, including one whose
    declared payload length is not a whole number of frames. The message names
    what was needed against what was available.
    """


class VadNotConfigured(DecibriError):
    """Raised when whole-recording analysis is requested without VAD."""


class FileConsumed(DecibriError):
    """Raised when a File is reused after its single pass.

    A File serves one operation: a whole-recording analysis or one
    iteration. Reusing it (analyzing twice, or iterating a File that has
    already been analyzed) raises this rather than yielding an empty stream.
    A lifecycle failure like a closed microphone or speaker stream; catch via
    DecibriError to handle it generically.
    """


class FileEngaged(DecibriError):
    """Raised when a File whose iteration has begun is analyzed.

    Whole-recording analysis times every window from the start of the
    recording, so it runs only on a File still at its start. Once iteration
    has pulled from the File, analyzing it raises this rather than reporting
    on the part not yet read. Construct a second File to do both. A lifecycle
    failure like FileConsumed; catch via DecibriError to handle it generically.
    """


# OrtError intermediate parent


class OrtError(DecibriError):
    """Base class for all ORT-related errors.

    Catches: OrtInitFailed, OrtSessionBuildFailed, OrtThreadsConfigFailed,
    VadModelLoadFailed, ModelLoadFailed, OrtInferenceFailed,
    OrtTensorCreateFailed, OrtTensorExtractFailed, OrtLoadFailed, OrtPathInvalid
    (the last two via OrtPathError intermediate). 10 instance classes total.

    Use this when you want to catch any ORT setup or runtime failure with
    a single except clause. For path-specific failures only (where the
    error has a `path` attribute), catch OrtPathError instead.

    This class has no instances of its own; it is a catch-target only.
    """


# Direct OrtError subclasses (8 instance classes)


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


class ModelLoadFailed(OrtError):
    """Raised when a bundled model loaded through the ONNX session seam fails.

    The model-agnostic counterpart to ``VadModelLoadFailed``: the capture
    denoise stage raises this when its ONNX model file cannot be opened, so a
    denoise load failure is not reported as a VAD error. The ``path`` attribute
    on the exception identifies which model file failed.
    """

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


# Fork-safety detection (Linux). Direct DecibriError subclass, NOT under
# OrtError, because this is a usage error (user initialized ORT in parent
# then forked) rather than an ORT-internal error. Catch via
# `except DecibriError` continues to work.


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


# Non-ORT ONNX backend failure. Direct DecibriError subclass, NOT under
# OrtError, because OrtError is the ORT-specific family: this variant is the
# reserved catch-all for other ONNX backends. Placed here alongside
# ForkAfterOrtInit, the other deliberate non-OrtError member of this region.


class OnnxBackendFailed(DecibriError):
    """Raised when an ONNX inference backend reports an error.

    The reserved catch-all for ONNX backend failures that are not the
    specific ORT setup or path failures grouped under OrtError. A direct
    DecibriError subclass (not an OrtError), so catch via DecibriError to
    handle it generically.
    """
