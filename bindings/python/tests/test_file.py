"""File: the offline source (conditioning + whole-recording analysis).

Covers the ``File`` / ``AsyncFile`` surface: the path and buffer
constructors (bare ``File(path)`` and ``File.open(path)`` equivalence),
conditioning correctness through the shared chain, the end-of-file flush
tail, per-chunk VAD metadata in FILE time, whole-recording ``analyze()`` /
``analyse()`` returning a ``VadReport``, the no-VAD hard error, and the
internal detector-feed resample for a non-detector target rate.

Silero-driven cases read the repo's golden speech fixture and skip when it
is absent (an installed-package run outside the repo); everything else is
hardware-free and model-free (energy mode), so the module is CI-safe.
"""

from __future__ import annotations

import asyncio
import math
import struct
import wave
from pathlib import Path

import pytest

import decibri
from decibri import File, Segment, Vad, VadReport, VadWindow
from decibri.exceptions import (
    FileConsumed,
    FileReadFailed,
    VadNotConfigured,
    WavInvalid,
)

# The golden TTS speech fixture the Rust core's VAD regression tests use.
# Resolved relative to the repo checkout; Silero cases skip when absent.
_GOLDEN_WAV = (
    Path(__file__).parents[3]
    / "crates"
    / "decibri"
    / "tests"
    / "assets"
    / "vad-golden-tts-speech-16k.wav"
)

requires_golden = pytest.mark.skipif(
    not _GOLDEN_WAV.is_file(),
    reason="golden speech fixture not available outside the repo checkout",
)


def sine_samples(rate: int, seconds: float, amplitude: float = 0.5) -> list[float]:
    """A mono sine at the given rate."""
    count = int(rate * seconds)
    return [
        amplitude * math.sin(2.0 * math.pi * 440.0 * i / rate) for i in range(count)
    ]


def write_wav(path: Path, samples: list[float], rate: int) -> None:
    """Write mono 16-bit PCM WAV via the standard library."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        clamped = (
            max(-32768, min(32767, int(s * 32768.0))) for s in samples
        )
        w.writeframes(struct.pack(f"<{len(samples)}h", *clamped))


def read_all(file: File) -> bytes:
    """Concatenate every conditioned chunk a File delivers."""
    out = bytearray()
    for chunk in file:
        assert isinstance(chunk, bytes)
        out.extend(chunk)
    return bytes(out)


# ---------------------------------------------------------------------------
# Constructors and conditioning
# ---------------------------------------------------------------------------


def test_bare_and_open_are_equivalent(tmp_path: Path) -> None:
    """File(path) and File.open(path) produce identical output."""
    path = tmp_path / "clip.wav"
    write_wav(path, sine_samples(16000, 0.5), 16000)
    bare = read_all(File(path))
    opened = read_all(File.open(path))
    assert bare == opened
    assert len(bare) > 0


def test_passthrough_roundtrip(tmp_path: Path) -> None:
    """A mono 16 kHz WAV with no conditioning passes through exactly."""
    path = tmp_path / "clip.wav"
    samples = sine_samples(16000, 0.5)  # 8000 samples
    write_wav(path, samples, 16000)
    original = path.read_bytes()[44:]  # the data payload after the header

    file = File(path)
    chunks = list(file)
    out = b"".join(chunks)
    assert out == original
    # Full-size chunks: 1600 samples of int16 per chunk.
    assert all(len(c) == 3200 for c in chunks)


def test_buffer_resamples_to_target() -> None:
    """The buffer path resamples input_rate to sample_rate, tail included."""
    samples = sine_samples(48000, 0.25)  # 12000 samples at 48 kHz
    file = File.buffer(samples, input_rate=48000, sample_rate=16000)
    out = read_all(file)
    out_samples = len(out) // 2
    # 3:1 resample of 12000 samples is 4000 out, plus the resampler's
    # flushed group-delay tail (a few hundred samples); the tail proves the
    # end-of-file flush delivers rather than truncates.
    assert 4000 <= out_samples <= 4000 + 512


def test_conditioning_removes_dc_offset() -> None:
    """dc_removal conditions the delivered audio (mean offset removed)."""
    offset = [0.25 + s for s in sine_samples(16000, 0.5, amplitude=0.1)]
    plain = File.buffer(offset, input_rate=16000)
    conditioned = File.buffer(offset, input_rate=16000, dc_removal=True)

    def mean_of(raw: bytes) -> float:
        values = struct.unpack(f"<{len(raw) // 2}h", raw)
        return sum(values) / len(values) / 32768.0

    plain_mean = mean_of(read_all(plain))
    conditioned_mean = mean_of(read_all(conditioned))
    assert plain_mean > 0.2
    assert abs(conditioned_mean) < 0.05


def test_dtype_float32_delivers_f32_bytes(tmp_path: Path) -> None:
    """dtype='float32' delivers 4 bytes per sample, mirroring Microphone."""
    path = tmp_path / "clip.wav"
    samples = sine_samples(16000, 0.2)
    write_wav(path, samples, 16000)
    out = read_all(File(path, dtype="float32"))
    assert len(out) == len(samples) * 4


def test_buffer_accepts_ndarray() -> None:
    """A numpy ndarray of samples is accepted by the buffer constructor."""
    np = pytest.importorskip("numpy")
    arr = np.asarray(sine_samples(16000, 0.2), dtype=np.float32)
    out = read_all(File.buffer(arr, input_rate=16000))
    assert len(out) == len(arr) * 2


def test_buffer_rejects_other_types() -> None:
    """Anything but a list of floats or an ndarray is rejected clearly."""
    with pytest.raises(TypeError, match="list of floats or a numpy ndarray"):
        File.buffer("not samples", input_rate=16000)  # type: ignore[arg-type]


def test_open_missing_file_raises() -> None:
    with pytest.raises(FileReadFailed):
        File("no-such-file.wav")


def test_open_invalid_wav_raises(tmp_path: Path) -> None:
    path = tmp_path / "junk.wav"
    path.write_bytes(b"this is not a wav file")
    with pytest.raises(WavInvalid):
        File(path)


def test_repr_shows_construction() -> None:
    file = File.buffer([0.0] * 1600, input_rate=16000)
    assert repr(file) == "File(sample_rate=16000, dtype='int16', vad=False)"


# ---------------------------------------------------------------------------
# Per-chunk VAD metadata in file time (energy mode: model-free, CI-safe)
# ---------------------------------------------------------------------------


def test_iter_with_metadata_reports_file_time() -> None:
    """Chunk timestamps are file positions in seconds, not wall-clock."""
    samples = sine_samples(16000, 0.5)
    file = File.buffer(samples, input_rate=16000, vad="energy")
    chunks = list(file.iter_with_metadata())
    assert len(chunks) == 5
    for i, chunk in enumerate(chunks):
        # 1600-sample chunks at 16 kHz are exactly 0.1 s of file time.
        assert chunk.timestamp == pytest.approx(i * 0.1)
        assert chunk.sequence == i
        assert isinstance(chunk.is_speaking, bool)
        assert 0.0 <= chunk.vad_score <= 1.0


def test_file_time_holdoff_is_processing_speed_independent() -> None:
    """The speaking holdoff elapses in FILE time, not wall-clock time.

    Half a second of loud signal then 0.6 s of silence: with the default
    300 ms holdoff, the speaking state must clear once 300 ms of FILE time
    silence has passed, no matter how fast the file processes.
    """
    loud = sine_samples(16000, 0.5)
    silence = [0.0] * 9600  # 0.6 s
    file = File.buffer(loud + silence, input_rate=16000, vad="energy")
    states = [chunk.is_speaking for chunk in file.iter_with_metadata()]
    # 11 chunks of 0.1 s: 5 loud, 6 silent.
    assert states[:5] == [True] * 5
    # Holdoff: silence begins at 0.5 s; 300 ms of file time later (inside
    # the chunk ending at 0.9 s) the state clears and stays cleared.
    assert states[-2:] == [False, False]
    assert True in states[5:]  # the holdoff kept it alive briefly


def test_vad_disabled_has_inert_flags() -> None:
    file = File.buffer(sine_samples(16000, 0.2), input_rate=16000)
    chunk = file.read_with_metadata()
    assert chunk is not None
    assert chunk.is_speaking is False
    assert chunk.vad_score == 0.0


# ---------------------------------------------------------------------------
# Whole-recording analysis
# ---------------------------------------------------------------------------


def test_analyze_requires_vad() -> None:
    """analyze() on a File built without vad= raises, never a silent VAD."""
    file = File.buffer(sine_samples(16000, 0.2), input_rate=16000)
    with pytest.raises(VadNotConfigured):
        file.analyze()
    file = File.buffer(sine_samples(16000, 0.2), input_rate=16000)
    with pytest.raises(VadNotConfigured):
        file.analyse()


def test_analyze_rejects_energy_mode() -> None:
    file = File.buffer(sine_samples(16000, 0.2), input_rate=16000, vad="energy")
    with pytest.raises(ValueError, match="analyze\\(\\) requires vad='silero'"):
        file.analyze()


@requires_golden
@pytest.mark.requires_bundled_ort
def test_analyze_returns_report_with_speech() -> None:
    """The golden speech recording analyzes into scores and segments."""
    report = File(_GOLDEN_WAV, vad="silero").analyze()
    assert isinstance(report, VadReport)
    assert len(report.scores) > 0
    assert all(isinstance(w, VadWindow) for w in report.scores)
    # Windows tile the recording in 32 ms steps of file time.
    for i, w in enumerate(report.scores):
        assert w.start == pytest.approx(i * 512 / 16000)
        assert w.end == pytest.approx((i + 1) * 512 / 16000)
        assert 0.0 <= w.vad_score <= 1.0
        assert w.is_speech == (w.vad_score >= 0.5)
    # Real speech: scores cross the threshold and merge into segments.
    assert max(w.vad_score for w in report.scores) >= 0.5
    assert len(report.segments) > 0
    duration = report.scores[-1].end
    for segment in report.segments:
        assert isinstance(segment, Segment)
        assert 0.0 <= segment.start < segment.end <= duration + 1e-9


@requires_golden
@pytest.mark.requires_bundled_ort
def test_analyse_equals_analyze() -> None:
    """Both spellings return the same analysis."""
    a = File(_GOLDEN_WAV, vad="silero").analyze()
    b = File(_GOLDEN_WAV, vad="silero").analyse()
    assert a.scores == b.scores
    assert a.segments == b.segments


@requires_golden
@pytest.mark.requires_bundled_ort
def test_analyze_with_non_detector_rate_resamples_internally() -> None:
    """A non-8k/16k target rate still analyzes: the detector feed is
    resampled internally, with no setting and no error."""
    report = File(_GOLDEN_WAV, vad="silero", sample_rate=22050).analyze()
    assert len(report.scores) > 0
    assert max(w.vad_score for w in report.scores) >= 0.5
    assert len(report.segments) > 0


@requires_golden
@pytest.mark.requires_bundled_ort
def test_analyze_matches_per_chunk_feed() -> None:
    """The detector-feed invariant, wrapper level: whole-recording scores
    and the per-chunk streaming scores read the same pre-conditioning
    signal, so the analysis maximum matches the streaming maximum."""
    report = File(_GOLDEN_WAV, vad="silero").analyze()

    streamed = File(_GOLDEN_WAV, vad="silero")
    stream_max = 0.0
    for chunk in streamed.iter_with_metadata():
        stream_max = max(stream_max, chunk.vad_score)
    analysis_max = max(w.vad_score for w in report.scores)
    assert analysis_max == pytest.approx(stream_max, abs=1e-6)


def test_analyze_consumed_file_raises(tmp_path: Path) -> None:
    """A File is one pass: analysis after close or a second analysis
    reports the consumed state clearly."""
    path = tmp_path / "clip.wav"
    write_wav(path, sine_samples(16000, 0.2), 16000)
    file = File(path, vad="energy")
    file.close()
    with pytest.raises(FileConsumed, match="File already consumed"):
        # Energy mode would be rejected first; bypass by checking the
        # bridge-level consumed state via a silero-free close: the
        # consumed check fires before mode validation at the bridge.
        file._bridge.analyze()


def test_iteration_after_analysis_raises() -> None:
    """A File whose source was taken by analysis raises on a later read,
    rather than yielding an empty stream indistinguishable from silence."""
    file = File.buffer(sine_samples(16000, 0.2), input_rate=16000)
    # analyze() on a no-VAD File takes the source, then reports the missing
    # VAD; the source is consumed either way.
    with pytest.raises(VadNotConfigured):
        file.analyze()
    with pytest.raises(FileConsumed):
        for _ in file:
            pass
    with pytest.raises(FileConsumed):
        file.read()


def test_iteration_after_exhaustion_is_quiet() -> None:
    """A File read to the end yields nothing on a second pass, no error."""
    file = File.buffer(sine_samples(16000, 0.2), input_rate=16000)
    assert len(read_all(file)) > 0
    assert read_all(file) == b""
    assert file.read() is None


def test_read_after_close_is_quiet() -> None:
    """A closed File reads as an ended stream, no error."""
    file = File.buffer(sine_samples(16000, 0.2), input_rate=16000)
    file.close()
    assert file.read() is None
    assert list(file) == []


def test_vad_object_configures_file() -> None:
    """The same Vad(...) config object the Microphone takes attaches to
    File, holdoff and threshold included."""
    file = File.buffer(
        sine_samples(16000, 0.2),
        input_rate=16000,
        vad=Vad(model="energy", threshold=0.2, holdoff_ms=100),
    )
    assert file._vad._threshold == pytest.approx(0.2)
    assert file._vad._holdoff_seconds == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# AsyncFile
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_file_iterates(tmp_path: Path) -> None:
    path = tmp_path / "clip.wav"
    samples = sine_samples(16000, 0.3)
    write_wav(path, samples, 16000)
    f = await decibri.AsyncFile.open(path)
    total = 0
    async with f:
        async for chunk in f:
            assert isinstance(chunk, bytes)
            total += len(chunk)
    assert total == len(samples) * 2


@pytest.mark.asyncio
async def test_async_file_metadata_and_buffer() -> None:
    f = await decibri.AsyncFile.buffer(
        sine_samples(16000, 0.3), input_rate=16000, vad="energy"
    )
    count = 0
    async for chunk in f.aiter_with_metadata():
        assert chunk.timestamp == pytest.approx(count * 0.1)
        count += 1
    assert count == 3


@requires_golden
@pytest.mark.requires_bundled_ort
@pytest.mark.asyncio
async def test_async_analyze() -> None:
    f = await decibri.AsyncFile.open(_GOLDEN_WAV, vad="silero")
    report = await f.analyze()
    assert len(report.segments) > 0
    f2 = await decibri.AsyncFile.open(_GOLDEN_WAV, vad="silero")
    report2 = await f2.analyse()
    assert report2.scores == report.scores
