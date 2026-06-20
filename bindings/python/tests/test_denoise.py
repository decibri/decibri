"""Denoise capture tests.

Two tiers:

- CI-safe: the closed-set ``denoise`` selector validates and resolves the
  bundled model path at construction time WITHOUT loading the model (the ONNX
  load happens at ``start()``), so construction, unknown-name rejection, and the
  off-by-default path all run without audio hardware or ORT.
- Hardware + ORT: the end-to-end denoise run and the model-load-failure surface
  need a real input device and the bundled ONNX Runtime; both are marked and
  auto-skip on CI runners (no audio) and dev installs (no bundled ``_ort``).

The denoise public surface mirrors the Silero VAD shape: a model name turns it
on, absence leaves it off, an unknown name is a clear error.
"""

from __future__ import annotations

import pytest

from decibri import AsyncMicrophone, Microphone, ModelLoadFailed


# ---------------------------------------------------------------------------
# CI-safe: construction + validation (no model load, no hardware, no ORT).
# ---------------------------------------------------------------------------


def test_denoise_constructs() -> None:
    """A valid model name constructs; the model loads later, at start()."""
    mic = Microphone(sample_rate=16000, channels=1, denoise="fastenhancer-t")
    assert mic is not None


def test_async_denoise_constructs() -> None:
    """AsyncMicrophone mirrors the sync surface and accepts denoise."""
    mic = AsyncMicrophone(sample_rate=16000, channels=1, denoise="fastenhancer-t")
    assert mic is not None


def test_denoise_off_by_default() -> None:
    """No denoise kwarg constructs identically to a plain microphone."""
    mic = Microphone(sample_rate=16000, channels=1)
    assert mic is not None


def test_denoise_unknown_name_raises() -> None:
    """An unrecognized model name is a clear ValueError, not a silent miss."""
    with pytest.raises(ValueError, match="Invalid denoise value"):
        Microphone(denoise="whisper")  # type: ignore[arg-type]


def test_async_denoise_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Invalid denoise value"):
        AsyncMicrophone(denoise="whisper")  # type: ignore[arg-type]


@pytest.mark.requires_bundled_ort
def test_denoise_composes_with_vad() -> None:
    """denoise and Silero VAD can be requested together (construction).

    Marked requires_bundled_ort because vad='silero' loads the Silero ONNX model
    at construction, so this needs the bundled ORT dylib (same gate the other
    vad='silero' construction tests use). denoise itself loads at start(), not
    construction, so the denoise-only construction tests above stay ORT-free.
    """
    mic = Microphone(
        sample_rate=16000,
        channels=1,
        denoise="fastenhancer-t",
        vad="silero",
    )
    assert mic is not None


# ---------------------------------------------------------------------------
# Hardware + ORT: end-to-end denoise run and the model-load-failure surface.
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
@pytest.mark.requires_bundled_ort
def test_denoise_capture_end_to_end() -> None:
    """Capturing with denoise on produces enhanced audio through the binding."""
    mic = Microphone(sample_rate=16000, channels=1, denoise="fastenhancer-t")
    chunks = 0
    total_bytes = 0
    with mic:
        for _ in range(10):
            chunk = mic.read(timeout_ms=500)
            if chunk is None:
                break
            assert isinstance(chunk, bytes)
            chunks += 1
            total_bytes += len(chunk)
    assert chunks > 0, "denoise capture produced no chunks"
    assert total_bytes > 0, "denoise capture produced no audio bytes"


@pytest.mark.requires_audio_input
@pytest.mark.requires_bundled_ort
def test_denoise_model_load_failure_surfaces_model_load_failed() -> None:
    """A bad denoise model path raises ModelLoadFailed, not the VAD-named error.

    Forced by constructing the bridge directly with a nonexistent model path
    (the public surface only exposes the bundled model). The resolved ORT dylib
    path is passed so the bridge forwards it to the denoise stage and ORT
    initializes, making the failure the model load itself, not an ORT-init
    failure.
    """
    from decibri import _decibri
    from decibri._ort_resolver import resolve_ort_dylib_path

    bridge = _decibri.MicrophoneBridge(
        sample_rate=16000,
        channels=1,
        frames_per_buffer=1600,
        format="int16",
        denoise="fastenhancer-t",
        denoise_model_path="/nonexistent/fastenhancer_t.onnx",
        ort_library_path=resolve_ort_dylib_path(None),
    )
    with pytest.raises(ModelLoadFailed):
        bridge.start()
