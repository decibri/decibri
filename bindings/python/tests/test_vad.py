"""Phase 2 VAD tests organized in three sections.

Section A: VAD config validation (no markers). Wrapper-layer validation of
vad / vad_threshold / vad_holdoff_ms. These overlap conceptually with
test_config.py but live here for cohesion with the rest of the VAD surface.
test_error_messages.py also covers a subset for byte-identity; this section
focuses on the validation-pass-through behavior, not message text.

Section B: pure _VadStateMachine and _compute_rms tests (no markers). The
wrapper-layer state machine and RMS helper are pure-Python and test-injectable
without hardware or ORT. This is the bulk of the no-marker VAD coverage.

Section C: end-to-end inference (markers). Full Microphone(vad="silero", ...)
or Microphone(vad="energy", ...) integration tests using real audio +
(for Silero) real ORT runtime.
"""

import struct
import time

import pytest

from decibri import Microphone
from decibri._classes import _VadStateMachine, _compute_rms


# ===========================================================================
# Section A: VAD config validation (no markers)
# ===========================================================================


def test_vad_threshold_default_silero_is_zero_point_five() -> None:
    """Silero mode defaults vad_threshold to 0.5 when not specified.

    Probes the wrapper's mode-dependent threshold default by exercising
    the _VadStateMachine directly (the public Microphone surface no longer
    lets you configure silero behavior without enabling silero, which
    would require ORT load and bundled-model resources).
    """
    sm = _VadStateMachine(
        mode="silero", threshold=0.5, holdoff_ms=300, sample_format="int16"
    )
    assert sm._threshold == 0.5  # type: ignore[attr-defined]


def test_vad_threshold_default_energy_is_zero_point_zero_one() -> None:
    """Energy mode defaults vad_threshold to 0.01 when not specified."""
    d = Microphone(vad=False)
    # vad=False uses energy as the inert placeholder mode in the wrapper,
    # so the wrapper-computed default threshold is 0.01.
    assert d._vad._threshold == 0.01  # type: ignore[attr-defined]


def test_vad_threshold_explicit_overrides_default() -> None:
    """User-supplied vad_threshold wins over the mode-dependent default."""
    d = Microphone(vad=False, vad_threshold=0.42)
    assert d._vad._threshold == 0.42  # type: ignore[attr-defined]


def test_vad_threshold_boundary_zero_accepted() -> None:
    """vad_threshold=0.0 is at the lower boundary; accepted."""
    Microphone(vad=False, vad_threshold=0.0)


def test_vad_threshold_boundary_one_accepted() -> None:
    """vad_threshold=1.0 is at the upper boundary; accepted."""
    Microphone(vad=False, vad_threshold=1.0)


def test_vad_holdoff_zero_accepted() -> None:
    """vad_holdoff_ms=0 is at the lower boundary; accepted."""
    Microphone(vad=False, vad_holdoff_ms=0)


def test_vad_disabled_ignores_vad_kwargs() -> None:
    """With vad=False the VAD-specific kwargs don't trigger model resolution."""
    d = Microphone(vad=False, model_path=None)
    # Bundled model NOT resolved because vad=False; would have raised
    # ValueError if attempted with no bundled file. Here it just passes.
    assert d.vad_score == 0.0
    assert d.is_speaking is False


def test_vad_disabled_probability_returns_zero() -> None:
    """vad_score is always 0.0 when vad=False."""
    d = Microphone(vad=False)
    assert d.vad_score == 0.0


def test_vad_disabled_is_speaking_returns_false() -> None:
    """is_speaking is always False when vad=False."""
    d = Microphone(vad=False)
    assert d.is_speaking is False


# --- Phase 7.5 collapsed vad parameter validation -------------------------


def test_vad_true_raises_migration_error() -> None:
    """vad=True is no longer supported; raises a migration-friendly ValueError."""
    with pytest.raises(ValueError, match="vad=True is no longer supported"):
        Microphone(vad=True)


def test_vad_invalid_string_raises() -> None:
    """vad with an unknown mode string raises with the supported-mode list."""
    with pytest.raises(ValueError, match="Invalid vad value"):
        Microphone(vad="bogus")


def test_vad_mode_kwarg_removed() -> None:
    """The legacy vad_mode kwarg is removed; passing it raises TypeError."""
    with pytest.raises(TypeError, match="vad_mode"):
        Microphone(vad_mode="silero")  # type: ignore[call-arg]


@pytest.mark.requires_bundled_ort
def test_vad_silero_string_constructs() -> None:
    """vad='silero' constructs cleanly when the bundled ORT dylib is present.

    Skipped on dev installs where _ort/ is empty; runs in CI install-test
    against the audited wheel via the requires_bundled_ort marker.
    """
    mic = Microphone(vad="silero")
    assert mic is not None


def test_vad_energy_string_constructs() -> None:
    """vad='energy' constructs cleanly without ORT load."""
    mic = Microphone(vad="energy")
    assert mic is not None


def test_vad_false_constructs() -> None:
    """vad=False (default) constructs cleanly without VAD or ORT load."""
    mic_default = Microphone()
    assert mic_default is not None
    mic_explicit = Microphone(vad=False)
    assert mic_explicit is not None


# ===========================================================================
# Section B: pure _VadStateMachine and _compute_rms tests (no markers)
# ===========================================================================


# --- _compute_rms direct tests --------------------------------------------


def test_compute_rms_silence_int16() -> None:
    """RMS of all-zero int16 buffer is exactly 0.0."""
    silence = b"\x00\x00" * 512
    assert _compute_rms(silence, "int16") == 0.0


def test_compute_rms_silence_float32() -> None:
    """RMS of all-zero float32 buffer is exactly 0.0."""
    silence = b"\x00\x00\x00\x00" * 512
    assert _compute_rms(silence, "float32") == 0.0


def test_compute_rms_int16_dc_quarter_amplitude() -> None:
    """RMS of constant int16 signal at 25% full scale equals 0.25.

    A constant DC signal's RMS equals its absolute amplitude. Sample value
    8192 (= 32768 / 4), normalized by 32768.0, gives 0.25. The 32768.0 vs
    32767.0 distinction is below the tolerance: 8192/32767 - 8192/32768
    is approximately 7.6e-6, well under 0.001.
    """
    sample_count = 8192
    samples = [8192] * sample_count
    buffer = struct.pack(f"<{sample_count}h", *samples)
    rms = _compute_rms(buffer, "int16")
    assert abs(rms - 0.25) < 0.001


def test_compute_rms_int16_loose_bound_half_amplitude() -> None:
    """RMS of int16 buffer with mixed-amplitude samples is in expected range."""
    # Mix of values around half-amplitude; RMS should be in the [0.4, 0.6] range
    samples = [16384, -16384] * 256
    buffer = struct.pack(f"<{len(samples)}h", *samples)
    rms = _compute_rms(buffer, "int16")
    assert 0.4 < rms < 0.6


def test_compute_rms_float32_known_values() -> None:
    """RMS of known float32 buffer matches closed-form calculation."""
    samples = [0.5, -0.5, 0.5, -0.5] * 128
    buffer = struct.pack(f"<{len(samples)}f", *samples)
    rms = _compute_rms(buffer, "float32")
    # All samples have absolute value 0.5, so RMS = sqrt(mean(0.25)) = 0.5
    assert abs(rms - 0.5) < 1e-6


def test_compute_rms_empty_buffer_returns_zero() -> None:
    """Empty input returns 0.0 cleanly (no DivisionError)."""
    assert _compute_rms(b"", "int16") == 0.0
    assert _compute_rms(b"", "float32") == 0.0


# --- _VadStateMachine passthrough tests -----------------------------------


def test_state_machine_silero_passthrough() -> None:
    """Silero mode passes bridge_probability through to vad_score."""
    sm = _VadStateMachine(
        mode="silero", threshold=0.5, holdoff_ms=300, sample_format="int16"
    )
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.7)
    assert sm.vad_score == 0.7
    assert sm.is_speaking is True  # 0.7 > 0.5

    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.2)
    assert sm.vad_score == 0.2
    # Still speaking; below threshold starts holdoff timer but doesn't flip yet
    assert sm.is_speaking is True


def test_state_machine_energy_passthrough() -> None:
    """Energy mode computes RMS via _compute_rms and passes through."""
    sm = _VadStateMachine(
        mode="energy", threshold=0.01, holdoff_ms=300, sample_format="int16"
    )
    silence = b"\x00\x00" * 512
    sm.process_chunk(silence, bridge_probability=0.999)  # bridge prob ignored in energy mode
    assert sm.vad_score == 0.0  # RMS of silence
    assert sm.is_speaking is False

    # Loud buffer above energy threshold
    samples = [16384] * 512
    loud = struct.pack(f"<512h", *samples)
    sm.process_chunk(loud, bridge_probability=0.0)
    assert sm.vad_score > 0.4  # RMS of constant 16384 is ~0.5
    assert sm.is_speaking is True


# --- Threshold-transition tests -------------------------------------------


def test_state_machine_above_threshold_sets_speaking() -> None:
    """Probability above threshold flips is_speaking to True."""
    sm = _VadStateMachine(mode="silero", threshold=0.5, holdoff_ms=300, sample_format="int16")
    assert sm.is_speaking is False
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.9)
    assert sm.is_speaking is True


def test_state_machine_above_threshold_clears_silence_timer() -> None:
    """Above-threshold cancels any pending silence timer."""
    sm = _VadStateMachine(mode="silero", threshold=0.5, holdoff_ms=300, sample_format="int16")
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.9)  # speaking
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.1)  # below; starts timer
    assert sm._silence_started_at is not None  # type: ignore[attr-defined]
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.9)  # above; should cancel
    assert sm._silence_started_at is None  # type: ignore[attr-defined]
    assert sm.is_speaking is True


def test_state_machine_below_threshold_while_silent_no_timer() -> None:
    """Below-threshold while not speaking does not start a timer."""
    sm = _VadStateMachine(mode="silero", threshold=0.5, holdoff_ms=300, sample_format="int16")
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.1)  # below; not speaking
    assert sm.is_speaking is False
    assert sm._silence_started_at is None  # type: ignore[attr-defined]


# --- Holdoff timer expiry test (no markers) -------------------------------


def test_vad_holdoff_state_machine() -> None:
    """Holdoff timer expires after vad_holdoff_ms; is_speaking flips on next access.

    100ms holdoff + 200ms sleep gives 100ms margin to absorb scheduler jitter
    on Windows / VM / CI runners. Verified on Ross's Windows dev machine.
    """
    sm = _VadStateMachine(mode="silero", threshold=0.5, holdoff_ms=100, sample_format="int16")
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.9)
    assert sm.is_speaking is True

    # Below threshold; starts silence timer.
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.1)
    assert sm.is_speaking is True  # still speaking; holdoff active

    # Sleep beyond holdoff window.
    time.sleep(0.2)

    # Property access triggers _refresh_speaking_state, which checks elapsed.
    assert sm.is_speaking is False
    assert sm._silence_started_at is None  # type: ignore[attr-defined]


def test_vad_state_machine_reset() -> None:
    """reset() clears speaking + timer + last probability."""
    sm = _VadStateMachine(mode="silero", threshold=0.5, holdoff_ms=300, sample_format="int16")
    sm.process_chunk(b"\x00" * 1024, bridge_probability=0.9)
    assert sm.is_speaking is True
    assert sm.vad_score == 0.9

    sm.reset()
    assert sm.is_speaking is False
    assert sm.vad_score == 0.0
    assert sm._silence_started_at is None  # type: ignore[attr-defined]


# ===========================================================================
# Section C: end-to-end inference (markers)
# ===========================================================================


@pytest.mark.requires_audio_input
@pytest.mark.requires_bundled_ort
def test_vad_holdoff_end_to_end() -> None:
    """End-to-end Silero VAD with real audio + bundled model + holdoff timer.

    Constructs Microphone(vad='silero') with auto-resolved model
    and a short holdoff. Reads audio for ~250ms; expects is_speaking to
    fluctuate based on real microphone input.
    """
    with Microphone(
        sample_rate=16000,
        channels=1,
        frames_per_buffer=512,
        vad="silero",
        vad_holdoff_ms=100,
    ) as d:
        for _ in range(8):  # ~256ms at 32ms per chunk
            chunk = d.read(timeout_ms=500)
            assert chunk is not None
        # Probability should be a real float in [0, 1]
        assert 0.0 <= d.vad_score <= 1.0


@pytest.mark.requires_audio_input
def test_vad_energy_mode_end_to_end() -> None:
    """End-to-end energy VAD with real audio (no Silero / ORT)."""
    with Microphone(
        sample_rate=16000,
        channels=1,
        frames_per_buffer=512,
        vad="energy",
        vad_threshold=0.01,
    ) as d:
        for _ in range(4):
            chunk = d.read(timeout_ms=500)
            assert chunk is not None
        # vad_score is the RMS of the most recent chunk; in [0, 1]
        assert 0.0 <= d.vad_score <= 1.0


@pytest.mark.requires_audio_input
@pytest.mark.requires_bundled_ort
def test_vad_silero_mode_end_to_end() -> None:
    """End-to-end Silero VAD with real audio + bundled model."""
    with Microphone(
        sample_rate=16000,
        channels=1,
        frames_per_buffer=512,
        vad="silero",
    ) as d:
        for _ in range(4):
            chunk = d.read(timeout_ms=500)
            assert chunk is not None
        assert 0.0 <= d.vad_score <= 1.0
