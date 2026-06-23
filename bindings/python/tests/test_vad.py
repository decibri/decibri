"""VAD tests organized in three sections.

Section A: VAD config validation (no markers). Wrapper-layer validation of
vad / vad_threshold / vad_holdoff_ms. These overlap conceptually with
test_config.py but live here for cohesion with the rest of the VAD surface.
test_error_messages.py also covers a subset for byte-identity; this section
focuses on the validation-pass-through behavior, not message text.

Section B: pure _VadStateMachine tests (no markers). The wrapper-layer state
machine applies threshold + holdoff policy to the score the bridge computes
for both VAD modes; it is pure-Python and test-injectable without hardware or
ORT. The energy RMS itself is computed natively (decibri::sample::rms) on the
pre-enhancement signal, so its correctness and transform-invariance are tested
in the Rust core, not here.

Section C: end-to-end inference (markers). Full Microphone(vad="silero", ...)
or Microphone(vad="energy", ...) integration tests using real audio +
(for Silero) real ORT runtime.
"""

import time

import pytest

from decibri import Microphone
from decibri._classes import _VadStateMachine


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
    sm = _VadStateMachine(threshold=0.5, holdoff_ms=300)
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


# --- collapsed vad parameter validation -------------------------


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
# Section B: pure _VadStateMachine tests (no markers)
# ===========================================================================
#
# The state machine applies the threshold + holdoff policy to the bridge score
# for both VAD modes. The energy RMS is computed natively
# (decibri::sample::rms) on the pre-enhancement signal; its correctness and
# transform-invariance are covered by the Rust core tests, not here. These
# tests cover the wrapper policy: the score feeds straight through to
# vad_score, and the threshold + holdoff state machine runs on it identically
# regardless of which mode produced it.


# --- _VadStateMachine passthrough tests -----------------------------------


def test_state_machine_silero_passthrough() -> None:
    """The bridge score (Silero probability) passes through to vad_score."""
    sm = _VadStateMachine(threshold=0.5, holdoff_ms=300)
    sm.process_chunk(0.7)
    assert sm.vad_score == 0.7
    assert sm.is_speaking is True  # 0.7 > 0.5

    sm.process_chunk(0.2)
    assert sm.vad_score == 0.2
    # Still speaking; below threshold starts holdoff timer but doesn't flip yet
    assert sm.is_speaking is True


def test_state_machine_energy_passthrough() -> None:
    """The bridge score (energy RMS) passes through with the energy threshold.

    Energy mode no longer computes RMS in the wrapper; the bridge computes it
    natively on the pre-enhancement signal and exposes it as vad_probability.
    The state machine applies the energy threshold default (0.01) to that
    score exactly as it applies the silero threshold to a probability.
    """
    sm = _VadStateMachine(threshold=0.01, holdoff_ms=300)
    sm.process_chunk(0.0)  # silence: RMS 0.0
    assert sm.vad_score == 0.0
    assert sm.is_speaking is False

    sm.process_chunk(0.5)  # loud: RMS well above the 0.01 threshold
    assert sm.vad_score == 0.5
    assert sm.is_speaking is True


# --- Threshold-transition tests -------------------------------------------


def test_state_machine_above_threshold_sets_speaking() -> None:
    """A score above threshold flips is_speaking to True."""
    sm = _VadStateMachine(threshold=0.5, holdoff_ms=300)
    assert sm.is_speaking is False
    sm.process_chunk(0.9)
    assert sm.is_speaking is True


def test_state_machine_above_threshold_clears_silence_timer() -> None:
    """Above-threshold cancels any pending silence timer."""
    sm = _VadStateMachine(threshold=0.5, holdoff_ms=300)
    sm.process_chunk(0.9)  # speaking
    sm.process_chunk(0.1)  # below; starts timer
    assert sm._silence_started_at is not None  # type: ignore[attr-defined]
    sm.process_chunk(0.9)  # above; should cancel
    assert sm._silence_started_at is None  # type: ignore[attr-defined]
    assert sm.is_speaking is True


def test_state_machine_below_threshold_while_silent_no_timer() -> None:
    """Below-threshold while not speaking does not start a timer."""
    sm = _VadStateMachine(threshold=0.5, holdoff_ms=300)
    sm.process_chunk(0.1)  # below; not speaking
    assert sm.is_speaking is False
    assert sm._silence_started_at is None  # type: ignore[attr-defined]


# --- Holdoff timer expiry test (no markers) -------------------------------


def test_vad_holdoff_state_machine() -> None:
    """Holdoff timer expires after vad_holdoff_ms; is_speaking flips on next access.

    100ms holdoff + 200ms sleep gives 100ms margin to absorb scheduler jitter
    on Windows / VM / CI runners. Verified on Ross's Windows dev machine.
    """
    sm = _VadStateMachine(threshold=0.5, holdoff_ms=100)
    sm.process_chunk(0.9)
    assert sm.is_speaking is True

    # Below threshold; starts silence timer.
    sm.process_chunk(0.1)
    assert sm.is_speaking is True  # still speaking; holdoff active

    # Sleep beyond holdoff window.
    time.sleep(0.2)

    # Property access triggers _refresh_speaking_state, which checks elapsed.
    assert sm.is_speaking is False
    assert sm._silence_started_at is None  # type: ignore[attr-defined]


def test_vad_state_machine_reset() -> None:
    """reset() clears speaking + timer + last probability."""
    sm = _VadStateMachine(threshold=0.5, holdoff_ms=300)
    sm.process_chunk(0.9)
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
