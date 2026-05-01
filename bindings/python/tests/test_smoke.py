"""Smoke tests for the decibri Python wheel.

Phase 1 scope: wheel import, extension module load, VersionInfo correctness.
Phase 2 additions: public API name surface (Microphone, Speaker, exception
hierarchy roots) and the Silero VAD model bundling check.

These tests are intentionally low-coverage and high-confidence: they verify
the wheel installs and exposes its top-level surface. Detailed coverage of
each surface lives in test_capture, test_output, test_vad, test_config,
test_exceptions, test_error_messages, and test_lifecycle.
"""

import importlib.resources

import decibri


def test_package_imports() -> None:
    assert decibri.__version__ == "0.1.0a1"
    # MicrophoneBridge remains accessible via attribute lookup even though
    # it is intentionally not in __all__ (Phase 7.6 Item B2).
    assert hasattr(decibri, "MicrophoneBridge")
    assert hasattr(decibri, "VersionInfo")


def test_extension_module_loads() -> None:
    # Importing from the underscore-prefixed internal module should succeed
    # and return the exact same objects the public __init__.py re-exports.
    from decibri import _decibri

    assert _decibri.MicrophoneBridge is decibri.MicrophoneBridge
    assert _decibri.VersionInfo is decibri.VersionInfo


def test_version_returns_version_info() -> None:
    info = decibri.MicrophoneBridge.version()
    assert isinstance(info, decibri.VersionInfo)


def test_version_decibri_matches_rust_core() -> None:
    # Literal equality per Phase 1 plan. When the Rust workspace bumps past
    # 3.3.2 this assertion breaks deliberately: force a conscious update of
    # the Python test expectation alongside the Rust version bump.
    assert decibri.MicrophoneBridge.version().decibri == "3.3.2"


def test_version_audio_backend_matches_cpal() -> None:
    # Sourced from the same `decibri::CPAL_VERSION` const in the Rust
    # core that the Node binding's version field reads. Contract enforced
    # by `crates/decibri/build.rs`. Phase 7.5: field renamed from
    # `portaudio` to `audio_backend` (more honest; we use cpal, not
    # portaudio).
    assert decibri.MicrophoneBridge.version().audio_backend == "cpal 0.17"


def test_version_info_fields() -> None:
    info = decibri.MicrophoneBridge.version()
    assert info.decibri == "3.3.2"
    assert info.audio_backend == "cpal 0.17"
    assert info.binding == "0.1.0a1"


def test_version_info_is_frozen() -> None:
    import pytest

    info = decibri.MicrophoneBridge.version()
    with pytest.raises(AttributeError):
        info.decibri = "hijacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Phase 2: public API name surface
# ---------------------------------------------------------------------------


def test_public_classes_importable() -> None:
    """The two public Python wrapper classes are reachable from the package."""
    from decibri import Microphone, Speaker

    assert isinstance(Microphone, type)
    assert isinstance(Speaker, type)


def test_exception_root_importable() -> None:
    """The exception hierarchy root is reachable from the package."""
    from decibri import DecibriError

    assert isinstance(DecibriError, type)
    assert issubclass(DecibriError, Exception)


def test_exception_intermediate_parents_importable() -> None:
    """OrtError and OrtPathError catch-target intermediates are reachable."""
    from decibri import DecibriError, OrtError, OrtPathError

    assert issubclass(OrtError, DecibriError)
    assert issubclass(OrtPathError, OrtError)


def test_public_surface_count() -> None:
    """The public ``__all__`` enumerates the trimmed top-level surface (19 names).

    Phase 7.6 trimmed the public surface in three stages:
      - Item B2 dropped the two internal bridges (MicrophoneBridge,
        SpeakerBridge); they remain importable via attribute lookup.
      - Item C3 (Shape C2) dropped the 29 granular exception subclasses
        from __all__; only the three catch-target roots (DecibriError,
        OrtError, OrtPathError) remain. All 32 exception classes are
        still importable both at decibri.<X> and at decibri.exceptions.<X>.
      - Item C5 added record_to_file and async_record_to_file.

    Final composition: 2 sync wrappers (Microphone, Speaker) + 2 async
    wrappers (AsyncMicrophone, AsyncSpeaker) + 4 lowercase factories
    (microphone, speaker, async_microphone, async_speaker) + 3 module-
    level convenience functions (devices, output_devices, version) +
    2 file convenience functions (record_to_file, async_record_to_file)
    + 3 value types (DeviceInfo, OutputDeviceInfo, VersionInfo) +
    3 exception roots (DecibriError, OrtError, OrtPathError)
    = 2 + 2 + 4 + 3 + 2 + 3 + 3 = 19.
    """
    assert len(decibri.__all__) == 19
    for name in decibri.__all__:
        assert hasattr(decibri, name), f"__all__ lists {name!r} but it is not exported"


# ---------------------------------------------------------------------------
# Phase 7.5: module-level convenience functions
# ---------------------------------------------------------------------------


def test_module_level_devices_aliases_microphone() -> None:
    """decibri.devices() is an alias for Microphone.devices() (input devices)."""
    result = decibri.devices()
    assert isinstance(result, list)
    expected = decibri.Microphone.devices()
    assert len(result) == len(expected)
    # DeviceInfo objects don't implement __eq__; compare by stable id field.
    assert [d.id for d in result] == [d.id for d in expected]


def test_module_level_output_devices_aliases_speaker() -> None:
    """decibri.output_devices() is an alias for Speaker.devices() (output devices)."""
    result = decibri.output_devices()
    assert isinstance(result, list)
    expected = decibri.Speaker.devices()
    assert len(result) == len(expected)
    assert [d.id for d in result] == [d.id for d in expected]


def test_module_level_version_aliases_microphone() -> None:
    """decibri.version() is an alias for Microphone.version()."""
    result = decibri.version()
    assert hasattr(result, "decibri")
    assert hasattr(result, "audio_backend")
    assert hasattr(result, "binding")
    expected = decibri.Microphone.version()
    assert result.decibri == expected.decibri
    assert result.audio_backend == expected.audio_backend
    assert result.binding == expected.binding


# ---------------------------------------------------------------------------
# Phase 2: Silero VAD model bundling
# ---------------------------------------------------------------------------


def test_silero_model_bundled_in_wheel() -> None:
    """The Silero VAD ONNX model ships in the wheel at the expected path.

    Validates the maturin include directive in pyproject.toml. The size
    floor of 2 MB catches truncated or empty bundled files; the canonical
    file is approximately 2.3 MB. Editable installs (maturin develop)
    resolve to the same path; this test runs identically in editable and
    standard wheel installs.
    """
    model_path = importlib.resources.files("decibri").joinpath(
        "models/silero_vad.onnx"
    )
    assert model_path.is_file(), f"bundled Silero model not found at {model_path}"
    size = model_path.stat().st_size
    assert size > 2 * 1024 * 1024, f"bundled Silero model is suspiciously small: {size} bytes"
