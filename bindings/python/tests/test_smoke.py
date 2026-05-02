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
    # Phase 7.7 Item A2: bridges are hidden behind the private _decibri
    # module. They are NOT accessible at decibri.<X> top level (sync or
    # async). The async pair was never accessible at the top level; the
    # sync pair (MicrophoneBridge, SpeakerBridge) used to be importable
    # via attribute lookup but was hidden in 7.7 for symmetry.
    assert not hasattr(decibri, "MicrophoneBridge")
    assert not hasattr(decibri, "SpeakerBridge")
    assert not hasattr(decibri, "AsyncMicrophoneBridge")
    assert not hasattr(decibri, "AsyncSpeakerBridge")
    assert hasattr(decibri, "VersionInfo")


def test_extension_module_loads() -> None:
    # The bridges remain accessible via the explicit private submodule
    # for advanced users; this is the only supported path post-7.7.
    from decibri import _decibri

    assert _decibri.MicrophoneBridge is not None
    assert _decibri.SpeakerBridge is not None
    assert _decibri.AsyncMicrophoneBridge is not None
    assert _decibri.AsyncSpeakerBridge is not None
    assert _decibri.VersionInfo is decibri.VersionInfo


def test_version_returns_version_info() -> None:
    from decibri import _decibri

    info = _decibri.MicrophoneBridge.version()
    assert isinstance(info, decibri.VersionInfo)


def test_version_decibri_matches_rust_core() -> None:
    from decibri import _decibri

    # Literal equality per Phase 1 plan. When the Rust workspace bumps past
    # 3.4.0 this assertion breaks deliberately: force a conscious update of
    # the Python test expectation alongside the Rust version bump.
    assert _decibri.MicrophoneBridge.version().decibri == "3.4.0"


def test_version_audio_backend_matches_cpal() -> None:
    from decibri import _decibri

    # Sourced from the same `decibri::CPAL_VERSION` const in the Rust
    # core that the Node binding's version field reads. Contract enforced
    # by `crates/decibri/build.rs`.
    assert _decibri.MicrophoneBridge.version().audio_backend == "cpal 0.17"


def test_version_info_fields() -> None:
    from decibri import _decibri

    info = _decibri.MicrophoneBridge.version()
    assert info.decibri == "3.4.0"
    assert info.audio_backend == "cpal 0.17"
    assert info.binding == "0.1.0a1"


def test_version_info_is_frozen() -> None:
    import pytest

    from decibri import _decibri

    info = _decibri.MicrophoneBridge.version()
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
    """DeviceError, OrtError, OrtPathError catch-target intermediates are reachable."""
    from decibri import DecibriError, DeviceError, OrtError, OrtPathError

    assert issubclass(DeviceError, DecibriError)
    assert issubclass(OrtError, DecibriError)
    assert issubclass(OrtPathError, OrtError)


def test_public_surface_count() -> None:
    """The public ``__all__`` enumerates the top-level surface (18 names).

    Phase 7.7 reshaped the public surface (17 names at lock):
      - Item A1 removed four lowercase factory functions
        (microphone, speaker, async_microphone, async_speaker).
      - Item B2 renamed `devices` -> `input_devices`.
      - Item B7 added `DeviceError` (catch-target intermediate).
      - Item B1 added `Chunk` (audio chunk with metadata).

    Phase 9 Item C7 added `ForkAfterOrtInit` (5th exception entry):
    additive single-class growth, surfaced because users are likely to
    catch it by name to distinguish "use spawn start method" from other
    DecibriError causes.

    Final composition: 2 sync wrappers (Microphone, Speaker) + 2 async
    wrappers (AsyncMicrophone, AsyncSpeaker) + 3 module-level
    convenience functions (input_devices, output_devices, version) +
    2 file convenience functions (record_to_file, async_record_to_file)
    + 3 value types (DeviceInfo, OutputDeviceInfo, VersionInfo)
    + 1 audio chunk dataclass (Chunk)
    + 5 exception entries (DecibriError, DeviceError, ForkAfterOrtInit,
      OrtError, OrtPathError)
    = 2 + 2 + 3 + 2 + 3 + 1 + 5 = 18.
    """
    assert len(decibri.__all__) == 18
    for name in decibri.__all__:
        assert hasattr(decibri, name), f"__all__ lists {name!r} but it is not exported"


def test_lowercase_factories_removed() -> None:
    """Phase 7.7 Item A1: the four lowercase factories are gone."""
    for removed_name in ("microphone", "speaker", "async_microphone", "async_speaker"):
        assert not hasattr(decibri, removed_name), (
            f"decibri.{removed_name} should have been removed in Phase 7.7"
        )


# ---------------------------------------------------------------------------
# Phase 7.5: module-level convenience functions
# ---------------------------------------------------------------------------


def test_module_level_input_devices_aliases_microphone() -> None:
    """decibri.input_devices() is an alias for Microphone.input_devices()."""
    result = decibri.input_devices()
    assert isinstance(result, list)
    expected = decibri.Microphone.input_devices()
    assert len(result) == len(expected)
    # DeviceInfo objects don't implement __eq__; compare by stable id field.
    assert [d.id for d in result] == [d.id for d in expected]


def test_module_level_output_devices_aliases_speaker() -> None:
    """decibri.output_devices() is an alias for Speaker.output_devices()."""
    result = decibri.output_devices()
    assert isinstance(result, list)
    expected = decibri.Speaker.output_devices()
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
