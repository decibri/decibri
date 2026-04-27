"""Smoke tests for the decibri Python wheel.

Phase 1 scope: wheel import, extension module load, VersionInfo correctness.
Phase 2 additions: public API name surface (Decibri, DecibriOutput, exception
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
    assert hasattr(decibri, "DecibriBridge")
    assert hasattr(decibri, "VersionInfo")


def test_extension_module_loads() -> None:
    # Importing from the underscore-prefixed internal module should succeed
    # and return the exact same objects the public __init__.py re-exports.
    from decibri import _decibri

    assert _decibri.DecibriBridge is decibri.DecibriBridge
    assert _decibri.VersionInfo is decibri.VersionInfo


def test_version_returns_version_info() -> None:
    info = decibri.DecibriBridge.version()
    assert isinstance(info, decibri.VersionInfo)


def test_version_decibri_matches_rust_core() -> None:
    # Literal equality per Phase 1 plan. When the Rust workspace bumps past
    # 3.3.2 this assertion breaks deliberately: force a conscious update of
    # the Python test expectation alongside the Rust version bump.
    assert decibri.DecibriBridge.version().decibri == "3.3.2"


def test_version_portaudio_matches_node_binding() -> None:
    # Byte-identical to the Node binding's `Decibri.version().portaudio`
    # output, sourced from the same `decibri::CPAL_VERSION` const in the
    # Rust core. Contract enforced by `crates/decibri/build.rs`.
    assert decibri.DecibriBridge.version().portaudio == "cpal 0.17"


def test_version_info_fields() -> None:
    info = decibri.DecibriBridge.version()
    assert info.decibri == "3.3.2"
    assert info.portaudio == "cpal 0.17"
    assert info.binding == "0.1.0a1"


def test_version_info_is_frozen() -> None:
    import pytest

    info = decibri.DecibriBridge.version()
    with pytest.raises(AttributeError):
        info.decibri = "hijacked"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Phase 2: public API name surface
# ---------------------------------------------------------------------------


def test_public_classes_importable() -> None:
    """The two public Python wrapper classes are reachable from the package."""
    from decibri import Decibri, DecibriOutput

    assert isinstance(Decibri, type)
    assert isinstance(DecibriOutput, type)


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
    """The public ``__all__`` enumerates the full Phase 2 surface (39 names).

    Composition: 2 public wrappers + 2 internal pyclasses + 3 value types
    + 32 exception classes (1 base + 20 direct subclasses + OrtError + 7
    direct OrtError subclasses + OrtPathError + 2 OrtPathError subclasses).
    """
    assert len(decibri.__all__) == 39
    for name in decibri.__all__:
        assert hasattr(decibri, name), f"__all__ lists {name!r} but it is not exported"


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
