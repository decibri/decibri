"""Smoke tests for the Phase 1 decibri scaffold.

Verifies the wheel imports, the extension module loads, and
``DecibriBridge.version()`` returns a well-formed ``VersionInfo`` whose
fields match the Rust workspace version and the Node binding's output
byte-for-byte.
"""

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
    # 3.3.0 this assertion breaks deliberately: force a conscious update of
    # the Python test expectation alongside the Rust version bump.
    assert decibri.DecibriBridge.version().decibri == "3.3.0"


def test_version_portaudio_matches_node_binding() -> None:
    # Byte-identical to the Node binding's `Decibri.version().portaudio`
    # output, sourced from the same `decibri::CPAL_VERSION` const in the
    # Rust core. Contract enforced by `crates/decibri/build.rs`.
    assert decibri.DecibriBridge.version().portaudio == "cpal 0.17"


def test_version_info_repr() -> None:
    info = decibri.DecibriBridge.version()
    rep = repr(info)
    assert rep == "VersionInfo(decibri='3.3.0', portaudio='cpal 0.17')"


def test_version_info_is_frozen() -> None:
    import pytest

    info = decibri.DecibriBridge.version()
    with pytest.raises(AttributeError):
        info.decibri = "hijacked"  # type: ignore[misc]
