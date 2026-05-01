"""Phase 7 wheel-install smoke tests.

These tests run against the production install surface (the wheel
installed in a clean venv), NOT against maturin develop builds.
They validate that the bundled artifacts (Silero VAD model, ORT
dylib) are present and discoverable, and that the public surface
constructs cleanly without audio hardware.

The tests are persistent (not Phase 7-specific) and serve as the
canonical "production install" gate going forward. They run as part
of the install-test step in python-ci.yml and on every push.

Audio-hardware-dependent tests live in test_capture.py and test_output.py
with @pytest.mark.requires_audio_input / requires_audio_output markers
(skipped on cloud CI). The smoke tests in THIS file deliberately
exclude audio paths so they run unconditionally on cloud runners.
"""

from __future__ import annotations

import importlib.metadata

import pytest

import decibri


def test_import_decibri() -> None:
    """The package imports successfully from the installed wheel."""
    assert decibri is not None
    assert hasattr(decibri, "__version__")
    assert decibri.__version__ == "0.1.0a1"


def test_construct_decibri_default() -> None:
    """Microphone() constructs cleanly without VAD (no ORT load needed)."""
    d = decibri.Microphone()
    assert d is not None


@pytest.mark.requires_bundled_ort
def test_construct_decibri_silero_vad() -> None:
    """Microphone(vad='silero') constructs cleanly with ORT.

    Verifies: ORT dylib loads from bundled _ort/, Silero ONNX model
    loads from bundled models/silero_vad.onnx, ONNX session initializes.
    Skipped if _ort/ is empty (dev installs without bundled ORT).
    """
    d = decibri.Microphone(vad="silero")
    assert d is not None


@pytest.mark.requires_bundled_ort
def test_resolver_finds_bundled() -> None:
    """The ORT resolver finds the bundled dylib after auditwheel/delocate repair.

    Verifies: the resolver's glob pattern (libonnxruntime*.so/.dylib,
    onnxruntime*.dll) finds the bundled dylib regardless of any hash
    suffix added by auditwheel's wheel repair, and regardless of whether
    delocate's -L flag preserved the _ort/ layout (Phase 7 Step 1 finding:
    we pass -L _ort to delocate-wheel so dylibs stay in _ort/).
    """
    from decibri._ort_resolver import resolve_ort_dylib_path

    path = resolve_ort_dylib_path(None)
    assert path is not None, (
        "Resolver returned None; bundled _ort/ directory may be empty "
        "or auditwheel/delocate moved the dylib out of _ort/"
    )


def test_async_construct() -> None:
    """AsyncMicrophone constructs cleanly (verifies pyo3-async-runtimes loads)."""
    d = decibri.AsyncMicrophone()
    assert d is not None


def test_numpy_construct() -> None:
    """Microphone(numpy=True) constructs cleanly (verifies rust-numpy bundling).

    Construction only; no actual read. The numpy=True flag exercises
    the rust-numpy code path's presence in the wheel; calling read()
    would also exercise it but requires audio hardware.
    """
    d = decibri.Microphone(numpy=True)
    assert d is not None


def test_wheel_size_within_budget() -> None:
    """The installed package's file enumeration succeeds (smoke).

    importlib.metadata.distribution('decibri').files enumerates the
    wheel's RECORD entries; the smoke proves the wheel's metadata is
    well-formed post-install and post-repair. Hard size enforcement
    happens in CI via python-ci.yml's wheel-size step (soft warn at
    150 MB, hard fail at 200 MB).
    """
    files = importlib.metadata.distribution("decibri").files
    assert files is not None
    assert len(files) > 0
