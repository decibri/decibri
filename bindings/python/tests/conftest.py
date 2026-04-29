"""Pytest configuration for the decibri Python binding test suite.

Registers three precondition markers with two distinct auto-skip policies.

Markers:
    requires_audio_input: test reads real audio frames from a microphone.
    requires_audio_output: test writes real audio frames to a speaker.
    requires_bundled_ort: test exercises the bundled ONNX Runtime dylib
        in decibri/_ort/, populated at wheel build time by python-ci.yml's
        download step. Independent of audio markers; gates ORT runtime
        availability, which is a separate precondition from
        microphone/speaker hardware.

Auto-skip semantics:
    Audio hardware markers (requires_audio_input, requires_audio_output):
        Skipped when the CI environment variable is set. CI runners have
        no real audio devices; local dev boxes do.

    Bundled ORT marker (requires_bundled_ort):
        Skipped when the bundled _ort/ directory is missing or contains
        no platform-appropriate dylib (libonnxruntime*.so on Linux,
        libonnxruntime*.dylib on macOS, onnxruntime*.dll on Windows). This
        is the inverse of "skip in CI": Phase 3 CI populates _ort/ before
        running tests, so these tests RUN in CI and SKIP on dev installs
        where maturin develop leaves _ort/ empty. The check inspects the
        installed package via importlib.resources, so editable installs
        and wheel installs both go through the same code path.
"""

from __future__ import annotations

import importlib.resources
import os
import sys
from pathlib import Path

import pytest

_PLATFORM_PATTERN: dict[str, str] = {
    "linux": "libonnxruntime*.so",
    "darwin": "libonnxruntime*.dylib",
    "win32": "onnxruntime*.dll",
}


def _bundled_ort_available() -> bool:
    """Return True iff decibri/_ort/ contains a platform-appropriate dylib.

    Mirrors the resolver's enumeration approach in _ort_resolver.py so the
    skip decision matches the resolver's would-it-resolve outcome. Defensive
    against the package not being importable yet (rare; a sibling test that
    breaks the import would surface that separately).
    """
    pattern = _PLATFORM_PATTERN.get(sys.platform)
    if pattern is None:
        return False
    try:
        ort_dir = importlib.resources.files("decibri") / "_ort"
        if not ort_dir.is_dir():
            return False
        return any(
            child.is_file() and Path(child.name).match(pattern)
            for child in ort_dir.iterdir()
        )
    except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError):
        return False


def pytest_configure(config: pytest.Config) -> None:
    """Register the three precondition markers."""
    config.addinivalue_line(
        "markers",
        "requires_audio_input: test needs a real audio input device (auto-skipped in CI)",
    )
    config.addinivalue_line(
        "markers",
        "requires_audio_output: test needs a real audio output device (auto-skipped in CI)",
    )
    config.addinivalue_line(
        "markers",
        "requires_bundled_ort: test needs decibri/_ort/ populated with the platform dylib "
        "(auto-skipped on dev installs where _ort/ is empty)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Apply the two auto-skip policies described in the module docstring."""
    in_ci = bool(os.environ.get("CI"))
    skip_audio_input = pytest.mark.skip(reason="no audio input device in CI")
    skip_audio_output = pytest.mark.skip(reason="no audio output device in CI")
    skip_bundled_ort = pytest.mark.skip(
        reason="_ort directory not bundled (dev install or platform mismatch)"
    )

    bundled_available = _bundled_ort_available()

    for item in items:
        if in_ci and "requires_audio_input" in item.keywords:
            item.add_marker(skip_audio_input)
        if in_ci and "requires_audio_output" in item.keywords:
            item.add_marker(skip_audio_output)
        if (not bundled_available) and "requires_bundled_ort" in item.keywords:
            item.add_marker(skip_bundled_ort)
