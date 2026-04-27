"""Pytest configuration for the decibri Python binding test suite.

Registers three hardware-required markers and auto-skips them in CI.

Markers:
    requires_audio_input: test reads real audio frames from a microphone.
    requires_audio_output: test writes real audio frames to a speaker.
    requires_bundled_ort: test constructs Decibri(vad=True, vad_mode='silero',
        ...). Independent of audio markers; gates ORT runtime availability,
        which is a separate precondition from microphone/speaker hardware.

Auto-skip semantics: when the CI environment variable is set (any non-empty
value), all hardware-marked tests are skipped. Local developer runs (CI
unset) execute every test.
"""

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    """Register the three hardware markers."""
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
        "requires_bundled_ort: test needs the bundled Silero model + ORT runtime (auto-skipped in CI)",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Auto-skip hardware-marked tests when running in CI."""
    if not os.environ.get("CI"):
        return
    skip_audio_input = pytest.mark.skip(reason="no audio input device in CI")
    skip_audio_output = pytest.mark.skip(reason="no audio output device in CI")
    skip_bundled_ort = pytest.mark.skip(reason="ORT runtime not available in CI")
    for item in items:
        if "requires_audio_input" in item.keywords:
            item.add_marker(skip_audio_input)
        if "requires_audio_output" in item.keywords:
            item.add_marker(skip_audio_output)
        if "requires_bundled_ort" in item.keywords:
            item.add_marker(skip_bundled_ort)
