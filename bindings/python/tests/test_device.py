"""Device selector tests.

``Device`` carries a device's stable per-host identifier and is passed on
the ``device`` parameter beside the index and name-substring forms. The
object's construction, its export, its acceptance on every constructor and
the refusal of any other shape need no hardware. Resolving an identifier
against a real device does, so those tests carry the audio markers and CI
auto-skips them.
"""

from __future__ import annotations

import dataclasses
import inspect
import typing

import pytest

import decibri
from decibri import (
    AsyncMicrophone,
    AsyncSpeaker,
    Device,
    Microphone,
    MicrophoneNotFound,
    Speaker,
    SpeakerNotFound,
)

_REFUSAL = "device must be None, int, str, or Device"


# ---------------------------------------------------------------------------
# The object itself
# ---------------------------------------------------------------------------


def test_device_constructs_positionally_and_by_keyword() -> None:
    """Both spellings carry the identifier through unchanged."""
    assert Device("wasapi:{x}").id == "wasapi:{x}"
    assert Device(id="coreaudio:uid").id == "coreaudio:uid"


def test_device_is_frozen() -> None:
    """The selector is immutable, like the Vad and Aec config objects."""
    selector = Device("alsa:hw:0,0")
    with pytest.raises(dataclasses.FrozenInstanceError):
        selector.id = "other"  # type: ignore[misc]


@pytest.mark.parametrize(
    "bad_id",
    [
        pytest.param(3, id="int"),
        pytest.param(None, id="none"),
        pytest.param(b"wasapi:{x}", id="bytes"),
    ],
)
def test_device_rejects_non_string_id(bad_id: object) -> None:
    """A non-string identifier is refused at construction with TypeError."""
    with pytest.raises(TypeError, match="id must be a string"):
        Device(bad_id)  # type: ignore[arg-type]


def test_device_exported_from_package_root() -> None:
    """``from decibri import Device`` resolves to the one class and is listed."""
    assert decibri.Device is Device
    assert "Device" in decibri.__all__


# ---------------------------------------------------------------------------
# Acceptance on every constructor (device resolution happens at start())
# ---------------------------------------------------------------------------


def test_every_constructor_accepts_a_device_object() -> None:
    """All four classes construct with a Device and keep it for repr."""
    selector = Device("host:never-resolved")
    for obj in (
        Microphone(device=selector),
        AsyncMicrophone(device=selector),
        Speaker(device=selector),
        AsyncSpeaker(device=selector),
    ):
        assert "device=Device(id='host:never-resolved')" in repr(obj)


@pytest.mark.parametrize(
    "target",
    [
        pytest.param(Microphone.__init__, id="Microphone"),
        pytest.param(AsyncMicrophone.__init__, id="AsyncMicrophone"),
        pytest.param(Speaker.__init__, id="Speaker"),
        pytest.param(AsyncSpeaker.__init__, id="AsyncSpeaker"),
        pytest.param(decibri.record_to_file, id="record_to_file"),
        pytest.param(decibri.async_record_to_file, id="async_record_to_file"),
    ],
)
def test_device_parameter_is_annotated_with_device(target: object) -> None:
    """Every ``device`` parameter names Device in its annotation.

    The stub side of the same contract is held by ``mypy --strict``.
    """
    assert callable(target)
    assert "device" in inspect.signature(target).parameters
    assert typing.get_type_hints(target)["device"] == int | str | Device | None


# ---------------------------------------------------------------------------
# Refusal of any other shape
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_device",
    [
        pytest.param({"id": "wasapi:{x}"}, id="mapping"),
        pytest.param(3.5, id="float"),
        pytest.param(["wasapi:{x}"], id="list"),
    ],
)
def test_wrong_shape_is_refused_on_capture_and_playback(bad_device: object) -> None:
    """A value of any other type raises ValueError naming the four forms."""
    with pytest.raises(ValueError, match=_REFUSAL):
        Microphone(device=bad_device)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match=_REFUSAL):
        Speaker(device=bad_device)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Resolution against real devices
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
def test_microphone_selects_every_input_device_by_id() -> None:
    """Each enumerated input device opens through its own identifier.

    Catches the identifier never reaching the core as its own selector:
    read as a name it matches nothing and start() raises.
    """
    devices = Microphone.devices()
    assert devices
    for info in devices:
        mic = Microphone(device=Device(id=info.id))
        mic.start()
        try:
            assert mic.is_open is True
        finally:
            mic.stop()


@pytest.mark.requires_audio_input
def test_microphone_id_is_matched_exactly_not_as_a_name() -> None:
    """A display name wrapped in Device is not an identifier and is not found.

    Catches the object collapsing into the name-substring path, which would
    resolve (or report an ambiguous name) instead of raising not found.
    """
    info = Microphone.devices()[0]
    mic = Microphone(device=Device(id=info.name))
    with pytest.raises(MicrophoneNotFound):
        mic.start()


@pytest.mark.requires_audio_output
def test_speaker_selects_every_output_device_by_id() -> None:
    """Each enumerated output device opens through its own identifier."""
    devices = Speaker.devices()
    assert devices
    for info in devices:
        spk = Speaker(device=Device(id=info.id))
        spk.start()
        try:
            assert spk.is_playing is True
        finally:
            spk.stop()


@pytest.mark.requires_audio_output
def test_speaker_id_is_matched_exactly_not_as_a_name() -> None:
    """A display name wrapped in Device is not found on the output side."""
    info = Speaker.devices()[0]
    spk = Speaker(device=Device(id=info.name))
    with pytest.raises(SpeakerNotFound):
        spk.start()
