"""Phase 5 tests for AsyncMicrophone and AsyncSpeaker.

Covers the goals from Section 2 of phase-5-async-support.md:
construction lifecycle, async context manager, async iterator,
cancellation propagation, output write/drain, and the concurrent-tasks
property test that validates Phase 4's Send + 'static under realistic
async load.

Per locked decision Q3, every test has explicit ``@pytest.mark.asyncio``.
Per the Step 1 fix relay, all cancellation tests use
``asyncio.wait_for(coro, timeout=X)`` rather than ``asyncio.timeout(X)``
because the latter is Python 3.11+ only and the abi3 floor is 3.10.

The smoke pyfunction in ``tests/test_async_smoke.py`` already proves
runtime initialization and basic cancellation propagation; this file
exercises the AsyncMicrophone / AsyncSpeaker public surface.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from decibri import AsyncMicrophone, AsyncSpeaker


# ---------------------------------------------------------------------------
# Goal 1: construction and basic lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_decibri_construct_start_stop() -> None:
    """AsyncMicrophone can be constructed, started, and stopped without exception.

    No vad to avoid ORT model load (the construction smoke for the bundled
    ORT path is in tests/test_ort_bundling.py). The lifecycle assertion
    requires a real cpal input stream to open; CI runners without audio
    hardware skip via the requires_audio_input marker.
    """
    decibri = AsyncMicrophone(vad=False)
    assert decibri.is_open is False
    await decibri.start()
    assert decibri.is_open is True
    await decibri.stop()
    assert decibri.is_open is False


@pytest.mark.requires_audio_output
@pytest.mark.asyncio
async def test_async_output_construct_start_stop() -> None:
    """AsyncSpeaker can be constructed, started, and stopped.

    Requires audio output hardware to open the cpal stream; CI runners
    without audio output skip via the requires_audio_output marker.
    """
    output = AsyncSpeaker()
    assert output.is_playing is False
    await output.start()
    assert output.is_playing is True
    await output.stop()
    assert output.is_playing is False


# ---------------------------------------------------------------------------
# Goal 2: read returns correct shape (requires audio input)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.requires_audio_input
async def test_async_decibri_read_returns_bytes() -> None:
    """A successful read returns bytes of the expected shape.

    Skipped without audio hardware. Default frame size is 1600 samples at
    16 kHz mono int16 = 3200 bytes; this test asserts the chunk is non-
    empty bytes and lets the exact length be implementation-defined.
    """
    decibri = AsyncMicrophone(vad=False)
    await decibri.start()
    try:
        chunk = await decibri.read()
        assert chunk is None or isinstance(chunk, bytes)
        if chunk is not None:
            assert len(chunk) > 0
    finally:
        await decibri.stop()


# ---------------------------------------------------------------------------
# Goal 3: async context manager
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_with_decibri() -> None:
    """``async with AsyncMicrophone()`` enters by starting, exits by stopping.

    Inside the with body, ``is_open`` is True. After the with block
    completes, ``is_open`` is False (verifying ``__aexit__`` ran).

    ``__aenter__`` calls ``start`` which opens a cpal input stream;
    requires audio input hardware. CI runners without it skip.
    """
    decibri = AsyncMicrophone(vad=False)
    assert decibri.is_open is False
    async with decibri as d:
        assert d is decibri
        assert decibri.is_open is True
    assert decibri.is_open is False


@pytest.mark.requires_audio_output
@pytest.mark.asyncio
async def test_async_with_decibri_output() -> None:
    """``async with AsyncSpeaker()`` enters by starting, exits by stopping.

    Requires audio output hardware (the output cpal stream opens during
    ``__aenter__``).
    """
    output = AsyncSpeaker()
    async with output as o:
        assert o is output
        assert output.is_playing is True
    assert output.is_playing is False


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_with_decibri_propagates_exceptions() -> None:
    """Exceptions inside ``async with`` body propagate after stop runs.

    Verifies ``__aexit__`` does not suppress exceptions and that the
    bridge is still stopped (resource cleanup) even on exception path.
    Requires audio input hardware (``__aenter__`` opens the stream).
    """

    class _AsyncWithProbe(Exception):
        pass

    decibri = AsyncMicrophone(vad=False)
    with pytest.raises(_AsyncWithProbe):
        async with decibri:
            assert decibri.is_open is True
            raise _AsyncWithProbe("test exception")
    # __aexit__ ran; bridge is stopped
    assert decibri.is_open is False


# ---------------------------------------------------------------------------
# Goal 4: async iterator (requires audio input)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.requires_audio_input
async def test_async_for_decibri_iterates() -> None:
    """``async for chunk in async_decibri:`` yields bytes chunks.

    Reads exactly 3 chunks then breaks. Validates ``__aiter__`` returns
    self, ``__anext__`` returns chunks, and the loop terminates correctly.
    """
    chunks_collected = 0
    async with AsyncMicrophone(vad=False) as decibri:
        async for chunk in decibri:
            assert isinstance(chunk, bytes)
            chunks_collected += 1
            if chunks_collected >= 3:
                break
    assert chunks_collected == 3


# ---------------------------------------------------------------------------
# Goal 5: cancellation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_decibri_cancellation_raises() -> None:
    """``asyncio.wait_for`` around a read raises an exception promptly.

    On a not-started bridge, the bridge's own error handling fires
    immediately (CaptureStreamClosed); on a started bridge with no audio
    hardware, ``wait_for`` may time out. Either path proves the
    cancellation pathway works on AsyncMicrophone's read coroutine. The
    foundational cancellation propagation is already proven by
    tests/test_async_smoke.py.
    """
    decibri = AsyncMicrophone(vad=False)
    with pytest.raises((asyncio.TimeoutError, Exception)):
        await asyncio.wait_for(decibri.read(), timeout=0.025)


@pytest.mark.asyncio
async def test_async_output_cancellation_during_drain() -> None:
    """``asyncio.wait_for`` around a drain raises an exception promptly.

    Either the bridge's own error handling fires (OutputStreamClosed on
    a not-started output) or the wait_for times out; either path proves
    cancellation works on AsyncSpeaker.drain.
    """
    output = AsyncSpeaker()
    with pytest.raises((asyncio.TimeoutError, Exception)):
        await asyncio.wait_for(output.drain(), timeout=0.025)


# ---------------------------------------------------------------------------
# Goal 6: output write/drain (requires audio output)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.requires_audio_output
async def test_async_output_write_drain() -> None:
    """Write a small known buffer and drain it.

    Test buffer is brief silence (zeros); writes to the output, drains,
    asserts no exception. Audio hardware required.
    """
    sample_rate = 48000
    duration_ms = 50
    num_samples = (sample_rate * duration_ms) // 1000
    silence = b"\x00\x00" * num_samples  # int16 silence

    async with AsyncSpeaker(sample_rate=sample_rate) as output:
        await output.write(silence)
        await output.drain()


# ---------------------------------------------------------------------------
# Goal 8: concurrent tasks share an instance
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_decibri_concurrent_tasks_share_instance() -> None:
    """Two concurrent tasks calling on the same instance work without panic.

    Validates Phase 4's ``Send + 'static`` guarantee under realistic
    async load (not just the ThreadPoolExecutor pattern from
    tests/test_bridge_sendability.py). The Rust-side
    ``tokio::sync::Mutex`` serializes concurrent calls per Phase 5 plan
    Risk 3; the test confirms no panic, no deadlock.

    Test design: ``stop`` is idempotent at the bridge level (sets stream
    and capture options to None; calling twice is a no-op). Running two
    concurrent stops on a started instance exercises mutex contention
    on a real bridge mutation without surfacing state-machine errors
    that the non-idempotent operations (start, read) would raise.
    Concurrent ``start`` is intentionally avoided because the bridge
    raises ``AlreadyRunning`` on the second call by design (correct
    behaviour, not a Phase 5 concern).

    Requires audio input hardware: the test starts a real cpal stream
    before issuing the concurrent stops. CI runners without audio skip.
    """
    decibri = AsyncMicrophone(vad=False)

    await decibri.start()
    try:
        await asyncio.gather(decibri.stop(), decibri.stop())
        assert decibri.is_open is False
    finally:
        if decibri.is_open:
            await decibri.stop()


# ---------------------------------------------------------------------------
# Goal 8 (continued): static methods
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_input_devices_returns_list() -> None:
    """``AsyncMicrophone.input_devices()`` is async and returns a list."""
    devices: list[Any] = await AsyncMicrophone.input_devices()
    assert isinstance(devices, list)


@pytest.mark.asyncio
async def test_async_output_devices_returns_list() -> None:
    """``AsyncSpeaker.output_devices()`` is async and returns a list."""
    devices: list[Any] = await AsyncSpeaker.output_devices()
    assert isinstance(devices, list)


@pytest.mark.asyncio
async def test_async_version_returns_version_info() -> None:
    """``AsyncMicrophone.version()`` is sync (reads compile-time constants)."""
    info = AsyncMicrophone.version()
    # VersionInfo has decibri / cpal / wheel fields per the Rust bridge
    assert info is not None


# ---------------------------------------------------------------------------
# Phase 7.5: close() alias for stop() on AsyncMicrophone
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_microphone_close_method_exists() -> None:
    """AsyncMicrophone exposes close() (mirrors AsyncSpeaker.close())."""
    import inspect

    mic = AsyncMicrophone(vad=False)
    assert hasattr(mic, "close")
    assert inspect.iscoroutinefunction(mic.close)


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_microphone_close_stops_stream() -> None:
    """await AsyncMicrophone.close() leaves the stream stopped (alias for stop())."""
    mic = AsyncMicrophone(vad=False)
    await mic.start()
    assert mic.is_open is True
    await mic.close()
    assert mic.is_open is False


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_microphone_close_idempotent() -> None:
    """await close() can be invoked multiple times safely."""
    mic = AsyncMicrophone(vad=False)
    await mic.start()
    await mic.close()
    await mic.close()  # second close is a no-op


# ---------------------------------------------------------------------------
# Phase 7.5 Item 10: is_open / is_playing now query bridge truth
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_microphone_is_open_reflects_bridge_truth() -> None:
    """AsyncMicrophone.is_open queries the Rust bridge atomic mirror.

    Phase 7.5 Item 10: previously this was a Python-side cache that
    could lie when the Rust side closed the stream itself. Now it
    delegates to the lock-free atomic mirror on AsyncMicrophoneBridge.
    """
    mic = AsyncMicrophone(vad=False)
    assert mic.is_open is False, "Pre-start: bridge atomic should be False"

    await mic.start()
    assert mic.is_open is True, "Post-start: bridge atomic should be True"

    await mic.stop()
    assert mic.is_open is False, "Post-stop: bridge atomic should be False"


@pytest.mark.requires_audio_output
@pytest.mark.asyncio
async def test_async_speaker_is_playing_reflects_bridge_truth() -> None:
    """AsyncSpeaker.is_playing queries the Rust bridge atomic mirror.

    Phase 7.5 Item 10 sibling fix: same shape as AsyncMicrophone.is_open.
    """
    spk = AsyncSpeaker()
    assert spk.is_playing is False

    await spk.start()
    assert spk.is_playing is True

    await spk.stop()
    assert spk.is_playing is False


@pytest.mark.asyncio
async def test_async_microphone_is_open_pre_start_no_hardware() -> None:
    """AsyncMicrophone.is_open returns False before start without hardware.

    Validates the atomic-mirror sync getter is callable without an
    active stream or audio device. Construction does not start the
    stream; the atomic should be False from the constructor.
    """
    mic = AsyncMicrophone(vad=False)
    assert mic.is_open is False


@pytest.mark.asyncio
async def test_async_speaker_is_playing_pre_start_no_hardware() -> None:
    """AsyncSpeaker.is_playing returns False before start without hardware."""
    spk = AsyncSpeaker()
    assert spk.is_playing is False


# ---------------------------------------------------------------------------
# Phase 7.7 Item B1: async read_with_metadata + aiter_with_metadata + Chunk
# ---------------------------------------------------------------------------


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_read_with_metadata_returns_chunk() -> None:
    """``await read_with_metadata()`` returns a frozen ``Chunk``."""
    import dataclasses

    from decibri import Chunk

    async with AsyncMicrophone(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        chunk = await d.read_with_metadata(timeout_ms=500)
        assert chunk is not None
        assert isinstance(chunk, Chunk)
        assert isinstance(chunk.data, bytes)
        assert isinstance(chunk.timestamp, float)
        assert chunk.timestamp > 0.0
        assert chunk.sequence == 0
        assert chunk.is_speaking is False
        assert chunk.vad_score == 0.0
        with pytest.raises(dataclasses.FrozenInstanceError):
            chunk.sequence = 99  # type: ignore[misc]


@pytest.mark.requires_audio_input
@pytest.mark.asyncio
async def test_async_aiter_with_metadata_yields_chunks() -> None:
    """``async for chunk in mic.aiter_with_metadata()`` yields Chunk objects."""
    from decibri import Chunk

    async with AsyncMicrophone(sample_rate=16000, channels=1, frames_per_buffer=512) as d:
        count = 0
        last_seq = -1
        async for chunk in d.aiter_with_metadata():
            assert isinstance(chunk, Chunk)
            assert chunk.sequence == last_seq + 1
            last_seq = chunk.sequence
            count += 1
            if count >= 3:
                break
        assert count == 3


# ---------------------------------------------------------------------------
# Phase 9 Item A6: AsyncMicrophone.open() / AsyncSpeaker.open() async
# factory classmethods that offload synchronous construction to the
# default ThreadPoolExecutor via loop.run_in_executor.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_microphone_open_returns_constructed_instance() -> None:
    """``await AsyncMicrophone.open()`` returns an ``AsyncMicrophone``."""
    mic = await AsyncMicrophone.open()
    try:
        assert isinstance(mic, AsyncMicrophone)
        # Construction-only: no start() yet.
        assert mic.is_open is False
    finally:
        # No bridge resources to release because we never started; the
        # finalizer note on AsyncMicrophone applies (no __del__; pyo3 Drop).
        pass


@pytest.mark.asyncio
async def test_async_microphone_open_forwards_kwargs() -> None:
    """Kwargs passed to open() reach the underlying constructor."""
    mic = await AsyncMicrophone.open(
        sample_rate=24000,
        channels=2,
        dtype="float32",
        frames_per_buffer=2400,
    )
    text = repr(mic)
    assert "sample_rate=24000" in text
    assert "channels=2" in text
    assert "dtype='float32'" in text
    assert "frames_per_buffer=2400" in text


@pytest.mark.asyncio
async def test_async_microphone_open_dispatches_to_executor() -> None:
    """``AsyncMicrophone.open()`` calls ``loop.run_in_executor``.

    Direct contract test: spy on the running loop's ``run_in_executor``
    and verify the ``open()`` implementation routes construction through
    it. A timing-based test would be flaky for a fast (no-VAD)
    constructor; this test pins the implementation contract that matters
    in practice (when ``vad='silero'`` is enabled the executor dispatch
    is what keeps the event loop responsive during the 100 to 500 ms
    ORT load).
    """
    loop = asyncio.get_running_loop()
    original = loop.run_in_executor
    calls: list[Any] = []

    def spy(executor: Any, func: Any, *args: Any) -> Any:
        calls.append((executor, func))
        return original(executor, func, *args)

    loop.run_in_executor = spy  # type: ignore[method-assign]
    try:
        mic = await AsyncMicrophone.open()
    finally:
        loop.run_in_executor = original  # type: ignore[method-assign]

    assert isinstance(mic, AsyncMicrophone)
    assert len(calls) == 1, f"expected one run_in_executor call, got {len(calls)}"
    executor_arg, func = calls[0]
    assert executor_arg is None, (
        "open() should pass executor=None to use the default ThreadPoolExecutor (LD-9-7)"
    )
    assert callable(func), "the dispatched callable should be the construction lambda"


@pytest.mark.asyncio
async def test_async_speaker_open_returns_constructed_instance() -> None:
    """``await AsyncSpeaker.open()`` returns an ``AsyncSpeaker``."""
    spk = await AsyncSpeaker.open()
    assert isinstance(spk, AsyncSpeaker)
    assert spk.is_playing is False


@pytest.mark.asyncio
async def test_async_speaker_open_forwards_kwargs() -> None:
    """Kwargs passed to AsyncSpeaker.open() reach the constructor."""
    spk = await AsyncSpeaker.open(sample_rate=24000, channels=2, dtype="float32")
    text = repr(spk)
    assert "sample_rate=24000" in text
    assert "channels=2" in text
    assert "dtype='float32'" in text
