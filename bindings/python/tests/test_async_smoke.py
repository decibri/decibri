"""Phase 5 Step 1 empirical smoke tests for pyo3-async-runtimes.

Verifies two preconditions for the AsyncDecibri / AsyncDecibriOutput
surface that lands in subsequent Phase 5 relays:

1. The Tokio runtime initializes correctly in our extension-module
   context. ``pyo3-async-runtimes`` documents
   ``#[pyo3_async_runtimes::tokio::main]`` as the entry-point pattern,
   but Decibri is a cdylib (Python extension module) with no main()
   to attribute. We rely on the crate's lazy-init path.

2. ``asyncio.CancelledError`` (and its ``TimeoutError`` subclass)
   propagate correctly from Python through ``pyo3-async-runtimes``
   to the underlying Tokio future. This is the foundation Phase 5's
   abort-immediately cancellation policy is built on (locked decision
   Q3 of phase-5-async-support.md).

These tests stay as persistent regression coverage (locked decision Q4):
if a future ``pyo3-async-runtimes`` upgrade breaks runtime init or
cancellation propagation, these tests fail first and surface the
regression at the build-pipeline level rather than at AsyncDecibri
integration time.

The smoke pyfunction lives at ``decibri._decibri._async_smoke`` and
returns 42 after a 50 ms ``tokio::time::sleep``. The 25 ms timeout in
the cancellation test sits comfortably below the 50 ms sleep duration
to avoid flaky CI on cold-start runners.
"""

from __future__ import annotations

import asyncio

import pytest

from decibri._decibri import _async_smoke  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_async_smoke_returns_expected_value() -> None:
    """Tokio runtime is reachable and ``future_into_py`` resolves correctly.

    Establishes the baseline: a Rust async block (50 ms tokio sleep
    followed by ``Ok(42_i64)``) is awaitable from Python and yields the
    expected value. If this test fails with a runtime-not-initialised
    error, the lazy-init assumption is wrong and Phase 5's plan needs
    revision before any AsyncDecibri code lands.
    """
    result = await _async_smoke()
    assert result == 42, f"Expected 42, got {result!r}"


@pytest.mark.asyncio
async def test_async_smoke_cancellation_propagates() -> None:
    """``asyncio.timeout`` raises ``TimeoutError`` before the 50 ms sleep completes.

    ``TimeoutError`` is a subclass of ``asyncio.CancelledError`` in
    Python 3.11+, so this test exercises the locked abort-immediately
    cancellation pathway. If this fails (the future runs to completion
    despite the timeout, or hangs, or raises a different exception),
    Phase 5's cancellation design needs revision before the full async
    surface lands.

    Note: per Phase 5 plan Risk 2, ``tokio::task::spawn_blocking`` does
    not cooperatively cancel the underlying OS thread; the Python side
    sees ``CancelledError`` immediately while the Rust thread continues.
    This smoke test exercises only the Python-visible behaviour, which
    is what the abort-immediately design contracts to.
    """
    with pytest.raises(asyncio.TimeoutError):
        async with asyncio.timeout(0.025):
            await _async_smoke()
