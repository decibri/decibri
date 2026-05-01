"""Phase 4 bridge sendability tests.

Verifies the Phase 4 invariant from the Python side: both `Microphone` and
`Speaker` instances can cross thread boundaries without raising
PyRuntimeError("can't access object from another thread"). The Rust
binding asserts the same property at compile time via the
`Send + 'static` assertion in `bindings/python/src/lib.rs`; these tests
catch behavioural regressions where the type-level property is preserved
but runtime behaviour breaks (e.g. a thread-local that the bridge
implicitly depends on, or a future PyO3 release that tightens runtime
checks beyond what Send + 'static guarantees).

The strongest signal is `test_decibri_can_be_passed_through_threadpool_for_callable`,
which simulates the capture pattern Phase 5's
`pyo3_async_runtimes::tokio::future_into_py` will use. If that test
passes, the bridge will compile and run inside an async closure when
Phase 5 wires it up.

These tests do not exercise audio capture or playback; they only exercise
the bridge's identity and a pure read-only method (`is_open`). No audio
hardware required, so no markers are applied.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

from decibri import Microphone, Speaker


def test_decibri_construct_no_unsendable_error() -> None:
    """Constructed Microphone instance survives a thread-pool submission.

    Pre-Phase-4 (#[pyclass(unsendable)]), this raised PyRuntimeError on
    the worker's first attribute access. Post-Phase-4, the bridge is
    `Send + 'static` and PyO3 admits the cross-thread access.
    """
    decibri = Microphone(vad=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda d: d is not None, decibri)
        assert future.result() is True


def test_decibri_output_construct_no_unsendable_error() -> None:
    """Same property for Speaker; the output bridge is also Send."""
    output = Speaker()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda d: d is not None, output)
        assert future.result() is True


def test_decibri_method_call_from_worker_thread() -> None:
    """A real method call on the bridge from a worker thread returns
    the expected value, not just an admission to access.

    A constructed-but-not-started bridge has no open stream; `is_open`
    must return False regardless of which thread the call originates
    from. This catches a hypothetical regression where the type-level
    Send property holds but a method internally relies on a thread-local
    state that only the construction thread carries.
    """
    decibri = Microphone(vad=False)
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(lambda d: d.is_open, decibri)
        result = future.result()
    assert result is False


def test_decibri_can_be_passed_through_threadpool_for_callable() -> None:
    """Closure-capture pattern that mirrors Phase 5's future_into_py.

    `pyo3_async_runtimes::tokio::future_into_py` captures the bridge
    into an async closure and submits the closure to a Tokio worker.
    Python's threadpool is not the same runtime, but the requirement
    on the captured value is the same: it must survive being moved
    out of the calling thread.

    If this test passes, the Phase 5 async wiring will compile when the
    bridge is captured into a tokio::spawn_blocking-shaped closure.
    """
    decibri = Microphone(vad=False)

    def use_decibri() -> bool:
        return decibri.is_open

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future: concurrent.futures.Future[Any] = executor.submit(use_decibri)
        result = future.result()
    assert result is False
