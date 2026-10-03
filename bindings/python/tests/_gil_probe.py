"""Checks that a call releases the GIL, by whether another thread runs during it.

Shared by the tests that check a call releases the GIL. A daemon thread waits
for the GIL and, each time it gets it, adds one to a count and releases the GIL
again through time.sleep(0), which releases and retakes it on every platform.
The thread therefore runs whenever the GIL is free and at no other time.

While a GilProbe is active the switch interval is SWITCH_INTERVAL_S, longer
than any test runs, so a thread waiting for the GIL never makes the thread
holding it give it up, and the GIL changes hands only where the thread holding
it releases it. The interval is set before the counting thread starts, and the
GIL is released before the first bracket, so every wait for the GIL that is
still in progress at a bracket began under that interval.

A bracket reads the count, makes exactly one call, and reads the count again.
Garbage is collected beforehand and the collector stays disabled until the
second read, so no finaliser runs in between. Nothing between the two reads
can release the GIL except the call itself, so the count advances during a
bracket only if the call released the GIL, and a call that holds the GIL
throughout leaves the count unchanged in every bracket, whatever the machine's
speed or load.

A test brackets BRACKETS calls of each kind it checks and passes when the count
advanced during at least one of them.
"""

from __future__ import annotations

import contextlib
import gc
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterator, TypeVar

T = TypeVar("T")

# Calls of each kind a test brackets. One advance of the count is enough; the
# other calls let the counting thread be seen when a busy machine keeps it off
# every processor for part of the test.
BRACKETS = 20
# The switch interval while a probe is active, in seconds. It is longer than
# any test runs, and inside the 4294 seconds CPython can hold on Windows, where
# it keeps the interval as a 32-bit count of microseconds.
SWITCH_INTERVAL_S = 1000.0
# Longest wait for the counting thread's first run.
_START_WAIT_S = 10.0


@dataclass
class Bracket:
    """One bracketed call: how many times the counting thread ran during it,
    and how long the call took."""

    runs: int
    ms: float


class GilProbe:
    """The counting thread, the switch interval and the bracket."""

    def __init__(self) -> None:
        self.count = 0
        self._stopping = False
        self._collector_paused = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._interval = sys.getswitchinterval()

    def _run(self) -> None:
        while not self._stopping:
            self.count += 1
            time.sleep(0)

    def __enter__(self) -> GilProbe:
        sys.setswitchinterval(SWITCH_INTERVAL_S)
        self._thread.start()
        # Sleeping releases the GIL, which lets the counting thread run and
        # wakes any thread already waiting for the GIL, whose next wait then
        # begins under the new interval.
        deadline = time.perf_counter() + _START_WAIT_S
        while self.count == 0 and time.perf_counter() < deadline:
            time.sleep(0.001)
        if self.count == 0:
            self.__exit__()
            raise AssertionError(
                f"the counting thread did not run within {_START_WAIT_S:.0f}s of starting"
            )
        return self

    def __exit__(self, *exc: object) -> None:
        self._stopping = True
        self._thread.join(timeout=5.0)
        sys.setswitchinterval(self._interval)

    @contextlib.contextmanager
    def collector_paused(self) -> Iterator[None]:
        """Collect garbage, then keep the collector disabled until the block
        ends. A bracket inside the block makes no collection of its own."""
        gc.collect()
        enabled = gc.isenabled()
        gc.disable()
        self._collector_paused = True
        try:
            yield
        finally:
            self._collector_paused = False
            if enabled:
                gc.enable()

    def bracket(self, call: Callable[..., object], *args: Any, **kwargs: Any) -> Bracket:
        """Bracket one call of `call` with `args` and `kwargs`."""
        return self._bracket(call, args, kwargs)[1]

    def bracketed(self, call: Callable[..., T], into: list[Bracket]) -> Callable[..., T]:
        """`call`, with each call through it bracketed and its bracket added to
        `into`, for a caller that makes the call itself."""

        def bracketed_call(*args: Any, **kwargs: Any) -> T:
            result, bracket = self._bracket(call, args, kwargs)
            into.append(bracket)
            return result

        return bracketed_call

    def _bracket(
        self, call: Callable[..., T], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[T, Bracket]:
        with contextlib.ExitStack() as stack:
            if not self._collector_paused:
                stack.enter_context(self.collector_paused())
            began = time.perf_counter()
            before = self.count
            result = call(*args, **kwargs)
            after = self.count
            ms = (time.perf_counter() - began) * 1000.0
        return result, Bracket(after - before, ms)


def assert_releases_gil(brackets: list[Bracket], what: str, holds: str) -> None:
    """Assert the counting thread ran during at least one bracketed call."""
    assert brackets, f"no call ({what}) was bracketed"
    lengths = sorted(b.ms for b in brackets)
    assert any(b.runs > 0 for b in brackets), (
        f"the counting thread ran during none of {len(brackets)} calls ({what}) "
        f"lasting {lengths[0]:.1f} to {lengths[-1]:.1f}ms; {holds}"
    )
