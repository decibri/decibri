"""Measures whether calls let other threads run while they work.

Shared by the tests that check a call releases the GIL. A daemon thread
increments a counter and sleeps 1ms, in a loop, at a raised scheduling priority
where the platform allows one, so its progress follows whether the GIL is free
rather than whether a core is. Timed GIL switching is off for the whole
measurement, so the GIL changes hands only where a thread blocks or a call
releases it. While a call releases the GIL the counter advances, and while a
call holds it the counter cannot.

Every measurement calibrates itself in the same run. After the calls, each
measured window is repeated as two controls of the same length: a sleep, which
releases the GIL, and a sort of a list, which holds it for its whole duration.
The calls pass when their ticks reach the sleep control's divided by RATIO and
RATIO times the sort control's, for the same total length. A call that holds
the GIL can still let the counter advance briefly where the code around it
releases the GIL, to check that a file exists for example. Those ticks grow
with the number of calls, not with their length, so the threshold is a constant
share of the sleep control's ticks. A test measures calls until a sleep is
expected to give EXPECTED_TICKS across them, so the windows hold enough ticks
to decide whatever the platform's sleep granularity.
"""

from __future__ import annotations

import ctypes
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable

# Ticks a sleep is expected to give across a test's measured calls.
EXPECTED_TICKS = 100.0
# The calls pass when their ticks reach the sleep control's divided by this,
# and this times the sort control's.
RATIO = 6.0
# Fewest ticks the sleep control may give across the calls for the comparison
# to decide.
_MIN_EXPECTED_TICKS = 50.0
# Ceiling on the total time a test measures calls for.
_MAX_MEASURED_MS = 5000.0
# The rate a test is sized from is measured over at least this many ticks, or
# for at most this long.
_RATE_TICKS = 20
_RATE_WAIT_S = 2.0
# Longest wait for the ticking thread's first tick.
_START_WAIT_S = 10.0
# Scheduling the ticking thread asks for. THREAD_PRIORITY_HIGHEST on Windows,
# QOS_CLASS_USER_INTERACTIVE on macOS.
_THREAD_PRIORITY_HIGHEST = 2
_QOS_CLASS_USER_INTERACTIVE = 0x21
# A switch interval no measurement reaches, so the GIL stays with the thread
# holding it until that thread blocks or releases it.
_NO_TIMED_SWITCHING = 10.0
# Length of the list the GIL-holding control sorts.
_SORT_ITEMS = 200_000


def _raise_priority() -> None:
    """Ask the platform to run the calling thread ahead of ordinary threads,
    where an unprivileged process may: the highest thread priority of the
    normal class on Windows and the user-interactive class on macOS. Elsewhere,
    or if the platform refuses, the thread keeps its priority."""
    try:
        if sys.platform == "win32":
            kernel32 = ctypes.WinDLL("kernel32")
            kernel32.GetCurrentThread.restype = ctypes.c_void_p
            kernel32.SetThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
            kernel32.SetThreadPriority.restype = ctypes.c_int
            kernel32.SetThreadPriority(kernel32.GetCurrentThread(), _THREAD_PRIORITY_HIGHEST)
        elif sys.platform == "darwin":
            libc = ctypes.CDLL(None)
            libc.pthread_set_qos_class_self_np.argtypes = [ctypes.c_uint, ctypes.c_int]
            libc.pthread_set_qos_class_self_np.restype = ctypes.c_int
            libc.pthread_set_qos_class_self_np(_QOS_CLASS_USER_INTERACTIVE, 0)
    except (AttributeError, OSError):
        pass


@dataclass
class Window:
    """The counter's advance during one measured window, and its length."""

    ticks: int
    ms: float


class GilMeter:
    """The ticking thread, the window measurement and the two controls."""

    def __init__(self) -> None:
        self.ticks = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._interval = sys.getswitchinterval()
        rng = random.Random(0)
        self._items = [rng.random() for _ in range(_SORT_ITEMS)]
        self._ms_per_item = 0.0

    def _run(self) -> None:
        _raise_priority()
        while not self._stop.is_set():
            self.ticks += 1
            time.sleep(0.001)

    def __enter__(self) -> GilMeter:
        sys.setswitchinterval(_NO_TIMED_SWITCHING)
        began = time.perf_counter()
        sorted(self._items)
        self._ms_per_item = (time.perf_counter() - began) * 1000.0 / _SORT_ITEMS
        self._thread.start()
        # No window may begin before the ticking thread has run once. A busy
        # machine can take a while to schedule a new thread.
        deadline = time.perf_counter() + _START_WAIT_S
        while self.ticks == 0 and time.perf_counter() < deadline:
            time.sleep(0.001)
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5.0)
        sys.setswitchinterval(self._interval)

    def measure(self, call: Callable[[], object]) -> Window:
        """Run `call` as one measured window."""
        before = self.ticks
        began = time.perf_counter()
        call()
        ms = (time.perf_counter() - began) * 1000.0
        return Window(self.ticks - before, ms)

    def target_ms(self) -> float:
        """The total call time across which a sleep gives EXPECTED_TICKS.

        The rate comes from sleeping in 50ms steps until the counter has
        advanced _RATE_TICKS times, or for _RATE_WAIT_S at most."""
        before = self.ticks
        began = time.perf_counter()
        while self.ticks - before < _RATE_TICKS and time.perf_counter() - began < _RATE_WAIT_S:
            time.sleep(0.05)
        ms = (time.perf_counter() - began) * 1000.0
        rate = (self.ticks - before) / ms
        # The sleep is the yardstick, so a stalled one makes every comparison
        # meaningless rather than merely generous.
        assert rate > 0, (
            f"background thread made no progress during a {ms:.0f}ms "
            "sleep; the baseline is unusable"
        )
        return min(EXPECTED_TICKS / rate, _MAX_MEASURED_MS)

    def released(self, ms: float) -> Window:
        """A control window of `ms` that releases the GIL: a sleep."""
        return self.measure(lambda: time.sleep(ms / 1000.0))

    def held(self, ms: float) -> Window:
        """A control window of about `ms` that holds the GIL: sorts of the list,
        each sized to the window, repeated until the window has passed."""
        count = max(1000, min(_SORT_ITEMS, int(ms / max(self._ms_per_item, 1e-9))))

        def hold() -> None:
            deadline = time.perf_counter() + ms / 1000.0
            while True:
                sorted(self._items[:count])
                if time.perf_counter() >= deadline:
                    break

        return self.measure(hold)


def measure_calls(
    meter: GilMeter,
    next_call: Callable[[int], Callable[[], object]],
    *,
    minimum: int = 1,
    maximum: int = 200,
) -> list[Window]:
    """Measure calls until they are long enough to decide.

    `next_call(i)` prepares the i-th call outside its window and returns it.
    Calls are measured, 20ms apart, until their total length reaches the
    meter's target, with at least `minimum` and at most `maximum` of them.
    """
    target = meter.target_ms()
    windows: list[Window] = []
    while len(windows) < minimum or (
        sum(w.ms for w in windows) < target and len(windows) < maximum
    ):
        call = next_call(len(windows))
        windows.append(meter.measure(call))
        time.sleep(0.02)
    return windows


def assert_releases_gil(meter: GilMeter, windows: list[Window], what: str, holds: str) -> None:
    """Assert the measured calls let the counter advance like a sleep of the
    same length does, and clearly unlike a GIL-holding sort of that length."""
    calls_ticks = sum(w.ticks for w in windows)
    calls_ms = sum(w.ms for w in windows)
    released = [meter.released(w.ms) for w in windows]
    held = [meter.held(w.ms) for w in windows]
    sleep_ticks = sum(w.ticks for w in released) * calls_ms / sum(w.ms for w in released)
    sort_ticks = sum(w.ticks for w in held) * calls_ms / sum(w.ms for w in held)
    summary = (
        f"{len(windows)} calls ({what}) totalling {calls_ms:.1f}ms advanced the "
        f"background thread {calls_ticks} ticks, where sleeps of the same lengths "
        f"give {sleep_ticks:.1f} and GIL-holding sorts give {sort_ticks:.1f}"
    )
    assert sleep_ticks >= _MIN_EXPECTED_TICKS, (
        f"{summary}; too few ticks for the comparison to decide"
    )
    assert sleep_ticks >= RATIO * RATIO * max(sort_ticks, 1.0), (
        f"{summary}; the two controls do not separate in this run"
    )
    threshold = max(sleep_ticks / RATIO, RATIO * max(sort_ticks, 1.0))
    assert calls_ticks >= threshold, f"{summary}, against a threshold of {threshold:.1f}; {holds}"
