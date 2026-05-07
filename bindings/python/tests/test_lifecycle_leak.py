"""Soak / leak detection tests for the Microphone lifecycle (0.1.3 Tier 1).

Bounded soak loops with tracemalloc + psutil RSS sampling. Catches leaks
in the binding's wrapper layer (Python-allocated; tracemalloc) and in
native allocations underneath cpal / pyo3 / ort (RSS via psutil).

Tier classification: Tier 1 ("ship pre-0.2.0"; cheap; high-leverage) per
prerelease-decibri-additional-testing-reqs.md Test 4. Bounded suite ships
in 0.1.3; multi-hour soaks (Tier 3) defer past 0.2.0.

Test design:

- 5-iteration warmup, 15-iteration measurement window. Warmup settles the
  interpreter's allocator caches; the measurement window detects monotonic
  growth past steady state. Total 20 iterations per test.
- tracemalloc tracks Python-side allocations (pyo3 wrapper objects, the
  exception hierarchy's class machinery, dataclass instances). A real
  Python leak would compound to several MB across 15 iterations.
- psutil RSS tracks native allocations the Python interpreter cannot see
  (cpal stream buffers, ORT session arenas, alsa-lib internals on Linux).
  RSS thresholds are wider (5 MB) because the OS allocator's high-water
  mark drifts naturally on busy systems.
- gc.collect() fires before each snapshot to reclaim cycle-collected
  objects deterministically; otherwise tracemalloc deltas would include
  uncollected cycles as if they were leaks.

Runtime budget:

- test_microphone_construct_destroy: ~5 seconds (no ORT load)
- test_async_microphone_construct_destroy: ~5 seconds (no ORT load)
- test_silero_vad_construct_destroy: ~10 seconds (ORT init dominates)

Combined target: under 30 seconds. Combined with test_microphone_properties.py
the 0.1.3 Tier 1 testing addition stays under the relay's 90-second budget.

Threshold rationale:

- 1 MB tracemalloc growth across 15 iterations is the noise ceiling
  observed across pyo3-style projects; a real wrapper-layer leak
  reproduces consistently above this.
- 5 MB RSS growth allows for OS allocator high-water-mark drift while
  still flagging the canonical leak shape (per-iteration multi-MB native
  allocation that never returns to the heap).
"""

from __future__ import annotations

import asyncio
import gc
import tracemalloc

import psutil
import pytest

from decibri import AsyncMicrophone, Microphone


_WARMUP_ITERATIONS = 5
_MEASUREMENT_ITERATIONS = 15
_TRACEMALLOC_THRESHOLD_MB = 1.0
_RSS_THRESHOLD_MB = 5.0


def _tracemalloc_total_growth_mb(
    snapshot_before: tracemalloc.Snapshot,
    snapshot_after: tracemalloc.Snapshot,
) -> float:
    """Return positive-only allocation growth across the two snapshots, in MB."""
    stats = snapshot_after.compare_to(snapshot_before, "filename")
    growth_bytes = sum(stat.size_diff for stat in stats if stat.size_diff > 0)
    return growth_bytes / 1024 / 1024


def _rss_mb() -> float:
    """Current process RSS in MB."""
    return psutil.Process().memory_info().rss / 1024 / 1024


# ---------------------------------------------------------------------------
# Section 1: synchronous Microphone lifecycle.
#
# 20 cycles of construct + stop + del. Microphone's wrapper does not start
# the cpal stream until .start() is called, so this loop exercises the
# wrapper layer's allocation / cleanup path without touching real audio
# hardware (CI-safe).
# ---------------------------------------------------------------------------


def test_microphone_construct_destroy_no_growth() -> None:
    """20 construct/destroy cycles must not leak Python or native memory."""
    tracemalloc.start()
    try:
        for _ in range(_WARMUP_ITERATIONS):
            mic = Microphone()
            mic.stop()
            del mic
        gc.collect()
        snapshot_warmup = tracemalloc.take_snapshot()
        rss_warmup_mb = _rss_mb()

        for _ in range(_MEASUREMENT_ITERATIONS):
            mic = Microphone()
            mic.stop()
            del mic
        gc.collect()
        snapshot_steady = tracemalloc.take_snapshot()
        rss_steady_mb = _rss_mb()
    finally:
        tracemalloc.stop()

    tracemalloc_growth_mb = _tracemalloc_total_growth_mb(
        snapshot_warmup, snapshot_steady
    )
    rss_growth_mb = rss_steady_mb - rss_warmup_mb

    assert tracemalloc_growth_mb < _TRACEMALLOC_THRESHOLD_MB, (
        f"tracemalloc shows {tracemalloc_growth_mb:.3f} MB growth across "
        f"{_MEASUREMENT_ITERATIONS} construct/destroy cycles; "
        f"threshold {_TRACEMALLOC_THRESHOLD_MB} MB exceeded"
    )
    assert rss_growth_mb < _RSS_THRESHOLD_MB, (
        f"RSS grew {rss_growth_mb:.3f} MB across {_MEASUREMENT_ITERATIONS} "
        f"construct/destroy cycles; threshold {_RSS_THRESHOLD_MB} MB exceeded"
    )


# ---------------------------------------------------------------------------
# Section 2: async Microphone lifecycle.
#
# Same shape as Section 1 but inside an asyncio event loop. Catches leaks
# specific to the async wrapper (tokio runtime handles, asyncio-bridged
# Future objects). pytest-asyncio is already a dev dependency.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_microphone_construct_destroy_no_growth() -> None:
    """20 async construct/destroy cycles must not leak Python or native memory."""
    tracemalloc.start()
    try:
        for _ in range(_WARMUP_ITERATIONS):
            mic = AsyncMicrophone()
            await mic.stop()
            del mic
        gc.collect()
        snapshot_warmup = tracemalloc.take_snapshot()
        rss_warmup_mb = _rss_mb()

        for _ in range(_MEASUREMENT_ITERATIONS):
            mic = AsyncMicrophone()
            await mic.stop()
            del mic
        gc.collect()
        snapshot_steady = tracemalloc.take_snapshot()
        rss_steady_mb = _rss_mb()
    finally:
        tracemalloc.stop()

    tracemalloc_growth_mb = _tracemalloc_total_growth_mb(
        snapshot_warmup, snapshot_steady
    )
    rss_growth_mb = rss_steady_mb - rss_warmup_mb

    assert tracemalloc_growth_mb < _TRACEMALLOC_THRESHOLD_MB, (
        f"tracemalloc shows {tracemalloc_growth_mb:.3f} MB growth across "
        f"{_MEASUREMENT_ITERATIONS} async construct/destroy cycles; "
        f"threshold {_TRACEMALLOC_THRESHOLD_MB} MB exceeded"
    )
    assert rss_growth_mb < _RSS_THRESHOLD_MB, (
        f"RSS grew {rss_growth_mb:.3f} MB across {_MEASUREMENT_ITERATIONS} "
        f"async construct/destroy cycles; threshold {_RSS_THRESHOLD_MB} MB exceeded"
    )


# ---------------------------------------------------------------------------
# Section 3: Silero VAD construct/destroy lifecycle.
#
# Microphone(vad="silero") triggers ORT session init during construction.
# The session loads the bundled silero_vad.onnx model (~2.3 MB) and
# allocates an arena under ORT's memory manager. A leak here would surface
# as monotonic RSS growth equal to the per-cycle session size.
#
# Marker: requires_bundled_ort. conftest.py auto-skips on dev installs
# where bindings/python/python/decibri/_ort/ is empty (maturin develop
# does not stage the dylib). CI populates the directory before pytest
# runs, so this test executes there.
#
# RSS threshold widened to 10 MB to absorb ORT's allocator behavior;
# tracemalloc threshold stays at 1 MB because ORT's allocations are
# native and invisible to tracemalloc.
# ---------------------------------------------------------------------------


@pytest.mark.requires_bundled_ort
def test_silero_vad_construct_destroy_no_growth() -> None:
    """20 silero-VAD construct/destroy cycles must not leak ORT session memory."""
    tracemalloc.start()
    try:
        for _ in range(_WARMUP_ITERATIONS):
            mic = Microphone(vad="silero")
            mic.stop()
            del mic
        gc.collect()
        snapshot_warmup = tracemalloc.take_snapshot()
        rss_warmup_mb = _rss_mb()

        for _ in range(_MEASUREMENT_ITERATIONS):
            mic = Microphone(vad="silero")
            mic.stop()
            del mic
        gc.collect()
        snapshot_steady = tracemalloc.take_snapshot()
        rss_steady_mb = _rss_mb()
    finally:
        tracemalloc.stop()

    tracemalloc_growth_mb = _tracemalloc_total_growth_mb(
        snapshot_warmup, snapshot_steady
    )
    rss_growth_mb = rss_steady_mb - rss_warmup_mb

    # RSS threshold widened for ORT's native allocator. The wrapper-layer
    # leak surface (tracemalloc) keeps the canonical 1 MB threshold; only
    # the native-allocator surface absorbs the wider band.
    rss_silero_threshold_mb = 10.0

    assert tracemalloc_growth_mb < _TRACEMALLOC_THRESHOLD_MB, (
        f"tracemalloc shows {tracemalloc_growth_mb:.3f} MB growth across "
        f"{_MEASUREMENT_ITERATIONS} silero-VAD construct/destroy cycles; "
        f"threshold {_TRACEMALLOC_THRESHOLD_MB} MB exceeded"
    )
    assert rss_growth_mb < rss_silero_threshold_mb, (
        f"RSS grew {rss_growth_mb:.3f} MB across {_MEASUREMENT_ITERATIONS} "
        f"silero-VAD construct/destroy cycles; "
        f"threshold {rss_silero_threshold_mb} MB exceeded"
    )
