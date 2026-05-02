"""Phase 9 Item C7: ForkAfterOrtInit detection on Linux.

These tests pin the contract that decibri raises ``ForkAfterOrtInit`` in
a child process when ONNX Runtime was initialized in the parent before
the fork. Without this detection, the child would silently produce wrong
VAD probabilities, segfault, or hang depending on the exact ORT
configuration. See ``docs/ecosystem/multiprocessing.md`` for the user-
facing remediation guidance.

All three tests are gated to Linux: the ``fork`` start method is not
available on Windows or macOS (macOS supports it nominally but Apple has
disabled it for any process linking Objective-C frameworks, which
includes Silero VAD's CoreML provider). The ``requires_bundled_ort``
marker further skips when the ORT dylib is not bundled, so editable
``maturin develop`` installs that don't populate ``_ort/`` skip
gracefully; CI builds wheels with the dylib bundled and exercises the
tests fully.

Inter-process result reporting uses ``multiprocessing.Queue`` rather
than child exit code so the test asserts the specific exception that
was raised, not just "child died." This catches the case where the
child raises a different exception (e.g. ``OrtInferenceFailed`` from a
silent corruption that happened to surface) instead of the proactive
``ForkAfterOrtInit`` raise.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys

import pytest

import decibri
from decibri.exceptions import ForkAfterOrtInit


pytestmark = [
    pytest.mark.skipif(
        sys.platform != "linux",
        reason="fork() start method is Linux-only in scope; Windows has no fork(), macOS disables it for ObjC-linking processes",
    ),
    pytest.mark.requires_bundled_ort,
]


# ---------------------------------------------------------------------------
# Worker functions (module-level for picklability under spawn).
# ---------------------------------------------------------------------------


def _child_use_silero_after_fork(queue: "mp.Queue[tuple[str, str]]") -> None:
    """Run inside a forked child: try to use Silero, expect ForkAfterOrtInit.

    The child inherits the parent's ``ORT_INIT_PID`` via the OnceLock
    that fork() duplicates into the child's address space; calling any
    Silero inference path then trips the pid mismatch check and raises
    ``ForkAfterOrtInit``. Reports the outcome via the queue.
    """
    try:
        mic = decibri.Microphone(vad="silero")
        mic.start()
        try:
            mic.read(timeout_ms=500)
        finally:
            mic.stop()
    except ForkAfterOrtInit as exc:
        queue.put(("fork_after_init_raised", str(exc)))
        return
    except Exception as exc:
        queue.put((f"unexpected_{type(exc).__name__}", str(exc)))
        return
    queue.put(("read_succeeded_unexpectedly", ""))


def _child_init_silero_in_child(queue: "mp.Queue[tuple[str, str]]") -> None:
    """Run inside a forked child: do the Silero init in the child itself.

    When the parent never initialized ORT, the child's first init is a
    fresh init in its own address space; ``ORT_INIT_PID`` becomes the
    child's pid and the detection check passes. Any non-fork-related
    failures (missing model, bad library path) surface here as
    different exception types so the test fails loudly rather than
    silently passing the wrong way.
    """
    try:
        mic = decibri.Microphone(vad="silero")
        # Construction alone is enough to verify ORT init succeeded;
        # we deliberately avoid start()/read() because the child has no
        # real audio device in CI and would hang or NoMicrophoneFound.
        queue.put(("init_succeeded", f"pid={os.getpid()}"))
    except Exception as exc:
        queue.put((f"init_failed_{type(exc).__name__}", str(exc)))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fork_after_silero_init_raises_in_child() -> None:
    """Parent inits Silero, then forks; child raises ForkAfterOrtInit on use.

    This is the headline behavior of Phase 9 C7: silent ORT corruption
    on fork is now a clean exception with a remediation message.
    """
    # Parent: trigger Silero ORT init.
    parent_mic = decibri.Microphone(vad="silero")
    del parent_mic  # release the bridge; ORT_INIT_PID stays set

    ctx = mp.get_context("fork")
    queue: "mp.Queue[tuple[str, str]]" = ctx.Queue()
    child = ctx.Process(target=_child_use_silero_after_fork, args=(queue,))
    child.start()
    child.join(timeout=15)
    assert not child.is_alive(), "child failed to terminate within timeout"

    assert not queue.empty(), "child did not report a result before exit"
    outcome, detail = queue.get(timeout=5)
    assert outcome == "fork_after_init_raised", (
        f"Expected ForkAfterOrtInit raise; child reported {outcome!r}: {detail}"
    )
    assert "fork()" in detail, (
        f"Exception message should mention fork() for actionable user feedback; got: {detail}"
    )
    assert "spawn" in detail, (
        f"Exception message should suggest spawn remediation; got: {detail}"
    )


def test_fork_before_silero_init_works() -> None:
    """Forking BEFORE the parent initializes Silero is fine.

    The child does its own ORT init from scratch in its own address
    space; no inherited state to corrupt; no fork detection trips.
    Sanity check that the C7 detection isn't over-eager.
    """
    ctx = mp.get_context("fork")
    queue: "mp.Queue[tuple[str, str]]" = ctx.Queue()
    child = ctx.Process(target=_child_init_silero_in_child, args=(queue,))
    child.start()
    child.join(timeout=30)
    assert not child.is_alive(), "child failed to terminate within timeout"

    assert not queue.empty(), "child did not report a result before exit"
    outcome, detail = queue.get(timeout=5)
    assert outcome == "init_succeeded", (
        f"Expected fresh ORT init in child to succeed; child reported {outcome!r}: {detail}"
    )


def test_spawn_after_silero_init_works() -> None:
    """Spawn ignores the parent's ORT state.

    Parent inits ORT, then spawns; the child re-imports decibri and
    re-runs init from scratch. ``ORT_INIT_PID`` becomes the child's pid
    in the child's process; no fork detection trips because spawn does
    not duplicate the parent's address space.
    """
    parent_mic = decibri.Microphone(vad="silero")
    del parent_mic

    ctx = mp.get_context("spawn")
    queue: "mp.Queue[tuple[str, str]]" = ctx.Queue()
    child = ctx.Process(target=_child_init_silero_in_child, args=(queue,))
    child.start()
    child.join(timeout=30)
    assert not child.is_alive(), "child failed to terminate within timeout"

    assert not queue.empty(), "child did not report a result before exit"
    outcome, detail = queue.get(timeout=5)
    assert outcome == "init_succeeded", (
        f"Expected spawn to ignore parent ORT state; child reported {outcome!r}: {detail}"
    )
