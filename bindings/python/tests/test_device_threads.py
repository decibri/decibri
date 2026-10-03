"""Device calls work from any thread, including after the thread that made the
process's first device call has exited.

Each case runs in a fresh interpreter, so the case's first device call is the
first in its process, and a fault ends the child rather than the test run. The
child must exit normally and print the case's completion line.

The cases need no audio device. With none present the calls return empty lists
or raise, and the child ignores what they raise.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

# What the first thread does. An index past the end of the device list makes
# start() look the device up by position, which raises before any stream opens.
_FIRST_CALLS = {
    "listing": "decibri.Microphone.devices(); decibri.Speaker.devices()",
    "microphone-start": "decibri.Microphone(device=1_000_000).start()",
    "speaker-start": "decibri.Speaker(device=1_000_000).start()",
}

# A thread makes the process's first device call and exits, then a second
# thread lists the devices and exits, then the main thread lists them.
_CHILD = """
import threading

import decibri


def first():
    try:
        {first}
    except decibri.DecibriError:
        pass


def listing():
    decibri.Microphone.devices()
    decibri.Speaker.devices()


for target in (first, listing):
    thread = threading.Thread(target=target)
    thread.start()
    thread.join()
listing()
print("case complete")
"""


@pytest.mark.parametrize("case", sorted(_FIRST_CALLS))
def test_device_calls_work_after_the_first_device_thread_exits(case: str) -> None:
    """A thread makes the process's first device call and exits, and device
    calls from other threads then complete."""
    script = textwrap.dedent(_CHILD).format(first=_FIRST_CALLS[case])
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0 and "case complete" in proc.stdout, (
        f"the child ended with {proc.returncode}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
