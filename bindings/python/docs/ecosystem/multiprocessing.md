# decibri with Python multiprocessing

decibri composes cleanly with `multiprocessing.Process` and `concurrent.futures.ProcessPoolExecutor` when the `spawn` start method is used. On Linux, where the default start method is `fork`, decibri detects the dangerous case (Silero VAD initialized in the parent process before fork) and raises a clean `ForkAfterOrtInit` exception in the child rather than allowing silent ONNX Runtime corruption.

This document covers the patterns verified in the Phase 9 prep research plus the Phase 9 fork-detection fix.

## TL;DR

- **`spawn` is safe for everything decibri does.** Recommended on all platforms.
- **`fork` (Linux default) is safe ONLY if you do not touch Silero VAD before forking.** Construct `Microphone(vad="silero")` inside the child, never in the parent before fork.
- **Pickling Microphone / Speaker / AsyncMicrophone / AsyncSpeaker raises `TypeError`.** Construct them inside worker processes; do not try to pass instances across process boundaries.

## Spawn start method works for everything

Verified on Windows (Phase 9 prep research; spawn is the only option) and consistent with the cross-platform Python multiprocessing semantics. Spawn re-imports your module in the child process and re-runs construction from scratch, so each child gets its own decibri state without inheriting anything from the parent.

```python
# spawn_basic.py: import + version in spawned children
import multiprocessing as mp


def child_work(_):
    import decibri
    info = decibri.version()
    return f"{info.decibri}|{info.audio_backend}|{info.binding}"


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=2) as pool:
        for r in pool.map(child_work, range(2)):
            print(r)
```

Verified output:

```
3.4.0|cpal 0.17|0.1.0
3.4.0|cpal 0.17|0.1.0
```

The same pattern works for `Microphone`, `Speaker`, `AsyncMicrophone`, `AsyncSpeaker`, and `decibri.input_devices()`. Each child constructs its own bridge; no cross-process state is shared.

## ProcessPoolExecutor works

`concurrent.futures.ProcessPoolExecutor` uses `spawn` by default on Windows and macOS, and respects `mp.get_context("spawn")` everywhere:

```python
# pool_input_devices.py
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp


def list_devices(_):
    import decibri
    return len(decibri.input_devices())


if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        for n in ex.map(list_devices, range(2)):
            print(f"worker saw {n} input devices")
```

## Pickling Microphone fails cleanly

Microphone, Speaker, and their async counterparts cannot be pickled. The underlying cpal stream lives in process-local memory and cannot be sent across process boundaries; PyO3 pyclasses are non-picklable by default. Attempting to pickle raises a clean `TypeError`:

```python
# pickle_attempt.py
import pickle
import decibri

try:
    pickle.dumps(decibri.Microphone())
except TypeError as exc:
    print(f"pickle failed cleanly: {exc}")
# -> pickle failed cleanly: cannot pickle 'decibri._decibri.MicrophoneBridge' object
```

This is the correct behavior; the failure mode is loud and the error message identifies the specific bridge class. The recommended pattern is to construct decibri objects inside the worker process and stream raw audio bytes back through a `multiprocessing.Queue`.

## Recommended pattern: construct inside the worker

```python
# producer_worker.py: capture in the worker, send bytes back via Queue
import multiprocessing as mp


def producer(queue, duration_seconds: float):
    import decibri
    with decibri.Microphone(sample_rate=16000, channels=1) as mic:
        chunks_per_second = 10  # 100 ms chunks at 16 kHz with 1600 frames buffer
        for _ in range(int(duration_seconds * chunks_per_second)):
            chunk = mic.read(timeout_ms=2000)
            if chunk is None:
                break
            queue.put(chunk)
    queue.put(None)  # sentinel


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=64)
    proc = ctx.Process(target=producer, args=(queue, 3.0))
    proc.start()
    total_bytes = 0
    while True:
        chunk = queue.get()
        if chunk is None:
            break
        total_bytes += len(chunk)
    proc.join()
    print(f"received {total_bytes} bytes from worker")
```

The same shape works for any consumer-producer split: capture in one worker, ML inference in another, output playback in a third, all communicating via `multiprocessing.Queue` with raw `bytes` payloads.

## Linux fork() and Silero VAD: ForkAfterOrtInit

ONNX Runtime is not safe to share across forked processes. The runtime's internal state (thread pools, allocators, model graph state) is initialized in the parent's address space and is not transparently fork-safe. A child that inherits an ORT-initialized parent and then calls `session.run` produces silent wrong answers, segfaults, or hangs depending on the specific configuration.

Phase 9 added proactive detection: decibri records the parent process id alongside the global ORT init, and on every Silero inference call compares the current pid to the recorded pid. If they differ, decibri raises `ForkAfterOrtInit` (a `DecibriError` subclass) instead of allowing ORT to silently misbehave.

```python
# fork_after_init.py: the dangerous pattern, now caught proactively
import multiprocessing as mp
import decibri
from decibri.exceptions import ForkAfterOrtInit


def child_inference(queue):
    # Inherits the parent's ORT_INIT_PID via fork; the next Silero call
    # detects the pid mismatch and raises ForkAfterOrtInit.
    try:
        mic = decibri.Microphone(vad="silero")
        mic.start()
        mic.read()
    except ForkAfterOrtInit as exc:
        queue.put(("fork_after_init_raised", str(exc)))
        return
    queue.put(("read_succeeded_unexpectedly", ""))


if __name__ == "__main__":
    # Trigger Silero ORT init in the parent.
    parent_mic = decibri.Microphone(vad="silero")

    ctx = mp.get_context("fork")  # Linux only
    queue = ctx.Queue()
    p = ctx.Process(target=child_inference, args=(queue,))
    p.start()
    p.join(timeout=10)
    print(queue.get(timeout=5))
```

Two clean remediations:

**Remediation A: switch to spawn**

```python
mp.set_start_method("spawn", force=True)
```

`spawn` re-imports your module in the child and re-runs ORT init from scratch. The fork detection passes because ORT_INIT_PID is set to the child's own pid.

**Remediation B: construct VAD inside the child**

```python
# fork_construct_in_child.py: also safe under fork
import multiprocessing as mp


def child(_):
    import decibri
    mic = decibri.Microphone(vad="silero")  # ORT_INIT_PID = child pid
    return "ok"


if __name__ == "__main__":
    ctx = mp.get_context("fork")
    with ctx.Pool(2) as pool:
        print(pool.map(child, range(2)))
```

When the parent never initialized ORT, the child performs its own first init and ORT_INIT_PID matches; no fork detection trips.

## Catching ForkAfterOrtInit

`ForkAfterOrtInit` is a direct subclass of `DecibriError` (not under `OrtError`, since it is a usage error rather than an ORT-internal error). Catch via either name:

```python
import decibri
from decibri.exceptions import ForkAfterOrtInit

try:
    mic.read()
except ForkAfterOrtInit as exc:
    # Specific catch: tell the operator to fix their start method
    raise RuntimeError("Use mp.set_start_method('spawn')") from exc

try:
    mic.read()
except decibri.DecibriError:
    # Generic catch: still works because ForkAfterOrtInit subclasses DecibriError
    ...
```

The `__all__` count grew from 17 to 18 in Phase 9 to expose `ForkAfterOrtInit` as a top-level catch target. The class is also accessible at `decibri.exceptions.ForkAfterOrtInit`.

## cpal stream is process-local

The cpal capture stream uses platform-specific OS resources (Windows WASAPI handles, Linux ALSA file descriptors, macOS CoreAudio AudioUnit objects) that are not transparently shared across forked processes. Constructing a `Microphone` in the parent and using it in a forked child does not produce a shared stream; depending on the platform, it produces either a dead handle in the child or undefined behavior. The recommended pattern remains: construct decibri objects inside the worker, never share via fork.

This is a less consequential failure mode than the ORT case because it surfaces immediately on the first read (dead handle) rather than producing silent wrong answers; no proactive detection is added in Phase 9 for this case.

## Decision matrix

| Scenario                                       | Safe? |
| ---------------------------------------------- | ----- |
| `spawn` + construct in child + any decibri op  | yes   |
| `spawn` + parent uses VAD + child uses VAD     | yes (each gets own ORT init) |
| `fork` + only construct in child               | yes   |
| `fork` + parent uses VAD + child does anything VAD | **raises ForkAfterOrtInit** |
| `fork` + cpal stream constructed in parent     | undefined (do not do this) |
| Pickling Microphone / Speaker                  | raises TypeError |
| `concurrent.futures.ProcessPoolExecutor` (default) | yes (uses spawn) |
