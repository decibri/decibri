# decibri in Jupyter notebooks

decibri composes cleanly with the IPython kernel (ipykernel 7.x; IPython 9.x; verified 2026-05-02 against IPython 9.13.0 + ipykernel 7.2.0). The two integration points worth knowing about are the ambient asyncio event loop (top-level `await` works) and ORT model loading (use the async factory to avoid blocking cell rendering).

This document covers the patterns verified in the Phase 9 prep research, with copy-paste examples that run end-to-end in a fresh notebook.

## Top-level await works

IPython 9 + ipykernel 7 runs a persistent asyncio event loop per kernel (`_WindowsSelectorEventLoop` on Windows; `_UnixSelectorEventLoop` on Linux/macOS). Any coroutine awaited at the top of a cell is scheduled on that loop. `decibri.AsyncMicrophone` and `decibri.AsyncSpeaker` cooperate with whatever loop ipykernel provides; you do not need `asyncio.run` and you should not call it inside a cell.

```python
# Cell 1
import decibri

amic = decibri.AsyncMicrophone(sample_rate=16000, channels=1)
await amic.start()
chunk = await amic.read(timeout_ms=1000)
print(f"got {len(chunk)} bytes")
await amic.stop()
```

```python
# Cell 2
async with decibri.AsyncMicrophone() as amic:
    async for chunk in amic:
        if chunk is None:
            break
        print(len(chunk))
        if amic.is_speaking:
            print("speech")
        break  # one chunk for demo
```

Constructing a fresh `AsyncMicrophone` in a subsequent cell works without any loop-state cleanup; the wrapper does not cache the loop reference across constructions.

## Use AsyncMicrophone.open() with vad="silero" in async cells

`Microphone(vad="silero")` and `AsyncMicrophone(vad="silero")` perform synchronous ONNX Runtime model loading inside `__init__`, which takes 100 to 500 ms (CPU-bound; slower on cold disk caches). Calling the constructor from inside an `async` cell blocks the kernel's event loop for the duration of the load, which can stall other coroutines (websocket pings, timer callbacks, JupyterLab UI updates).

The async factory `AsyncMicrophone.open()` (added in Phase 9) dispatches construction to the default ThreadPoolExecutor via `loop.run_in_executor`, returning the constructed instance awaitably.

```python
# Cell: prefer .open() over the constructor in async code
import decibri

# Blocks the event loop for ~100-500 ms. Acceptable for occasional
# construction; not acceptable in latency-sensitive pipelines.
amic = decibri.AsyncMicrophone(vad="silero")

# Non-blocking; the construction runs in a worker thread.
amic = await decibri.AsyncMicrophone.open(vad="silero")

# Symmetric on AsyncSpeaker.
spk = await decibri.AsyncSpeaker.open(sample_rate=24000)
```

The synchronous `__init__` constructor remains supported for back-compat and for users who construct outside an async context. After construction, `AsyncMicrophone.start()`, `read()`, `stop()`, and the async-context-manager / async-iterator protocols behave identically regardless of which factory was used.

## Auto-display shows useful state

A bare `Microphone` or `AsyncMicrophone` instance evaluated as the last expression in a cell auto-displays the `__repr__` (Phase 9 added a useful `__repr__` to all four wrapper classes). The repr shows construction parameters plus `is_open` so you can see at a glance whether the stream is currently running:

```python
# Cell: auto-display shows construction state
mic = decibri.Microphone(sample_rate=24000, channels=2, vad="silero")
mic
# -> Microphone(sample_rate=24000, channels=2, dtype='int16',
#               frames_per_buffer=1600, vad='silero', is_open=False)
```

After `mic.start()` (or entering the context manager), `is_open` flips to `True` and the repr reflects it.

## Kernel restart and explicit cleanup

When you click "Restart Kernel," the kernel process dies and the OS releases all of its resources, including any cpal audio streams and the ORT runtime. From decibri's perspective, kernel restart is safe; there is no cross-process state to clean up. The new kernel process starts fresh.

That said, calling `mic.close()` (or `mic.stop()`; the two are aliases) explicitly before a long-running cell ends is a cleaner pattern than relying on the defensive `__del__` finalizer. The finalizer catches the common case of a cell that forgot the context manager, but it runs at arbitrary GC points and cannot raise.

```python
# Cell: explicit cleanup is the recommended pattern
mic = decibri.Microphone()
try:
    mic.start()
    chunk = mic.read(timeout_ms=1000)
finally:
    mic.close()  # alias for stop(); also called by __exit__
```

## Logging output

Python `print()` and `logging` from your own code work normally; ipykernel captures stdout/stderr per cell. Rust-side `tracing` output (from inside the decibri binding) does not appear in cell output by default; it goes to the file descriptors the Jupyter server inherits, which usually means the terminal where you launched `jupyter lab` or `jupyter notebook`.

To see decibri's Rust-side trace output, set `RUST_LOG` before launching the Jupyter server:

```bash
RUST_LOG=info jupyter lab    # Linux/macOS
$env:RUST_LOG="info"; jupyter lab    # PowerShell
```

This is consistent with how every Rust-PyO3 binding behaves; it is a `tracing` subscriber configuration question, not a decibri-specific limitation.

## Windows: zmq RuntimeWarning at kernel boot

On Windows you may see a one-time `RuntimeWarning` from pyzmq at first kernel startup:

```
zmq/_future.py:718: RuntimeWarning: Proactor event loop does not implement
add_reader family of methods required for zmq. Registering an additional
selector thread for add_reader support via tornado.
```

This is a pyzmq + Windows + ipykernel issue; pyzmq emits the warning and falls back to a selector thread automatically. It does not affect decibri or your code; you can ignore it.

## VAD-load timing reference

For users tuning latency budgets:

| Construction                          | Typical first-call cost | Subsequent constructions |
| ------------------------------------- | ----------------------- | ------------------------ |
| `Microphone()` (no VAD)               | <10 ms                  | <10 ms                   |
| `Microphone(vad="energy")`            | <10 ms                  | <10 ms                   |
| `Microphone(vad="silero")`            | 100 to 500 ms           | <10 ms                   |
| `AsyncMicrophone.open(vad="silero")`  | 100 to 500 ms (offloaded) | <10 ms (offloaded)     |

ONNX Runtime initialization is process-global; once the first `Microphone(vad="silero")` warms the runtime, subsequent constructions in the same kernel process see the cached state and complete in under 10 ms. Restarting the kernel resets this; the next construction pays the full first-call cost again.
