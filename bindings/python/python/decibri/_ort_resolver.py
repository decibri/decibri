"""ONNX Runtime dynamic library path resolver for the Python wheel.

The Phase 2 wheel build leaves ORT loading to the user (manual
``ORT_DYLIB_PATH``, or a separate ``pip install onnxruntime``), which is
incompatible with the premium-quality remit. Phase 3 closes that gap by
bundling the per-platform dylib into ``decibri/_ort/`` at CI build time and
resolving its path at first VAD use through this module.

The resolver implements a four-arm priority order so that:

1. Power users can override at the call site (constructor argument).
2. Operators can override per-deployment via a decibri-namespaced env var
   without touching code.
3. Existing users of the upstream ``ort`` crate convention
   (``ORT_DYLIB_PATH``) keep working.
4. Default ``pip install decibri`` users get the bundled dylib for free.

The module is intentionally policy-only. It never touches ORT itself, never
loads a dylib, and never raises on a missing or invalid path. Path validity
is the Rust core's responsibility (``init_ort_once`` in
``crates/decibri/src/vad.rs``); this resolver returns a string path or
``None`` and lets the bridge layer be authoritative on errors. That keeps
the Python layer testable in isolation and avoids drift between Python-side
and Rust-side validation messages.
"""

from __future__ import annotations

import importlib.resources
import logging
import os
import sys
from pathlib import Path

__all__ = ["resolve_ort_dylib_path"]

_LOGGER = logging.getLogger(__name__)

# Platform-appropriate filename glob for the bundled dylib. Pattern (not
# exact match) tolerates auditwheel and delocate hash-suffix renaming
# applied at wheel-repair time (e.g. ``libonnxruntime-abc123.so``). One
# entry per supported ``sys.platform`` value: ``linux``, ``darwin``,
# ``win32``. Other platforms fall through to ``None`` and the bundled
# lookup returns nothing.
_PLATFORM_PATTERNS: dict[str, str] = {
    "linux": "libonnxruntime*.so",
    "darwin": "libonnxruntime*.dylib",
    "win32": "onnxruntime*.dll",
}


def _bundled_dylib_path() -> str | None:
    """Return the absolute path to the bundled ORT dylib, or None.

    Enumerates ``decibri/_ort/`` for files matching the platform glob and
    returns the first hit (sorted for determinism when multiple match).
    Returns ``None`` if the directory is missing, empty, or the platform
    is unsupported.

    Defensive against edge cases: if ``importlib.resources`` raises (e.g.
    the ``decibri`` package can't be located when running from an unusual
    import path), the exception is logged at debug and ``None`` is
    returned. Loud failures here would prevent users from constructing
    ``Decibri(vad=False)`` which doesn't need ORT at all.
    """
    pattern = _PLATFORM_PATTERNS.get(sys.platform)
    if pattern is None:
        return None

    try:
        ort_dir = importlib.resources.files("decibri") / "_ort"
        if not ort_dir.is_dir():
            return None
        candidates = sorted(
            child.name
            for child in ort_dir.iterdir()
            if child.is_file() and Path(child.name).match(pattern)
        )
        if not candidates:
            return None
        match = ort_dir / candidates[0]
        return str(match)
    except (ModuleNotFoundError, FileNotFoundError, NotADirectoryError) as exc:
        _LOGGER.debug("Bundled ORT lookup failed: %s", exc)
        return None


def resolve_ort_dylib_path(user_override: str | Path | None) -> str | None:
    """Resolve the ONNX Runtime dynamic library path via the decibri priority order.

    Priority (first match wins):

    1. ``user_override`` (the ``ort_library_path`` constructor argument).
       Returned as a string regardless of input type. Not pre-validated;
       the Rust core's ``init_ort_once`` raises ``OrtPathInvalid`` if the
       path doesn't exist or isn't a regular file.
    2. ``DECIBRI_ORT_DYLIB_PATH`` environment variable.
    3. ``ORT_DYLIB_PATH`` environment variable. Respected for
       compatibility with the upstream ``ort`` crate's standard env var,
       so existing users of bare ``ort``-based deployments continue to
       work without code changes.
    4. The bundled dylib at ``decibri/_ort/<platform-glob>``, included
       in the wheel by maturin's include directive.

    Returns the resolved path as a string, or ``None`` if nothing
    resolves. A ``None`` return indicates the resolver could not find a
    bundled or configured ORT dylib; the caller (typically the Rust
    bridge) falls back to ORT's default loader, which itself respects
    ``ORT_DYLIB_PATH`` if set after decibri import.

    Empty-string env var values are treated as unset, mirroring how POSIX
    and Windows shells typically intend ``FOO=`` to disable a setting.
    """
    if user_override is not None:
        return str(Path(user_override))

    decibri_env = os.environ.get("DECIBRI_ORT_DYLIB_PATH")
    if decibri_env:
        return decibri_env

    ort_env = os.environ.get("ORT_DYLIB_PATH")
    if ort_env:
        return ort_env

    return _bundled_dylib_path()
