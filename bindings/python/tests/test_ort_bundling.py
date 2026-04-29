"""Phase 3 ORT bundling and resolver tests.

Three tiers:

1. Resolver unit tests (no hardware, no ORT load): verify the four-arm
   priority order in `_ort_resolver.resolve_ort_dylib_path` using
   `monkeypatch` for env vars and `tmp_path` for synthetic `_ort/`
   layouts. Always run; no precondition markers.

2. Wheel-content integration tests (no ORT load, requires bundled `_ort/`):
   verify that the wheel actually ships a platform-appropriate dylib and
   that the resolver picks it. Marked `requires_bundled_ort`; auto-skipped
   on dev installs where `_ort/` is empty (per `conftest.py`).

3. End-to-end construction smoke (requires bundled `_ort/`): construct
   `Decibri(vad=True, vad_mode='silero')` and verify the load path
   reaches Silero session creation without raising. Marked
   `requires_bundled_ort`. Inference (forward pass against synthesized
   audio) is intentionally deferred to Phase 4+; the existing Decibri
   class consumes from a real microphone and does not expose a
   buffer-injection API, so a deterministic inference smoke would need
   an additional surface that is out of Phase 3 scope.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterator

import pytest

import decibri
from decibri._ort_resolver import resolve_ort_dylib_path

_ENV_VARS = ("DECIBRI_ORT_DYLIB_PATH", "ORT_DYLIB_PATH")

_PLATFORM_FILENAME = {
    "linux": "libonnxruntime.so",
    "darwin": "libonnxruntime.dylib",
    "win32": "onnxruntime.dll",
}

_PLATFORM_HASH_SUFFIXED = {
    "linux": "libonnxruntime-abc123.so",
    "darwin": "libonnxruntime-abc123.dylib",
    "win32": "onnxruntime-abc123.dll",
}

_PLATFORM_PATTERN = {
    "linux": "libonnxruntime*.so",
    "darwin": "libonnxruntime*.dylib",
    "win32": "onnxruntime*.dll",
}


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Strip both ORT-related env vars so tests run in known-state isolation."""
    for var in _ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    yield


# ---------------------------------------------------------------------------
# Tier 1: Resolver unit tests
# ---------------------------------------------------------------------------


def test_resolver_user_override_path_object(
    clean_env: None, tmp_path: Path
) -> None:
    """A Path user override is converted to a string and returned verbatim."""
    fake = tmp_path / "user.so"
    fake.write_bytes(b"")
    result = resolve_ort_dylib_path(fake)
    assert result == str(fake)


def test_resolver_user_override_str(clean_env: None, tmp_path: Path) -> None:
    """A str user override is returned through Path normalization."""
    fake = tmp_path / "user.so"
    fake.write_bytes(b"")
    result = resolve_ort_dylib_path(str(fake))
    assert result == str(fake)


def test_resolver_user_override_beats_env_vars(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """User override takes priority over both env var arms."""
    user_path = tmp_path / "user.so"
    decibri_path = tmp_path / "decibri_env.so"
    ort_path = tmp_path / "ort_env.so"
    for f in (user_path, decibri_path, ort_path):
        f.write_bytes(b"")
    monkeypatch.setenv("DECIBRI_ORT_DYLIB_PATH", str(decibri_path))
    monkeypatch.setenv("ORT_DYLIB_PATH", str(ort_path))
    assert resolve_ort_dylib_path(user_path) == str(user_path)


def test_resolver_decibri_env(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """DECIBRI_ORT_DYLIB_PATH env var is honored when no user override."""
    target = tmp_path / "env.so"
    target.write_bytes(b"")
    monkeypatch.setenv("DECIBRI_ORT_DYLIB_PATH", str(target))
    assert resolve_ort_dylib_path(None) == str(target)


def test_resolver_ort_env_compat(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """ORT_DYLIB_PATH env var is honored for upstream ort crate compat."""
    target = tmp_path / "ort.so"
    target.write_bytes(b"")
    monkeypatch.setenv("ORT_DYLIB_PATH", str(target))
    assert resolve_ort_dylib_path(None) == str(target)


def test_resolver_decibri_env_beats_ort_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When both env vars are set, DECIBRI_ORT_DYLIB_PATH wins."""
    decibri_path = tmp_path / "decibri.so"
    ort_path = tmp_path / "ort.so"
    decibri_path.write_bytes(b"")
    ort_path.write_bytes(b"")
    monkeypatch.setenv("DECIBRI_ORT_DYLIB_PATH", str(decibri_path))
    monkeypatch.setenv("ORT_DYLIB_PATH", str(ort_path))
    assert resolve_ort_dylib_path(None) == str(decibri_path)


def test_resolver_empty_decibri_env_falls_through(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Empty-string env var is treated as unset; resolver falls through."""
    target = tmp_path / "ort.so"
    target.write_bytes(b"")
    monkeypatch.setenv("DECIBRI_ORT_DYLIB_PATH", "")
    monkeypatch.setenv("ORT_DYLIB_PATH", str(target))
    assert resolve_ort_dylib_path(None) == str(target)


def test_resolver_returns_none_when_empty(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No user override, no env vars, empty bundled dir: returns None.

    Mocks ``importlib.resources.files`` so the test is portable regardless
    of whether the local dev install has ``_ort/`` populated. The
    substitute returns ``tmp_path``; ``tmp_path / '_ort'`` does not exist,
    so the resolver's ``is_dir()`` check returns False.
    """
    import importlib.resources

    monkeypatch.setattr(
        importlib.resources, "files", lambda _pkg: tmp_path
    )
    assert resolve_ort_dylib_path(None) is None


def test_resolver_bundled_handles_hash_suffix(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Resolver picks an auditwheel/delocate-renamed file via glob match.

    auditwheel and delocate may rename bundled libraries to include a
    content hash (e.g. ``libonnxruntime-abc123.so``). The resolver must
    match these via glob pattern, not exact filename. This test
    synthesizes a fake ``_ort/`` directory containing only a hash-suffixed
    file and asserts the resolver still finds it.
    """
    import importlib.resources

    suffixed_name = _PLATFORM_HASH_SUFFIXED.get(sys.platform)
    if suffixed_name is None:
        pytest.skip(f"unsupported platform for hash-suffix test: {sys.platform}")

    ort_dir = tmp_path / "_ort"
    ort_dir.mkdir()
    suffixed = ort_dir / suffixed_name
    suffixed.write_bytes(b"")

    monkeypatch.setattr(
        importlib.resources, "files", lambda _pkg: tmp_path
    )

    result = resolve_ort_dylib_path(None)
    assert result is not None
    assert Path(result).name == suffixed_name


# ---------------------------------------------------------------------------
# Tier 2: Wheel-content integration tests
# ---------------------------------------------------------------------------


@pytest.mark.requires_bundled_ort
def test_wheel_bundle_present() -> None:
    """The installed wheel contains a platform-appropriate ORT dylib.

    Uses ``importlib.resources`` to enumerate the installed package's
    ``_ort/`` directory. Validates the maturin include directive in
    pyproject.toml landed correctly and python-ci.yml's download step
    populated the directory before the wheel was packed.
    """
    import importlib.resources

    pattern = _PLATFORM_PATTERN.get(sys.platform)
    if pattern is None:
        pytest.skip(f"unsupported platform: {sys.platform}")

    ort_dir = importlib.resources.files("decibri") / "_ort"
    assert ort_dir.is_dir(), "decibri/_ort/ is missing from the installed package"

    matches = sorted(
        child.name
        for child in ort_dir.iterdir()
        if child.is_file() and Path(child.name).match(pattern)
    )
    assert matches, (
        f"decibri/_ort/ contains no files matching {pattern!r}; "
        f"directory contents: {sorted(c.name for c in ort_dir.iterdir())}"
    )


@pytest.mark.requires_bundled_ort
def test_resolver_bundled_default(
    clean_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Default resolution (no override, no env vars) returns the bundled path.

    Exercises the live ``importlib.resources`` lookup against the actual
    installed package; no monkeypatching of ``files()``. The returned
    path is verified to exist and to match the platform glob.
    """
    pattern = _PLATFORM_PATTERN.get(sys.platform)
    if pattern is None:
        pytest.skip(f"unsupported platform: {sys.platform}")

    result = resolve_ort_dylib_path(None)
    assert result is not None, "expected bundled fallback to return a path"
    resolved = Path(result)
    assert resolved.is_file(), f"resolver returned non-existent path: {result}"
    assert resolved.match(pattern), (
        f"resolver returned {resolved.name!r} which does not match {pattern!r}"
    )


# ---------------------------------------------------------------------------
# Tier 3: End-to-end construction smoke
# ---------------------------------------------------------------------------


@pytest.mark.requires_bundled_ort
def test_decibri_silero_construction() -> None:
    """``Decibri(vad=True, vad_mode='silero')`` constructs end-to-end.

    Exercises the full Phase 3 load path in one assertion: the resolver
    finds the bundled dylib, the Rust bridge receives the resolved path,
    ``init_ort_once`` validates and loads it via ``ort::init_from()``, and
    the Silero session is built against the bundled model. If any link
    in that chain breaks (resolver returns None unexpectedly, dylib is
    invalid, model is missing, ort::init_from rejects the path), this
    test surfaces the failure. The construction does not start audio
    capture; no microphone is required.
    """
    instance = decibri.Decibri(vad=True, vad_mode="silero")
    assert instance is not None
    assert isinstance(instance, decibri.Decibri)
