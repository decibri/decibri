"""Phase 6 Step 1 empirical smoke tests for rust-numpy 0.28.0.

Verifies three preconditions for the AsyncMicrophone / Speaker numpy
surfaces that land in subsequent Phase 6 relays:

1. rust-numpy 0.28.0 builds and runs cleanly in our extension-module
   context with PyO3 0.28.x.

2. ``Vec<i16>::into_pyarray(py)`` produces a ``numpy.ndarray`` with the
   expected dtype, shape, and values. This is the primary API path for
   the upcoming ``MicrophoneBridge::read`` numpy branch (when the
   ``numpy=True`` constructor flag is set, the read returns a PyArray
   constructed from the cpal-derived ``Vec<i16>`` or ``Vec<f32>``).

3. The returned ndarray's ``flags.owndata == False`` documents the
   actual rust-numpy 0.28.0 behavior. The Phase 6 Step 1 empirical
   experiment surfaced this as outcome (ii): rust-numpy's
   ``IntoPyArray::into_pyarray`` consumes the Rust ``Vec`` (correct
   ownership semantics; no double-free) but allocates a fresh NumPy
   buffer and copies the data into it. The original Vec's heap
   allocation is freed; ``owndata=False`` means NumPy is borrowing
   the buffer it created (it owns the underlying data but not the
   ndarray-object container in the way ``owndata=True`` would imply).
   This is documented behavior we depend on, not a bug; the test
   exists to surface a CHANGE in this behavior across rust-numpy
   upgrades, which would warrant investigation.

These tests stay as persistent regression coverage (locked decision Q4
of phase-6-numpy-zerocopy.md): if a future rust-numpy upgrade breaks
the ``into_pyarray`` semantics or pyo3 0.28 compat, these tests fail
first and surface the regression at the build-pipeline level rather
than at AsyncMicrophone integration time.
"""

from __future__ import annotations

import numpy as np

from decibri._decibri import _numpy_smoke  # type: ignore[attr-defined]


def test_numpy_smoke_returns_ndarray() -> None:
    """The smoke pyfunction returns an actual ``numpy.ndarray``.

    Tests that rust-numpy's ``into_pyarray`` produces an ndarray (not a
    list, not a memoryview, not a buffer object). If this fails, the
    rust-numpy API shape has changed in a way that breaks our usage.
    """
    arr = _numpy_smoke()
    assert isinstance(arr, np.ndarray), f"Expected ndarray, got {type(arr)!r}"


def test_numpy_smoke_dtype_is_int16() -> None:
    """The smoke ndarray has dtype ``np.int16`` matching the ``Vec<i16>`` input.

    Validates the dtype mapping that Phase 6's read path will rely on:
    Rust ``i16`` -> NumPy ``np.int16`` for ``format='int16'`` mode.
    """
    arr = _numpy_smoke()
    assert arr.dtype == np.int16, f"Expected int16, got {arr.dtype!r}"


def test_numpy_smoke_shape_is_1d() -> None:
    """The smoke ndarray has shape ``(5,)`` matching the Rust Vec's length.

    Validates the 1-D shape convention for mono audio. Multi-channel
    audio (Phase 6 §3e) uses 2-D ``(N, channels)``; this smoke covers
    the mono case.
    """
    arr = _numpy_smoke()
    assert arr.shape == (5,), f"Expected (5,), got {arr.shape!r}"


def test_numpy_smoke_values_match() -> None:
    """The smoke ndarray contains ``[0, 1, 2, 3, 4]`` matching the Rust Vec.

    Sanity check that no byte-order or endianness issue corrupts the
    values during the Rust-NumPy boundary crossing.
    """
    arr = _numpy_smoke()
    expected = np.array([0, 1, 2, 3, 4], dtype=np.int16)
    np.testing.assert_array_equal(arr, expected)


def test_numpy_smoke_owndata_documents_copy_semantics() -> None:
    """The smoke ndarray's ``owndata`` flag is False on rust-numpy 0.28.0.

    Documents the empirical finding from Phase 6 Step 1's smoke
    experiment (outcome ii): rust-numpy's ``IntoPyArray::into_pyarray``
    is a copy-at-the-boundary operation, not strict ownership transfer.
    The Rust ``Vec`` is consumed (so there's no double-free or
    use-after-free risk; ownership semantics are correct), but the data
    is copied into a freshly-allocated NumPy buffer rather than
    re-using the Vec's heap allocation in place.

    Concretely:

    - ``owndata=False`` on the returned ndarray means NumPy is using a
      buffer it does not own as its primary backing store. In our case,
      that buffer was allocated by rust-numpy at construction time,
      filled with a copy of the Vec's contents, and handed to the
      ndarray as a borrowed view. NumPy frees the buffer when the
      ndarray is GC'd (no leak), but the ``OWNDATA`` flag remains
      False because the ndarray itself does not own the allocation in
      the way a ``np.zeros(5)``-constructed array would.

    - The original Rust Vec's heap allocation is freed when the Vec
      is dropped at the end of the ``into_pyarray`` call. There is no
      double-allocation in long-term steady state; only the brief
      window during the boundary crossing.

    - For audio chunks (typical sizes ~1-3 KB at 100 ms 16 kHz mono
      int16), the memcpy cost is nanoseconds. Negligible against the
      cpal callback frequency and the Python GIL acquire/release
      overhead. Phase 6 accepts this cost as part of the contract.

    - The Phase 6 plan's original "NumPy zero-copy support" framing
      was technically inaccurate; the rebrand to "NumPy ndarray
      support" lands as a separate plan-storage edit. The
      functionality this enables is unchanged: callers get
      ``numpy.ndarray`` returns from ``Microphone(numpy=True).read()``
      with the expected dtype and shape; only the marketing word
      "zero-copy" is wrong.

    If a future rust-numpy upgrade flips this flag to ``True`` (e.g.
    by switching ``IntoPyArray`` to a true zero-copy buffer-sharing
    implementation), this assertion fails and surfaces the change as
    something worth investigating. We may then choose to celebrate
    the upgrade (true zero-copy is strictly better) or to verify it
    does not introduce new lifetime/safety constraints we need to
    handle.
    """
    arr = _numpy_smoke()
    assert arr.flags.owndata is False, (
        f"Expected owndata=False (rust-numpy 0.28.0 documented copy "
        f"semantics; into_pyarray allocates a fresh NumPy buffer and "
        f"copies the Vec's contents), got {arr.flags.owndata!r}. If "
        f"this happened after a rust-numpy upgrade, the upgrade may "
        f"have introduced true zero-copy buffer sharing; investigate "
        f"and update the Phase 6 documentation accordingly."
    )
