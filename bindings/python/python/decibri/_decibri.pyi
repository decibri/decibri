"""Type stubs for the compiled ``decibri._decibri`` Rust extension module.

Matches the Rust surface in ``bindings/python/src/lib.rs``. Kept alongside
the compiled ``.pyd`` / ``.so`` so mypy and IDEs see types without loading
the extension. Update this file when the Rust surface changes.
"""

class VersionInfo:
    """Version information for the decibri core and its audio runtime.

    The ``decibri`` field is the Rust core version (``CARGO_PKG_VERSION``
    at build time), not the Python wheel version. For the wheel version,
    use :data:`decibri.__version__`.
    """

    @property
    def decibri(self) -> str: ...
    @property
    def portaudio(self) -> str: ...
    def __repr__(self) -> str: ...

class DecibriBridge:
    """Internal bridge class. Public ``Decibri`` / ``AsyncDecibri`` wrappers
    ship in Phase 2 / Phase 3."""

    @staticmethod
    def version() -> VersionInfo: ...
