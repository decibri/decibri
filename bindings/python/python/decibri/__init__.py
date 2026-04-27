"""decibri Python bindings.

Phase 1 scaffold. Public surface grows through Phases 2-4. Use
``decibri.DecibriBridge.version()`` for runtime version information and
``decibri.__version__`` for the installed wheel version.
"""

from decibri._decibri import DecibriBridge, VersionInfo
from decibri._classes import Decibri, DecibriOutput

__all__ = ["Decibri", "DecibriBridge", "DecibriOutput", "VersionInfo"]
__version__ = "0.1.0a1"
