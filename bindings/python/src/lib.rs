//! decibri Python bindings.
//!
//! Phase 1 scaffold: exposes only the `DecibriBridge.version()` static method.
//! Phase 2 will add `Decibri` and `DecibriOutput` sync classes, all 28 error
//! types, and the full capture / output / VAD surface. Phase 3 adds `numpy`
//! and the async `AsyncDecibri` / `AsyncDecibriOutput` counterparts.
//!
//! The extension module is named `decibri._decibri` (underscore-prefixed,
//! internal). The public Python surface lives at `python/decibri/__init__.py`
//! and re-exports from `_decibri`. Commit 1 of Phase 1 ships the Rust side
//! only; the Python package layer lands in Commit 2.

use pyo3::prelude::*;

use decibri::CPAL_VERSION;

/// Version information for the decibri core and its audio runtime.
///
/// Returned from [`DecibriBridge::version`]. Fields mirror the Node binding's
/// `VersionInfoJs` exactly so that downstream tooling parsing either side's
/// output sees identical field names and identical field values.
#[pyclass(frozen, get_all, module = "decibri._decibri")]
pub struct VersionInfo {
    pub decibri: String,
    pub portaudio: String,
}

#[pymethods]
impl VersionInfo {
    fn __repr__(&self) -> String {
        format!(
            "VersionInfo(decibri='{}', portaudio='{}')",
            self.decibri, self.portaudio
        )
    }
}

/// Internal bridge pyclass. Phase 1 has a single staticmethod.
///
/// The `DecibriBridge` name mirrors `DecibriBridge` in the Node binding and
/// keeps the Rust surface consistent across FFI layers. The public Python
/// `Decibri` class (Phase 2) will wrap this bridge.
#[pyclass(module = "decibri._decibri")]
pub struct DecibriBridge;

#[pymethods]
impl DecibriBridge {
    /// Returns version info for the decibri Rust core and its audio runtime.
    ///
    /// The `decibri` field is the Rust workspace version (resolved at build
    /// time via `env!("CARGO_PKG_VERSION")`), NOT the Python wheel version.
    /// For the wheel version, use `decibri.__version__`.
    #[staticmethod]
    fn version() -> VersionInfo {
        VersionInfo {
            decibri: env!("CARGO_PKG_VERSION").to_string(),
            portaudio: format!("cpal {}", CPAL_VERSION),
        }
    }
}

/// Module entry point. maturin looks for a function named exactly the module
/// name (`_decibri`) decorated with `#[pymodule]`.
#[pymodule]
fn _decibri(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<DecibriBridge>()?;
    m.add_class::<VersionInfo>()?;
    Ok(())
}
