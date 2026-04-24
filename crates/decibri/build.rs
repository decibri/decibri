//! Build script: extracts the resolved cpal version from the workspace
//! Cargo.lock and exposes it to crate code via the `DECIBRI_CPAL_VERSION`
//! environment variable.
//!
//! This is the single-source-of-truth mechanism for the `CPAL_VERSION` const
//! in `lib.rs`. Previously, the cpal version string was hardcoded in the
//! Node binding's `version()` function (`"cpal 0.17".to_string()`, in two
//! places): duplicated state that would silently drift when the workspace
//! bumped cpal. By reading the resolved version here and emitting a
//! build-time env var, every binding pulls `decibri::CPAL_VERSION` from one
//! place. When the workspace bumps cpal, Cargo.lock updates, this script
//! re-runs, and all downstream consumers see the new value automatically.
//!
//! Output format: major.minor only (`"0.17"`, not `"0.17.0"`). Preserves
//! byte-identical output with the pre-refactor hardcoded string. If a
//! future scope decision wants the full resolved version, change
//! `truncate_to_major_minor` to return the unmodified string.

use std::fs;
use std::path::PathBuf;

fn main() {
    // crates/decibri -> crates -> workspace root
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo"),
    );
    let workspace_lock = manifest_dir
        .parent()
        .and_then(|p| p.parent())
        .map(|p| p.join("Cargo.lock"))
        .expect("workspace root must be reachable from crates/decibri");

    // Re-run only when Cargo.lock or this build script changes. Without
    // these directives, cargo re-runs build.rs on every rebuild, defeating
    // incremental compilation caches.
    println!("cargo:rerun-if-changed={}", workspace_lock.display());
    println!("cargo:rerun-if-changed=build.rs");

    let lock_content = fs::read_to_string(&workspace_lock)
        .unwrap_or_else(|e| panic!("failed to read {}: {}", workspace_lock.display(), e));

    let cpal_version = extract_cpal_version(&lock_content).unwrap_or_else(|| {
        panic!(
            "expected a [[package]] entry for `cpal` in {}; Cargo.lock is \
             malformed or cpal has been removed from the dep graph",
            workspace_lock.display()
        )
    });

    let major_minor = truncate_to_major_minor(&cpal_version);
    println!("cargo:rustc-env=DECIBRI_CPAL_VERSION={}", major_minor);
}

/// Finds the `cpal` package's `version` line in a Cargo.lock file.
///
/// Cargo.lock format is stable since Cargo 1.0:
///
/// ```text
/// [[package]]
/// name = "cpal"
/// version = "0.17.3"
/// source = "registry+..."
/// ```
///
/// Walks the file in order, tracking whether we're inside a `[[package]]`
/// block, whether that block's `name` is `cpal`, and whether we've seen
/// `version` for that block. Returns the first match; the workspace
/// currently has exactly one cpal entry.
fn extract_cpal_version(lock_content: &str) -> Option<String> {
    let mut in_package = false;
    let mut is_cpal = false;
    for line in lock_content.lines() {
        let trimmed = line.trim();
        if trimmed == "[[package]]" {
            in_package = true;
            is_cpal = false;
            continue;
        }
        if in_package && trimmed == "name = \"cpal\"" {
            is_cpal = true;
            continue;
        }
        if in_package && is_cpal {
            if let Some(rest) = trimmed.strip_prefix("version = \"") {
                if let Some(v) = rest.strip_suffix('"') {
                    return Some(v.to_string());
                }
            }
        }
    }
    None
}

/// Truncates a semver string like `"0.17.3"` to `"0.17"`.
///
/// Preserves byte-identical output with the pre-refactor hardcoded string
/// `"cpal 0.17"` that lived in `bindings/node/src/lib.rs`. Accepts any
/// input; if the version has fewer than two dot-separated components, the
/// original string is returned unchanged (defensive, though semver always
/// has at least major.minor.patch for non-prerelease).
fn truncate_to_major_minor(version: &str) -> String {
    let parts: Vec<&str> = version.split('.').collect();
    if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        version.to_string()
    }
}
