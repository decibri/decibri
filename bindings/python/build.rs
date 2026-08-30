//! Build script: reads the project version from `pyproject.toml` and exposes
//! it to crate code via the `DECIBRI_PYTHON_VERSION` environment variable.
//!
//! `pyproject.toml` is the single source of truth for the package version.
//! `VersionInfo.binding` in `lib.rs` reads the emitted value, so the two never
//! drift.
//!
//! # Locating pyproject.toml
//!
//! Two build layouts put the file in different places relative to this crate's
//! manifest, so both are searched in order:
//!
//! 1. `CARGO_MANIFEST_DIR/pyproject.toml`, the in-repo layout, where the file
//!    sits beside `bindings/python/Cargo.toml`.
//! 2. `CARGO_MANIFEST_DIR/../../pyproject.toml`, the source-distribution
//!    layout, where maturin packs `pyproject.toml` at the archive root and the
//!    crate keeps its `bindings/python/` path.
//!
//! Exactly one `pyproject.toml` is tracked, so the two paths never both
//! resolve within a single build.
//!
//! The change-tracking directive (`cargo:rerun-if-changed`) names a candidate
//! only when it exists. Cargo treats a named path that is absent as always
//! stale, so naming the other layout's candidate would rerun this script, and
//! relink the extension, on every build.
//!
//! # Parsing
//!
//! The parse is anchored to the `[project]` table and to the `version` key
//! within it. `pyproject.toml` holds other keys that a looser search matches:
//! `python_version` under `[tool.mypy]`, `requires-python` under `[project]`,
//! and the version constraints inside dependency specifiers.
//!
//! Every failure is a build failure. No candidate path readable, no match, or
//! more than one match all panic with a message naming the cause. There is no
//! default and no fallback value: a wrong version reported silently is worse
//! than a build that stops.

use std::fs;
use std::path::PathBuf;

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo"),
    );

    let mut candidates = vec![manifest_dir.join("pyproject.toml")];
    if let Some(sdist_root) = manifest_dir.parent().and_then(|p| p.parent()) {
        candidates.push(sdist_root.join("pyproject.toml"));
    }

    println!("cargo:rerun-if-changed=build.rs");
    // Only a candidate present on disk is named: cargo treats an absent
    // `rerun-if-changed` path as always stale, which would rerun this script
    // on every build in the layout that lacks it.
    for candidate in candidates.iter().filter(|c| c.exists()) {
        println!("cargo:rerun-if-changed={}", candidate.display());
    }

    let (path, content) = candidates
        .iter()
        .find_map(|c| fs::read_to_string(c).ok().map(|content| (c, content)))
        .unwrap_or_else(|| {
            panic!(
                "decibri-python build.rs: no readable pyproject.toml at any of {:?}",
                candidates
            )
        });

    let version = match project_versions(&content).as_slice() {
        [v] => v.clone(),
        [] => panic!(
            "decibri-python build.rs: no `version` key found in the [project] table of {}",
            path.display()
        ),
        found => panic!(
            "decibri-python build.rs: found {} `version` keys in the [project] table of {}: {:?}; \
             exactly one is required",
            found.len(),
            path.display(),
            found
        ),
    };

    println!("cargo:rustc-env=DECIBRI_PYTHON_VERSION={version}");
}

/// Collect every `version = "..."` value declared directly in the `[project]`
/// table.
///
/// Sub-tables such as `[project.urls]` and `[project.optional-dependencies]`
/// are separate tables and are not searched. Values inside multi-line arrays
/// (`classifiers`, `authors`, `dependencies`) carry no bare `version` key, so
/// the key-equality check excludes them.
///
/// Returns every match rather than the first, so the caller can reject an
/// ambiguous file instead of guessing.
fn project_versions(content: &str) -> Vec<String> {
    let mut versions = Vec::new();
    let mut in_project = false;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            in_project = trimmed == "[project]";
            continue;
        }
        if !in_project {
            continue;
        }

        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        if key.trim() != "version" {
            continue;
        }
        if let Some(v) = quoted_value(value) {
            versions.push(v);
        }
    }

    versions
}

/// Extract the contents of the first double-quoted string in a TOML value,
/// tolerating surrounding whitespace and a trailing comment.
fn quoted_value(value: &str) -> Option<String> {
    let after_quote = value.trim_start().strip_prefix('"')?;
    let (inner, _) = after_quote.split_once('"')?;
    Some(inner.to_string())
}
