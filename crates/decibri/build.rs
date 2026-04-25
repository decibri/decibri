//! Build script: extracts the resolved cpal version and exposes it to crate
//! code via the `DECIBRI_CPAL_VERSION` environment variable.
//!
//! This is the single-source-of-truth mechanism for the `CPAL_VERSION` const
//! in `lib.rs`. Previously, the cpal version string was hardcoded in the
//! Node binding's `version()` function (`"cpal 0.17".to_string()`, in two
//! places): duplicated state that would silently drift when the workspace
//! bumped cpal. By reading the resolved version here and emitting a
//! build-time env var, every binding pulls `decibri::CPAL_VERSION` from one
//! place. When the workspace bumps cpal, Cargo.toml updates, this script
//! re-runs, and all downstream consumers see the new value automatically.
//!
//! # Resolution strategy (belt-and-suspenders, in priority order)
//!
//! 1. **`DECIBRI_CPAL_VERSION` env var override.** If set, used verbatim and
//!    no parsing occurs. Provides an escape hatch for unforeseen build
//!    contexts and supports release-pipeline injection if ever needed.
//! 2. **Cargo.toml at `CARGO_MANIFEST_DIR`.** Always present in any build
//!    context (workspace, `cargo publish` verify, `cargo install` consumer).
//!    Parses the cpal dependency entry; handles the four cargo-emitted
//!    forms (literal string, inline table with `version = "..."`, workspace
//!    inherit, and dedicated `[dependencies.cpal]` table).
//! 3. **Workspace `Cargo.toml` fallback** (in-workspace builds only). When
//!    the manifest's cpal entry is `{ workspace = true, ... }`, the version
//!    lives in the workspace `[workspace.dependencies]` block at
//!    `../../Cargo.toml`. Read and parse from there.
//! 4. **Hardcoded `"0.17"` constant fallback** with `cargo:warning=`
//!    visibility. Belt-and-suspenders for unforeseen build contexts where
//!    none of the above paths succeed. Prevents a build failure but emits
//!    a loud warning so the fallback is visible in build output.
//!
//! # Why Cargo.toml is the primary source (changed from Cargo.lock in v3.3.2)
//!
//! Previous build.rs (v3.3.0 - v3.3.1) read the workspace `Cargo.lock` via
//! `..\..\Cargo.lock` traversal from `CARGO_MANIFEST_DIR`. This worked in
//! workspace builds but panicked in `cargo publish` verify because the
//! published tarball is flat (`decibri-X.Y.Z/Cargo.toml` and
//! `decibri-X.Y.Z/Cargo.lock` are siblings, not in workspace structure);
//! the traversal landed at `target/Cargo.lock` (nonexistent). The v3.3.1
//! crates.io publish failed on this defect.
//!
//! Cargo.toml is unambiguously present in every build context because cargo
//! guarantees `CARGO_MANIFEST_DIR` points at the manifest's directory.
//! Cargo's publish-time normalization fully resolves `{ workspace = true }`
//! to literal versions in the published Cargo.toml, so workspace-fallback
//! is only needed for in-workspace builds.
//!
//! Output format: major.minor only (`"0.17"`, not `"0.17.0"`). Preserves
//! byte-identical output with the pre-refactor hardcoded string. If a
//! future scope decision wants the full resolved version, change
//! `truncate_to_major_minor` to return the unmodified string.

use std::fs;
use std::path::{Path, PathBuf};

const FALLBACK_VERSION: &str = "0.17";

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by cargo"),
    );
    let manifest_toml = manifest_dir.join("Cargo.toml");

    println!("cargo:rerun-if-env-changed=DECIBRI_CPAL_VERSION");
    println!("cargo:rerun-if-changed={}", manifest_toml.display());
    println!("cargo:rerun-if-changed=build.rs");

    // Layer 1: env var override.
    if let Ok(v) = std::env::var("DECIBRI_CPAL_VERSION") {
        if !v.is_empty() {
            emit(&v);
            return;
        }
    }

    // Layer 2: read CARGO_MANIFEST_DIR / Cargo.toml.
    match fs::read_to_string(&manifest_toml) {
        Ok(content) => match resolve_cpal_version(&content, &manifest_dir) {
            Some(v) => {
                emit(&v);
                return;
            }
            None => {
                println!(
                    "cargo:warning=decibri build.rs: cpal version not found in {}; \
                     falling back to hardcoded \"{}\"",
                    manifest_toml.display(),
                    FALLBACK_VERSION
                );
            }
        },
        Err(e) => {
            println!(
                "cargo:warning=decibri build.rs: failed to read {}: {}; \
                 falling back to hardcoded \"{}\"",
                manifest_toml.display(),
                e,
                FALLBACK_VERSION
            );
        }
    }

    // Layer 4: hardcoded fallback.
    emit(FALLBACK_VERSION);
}

fn emit(version: &str) {
    let major_minor = truncate_to_major_minor(version);
    println!("cargo:rustc-env=DECIBRI_CPAL_VERSION={}", major_minor);
}

/// Resolve cpal version from a manifest's content, with workspace fallback.
///
/// Handles all four cargo-emitted forms of the cpal dep:
/// - `cpal = "0.17"` (Form 1: inline string)
/// - `cpal = { version = "0.17", optional = true }` (Form 2: inline table)
/// - `cpal = { workspace = true, optional = true }` (Form 3: workspace inherit)
/// - `[dependencies.cpal]\nversion = "0.17"\n...` (Form 4: dedicated table)
///
/// Forms 1, 2, 4 are self-contained: version returned directly.
/// Form 3 triggers a workspace-fallback read of `../../Cargo.toml`'s
/// `[workspace.dependencies] cpal` entry, which is always Form 1 or Form 2.
fn resolve_cpal_version(manifest_content: &str, manifest_dir: &Path) -> Option<String> {
    if let Some(form) = find_cpal_in_dependencies(manifest_content) {
        match form {
            CpalEntry::Literal(v) => return Some(v),
            CpalEntry::WorkspaceInherit => {
                // Form 3: walk up to workspace Cargo.toml.
                let workspace_toml = manifest_dir.parent()?.parent()?.join("Cargo.toml");
                let ws_content = fs::read_to_string(&workspace_toml).ok()?;
                if let Some(CpalEntry::Literal(v)) =
                    find_cpal_in_workspace_dependencies(&ws_content)
                {
                    return Some(v);
                }
                return None;
            }
        }
    }
    None
}

#[derive(Debug)]
enum CpalEntry {
    Literal(String),
    WorkspaceInherit,
}

/// Search a manifest's `[dependencies]` block (inline forms 1-3) and any
/// `[dependencies.cpal]` table (form 4) for the cpal entry.
fn find_cpal_in_dependencies(content: &str) -> Option<CpalEntry> {
    // Form 4: `[dependencies.cpal]` dedicated table.
    if let Some(v) = extract_version_after_section(content, "[dependencies.cpal]") {
        return Some(CpalEntry::Literal(v));
    }
    // Forms 1-3: inline within `[dependencies]` block.
    extract_inline_cpal_in_section(content, "[dependencies]")
}

/// Search a workspace manifest's `[workspace.dependencies]` block.
/// Workspace deps are always Form 1 or Form 2 (workspace-inherit doesn't
/// nest); WorkspaceInherit is not a valid CpalEntry here.
fn find_cpal_in_workspace_dependencies(content: &str) -> Option<CpalEntry> {
    if let Some(v) = extract_version_after_section(content, "[workspace.dependencies.cpal]") {
        return Some(CpalEntry::Literal(v));
    }
    extract_inline_cpal_in_section(content, "[workspace.dependencies]")
}

/// For a `[section.cpal]` dedicated-table form, walk forward from the
/// section header to the next `[` line, finding `version = "..."`.
fn extract_version_after_section(content: &str, section_header: &str) -> Option<String> {
    let mut in_section = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == section_header {
            in_section = true;
            continue;
        }
        if in_section {
            // Next section header ends this table.
            if trimmed.starts_with('[') && trimmed != section_header {
                return None;
            }
            if let Some(rest) = trimmed.strip_prefix("version") {
                let after_eq = rest.trim_start().strip_prefix('=')?.trim_start();
                if let Some(stripped) = after_eq.strip_prefix('"') {
                    if let Some(v) = stripped.strip_suffix('"') {
                        return Some(v.to_string());
                    }
                }
            }
        }
    }
    None
}

/// For an inline `cpal = ...` line within a `[section]` block, parse:
///   - `cpal = "0.17"` -> Literal("0.17")
///   - `cpal = { version = "0.17", ... }` -> Literal("0.17")
///   - `cpal = { workspace = true, ... }` -> WorkspaceInherit
///
/// Walks lines from `[section]` header until the next `[` section starts.
/// Returns the first cpal line found.
fn extract_inline_cpal_in_section(content: &str, section_header: &str) -> Option<CpalEntry> {
    let mut in_section = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == section_header {
            in_section = true;
            continue;
        }
        if in_section {
            if trimmed.starts_with('[') && trimmed != section_header {
                return None;
            }
            if let Some(value_part) = trimmed.strip_prefix("cpal") {
                let after_eq = value_part.trim_start().strip_prefix('=')?.trim_start();
                return parse_inline_value(after_eq);
            }
        }
    }
    None
}

/// Parse the value side of a `cpal = ...` inline declaration:
///   `"0.17"` -> Literal("0.17")
///   `{ version = "0.17", ... }` -> Literal("0.17")
///   `{ workspace = true, ... }` -> WorkspaceInherit
fn parse_inline_value(value: &str) -> Option<CpalEntry> {
    let trimmed = value.trim();
    // Form 1: bare quoted string.
    if let Some(stripped) = trimmed.strip_prefix('"') {
        if let Some(v) = stripped.split('"').next() {
            return Some(CpalEntry::Literal(v.to_string()));
        }
    }
    // Forms 2/3: inline table.
    if let Some(inner) = trimmed.strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
        // Look for `workspace = true` first (Form 3).
        for part in inner.split(',') {
            let part = part.trim();
            if part.starts_with("workspace") {
                let after_eq = part.split('=').nth(1)?.trim();
                if after_eq == "true" {
                    return Some(CpalEntry::WorkspaceInherit);
                }
            }
        }
        // Otherwise look for `version = "..."` (Form 2).
        for part in inner.split(',') {
            let part = part.trim();
            if let Some(rest) = part.strip_prefix("version") {
                let after_eq = rest.trim_start().strip_prefix('=')?.trim_start();
                if let Some(stripped) = after_eq.strip_prefix('"') {
                    if let Some(v) = stripped.split('"').next() {
                        return Some(CpalEntry::Literal(v.to_string()));
                    }
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
