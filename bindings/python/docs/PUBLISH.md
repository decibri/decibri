# Publishing decibri to PyPI

This document describes how decibri Python wheels are published to PyPI and TestPyPI via the [`publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml) GitHub Actions workflow. The workflow uses Trusted Publisher OIDC (no API tokens) and produces PEP 740 attestations via Sigstore.

Phase 10 (2026-05-02) shipped this workflow. The first publish (decibri 0.1.0a1 to TestPyPI) is Phase 11 work.

## Tag patterns

decibri's repository is polyglot: the Rust core, npm package, and Python wheel each have their own version cadences. Tag namespacing keeps each registry's publish workflow scoped to its own tags.

| Tag pattern              | Workflow                                                          | Registry                  |
| ------------------------ | ----------------------------------------------------------------- | ------------------------- |
| `python-v0.1.0a1`        | [`publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml) | TestPyPI (alpha)          |
| `python-v0.1.0b1`        | [`publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml) | TestPyPI (beta)           |
| `python-v0.1.0rc1`       | [`publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml) | TestPyPI (release candidate) |
| `python-v0.1.0`          | [`publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml) | PyPI (stable)             |
| `v3.4.0`, `v3.5.0`, etc. | [`release.yml`](../../../.github/workflows/release.yml)           | npm + crates.io           |

All `python-v*` patterns are PEP 440 compliant (the `python-v` prefix is workflow-routing only; the `v0.1.0a1` portion is what PyPI sees as the version after the wheel filename strips the namespace). Detection of TestPyPI vs PyPI is done via `contains(github.ref_name, 'a' | 'b' | 'rc')` inside the workflow `if:` conditions.

`release.yml`'s tag filter explicitly excludes `python-v*` (`'!python-v*'` in its trigger), so the npm + crates.io workflow does not fire on Python wheel tags.

## End-to-end publish flow

1. Bump the wheel version in [`bindings/python/Cargo.toml`](../Cargo.toml) (the binding crate version becomes the wheel version via maturin).
2. Update [`bindings/python/CHANGELOG.md`](../CHANGELOG.md) with the new version's entry.
3. Commit and push to `main` (typically via squash-merge from `development`).
4. Create and push the tag:
   ```bash
   git tag python-v0.1.0a1
   git push origin python-v0.1.0a1
   ```
5. CI runs `publish-pypi.yml`:
   - `build-and-validate` matrix builds wheels across 4 platforms (ubuntu-latest, ubuntu-24.04-arm, macos-14, windows-latest). x86_64-apple-darwin (macos-13) was dropped 2026-05-03 because the Intel Mac platform is deprecated (last shipped 2020-2022; macOS 26 dropped support for many Intel models) and the macos-13 GitHub-hosted runner pool is severely under-provisioned. Intel Mac users can install from source via `pip install decibri --no-binary :all:`.
   - Each platform: maturin build, auditwheel/delocate repair, abi3audit gate, install-test in clean venv with `CI=true`, upload artifact
   - `fail-fast: true` means any platform failure blocks both publish jobs
6. Tag-pattern routing kicks in:
   - Prerelease tags (`python-v*a*`, `python-v*b*`, `python-v*rc*`): `publish-testpypi` job runs after `build-and-validate` succeeds. The `testpypi` GitHub environment has a 1-minute wait timer.
   - Stable tags (`python-v\d+.\d+.\d+`): `publish-pypi` job runs after `build-and-validate` succeeds. The `pypi` GitHub environment requires reviewer approval. Production PyPI publish blocks until approved.
7. PyPI / TestPyPI receives the wheel via Trusted Publisher OIDC. Sigstore attestations generated automatically.
8. The project page on PyPI / TestPyPI shows the new version with the attestation badge.

## Trusted Publisher setup (already configured)

For future-maintainer reference; do NOT need to redo unless the workflow filename changes. TPP records key on the workflow filename and the GitHub environment name; they survive all other workflow edits.

**PyPI** (`https://pypi.org/manage/project/decibri/settings/publishing/`):

| Field | Value |
| ----- | ----- |
| Owner | `decibri` |
| Repository | `decibri` |
| Workflow | `publish-pypi.yml` |
| Environment | `pypi` |

**TestPyPI** (`https://test.pypi.org/manage/project/decibri/settings/publishing/`):

| Field | Value |
| ----- | ----- |
| Owner | `decibri` |
| Repository | `decibri` |
| Workflow | `publish-pypi.yml` |
| Environment | `testpypi` |

**GitHub `pypi` environment** (repo Settings -> Environments):
- Required reviewer: `rossarmstrong`
- Deployment tag rule: `python-v*.*.*` (stable tags only; prereleases blocked)

**GitHub `testpypi` environment**:
- Wait timer: 1 minute
- Deployment tag rule: `python-v*` (all tags including prereleases)

If the workflow filename ever changes from `publish-pypi.yml`, the TPP records on BOTH registries must be deleted and re-added. The OIDC trust chain breaks otherwise.

## Dry-run procedure (workflow_dispatch)

For workflow file changes pre-tag, run the workflow manually from the Actions UI:

1. Push the workflow file change to `development` (or `main`).
2. Navigate to `https://github.com/decibri/decibri/actions/workflows/publish-pypi.yml`.
3. Click "Run workflow" -> select the branch -> confirm.
4. The `build-and-validate` matrix runs across all 4 platforms.
5. The `publish-testpypi` and `publish-pypi` jobs both skip cleanly because their `if:` conditions key on `github.event_name == 'push'`.

If `build-and-validate` passes on all 4 platforms during a manual dispatch, the workflow file is wired correctly. If a platform fails, fix before tagging.

## Failure recovery

### abi3audit fails

Symptom: `build-and-validate` matrix entry fails on the "Verify abi3 compliance" step with output like `1 ABI version mismatches and N ABI violations found`.

Cause: a Rust dependency added a non-abi3 export that ended up in the compiled extension. Phase 10 prep research empirically verified the existing wheel passes; a regression here is from a Phase 8 / Phase 9 / later change.

Recovery: inspect `Cargo.lock` for new dependencies since the last successful publish; check whether the new dep links a non-abi3 symbol (`pyo3-build-config` output is the usual diagnostic). Pin the dep to a known-good version or remove the offending export.

### Install-test fails on a specific platform

Symptom: `build-and-validate` matrix entry fails on "Wheel install-test in clean venv" for a single platform (commonly macOS due to rpath, or Linux due to manylinux compatibility).

Recovery: check the platform-specific build logs:
- **macOS:** look for `LC_LOAD_DYLIB` / install-name issues; verify `delocate-listdeps` output shows only `/System/Library/Frameworks/...` deps. A new external dep slipping into the wheel needs a delocate rerun.
- **Linux:** look for auditwheel violations; run `auditwheel show` on the failing wheel to see what symbol versioning broke. If a new glibc symbol crept in, the manylinux floor needs to bump (currently `manylinux_2_28`).
- **Windows:** look for missing CRT or Windows SDK DLLs in the import table; `dumpbin /imports` output during build shows what's pulled in.

### Trusted Publisher rejects OIDC token

Symptom: `publish-testpypi` or `publish-pypi` fails with an error like `the OIDC token itself is well-formed... but doesn't match any known (pending) OIDC publisher`.

Cause: workflow filename in the TPP record does not match the running workflow's filename. This is the documented contract per [PyPI's troubleshooting docs](https://docs.pypi.org/trusted-publishers/troubleshooting/).

Recovery: verify the TPP record on PyPI / TestPyPI matches:
- Owner: `decibri`
- Repository: `decibri`
- Workflow: `publish-pypi.yml` (basename, not full path)
- Environment: `pypi` or `testpypi` matching the failing job

If the workflow file was renamed, delete the TPP record and add a new one with the new filename. Same on the other registry.

### Required reviewer block on `pypi` environment

Symptom: `publish-pypi` job sits in "Waiting for review" state.

This is intentional gating per the GitHub `pypi` environment configuration. The required reviewer approves via the Actions UI's reviewer prompt before the publish proceeds.

To remove the gate (e.g., at 0.2.0+ when the workflow is proven), edit the `pypi` environment in repo Settings -> Environments and remove the required reviewer or replace with a wait timer.

### Tag pushed to wrong workflow

Symptom: tag does not match the expected `python-v*` namespace, so `publish-pypi.yml` does not fire (or `release.yml` fires when it shouldn't).

Recovery: delete the tag locally and remotely, push the correctly-namespaced tag:
```bash
git tag -d python-v0.1.0a1
git push origin --delete python-v0.1.0a1
git tag python-v0.1.0a1 <correct-sha>
git push origin python-v0.1.0a1
```

Note: deleting a tag does NOT undo a successful PyPI publish. If the wheel already uploaded, you must bump the version (PyPI does not allow re-uploading the same version).

## Phase 11 alpha cycle expectations

Phase 11 (next phase) ships the first real publishes:

1. Push `python-v0.1.0a1`. CI runs build-and-validate (4 platforms); abi3audit + install-test gate; `publish-testpypi` runs. Wheel lands on TestPyPI.
2. Manual validation: in a fresh local venv, run `pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ decibri==0.1.0a1` and exercise the public API (sync construct, async construct, VAD path).
3. If issues surface: fix on `main`, bump to `python-v0.1.0a2`, push, repeat.
4. When the alpha is clean and the user-facing surface is finalized, push `python-v0.1.0`. CI runs build-and-validate; the `publish-pypi` job blocks on reviewer approval; on approval, the wheel ships to production PyPI.

The alpha cycle's purpose is to catch issues that the install-test gate cannot reproduce: real-world `pip install` from a registry, dependency resolution conflicts with user environments, missing classifiers in the project metadata, README rendering on the PyPI project page, and so on.

## Implementation references

- Workflow file: [`.github/workflows/publish-pypi.yml`](../../../.github/workflows/publish-pypi.yml)
- Existing matrix patterns mirrored: [`.github/workflows/python-ci.yml`](../../../.github/workflows/python-ci.yml) lines 393-601 (manylinux container build, auditwheel, delocate) and lines 603-670 (install-test gate)
- Tag-routing rationale: `~/.claude/plans/phase-10-design-adjudication.md` (Option D verdict; cohort survey of ruff, uv, polars, tokenizers, pydantic-core)
- abi3audit empirical verification: `~/.claude/plans/phase-10-prep-research.md` Section 2.3 (existing decibri wheel passes `abi3audit --strict --assume-minimum-abi3 3.10` with 0 violations and 0 mismatches)
- PEP 740 attestation specification: https://peps.python.org/pep-0740/
- pypa/gh-action-pypi-publish v1.14: https://github.com/pypa/gh-action-pypi-publish (attestations on by default for OIDC publishes)
- PyPI Trusted Publisher docs: https://docs.pypi.org/trusted-publishers/
