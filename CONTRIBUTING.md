# Welcome to the Decibri Contribution Guide

Thank you for investing your time in contributing to decibri. We welcome all sorts of different contributions.

Before making any type of contribution, please read our [Code of Conduct](https://github.com/decibri/decibri/blob/main/CODE_OF_CONDUCT.md) to keep our community approachable and respectable.

This guide walks through the contribution workflow, from opening an issue to submitting a pull request.


## New contributor resources

For a good overview of the project, please first read the [README](https://github.com/decibri/decibri/blob/main/README.md). General resources for getting started with open-source contributions:

- [Finding ways to contribute to open source on GitHub](https://docs.github.com/en/get-started/exploring-projects-on-github/finding-ways-to-contribute-to-open-source-on-github)
- [Collaborating with pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests)


## Ways to contribute

There are multiple ways you can contribute to this project:

- Reporting a bug
- Submitting a fix
- Suggesting new features or improvements
- Adding or updating documentation
- Adding or improving integration guides
- Improving test coverage
- Anything else we may have forgotten


## Getting started

### Prerequisites

decibri is built in Rust with Node.js bindings via napi-rs. To build from source you will need:

- **Rust** (latest stable) via [rustup](https://rustup.rs/)
- **Node.js 20 or later**
- Platform-specific audio development headers:
  - **Windows**: no extra dependencies (cpal uses WASAPI via the Windows SDK)
  - **macOS**: no extra dependencies (cpal uses CoreAudio via the system SDK)
  - **Linux**: `libasound2-dev` (`sudo apt-get install libasound2-dev`)

You do NOT need Python, node-gyp, or a C/C++ compiler toolchain. The previous v1.x C++/PortAudio implementation has been replaced by a native Rust implementation using [cpal](https://crates.io/crates/cpal).

### Setting up the development environment

1. Fork this repository to your own account and clone it to your local machine:
   ```bash
   git clone https://github.com/YOUR_USERNAME/decibri.git
   cd decibri
   ```

2. Build the Rust crate and run its tests:
   ```bash
   cargo build --workspace
   cargo test -p decibri
   ```

3. Build the Node.js native addon:
   ```bash
   cd npm/decibri
   npm install --ignore-optional
   npm run build
   ```
   The `--ignore-optional` flag skips the prebuilt platform packages (which only exist on npm) and forces a local build via napi-rs.

4. Run the CI-safe Node.js test suite (no hardware required):
   ```bash
   cd ../..
   node tests/test-ci.js
   ```
   This runs approximately 38 static assertions that verify the module loads, error messages are correct, and the API surface is intact.

5. (Optional) Run the live capture, output, and VAD tests if you have a microphone and speakers connected:
   ```bash
   node tests/test-capture.js
   node tests/test-output.js
   node tests/test-vad-silero.js
   node tests/test-api.js
   ```

6. (Optional) Run the browser unit tests (mocked, no hardware required):
   ```bash
   npm install
   npx vitest run
   ```

### Architecture overview

The project is a Cargo workspace with three layers:

- **Rust crate** (`crates/decibri/`): Core audio capture, playback, and VAD logic built on [cpal](https://crates.io/crates/cpal) and [ort](https://crates.io/crates/ort). Published to [crates.io](https://crates.io/crates/decibri) as the `decibri` crate.
- **Node.js bindings** (`bindings/node/`): napi-rs wrapper exposing the Rust crate to Node.js. Produces platform-specific `.node` binaries (Windows x64, macOS ARM64, Linux x64, Linux ARM64).
- **JavaScript API** (`npm/decibri/src/`): Thin JavaScript layer wrapping the native addon as Node.js Readable and Writable streams. Includes browser support via conditional exports (`npm/decibri/src/browser/`).

Platform-specific packages are published under the `@decibri` scope (`@decibri/decibri-win32-x64-msvc`, etc.) and resolved automatically via npm's `optionalDependencies`.

### Reporting a bug

We use GitHub Issues to track bugs. All open, pending, and closed cases are at [decibri Issue Tracking](https://github.com/decibri/decibri/issues).

Before opening a new issue, please search [existing issues](https://github.com/decibri/decibri/issues) to see if the bug has already been reported. You may be able to add more information or your own experience to an existing issue.

If no related issue exists, you can open a new one using the [issues form](https://github.com/decibri/decibri/issues/new).

To help us reproduce and fix bugs quickly, please include the following where applicable:

- A quick summary and background
- Your operating system and architecture (e.g. Windows 11 x64, macOS arm64, Ubuntu 22.04 x64)
- Node.js version (`node --version`)
- Rust version if building from source (`rustc --version`)
- Steps to reproduce the bug
- Code samples that trigger the issue
- What you expected to happen vs what actually happened
- Exact error messages (screenshots are fine, de-identified if needed)

### Proposing codebase changes

We welcome contributions from everyone interested in making decibri better. To propose a change:

1. Fork this repository and clone it to your local machine.
2. Create a new branch from `main` with a descriptive name that reflects your changes.
3. Make your changes.
4. Test your changes thoroughly:
   - `cargo test -p decibri` for Rust changes
   - `cd npm/decibri && npm run build && cd ../.. && node tests/test-ci.js` for Node.js binding changes
   - `npx vitest run` for browser-side changes
   - Live hardware tests if your change affects capture or playback behavior
5. Run the linters:
   - `cargo clippy --workspace -- -D warnings`
   - `cargo fmt --all -- --check`
6. Commit your changes with a clear and descriptive commit message.
7. Push your branch to your fork.
8. Open a pull request against the `main` branch of this repository. Include a description of your changes, the reasons for them, and the benefits they provide.

Our team will review your PR and provide feedback. We may ask for additional changes, so please be prepared to iterate before merging.

### Important notes for native code changes

- Changes to `crates/decibri/src/` or `bindings/node/src/` require rebuilding the native addon. Run `cd npm/decibri && npm run build` to regenerate the `.node` file and the napi-generated `index.js` loader.
- Prebuilt platform binaries for all four supported targets (Windows x64, macOS ARM64, Linux x64, Linux ARM64) are produced by the CI pipeline on release tags. You do not need to produce prebuilt binaries for a PR.
- If your change affects the public API surface, please update both the Rust crate docs and `npm/decibri/src/decibri.d.ts` to match.
- Run `cargo fmt --all` before committing to ensure consistent formatting. CI enforces formatting via `cargo fmt --all -- --check`.

### CI pipeline

Every pull request runs an automated CI pipeline covering lint, format, security audit, and test suites across Rust, Node.js, and browser targets. Your PR must pass CI before it can be merged. Details of the pipeline live in `.github/workflows/ci.yml`.

We appreciate your contributions and thank you for your time in submitting a pull request.


## License

By contributing to this repository, you agree to license your contributions under the [Apache License 2.0](https://github.com/decibri/decibri/blob/main/LICENSE).

Any contributed code or content must be your original work, and you warrant that you have the right to license it under the terms of the Apache License 2.0.

By contributing, you also acknowledge that your contribution will be included in the project under the same license as the rest of the repository.
