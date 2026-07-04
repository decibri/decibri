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

To build from source you will need:

- **Rust** (latest stable) via [rustup](https://rustup.rs/)
- **Node.js 20 or later**
- Platform-specific dependencies:
  - **Windows**: none
  - **macOS**: none
  - **Linux**: `sudo apt-get install libasound2-dev`

### Setting up the development environment

1. Fork this repository to your own account and clone it to your local machine:

   ```bash
   git clone https://github.com/YOUR_USERNAME/decibri.git
   cd decibri
   ```

2. Build the Rust crate and run its tests:

   ```bash
   cargo build --workspace
   cargo test-decibri
   ```

3. Build the Node.js native addon:

   ```bash
   cd npm/decibri
   npm install --ignore-optional
   npm run build
   ```

4. Run the CI-safe Node.js test suite (no hardware required):

   ```bash
   cd ../..
   node tests/test-ci.js
   ```

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

### Workspace layout

- `crates/decibri/`: Rust crate
- `bindings/node/`: Node.js bindings
- `npm/decibri/`: JavaScript package (Node.js + browser)
- `npm/platform-*/`: platform-specific binary packages

### Dependencies

The workspace `Cargo.lock` is committed so CI builds from a fixed dependency
resolution. It holds `bitflags` at 2.11.0: `bitflags` 2.12.0 expands the
`dispatch2` flag set past the default macro recursion limit and fails to compile
for Apple targets. If a `cargo update` raises `bitflags` and the macOS build
fails with a recursion-limit error in `dispatch2`, hold `bitflags` at 2.11.0.

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
   - `cargo test-decibri` for Rust changes
   - `cd npm/decibri && npm run build && cd ../.. && node tests/test-ci.js` for Node.js binding changes
   - `npx vitest run` for browser-side changes
   - Live hardware tests if your change affects capture or playback behavior
   - Update `npm/decibri/src/decibri.d.ts` if your change affects the public API surface
5. Run the linters:
   - `cargo clippy --workspace -- -D warnings`
   - `cargo fmt --all -- --check`
6. Commit your changes with a clear and descriptive commit message.
7. Push your branch to your fork.
8. Open a pull request against the `main` branch of this repository. Include a description of your changes, the reasons for them, and the benefits they provide.

Our team will review your PR and provide feedback. We may ask for additional changes, so please be prepared to iterate before merging.

### CI pipeline

Every pull request runs an automated CI pipeline covering lint, format, security audit, and test suites across Rust, Node.js, and browser targets. Your PR must pass CI before it can be merged. Details of the pipeline live in `.github/workflows/ci.yml`.

We appreciate your contributions and thank you for your time in submitting a pull request.

## Contributor License Agreement

Before your first contribution can be merged, we ask you to agree to the decibri Contributor License Agreement. It is a one-time step that lets the project include your work under its current and future licenses, with clear provenance, and it does not take away your copyright in what you contribute. You are welcome to read the full agreements first: the [Individual CLA](https://github.com/decibri/decibri-cla-action/blob/main/agreements/Individual-CLA-v1.md) and, for contributions made on behalf of a company, the [Corporate CLA](https://github.com/decibri/decibri-cla-action/blob/main/agreements/Corporate-CLA-v1.md).

When you open a pull request, an automated check looks at whether you are already covered. If you are not, it leaves a comment with a short sentence to agree to. Reply with that exact sentence as a comment on your own pull request, and the check turns green. That is the whole process, and once you have done it you are covered for your future contributions too. Until the check passes, the pull request cannot be merged.

If you are contributing as part of your work, your employer may need a Corporate CLA on file instead of an individual one. If that applies to you, or the check asks about it, contact the maintainers and we will sort it out.

The record we keep is deliberately minimal: your GitHub username and account ID, which version of the agreement you agreed to, and the date. How we handle that information, and how to request its removal, is set out in our [Privacy Policy](https://decibri.com/privacy).

The CLA covers your contributions across the decibri organisation's repositories, so you only need to agree once.

## License

The decibri source is released under the [Apache License 2.0](https://github.com/decibri/decibri/blob/main/LICENSE).

Contributions are governed by the Contributor License Agreement described above. Under the CLA you keep your copyright in what you contribute and grant the project the rights it needs to include and license your work, including under future licenses. Contributed code or content must be your own work, and you confirm that you have the right to grant those rights.
