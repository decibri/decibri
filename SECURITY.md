# Security

Decibri takes security seriously. As a cross-platform audio library that ships prebuilt native binaries for Node.js and a Rust crate, we are especially attentive to supply chain, binary integrity, and runtime safety concerns. If you believe you have found a security vulnerability in this repository, please report it as described below.

## Responsible Disclosure

We are strongly committed to the responsible disclosure of security vulnerabilities. Please follow these guidelines when reporting security issues:

- Email [hello@decibri.com](mailto:hello@decibri.com) with "SECURITY - Decibri" in the subject line.
- Alternatively, use [GitHub's private vulnerability reporting](https://github.com/decibri/decibri/security/advisories/new) to report directly through GitHub.
- Please do not report security vulnerabilities through public GitHub issues.

When reporting, please include the following details where applicable:

- A description of the vulnerability and how it can be exploited
- The affected version of the package or crate
- The platform and architecture (e.g., Windows x64, macOS arm64, Linux x64, Linux arm64)
- Steps to reproduce the issue
- The runtime where the issue occurs (Node.js, browser, Rust)
- Any other relevant information that could help us fix the vulnerability

We review reports as quickly as possible and work with reporters to coordinate remediation and disclosure.

## Supply Chain and Binary Integrity

Decibri ships prebuilt native binaries for four platforms (Windows x64, macOS ARM64, Linux x64, Linux ARM64) across multiple package registries. We take a defense-in-depth approach to supply chain integrity.

### Build integrity

- All binaries are compiled exclusively in GitHub Actions CI on GitHub-hosted runners. No binaries are built or uploaded manually.
- Builds are triggered only by tagged releases from the `main` branch of the `decibri/decibri` repository.
- The release workflow verifies all four platform binaries are present and pass smoke tests before publishing to npm or crates.io.
- The full build configuration is open source and auditable in `.github/workflows/release.yml`.

### Publishing and authentication

- npm publishing uses [Trusted Publishing via OIDC](https://docs.npmjs.com/trusted-publishers/). No long-lived npm tokens are stored in the repository or CI system. Each publish uses a short-lived, workflow-specific credential issued by npm.
- All npm packages (`decibri`, `@decibri/decibri-win32-x64-msvc`, `@decibri/decibri-darwin-arm64`, `@decibri/decibri-linux-x64-gnu`, `@decibri/decibri-linux-arm64-gnu`) are configured to require 2FA and disallow legacy token publishing.
- The Rust crate on crates.io is published from the same CI pipeline with a scoped, crate-specific token stored as a GitHub Actions secret.

### Provenance and attestation

- Every npm package publishes with automatic [npm provenance attestations](https://docs.npmjs.com/generating-provenance-statements/), cryptographically linking each published version to the exact source commit and build workflow.
- Each `.node` native binary is signed with a [SLSA Build Provenance](https://slsa.dev/) attestation via `actions/attest-build-provenance`, providing verifiable evidence of the build environment and inputs.
- Provenance attestations are recorded in the public [Sigstore transparency log](https://search.sigstore.dev/) and can be verified with `npm audit signatures`.

### Dependency monitoring

- Cargo and npm dependencies are continuously monitored by Dependabot for security advisories and version updates.
- `cargo-audit` runs on every PR via the RustSec advisory database.
- `npm audit` runs on every PR against the production dependency tree.
- Upstream critical dependencies (`cpal`, `ort`) are tracked by a weekly automated workflow that opens a GitHub issue when new versions are released.

## Supported Versions

This security policy applies to the following versions:

| Version | Supported                |
|:-------:|:------------------------:|
| 3.x     | :white_check_mark: Yes   |
| < 3.0   | :x: No longer supported  |

Security fixes are applied to the latest 3.x release only. Version 1.x (C++/PortAudio implementation) is no longer maintained. Version 2.x was never published.

## CVE Policy

For confirmed vulnerabilities, we will request a CVE identifier where appropriate and publish a GitHub Security Advisory with details of the issue, affected versions, and remediation steps. Security advisories are visible at the [decibri security advisories page](https://github.com/decibri/decibri/security/advisories).

## Security Best Practices for Users

Follow these best practices when using Decibri in your applications:

- Keep your dependencies up to date regularly
- Only install packages from the official npm registry and crates.io
- Enable two-factor authentication (2FA) on your npm and crates.io accounts
- Verify provenance attestations on installed packages with `npm audit signatures`
- Use `npm audit` and `cargo audit` regularly to check for known vulnerabilities in your dependency tree
- Pin dependencies to specific versions in production environments
- Follow the principle of least privilege when granting microphone access in your application
- For browser deployments, always serve over HTTPS and request audio permissions only when needed
- Review the [Sigstore transparency log](https://search.sigstore.dev/) entries for published versions if you need to verify build origin

## Reporting Concerns About This Policy

If you have questions about this security policy itself, or suggestions for improvement, please open a regular issue on the repository. These are not security vulnerability reports and do not require private disclosure.

## Acknowledgments

Thank you to the researchers and community members who help keep Decibri users secure. If you report a valid vulnerability and would like public acknowledgment, we will credit you in the security advisory and release notes.

## Contact

For security questions, email [hello@decibri.com](mailto:hello@decibri.com) with "SECURITY - Decibri" in the subject line.
