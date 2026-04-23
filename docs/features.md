# decibri feature flags

Reference guide to the Cargo features exposed by the `decibri` crate.
Aimed at Rust crate consumers evaluating which feature set to build with,
and at FFI binding authors (Node, Python, future mobile) who need to
understand the ORT distribution tradeoffs.

## Full feature list

| Flag | Default | Purpose |
|------|---------|---------|
| `capture` | on | Microphone input stream support (pulls in `cpal`, `crossbeam-channel`) |
| `output` | on | Speaker output stream support (pulls in `cpal`, `crossbeam-channel`) |
| `vad` | on | Silero VAD ONNX inference (pulls in `ort`) |
| `denoise` | on | Reserved stub for future DSP work; no runtime cost when off |
| `gain` | on | Reserved stub for future DSP work; no runtime cost when off |
| `ort-load-dynamic` | **on** | ORT loaded at runtime from a user-supplied path |
| `ort-download-binaries` | off | ORT downloaded at build time, statically embedded |
| `coreml` | off | ORT CoreML execution provider passthrough (macOS) |
| `cuda` | off | ORT CUDA execution provider passthrough |
| `directml` | off | ORT DirectML execution provider passthrough (Windows) |
| `rocm` | off | ORT ROCm execution provider passthrough (Linux AMD) |

The default feature set is
`["capture", "output", "vad", "denoise", "gain", "ort-load-dynamic"]`,
giving the full decibri experience with runtime ORT loading. This is the
set decibri's own Node.js and Python bindings build against.

## Choosing an ORT distribution mode

The `vad` feature pulls in the [`ort`](https://docs.rs/ort) crate, which
wraps Microsoft's ONNX Runtime. ORT is a ~13.5 MB C++ library that
decibri invokes for Silero VAD inference. Two mutually-exclusive Cargo
features control how ORT reaches the running binary:

- **`ort-load-dynamic`** (default): decibri is built without a specific
  ORT. At runtime, a path to `onnxruntime.{dll,dylib,so}` is supplied via
  `VadConfig::ort_library_path` or the `ORT_DYLIB_PATH` environment
  variable. The path is loaded on the first `SileroVad::new` call.

- **`ort-download-binaries`** (opt-in): the `ort` crate's build script
  downloads the appropriate ORT binary from Microsoft's GitHub release
  mirror and statically embeds it. No runtime path needed; no
  `ORT_DYLIB_PATH` lookup; consumers of your binary get ORT for free.

Selecting both at once is a compile error, enforced by a
`compile_error!` in `crates/decibri/src/lib.rs`.

### Tradeoffs

| Aspect | `ort-load-dynamic` | `ort-download-binaries` |
|--------|---------------------|-------------------------|
| Binary size | ~800 KB (ORT not embedded) | ~13.5 MB larger (ORT embedded) |
| Build-time network | Not required | Downloads ORT from GitHub during build |
| First `SileroVad::new` call | Loads ORT dylib from path; fails fast if path invalid | No dylib load (ORT already linked) |
| User experience | Consumer must have ORT available (bundled or system-installed) | Works out of the box |
| Cross-compilation | Simpler (no target-specific ORT binary needed at build time) | Requires matching ORT binary for the target platform |

### Why decibri's own bindings use `ort-load-dynamic`

The Node.js and Python bindings distribute ORT as a separate
platform-specific file inside the published package. For Node, the file
sits in each `npm/platform-*` optional-dependency package as
`onnxruntime.{dll,dylib,so}`. For Python, the file sits inside each
wheel's package data. The JS or Python wrapper resolves the bundled
path at runtime and passes it into Rust as `ort_library_path`.

This pattern lets binding authors:

- Keep the native `.node` / `.so` / `.pyd` artifact small (easier to
  review, faster to download).
- Update the bundled ORT version independently of the Rust source
  (swap the dylib file, no Rust rebuild required).
- Support multiple platforms from one Rust build (each platform package
  ships its own dylib).

Static linking via `ort-download-binaries` would grow every native
artifact by ~13.5 MB and complicate the cross-platform publishing
workflow.

### When a Rust crate consumer should pick `ort-download-binaries`

Pick `ort-download-binaries` when your Rust binary is the deliverable
and you want a single self-contained executable:

- CLI tools distributed via `cargo install <yourcrate>`, where users
  have no wrapper layer to bundle ORT. Static linking is the cleanest
  UX.
- Server processes on known platforms where ORT version pinning is a
  feature, not a constraint.
- Embedded deployments with constrained network access or filesystem
  layouts.

Pick `ort-load-dynamic` (the default) when:

- Authoring an FFI binding (Node, Python, mobile, etc.) with a wrapper
  layer that can distribute ORT alongside the native artifact.
- Sharing a single ORT install across multiple decibri-based binaries
  on the same host.
- Needing to update ORT independently of the Rust source (a critical
  CVE fix in ORT, for example).

To switch modes in a consumer crate, disable default features and
enable the one you want:

```toml
# Cargo.toml of a consumer
[dependencies]
decibri = { version = "3.3", default-features = false, features = [
    "capture",
    "output",
    "vad",
    "denoise",
    "gain",
    "ort-download-binaries",
] }
```

## Execution provider features

The four execution provider features (`coreml`, `cuda`, `directml`,
`rocm`) are passthroughs to the underlying `ort` crate's equivalents.
They enable GPU and accelerator backends in ONNX Runtime. They compose
with either ORT distribution mode, but the ORT binary in use must
itself include the relevant provider:

- `coreml`: macOS only. Present in Microsoft's default ORT releases
  for macOS.
- `cuda`: requires an ORT binary built with CUDA support. Microsoft's
  default ORT releases do NOT include CUDA; you need the GPU variant
  and typically combine this feature with `ort-load-dynamic` plus a
  manually-installed ORT-GPU dylib.
- `directml`: Windows only. Present in Microsoft's default ORT
  releases for Windows x64.
- `rocm`: Linux AMD GPU only. Similar to CUDA, requires a custom ORT
  build.

decibri's own bindings do not enable any execution provider by
default. Silero VAD is light enough to run in real time on CPU across
all supported platforms.

## Feature constraints

- `capture` and `output` both depend on `cpal`. Disabling both removes
  all cpal code. Without either you have only VAD and sample
  conversion, which is rarely useful on its own.
- `vad` pulls in `ort` as an optional dependency (`dep:ort`). Without
  `vad`, the `ort-load-dynamic` / `ort-download-binaries` features are
  no-ops since the `ort` crate is not in the dependency graph.
- **Known constraint**: `crates/decibri/src/error.rs` references
  `ort::Error` in several variant definitions without feature gating,
  so building with `--no-default-features --features capture,output`
  (attempting to skip `vad`) currently fails to compile. Production
  consumers always enable `vad`; feature-gating the `ort::Error`
  references is tracked as future cleanup.
- `denoise` and `gain` compile to empty stub modules today. They
  exist to reserve the feature names against a future
  API-surface-stable release.
- `docs.rs` builds the published documentation with
  `["capture", "output", "vad", "denoise", "gain", "ort-download-binaries"]`
  so docs.rs builders do not need a runtime ORT dylib. See
  `[package.metadata.docs.rs]` in `crates/decibri/Cargo.toml`.

## Related documentation

- Crate-level rustdoc: `cargo doc --package decibri --open`, or
  [docs.rs/decibri](https://docs.rs/decibri).
- Per-release feature changes: `CHANGELOG.md` at the repo root.
- Build configuration for docs.rs: `[package.metadata.docs.rs]` in
  `crates/decibri/Cargo.toml`.
