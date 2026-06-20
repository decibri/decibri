//! Internal ONNX session abstraction.
//!
//! Defines the [`OnnxSession`] trait that decibri uses to talk to ONNX Runtime,
//! decoupling the VAD code from the `ort` API behind a small backend-agnostic
//! interface.
//!
//! # Visibility
//!
//! Everything in this module is `pub(crate)`; the trait is not part of the
//! public API.
//!
//! # Trait shape
//!
//! The trait surface mirrors the four operations [`crate::vad`] performs
//! against an `ort` session: builder construction, input tensor creation, run,
//! and output tensor extraction. The closed-enum [`OnnxTensorData`] /
//! [`OnnxTensorOwned`] design avoids an `ndarray` dependency and keeps the VAD
//! hot path (~50 calls per second) free of tensor-type vtable dispatch.
//!
//! # Backend selection
//!
//! [`OnnxSessionBuilder::build`] returns the ORT-backed implementation, which
//! is the only backend.
//!
//! # Error handling
//!
//! The trait reuses [`crate::error::DecibriError`]: the ORT variants cover ORT
//! failures, and [`crate::error::DecibriError::OnnxBackendFailed`] is reserved
//! for non-ORT backend errors.

use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::error::DecibriError;

/// ONNX Runtime execution provider for a session.
///
/// `pub(crate)`: internal. Today only [`ExecutionProvider::Cpu`] is requested
/// (by [`crate::vad::SileroVad`]); the accelerator variants are wired so the
/// per-session build path ([`OnnxSessionBuilder`]) can register them. Each
/// accelerator is gated behind the matching Cargo feature: requesting one whose
/// feature is not compiled in yields a clear
/// [`DecibriError::OnnxBackendFailed`] naming the feature to enable. CPU is
/// always available and is the automatic fallback for every accelerator.
///
/// The accelerator variants are a reserved seam: they are matched in
/// `ort_impl::provider_dispatch` but not yet constructed in non-test code (no
/// consumer selects a non-CPU provider until the bindings expose a device
/// option). This mirrors the reserved-variant convention already used in this
/// module (see `OnnxTensorOwned::I64`), hence `allow(dead_code)`.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub(crate) enum ExecutionProvider {
    /// ORT's built-in CPU provider (MLAS). Always available; needs no Cargo
    /// feature. The default, and the only provider decibri requests today.
    #[default]
    Cpu,
    /// Apple CoreML (macOS, iOS). Gated behind the `coreml` Cargo feature.
    CoreMl,
    /// NVIDIA CUDA. Gated behind the `cuda` Cargo feature.
    Cuda,
    /// Microsoft DirectML (Windows). Gated behind the `directml` Cargo feature.
    DirectMl,
    /// AMD ROCm (Linux). Gated behind the `rocm` Cargo feature.
    Rocm,
}

// ── ONNX Runtime process-global initialization ──────────────────────────────
//
// ORT initialization is a process-global, first-init-wins singleton. It lives
// here (in the capability-neutral `onnx` module) rather than on any single
// model so that every local-ONNX consumer (Silero VAD today, future models)
// shares one ORT environment and one fork guard, with no single model owning
// the process-global state. This is a relocation of the machinery that
// previously lived in `vad.rs`; the first-wins semantics are unchanged.

/// Process-global ORT init tracker. Stores the library path used on first
/// successful init (None if init used `ort::init()` with env-var fallback).
/// Subsequent `init_ort_once` calls see this as populated and return immediately.
static ORT_INIT: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Process id captured alongside [`ORT_INIT`] on first successful init.
/// Compared against the current pid at every Silero inference call by
/// [`check_pid_for_ort`]; a mismatch indicates the process forked after
/// init and ONNX Runtime's internal state is no longer safe to use.
///
/// Recorded via [`OnceLock::set`] in the `Ok` arm of [`init_ort_once`] (after
/// `do_ort_init` returns successfully) so the pid stamp is paired with the
/// successful ORT init, not set speculatively before init returned.
static ORT_INIT_PID: OnceLock<u32> = OnceLock::new();

/// Wrap an ORT init failure with a decibri-specific actionable message.
///
/// Kept as a standalone function (rather than an inline closure in
/// `init_ort_once`) so it can be unit-tested without triggering a real ORT
/// init failure. Tests construct a synthetic `ort::Error` via
/// [`ort::Error::new`].
///
/// Returns one of two typed variants depending on whether a path was provided:
/// [`DecibriError::OrtLoadFailed`] when `path.is_some()`, or
/// [`DecibriError::OrtInitFailed`] otherwise. The inner `ort::Error` is
/// carried via `#[source]` so consumers can walk the error chain.
fn wrap_init_error(path: Option<&Path>, err: ort::Error) -> DecibriError {
    match path {
        Some(p) => DecibriError::OrtLoadFailed {
            path: p.to_path_buf(),
            source: err,
        },
        None => DecibriError::OrtInitFailed { source: err },
    }
}

/// Perform the actual ORT init call. Split out by distribution-mode feature
/// so `init_ort_once` stays single-path.
///
/// Under `ort-load-dynamic`: `ort::init_from(path)` is available and is
/// fallible (validates the dylib up-front).
///
/// Under `ort-download-binaries`: `ort::init_from` does NOT exist. ORT is
/// statically linked into the binary and any path argument is meaningless.
/// The path is ignored; we call `ort::init()` to commit an
/// `EnvironmentBuilder` so our OnceLock bookkeeping fires.
#[cfg(feature = "ort-load-dynamic")]
fn do_ort_init(path: Option<&Path>) -> Result<bool, ort::Error> {
    match path {
        Some(p) => ort::init_from(p).map(|b| b.with_name("decibri").commit()),
        None => Ok(ort::init().with_name("decibri").commit()),
    }
}

#[cfg(not(feature = "ort-load-dynamic"))]
fn do_ort_init(_path: Option<&Path>) -> Result<bool, ort::Error> {
    Ok(ort::init().with_name("decibri").commit())
}

/// Initialize ORT exactly once per process.
///
/// - If ORT is already initialized (by this or any prior caller), returns
///   immediately. The `path` argument is silently ignored. ORT's global
///   state cannot be re-initialized. See `VadConfig::ort_library_path` docs.
/// - Otherwise delegates to the feature-gated `do_ort_init` helper.
/// - On failure, the `OnceLock` is NOT set, so a subsequent caller can retry.
///
/// Note: `e` in the error-wrapping path below is always an `ort::Error` (ORT's
/// own error type), never a `DecibriError`. This function is the single place
/// where ORT errors enter decibri's error hierarchy. Do NOT apply the same
/// wrapping pattern elsewhere or the `decibri:` prefix and guidance string
/// will be duplicated in the message users see.
pub(crate) fn init_ort_once(path: Option<&Path>) -> Result<(), DecibriError> {
    // Fast path: ORT already initialized.
    if ORT_INIT.get().is_some() {
        // Guard fork-after-init at construction time, not only at inference: in
        // a forked child the inherited OnceLock makes this the fast path, and
        // building a session against inherited ORT state is the exact unsafe
        // operation `check_pid_for_ort` exists to prevent. Without this the
        // typed `ForkAfterOrtInit` would not fire until the first inference,
        // after the unsafe session build had already run.
        check_pid_for_ort()?;
        return Ok(());
    }

    // Defensive pre-check (load-dynamic only): verify the path points at a
    // regular file before handing it to `ort::init_from`. A nonexistent path
    // or a path that points at a directory causes `ort::init_from` on Windows
    // to hang indefinitely (reproduced 2026-04-22 against pyke/ort
    // 2.0.0-rc.12 + onnxruntime 1.24.4). Failing fast here turns that hang
    // into a clean typed error that Node, Python, and mobile consumers can
    // surface to users.
    //
    // The check is intentionally scoped to `load-dynamic`: under
    // `download-binaries`, ORT is statically linked and the path argument
    // is ignored, so there is nothing to pre-validate.
    #[cfg(feature = "ort-load-dynamic")]
    if let Some(p) = path {
        if !p.is_file() {
            // Use `OrtPathInvalid`, not `OrtLoadFailed`, precisely because
            // constructing an `ort::Error` here would call `ortsys![
            // CreateStatus]`, which triggers the ORT dylib load that this
            // pre-check is designed to prevent. `OrtPathInvalid` is
            // string-only and never touches ORT symbols.
            return Err(DecibriError::OrtPathInvalid {
                path: p.to_path_buf(),
                reason: "path does not exist or is not a regular file",
            });
        }
    }

    match do_ort_init(path) {
        Ok(_committed) => {
            // First-caller-wins. If another thread set it first, discard ours;
            // ORT's own global init is idempotent (first takes effect).
            let _ = ORT_INIT.set(path.map(|p| p.to_path_buf()));
            // Pair the ORT init with the pid that
            // performed it. Subsequent `check_pid_for_ort` calls compare
            // the current pid against this stamp and raise
            // `ForkAfterOrtInit` on mismatch. `_ = .set(...)` because
            // another thread may have set the pid first; under that race
            // the existing value wins and our value is silently dropped,
            // matching the behavior of ORT_INIT itself.
            let _ = ORT_INIT_PID.set(std::process::id());
            Ok(())
        }
        Err(e) => Err(wrap_init_error(path, e)),
    }
}

/// Verify that the current pid matches the pid that initialized ORT.
/// Called at the start of every Silero inference call to detect Linux
/// `fork()`-after-init silent ORT corruption.
///
/// On Linux, when a parent process initializes ORT (e.g. by constructing
/// `Microphone(vad="silero")`) and then forks a child via Python's
/// default `fork` start method, the child inherits the OnceLock's set
/// state but the underlying ORT runtime data structures are not safe to
/// share. Calling inference in the child without re-initializing
/// produces silent wrong probabilities, segfaults, or hangs depending
/// on the specific ORT configuration.
///
/// This function compares `std::process::id()` against the recorded
/// `ORT_INIT_PID`. On mismatch it returns
/// [`DecibriError::ForkAfterOrtInit`] with both pids attached so the
/// caller can diagnose. On match (the common case: same process that
/// did init is now doing inference) it returns `Ok(())`.
///
/// Returns `Ok(())` cheaply if `ORT_INIT_PID` is unset (ORT was never
/// initialized in this process), so non-VAD code paths pay nothing.
///
/// Pre-fork detection: not in scope here; `init_ort_once` is the single
/// place that sets the pid and runs only when ORT is actually used.
/// Code paths that never construct a Silero VAD never trip this check.
///
/// # Errors
/// - [`DecibriError::ForkAfterOrtInit`] if `current_pid != init_pid`.
pub(crate) fn check_pid_for_ort() -> Result<(), DecibriError> {
    if let Some(&init_pid) = ORT_INIT_PID.get() {
        let current_pid = std::process::id();
        if current_pid != init_pid {
            return Err(DecibriError::ForkAfterOrtInit {
                init_pid,
                current_pid,
            });
        }
    }
    Ok(())
}

/// Borrowed view into a single named input tensor.
///
/// Lifetimes: input data is borrowed from the caller. The ORT backend copies
/// into a `Vec` inside `Tensor::from_array`, so the borrow lifetime can be as
/// short as one [`OnnxSession::run`] call.
pub(crate) struct OnnxTensorView<'a> {
    pub shape: &'a [i64],
    pub data: OnnxTensorData<'a>,
}

/// Borrowed input tensor data.
///
/// Closed enum over the dtypes the local-ONNX consumers need, kept model-
/// agnostic so any model can pass either. Adding a variant requires a matching
/// arm in every [`OnnxSession`] impl.
///
/// `I64` is part of the seam API (Silero VAD feeds an i64 `sr` input) but is
/// unused by the f32-only denoise model, so a denoise-without-vad build
/// constructs no i64 input. Kept allowed-dead in that build rather than gated on
/// `vad`, which would couple this model-agnostic seam to one model's feature;
/// mirrors the sibling [`OnnxTensorOwned::I64`] convention below.
pub(crate) enum OnnxTensorData<'a> {
    F32(&'a [f32]),
    #[allow(dead_code)]
    I64(&'a [i64]),
}

/// Named-input bag for [`OnnxSession::run`].
///
/// Slice rather than `Vec` to avoid an allocation on each VAD inference
/// (Silero VAD calls `run()` ~50 times per second of audio at 16 kHz).
pub(crate) struct OnnxInputs<'a> {
    pub items: &'a [(&'a str, OnnxTensorView<'a>)],
}

/// Named-output bag returned from [`OnnxSession::run`].
///
/// Owns its data: ORT's `try_extract_tensor` returns a borrow tied to the
/// `SessionOutputs` value, which is itself bounded by `&'s mut Session`.
/// Copying outputs into owned [`Vec`]s ends that borrow before `run` returns,
/// keeping the trait method usable from `&mut self` without HRTB gymnastics.
pub(crate) struct OnnxOutputs {
    pub tensors: Vec<(String, OnnxOutputTensor)>,
}

/// One owned named output.
///
/// `shape` is part of the trait API (workloads with dynamic-shape outputs
/// need it) but is unused by Silero VAD's known-shape outputs. Kept
/// allowed-dead until a non-Silero consumer reads it.
pub(crate) struct OnnxOutputTensor {
    #[allow(dead_code)]
    pub shape: Vec<i64>,
    pub data: OnnxTensorOwned,
}

/// Owned output tensor data.
///
/// `I64` is part of the trait API (some ONNX workloads emit i64 outputs;
/// e.g., classification logits-as-class-ids) but is unused by Silero VAD's
/// f32-only outputs. Kept allowed-dead until a non-Silero consumer emits an
/// i64 output.
pub(crate) enum OnnxTensorOwned {
    F32(Vec<f32>),
    #[allow(dead_code)]
    I64(Vec<i64>),
}

impl OnnxOutputs {
    /// Look up an output tensor by name. Returns `None` if no output with
    /// that name was produced by the session.
    pub(crate) fn get(&self, name: &str) -> Option<&OnnxOutputTensor> {
        self.tensors.iter().find(|(n, _)| n == name).map(|(_, t)| t)
    }
}

/// Internal ONNX session abstraction.
///
/// `pub(crate)`: not part of the public API. A single backend (ORT) implements
/// it today.
///
/// `Send + Sync` is required because [`crate::vad::SileroVad`] can be moved into
/// a capture-thread processing path. ORT's `Session` already satisfies both
/// bounds.
///
/// `run` takes `&mut self` to match ORT's `Session::run` signature; a backend
/// whose native API takes `&self` implements the mutable receiver trivially.
pub(crate) trait OnnxSession: Send + Sync {
    /// Execute one inference.
    ///
    /// `inputs` is a borrowed slice of named tensor views; the session
    /// must not retain any reference past the return. `OnnxOutputs` owns
    /// its tensors so the caller can drop the borrow before reading them.
    fn run(&mut self, inputs: OnnxInputs<'_>) -> Result<OnnxOutputs, DecibriError>;
}

/// Builder for an [`OnnxSession`].
///
/// Concrete struct, not a trait: backend selection happens inside
/// [`Self::build`], which returns the ORT-backed `OrtSession` (the only
/// backend).
///
/// The execution provider defaults to [`ExecutionProvider::Cpu`] and can be
/// overridden with [`Self::with_execution_provider`]. For `Cpu` the session is
/// built exactly as before this builder gained provider awareness; an
/// accelerator is registered ahead of a CPU fallback (see
/// [`ort_impl::OrtSession::open`]).
pub(crate) struct OnnxSessionBuilder {
    model_path: PathBuf,
    intra_threads: usize,
    execution_provider: ExecutionProvider,
}

impl OnnxSessionBuilder {
    /// Start a builder targeting the given ONNX model file path.
    ///
    /// The execution provider defaults to [`ExecutionProvider::Cpu`].
    pub(crate) fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: path.into(),
            intra_threads: 1,
            execution_provider: ExecutionProvider::Cpu,
        }
    }

    /// Set the intra-op thread count for the session.
    pub(crate) fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }

    /// Select the execution provider for the session. Defaults to
    /// [`ExecutionProvider::Cpu`].
    ///
    /// Part of the reserved execution-provider seam: no in-crate caller selects
    /// a non-default provider yet, so this is `allow(dead_code)` for now (same
    /// convention as the accelerator variants of [`ExecutionProvider`]).
    #[allow(dead_code)]
    pub(crate) fn with_execution_provider(mut self, ep: ExecutionProvider) -> Self {
        self.execution_provider = ep;
        self
    }

    /// Build the session.
    ///
    /// Always returns the ORT-backed implementation. The entire `onnx` module
    /// is gated on `feature = "vad"` (see `lib.rs`), so this method only exists
    /// when ORT is available.
    pub(crate) fn build(self) -> Result<Box<dyn OnnxSession>, DecibriError> {
        let session = ort_impl::OrtSession::open(
            &self.model_path,
            self.intra_threads,
            self.execution_provider,
        )?;
        Ok(Box::new(session))
    }
}

mod ort_impl {
    use std::borrow::Cow;
    use std::path::Path;

    use ort::session::{Session, SessionInputValue};
    use ort::value::{Tensor, TensorElementType};

    use super::{
        DecibriError, ExecutionProvider, OnnxInputs, OnnxOutputTensor, OnnxOutputs, OnnxSession,
        OnnxTensorData, OnnxTensorOwned,
    };

    /// Shorthand for ORT's per-session execution-provider handle.
    type Epd = ort::execution_providers::ExecutionProviderDispatch;

    /// Generate a `Result<Epd, DecibriError>` constructor for one accelerator
    /// as a pair of whole-function `#[cfg]` definitions, mirroring the
    /// `do_ort_init` distribution-mode split. When the Cargo feature is
    /// enabled the provider is built; otherwise a clear `OnnxBackendFailed`
    /// error (the reserved non-ORT-backend variant, whose docs name `"coreml"`
    /// as an example) names the feature to enable. Whole-function gating keeps
    /// the feature-off build free of any reference to the gated ORT provider
    /// type.
    macro_rules! accelerator_dispatch {
        ($name:ident, $feature:literal, $ty:ident, $display:literal) => {
            #[cfg(feature = $feature)]
            fn $name() -> Result<Epd, DecibriError> {
                Ok(ort::execution_providers::$ty::default().build())
            }
            #[cfg(not(feature = $feature))]
            fn $name() -> Result<Epd, DecibriError> {
                Err(DecibriError::OnnxBackendFailed {
                    backend: $display,
                    source: concat!(
                        "execution provider not available in this build; rebuild decibri \
                         with the `",
                        $feature,
                        "` feature enabled"
                    )
                    .into(),
                })
            }
        };
    }

    accelerator_dispatch!(coreml_dispatch, "coreml", CoreML, "CoreML");
    accelerator_dispatch!(cuda_dispatch, "cuda", CUDA, "CUDA");
    accelerator_dispatch!(directml_dispatch, "directml", DirectML, "DirectML");
    accelerator_dispatch!(rocm_dispatch, "rocm", ROCm, "ROCm");

    /// Resolve the per-session execution-provider list for `ep`.
    ///
    /// `Ok(None)` means "register nothing and use ORT's built-in CPU
    /// provider", which preserves the pre-seam behaviour exactly. For an
    /// accelerator, returns `Ok(Some([accelerator, cpu]))` so ORT tries the
    /// accelerator first and falls back to CPU. Returns
    /// [`DecibriError::OnnxBackendFailed`] when the accelerator's Cargo feature
    /// is not compiled in.
    fn provider_dispatch(ep: ExecutionProvider) -> Result<Option<Vec<Epd>>, DecibriError> {
        let accelerator = match ep {
            ExecutionProvider::Cpu => return Ok(None),
            ExecutionProvider::CoreMl => coreml_dispatch()?,
            ExecutionProvider::Cuda => cuda_dispatch()?,
            ExecutionProvider::DirectMl => directml_dispatch()?,
            ExecutionProvider::Rocm => rocm_dispatch()?,
        };
        // CPU is always the final fallback in the priority list. Arena
        // allocation is left enabled to match ORT's built-in CPU default.
        let cpu = ort::execution_providers::CPU::default()
            .with_arena_allocator(true)
            .build();
        Ok(Some(vec![accelerator, cpu]))
    }

    /// ORT-backed [`OnnxSession`]. Wraps `ort::session::Session`.
    ///
    /// LSTM state for stateful models (Silero VAD) lives on the consumer
    /// (`SileroVad`), not the session: the session holds only ORT runtime
    /// state. This matches ORT's own model where `Session` is the engine
    /// and per-call state is supplied as named input tensors.
    pub(super) struct OrtSession {
        inner: Session,
    }

    impl OrtSession {
        /// Construct an [`OrtSession`] from a model file path: builder,
        /// `with_intra_threads`, optional execution-provider registration,
        /// `commit_from_file`.
        pub(super) fn open(
            path: &Path,
            intra: usize,
            execution_provider: ExecutionProvider,
        ) -> Result<Self, DecibriError> {
            // Resolve the provider list first so an unavailable accelerator
            // fails before any ORT interaction.
            let providers = provider_dispatch(execution_provider)?;

            let builder = Session::builder()
                .map_err(DecibriError::OrtSessionBuildFailed)?
                .with_intra_threads(intra)
                .map_err(|e| DecibriError::OrtThreadsConfigFailed(e.into()))?;

            // Cpu (the default, and the only provider decibri requests today)
            // resolves to `None` above and keeps ORT's built-in CPU provider
            // with no explicit registration, so its behaviour is byte-for-byte
            // identical to before this seam existed. A non-Cpu provider was
            // resolved as [accelerator, cpu] and is registered here, CPU last.
            let mut builder = match providers {
                Some(providers) => builder
                    .with_execution_providers(providers)
                    .map_err(|e| DecibriError::OrtSessionBuildFailed(e.into()))?,
                None => builder,
            };

            let session =
                builder
                    .commit_from_file(path)
                    .map_err(|e| DecibriError::VadModelLoadFailed {
                        path: path.to_path_buf(),
                        source: e,
                    })?;
            Ok(Self { inner: session })
        }
    }

    /// Map a runtime input or output name back to a `&'static str` for the
    /// existing [`DecibriError::OrtTensorCreateFailed`] /
    /// [`DecibriError::OrtTensorExtractFailed`] `kind` field. Preserves the
    /// error message text for the well-known Silero names without changing the
    /// error variant signatures.
    fn known_kind(name: &str) -> &'static str {
        match name {
            "input" => "input",
            "state" => "state",
            "sr" => "sr",
            "output" => "output",
            "stateN" => "state",
            _ => "trait_tensor",
        }
    }

    impl OnnxSession for OrtSession {
        /// Run one inference through the ORT session.
        ///
        /// Hardcoded to the ValueMap (runtime-named) input path: Silero VAD has
        /// exactly 3 inputs and ORT 2.x's `SessionInputs<'i, 'v, N>`
        /// const-generic input count is awkward to use from a runtime slice. The
        /// ValueMap variant accepts a `Vec<(Cow<'_, str>, SessionInputValue<'_>)>`
        /// which `Session::run` consumes via
        /// `From<Vec<(K, V)>> for SessionInputs<_, _, 0>`.
        ///
        /// Output dtype dispatch reads `Value::dtype().tensor_type()` per
        /// output and copies into an owned `OnnxOutputTensor`. Copy is trivial
        /// for Silero's outputs (`output` is 1 float, `stateN` is 256 floats).
        fn run(&mut self, inputs: OnnxInputs<'_>) -> Result<OnnxOutputs, DecibriError> {
            let mut input_pairs: Vec<(Cow<'_, str>, SessionInputValue<'_>)> =
                Vec::with_capacity(inputs.items.len());

            for (name, view) in inputs.items.iter() {
                let kind = known_kind(name);
                let shape: Vec<i64> = view.shape.to_vec();
                let value: SessionInputValue<'_> = match &view.data {
                    OnnxTensorData::F32(slice) => {
                        let tensor = Tensor::from_array((shape, slice.to_vec()))
                            .map_err(|e| DecibriError::OrtTensorCreateFailed { kind, source: e })?;
                        tensor.into()
                    }
                    OnnxTensorData::I64(slice) => {
                        let tensor = Tensor::from_array((shape, slice.to_vec()))
                            .map_err(|e| DecibriError::OrtTensorCreateFailed { kind, source: e })?;
                        tensor.into()
                    }
                };
                input_pairs.push((Cow::Owned((*name).to_string()), value));
            }

            let outputs = self
                .inner
                .run(input_pairs)
                .map_err(DecibriError::OrtInferenceFailed)?;

            let names: Vec<String> = outputs.keys().map(|k| k.to_string()).collect();
            let mut tensors: Vec<(String, OnnxOutputTensor)> = Vec::with_capacity(names.len());
            for name in names {
                let kind = known_kind(&name);
                let value = &outputs[name.as_str()];
                let dtype = value.dtype().tensor_type();
                let owned = match dtype {
                    Some(TensorElementType::Float32) => {
                        let (shape, data) = value.try_extract_tensor::<f32>().map_err(|e| {
                            DecibriError::OrtTensorExtractFailed { kind, source: e }
                        })?;
                        OnnxOutputTensor {
                            shape: shape.to_vec(),
                            data: OnnxTensorOwned::F32(data.to_vec()),
                        }
                    }
                    Some(TensorElementType::Int64) => {
                        let (shape, data) = value.try_extract_tensor::<i64>().map_err(|e| {
                            DecibriError::OrtTensorExtractFailed { kind, source: e }
                        })?;
                        OnnxOutputTensor {
                            shape: shape.to_vec(),
                            data: OnnxTensorOwned::I64(data.to_vec()),
                        }
                    }
                    _ => {
                        return Err(DecibriError::OnnxBackendFailed {
                            backend: "ort",
                            source: format!(
                                "unsupported output tensor element type for {name}: {dtype:?}"
                            )
                            .into(),
                        });
                    }
                };
                tensors.push((name, owned));
            }

            Ok(OnnxOutputs { tensors })
        }
    }
}

// Inline tests: the trait is `pub(crate)`, so an integration test under
// `tests/` could not see it. Inline `#[cfg(test)] mod tests` exercises the
// trait API in isolation (a MockSession plus an ORT-backed round trip) from
// within the crate.
#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity-check that `Box<dyn OnnxSession>` is `Send + Sync`. This is the
    /// load-bearing bound (`SileroVad` can live on a capture thread; the
    /// bindings already assert `Send + 'static`). If a future trait change
    /// drops one of these bounds, this assertion fails at compile time.
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn box_dyn_session_is_send_sync() {
        assert_send_sync::<Box<dyn OnnxSession>>();
    }

    #[test]
    fn execution_provider_default_is_cpu() {
        assert_eq!(ExecutionProvider::default(), ExecutionProvider::Cpu);

        // Construct and compare every variant so the `pub(crate)` enum's
        // variants are all exercised regardless of which execution-provider
        // features a given build enables.
        let all = [
            ExecutionProvider::Cpu,
            ExecutionProvider::CoreMl,
            ExecutionProvider::Cuda,
            ExecutionProvider::DirectMl,
            ExecutionProvider::Rocm,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a, b, "execution provider variants must be distinct");
            }
        }
    }

    /// Requesting an accelerator whose Cargo feature is not compiled in is a
    /// precise error raised before any ORT interaction. Gated off when `cuda`
    /// is enabled (then CUDA is available and this would not error). Exercises
    /// the builder's execution-provider setter and the missing-feature path.
    #[cfg(not(feature = "cuda"))]
    #[test]
    fn builder_rejects_unavailable_execution_provider() {
        let result = OnnxSessionBuilder::from_file("does-not-need-to-exist.onnx")
            .with_execution_provider(ExecutionProvider::Cuda)
            .build();
        match result {
            Err(DecibriError::OnnxBackendFailed { backend, source }) => {
                assert_eq!(backend, "CUDA");
                let msg = source.to_string();
                assert!(
                    msg.contains("cuda"),
                    "error should name the `cuda` feature to enable, got: {msg}"
                );
            }
            Err(other) => panic!("expected OnnxBackendFailed, got error: {other:?}"),
            Ok(_) => panic!("expected OnnxBackendFailed, got a built session"),
        }
    }

    // ORT init error wrapping. These tests moved here with `wrap_init_error`
    // when ORT initialization was relocated off `SileroVad` into this module.

    #[test]
    fn wrap_init_error_with_path_is_load_failed() {
        use std::env;

        // Platform-agnostic invalid path (guaranteed to not exist on any OS).
        let bogus_path = env::temp_dir().join("does-not-exist-onnxruntime-xyz-test");
        let err = wrap_init_error(
            Some(bogus_path.as_path()),
            ort::Error::new("simulated ort loader failure"),
        );
        let msg = err.to_string();

        assert!(
            msg.contains(&bogus_path.display().to_string()),
            "error message should contain the attempted path, got: {msg}"
        );
        assert!(
            msg.contains("If ORT_DYLIB_PATH is set"),
            "error message should contain actionable guidance phrase, got: {msg}"
        );
        assert!(
            msg.contains("simulated ort loader failure"),
            "error message should include the underlying ort error, got: {msg}"
        );
    }

    #[test]
    fn wrap_init_error_without_path_is_init_failed() {
        let err = wrap_init_error(None, ort::Error::new("simulated ort init failure"));
        let msg = err.to_string();

        assert!(
            msg.contains("ort_library_path"),
            "None-path error should mention the VadConfig field, got: {msg}"
        );
        assert!(
            msg.contains("ORT_DYLIB_PATH"),
            "None-path error should mention the env var, got: {msg}"
        );
        assert!(
            msg.contains("ort-download-binaries"),
            "None-path error should mention the opt-out feature, got: {msg}"
        );
        assert!(
            msg.contains("simulated ort init failure"),
            "error message should include the underlying ort error, got: {msg}"
        );
    }

    /// `err.source()` walks to the underlying `ort::Error` for the wrapped
    /// variants, and deliberately returns `None` for `OrtPathInvalid` (which
    /// has no underlying ORT error because constructing one would trigger the
    /// hang the pre-check prevents).
    #[test]
    fn ort_error_source_chain_preserved() {
        use std::error::Error;

        // OrtInitFailed (no path) carries an ort::Error source.
        let inner = ort::Error::new("simulated underlying ort error");
        let err = wrap_init_error(None, inner);
        assert!(
            err.source().is_some(),
            "OrtInitFailed should carry an ort::Error source"
        );

        // OrtLoadFailed (with path) carries an ort::Error source.
        let inner_with_path = ort::Error::new("another simulated error");
        let path_err = wrap_init_error(Some(Path::new("/tmp/bogus")), inner_with_path);
        assert!(
            path_err.source().is_some(),
            "OrtLoadFailed should carry an ort::Error source"
        );

        // OrtPathInvalid intentionally does NOT carry a source (documented
        // asymmetry; see OrtPathInvalid's rustdoc in error.rs).
        let path_invalid = DecibriError::OrtPathInvalid {
            path: PathBuf::from("/tmp/nope"),
            reason: "test",
        };
        assert!(
            path_invalid.source().is_none(),
            "OrtPathInvalid intentionally has no source (constructing ort::Error \
             would trigger the hang the pre-check prevents)"
        );
    }

    /// Trivial in-memory [`OnnxSession`] for trait-shape testing without ORT.
    /// Stores expected inputs (asserted in `run`) and canned outputs (returned
    /// from `run`).
    struct MockSession {
        expected_input_count: usize,
        canned_outputs: Vec<(String, OnnxOutputTensor)>,
    }

    impl OnnxSession for MockSession {
        fn run(&mut self, inputs: OnnxInputs<'_>) -> Result<OnnxOutputs, DecibriError> {
            assert_eq!(
                inputs.items.len(),
                self.expected_input_count,
                "MockSession got unexpected number of inputs"
            );
            let mut tensors = Vec::with_capacity(self.canned_outputs.len());
            for (name, t) in self.canned_outputs.iter() {
                let cloned = OnnxOutputTensor {
                    shape: t.shape.clone(),
                    data: match &t.data {
                        OnnxTensorOwned::F32(v) => OnnxTensorOwned::F32(v.clone()),
                        OnnxTensorOwned::I64(v) => OnnxTensorOwned::I64(v.clone()),
                    },
                };
                tensors.push((name.clone(), cloned));
            }
            Ok(OnnxOutputs { tensors })
        }
    }

    #[test]
    fn mock_session_round_trip() {
        let canned = vec![(
            "output".to_string(),
            OnnxOutputTensor {
                shape: vec![1, 1],
                data: OnnxTensorOwned::F32(vec![0.42]),
            },
        )];
        let mut session: Box<dyn OnnxSession> = Box::new(MockSession {
            expected_input_count: 3,
            canned_outputs: canned,
        });
        let audio = vec![0.0f32; 512];
        let state = vec![0.0f32; 256];
        let sr = vec![16000i64];
        let inputs = OnnxInputs {
            items: &[
                (
                    "input",
                    OnnxTensorView {
                        shape: &[1, 512],
                        data: OnnxTensorData::F32(&audio),
                    },
                ),
                (
                    "state",
                    OnnxTensorView {
                        shape: &[2, 1, 128],
                        data: OnnxTensorData::F32(&state),
                    },
                ),
                (
                    "sr",
                    OnnxTensorView {
                        shape: &[1],
                        data: OnnxTensorData::I64(&sr),
                    },
                ),
            ],
        };
        let outputs = session.run(inputs).expect("MockSession should succeed");
        assert_eq!(outputs.tensors.len(), 1);
        let probe = outputs.get("output").expect("output present");
        assert_eq!(probe.shape, vec![1, 1]);
        match &probe.data {
            OnnxTensorOwned::F32(v) => assert_eq!(v.as_slice(), &[0.42f32]),
            OnnxTensorOwned::I64(_) => panic!("expected F32"),
        }
    }

    #[test]
    fn output_lookup_by_name_returns_none_for_missing() {
        let outputs = OnnxOutputs {
            tensors: vec![(
                "output".to_string(),
                OnnxOutputTensor {
                    shape: vec![1],
                    data: OnnxTensorOwned::F32(vec![1.0]),
                },
            )],
        };
        assert!(outputs.get("output").is_some());
        assert!(outputs.get("not_a_name").is_none());
    }

    #[test]
    fn input_slice_preserves_order_and_names() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![10i64, 20];
        let inputs = OnnxInputs {
            items: &[
                (
                    "first",
                    OnnxTensorView {
                        shape: &[3],
                        data: OnnxTensorData::F32(&a),
                    },
                ),
                (
                    "second",
                    OnnxTensorView {
                        shape: &[2],
                        data: OnnxTensorData::I64(&b),
                    },
                ),
            ],
        };
        assert_eq!(inputs.items.len(), 2);
        assert_eq!(inputs.items[0].0, "first");
        assert_eq!(inputs.items[1].0, "second");
        match inputs.items[0].1.data {
            OnnxTensorData::F32(s) => assert_eq!(s, a.as_slice()),
            OnnxTensorData::I64(_) => panic!("expected F32"),
        }
        match inputs.items[1].1.data {
            OnnxTensorData::I64(s) => assert_eq!(s, b.as_slice()),
            OnnxTensorData::F32(_) => panic!("expected I64"),
        }
    }

    /// ORT-backed round trip: load the bundled Silero model, construct an
    /// [`OnnxSession`] via [`OnnxSessionBuilder::build`], run a single
    /// inference with deterministic zero-filled inputs, verify output
    /// structure (shape + dtype) without asserting exact values.
    ///
    /// The entire `onnx` module is `#[cfg(feature = "vad")]`-gated, so this
    /// test only runs with vad enabled (which is the default feature set).
    #[test]
    fn ort_backed_session_runs_silero_inference() {
        use std::path::Path;

        // Resolve model relative to the workspace root, matching the pattern
        // already used in `vad.rs::tests::model_path`.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let model = Path::new(manifest_dir)
            .join("..")
            .join("..")
            .join("models")
            .join("silero_vad.onnx");
        if !model.is_file() {
            eprintln!(
                "skipping ort_backed_session_runs_silero_inference: model not found at {}",
                model.display()
            );
            return;
        }

        // ORT must be initialized exactly once per process. Reuse the same
        // relocated `init_ort_once` path that `SileroVad` uses so this test
        // composes with the existing VAD tests in the same `cargo test`
        // invocation.
        super::init_ort_once(None).expect("ORT init should succeed");

        let mut session = OnnxSessionBuilder::from_file(&model)
            .with_intra_threads(1)
            .build()
            .expect("ORT session build should succeed for bundled model");

        let audio = vec![0.0f32; 512];
        let state = vec![0.0f32; 256];
        let sr = vec![16000i64];
        let outputs = session
            .run(OnnxInputs {
                items: &[
                    (
                        "input",
                        OnnxTensorView {
                            shape: &[1, 512],
                            data: OnnxTensorData::F32(&audio),
                        },
                    ),
                    (
                        "state",
                        OnnxTensorView {
                            shape: &[2, 1, 128],
                            data: OnnxTensorData::F32(&state),
                        },
                    ),
                    (
                        "sr",
                        OnnxTensorView {
                            shape: &[1],
                            data: OnnxTensorData::I64(&sr),
                        },
                    ),
                ],
            })
            .expect("ORT session run should succeed");

        let probe = outputs
            .get("output")
            .expect("Silero emits an `output` tensor");
        match &probe.data {
            OnnxTensorOwned::F32(v) => assert_eq!(v.len(), 1, "Silero `output` is one f32"),
            OnnxTensorOwned::I64(_) => panic!("Silero `output` is f32, not i64"),
        }
        let state_n = outputs
            .get("stateN")
            .expect("Silero emits a `stateN` tensor");
        match &state_n.data {
            OnnxTensorOwned::F32(v) => assert_eq!(v.len(), 256, "Silero `stateN` is 256 f32s"),
            OnnxTensorOwned::I64(_) => panic!("Silero `stateN` is f32, not i64"),
        }
    }
}
