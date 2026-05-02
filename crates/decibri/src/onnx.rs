//! Internal ONNX session abstraction.
//!
//! This module defines the [`OnnxSession`] trait that decibri uses to talk to
//! ONNX Runtime today, and that future backends (CoreML at iOS time, TFLite
//! at Android time, GPU EPs at the P4 GPU project) will plug into without
//! changing any consumer code.
//!
//! # Visibility
//!
//! Everything in this module is `pub(crate)`. The trait is invisible to
//! external consumers of `decibri` 3.x. It goes public at 4.0 alongside the
//! `decibri-onnx` workspace split, at which point this file becomes the
//! public API of the new `decibri-onnx` crate. See locked decision LD15 in
//! `~/.claude/plans/python-integration-project.md` and Phase 8 plan
//! `~/.claude/plans/phase-8-onnxsession-trait.md` LD-8-1.
//!
//! # Trait shape
//!
//! The trait surface mirrors the four operations [`crate::vad`] performs
//! against `ort::session::Session`: builder construction, input tensor
//! creation, run, output tensor extraction. The closed-enum
//! [`OnnxTensorData`] / [`OnnxTensorOwned`] design avoids an `ndarray`
//! workspace dep and keeps the VAD hot path (~50 calls per second) free of
//! tensor-type vtable dispatch.
//!
//! Adding a new dtype variant (e.g. `F16` or `I32`) is technically a closed-
//! enum break, but is acceptable while the trait stays `pub(crate)`. Once
//! the trait goes public at 4.0, dtype additions become a major-version
//! event.
//!
//! # Backend selection
//!
//! Backend dispatch happens inside [`OnnxSessionBuilder::build`]. In 3.x the
//! ORT-backed implementation is the only one, so `build` always returns an
//! `OrtSession`. At 0.3.0+ when the trait goes public and CoreML / TFLite
//! land, `build` becomes a `cfg!`/runtime dispatch point. Adding execution-
//! provider selection (`with_execution_providers`) at that point is purely
//! additive (LD-8-3).
//!
//! # Error handling
//!
//! The trait reuses [`crate::error::DecibriError`]. The 8 existing ORT
//! variants cover ORT failures. The forward-compat
//! [`crate::error::DecibriError::OnnxBackendFailed`] variant covers future
//! non-ORT backend errors and is not emitted by the ORT impl in 3.x.

use std::path::PathBuf;

use crate::error::DecibriError;

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
/// Closed enum over the dtypes Silero VAD needs. Adding a variant requires a
/// matching arm in every [`OnnxSession`] impl.
pub(crate) enum OnnxTensorData<'a> {
    F32(&'a [f32]),
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
/// `shape` is part of the trait API (future workloads with dynamic-shape
/// outputs need it) but is unused by Silero VAD's known-shape outputs in
/// 3.x. Kept allowed-dead until a non-Silero consumer reads it.
pub(crate) struct OnnxOutputTensor {
    #[allow(dead_code)]
    pub shape: Vec<i64>,
    pub data: OnnxTensorOwned,
}

/// Owned output tensor data.
///
/// `I64` is part of the trait API (some ONNX workloads emit i64 outputs;
/// e.g., classification logits-as-class-ids) but is unused by Silero VAD's
/// f32-only outputs in 3.x. Kept allowed-dead until a non-Silero consumer
/// emits an i64 output.
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
/// `pub(crate)` until 4.0 (LD-8-1). At 4.0 / `decibri-onnx` workspace split
/// this becomes the public surface that ORT, CoreML, and TFLite backends all
/// implement.
///
/// `Send + Sync` is required because [`crate::vad::SileroVad`] is moved into
/// the cpal capture-thread processing path (Phase 5 architecture). ORT's
/// `Session` already satisfies both bounds; future backends needing
/// `unsafe impl Send + Sync` wrappers absorb that cost in their impls.
///
/// `run` takes `&mut self` to match ORT's `Session::run` signature; backends
/// whose native API takes `&self` (CoreML's `MLModel.prediction`, TFLite's
/// `Interpreter::invoke`) implement the mutable receiver trivially.
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
/// Concrete struct, not a trait: backend dispatch happens inside
/// [`Self::build`]. In 3.x there is exactly one backend (ORT), so `build`
/// always returns an `OrtSession`. At 0.3.0+ when CoreML / TFLite / etc.
/// land, this is the dispatch point.
///
/// `with_execution_providers([EP])` is deferred to 0.3.0+ per LD-8-3. The
/// trait is `pub(crate)`, so no external consumer can call it; the only
/// internal consumer (`SileroVad`) does not need EP selection (Silero is
/// CPU-only). Adding EP selection at 0.3.0+ is purely additive.
pub(crate) struct OnnxSessionBuilder {
    model_path: PathBuf,
    intra_threads: usize,
}

impl OnnxSessionBuilder {
    /// Start a builder targeting the given ONNX model file path.
    pub(crate) fn from_file(path: impl Into<PathBuf>) -> Self {
        Self {
            model_path: path.into(),
            intra_threads: 1,
        }
    }

    /// Set the intra-op thread count for the session.
    pub(crate) fn with_intra_threads(mut self, n: usize) -> Self {
        self.intra_threads = n;
        self
    }

    /// Build the session.
    ///
    /// In 3.x always returns the ORT-backed implementation. The entire
    /// `onnx` module is gated on `feature = "vad"` (see `lib.rs`), so this
    /// method only exists when ORT is available.
    pub(crate) fn build(self) -> Result<Box<dyn OnnxSession>, DecibriError> {
        let session = ort_impl::OrtSession::open(&self.model_path, self.intra_threads)?;
        Ok(Box::new(session))
    }
}

mod ort_impl {
    use std::borrow::Cow;
    use std::path::Path;

    use ort::session::{Session, SessionInputValue};
    use ort::value::{Tensor, TensorElementType};

    use super::{
        DecibriError, OnnxInputs, OnnxOutputTensor, OnnxOutputs, OnnxSession, OnnxTensorData,
        OnnxTensorOwned,
    };

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
        /// Construct an [`OrtSession`] from a model file path. Pattern
        /// matches the pre-Phase-8 `crates/decibri/src/vad.rs:243-254`
        /// session-creation block: builder, with_intra_threads, commit_from_file.
        pub(super) fn open(path: &Path, intra: usize) -> Result<Self, DecibriError> {
            let session = Session::builder()
                .map_err(DecibriError::OrtSessionBuildFailed)?
                .with_intra_threads(intra)
                .map_err(|e| DecibriError::OrtThreadsConfigFailed(e.into()))?
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
    /// exact pre-Phase-8 error message text for the well-known Silero names
    /// without forcing a breaking change to the error variant signatures
    /// (LD-8-1: public API byte-identical).
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
        /// Hardcoded to the ValueMap (runtime-named) input path per LD-8-10:
        /// Silero VAD has exactly 3 inputs and ORT 2.x's
        /// `SessionInputs<'i, 'v, N>` const-generic input count is awkward
        /// to use from a runtime slice. The ValueMap variant accepts a
        /// `Vec<(Cow<'_, str>, SessionInputValue<'_>)>` which `Session::run`
        /// consumes via `From<Vec<(K, V)>> for SessionInputs<_, _, 0>`.
        /// Generic-N construction lands at 0.3.0+ when a second backend or
        /// non-VAD workload needs different input cardinality.
        ///
        /// Output dtype dispatch reads `Value::dtype().tensor_type()` per
        /// output and copies into an owned `OnnxOutputTensor`. Copy is
        /// trivial for Silero's outputs (`output` is 1 float, `stateN` is
        /// 256 floats). Whisper-style large outputs would benefit from a
        /// future `run_zero_copy` variant; deferred per Phase 8 plan
        /// Section 4 "Items NOT in scope".
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

// Inline tests: the trait is `pub(crate)` (LD-8-1), so an integration test in
// `tests/onnx_trait.rs` could not see it. Inline `#[cfg(test)] mod tests`
// preserves the plan's intent (test the trait API in isolation, with a
// MockSession and an ORT-backed round trip) while honoring the visibility
// constraint. Documented in the Phase 8 implementation report.
#[cfg(test)]
mod tests {
    use super::*;

    /// Sanity-check that `Box<dyn OnnxSession>` is `Send + Sync`. This is the
    /// load-bearing bound (`SileroVad` lives in the cpal capture-thread path
    /// per Phase 5; the bridge already asserts `Send + 'static`). If a future
    /// trait change drops one of these bounds, this assertion fails at
    /// compile time.
    #[allow(dead_code)]
    fn assert_send_sync<T: Send + Sync>() {}

    #[test]
    fn box_dyn_session_is_send_sync() {
        assert_send_sync::<Box<dyn OnnxSession>>();
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
        // `init_ort_once` path that `vad.rs` uses so this test composes with
        // the existing VAD tests in the same `cargo test` invocation.
        crate::vad::init_ort_once(None).expect("ORT init should succeed");

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
