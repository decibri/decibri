// Voice Activity Detection — stub for Phase 3.
//
// Phase 1 (current): Energy-based VAD lives in the JS wrapper (computeRMS).
// Phase 3: Silero VAD model (MIT, ~1.6MB) via ort crate (ONNX Runtime).
// Phase 3.1: Custom trained VAD model replaces Silero.
//
// The Rust VAD module will provide:
//
//   let vad = SileroVad::new(VadConfig { model_path, threshold: 0.5, ... })?;
//   let result = vad.process_chunk(&samples, sample_rate)?;
//   // result.probability: f32, result.is_speech: bool
