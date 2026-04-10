// Noise reduction — stub for Phase 2.
//
// Two paths planned:
//   1. Basic DSP: spectral subtraction (open source, works but has artifacts)
//   2. ML model: custom trained ONNX model (proprietary, better quality, default)
//
// Both compile to native and WASM from the same Rust code.
