//! Voice Activity Detection using the Silero VAD v5 ONNX model.
//!
//! Silero VAD is a pre-trained ML model that detects human speech in audio.
//! It runs inference on 512-sample windows (at 16kHz) and returns a speech
//! probability between 0.0 and 1.0.
//!
//! # Model tensor specification (verified from silero_vad.onnx)
//!
//! Inputs:
//!   - `input`:  f32[batch, window_size]  — audio samples
//!   - `state`:  f32[2, batch, 128]       — LSTM state (hidden + cell combined)
//!   - `sr`:     i64 scalar               — sample rate
//!
//! Outputs:
//!   - `output`: f32[batch, 1]            — speech probability
//!   - `stateN`: f32[2, batch, 128]       — updated LSTM state

#[cfg(feature = "vad")]
use std::path::PathBuf;

use crate::error::DecibriError;

/// Configuration for Silero VAD.
#[derive(Debug, Clone)]
pub struct VadConfig {
    /// Path to the silero_vad.onnx model file.
    pub model_path: PathBuf,
    /// Sample rate: 8000 or 16000 Hz.
    pub sample_rate: u32,
    /// Speech probability threshold (0.0–1.0). Default: 0.5.
    pub threshold: f32,
}

impl Default for VadConfig {
    fn default() -> Self {
        Self {
            model_path: PathBuf::from("silero_vad.onnx"),
            sample_rate: 16000,
            threshold: 0.5,
        }
    }
}

/// Result of processing an audio chunk through Silero VAD.
#[derive(Debug, Clone)]
pub struct VadResult {
    /// Maximum speech probability across all windows in the chunk.
    pub probability: f32,
    /// Whether probability >= threshold.
    pub is_speech: bool,
}

/// Silero VAD v5 inference engine.
///
/// Stateful: maintains LSTM hidden/cell state across calls.
/// Call `process()` with each audio chunk. Call `reset()` to clear state.
#[cfg(feature = "vad")]
pub struct SileroVad {
    session: ort::session::Session,
    /// Combined LSTM state [2, 1, 128] = 256 floats.
    state: Vec<f32>,
    sample_rate: u32,
    threshold: f32,
    /// Carries leftover samples between process() calls.
    accumulator: Vec<f32>,
    /// 512 for 16kHz, 256 for 8kHz.
    window_size: usize,
}

/// State size: 2 (hidden + cell) * 1 (batch) * 128 (hidden_dim) = 256.
/// State size: 2 (hidden + cell layers) × batch(1) × hidden_dim(128).
const STATE_SIZE: usize = 256;

#[cfg(feature = "vad")]
impl SileroVad {
    /// Create a new Silero VAD instance by loading the ONNX model.
    pub fn new(config: VadConfig) -> Result<Self, DecibriError> {
        let window_size = match config.sample_rate {
            16000 => 512,
            8000 => 256,
            _ => {
                return Err(DecibriError::Other(
                    "Silero VAD only supports sample rates 8000 and 16000".to_string(),
                ))
            }
        };

        if config.threshold < 0.0 || config.threshold > 1.0 {
            return Err(DecibriError::Other(
                "VAD threshold must be between 0.0 and 1.0".to_string(),
            ));
        }

        let session = ort::session::Session::builder()
            .map_err(|e| DecibriError::Other(format!("Failed to create ort session builder: {e}")))?
            .with_intra_threads(1)
            .map_err(|e| DecibriError::Other(format!("Failed to set ort threads: {e}")))?
            .commit_from_file(&config.model_path)
            .map_err(|e| {
                DecibriError::Other(format!(
                    "Failed to load Silero VAD model from {}: {e}",
                    config.model_path.display()
                ))
            })?;

        Ok(Self {
            session,
            state: vec![0.0f32; STATE_SIZE],
            sample_rate: config.sample_rate,
            threshold: config.threshold,
            accumulator: Vec::new(),
            window_size,
        })
    }

    /// Process audio samples and return a VAD result.
    ///
    /// Samples are f32 in the range [-1.0, 1.0]. The chunk can be any size;
    /// internally it is split into `window_size` windows. Leftover samples
    /// carry over to the next call.
    ///
    /// Returns the maximum speech probability across all windows processed.
    /// If no complete windows were formed (chunk too small), returns probability 0.0.
    pub fn process(&mut self, samples: &[f32]) -> Result<VadResult, DecibriError> {
        self.accumulator.extend_from_slice(samples);

        let mut max_probability: f32 = 0.0;
        let mut windows_processed = 0;

        while self.accumulator.len() >= self.window_size {
            let window: Vec<f32> = self.accumulator.drain(..self.window_size).collect();
            let probability = self.infer_window(&window)?;
            max_probability = max_probability.max(probability);
            windows_processed += 1;
        }

        // If no windows processed, return 0 probability (not enough data yet)
        if windows_processed == 0 {
            return Ok(VadResult {
                probability: 0.0,
                is_speech: false,
            });
        }

        Ok(VadResult {
            probability: max_probability,
            is_speech: max_probability >= self.threshold,
        })
    }

    /// Reset LSTM state to zeros (start of new utterance).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
        self.accumulator.clear();
    }

    /// Run inference on a single window of exactly `window_size` samples.
    fn infer_window(&mut self, window: &[f32]) -> Result<f32, DecibriError> {
        // Create input tensors using actual model tensor names:
        //   input: f32[1, window_size]
        //   state: f32[2, 1, 128]
        //   sr:    i64 scalar
        let input_tensor =
            ort::value::Tensor::from_array(([1i64, self.window_size as i64], window.to_vec()))
                .map_err(|e| DecibriError::Other(format!("Failed to create input tensor: {e}")))?;
        let state_tensor =
            ort::value::Tensor::from_array(([2i64, 1i64, 128i64], self.state.clone()))
                .map_err(|e| DecibriError::Other(format!("Failed to create state tensor: {e}")))?;
        let sr_tensor = ort::value::Tensor::from_array(([1i64], vec![self.sample_rate as i64]))
            .map_err(|e| DecibriError::Other(format!("Failed to create sr tensor: {e}")))?;

        let input_values = ort::inputs![
            "input" => input_tensor,
            "state" => state_tensor,
            "sr" => sr_tensor,
        ];

        let outputs = self
            .session
            .run(input_values)
            .map_err(|e| DecibriError::Other(format!("Silero VAD inference failed: {e}")))?;

        // Read outputs using actual model tensor names:
        //   output: f32[1, 1] — speech probability
        //   stateN: f32[2, 1, 128] — updated state
        let prob_tensor = outputs["output"]
            .try_extract_tensor::<f32>()
            .map_err(|e| DecibriError::Other(format!("Failed to extract output tensor: {e}")))?;
        let probability = prob_tensor.1[0];

        let state_tensor = outputs["stateN"]
            .try_extract_tensor::<f32>()
            .map_err(|e| DecibriError::Other(format!("Failed to extract state tensor: {e}")))?;
        self.state.copy_from_slice(state_tensor.1);

        Ok(probability)
    }
}

#[cfg(all(test, feature = "vad"))]
mod tests {
    use super::*;
    use std::path::Path;

    fn model_path() -> PathBuf {
        // Resolve relative to the workspace root
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir)
            .join("..")
            .join("..")
            .join("models")
            .join("silero_vad.onnx")
    }

    fn default_config() -> VadConfig {
        VadConfig {
            model_path: model_path(),
            sample_rate: 16000,
            threshold: 0.5,
        }
    }

    #[test]
    fn test_vad_config_validation() {
        let bad_rate = VadConfig {
            sample_rate: 44100,
            ..default_config()
        };
        assert!(SileroVad::new(bad_rate).is_err());
    }

    #[test]
    fn test_vad_loads_model() {
        let vad = SileroVad::new(default_config());
        assert!(vad.is_ok(), "Model should load: {:?}", vad.err());
    }

    #[test]
    fn test_vad_silence() {
        let mut vad = SileroVad::new(default_config()).unwrap();
        let silence = vec![0.0f32; 512];
        let result = vad.process(&silence).unwrap();
        // Silence should have low probability
        assert!(
            result.probability < 0.5,
            "Silence probability should be low, got {}",
            result.probability
        );
    }

    #[test]
    fn test_vad_state_persistence() {
        let mut vad = SileroVad::new(default_config()).unwrap();
        let initial_state = vad.state.clone();

        let samples = vec![0.0f32; 512];
        vad.process(&samples).unwrap();

        // State should have changed after inference
        assert_ne!(
            vad.state, initial_state,
            "State should change after inference"
        );
    }

    #[test]
    fn test_vad_accumulator_windows() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // 1600 samples = 3 windows of 512 + 64 leftover
        let samples = vec![0.0f32; 1600];
        let result = vad.process(&samples).unwrap();
        assert!(result.probability >= 0.0); // Just verify it runs

        // 64 samples should remain in accumulator
        assert_eq!(vad.accumulator.len(), 64);
    }

    #[test]
    fn test_vad_accumulator_carry() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // First chunk: 1600 samples → 3 windows, 64 leftover
        let chunk1 = vec![0.0f32; 1600];
        vad.process(&chunk1).unwrap();
        assert_eq!(vad.accumulator.len(), 64);

        // Second chunk: 1600 samples → 64 + 1600 = 1664 → 3 windows, 128 leftover
        let chunk2 = vec![0.0f32; 1600];
        vad.process(&chunk2).unwrap();
        assert_eq!(vad.accumulator.len(), 128);
    }

    #[test]
    fn test_vad_small_chunk() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // Less than one window — no inference should run
        let samples = vec![0.0f32; 100];
        let result = vad.process(&samples).unwrap();
        assert_eq!(result.probability, 0.0);
        assert_eq!(vad.accumulator.len(), 100);
    }

    #[test]
    fn test_vad_reset() {
        let mut vad = SileroVad::new(default_config()).unwrap();

        // Process something to change state
        let samples = vec![0.0f32; 512];
        vad.process(&samples).unwrap();
        vad.accumulator.extend_from_slice(&[0.0; 100]); // add some leftover

        vad.reset();

        assert!(vad.state.iter().all(|&v| v == 0.0), "State should be zeros");
        assert!(vad.accumulator.is_empty(), "Accumulator should be empty");
    }
}
