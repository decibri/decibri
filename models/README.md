# Models

## silero_vad.onnx

- **Name:** Silero VAD v5
- **License:** MIT
- **Source:** https://github.com/snakers4/silero-vad
- **Size:** ~2.2 MB
- **Purpose:** Voice Activity Detection — determines whether an audio frame contains human speech

### Input/Output Specification

**Inputs:**
- `input`: f32[batch, window_size] — audio samples (512 at 16kHz, 256 at 8kHz)
- `sr`: i64[batch] — sample rate
- `h`: f32[2, batch, 64] — LSTM hidden state
- `c`: f32[2, batch, 64] — LSTM cell state

**Outputs:**
- `output`: f32[batch, 1] — speech probability (0.0 to 1.0)
- `hn`: f32[2, batch, 64] — updated hidden state
- `cn`: f32[2, batch, 64] — updated cell state

Note: Exact tensor names may differ — verified at build time from the model file.

### License Notice

This model is a third-party artifact, not proprietary to decibri.
It is distributed under the MIT License. See the Silero VAD repository for full license text.
