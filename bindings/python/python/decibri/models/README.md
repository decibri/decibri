# Models

This directory holds the third-party model files decibri bundles, together with
the tensor interface each one exposes. The license notice every model requires
is in `THIRD-PARTY-NOTICES.md` beside this file, and ships alongside the weights
it covers inside the published npm and PyPI packages.

## silero_vad.onnx

- **Model:** Silero VAD v6.2
- **Source:** https://github.com/snakers4/silero-vad (release `v6.2`)
- **Size:** ~2.2 MB (2,327,524 bytes)
- **Purpose:** Voice Activity Detection. Determines whether an audio frame contains human speech

### Input/Output Specification

Verified from the model file (ONNX IR version 8, opset 16, exported with spox).
All tensors are fp32 except `sr`, which is i64. The batch, sample-count and
state dimensions are declared dynamic; the shapes below are what the model takes
and produces at batch 1.

**Inputs:**
- `input`: f32[batch, context_size + window_size]: audio samples. Each call is fed the previous window's last `context_size` samples followed by the current window (64 + 512 = 576 at 16 kHz, 32 + 256 = 288 at 8 kHz), where `context_size` is `window_size / 8`.
- `state`: f32[2, batch, 128]: combined LSTM hidden and cell state
- `sr`: i64 scalar: sample rate

**Outputs:**
- `output`: f32[batch, 1]: speech probability (0.0 to 1.0)
- `stateN`: f32[2, batch, 128]: updated state

Streaming contract: the model carries no state of its own between calls. Initialize
`state` to zeros and the audio context to zeros at the start of a stream, feed each
call's `stateN` back as the next call's `state`, and carry the last `context_size`
samples of each window forward as the next call's context. Feeding a bare window
with no context prepended scores even loud speech at the non-speech floor.

## fastenhancer_t.onnx

- **Model:** FastEnhancer-T (tiny tier), VoiceBank-DEMAND checkpoint, waveform variant
- **Source:** https://github.com/aask1357/fastenhancer (release `onnx-vd-v1.0.0`)
- **Size:** ~122 KB (125,036 bytes)
- **Purpose:** Single-channel speech enhancement (denoise). Maps a window of noisy speech samples to a hop of cleaned speech samples, frame by frame, for streaming use.

### Input/Output Specification

Verified at build time from the model file (ONNX IR version 10, opset 18,
exported from PyTorch 2.7.1). All tensors are fp32.

**Inputs:**
- `wav_in`: f32[1, 512]: one analysis window of audio samples.
- `cache_in_0`: f32[1, 256]: the overlap-add tail buffer (one hop), carried across calls.
- `cache_in_1`: f32[1, 16, 20]: streaming recurrent cache, carried across calls.
- `cache_in_2`: f32[1, 16, 20]: streaming recurrent cache, carried across calls.

**Outputs:**
- `wav_out`: f32[1, 256]: one hop of enhanced audio samples.
- `cache_out_0`: f32[1, 256]: updated overlap-add buffer, feed back as `cache_in_0` on the next call.
- `cache_out_1`: f32[1, 16, 20]: updated cache, feed back as `cache_in_1` on the next call.
- `cache_out_2`: f32[1, 16, 20]: updated cache, feed back as `cache_in_2` on the next call.

Streaming contract: this is the waveform-in/waveform-out variant. The model bakes
the complete spectral path into the graph (windowing, the forward and inverse
`DFT`, power compression, and overlap-add), so the host feeds audio samples and
receives audio samples with no spectral processing of its own. The three cache
tensors hold the model's overlap-add and recurrent state. Initialize them to
zeros at the start of a stream and feed each call's `cache_out` values back as
the next call's `cache_in` values.
