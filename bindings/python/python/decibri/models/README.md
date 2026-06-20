# Models

This directory holds the third-party model files decibri bundles, together with
the license notices each one requires. These notices ship inside the published
npm and PyPI packages alongside the model weights, so the attribution travels
with the files it covers.

## silero_vad.onnx

- **Name:** Silero VAD v5
- **License:** MIT
- **Source:** https://github.com/snakers4/silero-vad
- **Size:** ~2.2 MB
- **Purpose:** Voice Activity Detection. Determines whether an audio frame contains human speech

### Input/Output Specification

**Inputs:**
- `input`: f32[batch, window_size]: audio samples (512 at 16kHz, 256 at 8kHz)
- `sr`: i64[batch]: sample rate
- `h`: f32[2, batch, 64]: LSTM hidden state
- `c`: f32[2, batch, 64]: LSTM cell state

**Outputs:**
- `output`: f32[batch, 1]: speech probability (0.0 to 1.0)
- `hn`: f32[2, batch, 64]: updated hidden state
- `cn`: f32[2, batch, 64]: updated cell state

Note: Exact tensor names may differ. Verified at build time from the model file.

### License Notice

This model is a third-party artifact, not proprietary to decibri. It is
distributed under the MIT License, reproduced in full below.

```text
MIT License

Copyright (c) 2020-present Silero Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## fastenhancer_t.onnx

- **Name:** FastEnhancer-T (tiny tier), VoiceBank-DEMAND checkpoint, waveform variant
- **License:** MIT
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

### License Notice

This model is a third-party artifact, not proprietary to decibri. The model
code and weights are distributed under the MIT License, reproduced in full
below.

```text
MIT License

Copyright (c) 2025 AHN Sung Hwan

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Training-Data Attribution

The bundled checkpoint is trained on the VoiceBank-DEMAND noisy speech dataset,
which pairs clean speech from the CSTR VCTK Corpus with noise from the DEMAND
database. Each source dataset requires attribution:

- **VoiceBank-DEMAND.** Valentini-Botinhao, Cassia. (2017). Noisy speech
  database for training speech enhancement algorithms and TTS models, 2016
  [sound]. University of Edinburgh, School of Informatics, Centre for Speech
  Technology Research (CSTR). Licensed under Creative Commons Attribution 4.0
  International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/.
  https://doi.org/10.7488/ds/2117

- **CSTR VCTK Corpus (version 0.92).** Yamagishi, Junichi; Veaux, Christophe;
  MacDonald, Kirsten. (2019). CSTR VCTK Corpus: English Multi-speaker Corpus
  for CSTR Voice Cloning Toolkit (version 0.92) [sound]. University of
  Edinburgh, Centre for Speech Technology Research (CSTR). Licensed under the
  Open Data Commons Attribution License (ODC-By) v1.0,
  https://opendatacommons.org/licenses/by/1-0/.
  https://doi.org/10.7488/ds/2645

- **DEMAND.** Thiemann, Joachim; Ito, Nobutaka; Vincent, Emmanuel. (2013).
  DEMAND: a collection of multi-channel recordings of acoustic noise in
  diverse environments. Licensed under Creative Commons
  Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0),
  https://creativecommons.org/licenses/by-sa/3.0/.
  https://doi.org/10.5281/zenodo.1227121
