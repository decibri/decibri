# Third-Party Notices

This directory redistributes the model files listed below, together with the
license notices they require. The notices ship inside the published npm and
PyPI packages alongside the model weights they cover, so the attribution
travels with the files. `README.md` beside this file documents each model's
tensor interface.

## Silero VAD

- **Name:** Silero VAD
- **Version:** v6.2
- **License:** MIT
- **Source:** https://github.com/snakers4/silero-vad (release `v6.2`)
- **Files covered:** `silero_vad.onnx`

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

## FastEnhancer

- **Name:** FastEnhancer-T (tiny tier), VoiceBank-DEMAND checkpoint, waveform variant
- **Version:** `onnx-vd-v1.0.0`
- **License:** MIT
- **Source:** https://github.com/aask1357/fastenhancer (release `onnx-vd-v1.0.0`)
- **Files covered:** `fastenhancer_t.onnx`

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
