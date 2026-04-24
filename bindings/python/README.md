# decibri: Python bindings

Alpha-quality Python bindings for the [decibri](https://github.com/decibri/decibri)
audio capture / playback / voice-activity-detection library.

**Status: Phase 1 scaffold (version-info only).** Capture, output, VAD, and
numpy integration ship in 0.1.0. Async surface ships in a later 0.1.x.

## Install

Not yet on PyPI. Built locally via:

    cd bindings/python
    uv sync --dev
    uv run maturin develop --uv

## License

Apache-2.0. See the [workspace LICENSE](../../LICENSE).
