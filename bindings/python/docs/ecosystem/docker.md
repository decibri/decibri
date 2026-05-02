# decibri in Docker containers

decibri runs cleanly in Linux containers. The integration points worth knowing about are the runtime ALSA library (always required, even headless), the audio device passthrough flags (three of them, not one), and the headless behavior on hosts without `/dev/snd` (cpal exposes a null device, not an empty list).

This document covers the patterns verified in the Phase 9 implementation against Docker Desktop 4.71 (Server: linux/amd64, Engine 29.4.1, manylinux_2_34 wheel built inline). Reference Dockerfiles ship under `bindings/python/docs/ecosystem/docker/`.

## Platform constraint

`--device /dev/snd` is a Linux-host concept. Docker Desktop on Windows or macOS runs containers in a Linux VM that has no bridge to the host audio device, so the audio-passthrough scenario is end-to-end verifiable on Linux hosts only. The image itself builds and runs anywhere Docker runs; only the audio-passthrough integration is Linux-host-only.

The base and headless scenarios in this document are verified on Docker Desktop 4.71 (Windows host, WSL2 Linux backend); the audio-passthrough scenario is verified for image build and structural correctness only. Phase 11 adds a Linux-host validation note from a CI runner or production deployment.

## Base scenario: minimal install + version smoke check

Until 0.1.0 ships to PyPI, the minimal Dockerfile is multi-stage: a builder stage that compiles the wheel from source via maturin, and a slim runtime stage that pip-installs the wheel. After PyPI publish (Phase 11), the runtime stage is reduced to a single `pip install decibri` line.

[Dockerfile.base](docker/Dockerfile.base):

```dockerfile
ARG ORT_VERSION=1.24.4

FROM rust:1-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv pkg-config libasound2-dev curl patchelf \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . /src

# Stage the per-platform ONNX Runtime dylib into the wheel's resource
# directory before maturin runs (matches python-ci.yml manylinux step).
ARG ORT_VERSION
RUN ORT_TARBALL="onnxruntime-linux-x64-${ORT_VERSION}.tgz" \
    && curl -sL --retry 3 --retry-delay 2 \
        -o /tmp/ort.tgz \
        "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_TARBALL}" \
    && tar -xzf /tmp/ort.tgz -C /tmp \
    && mkdir -p /src/bindings/python/python/decibri/_ort \
    && cp -L "/tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime.so" \
        /src/bindings/python/python/decibri/_ort/libonnxruntime.so \
    && cp "/tmp/onnxruntime-linux-x64-${ORT_VERSION}/lib/libonnxruntime_providers_shared.so" \
        /src/bindings/python/python/decibri/_ort/

RUN python3 -m venv /opt/build-venv \
    && /opt/build-venv/bin/pip install --upgrade pip maturin \
    && cd /src/bindings/python \
    && /opt/build-venv/bin/maturin build --release

FROM python:3.12-slim AS runtime

# libasound2 is the runtime ALSA library cpal loads at first device
# enumeration on Linux; required even in headless deployments.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libasound2 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/target/wheels/*manylinux*.whl /tmp/
RUN pip install --no-cache-dir /tmp/decibri-*.whl && rm /tmp/decibri-*.whl

CMD ["python", "-c", "import decibri\nprint('decibri', decibri.__version__)\nv = decibri.version()\nprint(v)\n"]
```

Build and run:

```bash
docker build -t decibri:base -f bindings/python/docs/ecosystem/docker/Dockerfile.base .
docker run --rm decibri:base
```

Verified output (Docker Desktop 4.71, manylinux_2_34 wheel):

```
decibri 0.1.0
VersionInfo(decibri='3.4.0', audio_backend='cpal 0.17', binding='0.1.0')
```

Image size: ~255 MB (`python:3.12-slim` ~50 MB + libasound2 ~3 MB + decibri wheel including bundled ORT ~50 MB + Python stdlib runtime). Well under typical production image budgets.

## Headless scenario: no audio passthrough

For batch processors, CI runners, and VAD-only pipelines that operate on pre-recorded audio bytes, run the container without `--device /dev/snd`. The image is identical to the base; only the runtime invocation differs.

[Dockerfile.headless](docker/Dockerfile.headless) is structurally identical to the base; the difference is in expected behavior at runtime.

```bash
docker run --rm decibri:headless
```

Verified output (Docker Desktop 4.71, no audio passthrough):

```
input_devices: [DeviceInfo(index=0, name='Discard all samples (playback) or generate zero samples (capture)', id='alsa:null', max_input_channels=2, default_sample_rate=44100, is_default=false)]
output_devices: [OutputDeviceInfo(index=0, name='Discard all samples (playback) or generate zero samples (capture)', id='alsa:null', max_output_channels=2, default_sample_rate=44100, is_default=false)]
Microphone() constructed UNEXPECTEDLY in headless container
```

**Important nuance:** in a headless container ALSA exposes a `null` device (`alsa:null`) regardless of whether `/dev/snd` is passed through. cpal happily lists it as a device, and `decibri.Microphone()` will construct successfully selecting it as the default. Reading from a null-device-backed Microphone yields zero-valued samples; it does not raise.

For deployments that want to fail loudly when no real audio device is available, check the device id explicitly:

```python
import decibri

devs = decibri.input_devices()
real_devs = [d for d in devs if d.id != "alsa:null"]
if not real_devs:
    raise RuntimeError(
        "No real audio input device available. "
        "If running in Docker, pass --device /dev/snd and --group-add audio."
    )
```

For VAD-only pipelines that operate on pre-recorded audio bytes, the headless null device is irrelevant; you do not call `Microphone` at all and instead feed bytes to a `vad="silero"` pipeline directly via the SileroVad core API (or, in 0.2.0+, via a documented Python helper). The container needs only the runtime image; no audio passthrough flags.

## Audio passthrough scenario: three flags, not one

`--device /dev/snd` is necessary but not sufficient for audio capture in a non-root container. Three flags work in concert:

| Flag                     | Why it is required                                              |
| ------------------------ | --------------------------------------------------------------- |
| `--device /dev/snd:/dev/snd` | Expose the host ALSA device nodes to the container.         |
| `--group-add audio`      | Grant the container user audio group membership. Without this, non-root containers see the device files but `open(2)` returns `EPERM`. |
| `-e ALSA_PCM_CARD=0`     | Pin cpal to the raw ALSA hardware device rather than the `default` PCM, which the host sound server (PipeWire / PulseAudio) holds exclusively. |

[Dockerfile.audio](docker/Dockerfile.audio) builds a non-root image (`USER decibri`, uid 1000) so the `--group-add audio` flag is meaningful at run time.

Build and run on a Linux host with audio:

```bash
docker build -t decibri:audio -f bindings/python/docs/ecosystem/docker/Dockerfile.audio .

docker run --rm \
    --device /dev/snd:/dev/snd \
    --group-add audio \
    -e ALSA_PCM_CARD=0 \
    decibri:audio
```

Expected output on a Linux host with at least one audio device (build verified on Docker Desktop 4.71; end-to-end audio runtime verified separately on Linux host as a Phase 11 docs note):

```
input_devices count: <N>
  - DeviceInfo(index=0, name='hw:0,0', ..., is_default=true)
  - ...
decibri version: VersionInfo(decibri='3.4.0', audio_backend='cpal 0.17', binding='0.1.0')
```

If `input_devices count: 1` and the single device id is `alsa:null`, audio passthrough is incomplete; check the three flags above.

If the host audio gid does not match the conventional 29, use `--group-add $(getent group audio | cut -d: -f3)` instead of the literal `audio` name.

If PipeWire or PulseAudio is running on the host and you want decibri to use the sound server's bridge rather than ALSA's `default`, mount the PulseAudio socket and set `device="pulse"` at the Python layer:

```bash
docker run --rm \
    --device /dev/snd:/dev/snd \
    --group-add audio \
    -v /run/user/$(id -u)/pulse:/run/user/1000/pulse \
    -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
    decibri:audio python -c "import decibri; print(decibri.Microphone(device='pulse'))"
```

PulseAudio socket integration is host-distro-specific; for production deployments with a strong opinion on backend, the cleaner path is to run decibri on bare metal or use ALSA passthrough with `ALSA_PCM_CARD` pinning.

## Image size and layer breakdown

Verified on Docker Desktop 4.71, all three images using `python:3.12-slim` runtime base:

| Image                    | Size   | Notes                                                  |
| ------------------------ | ------ | ------------------------------------------------------ |
| `decibri:base`           | 255 MB | python:3.12-slim + libasound2 + decibri wheel          |
| `decibri:headless`       | 255 MB | identical to base; differs only in expected behavior   |
| `decibri:audio`          | 267 MB | base + alsa-utils + non-root user setup                |

The bundled ORT dylib accounts for ~15 to 20 MB inside the wheel; the Python wheel itself ships at ~25 MB. A 0.2.0+ `decibri-no-ort` variant for `vad=False` / `vad="energy"` users would shave roughly 20 MB from the runtime layer; tracked in the 0.2.0 backlog.

## Recommended deployment patterns

| Use case                                | Pattern                                                    |
| --------------------------------------- | ---------------------------------------------------------- |
| Production live audio capture           | Linux host, ALSA passthrough with the three flags          |
| Batch processing of pre-recorded audio  | Headless container; do not call `Microphone()`             |
| CI runners, smoke tests                 | Headless container; use `vad="energy"` or pre-recorded bytes |
| Local dev on macOS / Windows host       | Use the binding outside Docker; Docker Desktop has no audio bridge |

## Reference Dockerfiles

- [docker/Dockerfile.base](docker/Dockerfile.base)
- [docker/Dockerfile.headless](docker/Dockerfile.headless)
- [docker/Dockerfile.audio](docker/Dockerfile.audio)

All three are verified to build cleanly against Docker Desktop 4.71 (Server: linux/amd64, Engine 29.4.1, BuildKit). The base and headless runtime behaviors are verified end-to-end; the audio-passthrough runtime requires a Linux host (Phase 11 verification).
