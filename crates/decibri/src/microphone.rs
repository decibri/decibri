//! Microphone capture: open an input device and pull [`AudioChunk`]s from it.
//!
//! Build a [`MicrophoneConfig`], construct a [`Microphone`], then
//! [`start`](Microphone::start) it to obtain a [`MicrophoneStream`]. Read audio
//! with [`next_chunk`](MicrophoneStream::next_chunk) (blocking, with a timeout)
//! or [`try_next_chunk`](MicrophoneStream::try_next_chunk) (non-blocking). Both
//! take a requested sample count and deliver exactly that many interleaved
//! samples per chunk, re-blocking the device's native capture buffers on the
//! consumer side. The final chunk at stream close may be shorter, carrying the
//! remaining tail (no captured sample is dropped). The playback counterpart is
//! [`crate::speaker`].

#[cfg(feature = "capture")]
use std::collections::VecDeque;
#[cfg(feature = "capture")]
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
#[cfg(feature = "capture")]
use std::sync::{Arc, Mutex, PoisonError};
#[cfg(feature = "capture")]
use std::time::{Duration, Instant};

#[cfg(feature = "capture")]
use crossbeam_channel::{Receiver, RecvTimeoutError, Sender, TryRecvError, TrySendError};

#[cfg(feature = "capture")]
use crate::backend::{
    AudioBackend, BackendDevice, BackendStream, CpalBackend, InputDataCallback,
    StreamErrorCallback, StreamParams,
};
use std::path::PathBuf;

use crate::device::DeviceSelector;
use crate::error::DecibriError;
#[cfg(feature = "capture")]
use crate::stage::{build_capture_stage, CaptureStage, Transforms};
#[cfg(all(feature = "capture", feature = "aec"))]
use crate::stage::{AecReferenceRing, AecSettings};

/// Single-channel speech-enhancement (denoise) model selector.
///
/// A closed, `#[non_exhaustive]` set: today the only value is
/// [`DenoiseModel::FastEnhancerT`]. Naming the model rather than taking a bool
/// keeps adding further models a non-breaking widening (a new variant), and
/// keeps the caller on record about which model, and which license, they
/// invoked. The model weights ship with the binding that bundles them; the core
/// loads them from [`MicrophoneConfig::denoise_model_path`] and embeds no model
/// bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum DenoiseModel {
    /// FastEnhancer-T: the tiny tier of FastEnhancer, the VoiceBank-DEMAND
    /// waveform checkpoint. Maps a window of noisy speech samples to a hop of
    /// cleaned speech samples, frame by frame, for streaming use.
    FastEnhancerT,
}

/// High-pass filter selector for the capture chain.
///
/// A closed, `#[non_exhaustive]` set that is intentionally designed to grow:
/// today the values are [`HighpassFilter::Hz80`] (an 80 Hz second-order
/// Butterworth high-pass, the conventional voice rumble cutoff) and
/// [`HighpassFilter::Hz100`] (a 100 Hz second-order Butterworth, the more
/// aggressive rumble cut). Both remove low-frequency rumble below the voice
/// band. Naming the cutoff rather than taking a bool or a free integer keeps
/// adding further cutoffs (a `300` Hz telephony cut, say) a non-breaking
/// widening (a new variant), and keeps the caller on record about which cutoff
/// they selected. The closed named set is deliberate: members are added without
/// a breaking change, the way [`DenoiseModel`] grows. The filter is pure DSP, so
/// it bundles no file and loads no runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum HighpassFilter {
    /// An 80 Hz second-order Butterworth high-pass, the conventional voice
    /// rumble cutoff.
    Hz80,
    /// A 100 Hz second-order Butterworth high-pass, a more aggressive rumble
    /// cut than [`HighpassFilter::Hz80`].
    Hz100,
}

impl HighpassFilter {
    /// The corner (-3 dB) cutoff frequency in Hz that this variant selects. The
    /// single source of the cutoff value: the biquad design reads it from here
    /// rather than carrying a separate magic number.
    pub(crate) fn cutoff_hz(self) -> f32 {
        match self {
            HighpassFilter::Hz80 => 80.0,
            HighpassFilter::Hz100 => 100.0,
        }
    }
}

/// The source of the detector feed: which of the delivered channels the
/// voice-activity detector reads.
///
/// A closed, `#[non_exhaustive]` set, designed to grow the way
/// [`DenoiseModel`] grows. The variants name DELIVERED channels, the 0-based
/// positions of the interleaved frames a consumer receives, after any
/// [`MicrophoneConfig::channel_map`] is applied: with a map present,
/// delivered channel `j` carries device channel `channel_map[j]`, and the
/// source names `j`, never the device index. A device channel the map
/// delivers at two positions is therefore named by position, without
/// ambiguity. Selecting a source changes which samples reach the detector
/// and nothing else: the delivered audio is untouched.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum DetectorSource {
    /// The frame average of every delivered channel, the default: each
    /// interleaved frame collapses to the arithmetic mean of its channels
    /// before the detector reads it. At one delivered channel this is the
    /// identity.
    #[default]
    Average,
    /// One delivered channel, by 0-based index: the detector reads that
    /// channel's samples alone. The index must be below the configured
    /// delivered channel count
    /// ([`DecibriError::DetectorSourceOutOfRange`] otherwise); the count is
    /// the only ceiling, so no fixed maximum exists.
    Channel(u16),
}

/// Configuration for a microphone capture session.
///
/// `#[non_exhaustive]`: construct it with [`MicrophoneConfig::default`] and then
/// assign the public fields you need. Direct struct-literal construction from
/// another crate is intentionally not supported, so adding a field later stays
/// backward compatible.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MicrophoneConfig {
    /// Sample rate in Hz. Range: 1000–384000. Default: 16000.
    pub sample_rate: u32,
    /// Number of channels the stream DELIVERS, interleaved frame by frame in
    /// [`AudioChunk::data`]. Bounded below at 1 (the default) by
    /// [`validate`](Self::validate); bounded above by the resolved device
    /// alone, which reports its own count when the stream starts. No fixed
    /// maximum exists.
    ///
    /// The device itself is opened at its own native channel count, exactly as
    /// it is opened at its native rate, and decibri derives the delivered
    /// channels from it. This mirrors [`sample_rate`](Self::sample_rate),
    /// which is also the delivered figure rather than the device-open one. The
    /// playback counterpart, [`crate::speaker::SpeakerConfig::channels`], is
    /// the count offered to the device, because playback has no delivered
    /// side; on both surfaces the field is the shape of the audio you exchange
    /// with decibri.
    ///
    /// How the delivered channels are derived from the device's, when
    /// [`channel_map`](Self::channel_map) is `None`:
    ///
    /// - `1`: the documented average of every opened channel.
    /// - equal to the device's own count: every device channel, in device
    ///   order.
    /// - above the device's own count:
    ///   [`DecibriError::MicrophoneChannelsUnsupported`] when the stream
    ///   starts.
    /// - above 1 and below the device's own count:
    ///   [`DecibriError::ChannelSelectionAmbiguous`] when the stream starts.
    ///   Which of the device's channels those should be has no single answer,
    ///   so [`channel_map`](Self::channel_map) names them rather than decibri
    ///   choosing.
    ///
    /// A [`channel_map`](Self::channel_map) names the device channels
    /// directly, so it answers every case above except a count the device does
    /// not have.
    ///
    /// The block size passed to
    /// [`MicrophoneStream::try_next_chunk`](MicrophoneStream::try_next_chunk)
    /// and [`MicrophoneStream::next_chunk`](MicrophoneStream::next_chunk) is a
    /// count of interleaved samples, so above 1 channel it must be a multiple
    /// of this count.
    pub channels: u16,
    /// Optional list of 0-based DEVICE channel indices selecting which device
    /// channels feed the delivered channels: delivered channel `j` carries
    /// device channel `channel_map[j]`. The length must equal
    /// [`channels`](Self::channels). Entries may repeat and may appear in any
    /// order, so a map both selects and permutes. `None` (the default)
    /// derives the delivered channels as [`channels`](Self::channels)
    /// documents.
    ///
    /// The same shape as CoreAudio AUHAL's channel map
    /// (`kAudioOutputUnitProperty_ChannelMap`: an array of device channel
    /// indices, one entry per client channel). NOT miniaudio's `channelMap`,
    /// which names a spatial layout (which channel is front-left, and so on).
    /// sounddevice's `mapping` is the same idea 1-based; decibri is 0-based,
    /// matching [`crate::device::DeviceSelector::Index`].
    ///
    /// Validated when the stream starts, against the resolved device's own
    /// report: every entry must be below the device's native channel count
    /// ([`DecibriError::ChannelMapOutOfRange`] otherwise), and the length must
    /// equal [`channels`](Self::channels)
    /// ([`DecibriError::ChannelMapLengthMismatch`] otherwise). The device's
    /// report is the only ceiling; no fixed maximum exists. Not checked by
    /// [`validate`](Self::validate), which has no device in scope. Default:
    /// `None`.
    pub channel_map: Option<Vec<u16>>,
    /// The source of the detector feed
    /// ([`MicrophoneStream::detector_feed`]): which of the delivered channels
    /// a voice-activity detector reads. [`DetectorSource::Average`] (the
    /// default) feeds the frame average of every delivered channel;
    /// [`DetectorSource::Channel`] feeds one delivered channel alone. Names a
    /// DELIVERED channel index, the position within the interleaved frames a
    /// consumer receives, never a device index: with a
    /// [`channel_map`](Self::channel_map) present, delivered channel `j`
    /// carries device channel `channel_map[j]` and the source names `j`.
    /// Affects only the detector feed; the delivered audio is untouched.
    ///
    /// A [`DetectorSource::Channel`] index is validated by
    /// [`validate`](Self::validate) against [`channels`](Self::channels), the
    /// delivered count ([`DecibriError::DetectorSourceOutOfRange`] when it is
    /// not below that count). The delivered count is the only ceiling; no
    /// fixed maximum exists. Default: [`DetectorSource::Average`].
    pub detector_source: DetectorSource,
    /// Frames per audio callback buffer. Range: 64–65536. Default: 1600.
    pub frames_per_buffer: u32,
    /// Device selection. Default: system default input.
    pub device: DeviceSelector,
    /// Remove a constant (DC) offset from captured audio with a one-pole
    /// DC-blocking high-pass, applied after the channel and rate normalization.
    /// Default: false (off).
    pub dc_removal: bool,
    /// Single-channel speech-enhancement (denoise) model to run on the captured
    /// audio, applied after DC removal. `None` (the default) leaves denoise off.
    /// Naming a model also requires [`denoise_model_path`](Self::denoise_model_path);
    /// with the model set but no path the stage stays off. Honoured only when the
    /// `denoise` feature is compiled in.
    pub denoise: Option<DenoiseModel>,
    /// Filesystem path to the denoise model's ONNX file, supplied by the caller
    /// (the bindings resolve it from their bundled copy; the core ships no model
    /// bytes). Required when [`denoise`](Self::denoise) names a model; ignored
    /// otherwise. Default: `None`.
    pub denoise_model_path: Option<PathBuf>,
    /// Filesystem path to the ONNX Runtime dynamic library, used to initialise
    /// ORT for the capture-path denoise stage (the same role
    /// [`VadConfig::ort_library_path`](crate::vad::VadConfig::ort_library_path)
    /// plays for the VAD). Consulted only when [`denoise`](Self::denoise) names a
    /// model and the `denoise` feature is compiled with `ort-load-dynamic`; under
    /// `ort-download-binaries` ORT is statically linked and this is ignored.
    /// `None` (the default) leaves ORT to its own discovery (the `ORT_DYLIB_PATH`
    /// environment variable, then the system loader). ORT initialises once per
    /// process (first-wins), so when a VAD has already initialised it this is a
    /// no-op. Default: `None`.
    pub ort_library_path: Option<PathBuf>,
    /// High-pass filter to apply to the captured audio, removing low-frequency
    /// rumble below the voice band. Runs in the transform chain after denoise,
    /// on the cleaned mono signal at the target rate. `None` (the default)
    /// builds no high-pass stage, leaving the captured audio full-range (a true
    /// byte-identical no-op). Pure DSP: it loads no model and needs no path.
    pub highpass: Option<HighpassFilter>,
    /// Automatic gain control target level in dBFS, applied to the captured
    /// audio. Drives the running level toward this target with a smoothed,
    /// rate-limited gain. Range: -40 to -3 dBFS (typical -18). `None` (the
    /// default) builds no level-control stage, leaving the level untouched (a
    /// true byte-identical no-op). Runs after the high-pass step, honoured only
    /// when the `gain` feature is compiled in. Pure DSP: no model, no path.
    pub agc: Option<i8>,
    /// Peak limiter ceiling in dBFS (sample-peak), applied to the captured audio.
    /// Holds the signal at or below this ceiling, the safety net that catches a
    /// transient the AGC's gain would let exceed full scale. Range: -3.0 to 0.0
    /// dBFS (typical -1.0). `None` (the default) builds no limiter stage, leaving
    /// the level untouched (a true byte-identical no-op). Runs last in the
    /// transform chain, after the level-control step, honoured only when the
    /// `gain` feature is compiled in. Pure DSP: no model, no path.
    pub limiter: Option<f32>,
    /// Acoustic echo canceller model to run on the captured audio, removing the
    /// echo of far-end audio the caller pushes through
    /// [`MicrophoneStream::push_aec_reference`]. `None` (the default) leaves echo
    /// cancellation off.
    ///
    /// Runs last in the normalize segment, on the delivered channels at the
    /// target rate, before the pre-transform detector tap: with it on, the
    /// detector reads the echo-removed signal. One canceller engine runs per
    /// delivered channel, each fed the same pushed reference and each finding
    /// its own channel's echo delay, so the processing and memory cost scale
    /// with [`channels`](Self::channels); a capture that overruns the
    /// machine's budget shows up as
    /// [`overrun_count`](MicrophoneStream::overrun_count) climbing, exactly as
    /// any other stage that outruns the consumer does. Requires
    /// [`sample_rate`](Self::sample_rate) in `8000..=48000`, narrower than the
    /// range that field otherwise accepts; a rate outside it is rejected by
    /// [`validate`](Self::validate). Honoured only when the `aec` feature is
    /// compiled in. Pure DSP: no model file, no path.
    ///
    /// With no reference pushed, the canceller's estimate is zero and the
    /// captured audio passes through unchanged.
    #[cfg(feature = "aec")]
    pub aec: Option<decibri_aec::AecModel>,
    /// Adaptive filter tail length in milliseconds for the echo canceller.
    /// Range: 16 to 500. `None` (the default) takes the canceller's own default,
    /// so the number lives in one place. Consulted only when
    /// [`aec`](Self::aec) names a model.
    #[cfg(feature = "aec")]
    pub aec_tail_ms: Option<u16>,
    /// Residual-suppression policy for the echo canceller. `None` (the default)
    /// takes the canceller's own default. Consulted only when
    /// [`aec`](Self::aec) names a model.
    #[cfg(feature = "aec")]
    pub aec_suppression: Option<decibri_aec::Suppression>,
    /// Sample rate in Hz of the far-end reference the caller pushes through
    /// [`MicrophoneStream::push_aec_reference`]. `None` (the default) means the
    /// reference is already at [`sample_rate`](Self::sample_rate). When it names
    /// a different rate, decibri converts the reference before the canceller
    /// sees it: the canceller reads one rate for both streams, and a reference
    /// at the wrong rate cancels nothing and reports no error, so the
    /// conversion is decibri's rather than the caller's. Range: 1000 to 384000.
    /// Consulted only when [`aec`](Self::aec) names a model.
    #[cfg(feature = "aec")]
    pub aec_reference_sample_rate: Option<u32>,
    /// Number of channels in the far-end reference the caller pushes through
    /// [`MicrophoneStream::push_aec_reference`]. The pushed samples are
    /// expected frame-interleaved at this count; when it is above 1, decibri
    /// collapses each frame to one mono sample (the channel average) before
    /// the samples enter the reference queue, so the queue and the canceller
    /// always carry mono. 1 (the default) means the pushed reference is
    /// already mono and is taken as is. A count of 0 is rejected by
    /// [`validate`](Self::validate); there is no upper bound. Consulted only
    /// when [`aec`](Self::aec) names a model.
    ///
    /// The declared count must match the buffer actually pushed. The
    /// reference arrives as a flat slice whose true channel count is not
    /// recoverable from its length, so a mismatch is not detected and raises
    /// no error: the frames are misread, nothing is cancelled, and the
    /// observable signature is `delay_samples` staying `None` in
    /// [`MicrophoneStream::aec_metrics`] while the canceller reports no
    /// fault.
    ///
    /// A mono reference against playback through more than one loudspeaker
    /// has a cancellation ceiling: the echo reaching the microphone is the
    /// sum of different room responses driven by different signals, and a
    /// single-reference canceller models one response applied to their
    /// average, so a placement where those paths differ leaves a residual
    /// that no amount of adaptation removes.
    #[cfg(feature = "aec")]
    pub aec_reference_channels: u16,
}

impl Default for MicrophoneConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            channel_map: None,
            detector_source: DetectorSource::Average,
            frames_per_buffer: 1600,
            device: DeviceSelector::Default,
            dc_removal: false,
            denoise: None,
            denoise_model_path: None,
            ort_library_path: None,
            highpass: None,
            agc: None,
            limiter: None,
            #[cfg(feature = "aec")]
            aec: None,
            #[cfg(feature = "aec")]
            aec_tail_ms: None,
            #[cfg(feature = "aec")]
            aec_suppression: None,
            #[cfg(feature = "aec")]
            aec_reference_sample_rate: None,
            #[cfg(feature = "aec")]
            aec_reference_channels: 1,
        }
    }
}

impl MicrophoneConfig {
    /// Validate the configuration: sample rate, channel count, detector
    /// source, buffer size, the AGC target, the limiter ceiling, and the
    /// echo-cancellation rates and reference channel count (each when set)
    /// must fall within the supported ranges.
    pub fn validate(&self) -> Result<(), DecibriError> {
        if !(1000..=384000).contains(&self.sample_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
        }
        if self.channels == 0 {
            return Err(DecibriError::ChannelsOutOfRange);
        }
        // No upper bound here. The delivered count is bounded by the resolved
        // device alone, which is not in scope in this pure function, so the
        // ceiling is applied at open time as
        // `DecibriError::MicrophoneChannelsUnsupported`.
        // The detector source names a delivered channel, and the delivered
        // count is `channels`, which IS in scope here unlike the device: an
        // index at or above it can never resolve whatever device answers at
        // start(). The delivered count is the only ceiling; no fixed maximum
        // exists.
        if let DetectorSource::Channel(index) = self.detector_source {
            if index >= self.channels {
                return Err(DecibriError::DetectorSourceOutOfRange {
                    index,
                    channels: self.channels,
                });
            }
        }
        if !(64..=65536).contains(&self.frames_per_buffer) {
            return Err(DecibriError::FramesPerBufferOutOfRange);
        }
        // The AGC target is `Option<i8>`, so an out-of-range value can reach the
        // core directly from a Rust consumer that bypasses the bindings. Guard it
        // here, the load-bearing backstop, returning an error rather than
        // clamping (matching `sample_rate`).
        if let Some(target) = self.agc {
            if !(-40..=-3).contains(&target) {
                return Err(DecibriError::AgcTargetOutOfRange);
            }
        }
        // The limiter ceiling is `Option<f32>`, so an out-of-range value can reach
        // the core directly from a Rust consumer that bypasses the bindings. Guard
        // it here, the load-bearing backstop, returning an error rather than
        // clamping (matching `agc`).
        if let Some(ceiling) = self.limiter {
            if !(-3.0..=0.0).contains(&ceiling) {
                return Err(DecibriError::LimiterCeilingOutOfRange);
            }
        }
        // The echo canceller accepts a narrower rate window than the range
        // checked above, so a target rate that is valid for a plain capture is
        // rejected here once echo cancellation is on. Checked after the general
        // range check, so a rate outside decibri's own range still reports that.
        // The window mirrors the one the canceller enforces at construction;
        // `aec_window_matches_the_cancellers_own` holds the two together.
        #[cfg(feature = "aec")]
        if self.aec.is_some() {
            if !(8000..=48000).contains(&self.sample_rate) {
                return Err(DecibriError::AecSampleRateUnsupported(self.sample_rate));
            }
            // The declared reference rate is converted to the target rate, so it
            // has to be a rate decibri resamples from at all. Guarded here rather
            // than left to the resampler, so it reads as the rate error it is.
            if let Some(rate) = self.aec_reference_sample_rate {
                if !(1000..=384000).contains(&rate) {
                    return Err(DecibriError::SampleRateOutOfRange);
                }
            }
            // The declared reference channel count sets the interleave stride
            // the push collapses by, and a stride of 0 describes no buffer at
            // all. There is deliberately no upper bound: the count declares the
            // shape of the caller's own buffer, not a capability decibri
            // enforces.
            if self.aec_reference_channels == 0 {
                return Err(DecibriError::AecConfigInvalid {
                    reason: "the reference channel count must be at least 1".to_string(),
                });
            }
        }
        Ok(())
    }
}

/// A chunk of captured audio data.
///
/// `#[non_exhaustive]`: produced by the capture path and read field by field by
/// consumers. Sealing it keeps future metadata additions backward compatible.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AudioChunk {
    /// Interleaved f32 samples, normally in the range [-1.0, 1.0]. The conditioning
    /// chain sanitizes non-finite input, so a conditioned capture always delivers
    /// finite samples. The range is guaranteed when the limiter is enabled, which
    /// bounds the output to its ceiling. Two stages can drive samples above full
    /// scale without it: automatic gain control, whose gain can lift a loud
    /// passage past 1.0, and echo cancellation, which subtracts its estimate of
    /// the echo from the capture and exceeds the capture wherever that estimate
    /// is wrong in phase. Enable the limiter, which runs after both, to keep
    /// every sample within range; a caller reading f32 without it should clamp
    /// its own output before anything that assumes the range.
    ///
    /// The `int16` sample format clamps, so an over-scale sample reaches an
    /// `int16` consumer as full scale rather than wrapping.
    pub data: Vec<f32>,
    /// Sample rate of this chunk.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u16,
}

/// Bound on the capture channel. A stalled consumer cannot grow memory without
/// limit: once this many `AudioChunk`s are queued, the realtime callback drops
/// new chunks (counting them, see [`MicrophoneStream::overrun_count`]) rather
/// than blocking the audio thread or allocating without bound.
///
/// This is a memory bound, not a fixed-duration guarantee. One queued item is
/// one cpal callback buffer, whose duration is backend-dependent: on WASAPI cpal
/// ignores `BufferSize::Fixed` and delivers the driver period (often ~10 ms, not
/// `frames_per_buffer`), so 64 items can be anywhere from well under a second to
/// several seconds of audio. It is sized so a consumer that keeps pace never
/// drops; only a genuine stall does.
#[cfg(feature = "capture")]
const CAPTURE_CHANNEL_CAPACITY: usize = 64;

/// Memory bound for the pre-transform VAD tap, in seconds of audio at the target
/// rate. The tap accumulates the post-normalize signal in lockstep with the
/// delivered output and is drained by [`MicrophoneStream::vad_input`]. A consumer
/// that enables an enhancement step but never drains the tap would otherwise grow
/// memory without limit; beyond this many seconds the oldest tapped samples are
/// dropped. Sized far above the in-flight reblock depth and any transform
/// latency, so an actively draining VAD never reaches it even when the tap leads
/// the delivered output (as it does once a length-changing denoise stage runs).
#[cfg(feature = "capture")]
const VAD_TAP_BOUND_SECS: usize = 2;

/// Memory bound for the far-end echo-cancellation reference queue, in seconds of
/// audio at the declared reference rate. The caller pushes played audio into the
/// queue with [`MicrophoneStream::push_aec_reference`] and the capture chain
/// reads it at the rate the capture consumes it, so a caller feeding in step with
/// the capture never holds more than one block of it.
///
/// What the bound is sized for is the caller who does not feed in step: one that
/// hands over a whole synthesized utterance in a single call, or hands over a
/// greeting before it has read its first chunk. The queue holds that push whole
/// up to this many seconds and it is read out over the capture it echoes into,
/// so the bound is how far AHEAD OF ITS OWN CAPTURE a caller may run. A caller
/// pushing a second of audio for every second of capture never approaches it,
/// however large its individual pushes are; only one pushing faster than real
/// time for longer than this does.
///
/// Two seconds covers the pattern this exists for, a synthesis pipeline handing
/// over one utterance at a time, with the next utterance able to arrive before
/// the previous one has finished playing.
///
/// A push larger than the bound is truncated at its newest end and the discarded
/// samples are counted by [`MicrophoneStream::aec_reference_dropped`]. The time
/// they occupied is still represented: the feed supplies silence for every
/// near-end sample the caller did not cover, so the cost is the cancellation of
/// the discarded span alone.
///
/// At 16 kHz the queue holds 32000 samples, 128 KB; at 48 kHz, 96000 samples,
/// 384 KB.
#[cfg(all(feature = "capture", feature = "aec"))]
pub(crate) const AEC_REFERENCE_BOUND_SECS: usize = 2;

/// An open capture stream you pull [`AudioChunk`]s from.
///
/// Obtained from [`Microphone::start`]. Read audio with
/// [`next_chunk`](Self::next_chunk) (blocking, with a timeout) or
/// [`try_next_chunk`](Self::try_next_chunk) (non-blocking). Call
/// [`stop`](Self::stop) to end capture and release the device; dropping the
/// handle also releases it. The type is `Send + Sync`.
#[cfg(feature = "capture")]
pub struct MicrophoneStream {
    /// Owns the platform audio stream for the lifetime of this handle.
    ///
    /// Held behind [`BackendStream`](crate::backend::BackendStream), which keeps
    /// the stream behind a `Mutex<Option<...>>` so [`stop`](Self::stop) can drop
    /// it under `&self` to release the device while this type stays
    /// `Send + Sync` (the bindings require `Sync` for their `#[pyclass]` and
    /// `py.detach`). Dropping this field also releases the device. The test-only
    /// `tests::test_stream()` helper builds it with `BackendStream::empty()`
    /// (no real device); production stores the opened stream.
    _stream: BackendStream,
    receiver: Receiver<AudioChunk>,
    running: Arc<AtomicBool>,
    sample_rate: u32,
    channels: u16,
    // Last device/driver error reported by the cpal error callback while
    // streaming, retrievable via [`take_last_error`](Self::take_last_error).
    // Lets a consumer that sees a closed stream distinguish a driver failure
    // from an explicit `stop()`. The cpal callback writes it; consumers drain
    // it. Uncontended in practice (written at most once, on failure).
    last_error: Arc<Mutex<Option<DecibriError>>>,
    // Count of capture buffers dropped because the channel was full (a stalled
    // consumer). Incremented by the realtime callback, read via
    // [`overrun_count`](Self::overrun_count). Bounds memory by dropping rather
    // than queuing without limit.
    overruns: Arc<AtomicU64>,
    // Consumer-side re-block buffer: a FIFO of interleaved `f32` samples pulled
    // from `receiver`, drained in fixed `samples`-sized blocks so
    // [`next_chunk`](Self::next_chunk) / [`try_next_chunk`](Self::try_next_chunk)
    // deliver exactly the requested size regardless of the device's native
    // buffer size. Lives behind a `Mutex` (kept off the realtime callback, which
    // only `try_send`s native buffers) so the type stays `Send + Sync` and
    // concurrent consumers serialize. `Mutex<VecDeque<f32>>` is `Send + Sync`
    // because `VecDeque<f32>` is `Send`.
    reblock_buffer: Mutex<VecDeque<f32>>,
    // The capture stage chain, applied to each native block before it lands in
    // `reblock_buffer`. `None` when no conditioning is needed (an already-mono
    // device), keeping the drain on the direct, zero-cost no-chain path. When `Some`,
    // the chain runs behind its own `Mutex` so the stage buffers mutate under a
    // shared `&self` while the type stays `Send + Sync` (a `CaptureStage` is
    // `Send`, so `Mutex<CaptureStage>` is `Send + Sync`).
    capture_stage: Option<Mutex<CaptureStage>>,
    // One-time guard for the close-path chain flush. The chain's stages carry
    // conditioning state between blocks (the resampler holds its anti-alias
    // filter's group-delay tail); at close that held tail is drained once into
    // `reblock_buffer` so it is delivered rather than dropped. Set the first time
    // close is detected with a chain present, under the `reblock_buffer` lock, so
    // the drain runs exactly once. `AtomicBool` keeps the type `Send + Sync`.
    //
    // Once set it also closes the chain to further input: the stages hold no
    // state a later block can continue from, so `next_chunk` / `try_next_chunk`
    // stop pulling from the channel and `ingest` rejects a block that reaches it
    // anyway. Stays false for the lifetime of a stream with no chain, which has
    // nothing to flush and keeps its direct delivery unchanged.
    chain_flushed: AtomicBool,
    // Pre-transform (post-normalize) side channel for the VAD feed. `Some` only
    // when the chain has a `transform` segment, so the delivered output (which is
    // post-transform) differs from the signal a detector should read. It holds the
    // post-normalize signal at real-time rate, filled per block during
    // `ingest`/`flush_chain` and drained by [`vad_input`](Self::vad_input). When
    // every transform is length-preserving (the DC-removal step), the tap and the
    // delivered output advance one-for-one. When a length-changing, latency-
    // introducing transform is present (denoise re-blocks into frames), the tap
    // LEADS the delivered enhanced output by the chain's latency: it accrues real-
    // time samples while the delivered stream lags by the framing delay. That lead
    // is bounded (`vad_tap_cap` caps it) and invisible to VAD consumers, which keep
    // only a rolling probability scalar and never pair a score to specific returned
    // samples; the exact tap-vs-returned skew is not compensated for here. `None`
    // when there is no transform: the delivered output already is the
    // post-normalize signal, so the tap is unused and `vad_input` returns `None`
    // with zero overhead. `Mutex<VecDeque<f32>>` is `Send + Sync`, so the type
    // stays `Send + Sync`.
    vad_tap: Option<Mutex<VecDeque<f32>>>,
    // The channel count the tap signal is interleaved at: the chain's
    // post-normalize count, read from the chain at construction, or the
    // device count when there is no chain (where the tap is never active).
    // [`detector_feed`](Self::detector_feed) collapses the tap at this count,
    // which is fixed for the stream's lifetime, so no lock is needed to read
    // it.
    tap_channels: u16,
    // The source of the detector feed
    // ([`MicrophoneConfig::detector_source`]): the collapse
    // [`detector_feed`](Self::detector_feed) applies, to the tap at
    // `tap_channels` and to a multichannel delivered chunk at `channels`.
    // Copied from the configuration at start and fixed for the stream's
    // lifetime.
    detector_source: DetectorSource,
    // Memory bound (in samples) for `vad_tap`, computed from the target rate at
    // construction (see `VAD_TAP_BOUND_SECS`). Oldest tapped samples are dropped
    // beyond this so a consumer that enables an enhancement step but never drains
    // `vad_input` cannot grow memory without limit. Never reached while a VAD
    // actively drains the tap, so it does not perturb alignment.
    vad_tap_cap: usize,
    // The far-end reference queue shared with the chain's echo-cancellation
    // stage. `Some` only when echo cancellation is on. Held here as well as in
    // the stage so [`push_aec_reference`](Self::push_aec_reference) reaches it
    // without taking `capture_stage`'s lock, which the chain holds for the
    // duration of every block.
    #[cfg(feature = "aec")]
    aec_reference: Option<Arc<AecReferenceRing>>,
    // The declared channel count of the pushed far-end reference
    // ([`MicrophoneConfig::aec_reference_channels`]). Above 1,
    // [`push_aec_reference`](Self::push_aec_reference) collapses each
    // interleaved frame to one mono sample before it enters the queue; the
    // queue itself stays sized and filled in mono samples.
    #[cfg(feature = "aec")]
    aec_reference_channels: u16,
}

#[cfg(feature = "capture")]
impl MicrophoneStream {
    /// Direct access to the underlying `crossbeam_channel::Receiver`.
    ///
    /// Intended for **in-process Rust consumers** and for bindings (like the
    /// decibri Node.js addon) that integrate the channel into their own
    /// drain pump or event loop.
    ///
    /// FFI bindings targeting languages without native `crossbeam_channel`
    /// support, such as Python and the eventual mobile platforms, should
    /// prefer [`try_next_chunk`](Self::try_next_chunk) and
    /// [`next_chunk`](Self::next_chunk): they expose the same data with a
    /// three-state return (`Some` / `None` / `Err(MicrophoneStreamClosed)`)
    /// that maps cleanly across a language boundary.
    pub fn receiver(&self) -> &Receiver<AudioChunk> {
        &self.receiver
    }

    /// Attempt to read exactly `samples` interleaved samples without blocking.
    ///
    /// `samples` is the requested block size in interleaved `f32` samples
    /// (frames times channels). The device's native capture buffers are
    /// re-blocked on the consumer side, so every returned chunk holds exactly
    /// `samples` samples. `samples` must be a whole number of frames, meaning
    /// a multiple of [`channels`](Self::channels); a size that is not is
    /// refused with [`DecibriError::BlockSizeNotFrameAligned`] rather than
    /// splitting a frame across the chunk boundary. Every size is a whole
    /// number of frames on a mono stream.
    ///
    /// # Returns
    /// - `Ok(Some(chunk))`: a full block of exactly `samples` samples was
    ///   available and has been dequeued.
    /// - `Ok(None)`: the stream is open and running, but fewer than `samples`
    ///   samples are buffered. Try again shortly, or call
    ///   [`next_chunk`](Self::next_chunk) to block until a full block arrives.
    /// - `Err(DecibriError::MicrophoneStreamClosed)`: the stream is closed
    ///   (either by explicit [`stop`](Self::stop) or by an audio-driver error
    ///   reported via the cpal error callback) and the buffer is now empty (any
    ///   final tail was already delivered as a short chunk). No further chunks
    ///   will ever be available.
    ///
    /// Buffered samples that form one or more full blocks are delivered first.
    /// Once the stream closes with fewer than `samples` remaining, the final
    /// partial block (1..`samples` samples) is delivered as one short chunk, and
    /// the closed signal is returned on the next call once the buffer is empty.
    /// No captured sample is dropped: every chunk is exactly `samples` long
    /// except the final chunk at close, which carries the remaining tail.
    ///
    /// # Thread safety
    /// May be called from any thread. Takes the re-block buffer mutex, so
    /// concurrent callers serialize on it; a non-blocking
    /// `crossbeam_channel::try_recv` drains the native buffers into it.
    ///
    /// # Stability
    /// Part of decibri's stable FFI-consumer API surface, alongside
    /// [`next_chunk`](Self::next_chunk).
    pub fn try_next_chunk(&self, samples: usize) -> Result<Option<AudioChunk>, DecibriError> {
        self.check_frame_alignment(samples)?;
        let mut buf = self
            .reblock_buffer
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        // The chain has already been drained, so the stream is over: deliver what
        // is buffered and leave the channel alone. Checked before the drain so a
        // block arriving now is never fed through the emptied chain. Never taken
        // on the no-chain path, which has nothing to flush.
        if self.chain_flushed.load(Ordering::Relaxed) {
            return self.take_block_or_closed(&mut buf, samples);
        }

        // Pull every immediately-available native buffer into the re-block
        // buffer without blocking.
        let mut disconnected = false;
        loop {
            match self.receiver.try_recv() {
                Ok(chunk) => self.ingest(&mut buf, chunk)?,
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    disconnected = true;
                    break;
                }
            }
        }

        if buf.len() >= samples {
            // A full block is ready; deliver buffered data first, even when the
            // stream has since closed.
            Ok(Some(self.take_block(&mut buf, samples)))
        } else if disconnected || !self.is_open() {
            // Closed with fewer than a full block left. Drain the chain's
            // end-of-stream tail (once) into the buffer first, then deliver the
            // remaining samples as full blocks plus one final short chunk.
            self.flush_chain(&mut buf)?;
            self.take_block_or_closed(&mut buf, samples)
        } else {
            // Open, but not yet a full block. Try again shortly.
            Ok(None)
        }
    }

    /// Read exactly `samples` interleaved samples, blocking the calling thread
    /// until a full block arrives, the stream closes, or `timeout` elapses.
    ///
    /// `samples` is the requested block size in interleaved `f32` samples
    /// (frames times channels). The device's native capture buffers are
    /// re-blocked on the consumer side, so a returned chunk holds exactly
    /// `samples` samples. `samples` must be a whole number of frames, meaning
    /// a multiple of [`channels`](Self::channels); a size that is not is
    /// refused with [`DecibriError::BlockSizeNotFrameAligned`] rather than
    /// splitting a frame across the chunk boundary. Every size is a whole
    /// number of frames on a mono stream.
    ///
    /// # Arguments
    /// - `timeout = None`: block indefinitely until a full block arrives or the
    ///   stream closes.
    /// - `timeout = Some(dur)`: block at most `dur`; return `Ok(None)` if the
    ///   deadline passes before a full block accumulates. The partial block is
    ///   retained for the next call.
    ///
    /// # Returns
    /// - `Ok(Some(chunk))`: a full block of exactly `samples` samples was
    ///   received within the deadline.
    /// - `Ok(None)`: `timeout` elapsed before a full block accumulated. The
    ///   stream is still open and the partial stays buffered.
    /// - `Err(DecibriError::MicrophoneStreamClosed)`: the stream closed and the
    ///   buffer is now empty. Any full blocks buffered at the time of close are
    ///   delivered first, then the final partial block (1..`samples` samples) as
    ///   one short chunk; this error is only returned once the buffer is empty.
    ///   No captured sample is dropped.
    ///
    /// # Thread safety
    /// May be called from any thread. Blocks only the calling thread; other
    /// threads can call [`stop`](Self::stop) concurrently to unblock this call
    /// within approximately 20 ms. Holds the re-block buffer mutex for the
    /// duration of the call, so concurrent reads serialize.
    ///
    /// Implementation note: this method polls both the channel and
    /// [`is_open`](Self::is_open) at a short interval. An explicit
    /// [`stop`](Self::stop) disconnects the channel (it drops the cpal `Stream`,
    /// and with it the sender), which wakes a blocked wait promptly; the poll
    /// additionally covers a driver-error stop, which flips
    /// [`is_open`](Self::is_open) without dropping the stream.
    ///
    /// # Stability
    /// Part of decibri's stable FFI-consumer API surface.
    pub fn next_chunk(
        &self,
        samples: usize,
        timeout: Option<Duration>,
    ) -> Result<Option<AudioChunk>, DecibriError> {
        // Poll both the channel and `is_open` at this cadence so concurrent
        // `stop()` calls unblock a waiter within one interval. 20 ms is well
        // below a typical audio frame period (100 ms at 16 kHz / 1600
        // frames) so the extra wakeups cost negligible CPU.
        const POLL_INTERVAL: Duration = Duration::from_millis(20);

        self.check_frame_alignment(samples)?;
        let deadline = timeout.map(|t| Instant::now() + t);
        let mut buf = self
            .reblock_buffer
            .lock()
            .unwrap_or_else(PoisonError::into_inner);

        // The chain has already been drained, so the stream is over: deliver what
        // is buffered without waiting on the channel. Checked before the wait so a
        // block arriving now is never fed through the emptied chain. Never taken
        // on the no-chain path, which has nothing to flush.
        if self.chain_flushed.load(Ordering::Relaxed) {
            return self.take_block_or_closed(&mut buf, samples);
        }

        loop {
            // Fast path: a full block is already buffered.
            if buf.len() >= samples {
                return Ok(Some(self.take_block(&mut buf, samples)));
            }

            let wait = match deadline {
                Some(dl) => {
                    let now = Instant::now();
                    if now >= dl {
                        // Deadline reached. Absorb any last-moment arrivals,
                        // then deliver a full block if one is now ready, else
                        // `None` (the partial stays buffered for the next call).
                        self.drain_available(&mut buf)?;
                        if buf.len() >= samples {
                            return Ok(Some(self.take_block(&mut buf, samples)));
                        }
                        return Ok(None);
                    }
                    std::cmp::min(dl - now, POLL_INTERVAL)
                }
                None => POLL_INTERVAL,
            };

            match self.receiver.recv_timeout(wait) {
                Ok(chunk) => {
                    self.ingest(&mut buf, chunk)?;
                    // Loop: re-check whether a full block is ready.
                }
                Err(RecvTimeoutError::Timeout) => {
                    // Re-check whether `stop()` or a driver error fired while we
                    // waited (a driver-error close flips `is_open` without
                    // disconnecting the channel).
                    if !self.is_open() {
                        self.drain_available(&mut buf)?;
                        self.flush_chain(&mut buf)?;
                        return self.take_block_or_closed(&mut buf, samples);
                    }
                    // Stream still alive; loop with whatever deadline remains.
                }
                Err(RecvTimeoutError::Disconnected) => {
                    // Channel closed and drained. Drain the chain's end-of-stream
                    // tail (once) into the buffer, then deliver any remaining full
                    // blocks and the final short tail before reporting closed.
                    self.flush_chain(&mut buf)?;
                    return self.take_block_or_closed(&mut buf, samples);
                }
            }
        }
    }

    /// Refuse a block size that is not a whole number of frames.
    ///
    /// The re-block buffer holds interleaved samples, so a size that is not a
    /// multiple of the delivered channel count would cut a frame at the chunk
    /// boundary and leave every following chunk's channels rotated by the
    /// remainder. The chunk still reports the right count and the right
    /// length, so the rotation is invisible to a consumer; it is refused here
    /// rather than delivered. Checked before anything is dequeued, so a
    /// refused call consumes nothing and leaves the stream readable at a
    /// correct size.
    ///
    /// Cannot fire on a mono stream: every size is a multiple of 1.
    fn check_frame_alignment(&self, samples: usize) -> Result<(), DecibriError> {
        if self.channels > 1 && !samples.is_multiple_of(self.channels as usize) {
            return Err(DecibriError::BlockSizeNotFrameAligned {
                samples,
                channels: self.channels,
            });
        }
        Ok(())
    }

    /// Drain exactly `samples` interleaved samples off the front of the re-block
    /// buffer into a fresh [`AudioChunk`], stamping it with this stream's sample
    /// rate and channel count. The caller guarantees `buf.len() >= samples`.
    /// Sample values and order are preserved; only the chunk boundary changes.
    fn take_block(&self, buf: &mut VecDeque<f32>, samples: usize) -> AudioChunk {
        AudioChunk {
            data: buf.drain(..samples).collect(),
            sample_rate: self.sample_rate,
            channels: self.channels,
        }
    }

    /// Close-path delivery. Drains a chunk off a closing/closed stream: a full
    /// `samples`-block if one remains, then any final tail (1..`samples`
    /// samples) as one short chunk, and finally `Err(MicrophoneStreamClosed)`
    /// once the buffer is empty. No captured sample is dropped at close.
    fn take_block_or_closed(
        &self,
        buf: &mut VecDeque<f32>,
        samples: usize,
    ) -> Result<Option<AudioChunk>, DecibriError> {
        if buf.len() >= samples {
            Ok(Some(self.take_block(buf, samples)))
        } else if buf.is_empty() {
            Err(DecibriError::MicrophoneStreamClosed)
        } else {
            // Final partial block: deliver the remaining 1..`samples` samples.
            let remaining = buf.len();
            Ok(Some(self.take_block(buf, remaining)))
        }
    }

    /// Move every immediately-available native buffer from the channel into the
    /// re-block buffer without blocking. Stops on an empty or disconnected
    /// channel; the caller inspects `buf.len()` and the close state afterwards.
    fn drain_available(&self, buf: &mut VecDeque<f32>) -> Result<(), DecibriError> {
        while let Ok(chunk) = self.receiver.try_recv() {
            self.ingest(buf, chunk)?;
        }
        Ok(())
    }

    /// Feed one native capture block into the re-block buffer, running the
    /// capture stage chain first when one is present.
    ///
    /// `None` chain: the block's samples are appended directly, byte-identical to
    /// the direct no-chain path, with no lock, allocation, or copy beyond the existing
    /// reblock. `Some` chain: the chain runs behind its `Mutex` and its
    /// conditioned output is appended instead.
    ///
    /// When the chain has a transform, the post-normalize, pre-transform tap is
    /// captured into the VAD side channel per block, so
    /// [`vad_input`](Self::vad_input) can hand a detector the signal before the
    /// enhancement step. The tap holds real-time samples, so a length-changing
    /// transform (denoise) leaves it leading the delivered enhanced output; see
    /// the [`vad_tap`](Self::vad_tap) field and `vad_input` docs.
    ///
    /// Rejects the block with [`DecibriError::MicrophoneStreamClosed`] once the
    /// chain has been flushed. The flushed chain holds no state a further block
    /// can continue from, so running one through it yields samples that do not
    /// follow the delivered stream. The callers short-circuit before reaching
    /// here, so this ends the stream on the already-documented closed signal
    /// rather than delivering that output.
    ///
    /// A debug assertion fires on that rejection: the callers make it
    /// unreachable, so reaching it means the read-path short-circuit no longer
    /// holds. Debug builds stop on it; release builds return the error.
    fn ingest(&self, buf: &mut VecDeque<f32>, chunk: AudioChunk) -> Result<(), DecibriError> {
        let flushed = self.chain_flushed.load(Ordering::Relaxed);
        debug_assert!(
            !flushed,
            "ingest into a flushed chain: the read-path short-circuit failed"
        );
        if flushed {
            return Err(DecibriError::MicrophoneStreamClosed);
        }
        match &self.capture_stage {
            None => {
                buf.extend(chunk.data);
                Ok(())
            }
            Some(stage) => {
                let mut stage = stage.lock().unwrap_or_else(PoisonError::into_inner);
                let out = stage.run(&chunk.data)?;
                buf.extend(out.iter().copied());
                // Copy the pre-transform tap out, release the stage lock, then
                // append it to the VAD buffer so the two locks are never nested.
                let tap = stage.has_transform().then(|| stage.tap().to_vec());
                drop(stage);
                if let Some(tap) = tap {
                    self.push_vad_tap(tap);
                }
                Ok(())
            }
        }
    }

    /// Append post-normalize samples to the VAD tap, dropping the oldest beyond
    /// the [`VAD_TAP_BOUND_SECS`] memory bound. A no-op when the tap is inactive
    /// (no transform). The bound only trips when a transform is enabled but the
    /// tap is not being drained, so it never perturbs an actively draining VAD.
    fn push_vad_tap(&self, samples: Vec<f32>) {
        if let Some(tap) = &self.vad_tap {
            let mut tap = tap.lock().unwrap_or_else(PoisonError::into_inner);
            tap.extend(samples);
            if tap.len() > self.vad_tap_cap {
                let excess = tap.len() - self.vad_tap_cap;
                tap.drain(..excess);
            }
        }
    }

    /// Drain the capture chain's end-of-stream tail into the re-block buffer,
    /// exactly once, when the stream closes.
    ///
    /// The chain's stages carry conditioning state between blocks (the resampler
    /// holds its anti-alias filter's group-delay tail). At close this drains that
    /// held tail through the chain and appends it to `buf`, so the existing
    /// reblock delivers it as part of the final chunk(s) rather than dropping it.
    /// Runs at most once, guarded by `chain_flushed`; a stream with no chain
    /// (`None`) drains nothing and the direct path is unchanged. The caller holds
    /// the `reblock_buffer` lock, which orders this drain against every other
    /// buffer access and makes the guard's check-and-set effectively atomic.
    ///
    /// Called after the last native block has been drained through
    /// [`ingest`](Self::ingest) and before the close-time delivery, so the tail
    /// is appended after all processed output and reblocked normally.
    fn flush_chain(&self, buf: &mut VecDeque<f32>) -> Result<(), DecibriError> {
        if let Some(stage) = &self.capture_stage {
            if !self.chain_flushed.swap(true, Ordering::Relaxed) {
                let mut stage = stage.lock().unwrap_or_else(PoisonError::into_inner);
                let mut tail = Vec::new();
                stage.flush(&mut tail)?;
                buf.extend(tail);
                // Capture the post-normalize flush tail into the VAD tap too, so
                // the tap does not desync from the delivered output at close: the
                // resampler's group-delay tail is part of the post-normalize
                // signal the detector should read.
                let tap = stage.has_transform().then(|| stage.tap().to_vec());
                drop(stage);
                if let Some(tap) = tap {
                    self.push_vad_tap(tap);
                }
            }
        }
        Ok(())
    }

    /// Check whether the stream is still actively capturing.
    pub fn is_open(&self) -> bool {
        self.running.load(Ordering::Relaxed)
    }

    /// The sample rate (Hz) this stream was opened with.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// The channel count this stream delivers, the count every delivered
    /// chunk's samples are interleaved at. The device itself is opened at its
    /// own native count; the capture chain collapses to this one.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// The pre-transform (post-normalize) samples for the VAD feed, drained as
    /// [`next_chunk`](Self::next_chunk) delivers blocks.
    ///
    /// Binding-internal plumbing: a binding that runs voice-activity detection
    /// calls this right after a `next_chunk` / `try_next_chunk` delivery, passing
    /// the delivered chunk's length, and feeds the returned samples to the
    /// detector instead of the delivered chunk. This makes the detector read the
    /// signal BEFORE the opt-in enhancement step, so enabling enhancement does not
    /// change detection. Not part of the stable FFI-consumer surface.
    ///
    /// # Returns
    /// - `Some(v)`: the chain has an enhancement step, so the delivered output is
    ///   the post-enhancement signal; `v` holds up to `samples` post-normalize
    ///   samples from the front of the tap, the pre-enhancement signal a detector
    ///   should read. With only a length-preserving transform (DC removal) those
    ///   are the exact pre-transform twin of the just-delivered block. With a
    ///   length-changing, latency-introducing transform (denoise), the tap is the
    ///   real-time pre-enhancement signal and LEADS the delivered enhanced block by
    ///   the chain's latency, so `v` is a bounded amount ahead rather than sample-
    ///   aligned. That lead is intended (the detector reads audio sooner, never
    ///   later) and invisible to consumers, which keep only a rolling probability
    ///   and never pair a score to specific returned samples.
    /// - `None`: the chain has no enhancement step, so the delivered chunk already
    ///   is the post-normalize signal; feed the detector the delivered chunk
    ///   exactly as before, with no allocation or copy.
    ///
    /// # Thread safety
    /// Takes only the tap mutex (never the reblock or chain locks), so it composes
    /// with a concurrent [`next_chunk`](Self::next_chunk). The lead stays bounded
    /// as long as the same consumer calls this once per delivered block, as the
    /// binding pump does.
    pub fn vad_input(&self, samples: usize) -> Option<Vec<f32>> {
        let tap = self.vad_tap.as_ref()?;
        let mut tap = tap.lock().unwrap_or_else(PoisonError::into_inner);
        let take = samples.min(tap.len());
        Some(tap.drain(..take).collect())
    }

    /// The detector feed for one delivered chunk: the mono samples a
    /// voice-activity detector reads.
    ///
    /// Binding-internal plumbing: a binding that runs voice-activity detection
    /// calls this right after a `next_chunk` / `try_next_chunk` delivery,
    /// passing the delivered chunk, and feeds the returned samples (or, on
    /// `None`, the delivered chunk itself) to the detector. Not part of the
    /// stable FFI-consumer surface.
    ///
    /// The feed is the pre-transform signal from
    /// [`vad_input`](Self::vad_input) when the tap is active, and the
    /// delivered chunk otherwise. Either signal is collapsed to mono when it
    /// carries more than one channel, as the configured
    /// [`MicrophoneConfig::detector_source`] directs: each interleaved frame
    /// becomes one sample, the average of its channels
    /// ([`DetectorSource::Average`], the default) or the sample of one named
    /// delivered channel ([`DetectorSource::Channel`]). The tap is collapsed
    /// at the chain's own post-normalize channel count, the count the tapped
    /// samples are interleaved at; the delivered chunk is collapsed at the
    /// stream's delivered count.
    ///
    /// # Returns
    /// - `Some(v)`: `v` is the mono detector feed for this delivery.
    /// - `None`: `delivered` already is the feed; read it directly, with no
    ///   allocation or copy.
    ///
    /// # Thread safety
    /// As [`vad_input`](Self::vad_input): takes only the tap mutex, so it
    /// composes with a concurrent [`next_chunk`](Self::next_chunk).
    pub fn detector_feed(&self, delivered: &[f32]) -> Option<Vec<f32>> {
        if let Some(tap) = self.vad_input(delivered.len()) {
            return Some(if self.tap_channels > 1 {
                Self::collapse(&tap, self.tap_channels, self.detector_source)
            } else {
                tap
            });
        }
        if self.channels > 1 {
            return Some(Self::collapse(
                delivered,
                self.channels,
                self.detector_source,
            ));
        }
        None
    }

    /// Collapse one interleaved signal to the mono detector feed, as the
    /// configured source directs: the frame average, or one delivered
    /// channel's samples. The single collapse site both
    /// [`detector_feed`](Self::detector_feed) paths run through, so the two
    /// cannot apply different sources.
    fn collapse(samples: &[f32], channels: u16, source: DetectorSource) -> Vec<f32> {
        match source {
            DetectorSource::Average => crate::sample::downmix_to_mono(samples, channels),
            DetectorSource::Channel(index) => {
                crate::sample::select_channel(samples, channels, index)
            }
        }
    }

    /// Take the last device/driver error reported while streaming, if any.
    ///
    /// When the cpal error callback fires (device unplug, driver failure) it
    /// records a typed [`DecibriError::DeviceFailed`] and closes the stream.
    /// Reading it here drains it (returns `None` afterwards), letting a
    /// consumer that observes a closed stream tell a driver failure apart from
    /// an explicit [`stop`](Self::stop).
    ///
    /// The slot holds one error and taking it clears it, so the first observer
    /// wins. Keep to one draining call site per consumer: a second site that
    /// takes the error for a different purpose swallows the one the first was
    /// going to report. A non-consuming read needs its own accessor.
    pub fn take_last_error(&self) -> Option<DecibriError> {
        self.last_error.lock().ok().and_then(|mut slot| slot.take())
    }

    /// Total number of capture buffers dropped because the channel was full
    /// (a consumer that could not keep up). Stays 0 while the consumer keeps
    /// pace; a rising count means audio is being dropped to bound memory.
    pub fn overrun_count(&self) -> u64 {
        self.overruns.load(Ordering::Relaxed)
    }

    /// Queue far-end reference audio for the echo canceller.
    ///
    /// `reference` is `f32` at
    /// [`MicrophoneConfig::aec_reference_sample_rate`] (the capture
    /// [`sample_rate`](MicrophoneConfig::sample_rate) when that is unset), in
    /// played order, frame-interleaved at
    /// [`MicrophoneConfig::aec_reference_channels`] (mono at the default of
    /// 1). With a declared count above 1, each frame is collapsed to one mono
    /// sample (the channel average) on this call, before the samples enter
    /// the queue; the queue itself always carries mono. A trailing partial
    /// frame is dropped.
    ///
    /// The declared count must match this buffer's actual interleaving. A
    /// mismatch is not detected and raises no error, exactly as a reference
    /// at an undeclared rate is not: the frames are misread, nothing is
    /// cancelled, and the signature is `delay_samples` staying `None` in
    /// [`aec_metrics`](Self::aec_metrics) while the canceller reports no
    /// fault.
    ///
    /// When it is pushed does not have to match when it plays. decibri hands the
    /// canceller the queued reference at the rate the capture consumes it, so a
    /// push made before the first chunk is read, or a whole utterance handed over
    /// in one call, is read out over the capture it echoes into rather than all
    /// at once. Every sample pushed reaches the canceller in played order.
    ///
    /// Silence between what is played need not be pushed. A caller that stops
    /// pushing has said that nothing is playing, and decibri supplies the
    /// silence that means, counted by
    /// [`aec_reference_silence`](Self::aec_reference_silence). Pushing the
    /// silence explicitly is equivalent, not better.
    ///
    /// Never blocks on the canceller and never fails, so it may be called from a
    /// renderer callback or a socket handler. The queue is bounded (see
    /// `AEC_REFERENCE_BOUND_SECS`), and because it is read at the capture's rate
    /// the bound is how far ahead of its own capture a caller may run; samples
    /// that do not fit are discarded from the newest end and counted by
    /// [`aec_reference_dropped`](Self::aec_reference_dropped), leaving what
    /// remains a contiguous run in played order and the time the discarded
    /// samples occupied represented as silence.
    ///
    /// A no-op on a stream with echo cancellation off.
    ///
    /// # Thread safety
    /// May be called from any thread, and from a different thread than the one
    /// reading chunks. It does not take the capture chain's lock.
    #[cfg(feature = "aec")]
    pub fn push_aec_reference(&self, reference: &[f32]) {
        if let Some(queue) = &self.aec_reference {
            // The collapse precedes the push, so the queue stays sized and
            // filled in mono samples whatever count the caller declares. At
            // the default count of 1 the slice goes in untouched, on the same
            // path as before the count existed.
            if self.aec_reference_channels > 1 {
                queue.push(&crate::sample::downmix_to_mono(
                    reference,
                    self.aec_reference_channels,
                ));
            } else {
                queue.push(reference);
            }
        }
    }

    /// Total number of far-end reference samples discarded because the queue was
    /// full, at the declared reference rate. Stays 0 while the reference is
    /// pushed as it plays; a rising count means the caller has run more than
    /// `AEC_REFERENCE_BOUND_SECS` ahead of its own capture, so that much of the
    /// played signal never reached the canceller and the echo of it cannot be
    /// removed. 0 on a stream with echo cancellation off.
    ///
    /// The span the discarded samples occupied is still represented, as silence,
    /// so the canceller's alignment survives a discard rather than being moved
    /// by it: what a rising count costs is the cancellation of that span, not of
    /// the rest of the stream.
    ///
    /// Distinct from [`AecMetrics::reference_dropped`](decibri_aec::AecMetrics::reference_dropped),
    /// which counts what the canceller's own far-end history overwrote after this
    /// queue handed it over.
    #[cfg(feature = "aec")]
    pub fn aec_reference_dropped(&self) -> u64 {
        self.aec_reference
            .as_ref()
            .map_or(0, |queue| queue.dropped())
    }

    /// Total number of far-end reference samples decibri supplied as silence
    /// because the caller had pushed none for them, at the capture
    /// [`sample_rate`](Self::sample_rate). 0 on a stream with echo cancellation
    /// off.
    ///
    /// An accounting figure, not a fault. The canceller reads one far-end sample
    /// for every near-end sample it cancels, and while nothing is playing the
    /// far end is silence, so a stream with occasional playback spends most of
    /// its length here. What it makes visible is the caller: a count that tracks
    /// the whole length of the stream means nothing was ever pushed, and a count
    /// that climbs while the caller believes it is playing means the reference
    /// is not reaching this stream.
    ///
    /// Read with [`AecMetrics::reference_starved`](decibri_aec::AecMetrics::reference_starved),
    /// which counts near-end samples the canceller could find no far-end sample
    /// for at all. That one stays at 0 while this one carries the shortfall.
    #[cfg(feature = "aec")]
    pub fn aec_reference_silence(&self) -> u64 {
        self.aec_reference
            .as_ref()
            .map_or(0, |queue| queue.silence())
    }

    /// The echo canceller's transport and cancellation metrics, or `None` on a
    /// stream with echo cancellation off.
    ///
    /// On a stream delivering more than one channel this reports the FIRST
    /// delivered channel's canceller; read
    /// [`aec_metrics_per_channel`](Self::aec_metrics_per_channel) for every
    /// channel's report. On a single-channel stream the two agree, this one
    /// holding the single entry.
    ///
    /// `acquisition_parked` climbing while `delay_samples` stays `None` and
    /// `erle_db` stays at zero is the signature of a canceller with no usable
    /// reference: none was pushed, it is not at the declared rate, or it is not
    /// the signal that produced the echo. `reference_reanchors` climbing is the
    /// canceller recovering its alignment after a break in either stream.
    ///
    /// `reference_starved` reports near-end samples the canceller could find no
    /// far-end sample for. decibri keeps the far-end stream level with the
    /// capture, so this stays at 0 on a stream whose caller simply stops
    /// pushing; the shortfall is reported by
    /// [`aec_reference_silence`](Self::aec_reference_silence) instead. A
    /// non-zero count here means the caller ran further ahead of the capture
    /// than the canceller's far-end history reaches.
    ///
    /// # Thread safety
    /// Takes the capture chain's lock, which the chain holds for the duration of
    /// each block, so a read serializes against block processing.
    #[cfg(feature = "aec")]
    pub fn aec_metrics(&self) -> Option<decibri_aec::AecMetrics> {
        self.aec_metrics_per_channel()?.into_iter().next()
    }

    /// Every delivered channel's canceller metrics, in delivered order, or
    /// `None` on a stream with echo cancellation off. One canceller engine
    /// runs per delivered channel, each fed the same pushed reference and each
    /// finding its own channel's echo delay, so the entries differ where the
    /// channels' acoustic paths differ. The vector is never empty.
    ///
    /// `delay_samples` is each engine's alignment offset from the reference
    /// frontier as the feeding established it, not a measurement of the room's
    /// echo path length.
    ///
    /// `erle_db` is not a quality ranking across channels: it rises with echo
    /// distance, because a weaker echo is easier to reduce in ratio terms, so
    /// a far microphone routinely reports a higher figure than a near one
    /// while removing less echo in absolute terms. Compare a channel against
    /// its own history, not against its neighbours.
    ///
    /// # Thread safety
    /// Takes the capture chain's lock, which the chain holds for the duration of
    /// each block, so a read serializes against block processing.
    #[cfg(feature = "aec")]
    pub fn aec_metrics_per_channel(&self) -> Option<Vec<decibri_aec::AecMetrics>> {
        self.capture_stage
            .as_ref()?
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .aec_metrics()
    }

    /// Stop capturing audio and release the device.
    ///
    /// Flips the running flag, then drops the held stream, so the OS releases
    /// the input device (and the mic-in-use indicator clears) at once rather
    /// than only when this handle is dropped. Dropping the stream also
    /// disconnects the capture channel; chunks already buffered remain readable
    /// via [`try_next_chunk`](Self::try_next_chunk) or
    /// [`next_chunk`](Self::next_chunk) until the buffer is drained, after which
    /// those methods return `Err(MicrophoneStreamClosed)`.
    ///
    /// May be called from any thread; wakes a concurrent
    /// [`next_chunk`](Self::next_chunk) waiter promptly (the channel disconnect
    /// unblocks it). Releasing the device blocks briefly while the audio thread
    /// tears down.
    pub fn stop(&self) {
        self.running.store(false, Ordering::Relaxed);
        // Drop the held stream to release the OS device now, not only on drop.
        // Poison-tolerant and idempotent (see `BackendStream::stop`).
        self._stream.stop();
    }
}

/// An input device you capture audio from.
///
/// Build one from a [`MicrophoneConfig`], then [`start`](Self::start) it to open
/// the device and obtain a [`MicrophoneStream`] of [`AudioChunk`]s. The playback
/// counterpart is [`Speaker`](crate::Speaker).
///
/// ```no_run
/// use decibri::{Microphone, MicrophoneConfig};
///
/// let mic = Microphone::new(MicrophoneConfig::default())?;
/// let stream = mic.start()?;
/// # Ok::<(), decibri::DecibriError>(())
/// ```
#[cfg(feature = "capture")]
pub struct Microphone {
    config: MicrophoneConfig,
    device: BackendDevice,
}

/// The channel count a stream delivers, given the chain it was built with and
/// the count its device is opened at.
///
/// The chain's own resolved output when there is a chain, and `device_channels`
/// unchanged when there is not: with no stage in the way, what the device
/// produces is what the consumer receives. The count is READ from the chain
/// rather than recomputed from the configuration, so the number a consumer is
/// told and the number the stages actually produce cannot drift apart.
///
/// A named function rather than an expression inline in
/// [`Microphone::start`], because that is what makes the derivation assertable
/// without a device to open. `delivered_channels_are_derived_from_the_chain`
/// holds it: a count fixed here as a constant agrees with the derivation for
/// every configuration capture accepts today, so nothing else in the suite
/// would notice the difference.
#[cfg(feature = "capture")]
fn delivered_channels(capture_stage: Option<&CaptureStage>, device_channels: u16) -> u16 {
    capture_stage.map_or(device_channels, CaptureStage::output_channels)
}

/// Validate a channel map against the delivered count the configuration names
/// and the channel count the resolved device reports.
///
/// The map carries one entry per delivered channel, so its length must equal
/// `target_channels`, and every entry names a device channel, so each must be
/// below `device_channels`. The device's report is the only ceiling: no fixed
/// maximum appears here or anywhere else on the path, and a device reporting
/// any count accepts an index right up to it. Duplicate entries are permitted
/// (each delivered channel is an independent copy of its source) and order is
/// meaningful.
///
/// A pure function so the rules are assertable without a device to resolve;
/// [`Microphone::start`] is its one production caller, passing the same device
/// report that sets the capture open count. It does not belong in
/// [`MicrophoneConfig::validate`], which has no device in scope: any ceiling
/// enforced there would be an invented one.
#[cfg(feature = "capture")]
fn validate_channel_map(
    map: &[u16],
    target_channels: u16,
    device_channels: u16,
) -> Result<(), DecibriError> {
    if map.len() != target_channels as usize {
        return Err(DecibriError::ChannelMapLengthMismatch {
            entries: map.len(),
            channels: target_channels,
        });
    }
    for &index in map {
        if index >= device_channels {
            return Err(DecibriError::ChannelMapOutOfRange {
                index,
                available: device_channels,
            });
        }
    }
    Ok(())
}

/// Validate a delivered channel count carrying no channel map against the
/// channel count the resolved device reports.
///
/// Without a map, decibri derives the delivered channels itself, and only two
/// derivations have a single meaning: collapsing every device channel to one
/// (the documented average) and carrying every device channel through in
/// device order. A count above the device's own is not derivable at all, since
/// decibri delivers device channels and cannot manufacture one that does not
/// exist. A count above one and below the device's own is derivable several
/// ways that no reading of the request chooses between, and every one of them
/// would deliver plausible audio from channels the caller did not name, which
/// nothing in the delivered chunk would reveal; `channel_map` names them
/// instead.
///
/// A map lifts both rules, and deliberately: its entries are checked against
/// the device individually by [`validate_channel_map`], so a map may repeat a
/// channel and may therefore be longer than the device's own count.
///
/// A pure function so the rules are assertable without a device to resolve;
/// [`Microphone::start`] is its one production caller, passing the same device
/// report that sets the capture open count. It does not belong in
/// [`MicrophoneConfig::validate`], which has no device in scope.
#[cfg(feature = "capture")]
fn validate_unmapped_channels(
    target_channels: u16,
    device_channels: u16,
) -> Result<(), DecibriError> {
    if target_channels > device_channels {
        return Err(DecibriError::MicrophoneChannelsUnsupported {
            requested: target_channels,
            available: device_channels,
        });
    }
    if target_channels > 1 && target_channels < device_channels {
        return Err(DecibriError::ChannelSelectionAmbiguous {
            requested: target_channels,
            available: device_channels,
        });
    }
    Ok(())
}

#[cfg(feature = "capture")]
impl Microphone {
    /// Create a microphone: validates the [`MicrophoneConfig`] and resolves the
    /// selected input device. Does not open the stream; call [`start`](Self::start)
    /// for that.
    pub fn new(config: MicrophoneConfig) -> Result<Self, DecibriError> {
        config.validate()?;
        let device = CpalBackend.resolve_input_device(&config.device)?;
        Ok(Self { config, device })
    }

    /// List the available input devices.
    pub fn devices() -> Result<Vec<crate::device::MicrophoneInfo>, DecibriError> {
        crate::device::input_devices()
    }

    /// Start capturing audio. Returns a stream handle with a receiver for audio chunks.
    pub fn start(&self) -> Result<MicrophoneStream, DecibriError> {
        let (sender, receiver): (Sender<AudioChunk>, Receiver<AudioChunk>) =
            crossbeam_channel::bounded(CAPTURE_CHANNEL_CAPACITY);

        let running = Arc::new(AtomicBool::new(true));
        let running_clone = running.clone();

        // The requested rate and channel count are the TARGET the consumer
        // receives. The device is opened at its native rate and native channel
        // count, both settled here from its default supported format, and the
        // capture chain resamples and collapses native -> target so delivery
        // is at exactly the requested format.
        let target_rate = self.config.sample_rate;
        let native_rate = CpalBackend.native_input_rate(&self.device)?;
        let native_channels = CpalBackend.native_input_channels(&self.device)?;
        let frames_per_buffer = self.config.frames_per_buffer;

        // Build the normalize chain for this device. The target is the
        // configured delivered count at the requested rate:
        // `build_capture_stage` adds a channel
        // stage for a multichannel device (the average, or the gather the
        // channel map selects) and a resample when the native rate differs,
        // and returns `None` only for a mono device already at the target rate
        // with no map. The stream reports the OUTPUT channel count (the count
        // the exact-size `samples` math is counted in) and the target rate.
        // `validate()` bounds the accepted set of `channels` below, so the
        // value reaching the chain here is non-zero; its upper bound is the
        // device's, applied just below.
        let target_channels: u16 = self.config.channels;

        // The channel map names device channels, so it is checked here, after
        // device resolution, against the same report that sets the open count.
        // Never checked in `validate()`, which has no device in scope.
        if let Some(map) = &self.config.channel_map {
            validate_channel_map(map, self.config.channels, native_channels)?;
        } else {
            validate_unmapped_channels(target_channels, native_channels)?;
        }
        // Denoise is enabled only when a model AND its path are both set; with a
        // model but no path (or vice versa) the chain leaves denoise off. The
        // path is borrowed for construction only (the stage loads the model and
        // does not retain the path).
        let denoise = self
            .config
            .denoise
            .zip(self.config.denoise_model_path.as_deref())
            .map(|(model, path)| (model, path, self.config.ort_library_path.as_deref()));
        // The far-end reference queue is created here rather than inside the
        // chain, because both the chain's stage and the stream's push method hold
        // it: the stage drains it under the chain's lock, the push method must not
        // wait on that lock. Sized in seconds at the rate the caller declares its
        // reference is at, which is the rate the queued samples carry.
        #[cfg(feature = "aec")]
        let (aec, aec_reference) = match self.config.aec {
            Some(model) => {
                let reference_rate = self.config.aec_reference_sample_rate.unwrap_or(target_rate);
                let reference = Arc::new(AecReferenceRing::new(
                    reference_rate as usize * AEC_REFERENCE_BOUND_SECS,
                ));
                let settings = AecSettings {
                    model,
                    tail_ms: self.config.aec_tail_ms,
                    suppression: self.config.aec_suppression,
                    reference_rate,
                    reference: Arc::clone(&reference),
                };
                (Some(settings), Some(reference))
            }
            None => (None, None),
        };
        let capture_stage = build_capture_stage(
            native_channels,
            target_channels,
            native_rate,
            target_rate,
            Transforms {
                channel_map: self.config.channel_map.as_deref(),
                dc_removal: self.config.dc_removal,
                denoise,
                highpass: self.config.highpass,
                agc: self.config.agc,
                limiter: self.config.limiter,
                #[cfg(feature = "aec")]
                aec,
            },
        )?;
        let output_channels = delivered_channels(capture_stage.as_ref(), native_channels);

        let err_running = running.clone();
        let last_error = Arc::new(Mutex::new(None));
        let err_last_error = last_error.clone();
        let overruns = Arc::new(AtomicU64::new(0));
        let overruns_cb = overruns.clone();

        // Realtime data callback: capture, copy, non-blocking send. Identical
        // work as before; the seam wraps only stream construction.
        let on_data: InputDataCallback = Box::new(move |data: &[f32]| {
            if !running_clone.load(Ordering::Relaxed) {
                return;
            }
            // The native chunk carries the device's capture format (its native
            // rate and channel count); the consumer-side chain normalizes it.
            let chunk = AudioChunk {
                data: data.to_vec(),
                sample_rate: native_rate,
                channels: native_channels,
            };
            // Non-blocking send: the realtime audio thread must never block. On
            // a full channel (a stalled consumer) drop this chunk and count it
            // rather than growing memory without bound; a disconnected receiver
            // also just discards.
            match sender.try_send(chunk) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    overruns_cb.fetch_add(1, Ordering::Relaxed);
                }
                Err(TrySendError::Disconnected(_)) => {}
            }
        });

        // Error callback: record the typed cause and mark the stream closed so a
        // consumer that sees the stream close can distinguish a driver failure
        // from stop(). The backend builds the typed `DeviceFailed`.
        let on_error: StreamErrorCallback = Box::new(move |err: DecibriError| {
            eprintln!("{err}");
            if let Ok(mut slot) = err_last_error.lock() {
                *slot = Some(err);
            }
            err_running.store(false, Ordering::Relaxed);
        });

        // Open the device at its native rate and native channel count; the
        // capture chain resamples and collapses to the target. A mono device
        // already at the target rate has neither stage.
        let params = StreamParams {
            channels: native_channels,
            sample_rate: native_rate,
            frames_per_buffer: Some(frames_per_buffer),
        };
        let _stream = CpalBackend.open_input_stream(&self.device, &params, on_data, on_error)?;

        // The VAD tap is active only when the chain has a transform segment, so
        // the delivered (post-transform) output differs from the pre-transform
        // signal a detector should read. With no transform the delivered output
        // already is that signal, so no tap is allocated.
        let vad_tap = match &capture_stage {
            Some(stage) if stage.has_transform() => Some(Mutex::new(VecDeque::new())),
            _ => None,
        };
        // The count the tap signal is interleaved at, read from the chain (see
        // the field docs). With no chain the tap is never active; the device
        // count stands in.
        let tap_channels = capture_stage
            .as_ref()
            .map_or(native_channels, CaptureStage::tap_channels);
        let vad_tap_cap = target_rate as usize * VAD_TAP_BOUND_SECS;
        // The tap memory bound must sit far above the chain's conditioning
        // latency, so an actively draining detector never reaches it even when a
        // length-changing stage leaves the tap leading the delivered output.
        // Checked once here, not per block, since the latency is fixed when the
        // chain is built.
        debug_assert!(
            capture_stage
                .as_ref()
                .map_or(0, CaptureStage::transform_latency)
                < vad_tap_cap,
            "the VAD tap memory bound must exceed the chain's transform latency"
        );

        Ok(MicrophoneStream {
            _stream,
            receiver,
            running,
            // Consumers receive the target rate; `take_block` stamps it on every
            // delivered chunk.
            sample_rate: target_rate,
            channels: output_channels,
            last_error,
            overruns,
            reblock_buffer: Mutex::new(VecDeque::new()),
            capture_stage: capture_stage.map(Mutex::new),
            chain_flushed: AtomicBool::new(false),
            vad_tap,
            tap_channels,
            detector_source: self.config.detector_source,
            vad_tap_cap,
            #[cfg(feature = "aec")]
            aec_reference,
            #[cfg(feature = "aec")]
            aec_reference_channels: self.config.aec_reference_channels,
        })
    }
}

#[cfg(all(test, feature = "capture"))]
mod tests {
    use super::*;
    use std::thread;

    /// Construct a synthetic `MicrophoneStream` with no underlying cpal device,
    /// the given stage chain, and the given output channel count. Returns the
    /// stream plus test-side handles to inject native chunks and flip running.
    /// As [`test_stream_with`], with the far-end reference queue the stream's push
    /// method feeds wired to the one the chain's canceller drains.
    #[cfg(feature = "aec")]
    fn test_stream_with_reference(
        capture_stage: Option<CaptureStage>,
        channels: u16,
        aec_reference: Arc<AecReferenceRing>,
    ) -> (MicrophoneStream, Sender<AudioChunk>, Arc<AtomicBool>) {
        let (stream, sender, running) = test_stream_with(capture_stage, channels);
        let stream = MicrophoneStream {
            aec_reference: Some(aec_reference),
            ..stream
        };
        (stream, sender, running)
    }

    fn test_stream_with(
        capture_stage: Option<CaptureStage>,
        channels: u16,
    ) -> (MicrophoneStream, Sender<AudioChunk>, Arc<AtomicBool>) {
        let (sender, receiver) = crossbeam_channel::unbounded::<AudioChunk>();
        let running = Arc::new(AtomicBool::new(true));
        let vad_tap = match &capture_stage {
            Some(stage) if stage.has_transform() => Some(Mutex::new(VecDeque::new())),
            _ => None,
        };
        // As `Microphone::start`: the tap count comes from the chain, with the
        // given count standing in when there is no chain. The `channels`
        // argument stays the stream's own (delivered) count, so a test can set
        // the two apart on purpose.
        let tap_channels = capture_stage
            .as_ref()
            .map_or(channels, CaptureStage::tap_channels);
        let stream = MicrophoneStream {
            _stream: BackendStream::empty(),
            receiver,
            running: running.clone(),
            sample_rate: 16000,
            channels,
            last_error: Arc::new(Mutex::new(None)),
            overruns: Arc::new(AtomicU64::new(0)),
            reblock_buffer: Mutex::new(VecDeque::new()),
            capture_stage: capture_stage.map(Mutex::new),
            vad_tap,
            tap_channels,
            detector_source: DetectorSource::Average,
            vad_tap_cap: 16000 * VAD_TAP_BOUND_SECS,
            chain_flushed: AtomicBool::new(false),
            #[cfg(feature = "aec")]
            aec_reference: None,
            #[cfg(feature = "aec")]
            aec_reference_channels: 1,
        };
        (stream, sender, running)
    }

    /// As [`test_stream_with`], with the detector source set apart from the
    /// default average.
    fn test_stream_with_source(
        capture_stage: Option<CaptureStage>,
        channels: u16,
        source: DetectorSource,
    ) -> (MicrophoneStream, Sender<AudioChunk>, Arc<AtomicBool>) {
        let (stream, sender, running) = test_stream_with(capture_stage, channels);
        let stream = MicrophoneStream {
            detector_source: source,
            ..stream
        };
        (stream, sender, running)
    }

    /// Mono stream with no stage chain (the `None`, no-chain path).
    fn test_stream() -> (MicrophoneStream, Sender<AudioChunk>, Arc<AtomicBool>) {
        test_stream_with(None, 1)
    }

    fn make_chunk(first_sample: f32) -> AudioChunk {
        AudioChunk {
            data: vec![first_sample],
            sample_rate: 16000,
            channels: 1,
        }
    }

    /// A native chunk carrying an arbitrary run of interleaved samples, for the
    /// re-blocking tests (native buffers are variable-size).
    fn make_native_chunk(data: Vec<f32>) -> AudioChunk {
        AudioChunk {
            data,
            sample_rate: 16000,
            channels: 1,
        }
    }

    #[test]
    fn test_try_next_chunk_returns_none_when_empty() {
        let (stream, _sender, _running) = test_stream();
        let result = stream.try_next_chunk(1).unwrap();
        assert!(
            result.is_none(),
            "try_next_chunk on empty open stream should return Ok(None)"
        );
    }

    #[test]
    fn test_try_next_chunk_returns_chunk_when_available() {
        let (stream, sender, _running) = test_stream();
        sender.send(make_chunk(0.42)).unwrap();

        let result = stream.try_next_chunk(1).unwrap();
        let chunk = result.expect("should have received the injected chunk");
        assert_eq!(chunk.data, vec![0.42]);
    }

    #[test]
    fn test_try_next_chunk_returns_err_when_closed() {
        let (stream, sender, running) = test_stream();
        drop(sender); // simulate cpal stream dropping the sender
        running.store(false, Ordering::Relaxed);

        let err = stream.try_next_chunk(1).unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    #[test]
    fn test_next_chunk_blocks_until_chunk_arrives() {
        let (stream, sender, _running) = test_stream();

        let producer = thread::spawn(move || {
            thread::sleep(Duration::from_millis(50));
            sender.send(make_chunk(0.77)).unwrap();
        });

        let result = stream.next_chunk(1, None).unwrap();
        let chunk = result.expect("should have received the eventually-pushed chunk");
        assert_eq!(chunk.data, vec![0.77]);

        producer.join().unwrap();
    }

    #[test]
    fn test_next_chunk_timeout_returns_none() {
        let (stream, _sender, _running) = test_stream();
        let start = Instant::now();
        let result = stream
            .next_chunk(1, Some(Duration::from_millis(50)))
            .unwrap();
        let elapsed = start.elapsed();

        assert!(
            result.is_none(),
            "next_chunk with timeout and no arrivals should return Ok(None)"
        );
        // Lower bound: at least the requested timeout.
        assert!(
            elapsed >= Duration::from_millis(40),
            "next_chunk returned too early: {elapsed:?}"
        );
    }

    #[test]
    fn test_next_chunk_flushes_buffered_before_closed_err() {
        let (stream, sender, running) = test_stream();
        sender.send(make_chunk(1.0)).unwrap();
        sender.send(make_chunk(2.0)).unwrap();
        drop(sender); // simulate cpal stream drop
        running.store(false, Ordering::Relaxed);

        // First two calls drain the buffer, third reports closed.
        let c1 = stream
            .next_chunk(1, Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(c1.unwrap().data, vec![1.0]);

        let c2 = stream
            .next_chunk(1, Some(Duration::from_millis(100)))
            .unwrap();
        assert_eq!(c2.unwrap().data, vec![2.0]);

        let err = stream
            .next_chunk(1, Some(Duration::from_millis(100)))
            .unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    #[test]
    fn test_next_chunk_returns_closed_within_polling_interval_after_stop() {
        let (stream, _sender, running) = test_stream();

        let r = running.clone();
        let stopper = thread::spawn(move || {
            thread::sleep(Duration::from_millis(30));
            r.store(false, Ordering::Relaxed);
        });

        // next_chunk with no timeout should wake up on the next 20 ms poll
        // after stop() flips the running flag. Give a generous 250 ms ceiling
        // to absorb scheduler jitter on loaded CI runners.
        let start = Instant::now();
        let err = stream.next_chunk(1, None).unwrap_err();
        let elapsed = start.elapsed();

        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
        assert!(
            elapsed < Duration::from_millis(250),
            "next_chunk took too long to detect stop(): {elapsed:?}"
        );

        stopper.join().unwrap();
    }

    /// Re-blocking delivers full blocks of exactly the requested size, in order,
    /// then a final short chunk carrying the tail, with no sample lost,
    /// reordered, or altered. Irregular native buffers carrying the values 0..14
    /// (not a multiple of the block size 4) are re-blocked; the concatenation of
    /// the delivered chunks equals the FULL input stream, tail included.
    #[test]
    fn test_next_chunk_delivers_exact_blocks_then_final_tail() {
        let (stream, sender, running) = test_stream();
        sender
            .send(make_native_chunk(vec![0.0, 1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        sender.send(make_native_chunk(vec![5.0, 6.0])).unwrap();
        sender
            .send(make_native_chunk(vec![
                7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
            ]))
            .unwrap();
        drop(sender); // no more native buffers will arrive
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let mut chunks: Vec<Vec<f32>> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(chunk)) => chunks.push(chunk.data),
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Every block is exactly `samples` long except the final tail (14 % 4 == 2).
        let (last, full) = chunks.split_last().expect("at least one chunk delivered");
        for block in full {
            assert_eq!(
                block.len(),
                samples,
                "non-final blocks are exactly `samples` long"
            );
        }
        assert_eq!(last.len(), 2, "the final chunk carries the 2-sample tail");

        // Sample identity: the full input stream is reconstructed, nothing dropped.
        let collected: Vec<f32> = chunks.into_iter().flatten().collect();
        assert_eq!(
            collected,
            (0..14).map(|n| n as f32).collect::<Vec<f32>>(),
            "re-blocking preserves every sample value and order, including the tail"
        );
    }

    /// A single native buffer larger than the requested block size is split
    /// into successive full blocks across calls, the leftover carried forward;
    /// the final sub-block remainder is delivered as a short chunk at close.
    #[test]
    fn test_next_chunk_splits_large_native_buffer_into_blocks() {
        let (stream, sender, running) = test_stream();
        sender
            .send(make_native_chunk((0..10).map(|n| n as f32).collect()))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 3;
        let b1 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("first block");
        let b2 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("second block");
        let b3 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("third block");
        assert_eq!(b1.data, vec![0.0, 1.0, 2.0]);
        assert_eq!(b2.data, vec![3.0, 4.0, 5.0]);
        assert_eq!(b3.data, vec![6.0, 7.0, 8.0]);

        // 10 % 3 == 1 leftover sample: delivered as a final short chunk, then closed.
        let tail = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("final tail");
        assert_eq!(tail.data, vec![9.0]);
        let err = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    /// `try_next_chunk` returns `Ok(None)` while fewer than a full block are
    /// buffered on an open stream, then `Ok(Some)` once enough native samples
    /// accumulate; the block is exactly `samples` long.
    #[test]
    fn test_try_next_chunk_short_returns_none_then_full_block() {
        let (stream, sender, _running) = test_stream();
        let samples = 4;

        sender.send(make_native_chunk(vec![10.0, 11.0])).unwrap();
        assert!(
            stream.try_next_chunk(samples).unwrap().is_none(),
            "fewer than `samples` on an open stream returns Ok(None)"
        );

        sender.send(make_native_chunk(vec![12.0, 13.0])).unwrap();
        let chunk = stream
            .try_next_chunk(samples)
            .unwrap()
            .expect("a full block is now available");
        assert_eq!(chunk.data, vec![10.0, 11.0, 12.0, 13.0]);

        // The buffer is empty again.
        assert!(stream.try_next_chunk(samples).unwrap().is_none());
    }

    /// The final partial block (fewer than `samples` samples at close) is
    /// delivered as one short chunk, then the stream reports closed.
    #[test]
    fn test_final_partial_tail_delivered_on_close() {
        let (stream, sender, running) = test_stream();
        sender
            .send(make_native_chunk(vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let c1 = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("first full block");
        assert_eq!(c1.data, vec![0.0, 1.0, 2.0, 3.0]);

        // The remaining 2 samples are delivered as a final short chunk.
        let tail = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap()
            .expect("final tail");
        assert_eq!(tail.data, vec![4.0, 5.0]);

        // Now the buffer is empty: the stream reports closed.
        let err = stream
            .next_chunk(samples, Some(Duration::from_millis(100)))
            .unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    /// `try_next_chunk` delivers the final tail as a short chunk when the stream
    /// closes with a partial buffered, then reports closed.
    #[test]
    fn test_try_next_chunk_delivers_tail_on_close() {
        let (stream, sender, running) = test_stream();
        sender.send(make_native_chunk(vec![1.0, 2.0])).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let tail = stream
            .try_next_chunk(samples)
            .unwrap()
            .expect("final tail delivered");
        assert_eq!(tail.data, vec![1.0, 2.0]);

        let err = stream.try_next_chunk(samples).unwrap_err();
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    /// Empty-chain no-op (the `None` path): with no stage chain (a mono device),
    /// `next_chunk` delivers the raw reblocked samples byte-identically to the
    /// direct no-chain path, exact size and final-tail behaviour included.
    #[test]
    fn test_empty_chain_is_byte_identical_noop() {
        let (stream, sender, running) = test_stream(); // mono, capture_stage = None
        assert!(stream.capture_stage.is_none(), "a mono stream has no chain");
        sender
            .send(make_native_chunk(vec![0.0, 1.0, 2.0, 3.0, 4.0]))
            .unwrap();
        sender
            .send(make_native_chunk(vec![5.0, 6.0, 7.0, 8.0]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 4;
        let mut collected: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => collected.extend_from_slice(&c.data),
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        // None path (direct reblock): 9 samples -> two blocks of 4 then a 1-sample
        // tail, every value passed through untransformed and in order.
        assert_eq!(collected, (0..9).map(|n| n as f32).collect::<Vec<f32>>());
    }

    /// Cost no-op (the `None` path): a mono device builds no chain, so the drain
    /// stays on the direct reblock. The `None` arm of `ingest` is a plain
    /// `buf.extend(chunk.data)` with no chain lock, allocation, or copy,
    /// structurally identical to the direct no-chain path.
    #[test]
    fn test_mono_device_builds_no_chain() {
        assert!(
            build_capture_stage(1, 1, 16000, 16000, Transforms::default())
                .unwrap()
                .is_none(),
            "a mono device at the target rate needs no normalize chain"
        );
        let (stream, _sender, _running) = test_stream();
        assert!(
            stream.capture_stage.is_none(),
            "no CaptureStage is allocated for a mono stream"
        );
    }

    /// The channel count a stream reports is DERIVED from its chain, not named
    /// beside it.
    ///
    /// Regression: `delivered_channels` answering with a constant. Capture is
    /// mono in and mono out for every configuration it accepts today, so a
    /// hardcoded `1` matches the derived value at every point the rest of the
    /// suite observes it. Planting one against the code before this test existed
    /// left all 277 tests green, which is the whole reason the test is here.
    ///
    /// Each arm below fails a constant a different way round: a chain that
    /// collapses must report the collapse, a chain that collapses nothing must
    /// report what it was handed, and no chain at all must leave the device's
    /// own count untouched.
    #[test]
    fn delivered_channels_are_derived_from_the_chain() {
        // A chain that collapses reports the collapse, whatever went in.
        for device_channels in [2u16, 6, 1024] {
            let chain =
                build_capture_stage(device_channels, 1, 16_000, 16_000, Transforms::default())
                    .expect("no channel count is rejected by the builder")
                    .expect("a device above the target gets a downmix chain");
            assert_eq!(
                delivered_channels(Some(&chain), device_channels),
                1,
                "a collapsing chain delivers mono"
            );
        }

        // A chain that collapses nothing reports the count it was handed.
        for channels in [2u16, 5, 32] {
            let chain =
                build_capture_stage(channels, channels, 48_000, 16_000, Transforms::default())
                    .expect("no channel count is rejected by the builder")
                    .expect("a rate change builds a resample chain");
            assert_eq!(
                delivered_channels(Some(&chain), channels),
                channels,
                "a chain that collapses nothing delivers what it was handed"
            );
        }

        // With no chain, what the device produces is what the consumer receives.
        for device_channels in [1u16, 2, 7] {
            assert_eq!(
                delivered_channels(None, device_channels),
                device_channels,
                "no chain means nothing changed the count"
            );
        }
    }

    /// Auto-normalize (the `Some([Downmix])` path): a multichannel source is
    /// downmixed to correct mono, exact size holds on the mono output, the
    /// downmix is sample-identical to `sample::downmix_to_mono`, and the final
    /// tail is delivered.
    #[test]
    fn test_downmix_chain_yields_correct_mono() {
        let stage = build_capture_stage(2, 1, 16000, 16000, Transforms::default()).unwrap();
        assert!(stage.is_some(), "a stereo device gets a downmix chain");
        let (stream, sender, running) = test_stream_with(stage, 1); // output is mono

        // Stereo native chunks (interleaved L,R). Frame means:
        //   [0.5,0.3]->0.4 [0.4,0.6]->0.5 ; [0.0,0.2]->0.1 [0.8,0.4]->0.6 [0.1,0.1]->0.1
        sender
            .send(make_native_chunk(vec![0.5, 0.3, 0.4, 0.6]))
            .unwrap();
        sender
            .send(make_native_chunk(vec![0.0, 0.2, 0.8, 0.4, 0.1, 0.1]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 2; // 2 MONO samples per block
        let mut collected: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.channels, 1, "the normalized output is mono");
                    collected.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Mono stream [0.4,0.5,0.1,0.6,0.1] -> blocks [0.4,0.5] [0.1,0.6], tail [0.1].
        let expected = [0.4, 0.5, 0.1, 0.6, 0.1];
        assert_eq!(
            collected.len(),
            expected.len(),
            "nothing dropped beyond the final-tail rule"
        );
        for (got, want) in collected.iter().zip(expected.iter()) {
            assert!((got - want).abs() < 1e-6, "{got} vs {want}");
        }
    }

    /// Resample-in-capture (the `Some([ResampleStage])` path): a mono device
    /// above the target rate has its audio resampled to the target inside
    /// `ingest`, exact-size delivery holds on the resampled output, every
    /// delivered chunk reports the target rate and mono, and the total sample
    /// count tracks the 1:3 rate ratio (48 kHz -> 16 kHz).
    #[test]
    fn test_resample_chain_delivers_exact_blocks_at_target_rate() {
        let stage = build_capture_stage(1, 1, 48_000, 16_000, Transforms::default())
            .unwrap()
            .expect("48k mono -> resample chain");
        // test_stream_with stamps the stream at 16 kHz (the target), mono.
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();
        sender.send(make_native_chunk(input.clone())).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320; // 20 ms at 16 kHz
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.channels, 1, "resampled output is mono");
                    assert_eq!(c.sample_rate, 16_000, "chunks report the target rate");
                    blocks.push(c.data);
                }
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        let (last, full) = blocks.split_last().expect("at least one block delivered");
        for block in full {
            assert_eq!(
                block.len(),
                samples,
                "non-final blocks are exactly `samples` long on the resampled output"
            );
        }
        assert!(
            !last.is_empty() && last.len() <= samples,
            "the final tail carries 1..=`samples` resampled samples"
        );

        let total: usize = blocks.iter().map(|b| b.len()).sum();
        // A 1:3 downsample of the 24000-sample input yields roughly 8000 output
        // samples, plus the resampler's group-delay tail now drained at close, so
        // the total sits just above input/3 and well below half the input.
        assert!(
            total > input.len() / 4 && total < input.len() / 2,
            "resampled total {total} tracks the 1:3 ratio of {} input",
            input.len()
        );
    }

    /// Resample close path, the no-sample-dropped proof: a known signal fed
    /// through the resampling capture chain and read to close delivers the
    /// COMPLETE resampled signal, group-delay tail included. The concatenation of
    /// every delivered chunk (steady full blocks plus the final short tail)
    /// equals a bare resampler fed the whole input then flushed once, bit for
    /// bit: no sample lost, reordered, requantized, or scaled. The flushed tail
    /// arrives as part of the final chunk(s) through the normal reblock (full
    /// blocks then one final partial), not a separate post-close emission.
    /// Bit-equality also proves the flush ran exactly once: a missing flush would
    /// drop the tail and a double flush would append extra samples, either of
    /// which breaks the equality.
    #[test]
    fn test_resample_close_delivers_full_signal_with_flushed_tail() {
        use decibri_resampler::{PolyphaseResampler, Resampler};

        let input: Vec<f32> = (0..24_000).map(|n| (n as f32 * 0.01).sin()).collect();

        // Ground truth: a bare resampler fed the whole input, then one flush.
        let mut reference = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut expected = Vec::new();
        reference.process(&input, &mut expected).unwrap();
        reference.flush(&mut expected);

        // The process-only count (no flush) shows the tail is a real contribution
        // that the resample path dropped before this drain existed.
        let mut process_only = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut process_out = Vec::new();
        process_only.process(&input, &mut process_out).unwrap();

        let stage = build_capture_stage(1, 1, 48_000, 16_000, Transforms::default())
            .unwrap()
            .expect("48k mono -> resample chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);
        sender.send(make_native_chunk(input)).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320; // 20 ms at 16 kHz
        let mut blocks: Vec<Vec<f32>> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.sample_rate, 16_000, "chunks report the target rate");
                    assert_eq!(c.channels, 1, "resampled output is mono");
                    blocks.push(c.data);
                }
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Delivered as full blocks then one final partial: the tail rides the
        // normal reblock, not a separate emission.
        let (last, full) = blocks.split_last().expect("at least one block delivered");
        for block in full {
            assert_eq!(
                block.len(),
                samples,
                "non-final blocks are exactly `samples` long, tail included"
            );
        }
        assert!(
            !last.is_empty() && last.len() <= samples,
            "the final chunk carries 1..=`samples` resampled samples"
        );

        // No sample dropped, none added: the delivered stream equals the full
        // resampled signal (steady output plus the flushed group-delay tail).
        let delivered: Vec<f32> = blocks.into_iter().flatten().collect();
        assert_eq!(
            delivered, expected,
            "the resample close path delivers the complete resampled signal, tail included"
        );
        assert!(
            expected.len() > process_out.len(),
            "the flushed tail adds the samples the process-only path dropped"
        );

        // Idempotent close: further reads stay closed with no resurrected audio.
        for _ in 0..3 {
            let err = stream
                .next_chunk(samples, Some(Duration::from_millis(20)))
                .unwrap_err();
            assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
        }
    }

    /// Enhancement-on, end to end through `ingest` (the `Some([transform])`
    /// path): with `dc_removal` enabled on a mono device already at the target
    /// rate, the chain is transform-only (no downmix, no resample), so a constant
    /// DC offset fed through the stream is delivered with the same sample count
    /// (the DC step preserves length and holds no tail) and a settled mean near
    /// zero (the offset is removed). This proves the transform segment is applied
    /// to delivered audio through the normal capture path.
    #[test]
    fn test_enhancement_on_removes_dc_end_to_end() {
        let enhancement = true;
        let stage = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: enhancement,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("dc_removal builds a transform-only chain for a mono device");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        // A constant 0.5 offset: pure DC, no audio content.
        let n = 16_000;
        let input = vec![0.5_f32; n];
        sender.send(make_native_chunk(input)).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320;
        let mut collected: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    assert_eq!(c.channels, 1, "the enhanced output stays mono");
                    assert_eq!(c.sample_rate, 16_000, "no resample, the rate is unchanged");
                    collected.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout while data was buffered"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        assert_eq!(
            collected.len(),
            n,
            "the DC step preserves the sample count exactly (no resample, no tail)"
        );
        // After the filter settles, the constant offset is gone: the mean of the
        // last quarter is essentially zero.
        let settled = &collected[n - n / 4..];
        let mean = settled.iter().sum::<f32>() / settled.len() as f32;
        assert!(
            mean.abs() < 1e-3,
            "the DC offset is removed end to end (settled mean {mean})"
        );
    }

    /// VAD tap inactive (enhancement off): a chain with no transform (here a
    /// downmix-only chain) leaves the tap unallocated, so `vad_input` is `None`
    /// throughout and a binding feeds VAD the delivered chunk exactly as before.
    #[test]
    fn test_vad_input_none_when_no_transform() {
        let stage = build_capture_stage(2, 1, 16_000, 16_000, Transforms::default())
            .unwrap()
            .expect("stereo -> downmix-only chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);
        assert!(
            stream.vad_input(4).is_none(),
            "no transform: the tap is inactive and vad_input returns None"
        );

        sender
            .send(make_native_chunk(vec![0.5, 0.3, 0.4, 0.6]))
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);
        loop {
            match stream.next_chunk(2, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => assert!(
                    stream.vad_input(c.data.len()).is_none(),
                    "vad_input stays None throughout when no transform is present"
                ),
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
    }

    /// A mono device builds no chain at all, so the tap is inactive and
    /// `vad_input` is `None` (the binding uses the delivered chunk).
    #[test]
    fn test_vad_input_none_for_no_chain() {
        let (stream, _sender, _running) = test_stream();
        assert!(stream.capture_stage.is_none(), "a mono stream has no chain");
        assert!(
            stream.vad_input(4).is_none(),
            "no chain: vad_input returns None"
        );
    }

    /// Enhancement-on tap correctness: with DC removal active, `vad_input` returns
    /// the post-normalize (pre-DC-removal) signal, NOT the post-transform delivered
    /// output. A constant DC offset proves it: the delivered output has the offset
    /// removed, but the VAD feed still carries it, so VAD reads the pre-transform
    /// signal. The tap and the delivered output stay aligned (same sample count).
    #[test]
    fn test_vad_input_returns_pre_transform_signal() {
        let enhancement = true;
        let stage = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: enhancement,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("dc-only chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        let n = 16_000;
        let input = vec![0.5_f32; n];
        sender.send(make_native_chunk(input)).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320;
        let mut delivered: Vec<f32> = Vec::new();
        let mut vad_feed: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    let pre = stream
                        .vad_input(c.data.len())
                        .expect("tap active when a transform is present");
                    vad_feed.extend_from_slice(&pre);
                    delivered.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        assert_eq!(delivered.len(), n, "no resample, length preserved");
        assert_eq!(
            vad_feed.len(),
            n,
            "the VAD feed is aligned with the delivered output"
        );
        // The VAD feed carries the DC offset (pre-transform).
        assert!(
            vad_feed.iter().all(|&s| (s - 0.5).abs() < 1e-6),
            "the VAD feed is the exact post-normalize (pre-DC) signal, offset intact"
        );
        // The delivered output has the offset removed (post-transform).
        let settled = &delivered[n - n / 4..];
        let d_mean = settled.iter().sum::<f32>() / settled.len() as f32;
        assert!(
            d_mean.abs() < 1e-3,
            "the delivered output has the DC offset removed (post-transform)"
        );
    }

    /// A capture stage chain whose normalize segment ends above one channel:
    /// no downmix (the device count equals the target count) and a DC-removal
    /// transform, so the tap is active and its resolved count is 2. Built
    /// through `build_capture_stage` exactly as `Microphone::start` builds it.
    /// The tests that use it fill the tap directly and never run a block
    /// through the chain: the in-place transform wrappers assert mono, so no
    /// multichannel signal can cross the transform segment.
    fn two_channel_tap_chain() -> CaptureStage {
        build_capture_stage(
            2,
            2,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a transform builds a chain")
    }

    /// `detector_feed` collapses a multichannel tap to mono: each interleaved
    /// frame becomes the average of its channels, so the feed halves in length
    /// and a detector reads one channel-averaged signal. Regression: a feed
    /// handed to a detector interleaved, where the score would read
    /// consecutive channels as successive mono samples.
    #[test]
    fn detector_feed_collapses_a_multichannel_tap() {
        let (stream, _sender, _running) = test_stream_with(Some(two_channel_tap_chain()), 2);

        let frames = 160;
        let left: Vec<f32> = (0..frames).map(|i| (i % 8) as f32 * 0.125 - 0.5).collect();
        let interleaved: Vec<f32> = left.iter().flat_map(|&l| [l, l + 0.5]).collect();
        let expected: Vec<f32> = left.iter().map(|&l| l + 0.25).collect();
        stream.push_vad_tap(interleaved.clone());

        // The delivered chunk only sizes the drain here; the tap is what the
        // feed is derived from.
        let feed = stream
            .detector_feed(&interleaved)
            .expect("a transform keeps the tap active");
        assert_eq!(
            feed.len(),
            frames,
            "the feed is mono: one sample per interleaved frame"
        );
        for (i, (got, want)) in feed.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "feed sample {i} is the frame's channel average (got {got}, want {want})"
            );
        }
    }

    /// `detector_feed` is the identity on the mono path: with a mono tap the
    /// feed is the pre-transform signal unchanged in length and value, and
    /// with no chain it returns `None` so the delivered chunk is read directly
    /// with no copy. Regression: a collapse that reshapes or rescales a signal
    /// that is already mono.
    #[test]
    fn detector_feed_is_the_identity_on_the_mono_path() {
        // Mono tap: the feed is the pre-transform (DC-offset-intact) signal.
        let stage = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                dc_removal: true,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("dc-only chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);
        let n = 320;
        sender.send(make_native_chunk(vec![0.5_f32; n])).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);
        let chunk = stream
            .next_chunk(n, Some(Duration::from_millis(100)))
            .expect("the chain delivers")
            .expect("a full block is buffered");
        let feed = stream
            .detector_feed(&chunk.data)
            .expect("a transform keeps the tap active");
        assert_eq!(feed.len(), n, "a mono feed keeps its length");
        assert!(
            feed.iter().all(|&s| (s - 0.5).abs() < 1e-6),
            "a mono feed is the pre-transform signal unchanged"
        );

        // No chain: the delivered chunk already is the feed.
        let (stream, sender, running) = test_stream();
        sender.send(make_chunk(0.25)).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);
        let chunk = stream
            .next_chunk(1, Some(Duration::from_millis(100)))
            .expect("the direct path delivers")
            .expect("a block is buffered");
        assert!(
            stream.detector_feed(&chunk.data).is_none(),
            "no chain: the delivered chunk already is the feed"
        );
    }

    /// `detector_feed` collapses the tap at the CHAIN's post-normalize count,
    /// not the stream's delivered count: a stream whose stored delivered count
    /// is set apart from the chain's tap count on purpose still averages each
    /// tapped frame at the chain's count. Regression: the derivation quietly
    /// switching to the delivered count, which reads a signal tapped mid-chain
    /// at the count of a different point in the chain.
    #[test]
    fn detector_feed_reads_the_chain_tap_count_not_the_delivered_count() {
        // The chain taps at 2 channels; the stream's own count is set to 1.
        let (stream, _sender, _running) = test_stream_with(Some(two_channel_tap_chain()), 1);

        let frames = 160;
        let left: Vec<f32> = (0..frames).map(|i| (i % 4) as f32 * 0.25 - 0.375).collect();
        let interleaved: Vec<f32> = left.iter().flat_map(|&l| [l, l + 0.25]).collect();
        let expected: Vec<f32> = left.iter().map(|&l| l + 0.125).collect();
        stream.push_vad_tap(interleaved.clone());

        let feed = stream
            .detector_feed(&interleaved)
            .expect("a transform keeps the tap active");
        assert_eq!(
            feed.len(),
            frames,
            "the collapse runs at the chain's tap count, not the stream's count"
        );
        for (i, (got, want)) in feed.iter().zip(&expected).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "feed sample {i} is the frame's channel average (got {got}, want {want})"
            );
        }
    }

    /// With no detector source set, the feed is byte-identical to the frame
    /// average on both `detector_feed` paths: the tap collapse and the
    /// delivered-chunk collapse each equal `downmix_to_mono` of their input
    /// exactly. The default-path safety pin: a configuration that sets
    /// nothing reads the same feed as before the option existed. Regression:
    /// the source plumbing disturbing the default collapse.
    #[test]
    fn the_default_detector_source_is_the_frame_average() {
        // Tap path: a 2-channel tap on a default-source stream.
        let (stream, _sender, _running) = test_stream_with(Some(two_channel_tap_chain()), 2);
        let frames = 160;
        let interleaved: Vec<f32> = (0..frames)
            .flat_map(|i| [(i % 8) as f32 * 0.125 - 0.5, (i % 5) as f32 * 0.2 - 0.4])
            .collect();
        stream.push_vad_tap(interleaved.clone());
        let feed = stream
            .detector_feed(&interleaved)
            .expect("a transform keeps the tap active");
        assert_eq!(
            feed,
            crate::sample::downmix_to_mono(&interleaved, 2),
            "the tap-path default is the frame average, byte for byte"
        );

        // Delivered path: no chain, 2 delivered channels, default source.
        let (stream, _sender, _running) = test_stream_with(None, 2);
        let feed = stream
            .detector_feed(&interleaved)
            .expect("a multichannel delivered chunk is collapsed");
        assert_eq!(
            feed,
            crate::sample::downmix_to_mono(&interleaved, 2),
            "the delivered-path default is the frame average, byte for byte"
        );
    }

    /// A `DetectorSource::Channel` feeds the named delivered channel alone,
    /// on both `detector_feed` paths. The channels carry distinguishable
    /// content, silence against a ramp, so a wrong selection is visible: the
    /// silent channel's feed is all zeros and the loud channel's is the ramp.
    /// Regression: a selection that reads the wrong channel while every
    /// length still agrees.
    #[test]
    fn a_channel_source_feeds_the_named_delivered_channel() {
        let frames = 160;
        let loud: Vec<f32> = (0..frames).map(|i| (i % 16) as f32 * 0.05 + 0.1).collect();
        // Delivered channel 0 is silent; delivered channel 1 carries the ramp.
        let interleaved: Vec<f32> = loud.iter().flat_map(|&s| [0.0, s]).collect();

        // Tap path.
        for (source, expected) in [
            (DetectorSource::Channel(0), vec![0.0_f32; frames]),
            (DetectorSource::Channel(1), loud.clone()),
        ] {
            let (stream, _sender, _running) =
                test_stream_with_source(Some(two_channel_tap_chain()), 2, source);
            stream.push_vad_tap(interleaved.clone());
            let feed = stream
                .detector_feed(&interleaved)
                .expect("a transform keeps the tap active");
            assert_eq!(feed, expected, "the tap path feeds {source:?} alone");
        }

        // Delivered path (no chain).
        for (source, expected) in [
            (DetectorSource::Channel(0), vec![0.0_f32; frames]),
            (DetectorSource::Channel(1), loud.clone()),
        ] {
            let (stream, _sender, _running) = test_stream_with_source(None, 2, source);
            let feed = stream
                .detector_feed(&interleaved)
                .expect("a multichannel delivered chunk is collapsed");
            assert_eq!(feed, expected, "the delivered path feeds {source:?} alone");
        }
    }

    /// `validate` refuses a detector source at or above the configured
    /// delivered count and accepts one below it, at any count: the check is
    /// against the configuration's own `channels`, and a large count admits a
    /// correspondingly large index. The negative control against a fixed
    /// maximum reintroduced on the index.
    #[test]
    fn detector_source_is_validated_against_the_delivered_count() {
        let mut config = MicrophoneConfig {
            channels: 2,
            detector_source: DetectorSource::Channel(2),
            ..Default::default()
        };
        assert!(matches!(
            config.validate(),
            Err(DecibriError::DetectorSourceOutOfRange {
                index: 2,
                channels: 2
            })
        ));

        config.detector_source = DetectorSource::Channel(1);
        assert!(config.validate().is_ok(), "an index below the count passes");

        // Mono: only channel 0 exists.
        config.channels = 1;
        config.detector_source = DetectorSource::Channel(1);
        assert!(matches!(
            config.validate(),
            Err(DecibriError::DetectorSourceOutOfRange {
                index: 1,
                channels: 1
            })
        ));
        config.detector_source = DetectorSource::Channel(0);
        assert!(config.validate().is_ok(), "channel 0 is valid on mono");

        // No fixed maximum: a large delivered count admits a large index.
        config.channels = 1024;
        config.detector_source = DetectorSource::Channel(1023);
        assert!(
            config.validate().is_ok(),
            "the delivered count is the only ceiling"
        );
    }

    /// The tap stays aligned with the delivered output through the resampler's
    /// flushed group-delay tail at close: the VAD feed equals the post-normalize
    /// signal (resample process + flush) bit for bit and matches the delivered
    /// count, so the close path does not desync the tap.
    #[test]
    fn test_vad_input_aligned_through_resampler_flush_tail() {
        use decibri_resampler::{PolyphaseResampler, Resampler};

        let enhancement = true;
        let stage = build_capture_stage(
            1,
            1,
            48_000,
            16_000,
            Transforms {
                dc_removal: enhancement,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("resample + DC chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        let input: Vec<f32> = (0..24_000).map(|k| (k as f32 * 0.01).sin() + 0.5).collect();
        sender.send(make_native_chunk(input.clone())).unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        let samples = 320;
        let mut delivered: Vec<f32> = Vec::new();
        let mut vad_feed: Vec<f32> = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                Ok(Some(c)) => {
                    let pre = stream.vad_input(c.data.len()).expect("tap active");
                    vad_feed.extend_from_slice(&pre);
                    delivered.extend_from_slice(&c.data);
                }
                Ok(None) => panic!("unexpected timeout"),
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }

        // Ground truth post-normalize signal: the resampler over the whole input,
        // process then flush (no DC removal).
        let mut resampler = PolyphaseResampler::new(48_000, 16_000).unwrap();
        let mut expected_norm = Vec::new();
        resampler.process(&input, &mut expected_norm).unwrap();
        resampler.flush(&mut expected_norm);

        assert_eq!(
            vad_feed, expected_norm,
            "the VAD feed is the post-normalize signal incl. the resampler flush tail"
        );
        assert_eq!(
            vad_feed.len(),
            delivered.len(),
            "tap and delivered stay aligned through the flushed tail"
        );
        assert_ne!(
            vad_feed, delivered,
            "the delivered output is post-transform (DC removed), so it differs from the feed"
        );
    }

    /// Energy-VAD invariance to enhancement: the energy score (the RMS the
    /// bindings compute on the pre-enhancement `vad_input` tap) is the SAME
    /// whether or not a transform is enabled, because both read the post-
    /// normalize, pre-transform signal. AGC is the worst trigger: it drives the
    /// DELIVERED level toward its target, so a score computed on the delivered
    /// chunk (the pre-fix wrapper behaviour) would shift sharply, while the score
    /// on the tap does not. This is the energy-mode analogue of
    /// `test_vad_input_returns_pre_transform_signal`, asserted through the RMS
    /// the bindings now compute in native.
    ///
    /// Needs the `gain` feature: with it compiled out the AGC transform is
    /// inert, the delivered level equals the pre-transform level, and the
    /// divergence assertion cannot hold.
    #[cfg(feature = "gain")]
    #[test]
    fn test_energy_score_invariant_to_transform() {
        // A quiet sine: AGC boosts it well above its input level, so the
        // delivered RMS differs sharply from the pre-transform RMS.
        let input: Vec<f32> = (0..16_000)
            .map(|k| 0.03 * (k as f32 * 0.05).sin())
            .collect();

        // Run the chain over the input, returning (energy score on the
        // pre-transform tap, RMS of the delivered output) exactly as a binding
        // would: feed the `vad_input` tap when present, else the delivered chunk.
        let run = |transforms: Transforms<'_>| -> (f32, f32) {
            let stage = build_capture_stage(1, 1, 16_000, 16_000, transforms).unwrap();
            let (stream, sender, running) = test_stream_with(stage, 1);
            sender.send(make_native_chunk(input.clone())).unwrap();
            drop(sender);
            running.store(false, Ordering::Relaxed);

            let samples = 320;
            let mut pre: Vec<f32> = Vec::new();
            let mut delivered: Vec<f32> = Vec::new();
            loop {
                match stream.next_chunk(samples, Some(Duration::from_millis(100))) {
                    Ok(Some(c)) => {
                        match stream.vad_input(c.data.len()) {
                            Some(v) => pre.extend_from_slice(&v),
                            None => pre.extend_from_slice(&c.data),
                        }
                        delivered.extend_from_slice(&c.data);
                    }
                    Ok(None) => panic!("unexpected timeout"),
                    Err(DecibriError::MicrophoneStreamClosed) => break,
                    Err(e) => panic!("unexpected error: {e}"),
                }
            }
            (crate::sample::rms(&pre), crate::sample::rms(&delivered))
        };

        let off = Transforms::default();
        let (baseline, baseline_delivered) = run(off);
        // No transform: the tap is inactive, so the pre feed IS the delivered
        // chunk and the two scores coincide.
        assert!(
            (baseline - baseline_delivered).abs() < 1e-6,
            "no-transform baseline is self-consistent ({baseline} vs {baseline_delivered})"
        );

        // AGC active (the worst trigger): the pre-transform score is unchanged,
        // while the delivered RMS shifts sharply (the pre-fix delivered-chunk
        // path would have diverged, since AGC boosts the quiet input).
        let agc = Transforms {
            agc: Some(-18),
            ..Default::default()
        };
        let (agc_pre, agc_delivered) = run(agc);
        assert!(
            (agc_pre - baseline).abs() < 1e-4,
            "energy score is unchanged by AGC: {agc_pre} vs baseline {baseline}"
        );
        assert!(
            agc_delivered > agc_pre * 1.5,
            "AGC raises the delivered level well above the input, so a delivered-chunk \
             score would diverge ({agc_delivered} vs pre-transform {agc_pre})"
        );

        // High-pass active: same invariance on the tap.
        let hp = Transforms {
            highpass: Some(HighpassFilter::Hz80),
            ..Default::default()
        };
        let (hp_pre, _) = run(hp);
        assert!(
            (hp_pre - baseline).abs() < 1e-4,
            "energy score is unchanged by the high-pass: {hp_pre} vs baseline {baseline}"
        );

        // DC removal active: same invariance on the tap.
        let dc = Transforms {
            dc_removal: true,
            ..Default::default()
        };
        let (dc_pre, _) = run(dc);
        assert!(
            (dc_pre - baseline).abs() < 1e-4,
            "energy score is unchanged by DC removal: {dc_pre} vs baseline {baseline}"
        );
    }

    /// Compile-time assertion that `Arc<Mutex<MicrophoneStream>>` is `Send + Sync`,
    /// which requires `MicrophoneStream: Send`. This is the wrapping strategy the
    /// bindings document for consumers needing shared access from multiple
    /// threads.
    ///
    /// If `MicrophoneStream` ever becomes `!Send` (for example by adding an `Rc<_>`
    /// or `RefCell<_>` field), this test fails to compile, catching the
    /// regression at build time rather than at a binding wrap call site.
    #[test]
    fn test_arc_mutex_microphone_stream_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Arc<std::sync::Mutex<MicrophoneStream>>>();
    }

    /// `Arc<Mutex<MicrophoneStream>>` exercised from two threads racing for
    /// the lock: each thread reads one injected chunk, the Mutex
    /// serializes access, and together they consume exactly the chunks
    /// injected (no duplicates, no losses, no deadlock).
    ///
    /// `Barrier` forces both threads to attempt the lock acquisition
    /// simultaneously so the test exercises contention rather than
    /// sequential non-overlapping access.
    #[test]
    fn test_arc_mutex_microphone_stream_serializes_two_threads() {
        use std::sync::{Barrier, Mutex};

        let (stream, sender, _running) = test_stream();
        sender.send(make_chunk(1.0)).unwrap();
        sender.send(make_chunk(2.0)).unwrap();

        let shared = Arc::new(Mutex::new(stream));
        let barrier = Arc::new(Barrier::new(2));

        let s1 = shared.clone();
        let b1 = barrier.clone();
        let t1 = thread::spawn(move || {
            b1.wait();
            let guard = s1.lock().unwrap();
            guard.try_next_chunk(1).unwrap()
        });

        let s2 = shared.clone();
        let b2 = barrier.clone();
        let t2 = thread::spawn(move || {
            b2.wait();
            let guard = s2.lock().unwrap();
            guard.try_next_chunk(1).unwrap()
        });

        let c1 = t1.join().unwrap().expect("thread 1 must receive a chunk");
        let c2 = t2.join().unwrap().expect("thread 2 must receive a chunk");

        let mut vals = [c1.data[0], c2.data[0]];
        vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(
            vals,
            [1.0, 2.0],
            "both chunks must be consumed exactly once with no duplicates or losses"
        );
    }

    /// Compile-time guard that `MicrophoneStream` is `Send + Sync`. The
    /// `_stream` field is a `Mutex` to keep this bound; a future change to
    /// `Cell`/`RefCell` would make the type `!Sync` and fail this test in the
    /// core crate, rather than only surfacing as a `decibri-python` build break
    /// (pyo3's `#[pyclass]` and `py.detach` both require `Sync`).
    #[test]
    fn test_microphone_stream_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MicrophoneStream>();
    }

    /// `stop()` takes the held stream out from under the mutex (releasing the
    /// device in production) and is idempotent. The test seam holds no real
    /// device, so this asserts the slot is empty after stop and that a second
    /// stop does not panic; the real `Some -> None` drop is exercised by the
    /// binding suites and integration capture.
    #[test]
    fn test_stop_empties_stream_and_is_idempotent() {
        let (stream, _sender, _running) = test_stream();
        stream.stop();
        assert!(!stream.is_open(), "stop() clears the running flag");
        assert!(
            !stream._stream.is_active(),
            "stop() leaves the stream slot empty (device released)"
        );
        stream.stop(); // idempotent: an already-empty slot must not panic
    }

    /// The AGC target is range-checked at the core as a load-bearing backstop: an
    /// out-of-range `agc` errors with `AgcTargetOutOfRange` rather than clamping,
    /// while an in-range target and the `None` default both validate. The guard
    /// runs in `MicrophoneConfig::validate`, the same site that range-checks
    /// `sample_rate`, so a Rust consumer that bypasses the bindings is protected.
    #[test]
    fn agc_target_out_of_range_is_a_core_error() {
        let mut cfg = MicrophoneConfig::default();
        assert!(cfg.validate().is_ok(), "the default (agc None) validates");

        cfg.agc = Some(-18);
        assert!(cfg.validate().is_ok(), "an in-range target validates");
        cfg.agc = Some(-40);
        assert!(cfg.validate().is_ok(), "the lower edge validates");
        cfg.agc = Some(-3);
        assert!(cfg.validate().is_ok(), "the upper edge validates");

        cfg.agc = Some(-41);
        assert!(
            matches!(cfg.validate(), Err(DecibriError::AgcTargetOutOfRange)),
            "below the range errors, not clamps"
        );
        cfg.agc = Some(-2);
        assert!(
            matches!(cfg.validate(), Err(DecibriError::AgcTargetOutOfRange)),
            "above the range errors"
        );
        cfg.agc = Some(-100);
        assert!(
            matches!(cfg.validate(), Err(DecibriError::AgcTargetOutOfRange)),
            "well below the range errors"
        );
    }

    /// The limiter ceiling is range-checked at the core as a load-bearing
    /// backstop: an out-of-range `limiter` errors with `LimiterCeilingOutOfRange`
    /// rather than clamping, while an in-range ceiling and the `None` default both
    /// validate. The guard runs in `MicrophoneConfig::validate`, the same site that
    /// range-checks `sample_rate` and `agc`, so a Rust consumer that bypasses the
    /// bindings is protected.
    #[test]
    fn limiter_ceiling_out_of_range_is_a_core_error() {
        let mut cfg = MicrophoneConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "the default (limiter None) validates"
        );

        cfg.limiter = Some(-1.0);
        assert!(cfg.validate().is_ok(), "an in-range ceiling validates");
        cfg.limiter = Some(-3.0);
        assert!(cfg.validate().is_ok(), "the lower edge validates");
        cfg.limiter = Some(0.0);
        assert!(cfg.validate().is_ok(), "the upper edge validates");

        cfg.limiter = Some(-3.5);
        assert!(
            matches!(cfg.validate(), Err(DecibriError::LimiterCeilingOutOfRange)),
            "below the range errors, not clamps"
        );
        cfg.limiter = Some(0.5);
        assert!(
            matches!(cfg.validate(), Err(DecibriError::LimiterCeilingOutOfRange)),
            "above the range errors"
        );
        cfg.limiter = Some(-100.0);
        assert!(
            matches!(cfg.validate(), Err(DecibriError::LimiterCeilingOutOfRange)),
            "well below the range errors"
        );
    }

    /// `MicrophoneConfig::validate` bounds `channels` below and not above. A
    /// zero channel count is `ChannelsOutOfRange`; every count above it
    /// validates, whatever its size, because the only ceiling is the resolved
    /// device's and this function has no device in scope. A fixed maximum
    /// added here later fails against the large counts below.
    #[test]
    fn validate_bounds_channels_below_and_not_above() {
        let mut cfg = MicrophoneConfig::default();
        assert_eq!(cfg.channels, 1, "the default channel count is mono");
        assert!(cfg.validate().is_ok(), "the default (channels 1) validates");

        for channels in [1u16, 2, 6, 32, 100, 1024, u16::MAX] {
            cfg.channels = channels;
            assert!(
                cfg.validate().is_ok(),
                "channels {channels} validates: no upper bound lives here"
            );
        }

        cfg.channels = 0;
        assert!(
            matches!(cfg.validate(), Err(DecibriError::ChannelsOutOfRange)),
            "zero channels is the one channel-count rejection validate makes"
        );
    }

    /// The unmapped derivation has exactly two meanings, and refuses the rest
    /// against the device's own report rather than choosing.
    ///
    /// Negative control: the two accepted rows below are the reason a blanket
    /// refusal cannot pass this test.
    #[test]
    fn unmapped_channels_accept_only_the_average_and_the_identity() {
        for (requested, device) in [(1u16, 1u16), (1, 2), (1, 6), (2, 2), (6, 6), (123, 123)] {
            assert!(
                validate_unmapped_channels(requested, device).is_ok(),
                "{requested} of {device} is the average or the identity"
            );
        }

        for (requested, device) in [(2u16, 1u16), (7, 6), (1024, 2)] {
            assert!(
                matches!(
                    validate_unmapped_channels(requested, device),
                    Err(DecibriError::MicrophoneChannelsUnsupported {
                        requested: r,
                        available: a,
                    }) if r == requested && a == device
                ),
                "{requested} of {device} exceeds the device and names both figures"
            );
        }

        for (requested, device) in [(2u16, 6u16), (2, 3), (5, 6), (63, 64)] {
            assert!(
                matches!(
                    validate_unmapped_channels(requested, device),
                    Err(DecibriError::ChannelSelectionAmbiguous {
                        requested: r,
                        available: a,
                    }) if r == requested && a == device
                ),
                "{requested} of {device} is a strict subset and needs a map"
            );
        }
    }

    /// A map lifts both unmapped rules, because its entries are checked
    /// against the device one at a time. A map may repeat a channel, so it may
    /// be longer than the device's own count: `channels: 4` on a mono device
    /// is legitimate through `[0, 0, 0, 0]` and refused without a map.
    #[test]
    fn a_map_may_deliver_more_channels_than_the_device_has() {
        assert!(
            validate_channel_map(&[0, 0, 0, 0], 4, 1).is_ok(),
            "four copies of a mono device's one channel is a valid map"
        );
        assert!(
            matches!(
                validate_unmapped_channels(4, 1),
                Err(DecibriError::MicrophoneChannelsUnsupported { .. })
            ),
            "the same count without a map is refused"
        );
        assert!(
            validate_channel_map(&[3, 1], 2, 6).is_ok(),
            "a strict subset in an arbitrary order is a valid map"
        );
        assert!(
            matches!(
                validate_unmapped_channels(2, 6),
                Err(DecibriError::ChannelSelectionAmbiguous { .. })
            ),
            "the same count without a map is refused"
        );
    }

    /// A channel map whose length differs from the configured delivered count
    /// is a configuration error naming both figures. The empty map is the
    /// same mismatch, not a special case.
    #[test]
    fn channel_map_length_must_match_the_delivered_count() {
        assert!(
            validate_channel_map(&[0], 1, 2).is_ok(),
            "one entry for one delivered channel validates"
        );
        assert!(
            matches!(
                validate_channel_map(&[0, 1], 1, 2),
                Err(DecibriError::ChannelMapLengthMismatch {
                    entries: 2,
                    channels: 1
                })
            ),
            "two entries against one delivered channel is a length mismatch"
        );
        assert!(
            matches!(
                validate_channel_map(&[], 1, 2),
                Err(DecibriError::ChannelMapLengthMismatch {
                    entries: 0,
                    channels: 1
                })
            ),
            "an empty map against one delivered channel is a length mismatch"
        );
    }

    /// A map entry at or above the device's reported count is rejected with
    /// the entry and the report both named; every entry below the report is
    /// accepted, right up to the report itself.
    ///
    /// The accepting arm is this path's negative control against the
    /// no-fixed-maximum rule: index 59999 on a device reporting 60000
    /// channels must pass, so a fixed maximum added later fails loudly here.
    #[test]
    fn channel_map_entries_are_bounded_only_by_the_device_report() {
        assert!(
            matches!(
                validate_channel_map(&[2], 1, 2),
                Err(DecibriError::ChannelMapOutOfRange {
                    index: 2,
                    available: 2
                })
            ),
            "the first index past a 2-channel report is rejected"
        );
        assert!(
            validate_channel_map(&[1], 1, 2).is_ok(),
            "the last channel of a 2-channel report is accepted"
        );
        assert!(
            validate_channel_map(&[0], 1, 1).is_ok(),
            "channel 0 of a mono report is accepted"
        );
        assert!(
            validate_channel_map(&[59_999], 1, 60_000).is_ok(),
            "the top index of a 60000-channel report is accepted"
        );
        assert!(
            matches!(
                validate_channel_map(&[60_000], 1, 60_000),
                Err(DecibriError::ChannelMapOutOfRange {
                    index: 60_000,
                    available: 60_000
                })
            ),
            "the first index past a 60000-channel report is rejected"
        );
        // Duplicates are permitted at the general width: each delivered
        // channel is an independent copy of its source.
        assert!(
            validate_channel_map(&[1, 1], 2, 2).is_ok(),
            "a duplicated entry validates"
        );
    }

    /// A stream whose chain gathers device channel 1 delivers exactly that
    /// channel's samples: the consumer-path twin of the stage-level selection
    /// test, driven through `try_next_chunk` with injected native stereo
    /// blocks and no device.
    #[test]
    fn selected_channel_flows_through_the_consumer_path() {
        let map = [1_u16];
        let stage = build_capture_stage(
            2,
            1,
            16_000,
            16_000,
            Transforms {
                channel_map: Some(&map),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("a mapped stereo device builds a select chain");
        let (stream, sender, running) = test_stream_with(Some(stage), 1);

        // Two native stereo blocks; channel 0 carries positives, channel 1
        // negatives.
        sender
            .send(AudioChunk {
                data: vec![1.0, -1.0, 2.0, -2.0],
                sample_rate: 16_000,
                channels: 2,
            })
            .unwrap();
        sender
            .send(AudioChunk {
                data: vec![3.0, -3.0, 4.0, -4.0],
                sample_rate: 16_000,
                channels: 2,
            })
            .unwrap();
        drop(sender);
        running.store(false, Ordering::Relaxed);

        assert_eq!(
            drain_try(&stream, 2),
            vec![-1.0, -2.0, -3.0, -4.0],
            "the delivered stream is device channel 1 exactly"
        );
    }

    /// A 48 kHz mono device resampled to the 16 kHz target: the chain whose
    /// stages carry state across blocks, so a block fed after the close-path
    /// flush would run through emptied stages.
    fn resample_stage() -> CaptureStage {
        build_capture_stage(1, 1, 48_000, 16_000, Transforms::default())
            .unwrap()
            .expect("48 kHz mono device -> 16 kHz resample chain")
    }

    /// Read a stream to exhaustion with `try_next_chunk`, returning every
    /// delivered sample in order. Stops on the closed signal.
    fn drain_try(stream: &MicrophoneStream, samples: usize) -> Vec<f32> {
        let mut out = Vec::new();
        loop {
            match stream.try_next_chunk(samples) {
                Ok(Some(c)) => out.extend_from_slice(&c.data),
                Ok(None) => break,
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        out
    }

    /// Read a stream to exhaustion with `next_chunk`, returning every delivered
    /// sample in order. Stops on the closed signal.
    fn drain_next(stream: &MicrophoneStream, samples: usize) -> Vec<f32> {
        let mut out = Vec::new();
        loop {
            match stream.next_chunk(samples, Some(Duration::from_millis(60))) {
                Ok(Some(c)) => out.extend_from_slice(&c.data),
                Ok(None) => break,
                Err(DecibriError::MicrophoneStreamClosed) => break,
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        out
    }

    /// A native buffer arriving after the close-path flush is not fed through the
    /// chain: the delivered stream is exactly what the same capture yields with no
    /// late buffer at all, down to the sample.
    ///
    /// The chain's stages hold their conditioning state between blocks and the
    /// flush empties it, so running a further block through them resumes from the
    /// drained state and yields samples that do not continue the stream. The
    /// driver-error close reaches this sequence: it clears the running flag
    /// without dropping the platform stream, so the capture channel stays live
    /// and a buffer already in flight can land after the flush.
    #[test]
    fn late_buffer_after_flush_is_not_fed_through_the_chain() {
        // The native buffer length is deliberately not a multiple of the 3:1
        // decimation factor, so a resumed chain would land off the decimation
        // grid as well as starting from emptied filter state.
        let warm: Vec<f32> = (0..4_801).map(|n| (n as f32 * 0.01).sin()).collect();
        let late: Vec<f32> = (0..4_800)
            .map(|n| ((n + 4_801) as f32 * 0.01).sin())
            .collect();
        let samples = 320;

        // Reference: the same capture, closed the same way, with no late buffer.
        let (reference, ref_sender, ref_running) = test_stream_with(Some(resample_stage()), 1);
        ref_sender.send(make_native_chunk(warm.clone())).unwrap();
        ref_running.store(false, Ordering::Relaxed);
        let expected = drain_try(&reference, samples);
        assert!(
            reference.chain_flushed.load(Ordering::Relaxed),
            "the reference read reached the close-path flush"
        );

        // The same capture, with a native buffer landing after the flush.
        let (stream, sender, running) = test_stream_with(Some(resample_stage()), 1);
        sender.send(make_native_chunk(warm)).unwrap();
        running.store(false, Ordering::Relaxed);
        let before = drain_try(&stream, samples);
        assert!(
            stream.chain_flushed.load(Ordering::Relaxed),
            "the read reached the close-path flush"
        );

        sender.send(make_native_chunk(late)).unwrap();
        let after = drain_try(&stream, samples);

        assert!(
            after.is_empty(),
            "a buffer arriving after the flush delivered {} samples",
            after.len()
        );
        assert_eq!(
            before, expected,
            "the delivered stream must match the capture with no late buffer"
        );
    }

    /// The same guarantee on the blocking read: `next_chunk` does not pull a
    /// buffer that arrives after the close-path flush.
    #[test]
    fn next_chunk_does_not_ingest_after_flush() {
        let warm: Vec<f32> = (0..4_801).map(|n| (n as f32 * 0.01).sin()).collect();
        let late: Vec<f32> = (0..4_800)
            .map(|n| ((n + 4_801) as f32 * 0.01).sin())
            .collect();
        let samples = 320;

        let (stream, sender, running) = test_stream_with(Some(resample_stage()), 1);
        sender.send(make_native_chunk(warm)).unwrap();
        running.store(false, Ordering::Relaxed);
        let before = drain_next(&stream, samples);
        assert!(
            !before.is_empty() && stream.chain_flushed.load(Ordering::Relaxed),
            "the read delivered the capture and reached the close-path flush"
        );

        sender.send(make_native_chunk(late)).unwrap();
        assert!(
            matches!(
                stream.next_chunk(samples, Some(Duration::from_millis(60))),
                Err(DecibriError::MicrophoneStreamClosed)
            ),
            "the stream stays closed rather than delivering the late buffer"
        );
    }

    /// The flushed chain rejects a block rather than conditioning it, so a caller
    /// reaching `ingest` after the flush ends the stream on the closed signal
    /// instead of delivering output the chain cannot produce.
    ///
    /// Both regimes of the guard are pinned here. A debug build stops on the
    /// assertion, which is why the test expects that panic when
    /// `debug_assertions` is on; a release build has no assertion and returns
    /// the error the body asserts.
    #[test]
    #[cfg_attr(
        debug_assertions,
        should_panic(expected = "ingest into a flushed chain")
    )]
    fn ingest_after_flush_is_rejected() {
        let (stream, _sender, running) = test_stream_with(Some(resample_stage()), 1);
        running.store(false, Ordering::Relaxed);
        let mut buf = VecDeque::new();
        stream.flush_chain(&mut buf).expect("close-path flush");
        assert!(stream.chain_flushed.load(Ordering::Relaxed));

        let err = stream
            .ingest(&mut buf, make_native_chunk(vec![0.5; 480]))
            .expect_err("a flushed chain rejects a further block");
        assert!(matches!(err, DecibriError::MicrophoneStreamClosed));
    }

    /// Restarting reads the new capture in full: the flushed state belongs to the
    /// stream that closed, not to the [`Microphone`] that produced it.
    ///
    /// [`Microphone::start`] builds a whole new `MicrophoneStream` for every
    /// call, so a restart is a second stream over the same configuration, which
    /// is what this pins. A latch held anywhere outlasting one stream would leave
    /// the restarted stream short-circuiting on its first read and reporting
    /// closed, which a consumer cannot tell apart from a stream that genuinely
    /// ended. The restart is read with `next_chunk`, the method the bindings
    /// pump.
    #[test]
    fn a_restarted_stream_reads_after_the_previous_one_was_flushed() {
        let block: Vec<f32> = (0..4_801).map(|n| (n as f32 * 0.01).sin()).collect();
        let samples = 320;

        // First session: fed, closed, and read to exhaustion, so it flushes.
        let (first, first_sender, first_running) = test_stream_with(Some(resample_stage()), 1);
        first_sender.send(make_native_chunk(block.clone())).unwrap();
        first_running.store(false, Ordering::Relaxed);
        let first_delivered = drain_try(&first, samples);
        assert!(
            first.chain_flushed.load(Ordering::Relaxed),
            "the first session reached the close-path flush"
        );
        assert!(!first_delivered.is_empty());

        // The restart, over the same configuration and the same input.
        let (second, second_sender, second_running) = test_stream_with(Some(resample_stage()), 1);
        assert!(
            !second.chain_flushed.load(Ordering::Relaxed),
            "a restarted stream starts with its chain unflushed"
        );
        second_sender.send(make_native_chunk(block)).unwrap();
        second_running.store(false, Ordering::Relaxed);
        let second_delivered = drain_next(&second, samples);

        assert!(
            !second_delivered.is_empty(),
            "the restarted stream reported closed without delivering its capture"
        );
        assert_eq!(
            second_delivered, first_delivered,
            "a restarted stream delivers the same capture as the first"
        );
    }

    /// A denoise-enabled capture that closes before a single buffer arrives
    /// delivers nothing and reports closed: the close-path flush of a chain that
    /// received no audio contributes no samples, so a consumer is never handed
    /// audio the stream did not capture.
    #[cfg(feature = "denoise")]
    #[test]
    fn a_capture_closed_before_any_buffer_delivers_nothing() {
        let (stream, _sender, running) = test_stream_with(Some(denoise_stage()), 1);
        running.store(false, Ordering::Relaxed);

        let delivered = drain_try(&stream, 320);
        assert!(
            stream.chain_flushed.load(Ordering::Relaxed),
            "the read reached the close-path flush"
        );
        assert_eq!(
            delivered.len(),
            0,
            "a capture that received no buffer delivered {} samples",
            delivered.len()
        );
        assert!(matches!(
            stream.next_chunk(320, Some(Duration::from_millis(60))),
            Err(DecibriError::MicrophoneStreamClosed)
        ));
    }

    /// A denoise-enabled capture that received even a fraction of one analysis
    /// frame still delivers that audio at close. The regression the unfed gate
    /// must not break: 100 samples form no frame, so the whole delivery is the
    /// flushed tail.
    #[cfg(feature = "denoise")]
    #[test]
    fn a_capture_fed_a_little_audio_still_delivers_its_tail() {
        let block: Vec<f32> = (0..100).map(|n| (n as f32 * 0.05).sin()).collect();

        let (stream, sender, running) = test_stream_with(Some(denoise_stage()), 1);
        sender.send(make_native_chunk(block)).unwrap();
        running.store(false, Ordering::Relaxed);

        let delivered = drain_try(&stream, 320);
        // Left-pad (256) + 100 real + one window of padding (512) = 868 samples,
        // which yields two whole 256-sample hops.
        assert_eq!(delivered.len(), 512, "the real tail is delivered in full");
        assert!(
            delivered.iter().any(|&s| s != 0.0),
            "the delivered tail carries the audio the stream captured"
        );
    }

    /// A mono 16 kHz chain whose only stage is denoise, so the `transform`
    /// segment is exercised with an empty `normalize`.
    #[cfg(feature = "denoise")]
    fn denoise_stage() -> CaptureStage {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
            .join("models")
            .join("fastenhancer_t.onnx");
        build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                denoise: Some((DenoiseModel::FastEnhancerT, path.as_path(), None)),
                ..Default::default()
            },
        )
        .unwrap()
        .expect("mono 16 kHz denoise chain")
    }

    /// A stream opened, fed while running, and closed normally is unaffected: the
    /// close check that now runs before the drain does not disturb delivery of
    /// buffers that arrive while the stream is open, and the full delivered stream
    /// still matches the chain applied directly.
    #[test]
    fn open_stream_delivers_every_buffer_then_the_flushed_tail() {
        let blocks: Vec<Vec<f32>> = (0..4)
            .map(|b| {
                (0..1_200)
                    .map(|n| ((b * 1_200 + n) as f32 * 0.01).sin())
                    .collect()
            })
            .collect();
        let samples = 320;

        let (stream, sender, running) = test_stream_with(Some(resample_stage()), 1);
        let mut delivered: Vec<f32> = Vec::new();
        for block in &blocks {
            sender.send(make_native_chunk(block.clone())).unwrap();
            // Read while the stream is open: the reordered check must not
            // short-circuit here.
            delivered.extend(drain_try(&stream, samples));
        }
        assert!(
            !stream.chain_flushed.load(Ordering::Relaxed),
            "an open stream is never flushed"
        );
        running.store(false, Ordering::Relaxed);
        delivered.extend(drain_try(&stream, samples));

        // The same chain applied directly to the same blocks, flushed once.
        let mut reference = resample_stage();
        let mut expected: Vec<f32> = Vec::new();
        for block in &blocks {
            expected.extend_from_slice(reference.run(block).unwrap());
        }
        reference.flush(&mut expected).unwrap();

        assert_eq!(
            delivered, expected,
            "an ordinary open-feed-close stream delivers the chain's output unchanged"
        );
    }

    /// The close path on a chained stream that was never fed. The flush still
    /// runs, and a resample stage that processed no audio contributes nothing
    /// to it, so the stream delivers no samples and then reports closed.
    /// Pinned because it is the same close path the flush ordering depends on,
    /// and because the tail's length and content are the chain's, not this
    /// module's.
    #[test]
    fn close_without_any_buffer_delivers_nothing_from_the_resample_stage() {
        let (stream, _sender, running) = test_stream_with(Some(resample_stage()), 1);
        running.store(false, Ordering::Relaxed);

        let delivered = drain_try(&stream, 320);
        assert!(
            stream.chain_flushed.load(Ordering::Relaxed),
            "the close path flushed the chain"
        );

        let mut expected = Vec::new();
        resample_stage().flush(&mut expected).unwrap();
        assert_eq!(
            delivered, expected,
            "a never-fed stream delivers exactly what the chain's flush appends"
        );

        // The closed signal repeats, now via the reordered short-circuit.
        assert!(matches!(
            stream.try_next_chunk(320),
            Err(DecibriError::MicrophoneStreamClosed)
        ));
        assert!(matches!(
            stream.next_chunk(320, Some(Duration::from_millis(20))),
            Err(DecibriError::MicrophoneStreamClosed)
        ));
    }

    /// A block size that is not a whole number of frames is refused, and
    /// refused before anything is dequeued, so the stream stays readable at a
    /// correct size afterwards.
    ///
    /// Regression: draining an arbitrary sample count off an interleaved
    /// buffer. A chunk cut mid-frame reports the right channel count and the
    /// right length, and every chunk after it carries the channels rotated by
    /// the remainder, so nothing in the delivered audio reveals it.
    #[test]
    fn a_block_size_that_splits_a_frame_is_refused_and_consumes_nothing() {
        let (stream, sender, _running) = test_stream_with(None, 2);
        sender
            .send(AudioChunk {
                data: (0..12).map(|value| value as f32).collect(),
                sample_rate: 16000,
                channels: 2,
            })
            .unwrap();

        for samples in [1usize, 3, 5, 1601] {
            assert!(
                matches!(
                    stream.try_next_chunk(samples),
                    Err(DecibriError::BlockSizeNotFrameAligned {
                        samples: s,
                        channels: 2,
                    }) if s == samples
                ),
                "a {samples}-sample block splits a stereo frame"
            );
            assert!(
                matches!(
                    stream.next_chunk(samples, Some(Duration::from_millis(1))),
                    Err(DecibriError::BlockSizeNotFrameAligned { .. })
                ),
                "the blocking read refuses the same {samples}-sample block"
            );
        }

        // Nothing above consumed a sample: the first frame is still frame 0.
        let chunk = stream
            .try_next_chunk(4)
            .expect("an aligned read succeeds")
            .expect("a full block is buffered");
        assert_eq!(
            chunk.data,
            vec![0.0, 1.0, 2.0, 3.0],
            "the refused reads dequeued nothing and left the frames aligned"
        );

        // Negative control: on a mono stream no size can split a frame, so the
        // guard is unreachable and the odd sizes above all read normally.
        let (mono, mono_sender, _mono_running) = test_stream_with(None, 1);
        mono_sender
            .send(AudioChunk {
                data: (0..12).map(|value| value as f32).collect(),
                sample_rate: 16000,
                channels: 1,
            })
            .unwrap();
        for samples in [1usize, 3, 5] {
            assert!(
                mono.try_next_chunk(samples).is_ok(),
                "a mono stream accepts a {samples}-sample block"
            );
        }
    }

    // ── Echo cancellation ──────────────────────────────────────────────

    /// The rate window `validate` enforces is exactly the one the canceller
    /// enforces at construction. decibri names the window itself so the
    /// rejection arrives at configuration time rather than at stream start, which
    /// makes it a second copy; this holds the two together. Regression: the copy
    /// drifting from the canceller's own range, which would either reject a rate
    /// that works or accept one that fails later.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_window_matches_the_cancellers_own() {
        for rate in [7_999u32, 8_000, 16_000, 44_100, 48_000, 48_001] {
            let config = MicrophoneConfig {
                sample_rate: rate,
                aec: Some(decibri_aec::AecModel::default()),
                ..Default::default()
            };

            let mut engine_config = decibri_aec::AecConfig::default();
            engine_config.sample_rate = rate;
            let engine_accepts = decibri_aec::Aec::new(engine_config).is_ok();

            assert_eq!(
                config.validate().is_ok(),
                engine_accepts,
                "validate must accept exactly the rates the canceller accepts (rate {rate})"
            );
        }
    }

    /// Echo cancellation validates at every delivered channel count, with and
    /// without a channel map: no capacity ceiling exists on the pair.
    ///
    /// The counts run well past any plausible machine budget on purpose. The
    /// cost of cancellation scales per delivered channel and the overload
    /// signal is `overrun_count` at runtime, so a capacity rejection at
    /// configuration time would be a number decibri cannot derive; this pins
    /// that no such number exists. A maximum added later fails here loudly.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_validates_at_every_delivered_channel_count() {
        let mut config = MicrophoneConfig {
            aec: Some(decibri_aec::AecModel::default()),
            ..Default::default()
        };

        for channels in [1u16, 2, 3, 8, 64] {
            config.channels = channels;
            config.channel_map = None;
            assert!(
                config.validate().is_ok(),
                "channels {channels} with the canceller validates"
            );

            config.channel_map = Some((0..channels).collect());
            assert!(
                config.validate().is_ok(),
                "channels {channels} with the canceller and a map validates"
            );
        }

        // One channel selected from an array cancels as before.
        config.channels = 1;
        config.channel_map = Some(vec![3]);
        assert!(
            config.validate().is_ok(),
            "one channel selected from an array with the canceller validates"
        );

        // Negative control: the same counts without the canceller also
        // validate, so the acceptance above is not an artifact of a validator
        // that stopped checking channels at all; both sides sit on the same
        // device-bounded rule.
        config.aec = None;
        for channels in [2u16, 3, 8, 64] {
            config.channels = channels;
            config.channel_map = None;
            assert!(
                config.validate().is_ok(),
                "channels {channels} validates with the canceller off"
            );
        }
    }

    /// A target rate outside the canceller's window is rejected with the variant
    /// that names echo cancellation, not the general sample-rate variant: the rate
    /// is one a capture without echo cancellation accepts, so the message has to
    /// say which of the two is refusing it. Regression: reusing
    /// `SampleRateOutOfRange`, whose message names a range the rate is inside.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_rejects_an_out_of_window_target_rate_by_name() {
        let mut config = MicrophoneConfig {
            sample_rate: 96_000,
            aec: Some(decibri_aec::AecModel::default()),
            ..Default::default()
        };
        assert!(
            matches!(
                config.validate(),
                Err(DecibriError::AecSampleRateUnsupported(96_000))
            ),
            "an out-of-window target rate is rejected by name"
        );

        // The same rate without echo cancellation stays valid.
        config.aec = None;
        assert!(
            config.validate().is_ok(),
            "the rate is only out of range for the canceller"
        );

        // A rate outside decibri's own range still reports that, so the general
        // check keeps precedence.
        config.sample_rate = 500_000;
        config.aec = Some(decibri_aec::AecModel::default());
        assert!(
            matches!(config.validate(), Err(DecibriError::SampleRateOutOfRange)),
            "a rate outside decibri's own range reports that first"
        );
    }

    /// A declared reference rate outside the range decibri resamples from is
    /// rejected at configuration time. Regression: the rejection surfacing later
    /// as a resampler construction failure, which names a rate pair rather than
    /// the field the caller set.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_rejects_a_reference_rate_outside_the_resampled_range() {
        let mut config = MicrophoneConfig {
            aec: Some(decibri_aec::AecModel::default()),
            ..Default::default()
        };
        for rate in [999u32, 384_001] {
            config.aec_reference_sample_rate = Some(rate);
            assert!(
                matches!(config.validate(), Err(DecibriError::SampleRateOutOfRange)),
                "a reference rate of {rate} is rejected"
            );
        }
        config.aec_reference_sample_rate = Some(24_000);
        assert!(
            config.validate().is_ok(),
            "a reference rate inside the range is accepted"
        );
    }

    /// The stream's push method reaches the queue the chain's canceller drains,
    /// the discard and substituted-silence counts are reachable from the stream,
    /// and the canceller's metrics are too. Regression: a push that lands in a
    /// queue nothing reads, and a discard count, a silence count, or a metrics
    /// read with no path out to a consumer.
    #[cfg(feature = "aec")]
    #[test]
    fn the_stream_reaches_the_reference_queue_and_the_metrics() {
        let queue = Arc::new(crate::stage::AecReferenceRing::new(4));
        let chain = build_capture_stage(
            1,
            1,
            16_000,
            16_000,
            Transforms {
                aec: Some(AecSettings {
                    model: decibri_aec::AecModel::default(),
                    tail_ms: None,
                    suppression: None,
                    reference_rate: 16_000,
                    reference: Arc::clone(&queue),
                }),
                ..Default::default()
            },
        )
        .expect("the chain builds")
        .expect("echo cancellation builds a chain");
        let (stream, _sender, _running) =
            test_stream_with_reference(Some(chain), 1, Arc::clone(&queue));

        assert_eq!(
            stream.aec_reference_dropped(),
            0,
            "nothing is discarded before anything is pushed"
        );
        stream.push_aec_reference(&[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        assert_eq!(
            stream.aec_reference_dropped(),
            2,
            "the samples past the queue's bound are counted through the stream"
        );

        assert_eq!(
            stream.aec_reference_silence(),
            0,
            "a stream that has processed no capture has substituted no silence"
        );
        let mut buf = VecDeque::new();
        stream
            .ingest(&mut buf, make_native_chunk(vec![0.1; 320]))
            .expect("the chain conditions one block");
        assert_eq!(
            stream.aec_reference_silence(),
            320 - 4,
            "the drain covers the near-end samples the four queued reference \
             samples did not"
        );

        let metrics = stream.aec_metrics().expect("the stream reports metrics");
        assert_eq!(
            metrics.reference_reanchors, 0,
            "a stream that processed one block has re-anchored nothing"
        );
        assert_eq!(
            metrics.reference_starved, 0,
            "a far end kept level with the capture starves nothing"
        );

        // A stream with echo cancellation off answers without a queue: the push is
        // a no-op, both counts are zero, and there are no metrics.
        let (plain, _sender, _running) = test_stream_with(None, 1);
        plain.push_aec_reference(&[0.1, 0.2]);
        assert_eq!(plain.aec_reference_dropped(), 0);
        assert_eq!(plain.aec_reference_silence(), 0);
        assert!(plain.aec_metrics().is_none());
    }

    /// On a stream delivering more than one channel, the per-channel accessor
    /// carries one entry per delivered channel and the single-report accessor
    /// is its first entry. Regression: the two accessors answering from
    /// different engines, or the per-channel report collapsing to one entry.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_metrics_per_channel_carries_every_delivered_channel() {
        let queue = Arc::new(crate::stage::AecReferenceRing::new(16_000));
        let chain = build_capture_stage(
            2,
            2,
            16_000,
            16_000,
            Transforms {
                aec: Some(AecSettings {
                    model: decibri_aec::AecModel::default(),
                    tail_ms: None,
                    suppression: None,
                    reference_rate: 16_000,
                    reference: Arc::clone(&queue),
                }),
                ..Default::default()
            },
        )
        .expect("the chain builds")
        .expect("echo cancellation builds a chain");
        let (stream, _sender, _running) =
            test_stream_with_reference(Some(chain), 2, Arc::clone(&queue));

        let mut buf = VecDeque::new();
        stream
            .ingest(
                &mut buf,
                AudioChunk {
                    data: vec![0.1; 2 * 512],
                    sample_rate: 16_000,
                    channels: 2,
                },
            )
            .expect("the chain conditions one block");

        let per_channel = stream
            .aec_metrics_per_channel()
            .expect("the stream reports per-channel metrics");
        assert_eq!(per_channel.len(), 2, "one entry per delivered channel");
        let single = stream.aec_metrics().expect("the stream reports metrics");
        assert_eq!(
            single, per_channel[0],
            "the single report is the first delivered channel's entry"
        );
    }

    /// A stereo push whose two channels are identical leaves the queue exactly
    /// what the mono push of the same content leaves, and the mono push itself
    /// carries the slice through untouched. The averaging arithmetic is pinned
    /// by `sample::downmix_to_mono`'s own tests; this pins the wiring from the
    /// declared count to the collapse, where a stride or offset error would
    /// land interleaved samples in the queue.
    #[cfg(feature = "aec")]
    #[test]
    fn a_duplicated_channel_stereo_push_matches_the_mono_push() {
        let mono_content = [0.1f32, -0.2, 0.3, -0.4];
        let stereo: Vec<f32> = mono_content.iter().flat_map(|&s| [s, s]).collect();

        let mono_queue = Arc::new(crate::stage::AecReferenceRing::new(16));
        let (mono_stream, _s, _r) = test_stream_with_reference(None, 1, Arc::clone(&mono_queue));
        mono_stream.push_aec_reference(&mono_content);
        assert_eq!(
            mono_queue.queued_samples(),
            mono_content,
            "the default count of 1 passes the slice through untouched"
        );

        let queue = Arc::new(crate::stage::AecReferenceRing::new(16));
        let (stream, _s2, _r2) = test_stream_with_reference(None, 1, Arc::clone(&queue));
        let stereo_stream = MicrophoneStream {
            aec_reference_channels: 2,
            ..stream
        };
        stereo_stream.push_aec_reference(&stereo);
        assert_eq!(
            queue.queued_samples(),
            mono_queue.queued_samples(),
            "duplicated channels collapse to the mono push's own content"
        );
    }

    /// A stereo push with different content per channel queues each frame's
    /// average, and the queue is sized and filled in mono samples after the
    /// collapse: six frames against a four-sample capacity leave four averages
    /// queued and two mono samples counted dropped. Catches a collapse that
    /// takes one channel instead of averaging, and a queue made channel-aware
    /// (sized or filled in interleaved samples), which would reintroduce the
    /// undeclared-stereo defect from the other side.
    #[cfg(feature = "aec")]
    #[test]
    fn a_declared_stereo_push_averages_frames_and_fills_the_queue_in_mono() {
        let frames = [
            (0.2f32, 0.4f32),
            (0.0, 1.0),
            (-0.5, 0.5),
            (0.6, 0.2),
            (-1.0, -0.5),
            (0.3, 0.1),
        ];
        let interleaved: Vec<f32> = frames.iter().flat_map(|&(l, r)| [l, r]).collect();
        let expected: Vec<f32> = frames.iter().map(|&(l, r)| (l + r) / 2.0).collect();

        let queue = Arc::new(crate::stage::AecReferenceRing::new(4));
        let (stream, _s, _r) = test_stream_with_reference(None, 1, Arc::clone(&queue));
        let stream = MicrophoneStream {
            aec_reference_channels: 2,
            ..stream
        };
        stream.push_aec_reference(&interleaved);
        assert_eq!(
            queue.queued_samples(),
            expected[..4],
            "the queue holds the oldest four frame averages"
        );
        assert_eq!(
            stream.aec_reference_dropped(),
            2,
            "the discard is counted in mono samples, after the collapse"
        );
    }

    /// A declared reference channel count of 0 is a configuration error
    /// carrying the echo-canceller configuration identity, and no count above
    /// 0 is rejected: the count declares the shape of the caller's own buffer,
    /// so there is no upper bound to enforce. Driven to `u16::MAX` so a fixed
    /// maximum added later fails here loudly. With echo cancellation off the
    /// count is not consulted, matching the reference rate.
    #[cfg(feature = "aec")]
    #[test]
    fn aec_rejects_a_zero_reference_channel_count_and_caps_nothing() {
        let mut config = MicrophoneConfig {
            aec: Some(decibri_aec::AecModel::default()),
            aec_reference_channels: 0,
            ..Default::default()
        };
        let err = config.validate().expect_err("a zero count is rejected");
        assert!(
            matches!(err, DecibriError::AecConfigInvalid { .. }),
            "a zero count reports the echo-canceller configuration identity, got {err:?}"
        );
        assert_eq!(
            err.to_string(),
            "echo canceller configuration error: the reference channel count must be at least 1"
        );
        for count in [1u16, 2, 8, 255, 4096, u16::MAX] {
            config.aec_reference_channels = count;
            assert!(config.validate().is_ok(), "a count of {count} is accepted");
        }
        config.aec = None;
        config.aec_reference_channels = 0;
        assert!(
            config.validate().is_ok(),
            "the count is consulted only when echo cancellation is on"
        );
    }

    /// The failure this field exists for, made deterministic: a stereo far end
    /// pushed with the count declared lets the delay search lock, and the same
    /// bytes pushed undeclared never lock and only overfill the queue. Runs
    /// entirely on synthetic audio through the offline ingest path; the
    /// estimator cuts frames on sample counts and uses exact arithmetic, so
    /// the outcome is the same on every run.
    #[cfg(feature = "aec")]
    #[test]
    fn a_declared_stereo_reference_lets_the_delay_search_lock() {
        /// Deterministic wideband noise from a linear congruential generator,
        /// scaled to `amplitude`.
        fn noise(seed: u32, len: usize, amplitude: f32) -> Vec<f32> {
            let mut state = seed;
            (0..len)
                .map(|_| {
                    state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                    ((state >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0 * amplitude
                })
                .collect()
        }

        const RATE: usize = 16_000;
        const SECONDS: usize = 10;
        const BLOCK: usize = 320;
        const DELAY: usize = 640;
        let len = RATE * SECONDS;
        // The played (mono) signal, plus a difference signal that puts
        // different content on each channel while keeping every frame average
        // exactly the played signal: left carries the sum, right the
        // difference.
        let played = noise(0x2001, len, 0.25);
        let side = noise(0x7f4a_11c3, len, 0.25);
        let left: Vec<f32> = played.iter().zip(&side).map(|(&m, &d)| m + d).collect();
        let right: Vec<f32> = played.iter().zip(&side).map(|(&m, &d)| m - d).collect();
        // The capture is the played signal through a synthetic echo path:
        // delayed by `DELAY` samples and attenuated.
        let near: Vec<f32> = (0..len)
            .map(|i| {
                if i >= DELAY {
                    played[i - DELAY] * 0.5
                } else {
                    0.0
                }
            })
            .collect();

        for (declared, expect_lock) in [(2u16, true), (1u16, false)] {
            let queue = Arc::new(crate::stage::AecReferenceRing::new(
                RATE * AEC_REFERENCE_BOUND_SECS,
            ));
            let chain = build_capture_stage(
                1,
                1,
                16_000,
                16_000,
                Transforms {
                    aec: Some(AecSettings {
                        model: decibri_aec::AecModel::default(),
                        tail_ms: None,
                        suppression: None,
                        reference_rate: 16_000,
                        reference: Arc::clone(&queue),
                    }),
                    ..Default::default()
                },
            )
            .expect("the chain builds")
            .expect("echo cancellation builds a chain");
            let (stream, _sender, _running) =
                test_stream_with_reference(Some(chain), 1, Arc::clone(&queue));
            let stream = MicrophoneStream {
                aec_reference_channels: declared,
                ..stream
            };

            let mut delivered = VecDeque::new();
            let mut locked = None;
            for block in 0..len / BLOCK {
                let start = block * BLOCK;
                let interleaved: Vec<f32> = (start..start + BLOCK)
                    .flat_map(|i| [left[i], right[i]])
                    .collect();
                stream.push_aec_reference(&interleaved);
                stream
                    .ingest(
                        &mut delivered,
                        make_native_chunk(near[start..start + BLOCK].to_vec()),
                    )
                    .expect("the chain conditions the block");
                let metrics = stream.aec_metrics().expect("the stream reports metrics");
                if let Some(delay) = metrics.delay_samples {
                    locked = Some(delay);
                    break;
                }
            }
            if expect_lock {
                let delay = locked.expect("the declared stereo reference locks the delay search");
                // The locked value is the engine's alignment offset, not the
                // bare echo-path delay: the drain feeds each block's reference
                // before the block's near samples are processed, so the far
                // frontier leads the near end by up to one block at the
                // anchor, and the estimator backs the peak off by its onset
                // margin. The window below covers the path delay plus that
                // feed lead.
                assert!(
                    (DELAY - 64..=DELAY + BLOCK).contains(&delay),
                    "the locked delay ({delay}) sits at the synthetic echo path ({DELAY}) \
                     plus the feed lead"
                );
            } else {
                assert!(
                    locked.is_none(),
                    "the undeclared stereo reference never locks"
                );
                assert!(
                    stream.aec_reference_dropped() > 0,
                    "the undeclared stereo push overfills the queue"
                );
            }
        }
    }
}
