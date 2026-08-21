//! Offline audio source.
//!
//! A [`File`] conditions audio you already have, a recording on disk or
//! in-memory samples, through the same chain a live [`crate::Microphone`]
//! drives: channel and rate normalization first, then the opt-in conditioning
//! (DC removal, denoise, high-pass, level control, limiter). Iterating a
//! `File` yields conditioned [`AudioChunk`]s and ends at the last chunk, after
//! the chain's end-of-stream tail has been drained.
//!
//! Because a `File` is a complete recording rather than a live stream, it can
//! also analyze the whole recording for speech: [`File::analyze`] (spelled
//! [`File::analyse`] as well) runs voice-activity detection over the recording
//! and returns a [`VadReport`] with per-window scores and merged speech
//! segments, all timed in seconds of file time (sample positions over the
//! rate), never wall-clock time.
//!
//! Construction mirrors the microphone's config-first shape: build a
//! [`FileConfig`], then open a path with [`File::open`] (or the
//! [`File::new`] alias) or wrap in-memory samples with [`File::buffer`]. The
//! path form reads the input rate and channel count from the file's own
//! header; the buffer form takes both explicitly because raw samples carry
//! no header. [`FileConfig::channels`] and [`FileConfig::channel_map`] name
//! what is delivered, in the capture surface's own vocabulary.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use crate::error::DecibriError;
use crate::microphone::{AudioChunk, DenoiseModel, DetectorSource, HighpassFilter};
use crate::stage::{build_capture_stage, CaptureStage, Transforms};

#[cfg(feature = "vad")]
use crate::vad::{SileroVad, VadConfig};
#[cfg(feature = "vad")]
use decibri_resampler::{PolyphaseResampler, Resampler};

/// Number of input frames fed through the chain per internal step. Any block
/// size produces identical output (the chain is block-invariant); this one
/// matches the microphone's default delivery block so the two paths exercise
/// the chain the same way.
const FEED_FRAMES: usize = 1600;

/// Number of output frames per delivered chunk at the target rate (a chunk
/// carries this many frames times the delivered channel count in interleaved
/// samples), matching the microphone's default `frames_per_buffer`. The final
/// chunk may be shorter once the chain's end-of-stream tail has been drained.
const DELIVERY_FRAMES: usize = 1600;

/// Memory bound for the detector feed queue, in seconds of audio at the
/// detector rate. Mirrors the live capture path's tap bound: a consumer that
/// enables VAD but never drains the feed cannot grow memory without limit.
#[cfg(feature = "vad")]
const VAD_QUEUE_BOUND_SECS: usize = 2;

/// Silence tolerated inside one speech segment before it is closed, in
/// milliseconds of file time. The shared default the bindings also use for
/// their per-chunk speaking state.
#[cfg(feature = "vad")]
const DEFAULT_VAD_HOLDOFF_MS: u32 = 300;

/// Configuration for an offline [`File`] source.
///
/// `#[non_exhaustive]`: construct it with [`FileConfig::default`] and then
/// assign the public fields you need. Direct struct-literal construction from
/// another crate is intentionally not supported, so adding a field later stays
/// backward compatible.
///
/// The conditioning fields carry the same names, meanings, ranges, and
/// defaults as their [`crate::MicrophoneConfig`] counterparts.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct FileConfig {
    /// Target output rate in Hz; the rate every delivered chunk carries. The
    /// chain resamples the source's input rate to this rate. Range:
    /// 1000-384000. Default: 16000.
    pub sample_rate: u32,
    /// Number of channels the `File` DELIVERS, interleaved frame by frame in
    /// [`AudioChunk::data`]. Bounded below at 1 (the default) by
    /// [`validate`](Self::validate); bounded above by the source's own
    /// channel count alone, read from the container's header (or stated as
    /// the `input_channels` of [`File::buffer`]). No fixed maximum exists.
    ///
    /// The same meaning as [`crate::MicrophoneConfig::channels`], with the
    /// source's header count standing where the device's report stands on the
    /// live path. How the delivered channels are derived from the source's,
    /// when [`channel_map`](Self::channel_map) is `None`:
    ///
    /// - `1`: the documented average of every source channel.
    /// - equal to the source's own count: every source channel, in source
    ///   order.
    /// - above the source's own count:
    ///   [`DecibriError::FileChannelsUnsupported`] at construction.
    /// - above 1 and below the source's own count:
    ///   [`DecibriError::FileChannelSelectionAmbiguous`] at construction.
    ///   Which of the source's channels those should be has no single answer,
    ///   so [`channel_map`](Self::channel_map) names them rather than decibri
    ///   choosing.
    ///
    /// A [`channel_map`](Self::channel_map) names the source channels
    /// directly, so it answers every case above.
    pub channels: u16,
    /// Optional list of 0-based SOURCE channel indices selecting which source
    /// channels feed the delivered channels: delivered channel `j` carries
    /// source channel `channel_map[j]`. The length must equal
    /// [`channels`](Self::channels). Entries may repeat and may appear in any
    /// order, so a map both selects and permutes. `None` (the default)
    /// derives the delivered channels as [`channels`](Self::channels)
    /// documents.
    ///
    /// The same shape and semantics as
    /// [`crate::MicrophoneConfig::channel_map`], with source channels in the
    /// device channels' place. Validated at construction, against the
    /// source's own channel count: every entry must be below it
    /// ([`DecibriError::FileChannelMapOutOfRange`] otherwise), and the length
    /// must equal [`channels`](Self::channels)
    /// ([`DecibriError::ChannelMapLengthMismatch`] otherwise). The source's
    /// count is the only ceiling; no fixed maximum exists. Not checked by
    /// [`validate`](Self::validate), which has no source in scope. Default:
    /// `None`.
    pub channel_map: Option<Vec<u16>>,
    /// The source of the detector feed: which of the delivered channels a
    /// voice-activity detector reads, for the per-chunk feed
    /// ([`File::vad_input`]) and the whole-recording analysis
    /// ([`File::analyze`]) alike.
    /// [`DetectorSource::Average`](crate::microphone::DetectorSource::Average)
    /// (the default) feeds the frame average of every delivered channel;
    /// [`DetectorSource::Channel`](crate::microphone::DetectorSource::Channel)
    /// feeds one delivered channel alone. Names a DELIVERED channel index,
    /// never a source index: with a [`channel_map`](Self::channel_map)
    /// present, delivered channel `j` carries source channel
    /// `channel_map[j]` and the source names `j`. Affects only the detector
    /// feed; the delivered audio is untouched.
    ///
    /// The same meaning as
    /// [`crate::MicrophoneConfig::detector_source`], with the source's
    /// channels standing where the device's stand on the live path. A
    /// [`DetectorSource::Channel`](crate::microphone::DetectorSource::Channel)
    /// index is validated by [`validate`](Self::validate) against
    /// [`channels`](Self::channels), the delivered count
    /// ([`DecibriError::DetectorSourceOutOfRange`] when it is not below that
    /// count). The delivered count is the only ceiling; no fixed maximum
    /// exists. Honoured only when the `vad` feature is compiled in. Default:
    /// [`DetectorSource::Average`](crate::microphone::DetectorSource::Average).
    pub detector_source: DetectorSource,
    /// Remove a constant (DC) offset with a one-pole DC-blocking high-pass,
    /// applied after the channel and rate normalization. Default: false (off).
    pub dc_removal: bool,
    /// Single-channel speech-enhancement (denoise) model to run on the audio,
    /// applied after DC removal. `None` (the default) leaves denoise off.
    /// Naming a model also requires [`denoise_model_path`](Self::denoise_model_path);
    /// with the model set but no path the stage stays off. Honoured only when
    /// the `denoise` feature is compiled in.
    pub denoise: Option<DenoiseModel>,
    /// Filesystem path to the denoise model's ONNX file, supplied by the
    /// caller (the bindings resolve it from their bundled copy; the core ships
    /// no model bytes). Required when [`denoise`](Self::denoise) names a
    /// model; ignored otherwise. Default: `None`.
    pub denoise_model_path: Option<PathBuf>,
    /// Filesystem path to the ONNX Runtime dynamic library, consulted exactly
    /// as [`crate::MicrophoneConfig::ort_library_path`] is on the live path.
    /// Default: `None`.
    pub ort_library_path: Option<PathBuf>,
    /// High-pass filter applied after denoise, removing low-frequency rumble.
    /// `None` (the default) builds no high-pass stage.
    pub highpass: Option<HighpassFilter>,
    /// Automatic gain control target level in dBFS, applied after the
    /// high-pass. Range: -40 to -3. `None` (the default) builds no
    /// level-control stage. Honoured only when the `gain` feature is compiled
    /// in.
    pub agc: Option<i8>,
    /// Peak limiter ceiling in dBFS, applied last. Range: -3.0 to 0.0. `None`
    /// (the default) builds no limiter stage. Honoured only when the `gain`
    /// feature is compiled in.
    pub limiter: Option<f32>,
    /// Voice-activity detection over the recording. `None` (the default)
    /// leaves VAD off: iteration delivers conditioned audio only and
    /// [`File::analyze`] returns [`DecibriError::VadNotConfigured`].
    ///
    /// The configuration's `model_path`, `threshold`, and `ort_library_path`
    /// are honoured. Its `sample_rate` is managed by the `File`: detection
    /// runs at the target rate when that rate is 8000 or 16000, and otherwise
    /// on an internal copy of the detector feed resampled to 16000, so any
    /// target rate works without a setting or an error.
    #[cfg(feature = "vad")]
    pub vad: Option<VadConfig>,
    /// Silence tolerated inside one speech segment before it is closed, in
    /// milliseconds of FILE time (sample positions, not wall-clock time).
    /// Applies to [`File::analyze`] segment merging. Default: 300.
    #[cfg(feature = "vad")]
    pub vad_holdoff_ms: u32,
}

impl Default for FileConfig {
    fn default() -> Self {
        Self {
            sample_rate: 16000,
            channels: 1,
            channel_map: None,
            detector_source: DetectorSource::Average,
            dc_removal: false,
            denoise: None,
            denoise_model_path: None,
            ort_library_path: None,
            highpass: None,
            agc: None,
            limiter: None,
            #[cfg(feature = "vad")]
            vad: None,
            #[cfg(feature = "vad")]
            vad_holdoff_ms: DEFAULT_VAD_HOLDOFF_MS,
        }
    }
}

impl FileConfig {
    /// Validate the configuration: the target sample rate, the channel floor,
    /// the detector source, the AGC target, and the limiter ceiling (each
    /// when set) must fall within the supported ranges, matching the live
    /// capture path's validation exactly. With VAD configured, the detector
    /// threshold is validated fail-fast as well.
    ///
    /// # Errors
    /// - [`DecibriError::SampleRateOutOfRange`] when `sample_rate` is outside
    ///   1000-384000.
    /// - [`DecibriError::ChannelsOutOfRange`] when `channels` is 0.
    /// - [`DecibriError::DetectorSourceOutOfRange`] when the detector source
    ///   names a delivered channel at or above `channels`.
    /// - [`DecibriError::AgcTargetOutOfRange`] when `agc` is outside -40..=-3.
    /// - [`DecibriError::LimiterCeilingOutOfRange`] when `limiter` is outside
    ///   -3.0..=0.0.
    /// - [`DecibriError::VadThresholdOutOfRange`] when the VAD threshold is
    ///   outside 0.0..=1.0.
    pub fn validate(&self) -> Result<(), DecibriError> {
        if !(1000..=384000).contains(&self.sample_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
        }
        if self.channels == 0 {
            return Err(DecibriError::ChannelsOutOfRange);
        }
        // No upper bound here. The delivered count is bounded by the source's
        // own channel count alone, which is not in scope in this pure
        // function, so the ceiling is applied at construction as
        // `DecibriError::FileChannelsUnsupported`.
        // The detector source names a delivered channel, and the delivered
        // count is `channels`, which IS in scope here unlike the source's own
        // count: an index at or above it can never resolve whatever the
        // source turns out to carry. The delivered count is the only ceiling;
        // no fixed maximum exists.
        if let DetectorSource::Channel(index) = self.detector_source {
            if index >= self.channels {
                return Err(DecibriError::DetectorSourceOutOfRange {
                    index,
                    channels: self.channels,
                });
            }
        }
        if let Some(target) = self.agc {
            if !(-40..=-3).contains(&target) {
                return Err(DecibriError::AgcTargetOutOfRange);
            }
        }
        if let Some(ceiling) = self.limiter {
            if !(-3.0..=0.0).contains(&ceiling) {
                return Err(DecibriError::LimiterCeilingOutOfRange);
            }
        }
        #[cfg(feature = "vad")]
        if let Some(vad) = &self.vad {
            // Validate the threshold now rather than at analysis time. The
            // detector rate the File selects is always 8000 or 16000, so a
            // rate error cannot reach the user from here; only the threshold
            // range is load-bearing.
            let mut detector = vad.clone();
            detector.sample_rate = 16000;
            detector.validate()?;
        }
        Ok(())
    }
}

/// The container format a [`File::save`] call writes.
///
/// Selected from the path's extension, or explicitly via
/// [`SaveOptions::format`]. Every format writes 16-bit PCM at the `File`'s
/// target rate, frame-interleaved at the delivered channel count; FLAC
/// compresses it losslessly at [`SaveOptions::compression`]. Each
/// container's own channel ceiling applies, as [`File::save`] documents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaveFormat {
    /// RIFF/WAVE carrying 16-bit PCM. The `.wav` extension.
    Wav,
    /// AIFF carrying 16-bit big-endian PCM. The `.aiff`, `.aif` and `.aifc`
    /// extensions.
    Aiff,
    /// FLAC carrying 16-bit samples, losslessly compressed. The `.flac`
    /// extension.
    Flac,
}

impl SaveFormat {
    /// The save format the path's extension names: the resolution
    /// [`File::save`] applies when [`SaveOptions::format`] is unset.
    ///
    /// The write side has no bytes to identify, so the name is the signal:
    /// `.wav`, `.aiff`, `.aif`, `.aifc` and `.flac` are recognised, ASCII
    /// case-insensitively. Anything else is refused rather than defaulted,
    /// naming the extension found and the accepted set.
    ///
    /// # Errors
    /// - [`DecibriError::AudioFormatUnsupported`] when the extension names
    ///   no format decibri writes, or the path has no extension.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, DecibriError> {
        let extension = path
            .as_ref()
            .extension()
            .and_then(|e| e.to_str())
            .map(str::to_ascii_lowercase);
        match extension.as_deref() {
            Some("wav") => Ok(SaveFormat::Wav),
            Some("aiff" | "aif" | "aifc") => Ok(SaveFormat::Aiff),
            Some("flac") => Ok(SaveFormat::Flac),
            Some(other) => Err(DecibriError::AudioFormatUnsupported {
                reason: format!(
                    "the extension '.{other}' does not name a format decibri writes; \
                     use .wav, .aiff, .aif, .aifc or .flac, or set an explicit format"
                ),
            }),
            None => Err(DecibriError::AudioFormatUnsupported {
                reason: "the path has no extension to name a format; use .wav, .aiff, \
                         .aif, .aifc or .flac, or set an explicit format"
                    .to_string(),
            }),
        }
    }
}

/// Options for [`File::save`].
///
/// `#[non_exhaustive]`: construct it with [`SaveOptions::default`] and then
/// assign the public fields you need, exactly as [`FileConfig`] is built.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct SaveOptions {
    /// The container format to write. `None` (the default) takes it from the
    /// path's extension: `.wav`, `.aiff`, `.aif`, `.aifc` or `.flac`, ASCII
    /// case-insensitively. decibri reads a file by its content and writes one
    /// by its name; an extension it does not recognise is refused, never
    /// defaulted.
    pub format: Option<SaveFormat>,
    /// FLAC compression level, 0 through 8. Higher levels search harder for a
    /// smaller file; every level decodes to identical audio. `None` (the
    /// default) is level 5. Range checked whenever set; consulted only when
    /// the resolved format is [`SaveFormat::Flac`] and ignored otherwise,
    /// exactly as [`FileConfig`] fields that belong to one stage are ignored
    /// when that stage is off.
    pub compression: Option<u8>,
}

/// What a [`File::save`] call did to the samples on their way into the file:
/// the full-scale clamp and the non-finite repair, each counted.
///
/// `#[non_exhaustive]`: read field by field; sealing it keeps future result
/// fields backward compatible.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct SaveReport {
    /// Finite samples outside `[-1.0, 1.0]`, clamped to full scale. The
    /// conditioned output can exceed full scale (AGC or AEC without a
    /// limiter), and a 16-bit encoding cannot hold that, so the overshoot
    /// clips and the count says how much. The count is a statement about
    /// integer encodings: a float encoding preserves an overscale sample
    /// rather than clipping it, and its count would be zero.
    pub clipped_samples: u64,
    /// Non-finite samples replaced before writing: NaN with silence, an
    /// infinity with full scale. The same replacement on every format, so the
    /// same samples produce the same audio whatever container carries them.
    pub non_finite_samples: u64,
}

/// FLAC compression level written when [`SaveOptions::compression`] is unset;
/// the encoder's own default.
const DEFAULT_FLAC_COMPRESSION: u8 = 5;

/// One scored voice-activity window of a recording.
///
/// Produced by [`File::analyze`]. Windows tile the recording from the start:
/// window `i` covers `i * window / rate` to `(i + 1) * window / rate` seconds
/// of file time. A trailing remainder shorter than one window is not scored,
/// exactly as the live detector leaves a sub-window remainder unscored.
///
/// `#[non_exhaustive]`: read field by field; sealing it keeps future result
/// fields backward compatible.
#[cfg(feature = "vad")]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct VadWindow {
    /// Window start, in seconds of file time.
    pub start: f64,
    /// Window end, in seconds of file time.
    pub end: f64,
    /// Speech probability for this window (0.0 to 1.0).
    pub probability: f32,
    /// Whether `probability` meets the configured threshold.
    pub is_speech: bool,
}

/// One merged speech region of a recording.
///
/// Produced by [`File::analyze`]: consecutive speech windows whose silence
/// gaps are within the configured holdoff collapse into one segment. The
/// segment ends at the last speech window, not at the holdoff expiry.
///
/// `#[non_exhaustive]`: read field by field; sealing it keeps future result
/// fields backward compatible.
#[cfg(feature = "vad")]
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub struct Segment {
    /// Region start, in seconds of file time.
    pub start: f64,
    /// Region end, in seconds of file time.
    pub end: f64,
}

/// The whole-recording voice-activity analysis a [`File::analyze`] call
/// returns: the per-window scores and the merged speech segments.
///
/// `#[non_exhaustive]`: read field by field; sealing it keeps future result
/// fields backward compatible.
#[cfg(feature = "vad")]
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct VadReport {
    /// Per-window speech scores across the whole recording, in file order.
    pub scores: Vec<VadWindow>,
    /// Merged speech regions across the whole recording, in file order.
    pub segments: Vec<Segment>,
}

/// An offline audio source: a recording or in-memory samples conditioned
/// through the same chain as live capture.
///
/// Iterate it for conditioned [`AudioChunk`]s; the iteration is finite and
/// ends after the chain's end-of-stream tail. With VAD configured, call
/// [`analyze`](Self::analyze) (or [`analyse`](Self::analyse)) to score the
/// whole recording instead. Iteration and analysis are separate single
/// passes: each consumes the source once, so use one `File` per operation.
/// Analysis is refused once iteration has begun, rather than scoring the
/// unread remainder.
pub struct File {
    /// Interleaved source samples at `input_rate` with `input_channels`.
    source: Vec<f32>,
    /// The source's native rate in Hz (from the file's header, or the explicit
    /// `input_rate` of [`File::buffer`]).
    input_rate: u32,
    /// The source's channel count (from the file's header; 1 for buffers).
    input_channels: u16,
    /// Cursor into `source`, in interleaved samples.
    pos: usize,
    /// Whether iteration has driven the cursor. Set by [`Iterator::next`] and
    /// never by [`File::analyze`], so it distinguishes a cursor the caller
    /// moved from one the analysis pass moved itself.
    engaged: bool,
    /// The conditioning chain; `None` when the source is already mono at the
    /// target rate with no conditioning enabled (the zero-cost direct path).
    stage: Option<CaptureStage>,
    /// Delivered-output re-block buffer: conditioned samples at the target
    /// rate, interleaved at the delivered channel count, drained in
    /// [`DELIVERY_FRAMES`]-frame chunks.
    reblock: VecDeque<f32>,
    /// Whether the chain's end-of-stream tail has been drained.
    flushed: bool,
    /// Whether iteration has delivered its final chunk.
    finished: bool,
    /// The target output rate every delivered chunk carries.
    target_rate: u32,
    /// Scratch buffer holding each step's detector-feed source (the
    /// pre-conditioning signal at the target rate), reused across steps.
    scratch: Vec<f32>,
    /// The detector configuration for [`File::analyze`], kept from
    /// [`FileConfig::vad`].
    #[cfg(feature = "vad")]
    vad: Option<VadConfig>,
    /// The rate detection runs at: the target rate when it is 8000 or 16000,
    /// otherwise 16000 via `vad_resampler`.
    #[cfg(feature = "vad")]
    vad_rate: u32,
    /// Resamples the detector feed from the target rate to `vad_rate` when
    /// the two differ; `None` when the target rate is already a detector
    /// rate.
    #[cfg(feature = "vad")]
    vad_resampler: Option<PolyphaseResampler>,
    /// The detector feed: the pre-conditioning signal at `vad_rate`, appended
    /// in step with delivery and drained by [`File::vad_input`] (or consumed
    /// by [`File::analyze`]).
    #[cfg(feature = "vad")]
    vad_queue: VecDeque<f32>,
    /// Memory bound for `vad_queue`, in samples.
    #[cfg(feature = "vad")]
    vad_queue_cap: usize,
    /// The source of the detector feed
    /// ([`FileConfig::detector_source`]): the collapse
    /// [`File::push_vad_feed`] applies before samples enter the feed. Copied
    /// from the configuration at construction.
    #[cfg(feature = "vad")]
    detector_source: DetectorSource,
    /// Segment-merge holdoff in milliseconds of file time.
    #[cfg(feature = "vad")]
    vad_holdoff_ms: u32,
}

impl File {
    /// Open an audio file and prepare it for conditioning (and, with VAD
    /// configured, analysis). Reads the whole file; the input rate and channel
    /// count come from the file's own header. WAV, AIFF, AIFF-C and FLAC are
    /// read, in every encoding the reader carries. The container is
    /// identified from the bytes, so the path's extension does not decide how
    /// the file is read.
    ///
    /// # Errors
    /// - [`DecibriError::FileReadFailed`] when the file cannot be read.
    /// - [`DecibriError::AudioFormatUnsupported`] when the bytes are in a
    ///   container or encoding the reader does not carry.
    /// - [`DecibriError::AudioFileMalformed`] when the bytes are structurally
    ///   wrong.
    /// - [`DecibriError::AudioFileTruncated`] when the file ends before the
    ///   audio it declares.
    /// - [`DecibriError::SampleRateOutOfRange`] when the file's own rate is
    ///   outside 1000-384000.
    /// - [`DecibriError::FileChannelsUnsupported`],
    ///   [`DecibriError::FileChannelSelectionAmbiguous`],
    ///   [`DecibriError::FileChannelMapOutOfRange`] and
    ///   [`DecibriError::ChannelMapLengthMismatch`] exactly as
    ///   [`FileConfig::channels`] and [`FileConfig::channel_map`] document,
    ///   checked against the file's own header count.
    /// - Configuration errors exactly as [`FileConfig::validate`] reports.
    pub fn open(path: impl AsRef<Path>, config: FileConfig) -> Result<Self, DecibriError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| DecibriError::FileReadFailed {
            path: path.to_path_buf(),
            source,
        })?;
        let audio = decibri_decode::decode(&bytes)?;
        // decibri's accepted input range, which the reader does not carry. It
        // sits here rather than inside any one container's reader because
        // every container can declare a rate outside it, and the same range
        // and the same error identity apply to `File::buffer`'s explicit rate.
        let sample_rate = audio.sample_rate();
        if !(1000..=384000).contains(&sample_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
        }
        let channels = audio.channels();
        Self::from_source(audio.into_samples(), sample_rate, channels, config)
    }

    /// Alias of [`File::open`]: the bare-constructor spelling. Both forms
    /// produce the same `File`.
    ///
    /// # Errors
    /// Exactly as [`File::open`].
    pub fn new(path: impl AsRef<Path>, config: FileConfig) -> Result<Self, DecibriError> {
        Self::open(path, config)
    }

    /// Wrap in-memory samples and prepare them for conditioning (and, with
    /// VAD configured, analysis). `samples` are f32 in [-1.0, 1.0] at
    /// `input_rate`, frame-interleaved at `input_channels` (1 for mono); raw
    /// samples carry no header, so the rate and the channel count are both
    /// explicit.
    ///
    /// # Errors
    /// - [`DecibriError::SampleRateOutOfRange`] when `input_rate` is outside
    ///   1000-384000.
    /// - [`DecibriError::ChannelsOutOfRange`] when `input_channels` is 0.
    /// - [`DecibriError::BlockSizeNotFrameAligned`] when `samples` is not a
    ///   whole number of `input_channels`-channel frames.
    /// - Configuration and channel errors exactly as [`File::open`] reports.
    pub fn buffer(
        samples: Vec<f32>,
        input_rate: u32,
        input_channels: u16,
        config: FileConfig,
    ) -> Result<Self, DecibriError> {
        if !(1000..=384000).contains(&input_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
        }
        // A zero count describes no buffer at all, and a partial trailing
        // frame would rotate the channel identities of everything after it,
        // exactly as a mid-stream misalignment would on the live path. Both
        // are refused before the source is accepted, so a refused call leaves
        // nothing half-ingested.
        if input_channels == 0 {
            return Err(DecibriError::ChannelsOutOfRange);
        }
        if input_channels > 1 && !samples.len().is_multiple_of(input_channels as usize) {
            return Err(DecibriError::BlockSizeNotFrameAligned {
                samples: samples.len(),
                channels: input_channels,
            });
        }
        Self::from_source(samples, input_rate, input_channels, config)
    }

    /// Shared constructor tail: validate the configuration, build the chain
    /// for this source, and set up the detector feed when VAD is configured.
    fn from_source(
        source: Vec<f32>,
        input_rate: u32,
        input_channels: u16,
        config: FileConfig,
    ) -> Result<Self, DecibriError> {
        config.validate()?;
        let target_rate = config.sample_rate;

        // The channel map names source channels, so it is checked here, where
        // the source's own count is first known, against the same figure that
        // sets the chain's input count. Never checked in `validate()`, which
        // has no source in scope. The rules are the live capture path's, with
        // the source's header count standing where the device's report
        // stands; the errors are the file surface's own because the capture
        // messages name a device.
        if let Some(map) = &config.channel_map {
            if map.len() != config.channels as usize {
                return Err(DecibriError::ChannelMapLengthMismatch {
                    entries: map.len(),
                    channels: config.channels,
                });
            }
            for &index in map {
                if index >= input_channels {
                    return Err(DecibriError::FileChannelMapOutOfRange {
                        index,
                        available: input_channels,
                    });
                }
            }
        } else {
            if config.channels > input_channels {
                return Err(DecibriError::FileChannelsUnsupported {
                    requested: config.channels,
                    available: input_channels,
                });
            }
            if config.channels > 1 && config.channels < input_channels {
                return Err(DecibriError::FileChannelSelectionAmbiguous {
                    requested: config.channels,
                    available: input_channels,
                });
            }
        }

        // Denoise is enabled only when a model AND its path are both set,
        // exactly as on the live path.
        let denoise = config
            .denoise
            .zip(config.denoise_model_path.as_deref())
            .map(|(model, path)| (model, path, config.ort_library_path.as_deref()));
        let stage = build_capture_stage(
            input_channels,
            config.channels,
            input_rate,
            target_rate,
            Transforms {
                channel_map: config.channel_map.as_deref(),
                dc_removal: config.dc_removal,
                denoise,
                highpass: config.highpass,
                agc: config.agc,
                limiter: config.limiter,
                // The offline path has no far-end reference to cancel against, so
                // echo cancellation is not offered on it.
                #[cfg(feature = "aec")]
                aec: None,
            },
        )?;

        // Detection runs at the target rate when that rate is one the
        // detector accepts, and otherwise on a copy of the feed resampled to
        // 16000. The conditioned output always stays at the target rate; this
        // internal step never surfaces as a setting or an error.
        #[cfg(feature = "vad")]
        let (vad_rate, vad_resampler) = if config.vad.is_some() {
            match target_rate {
                8000 | 16000 => (target_rate, None),
                _ => (16000, Some(PolyphaseResampler::new(target_rate, 16000)?)),
            }
        } else {
            (16000, None)
        };

        Ok(Self {
            source,
            input_rate,
            input_channels,
            pos: 0,
            engaged: false,
            stage,
            reblock: VecDeque::new(),
            flushed: false,
            finished: false,
            target_rate,
            scratch: Vec::new(),
            #[cfg(feature = "vad")]
            vad: config.vad,
            #[cfg(feature = "vad")]
            vad_rate,
            #[cfg(feature = "vad")]
            vad_resampler,
            #[cfg(feature = "vad")]
            vad_queue: VecDeque::new(),
            #[cfg(feature = "vad")]
            vad_queue_cap: vad_rate.max(target_rate) as usize * VAD_QUEUE_BOUND_SECS,
            #[cfg(feature = "vad")]
            detector_source: config.detector_source,
            #[cfg(feature = "vad")]
            vad_holdoff_ms: config.vad_holdoff_ms,
        })
    }

    /// The target output rate every delivered chunk carries.
    pub fn sample_rate(&self) -> u32 {
        self.target_rate
    }

    /// The source's native rate, from the file's header or the explicit
    /// [`File::buffer`] input rate.
    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    /// Whether the detector feed is maintained: when VAD is configured, when
    /// the chain has a conditioning transform (the delivered output then
    /// differs from the pre-conditioning signal a detector should read), or
    /// when the delivery carries more than one channel (the feed is that
    /// signal's mono collapse, which the delivered chunk is not). The same
    /// three conditions under which the live capture path's
    /// [`crate::microphone::MicrophoneStream::detector_feed`] returns a feed
    /// distinct from the delivered chunk.
    fn feed_active(&self) -> bool {
        #[cfg(feature = "vad")]
        {
            self.vad.is_some()
                || self.stage.as_ref().is_some_and(CaptureStage::has_transform)
                || self.delivered_channels() > 1
        }
        #[cfg(not(feature = "vad"))]
        {
            false
        }
    }

    /// The rate the detector runs at (8000 or 16000), or `None` when VAD is
    /// not configured.
    ///
    /// Binding-internal plumbing, like
    /// [`crate::microphone::MicrophoneStream::vad_input`]: a binding that
    /// runs its own per-chunk detector constructs it at this rate. Not part
    /// of the stable FFI-consumer surface.
    #[cfg(feature = "vad")]
    pub fn vad_rate(&self) -> Option<u32> {
        self.vad.as_ref().map(|_| self.vad_rate)
    }

    /// Drain the detector feed accumulated since the previous drain: the
    /// pre-conditioning signal, in file order, always mono (a signal carrying
    /// more than one channel is collapsed before it enters the feed, as the
    /// configured [`FileConfig::detector_source`] directs: the frame average
    /// by default, or one named delivered channel, the live path's own
    /// collapse). With VAD configured the feed is at
    /// [`vad_rate`](Self::vad_rate); without VAD it stays at the target
    /// rate. Returns `None` when the feed is inactive (no VAD, no transform,
    /// and a mono delivery), in which case the delivered chunk already is
    /// the mono pre-conditioning signal, exactly as
    /// [`crate::microphone::MicrophoneStream::detector_feed`] contracts.
    ///
    /// Binding-internal plumbing: a binding calls this after each delivered
    /// chunk and feeds the samples (or, on `None`, the delivered chunk) to
    /// its detector. Not part of the stable FFI-consumer surface.
    #[cfg(feature = "vad")]
    pub fn vad_input(&mut self) -> Option<Vec<f32>> {
        if !self.feed_active() {
            return None;
        }
        Some(self.vad_queue.drain(..).collect())
    }

    /// Analyze the whole recording for speech and return the per-window
    /// scores and merged speech segments, timed in seconds of file time.
    ///
    /// Runs one pass over the recording, scoring the pre-conditioning signal
    /// window by window (512 samples at 16 kHz, 256 at 8 kHz), then merges
    /// consecutive speech windows whose silence gaps are within
    /// [`FileConfig::vad_holdoff_ms`] of file time. Consumes the `File`: the
    /// analysis is a single pass, separate from iteration.
    ///
    /// The scored signal is the one the channel and rate normalization
    /// produces, taken before any conditioning transform, so the report is the
    /// same whatever conditioning the [`FileConfig`] enables.
    ///
    /// Runs only on a source still at its start. Once iteration has pulled
    /// from the `File`, analysis reports [`DecibriError::FileEngaged`].
    ///
    /// # Errors
    /// - [`DecibriError::FileEngaged`] when iteration has already begun.
    /// - [`DecibriError::VadNotConfigured`] when the `File` was built without
    ///   [`FileConfig::vad`].
    /// - Detector construction and inference errors exactly as [`SileroVad`]
    ///   reports them.
    /// - Chain errors exactly as iteration reports them.
    #[cfg(feature = "vad")]
    pub fn analyze(mut self) -> Result<VadReport, DecibriError> {
        // Window and segment times are absolute, measured from the start of
        // the recording, so the pass runs only from the start of the source.
        if self.engaged {
            return Err(DecibriError::FileEngaged);
        }
        let Some(vad) = self.vad.clone() else {
            return Err(DecibriError::VadNotConfigured);
        };
        let mut detector_config = vad;
        detector_config.sample_rate = self.vad_rate;
        let threshold = detector_config.threshold;
        let window = detector_config.validate()?;
        let mut detector = SileroVad::new(detector_config)?;

        // Drive the whole source through the normalize tier, scoring the
        // detector feed one exact window at a time so each call yields exactly
        // one score. The conditioning transform does not run: analysis returns
        // the report, not the conditioned chunks.
        let mut probabilities: Vec<f32> = Vec::new();
        let mut pending: Vec<f32> = Vec::new();
        while !self.flushed {
            self.advance_feed()?;
            pending.extend(self.vad_queue.drain(..));
            let mut offset = 0;
            while pending.len() - offset >= window {
                let result = detector.process(&pending[offset..offset + window])?;
                probabilities.push(result.probability);
                offset += window;
            }
            pending.drain(..offset);
        }
        // A trailing remainder shorter than one window stays unscored,
        // exactly as the live detector's accumulator leaves it.

        let window_secs = window as f64 / f64::from(self.vad_rate);
        let scores: Vec<VadWindow> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &probability)| VadWindow {
                start: i as f64 * window_secs,
                end: (i + 1) as f64 * window_secs,
                probability,
                is_speech: probability >= threshold,
            })
            .collect();
        let segments = merge_segments(&scores, f64::from(self.vad_holdoff_ms) / 1000.0);
        Ok(VadReport { scores, segments })
    }

    /// Alias of [`File::analyze`]: the same whole-recording analysis under
    /// the international spelling. Both spellings ship in every binding.
    ///
    /// # Errors
    /// Exactly as [`File::analyze`].
    #[cfg(feature = "vad")]
    pub fn analyse(self) -> Result<VadReport, DecibriError> {
        self.analyze()
    }

    /// Write the conditioned recording to `path` as an audio file.
    ///
    /// Runs the recording once through the same conditioning pass iteration
    /// delivers, whole, and writes the result as 16-bit PCM at the target
    /// rate, frame-interleaved at the delivered channel count (the count
    /// every delivered [`AudioChunk`] carries). The container comes from the
    /// path's extension (`.wav`, `.aiff`, `.aif`, `.aifc` or `.flac`), or
    /// from [`SaveOptions::format`] when set: decibri reads a file by its
    /// content and writes one by its name. Consumes the `File`: the save is
    /// a single pass, separate from iteration and analysis.
    ///
    /// Each container's own channel ceiling applies, reported as
    /// [`DecibriError::AudioFormatUnsupported`] carrying the container
    /// layer's own text: a FLAC frame carries at most 8 channels, and a WAV
    /// `fmt ` chunk's `nBlockAlign` is a 16-bit field, so at the 16-bit
    /// samples decibri writes a WAV holds at most 32767 channels. decibri
    /// enforces no ceiling of its own on any format.
    ///
    /// A finite sample outside `[-1.0, 1.0]`, which AGC or AEC without a
    /// limiter can produce, is clamped to full scale and counted in the
    /// returned [`SaveReport`]. A non-finite sample never reaches the file:
    /// NaN becomes silence and an infinity becomes full scale, the same on
    /// every format, counted separately in the report.
    ///
    /// Runs only on a source still at its start. Once iteration has pulled
    /// from the `File`, saving reports [`DecibriError::FileEngaged`].
    ///
    /// # Errors
    /// - [`DecibriError::FileEngaged`] when iteration has already begun.
    /// - [`DecibriError::FlacCompressionOutOfRange`] when
    ///   [`SaveOptions::compression`] is outside 0-8.
    /// - [`DecibriError::AudioFormatUnsupported`] when neither the path's
    ///   extension nor [`SaveOptions::format`] names a format decibri
    ///   writes.
    /// - [`DecibriError::FileWriteFailed`] when the encoded file cannot be
    ///   written to disk.
    /// - Chain errors exactly as iteration reports them.
    pub fn save(
        mut self,
        path: impl AsRef<Path>,
        options: SaveOptions,
    ) -> Result<SaveReport, DecibriError> {
        let path = path.as_ref();
        // The engaged state is reported first, before any argument is
        // interpreted, matching the check order of `analyze`.
        if self.engaged {
            return Err(DecibriError::FileEngaged);
        }
        let compression = match options.compression {
            None => DEFAULT_FLAC_COMPRESSION,
            Some(level) if level <= 8 => level,
            Some(_) => return Err(DecibriError::FlacCompressionOutOfRange),
        };
        let format = match options.format {
            Some(format) => format,
            None => SaveFormat::from_path(path)?,
        };

        // The same single pass iteration delivers, taken whole: every
        // conditioned sample including the chain's end-of-stream tail.
        let mut samples: Vec<f32> = Vec::new();
        loop {
            self.advance()?;
            samples.extend(self.reblock.drain(..));
            if self.flushed && self.reblock.is_empty() {
                break;
            }
        }

        let non_finite_samples = replace_non_finite(&mut samples);
        let clipped_samples = clamp_overscale(&mut samples);

        // The written layout is the delivered layout: the same rate and the
        // same interleaved channel count iteration stamps on every chunk.
        // Each writer refuses a count its container cannot carry, with the
        // container layer's own message forwarded unchanged.
        let spec = decibri_decode::AudioSpec::new(self.target_rate, self.delivered_channels());
        let mut bytes = Vec::new();
        match format {
            SaveFormat::Wav => {
                decibri_decode::WavWriter::new(spec, decibri_decode::WavCodec::PcmI16)
                    .write(&samples, &mut bytes)?;
            }
            SaveFormat::Aiff => {
                decibri_decode::AiffWriter::new(spec, decibri_decode::AiffCodec::PcmI16)
                    .write(&samples, &mut bytes)?;
            }
            SaveFormat::Flac => {
                decibri_decode::FlacWriter::new(spec, 16)
                    .with_level(compression)
                    .write(&samples, &mut bytes)?;
            }
        }
        std::fs::write(path, &bytes).map_err(|source| DecibriError::FileWriteFailed {
            path: path.to_path_buf(),
            source,
        })?;
        Ok(SaveReport {
            clipped_samples,
            non_finite_samples,
        })
    }

    /// The next source block's range, moving the cursor past it; `None` once
    /// the source is exhausted. The one place either pass advances the cursor,
    /// so both step the source in the same blocks.
    fn next_range(&mut self) -> Option<std::ops::Range<usize>> {
        if self.pos >= self.source.len() {
            return None;
        }
        let feed = FEED_FRAMES * self.input_channels as usize;
        let end = (self.pos + feed).min(self.source.len());
        let range = self.pos..end;
        self.pos = end;
        Some(range)
    }

    /// Number of interleaved samples in one full delivered chunk:
    /// [`DELIVERY_FRAMES`] frames at the delivered channel count.
    fn delivery_samples(&self) -> usize {
        DELIVERY_FRAMES * self.delivered_channels() as usize
    }

    /// Feed source blocks through the chain until the re-block buffer can
    /// deliver one chunk or the source is exhausted and flushed. One step of
    /// the iteration pass, which delivers the conditioned audio.
    fn advance(&mut self) -> Result<(), DecibriError> {
        let delivery_samples = self.delivery_samples();
        while self.reblock.len() < delivery_samples && !self.flushed {
            match self.next_range() {
                Some(range) => self.ingest(range)?,
                None => {
                    self.flush_chain()?;
                    self.flushed = true;
                }
            }
        }
        Ok(())
    }

    /// Feed one source block through the normalize tier and append the
    /// detector feed. One step of the analysis pass, which returns the report
    /// rather than the conditioned audio, so the conditioning transform is not
    /// run.
    #[cfg(feature = "vad")]
    fn advance_feed(&mut self) -> Result<(), DecibriError> {
        match self.next_range() {
            Some(range) => self.ingest_feed(range)?,
            None => {
                self.flush_feed()?;
                self.flushed = true;
            }
        }
        Ok(())
    }

    /// Run one source block through the chain, appending the conditioned
    /// output to the re-block buffer and the pre-conditioning signal to the
    /// detector feed. Mirrors the live path's ingest: the detector reads the
    /// signal captured at the normalize/transform boundary when a transform
    /// is present, and the delivered block already is that signal otherwise.
    fn ingest(&mut self, range: std::ops::Range<usize>) -> Result<(), DecibriError> {
        let want_feed = self.feed_active();
        self.scratch.clear();
        match &mut self.stage {
            None => {
                self.reblock.extend(&self.source[range.clone()]);
                if want_feed {
                    self.scratch.extend_from_slice(&self.source[range]);
                }
            }
            Some(stage) => {
                let has_transform = stage.has_transform();
                let out = stage.run(&self.source[range])?;
                self.reblock.extend(out.iter().copied());
                if want_feed {
                    if has_transform {
                        self.scratch.extend_from_slice(stage.tap());
                    } else {
                        self.scratch.extend_from_slice(out);
                    }
                }
            }
        }
        #[cfg(feature = "vad")]
        if want_feed {
            self.push_vad_feed()?;
        }
        Ok(())
    }

    /// Run one source block through the normalize tier, appending the result
    /// to the detector feed. The analysis counterpart of
    /// [`ingest`](Self::ingest): the feed is the post-normalize signal on
    /// either path, so the transform tier stays unrun here.
    #[cfg(feature = "vad")]
    fn ingest_feed(&mut self, range: std::ops::Range<usize>) -> Result<(), DecibriError> {
        self.scratch.clear();
        match &mut self.stage {
            None => self.scratch.extend_from_slice(&self.source[range]),
            Some(stage) => {
                let normalized = stage.run_normalize(&self.source[range])?;
                self.scratch.extend_from_slice(normalized);
            }
        }
        self.push_vad_feed()?;
        Ok(())
    }

    /// Drain the normalize tier's end-of-stream tail into the detector feed,
    /// once, when the source is exhausted. The analysis counterpart of
    /// [`flush_chain`](Self::flush_chain).
    #[cfg(feature = "vad")]
    fn flush_feed(&mut self) -> Result<(), DecibriError> {
        self.scratch.clear();
        if let Some(stage) = &mut self.stage {
            let tail = stage.flush_normalize()?;
            self.scratch.extend_from_slice(tail);
        }
        self.push_vad_feed()?;
        // Drain the detector-feed resampler's own tail so trailing windows are
        // scored rather than stranded.
        if let Some(resampler) = &mut self.vad_resampler {
            let mut tail = Vec::new();
            resampler.flush(&mut tail);
            self.vad_queue.extend(tail);
            self.cap_vad_queue();
        }
        Ok(())
    }

    /// Drain the chain's end-of-stream tail into the re-block buffer and the
    /// detector feed, once, when the source is exhausted. The offline
    /// analogue of the live close-path flush: the resampler's group-delay
    /// tail and any conditioning-stage tail are delivered rather than
    /// dropped, and the detector feed stays aligned through the close.
    fn flush_chain(&mut self) -> Result<(), DecibriError> {
        let want_feed = self.feed_active();
        self.scratch.clear();
        if let Some(stage) = &mut self.stage {
            let mut tail = Vec::new();
            stage.flush(&mut tail)?;
            self.reblock.extend(&tail);
            if want_feed {
                if stage.has_transform() {
                    self.scratch.extend_from_slice(stage.tap());
                } else {
                    self.scratch.extend_from_slice(&tail);
                }
            }
        }
        #[cfg(feature = "vad")]
        if want_feed {
            self.push_vad_feed()?;
            // Drain the detector-feed resampler's own tail so trailing
            // windows are scored rather than stranded.
            if let Some(resampler) = &mut self.vad_resampler {
                let mut tail = Vec::new();
                resampler.flush(&mut tail);
                self.vad_queue.extend(tail);
                self.cap_vad_queue();
            }
        }
        Ok(())
    }

    /// Append `scratch` (the pre-conditioning signal at the target rate) to
    /// the detector feed, collapsing it to mono first when it carries more
    /// than one channel, resampling to the detector rate when the two
    /// differ, and enforce the feed's memory bound.
    ///
    /// The collapse is the live path's own, directed by the configured
    /// [`FileConfig::detector_source`] exactly as
    /// [`crate::microphone::MicrophoneStream::detector_feed`] is directed by
    /// its own: each interleaved frame becomes one sample, the average of its
    /// channels ([`crate::sample::downmix_to_mono`], the default) or the
    /// sample of one named delivered channel
    /// ([`crate::sample::select_channel`]), at the chain's own post-normalize
    /// channel count, the count the scratch signal is interleaved at. The
    /// detector and the feed's resampler both read mono, so the collapse
    /// precedes both. At one channel the samples take the untouched path
    /// below, byte for byte.
    ///
    /// The detector-feed resampler rejects a block that arrives after its own
    /// flush. Every caller runs before that flush, so the rejection is
    /// propagated rather than assumed away.
    #[cfg(feature = "vad")]
    fn push_vad_feed(&mut self) -> Result<(), DecibriError> {
        if self.scratch.is_empty() {
            return Ok(());
        }
        let feed_channels = self
            .stage
            .as_ref()
            .map_or(self.input_channels, CaptureStage::tap_channels);
        let collapsed;
        let feed: &[f32] = if feed_channels > 1 {
            collapsed = match self.detector_source {
                DetectorSource::Average => {
                    crate::sample::downmix_to_mono(&self.scratch, feed_channels)
                }
                DetectorSource::Channel(index) => {
                    crate::sample::select_channel(&self.scratch, feed_channels, index)
                }
            };
            &collapsed
        } else {
            &self.scratch
        };
        match &mut self.vad_resampler {
            Some(resampler) => {
                let mut out = Vec::new();
                resampler.process(feed, &mut out)?;
                self.vad_queue.extend(out);
            }
            None => self.vad_queue.extend(feed.iter().copied()),
        }
        self.cap_vad_queue();
        Ok(())
    }

    /// Drop the oldest detector-feed samples beyond the memory bound. Only
    /// trips when VAD is configured but the feed is never drained; an
    /// actively draining consumer (a binding, or analysis) never reaches it.
    #[cfg(feature = "vad")]
    fn cap_vad_queue(&mut self) {
        if self.vad_queue.len() > self.vad_queue_cap {
            let excess = self.vad_queue.len() - self.vad_queue_cap;
            self.vad_queue.drain(..excess);
        }
    }

    /// The channel count the delivered chunks are interleaved at.
    ///
    /// The chain's own resolved output when there is a chain, and the source's
    /// count unchanged when there is not: with no stage in the way, what the
    /// decoder produced is what the consumer receives. Read from the chain
    /// rather than named here, so the count stamped on a chunk and the count
    /// the stages actually produce cannot drift apart.
    fn delivered_channels(&self) -> u16 {
        self.stage
            .as_ref()
            .map_or(self.input_channels, CaptureStage::output_channels)
    }
}

impl Iterator for File {
    type Item = Result<AudioChunk, DecibriError>;

    /// Deliver the next conditioned chunk: [`DELIVERY_FRAMES`] frames of
    /// interleaved samples at the delivered channel count and the target
    /// rate, with a possibly shorter final chunk once the chain's
    /// end-of-stream tail has been drained. Every chunk is a whole number of
    /// frames: the chunk boundary never cuts a frame, so the channel
    /// identities cannot rotate across chunks. Returns `None` after the
    /// final chunk; an error ends the iteration.
    fn next(&mut self) -> Option<Self::Item> {
        // The one place iteration reaches the source: the adapters and the
        // provided methods all arrive here. The mark precedes the finished
        // check and any delivery, so a call that moves the cursor and returns
        // nothing still counts.
        self.engaged = true;
        if self.finished {
            return None;
        }
        if let Err(e) = self.advance() {
            self.finished = true;
            return Some(Err(e));
        }
        let take = self.reblock.len().min(self.delivery_samples());
        // The chain emits whole frames (its planar runs are equal-length by
        // construction) and the full-chunk size is a whole number of frames,
        // so the take is one too; a partial frame here would mean the chain
        // itself broke that invariant.
        debug_assert!(
            take.is_multiple_of(self.delivered_channels() as usize),
            "the delivered take must be a whole number of frames"
        );
        if take == 0 {
            self.finished = true;
            return None;
        }
        let data: Vec<f32> = self.reblock.drain(..take).collect();
        if self.flushed && self.reblock.is_empty() {
            self.finished = true;
        }
        let channels = self.delivered_channels();
        Some(Ok(AudioChunk {
            data,
            sample_rate: self.target_rate,
            channels,
        }))
    }
}

/// Replace every non-finite sample in place, returning how many were
/// replaced: NaN becomes silence (0.0) and an infinity becomes full scale
/// (1.0 or -1.0), the values decode's integer encodings produce for the same
/// input. Applied before any writer runs so every format writes the same
/// audio for the same samples; a float encoding would otherwise carry NaN
/// and infinity into the file verbatim.
fn replace_non_finite(samples: &mut [f32]) -> u64 {
    let mut replaced = 0u64;
    for sample in samples.iter_mut() {
        if !sample.is_finite() {
            *sample = if sample.is_nan() {
                0.0
            } else if *sample > 0.0 {
                1.0
            } else {
                -1.0
            };
            replaced += 1;
        }
    }
    replaced
}

/// Clamp every sample outside `[-1.0, 1.0]` to full scale in place,
/// returning how many were clamped. Runs after [`replace_non_finite`], so
/// the count covers finite overshoot only and the writer's own clamp is a
/// no-op. Kept separate from the non-finite replacement: the clamp belongs
/// to integer encodings, while the replacement applies to every encoding,
/// and a float target must be able to take the replacement without the
/// clamp.
fn clamp_overscale(samples: &mut [f32]) -> u64 {
    let mut clipped = 0u64;
    for sample in samples.iter_mut() {
        if *sample > 1.0 {
            *sample = 1.0;
            clipped += 1;
        } else if *sample < -1.0 {
            *sample = -1.0;
            clipped += 1;
        }
    }
    clipped
}

/// Merge consecutive speech windows into segments: a silence gap within
/// `holdoff_secs` of file time keeps the current segment open; a longer gap
/// closes it. Each segment ends at its last speech window, so trailing
/// holdoff silence is never inside a segment.
#[cfg(feature = "vad")]
fn merge_segments(scores: &[VadWindow], holdoff_secs: f64) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut current: Option<(f64, f64)> = None;
    for w in scores.iter().filter(|w| w.is_speech) {
        current = Some(match current {
            None => (w.start, w.end),
            Some((start, end)) => {
                if w.start - end <= holdoff_secs {
                    (start, w.end)
                } else {
                    segments.push(Segment { start, end });
                    (w.start, w.end)
                }
            }
        });
    }
    if let Some((start, end)) = current {
        segments.push(Segment { start, end });
    }
    segments
}

#[cfg(test)]
mod tests {
    use super::*;
    // Only the test WAV builders convert samples to bytes; the reader the
    // `File` path uses does its own conversion.
    use crate::sample;
    // The module-level resampler import is `vad`-gated; the reference
    // resampler in the buffer tests must resolve under `capture` alone.
    use decibri_resampler::{PolyphaseResampler, Resampler};

    // ── Building input files by hand ─────────────────────────────────────
    //
    // Every input below is assembled here rather than committed, so the file's
    // shape reads as a statement about the format instead of as a blob, and
    // nothing binary enters the crate. Each builder names its fields in the
    // order and the byte order the specification fixes them in.

    /// A RIFF/WAVE file: the 12-byte `RIFF` header naming the `WAVE` form,
    /// then a `fmt ` chunk carrying `fmt_body` and a `data` chunk carrying
    /// `payload`. RIFF pads a chunk body to an even length and does not count
    /// the pad byte in the chunk size, which both chunks honour here.
    fn riff_wave(fmt_body: &[u8], payload: &[u8]) -> Vec<u8> {
        fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut out = Vec::with_capacity(8 + body.len() + 1);
            out.extend_from_slice(id);
            out.extend_from_slice(&(body.len() as u32).to_le_bytes());
            out.extend_from_slice(body);
            if !body.len().is_multiple_of(2) {
                out.push(0);
            }
            out
        }
        let mut form = b"WAVE".to_vec();
        form.extend_from_slice(&chunk(b"fmt ", fmt_body));
        form.extend_from_slice(&chunk(b"data", payload));

        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(form.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&form);
        bytes
    }

    /// The 16-byte `fmt ` chunk body every WAVE file carries: `wFormatTag`,
    /// `nChannels`, `nSamplesPerSec`, `nAvgBytesPerSec`, `nBlockAlign` and
    /// `wBitsPerSample`, in that order, all little-endian.
    fn wave_fmt(tag: u16, channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let block_align = channels * bits.div_ceil(8);
        let mut body = Vec::with_capacity(16);
        body.extend_from_slice(&tag.to_le_bytes());
        body.extend_from_slice(&channels.to_le_bytes());
        body.extend_from_slice(&rate.to_le_bytes());
        body.extend_from_slice(&(rate * u32::from(block_align)).to_le_bytes());
        body.extend_from_slice(&block_align.to_le_bytes());
        body.extend_from_slice(&bits.to_le_bytes());
        body
    }

    /// The 40-byte WAVE_FORMAT_EXTENSIBLE `fmt ` body: the 16 plain fields
    /// with `wFormatTag` 0xFFFE, then the extension, `cbSize` of 22,
    /// `wValidBitsPerSample`, `dwChannelMask` and the 16-byte SubFormat GUID.
    /// The GUID, not the tag, names the encoding.
    fn wave_fmt_extensible(subformat: [u8; 16], channels: u16, rate: u32, bits: u16) -> Vec<u8> {
        let mut body = wave_fmt(0xFFFE, channels, rate, bits);
        body.extend_from_slice(&22u16.to_le_bytes()); // cbSize
        body.extend_from_slice(&bits.to_le_bytes()); // wValidBitsPerSample
        body.extend_from_slice(&0u32.to_le_bytes()); // dwChannelMask
        body.extend_from_slice(&subformat);
        body
    }

    /// KSDATAFORMAT_SUBTYPE_PCM, the SubFormat GUID an extensible container
    /// carries for integer PCM, in file byte order.
    const SUBTYPE_PCM: [u8; 16] = [
        0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B,
        0x71,
    ];

    /// An EA IFF 85 `FORM` file of type `AIFF` or `AIFC`: the 12-byte header,
    /// the `FVER` chunk on the AIFF-C path, then `COMM` and `SSND`. IFF is
    /// big-endian throughout and pads a chunk body to an even length without
    /// counting the pad byte, the mirror of RIFF's rule.
    ///
    /// `SSND` does not begin with samples: its body opens with the 32-bit
    /// `offset` and `blockSize` fields, both zero here, and the sample data
    /// follows them.
    fn iff_form(form: &[u8; 4], comm_body: &[u8], payload: &[u8]) -> Vec<u8> {
        fn chunk(id: &[u8; 4], body: &[u8]) -> Vec<u8> {
            let mut out = Vec::with_capacity(8 + body.len() + 1);
            out.extend_from_slice(id);
            out.extend_from_slice(&(body.len() as u32).to_be_bytes());
            out.extend_from_slice(body);
            if !body.len().is_multiple_of(2) {
                out.push(0);
            }
            out
        }
        let mut ssnd = Vec::with_capacity(8 + payload.len());
        ssnd.extend_from_slice(&0u32.to_be_bytes()); // offset
        ssnd.extend_from_slice(&0u32.to_be_bytes()); // blockSize
        ssnd.extend_from_slice(payload);

        let mut body = form.to_vec();
        if form == b"AIFC" {
            // AIFC Version 1, the one timestamp the format has ever carried.
            body.extend_from_slice(&chunk(b"FVER", &0xA280_5140u32.to_be_bytes()));
        }
        body.extend_from_slice(&chunk(b"COMM", comm_body));
        body.extend_from_slice(&chunk(b"SSND", &ssnd));

        let mut bytes = b"FORM".to_vec();
        bytes.extend_from_slice(&(body.len() as u32).to_be_bytes());
        bytes.extend_from_slice(&body);
        bytes
    }

    /// A `COMM` chunk body: `numChannels`, `numSampleFrames`, `sampleSize`
    /// and the 80-bit `sampleRate`, all big-endian. An AIFF-C file appends
    /// the compression four-CC and a `compressionName` pascal string, empty
    /// here, padded to an even total.
    fn aiff_comm(
        channels: u16,
        frames: u32,
        bits: u16,
        rate: u32,
        compression: Option<&[u8; 4]>,
    ) -> Vec<u8> {
        let mut body = Vec::with_capacity(24);
        body.extend_from_slice(&channels.to_be_bytes());
        body.extend_from_slice(&frames.to_be_bytes());
        body.extend_from_slice(&bits.to_be_bytes());
        body.extend_from_slice(&aiff_extended_rate(rate));
        if let Some(four_cc) = compression {
            body.extend_from_slice(four_cc);
            body.push(0); // compressionName length
            body.push(0); // pad to an even total
        }
        body
    }

    /// The 80-bit IEEE 754 extended-precision float `COMM` stores its sample
    /// rate in: the sign bit clear, a 15-bit exponent biased by 16383, and a
    /// 64-bit mantissa whose leading integer bit is explicit rather than
    /// implied, which is what distinguishes this format from binary32 and
    /// binary64.
    ///
    /// For an integer rate, the exponent is the position of its highest set
    /// bit and the mantissa is the rate shifted up so that bit lands in the
    /// mantissa's top position. 16000 Hz is 2^13 scaled, so it encodes as
    /// exponent 16383 + 13 and mantissa 0xFA00_0000_0000_0000.
    fn aiff_extended_rate(rate: u32) -> [u8; 10] {
        assert!(rate > 0, "a zero rate has no normalized encoding");
        let top_bit = 31 - rate.leading_zeros();
        let exponent = (16383 + top_bit) as u16;
        let mantissa = u64::from(rate) << (63 - top_bit);
        let mut out = [0u8; 10];
        out[0..2].copy_from_slice(&exponent.to_be_bytes());
        out[2..10].copy_from_slice(&mantissa.to_be_bytes());
        out
    }

    /// Interleave a mono signal into a synthetic WAV byte vector: 16-bit PCM
    /// (format 1) or 32-bit float (format 3), any rate and channel count.
    fn wav_bytes(format: u16, channels: u16, rate: u32, samples: &[f32]) -> Vec<u8> {
        let payload = match format {
            1 => sample::f32_to_i16_le_bytes(samples),
            3 => sample::f32_to_f32_le_bytes(samples),
            _ => unreachable!("test formats are 1 and 3"),
        };
        let bits: u16 = if format == 1 { 16 } else { 32 };
        riff_wave(&wave_fmt(format, channels, rate, bits), &payload)
    }

    /// Write `bytes` to a uniquely named file under the system temp
    /// directory, hand the path to `read_it`, and remove the file. `File`
    /// opens a path, so an input assembled here has to reach the filesystem
    /// to be read at all.
    fn with_file<T>(name: &str, bytes: &[u8], read_it: impl FnOnce(&Path) -> T) -> T {
        let path = std::env::temp_dir().join(format!("decibri-decode-{name}"));
        std::fs::write(&path, bytes).expect("the temp input should be writable");
        let out = read_it(&path);
        std::fs::remove_file(&path).ok();
        out
    }

    /// Open assembled bytes through `File` with the given configuration and
    /// collect every delivered sample.
    fn decode_bytes(name: &str, bytes: &[u8], config: FileConfig) -> Vec<f32> {
        with_file(name, bytes, |path| {
            collect(File::open(path, config).unwrap_or_else(|e| panic!("{name} should open: {e}")))
        })
    }

    /// Open assembled bytes through `File` and return the failure.
    fn decode_error(name: &str, bytes: &[u8]) -> DecibriError {
        with_file(name, bytes, |path| {
            File::open(path, FileConfig::default())
                .err()
                .unwrap_or_else(|| panic!("{name} should be rejected"))
        })
    }

    /// Interleaved samples as big-endian 16-bit two's-complement words, the
    /// payload a plain AIFF `SSND` chunk carries. Big-endian is the format's
    /// default and the reason AIFF is worth testing separately from WAV.
    fn be_i16_bytes(samples: &[f32]) -> Vec<u8> {
        let mut out = Vec::with_capacity(samples.len() * 2);
        for &s in samples {
            out.extend_from_slice(&((s * 32768.0) as i16).to_be_bytes());
        }
        out
    }

    /// A short mono sine at the given rate, amplitude 0.5.
    fn sine(rate: u32, seconds: f64) -> Vec<f32> {
        let count = (f64::from(rate) * seconds) as usize;
        (0..count)
            .map(|i| 0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / f64::from(rate)).sin())
            .map(|s| s as f32)
            .collect()
    }

    /// Collect every conditioned sample a `File` iteration delivers,
    /// asserting each chunk carries the target rate and mono channel count.
    fn collect(file: File) -> Vec<f32> {
        let rate = file.sample_rate();
        let mut out = Vec::new();
        for chunk in file {
            let chunk = chunk.expect("iteration should not error");
            assert_eq!(chunk.sample_rate, rate);
            assert_eq!(chunk.channels, 1);
            out.extend(chunk.data);
        }
        out
    }

    // ── Conditioning: the pure-audio path ───────────────────────────────

    /// A mono source already at the target rate with no conditioning takes
    /// the direct path: the delivered samples equal the input exactly, in
    /// full-size chunks with one short final chunk.
    #[test]
    fn buffer_passthrough_is_byte_identical() {
        let input = sine(16000, 0.5); // 8000 samples
        let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
        let mut sizes = Vec::new();
        let mut out = Vec::new();
        for chunk in file {
            let chunk = chunk.unwrap();
            sizes.push(chunk.data.len());
            out.extend(chunk.data);
        }
        assert_eq!(out, input);
        assert_eq!(sizes, vec![1600, 1600, 1600, 1600, 1600]);
    }

    /// The buffer path resamples an input rate to the target rate, and the
    /// delivered output matches driving the resampler directly, including
    /// its flushed group-delay tail.
    #[test]
    fn buffer_resamples_and_flushes_tail() {
        let input = sine(48000, 0.25);
        let file = File::buffer(input.clone(), 48000, 1, FileConfig::default()).unwrap();
        let out = collect(file);

        let mut resampler = PolyphaseResampler::new(48000, 16000).unwrap();
        let mut expected = Vec::new();
        for block in input.chunks(FEED_FRAMES) {
            resampler.process(block, &mut expected).unwrap();
        }
        resampler.flush(&mut expected);
        assert_eq!(out, expected);
    }

    /// Conditioning through the `File` equals driving the shared chain
    /// directly with the same feed pattern: one chain, two drivers.
    #[test]
    fn conditioning_matches_direct_chain_drive() {
        let input: Vec<f32> = sine(16000, 0.5).iter().map(|s| s + 0.25).collect();
        let config = FileConfig {
            dc_removal: true,
            ..Default::default()
        };
        let file = File::buffer(input.clone(), 16000, 1, config).unwrap();
        let out = collect(file);

        let mut chain = build_capture_stage(
            1,
            1,
            16000,
            16000,
            Transforms {
                dc_removal: true,
                ..Default::default()
            },
        )
        .unwrap()
        .expect("dc removal builds a chain");
        let mut expected = Vec::new();
        for block in input.chunks(FEED_FRAMES) {
            expected.extend_from_slice(chain.run(block).unwrap());
        }
        chain.flush(&mut expected).unwrap();
        assert_eq!(out, expected);
    }

    /// `File::open` and the `File::new` alias produce identical output for
    /// the same path and configuration.
    #[test]
    fn open_and_new_are_equivalent() {
        let samples = sine(16000, 0.3);
        let bytes = wav_bytes(1, 1, 16000, &samples);
        let path = std::env::temp_dir().join("decibri-file-open-vs-new.wav");
        std::fs::write(&path, &bytes).unwrap();

        let opened = collect(File::open(&path, FileConfig::default()).unwrap());
        let newed = collect(File::new(&path, FileConfig::default()).unwrap());
        std::fs::remove_file(&path).ok();
        assert_eq!(opened, newed);
        assert!(!opened.is_empty());
    }

    /// The WAV path reads format, channels, and rate from the header: a
    /// stereo float WAV downmixes to mono and keeps sample values.
    #[test]
    fn open_reads_header_and_downmixes() {
        let mono = sine(16000, 0.2);
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s, s]).collect();
        let bytes = wav_bytes(3, 2, 16000, &stereo);
        let path = std::env::temp_dir().join("decibri-file-stereo-float.wav");
        std::fs::write(&path, &bytes).unwrap();

        let file = File::open(&path, FileConfig::default()).unwrap();
        assert_eq!(file.input_rate(), 16000);
        let out = collect(file);
        std::fs::remove_file(&path).ok();
        // Averaging two identical channels reproduces the mono signal.
        for (a, b) in out.iter().zip(mono.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        assert_eq!(out.len(), mono.len());
    }

    /// An empty source delivers no chunks and ends the iteration cleanly.
    #[test]
    fn empty_buffer_ends_immediately() {
        let file = File::buffer(Vec::new(), 16000, 1, FileConfig::default()).unwrap();
        assert_eq!(collect(file).len(), 0);
    }

    /// An empty source at a rate that builds a resample stage delivers no
    /// samples either: the close-path flush of a chain that processed nothing
    /// appends nothing, so no audio is invented at the head of the stream.
    #[test]
    fn empty_buffer_at_a_resampled_rate_delivers_nothing() {
        let config = FileConfig {
            sample_rate: 16000,
            ..FileConfig::default()
        };
        let file = File::buffer(Vec::new(), 48000, 1, config).unwrap();
        assert_eq!(
            collect(file).len(),
            0,
            "an unfed resample chain contributes no samples at close"
        );
    }

    /// An empty source with denoise enabled delivers no samples either: the
    /// close-path flush of a chain that processed nothing appends nothing, so a
    /// recording that held no audio yields none.
    #[cfg(feature = "denoise")]
    #[test]
    fn empty_buffer_with_denoise_delivers_nothing() {
        let config = FileConfig {
            denoise: Some(DenoiseModel::FastEnhancerT),
            denoise_model_path: Some(denoise_model_path()),
            ..FileConfig::default()
        };
        let file = File::buffer(Vec::new(), 16000, 1, config).unwrap();
        assert_eq!(
            collect(file).len(),
            0,
            "an unfed denoise chain contributes no samples at close"
        );
    }

    /// A source holding a fraction of one analysis frame still comes back out:
    /// 100 samples produce no output through the framing, so every delivered
    /// sample arrives via the close-path flush. The regression that the unfed
    /// gate must not break, on the `File` path.
    #[cfg(feature = "denoise")]
    #[test]
    fn short_buffer_with_denoise_still_delivers_its_tail() {
        let config = FileConfig {
            denoise: Some(DenoiseModel::FastEnhancerT),
            denoise_model_path: Some(denoise_model_path()),
            ..FileConfig::default()
        };
        // 100 samples at 16 kHz: well under the 512-sample analysis window.
        let source: Vec<f32> = (0..100)
            .map(|i| 0.5 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16_000.0).sin())
            .collect();
        let file = File::buffer(source, 16000, 1, config).unwrap();
        let out = collect(file);
        // Left-pad (256) + 100 real + one window of padding (512) = 868 samples,
        // which yields two whole 256-sample hops.
        assert_eq!(out.len(), 512, "the real tail is delivered in full");
        assert!(
            out.iter().any(|&s| s != 0.0),
            "the delivered tail carries the audio the source held"
        );
    }

    // ── Constructor and parse errors ─────────────────────────────────────

    /// A missing file reports the typed read failure with its path.
    #[test]
    fn open_missing_file_reports_read_failure() {
        let err = File::open("no-such-file.wav", FileConfig::default())
            .err()
            .expect("a missing file should fail to open");
        assert!(matches!(err, DecibriError::FileReadFailed { .. }));
    }

    // ── The file reader ──────────────────────────────────────────────────
    //
    // Inputs here are assembled byte by byte from the specifications rather
    // than committed, so a reader sees what makes a file a 999 Hz file or a
    // half-frame file without decoding a blob, and the crate ships no binary
    // for them. The one exception is `libflac-speech-clip-16k.flac`, which is
    // real encoder output for the reason recorded on `libflac_clip_path`.

    /// The one committed input on this path: 1600 samples of the golden
    /// recording's own speech, encoded by libFLAC.
    ///
    /// It is committed because FLAC is the one format here that cannot be
    /// assembled by hand into anything a real encoder produces. Every FLAC
    /// frame a real encoder emits is predictor-and-residual coded under a
    /// per-partition Rice parameter search; a hand-built one would carry
    /// verbatim subframes and exercise a path no encoder takes, which would
    /// read as evidence without being any.
    fn libflac_clip_path() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("assets")
            .join("libflac-speech-clip-16k.flac")
    }

    /// The span of the golden recording the committed FLAC encodes. Speech,
    /// not silence: the clip runs from the onset of the first utterance.
    const CLIP: std::ops::Range<usize> = 22400..24000;

    /// The golden recording's samples, read with a 16-bit reader written
    /// here rather than through the reader under test.
    ///
    /// The container comparisons below assert that three encodings deliver
    /// one set of samples, so the set they are compared against must not come
    /// from the thing being compared. This walks the chunk list, checks the
    /// format is the mono 16-bit PCM the fixture is, and divides by 32768,
    /// which is the whole of it.
    fn golden_samples() -> Vec<f32> {
        let bytes = std::fs::read(golden_wav_path()).expect("the golden recording reads");
        assert_eq!(&bytes[0..4], b"RIFF", "the golden recording is RIFF");
        assert_eq!(&bytes[8..12], b"WAVE", "the golden recording is WAVE");

        let le16 = |at: usize| u16::from_le_bytes([bytes[at], bytes[at + 1]]);
        let le32 = |at: usize| {
            u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
        };

        let (mut tag, mut channels, mut rate, mut bits) = (0u16, 0u16, 0u32, 0u16);
        let mut data: Option<(usize, usize)> = None;
        let mut pos = 12usize;
        while pos + 8 <= bytes.len() {
            let id = &bytes[pos..pos + 4];
            let size = le32(pos + 4) as usize;
            let body = pos + 8;
            if id == b"fmt " {
                tag = le16(body);
                channels = le16(body + 2);
                rate = le32(body + 4);
                bits = le16(body + 14);
            } else if id == b"data" && data.is_none() {
                data = Some((body, size));
            }
            pos = body + size + (size & 1);
        }
        assert_eq!(
            (tag, channels, rate, bits),
            (1, 1, 16000, 16),
            "the golden recording is mono 16-bit PCM at 16 kHz"
        );
        let (at, len) = data.expect("the golden recording has a data chunk");
        bytes[at..at + len]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|c| f32::from(i16::from_le_bytes(*c)) / 32768.0)
            .collect()
    }

    /// The samples the committed FLAC encodes, taken from the golden
    /// recording by the reader above.
    fn clip_samples() -> Vec<f32> {
        golden_samples()[CLIP].to_vec()
    }

    /// The strongest single statement available about the reader: one span of
    /// real speech, carried by three containers, delivers one set of samples.
    ///
    /// The WAV and the AIFF are assembled here from the specifications; the
    /// FLAC is libFLAC's own output. All three are lossless at the source's
    /// 16 bits and all three are compared against the samples a reader
    /// written here pulls out of the golden recording, so equality is exact
    /// and nothing in the comparison comes from the reader under test.
    ///
    /// Regression: a container reader that offsets, reorders, rescales or
    /// drops samples; a byte-order fault on the big-endian container; and a
    /// FLAC reader that disagrees with the encoder everybody else uses.
    #[test]
    fn one_span_of_speech_in_three_containers_decodes_to_one_set_of_samples() {
        let reference = clip_samples();
        assert_eq!(reference.len(), 1600);
        assert!(
            reference.iter().any(|&s| s.abs() > 0.15),
            "the clip carries speech, not silence"
        );

        let wav = wav_bytes(1, 1, 16000, &reference);
        assert_eq!(
            decode_bytes("identity.wav", &wav, FileConfig::default()),
            reference,
            "WAV"
        );

        let aiff = iff_form(
            b"AIFF",
            &aiff_comm(1, reference.len() as u32, 16, 16000, None),
            &be_i16_bytes(&reference),
        );
        assert_eq!(
            decode_bytes("identity.aiff", &aiff, FileConfig::default()),
            reference,
            "AIFF"
        );

        let flac = collect(File::open(libflac_clip_path(), FileConfig::default()).unwrap());
        assert_eq!(flac, reference, "FLAC");
    }

    /// The container comes from the bytes, not from the path: the committed
    /// FLAC under a `.wav` name decodes as the FLAC it is.
    ///
    /// Regression: dispatching on the path's extension.
    #[test]
    fn a_flac_named_wav_decodes_as_a_flac() {
        let bytes = std::fs::read(libflac_clip_path()).expect("the committed FLAC reads");
        assert_eq!(
            decode_bytes("named.wav", &bytes, FileConfig::default()),
            clip_samples()
        );
    }

    /// The WAV encodings that open for the first time deliver the samples
    /// their format defines.
    ///
    /// The three that can hold the source exactly are compared to it exactly:
    /// 24-bit, an extensible container, and the plain 16-bit control. The
    /// 8-bit path is compared against `(byte - 128) / 128`, which is the
    /// whole of that format's definition, over every one of its 256 codes.
    ///
    /// Regression: a sample conversion that scales, biases, sign-flips or
    /// takes the wrong end of a 24-bit word, none of which a tolerance
    /// against lossy audio would catch.
    #[test]
    fn the_wav_encodings_that_newly_open_deliver_what_their_format_defines() {
        let reference = clip_samples();

        // 24-bit little-endian PCM, the source's 16 bits in the top of each
        // word, so the value is unchanged.
        let mut pcm24 = Vec::with_capacity(reference.len() * 3);
        for &s in &reference {
            let word = i32::from((s * 32768.0) as i16) << 8;
            pcm24.extend_from_slice(&word.to_le_bytes()[0..3]);
        }
        let bytes = riff_wave(&wave_fmt(1, 1, 16000, 24), &pcm24);
        assert_eq!(
            decode_bytes("clip24.wav", &bytes, FileConfig::default()),
            reference,
            "24-bit"
        );

        // The same body in an extensible container, which names its encoding
        // by SubFormat GUID rather than by tag.
        let payload = sample::f32_to_i16_le_bytes(&reference);
        let bytes = riff_wave(&wave_fmt_extensible(SUBTYPE_PCM, 1, 16000, 16), &payload);
        assert_eq!(
            decode_bytes("clipx.wav", &bytes, FileConfig::default()),
            reference,
            "extensible"
        );

        // 8-bit unsigned PCM, over all 256 codes, against the format's own
        // definition. 0 is negative full scale, 128 is silence, 255 is one
        // step short of positive full scale.
        let codes: Vec<u8> = (0..=255u8).collect();
        let bytes = riff_wave(&wave_fmt(1, 1, 16000, 8), &codes);
        let decoded = decode_bytes("clip8.wav", &bytes, FileConfig::default());
        let expected: Vec<f32> = codes
            .iter()
            .map(|&b| (f32::from(b) - 128.0) / 128.0)
            .collect();
        assert_eq!(decoded, expected, "8-bit");
        assert_eq!(decoded[0], -1.0);
        assert_eq!(decoded[128], 0.0);
    }

    /// The two G.711 laws open in both containers that carry them, and the
    /// same law decodes identically whichever container carried it.
    ///
    /// Every one of the 256 codes is swept rather than a clip sampled, so a
    /// wrong entry anywhere in either expansion table is reached. The two
    /// laws are asserted distinct from each other, which is what a reader
    /// that had wired one table to both tags would fail.
    ///
    /// Regression: a mu-law table used for A-law or the reverse, a table
    /// misindexed at the segment boundaries, and an AIFF-C compression
    /// four-CC routed to the wrong decoder.
    #[test]
    fn the_g711_laws_decode_the_same_in_a_wav_and_in_an_aiff_c() {
        let codes: Vec<u8> = (0..=255u8).collect();
        let frames = codes.len() as u32;
        // Telephony's own rate, and the target rate too, so the chain is a
        // pass-through and the delivered samples are the decoded codes.
        let at_8k = || FileConfig {
            sample_rate: 8000,
            ..Default::default()
        };

        for (tag, four_cc, label) in [(7u16, b"ulaw", "mu-law"), (6u16, b"alaw", "A-law")] {
            let wav = riff_wave(&wave_fmt(tag, 1, 8000, 8), &codes);
            let from_wav = decode_bytes(&format!("g711-{label}.wav"), &wav, at_8k());

            let aifc = iff_form(
                b"AIFC",
                &aiff_comm(1, frames, 8, 8000, Some(four_cc)),
                &codes,
            );
            let from_aifc = decode_bytes(&format!("g711-{label}.aifc"), &aifc, at_8k());

            assert_eq!(from_wav.len(), 256, "{label}: one sample per code");
            assert_eq!(from_wav, from_aifc, "{label}: WAV against AIFF-C");
            assert!(
                from_wav.iter().all(|s| s.abs() <= 1.0),
                "{label}: every code lands inside full scale"
            );
            assert!(
                from_wav.iter().any(|&s| s > 0.9) && from_wav.iter().any(|&s| s < -0.9),
                "{label}: the sweep reaches both ends of the scale"
            );
            // A companded law spends most of its codes near zero: the median
            // magnitude is a small fraction of the peak. A linear table put
            // behind either tag fails this.
            let mut magnitudes: Vec<f32> = from_wav.iter().map(|s| s.abs()).collect();
            magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert!(
                magnitudes[128] < 0.15,
                "{label}: the law is companded, median magnitude {}",
                magnitudes[128]
            );
        }

        // The two laws are different tables, so the same codes decode
        // differently under each.
        let mu = decode_bytes(
            "g711-cmp-mu.wav",
            &riff_wave(&wave_fmt(7, 1, 8000, 8), &codes),
            at_8k(),
        );
        let a = decode_bytes(
            "g711-cmp-a.wav",
            &riff_wave(&wave_fmt(6, 1, 8000, 8), &codes),
            at_8k(),
        );
        assert_ne!(mu, a, "mu-law and A-law are not the same table");
    }

    /// The conditioning chain runs on the committed FLAC, at a target rate
    /// the source is resampled to, and produces what the WAV of the same
    /// samples produces.
    ///
    /// Regression: a reader that delivers audio the chain cannot be driven
    /// with (a wrong channel count, a wrong rate, an interleaving fault),
    /// none of which the pass-through comparison reaches.
    #[test]
    fn the_conditioning_chain_runs_on_a_flac_at_a_resampled_rate() {
        let reference = clip_samples();
        let config = FileConfig {
            sample_rate: 24000,
            dc_removal: true,
            highpass: Some(HighpassFilter::Hz80),
            ..Default::default()
        };

        let flac = collect(File::open(libflac_clip_path(), config.clone()).unwrap());
        let wav = decode_bytes(
            "chain.wav",
            &wav_bytes(1, 1, 16000, &reference),
            config.clone(),
        );
        assert_eq!(flac, wav);
        assert!(
            flac.len() > reference.len(),
            "the source was resampled up: {} from {}",
            flac.len(),
            reference.len()
        );
        assert!(flac.iter().any(|&s| s != 0.0));

        // The conditioning is doing something: the same source at the same
        // target rate with every option off is a different signal.
        let plain = collect(
            File::open(
                libflac_clip_path(),
                FileConfig {
                    sample_rate: 24000,
                    ..Default::default()
                },
            )
            .unwrap(),
        );
        assert_ne!(flac, plain, "the conditioning stages engaged");

        // The limiter's ceiling holds on audio that reached the chain
        // through the FLAC reader, where that stage is compiled in.
        #[cfg(feature = "gain")]
        {
            let limited = collect(
                File::open(
                    libflac_clip_path(),
                    FileConfig {
                        sample_rate: 24000,
                        agc: Some(-18),
                        limiter: Some(-1.0),
                        ..Default::default()
                    },
                )
                .unwrap(),
            );
            let peak = limited.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
            let ceiling = 10f32.powf(-1.0 / 20.0);
            assert!(peak <= ceiling + 1e-6, "the limiter ceiling holds: {peak}");
        }
    }

    /// decibri's accepted input range is enforced on every container, not on
    /// the one that used to carry the check.
    ///
    /// The out-of-range case that settles the placement is an AIFF, a
    /// container that never passes a WAV reader at all: a check reinstated
    /// inside one would let it through. Both boundaries are asserted
    /// inclusive from both sides.
    ///
    /// Regression: reinstating the range check inside a container's own
    /// reader, and a boundary written exclusive.
    #[test]
    fn the_sample_rate_range_is_enforced_on_every_container() {
        let frames = 100u32;
        let payload = vec![0u8; frames as usize * 2];

        // A non-WAV container out of range, against the same container in
        // range: the rejection is the rate and not the container.
        let aiff = iff_form(b"AIFF", &aiff_comm(1, frames, 16, 500, None), &payload);
        assert!(
            matches!(
                decode_error("rate500.aiff", &aiff),
                DecibriError::SampleRateOutOfRange
            ),
            "a 500 Hz AIFF is rejected by a check that is not in the WAV reader"
        );
        let control = iff_form(b"AIFF", &aiff_comm(1, frames, 16, 16000, None), &payload);
        assert_eq!(
            decode_bytes("rate16k.aiff", &control, FileConfig::default()).len(),
            frames as usize,
            "the same AIFF at an accepted rate opens"
        );

        for rate in [999u32, 384001] {
            let bytes = riff_wave(&wave_fmt(1, 1, rate, 16), &payload);
            assert!(
                matches!(
                    decode_error(&format!("rate{rate}.wav"), &bytes),
                    DecibriError::SampleRateOutOfRange
                ),
                "{rate} Hz is outside the range"
            );
        }
        for rate in [1000u32, 384000] {
            let bytes = riff_wave(&wave_fmt(1, 1, rate, 16), &payload);
            let out = decode_bytes(
                &format!("rate{rate}.wav"),
                &bytes,
                FileConfig {
                    sample_rate: rate,
                    ..Default::default()
                },
            );
            assert_eq!(out.len(), frames as usize, "{rate} Hz is inside the range");
        }
    }

    /// Each of the three read failures is reachable, and each reports its own
    /// variant.
    ///
    /// Every failing input here is a one-field edit of a control that opens,
    /// asserted first, so each rejection is the field named against it and
    /// not some other thing wrong with an assembled file.
    ///
    /// Regression: a mapping that collapses the three into one, leaving a
    /// caller unable to tell "re-encode this" from "fetch it again".
    #[test]
    fn each_read_failure_reports_its_own_variant() {
        let payload = vec![0u8; 200];

        // The control every WAV case below is an edit of: one channel,
        // 16-bit integer PCM at 16 kHz, a `data` chunk holding all 200 bytes.
        let control = riff_wave(&wave_fmt(1, 1, 16000, 16), &payload);
        assert_eq!(
            decode_bytes("control.wav", &control, FileConfig::default()).len(),
            100,
            "the control opens, so each rejection below is its own edit"
        );

        // Unsupported: a container magic nothing carries, a WAVE format tag
        // for a codec the reader does not decode (0x0011, IMA ADPCM), a
        // sample width it does not carry, and a zero channel count. The tag
        // and the width are read from the header, so the payload is never
        // reached and does not have to be real ADPCM.
        let unsupported: [(&str, Vec<u8>); 4] = [
            ("junk", b"this is definitely not audio at all".to_vec()),
            ("adpcm", riff_wave(&wave_fmt(0x0011, 1, 16000, 4), &payload)),
            ("pcm20", riff_wave(&wave_fmt(1, 1, 16000, 20), &payload)),
            ("chan0", riff_wave(&wave_fmt(1, 0, 16000, 16), &payload)),
        ];
        for (name, bytes) in unsupported {
            let err = decode_error(&format!("unsupported-{name}"), &bytes);
            assert!(
                matches!(err, DecibriError::AudioFormatUnsupported { .. }),
                "{name}: {err}"
            );
        }

        // Malformed: a RIFF/WAVE carrying a `fmt ` chunk and no `data` chunk.
        // The structure parses and then runs out of the chunk the format
        // requires, which is a different thing from the file being short.
        let mut no_data = b"WAVE".to_vec();
        let fmt = wave_fmt(1, 1, 16000, 16);
        no_data.extend_from_slice(b"fmt ");
        no_data.extend_from_slice(&(fmt.len() as u32).to_le_bytes());
        no_data.extend_from_slice(&fmt);
        let mut bytes = b"RIFF".to_vec();
        bytes.extend_from_slice(&(no_data.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&no_data);
        let err = decode_error("malformed-nodata.wav", &bytes);
        assert!(
            matches!(err, DecibriError::AudioFileMalformed { .. }),
            "{err}"
        );

        // Malformed, on real encoder output: one byte of the committed FLAC
        // flipped mid-stream, which the frame CRC catches before a sample of
        // that frame is delivered.
        let mut corrupt = std::fs::read(libflac_clip_path()).expect("the committed FLAC reads");
        let at = corrupt.len() / 2;
        corrupt[at] ^= 0xFF;
        let err = decode_error("malformed-bitflip.flac", &corrupt);
        assert!(
            matches!(err, DecibriError::AudioFileMalformed { .. }),
            "{err}"
        );

        // Truncated: nothing at all, less than a container header, a WAV
        // whose `data` chunk declares more than the file holds, and the
        // committed FLAC cut short.
        let mut short_data = riff_wave(&wave_fmt(1, 1, 16000, 16), &payload);
        short_data.truncate(short_data.len() - 40);
        let whole = std::fs::read(libflac_clip_path()).expect("the committed FLAC reads");
        let truncated: [(&str, Vec<u8>); 4] = [
            ("empty", Vec::new()),
            ("stub", b"RIFF\x00\x00\x00\x00".to_vec()),
            ("shortdata", short_data),
            ("flac", whole[..whole.len() * 6 / 10].to_vec()),
        ];
        for (name, bytes) in truncated {
            let err = decode_error(&format!("truncated-{name}"), &bytes);
            assert!(
                matches!(err, DecibriError::AudioFileTruncated { .. }),
                "{name}: {err}"
            );
        }
    }

    /// A behaviour change, pinned: a WAV whose declared data length is not a
    /// whole number of frames is reported rather than delivered slightly
    /// short. Two channels at 16 bits make a 4-byte frame, and the length
    /// here is two bytes past a whole number of them.
    ///
    /// A file shorter than its own declaration already failed, which is what
    /// an interrupted download leaves behind; the control below asserts that
    /// a whole number of frames of the same shape still opens, so the
    /// rejection is the misalignment and nothing else.
    ///
    /// Regression: silently dropping the partial frame, which is what the
    /// reader this replaced did.
    #[test]
    fn a_frame_misaligned_wav_is_reported_rather_than_delivered_short() {
        let fmt = wave_fmt(1, 2, 16000, 16);
        let aligned = vec![0u8; 200];
        let out = decode_bytes(
            "aligned.wav",
            &riff_wave(&fmt, &aligned),
            FileConfig::default(),
        );
        assert_eq!(out.len(), 50, "50 whole stereo frames open and downmix");

        let mut misaligned = aligned.clone();
        misaligned.extend_from_slice(&[0x55, 0x66]);
        let err = decode_error("halfframe.wav", &riff_wave(&fmt, &misaligned));
        assert!(
            matches!(err, DecibriError::AudioFileTruncated { .. }),
            "{err}"
        );
    }

    /// The reader's eight failures land in the three decibri names, pinned
    /// one by one, including the two the reader has no construction site for.
    ///
    /// The mapping's `_` arm cannot be reached from here, because the
    /// reader's error type is `#[non_exhaustive]` and a variant outside the
    /// eight cannot be built. `ContainerCodecMismatch` is the row that stands
    /// in for it: it is the shape a later addition takes, a codec named ahead
    /// of its decoder, and it is pinned to the same name the `_` arm carries.
    ///
    /// Regression: a later reader release moving a failure into a category
    /// that tells the caller to do the wrong thing about their file.
    #[test]
    fn the_reader_error_mapping_is_pinned() {
        use decibri_decode::{CodecId, DecodeError, FourCc};

        let unsupported: [DecodeError; 5] = [
            DecodeError::UnsupportedContainer {
                tag: FourCc(*b"OggS"),
            },
            DecodeError::UnsupportedCodec {
                codec: CodecId::WaveFormatTag(0x0011),
            },
            DecodeError::UnsupportedSampleFormat {
                format: CodecId::WaveFormatTag(1),
                bits_per_sample: 20,
            },
            DecodeError::UnsupportedChannelLayout { channels: 0 },
            // Unreachable through the reader's entry point, and the stand-in
            // for a variant a later release adds.
            DecodeError::ContainerCodecMismatch {
                declared: CodecId::WaveFormatTag(1),
                found: CodecId::WaveFormatTag(7),
            },
        ];
        for source in unsupported {
            let text = source.to_string();
            let mapped = DecibriError::from(source);
            assert!(
                matches!(mapped, DecibriError::AudioFormatUnsupported { .. }),
                "{text} -> {mapped}"
            );
            // The reader's own text reaches the caller: it names the tag,
            // four-CC or width decibri could only describe generically.
            assert!(mapped.to_string().ends_with(&text), "{mapped}");
        }

        let mapped = DecibriError::from(DecodeError::Malformed {
            expected: "a 'data' chunk header",
            offset: 36,
        });
        assert!(
            matches!(mapped, DecibriError::AudioFileMalformed { .. }),
            "{mapped}"
        );

        let mapped = DecibriError::from(DecodeError::Truncated {
            expected: 4096,
            available: 1200,
        });
        assert!(
            matches!(mapped, DecibriError::AudioFileTruncated { .. }),
            "{mapped}"
        );

        // Rate conversion keeps the identity it carries everywhere else,
        // rather than becoming a statement about the file.
        let mapped = DecibriError::from(DecodeError::Resample(
            decibri_resampler::ResamplerError::ZeroSampleRate,
        ));
        assert!(
            matches!(mapped, DecibriError::SampleRateOutOfRange),
            "{mapped}"
        );
    }

    /// The three new messages carry the prefixes the Node classifier keys on,
    /// and the codes the identity table assigns.
    ///
    /// Regression: a reworded message, which would leave the Node table
    /// classifying the failure as an unnamed decibri error.
    #[test]
    fn the_read_failure_messages_and_codes_are_pinned() {
        let cases = [
            (
                DecibriError::AudioFormatUnsupported {
                    reason: "why".to_string(),
                },
                "unsupported audio format: why",
                "AUDIO_FORMAT_UNSUPPORTED",
            ),
            (
                DecibriError::AudioFileMalformed {
                    reason: "why".to_string(),
                },
                "malformed audio file: why",
                "AUDIO_FILE_MALFORMED",
            ),
            (
                DecibriError::AudioFileTruncated {
                    reason: "why".to_string(),
                },
                "truncated audio file: why",
                "AUDIO_FILE_TRUNCATED",
            ),
        ];
        for (err, message, code) in cases {
            assert_eq!(err.to_string(), message);
            assert_eq!(err.code(), code);
        }
    }

    /// Configuration validation matches the live path: rate, channel floor,
    /// AGC target, and limiter ceiling ranges are enforced at construction.
    #[test]
    fn config_validation_matches_live_ranges() {
        let config = FileConfig {
            sample_rate: 999,
            ..Default::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0], 16000, 1, config),
            Err(DecibriError::SampleRateOutOfRange)
        ));

        // The floor only, exactly as the live path's `validate`: no upper
        // bound exists here, the source's own count answers at construction.
        let config = FileConfig {
            channels: 0,
            ..Default::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0], 16000, 1, config),
            Err(DecibriError::ChannelsOutOfRange)
        ));

        let config = FileConfig {
            agc: Some(-2),
            ..Default::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0], 16000, 1, config),
            Err(DecibriError::AgcTargetOutOfRange)
        ));

        let config = FileConfig {
            limiter: Some(0.5),
            ..Default::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0], 16000, 1, config),
            Err(DecibriError::LimiterCeilingOutOfRange)
        ));

        assert!(matches!(
            File::buffer(vec![0.0], 999, 1, FileConfig::default()),
            Err(DecibriError::SampleRateOutOfRange)
        ));
    }

    // ── Segment merging (pure file-time policy, no model involved) ──────

    #[cfg(feature = "vad")]
    fn window(i: usize, is_speech: bool) -> VadWindow {
        let secs = 512.0 / 16000.0;
        VadWindow {
            start: i as f64 * secs,
            end: (i + 1) as f64 * secs,
            probability: if is_speech { 0.9 } else { 0.1 },
            is_speech,
        }
    }

    /// Consecutive speech windows merge into one segment that ends at the
    /// last speech window.
    #[cfg(feature = "vad")]
    #[test]
    fn merge_joins_consecutive_speech() {
        let scores = vec![
            window(0, true),
            window(1, true),
            window(2, false),
            window(3, false),
        ];
        let segments = merge_segments(&scores, 0.3);
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].start, 0.0);
        assert_eq!(segments[0].end, scores[1].end);
    }

    /// A silence gap within the holdoff keeps one segment open; a longer gap
    /// splits two.
    #[cfg(feature = "vad")]
    #[test]
    fn merge_applies_file_time_holdoff() {
        // 32 ms windows: an 8-window gap is 256 ms (inside a 300 ms holdoff),
        // an 11-window gap is 352 ms (outside it).
        let mut inside: Vec<VadWindow> = Vec::new();
        inside.push(window(0, true));
        for i in 1..9 {
            inside.push(window(i, false));
        }
        inside.push(window(9, true));
        assert_eq!(merge_segments(&inside, 0.3).len(), 1);

        let mut outside: Vec<VadWindow> = Vec::new();
        outside.push(window(0, true));
        for i in 1..12 {
            outside.push(window(i, false));
        }
        outside.push(window(12, true));
        let segments = merge_segments(&outside, 0.3);
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0].end, inside[0].end);
        assert_eq!(segments[1].start, window(12, true).start);
    }

    /// No speech windows means no segments.
    #[cfg(feature = "vad")]
    #[test]
    fn merge_empty_and_silence_yield_no_segments() {
        assert!(merge_segments(&[], 0.3).is_empty());
        let silence = vec![window(0, false), window(1, false)];
        assert!(merge_segments(&silence, 0.3).is_empty());
    }

    // ── Whole-recording analysis (drives the Silero model) ──────────────

    #[cfg(feature = "vad")]
    fn silero_model_path() -> PathBuf {
        // Resolve relative to the workspace root, exactly as the detector's
        // own tests do.
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir)
            .join("..")
            .join("..")
            .join("models")
            .join("silero_vad.onnx")
    }

    #[cfg(feature = "denoise")]
    fn denoise_model_path() -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir)
            .join("..")
            .join("..")
            .join("models")
            .join("fastenhancer_t.onnx")
    }

    fn golden_wav_path() -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        Path::new(manifest_dir)
            .join("tests")
            .join("assets")
            .join("vad-golden-tts-speech-16k.wav")
    }

    /// Per-window scores a whole analysis of the golden recording returns.
    #[cfg(feature = "vad")]
    const GOLDEN_SCORE_COUNT: usize = 208;

    /// Merged speech segments a whole analysis of the golden recording
    /// returns.
    #[cfg(feature = "vad")]
    const GOLDEN_SEGMENT_COUNT: usize = 2;

    #[cfg(feature = "vad")]
    fn vad_file_config() -> FileConfig {
        FileConfig {
            vad: Some(VadConfig {
                model_path: silero_model_path(),
                ..VadConfig::default()
            }),
            ..Default::default()
        }
    }

    /// Analysis without VAD configured reports the typed error rather than
    /// silently constructing a detector.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_without_vad_errors() {
        let file = File::buffer(sine(16000, 0.1), 16000, 1, FileConfig::default()).unwrap();
        assert!(matches!(
            file.analyze(),
            Err(DecibriError::VadNotConfigured)
        ));
        let file = File::buffer(sine(16000, 0.1), 16000, 1, FileConfig::default()).unwrap();
        assert!(matches!(
            file.analyse(),
            Err(DecibriError::VadNotConfigured)
        ));
    }

    /// Analysis after iteration has begun reports the typed error rather than
    /// scoring the unread remainder and timing it from zero.
    ///
    /// The `File` here has no VAD configured, so the error also pins the
    /// check order: the engaged state is reported before the missing
    /// detector configuration.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_after_partial_iteration_errors() {
        let mut file = File::buffer(sine(16000, 0.5), 16000, 1, FileConfig::default()).unwrap();
        let first = file.next();
        assert!(matches!(first, Some(Ok(_))), "the first chunk is delivered");
        assert!(matches!(file.analyze(), Err(DecibriError::FileEngaged)));

        let mut file = File::buffer(sine(16000, 0.5), 16000, 1, FileConfig::default()).unwrap();
        let _ = file.next();
        assert!(matches!(file.analyse(), Err(DecibriError::FileEngaged)));
    }

    /// Every route that can advance the cursor and leave the `File` owned is
    /// guarded, not only the ones spelled `next()`.
    ///
    /// `Iterator`'s provided methods and its adapters all reach the source
    /// through `File::next`, whether called directly on `&mut File` or
    /// through `by_ref()`, so each of these must report the same error. The
    /// routes that take the `File` by value cannot be followed by an analysis
    /// at all, which the type system already covers.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_after_any_iteration_route_errors() {
        /// Drive a fresh `File` through one iteration route, then report
        /// whether the analysis that follows refuses.
        fn refuses_after(drive: impl FnOnce(&mut File)) -> bool {
            let mut file = File::buffer(sine(16000, 0.5), 16000, 1, FileConfig::default()).unwrap();
            drive(&mut file);
            matches!(file.analyze(), Err(DecibriError::FileEngaged))
        }

        // Provided methods that borrow the `File`.
        assert!(
            refuses_after(|f| {
                let _ = f.next();
            }),
            "after next()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.nth(1);
            }),
            "after nth()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.find(|_| false);
            }),
            "after find()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.find_map(|_| None::<()>);
            }),
            "after find_map()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.position(|_| false);
            }),
            "after position()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.any(|_| false);
            }),
            "after any()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.all(|_| true);
            }),
            "after all()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.try_fold(0usize, |n, _| Some(n + 1));
            }),
            "after try_fold()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.try_for_each(|_| Some(()));
            }),
            "after try_for_each()"
        );

        // Adapters, reached through by_ref() so the `File` survives them.
        assert!(
            refuses_after(|f| {
                let _ = f.by_ref().take(2).count();
            }),
            "after take()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.by_ref().filter(|r| r.is_ok()).count();
            }),
            "after filter().count()"
        );
        assert!(
            refuses_after(|f| {
                let _ = f.by_ref().last();
            }),
            "after last()"
        );
        assert!(
            refuses_after(|f| f.by_ref().for_each(drop)),
            "after for_each()"
        );
        // Peeking pulls a chunk and buffers it inside the adapter, so it moves
        // the cursor without the caller ever seeing a delivery.
        assert!(
            refuses_after(|f| {
                let mut lookahead = f.by_ref().peekable();
                let _ = lookahead.peek().is_some();
            }),
            "after peekable().peek()"
        );

        // `&mut File` is an `Iterator` in its own right, which reaches
        // `IntoIterator` and the collection traits without moving the `File`.
        assert!(
            refuses_after(|f| {
                for chunk in &mut *f {
                    drop(chunk);
                }
            }),
            "after a for loop over &mut File"
        );
        assert!(
            refuses_after(|f| {
                let _ = (&mut *f).collect::<Vec<_>>();
            }),
            "after collect() over &mut File"
        );

        assert!(
            refuses_after(|f| {
                for chunk in f.by_ref() {
                    drop(chunk);
                }
            }),
            "after a for loop over by_ref()"
        );
    }

    /// A `next()` that delivers no chunk still advances the cursor, so it
    /// counts as iteration: an empty source and a fully drained one both
    /// refuse a later analysis.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_after_iteration_without_delivery_errors() {
        // An empty source: the first `next()` flushes the chain and returns
        // nothing, having delivered no chunk at all.
        let mut file = File::buffer(Vec::new(), 16000, 1, FileConfig::default()).unwrap();
        assert!(file.next().is_none(), "an empty source delivers no chunk");
        assert!(matches!(file.analyze(), Err(DecibriError::FileEngaged)));

        // A drained source: iteration ran to its end, which is the shape a
        // caller reaches by streaming the whole recording first.
        let mut file = File::buffer(sine(16000, 0.5), 16000, 1, FileConfig::default()).unwrap();
        file.by_ref().for_each(drop);
        assert!(matches!(file.analyze(), Err(DecibriError::FileEngaged)));
    }

    /// A drained `File` keeps ending quietly: `next()` goes on returning
    /// `None` rather than reporting the engaged state, which belongs to
    /// analysis alone.
    #[test]
    fn iteration_past_exhaustion_stays_quiet() {
        let mut file = File::buffer(sine(16000, 0.5), 16000, 1, FileConfig::default()).unwrap();
        let delivered = file.by_ref().filter(|chunk| chunk.is_ok()).count();
        assert!(delivered > 0, "the first pass delivers audio");
        assert!(file.next().is_none(), "a drained File ends quietly");
        assert!(file.next().is_none(), "and stays ended on every later call");
    }

    /// A `File` nobody has iterated still analyzes: the guard costs the
    /// ordinary path nothing.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_on_a_pristine_file_still_succeeds() {
        let report = File::open(golden_wav_path(), vad_file_config())
            .unwrap()
            .analyze()
            .unwrap();
        assert!(!report.scores.is_empty());
        assert!(!report.segments.is_empty());
    }

    /// The analysis of the golden recording, pinned to exact counts. A change
    /// to either number means the analysis itself changed.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_of_the_golden_recording_is_pinned() {
        let report = File::open(golden_wav_path(), vad_file_config())
            .unwrap()
            .analyze()
            .unwrap();
        assert_eq!(report.scores.len(), GOLDEN_SCORE_COUNT);
        assert_eq!(report.segments.len(), GOLDEN_SEGMENT_COUNT);
    }

    /// The whole golden recording analyzes to one answer whether it arrives
    /// as the WAV it is committed as or as the AIFF assembled here from its
    /// samples: the same window scores and the same speech segments, at full
    /// length and with the pinned two segments in it.
    ///
    /// The FLAC needs no arm of its own. Analysis is a function of the
    /// samples and the rate alone, and the sample-identity test already
    /// pins the FLAC to the same samples, so a third arm here would restate
    /// that rather than test anything further.
    ///
    /// Regression: a reader whose output reaches the detector feed altered
    /// in the rate or the channel count the feed is built from, which the
    /// sample comparison does not reach.
    #[cfg(feature = "vad")]
    #[test]
    fn one_recording_in_two_containers_analyzes_to_one_answer() {
        let wav = File::open(golden_wav_path(), vad_file_config())
            .unwrap()
            .analyze()
            .unwrap();
        assert_eq!(wav.scores.len(), GOLDEN_SCORE_COUNT);
        assert_eq!(wav.segments.len(), GOLDEN_SEGMENT_COUNT);

        let samples = golden_samples();
        let aiff = iff_form(
            b"AIFF",
            &aiff_comm(1, samples.len() as u32, 16, 16000, None),
            &be_i16_bytes(&samples),
        );
        let report = with_file("analyze.aiff", &aiff, |path| {
            File::open(path, vad_file_config())
                .unwrap()
                .analyze()
                .unwrap()
        });
        assert_eq!(report.scores, wav.scores, "AIFF");
        assert_eq!(report.segments, wav.segments, "AIFF");
    }

    /// THE detector-feed invariant: analysis scores equal feeding the
    /// detector the same recording window by window directly. The `File`
    /// path adds no conditioning here, so its detector feed is the recording
    /// itself; the two probability sequences must match exactly.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_matches_direct_window_scoring() {
        let path = golden_wav_path();
        let report = File::open(&path, vad_file_config())
            .unwrap()
            .analyze()
            .unwrap();

        // Score the same samples directly, one 512-sample window at a time.
        let bytes = std::fs::read(&path).unwrap();
        let samples = decibri_decode::decode(&bytes).unwrap().into_samples();
        let mut detector = SileroVad::new(VadConfig {
            model_path: silero_model_path(),
            ..VadConfig::default()
        })
        .unwrap();
        let mut expected: Vec<f32> = Vec::new();
        for window in samples.as_chunks::<512>().0 {
            expected.push(detector.process(window).unwrap().probability);
        }

        assert_eq!(report.scores.len(), expected.len());
        for (w, e) in report.scores.iter().zip(expected.iter()) {
            assert_eq!(w.probability, *e);
        }
        // Window timing tiles the recording in 32 ms steps of file time.
        for (i, w) in report.scores.iter().enumerate() {
            assert!((w.start - i as f64 * 512.0 / 16000.0).abs() < 1e-9);
            assert!((w.end - w.start - 512.0 / 16000.0).abs() < 1e-9);
        }
        // The golden recording contains real speech: segments exist and each
        // ends at a speech window inside the recording.
        assert!(!report.segments.is_empty());
        let duration = samples.len() as f64 / 16000.0;
        for segment in &report.segments {
            assert!(segment.start < segment.end);
            assert!(segment.end <= duration + 1e-9);
        }
    }

    /// The analysis report does not depend on the conditioning settings: the
    /// detector reads the signal the normalize tier produces, taken before any
    /// conditioning transform. Analysis runs that tier alone, so this equality
    /// must hold for every conditioning setting.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_report_is_independent_of_the_dsp_tier() {
        let path = golden_wav_path();
        let plain = File::open(&path, vad_file_config())
            .unwrap()
            .analyze()
            .unwrap();

        let mut config = vad_file_config();
        config.dc_removal = true;
        config.highpass = Some(HighpassFilter::Hz80);
        config.agc = Some(-18);
        config.limiter = Some(-1.0);
        let conditioned = File::open(&path, config).unwrap().analyze().unwrap();

        assert_eq!(plain.scores, conditioned.scores);
        assert_eq!(plain.segments, conditioned.segments);
    }

    /// The same equality with denoise on. Denoise is framed and
    /// length-changing, unlike the same-length DSP stages, so it exercises the
    /// invariant on a transform whose output no longer lines up with its
    /// input. Checked at the detector's own rate and at a target rate that puts
    /// a resampler in the normalize tier, which covers the end-of-stream tail
    /// as well.
    #[cfg(all(feature = "vad", feature = "denoise"))]
    #[test]
    fn analyze_report_is_independent_of_denoise() {
        let path = golden_wav_path();
        for rate in [16000, 22050] {
            let mut plain_config = vad_file_config();
            plain_config.sample_rate = rate;
            let plain = File::open(&path, plain_config).unwrap().analyze().unwrap();

            let mut config = vad_file_config();
            config.sample_rate = rate;
            config.dc_removal = true;
            config.denoise = Some(DenoiseModel::FastEnhancerT);
            config.denoise_model_path = Some(denoise_model_path());
            let denoised = File::open(&path, config).unwrap().analyze().unwrap();

            assert_eq!(plain.scores, denoised.scores, "scores at {rate} Hz");
            assert_eq!(plain.segments, denoised.segments, "segments at {rate} Hz");
        }
    }

    /// Both spellings return the same analysis.
    #[cfg(feature = "vad")]
    #[test]
    fn analyse_equals_analyze() {
        let path = golden_wav_path();
        let a = File::open(&path, vad_file_config())
            .unwrap()
            .analyze()
            .unwrap();
        let b = File::open(&path, vad_file_config())
            .unwrap()
            .analyse()
            .unwrap();
        assert_eq!(a.scores, b.scores);
        assert_eq!(a.segments, b.segments);
    }

    /// A target rate the detector does not accept still analyzes: the
    /// detector feed is resampled internally to 16 kHz, no setting and no
    /// error, and the recording's speech is still found.
    #[cfg(feature = "vad")]
    #[test]
    fn analyze_resamples_feed_for_non_detector_rate() {
        let mut config = vad_file_config();
        config.sample_rate = 22050;
        let file = File::open(golden_wav_path(), config).unwrap();
        assert_eq!(file.vad_rate(), Some(16000));
        let report = file.analyze().unwrap();
        assert!(!report.scores.is_empty());
        let max = report
            .scores
            .iter()
            .map(|w| w.probability)
            .fold(0.0f32, f32::max);
        assert!(
            max >= 0.5,
            "speech should still be found through the internal feed resample, max {max}"
        );
        assert!(!report.segments.is_empty());
    }

    /// A detector-native target rate runs detection at that rate with no
    /// internal resample.
    #[cfg(feature = "vad")]
    #[test]
    fn vad_rate_follows_detector_native_targets() {
        let mut config = vad_file_config();
        config.sample_rate = 8000;
        let file = File::buffer(sine(8000, 0.1), 8000, 1, config).unwrap();
        assert_eq!(file.vad_rate(), Some(8000));

        let file = File::buffer(sine(16000, 0.1), 16000, 1, FileConfig::default()).unwrap();
        assert_eq!(file.vad_rate(), None);
    }

    /// The binding-internal detector feed drains the pre-conditioning signal
    /// in step with iteration, and is absent without VAD.
    #[cfg(feature = "vad")]
    #[test]
    fn vad_input_drains_feed_during_iteration() {
        let input = sine(16000, 0.2);
        let mut config = vad_file_config();
        config.dc_removal = true;
        let mut file = File::buffer(input.clone(), 16000, 1, config).unwrap();
        let mut fed: Vec<f32> = Vec::new();
        loop {
            let chunk = match file.next() {
                Some(Ok(chunk)) => chunk,
                Some(Err(e)) => panic!("iteration error: {e}"),
                None => break,
            };
            let _ = chunk;
            fed.extend(file.vad_input().expect("vad configured"));
        }
        // The feed is the PRE-conditioning signal: with only a same-length
        // DC-removal transform it equals the input exactly.
        assert_eq!(fed, input);

        let mut plain = File::buffer(input, 16000, 1, FileConfig::default()).unwrap();
        assert!(plain.vad_input().is_none());
    }

    // ── Save: writing the conditioned recording ─────────────────────────

    /// A uniquely named path under the system temp directory, handed to
    /// `save_it` and removed afterwards whether or not the save created it.
    fn with_save_path<T>(name: &str, save_it: impl FnOnce(&Path) -> T) -> T {
        let path = std::env::temp_dir().join(format!("decibri-save-{name}"));
        std::fs::remove_file(&path).ok();
        let out = save_it(&path);
        std::fs::remove_file(&path).ok();
        out
    }

    /// Samples that live exactly on the 16-bit grid (`k / 32768`), so the
    /// quantisation into a 16-bit file is the identity and a round trip can
    /// be asserted sample for sample rather than within a tolerance.
    fn grid_samples(count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| ((i as i32 % 1201) - 600) as f32 / 32768.0)
            .collect()
    }

    /// Quantise one sample the way every 16-bit writer in the save path
    /// does: clamp, scale by 32768, truncate toward zero, clamp in integer
    /// space, back to `f32` over the same scale.
    fn quantise_i16(sample: f32) -> f32 {
        let scaled = (sample.clamp(-1.0, 1.0) * 32768.0) as i32;
        scaled.clamp(-32768, 32767) as f32 / 32768.0
    }

    /// Save writes the delivered audio and open reads it back identically,
    /// in every container the extension rule names. Catches a writer wired
    /// to the wrong container, a container that loses samples, and a drain
    /// that differs from what iteration delivers.
    #[test]
    fn save_round_trip_is_exact_in_all_three_containers() {
        let input = grid_samples(4000);
        for (ext, magic) in [
            ("wav", &b"RIFF"[..]),
            ("aiff", &b"FORM"[..]),
            ("flac", &b"fLaC"[..]),
        ] {
            with_save_path(&format!("roundtrip.{ext}"), |path| {
                let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
                let report = file.save(path, SaveOptions::default()).unwrap();
                assert_eq!(report.clipped_samples, 0, "{ext}: nothing to clip");
                assert_eq!(report.non_finite_samples, 0, "{ext}: nothing to repair");
                let bytes = std::fs::read(path).expect("the saved file exists");
                assert_eq!(
                    &bytes[..4],
                    magic,
                    "{ext}: the extension picks the container"
                );
                let back = collect(File::open(path, FileConfig::default()).unwrap());
                assert_eq!(back, input, "{ext}: the round trip is exact");
            });
        }
    }

    /// A conditioned save writes the conditioned audio, not the input: the
    /// saved file differs from the source and reads back as exactly the
    /// quantised delivered stream. Catches a save path that bypasses the
    /// conditioning chain.
    #[test]
    fn save_writes_the_conditioned_audio_not_the_input() {
        // A DC offset the conditioning removes, on the 16-bit grid.
        let input: Vec<f32> = grid_samples(4000)
            .iter()
            .map(|s| s + 8192.0 / 32768.0)
            .collect();
        let config = FileConfig {
            dc_removal: true,
            ..Default::default()
        };
        let expected: Vec<f32> =
            collect(File::buffer(input.clone(), 16000, 1, config.clone()).unwrap())
                .iter()
                .map(|&s| quantise_i16(s))
                .collect();
        assert_ne!(expected, input, "the conditioning changed the audio");
        with_save_path("conditioned.wav", |path| {
            let file = File::buffer(input.clone(), 16000, 1, config.clone()).unwrap();
            file.save(path, SaveOptions::default()).unwrap();
            let back = collect(File::open(path, FileConfig::default()).unwrap());
            assert_eq!(back, expected, "the file holds the conditioned stream");
        });
    }

    /// Saving is refused once iteration has begun, with the same identity
    /// and the same check order as analysis: the engaged state is reported
    /// before any argument is interpreted. Catches a save that writes the
    /// unread remainder of a partially streamed source.
    #[test]
    fn save_after_iteration_reports_file_engaged() {
        let mut file = File::buffer(sine(16000, 0.1), 16000, 1, FileConfig::default()).unwrap();
        let _ = file.next();
        with_save_path("engaged.wav", |path| {
            let err = file.save(path, SaveOptions::default()).unwrap_err();
            assert!(matches!(err, DecibriError::FileEngaged));
        });

        // Engaged wins over an invalid compression level, matching the
        // check order pinned for analyze.
        let mut file = File::buffer(sine(16000, 0.1), 16000, 1, FileConfig::default()).unwrap();
        let _ = file.next();
        with_save_path("engaged-order.flac", |path| {
            let options = SaveOptions {
                compression: Some(99),
                ..Default::default()
            };
            let err = file.save(path, options).unwrap_err();
            assert!(matches!(err, DecibriError::FileEngaged));
        });
    }

    /// An explicit format override beats the extension, including an
    /// extension outside the recognised set. Catches an override consulted
    /// after the extension rather than instead of it.
    #[test]
    fn save_format_override_beats_the_extension() {
        let input = grid_samples(400);
        with_save_path("override.flac", |path| {
            let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
            let options = SaveOptions {
                format: Some(SaveFormat::Wav),
                ..Default::default()
            };
            file.save(path, options).unwrap();
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(&bytes[..4], b"RIFF", "the override picked WAV");
        });
        with_save_path("override.dat", |path| {
            let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
            let options = SaveOptions {
                format: Some(SaveFormat::Flac),
                ..Default::default()
            };
            file.save(path, options).unwrap();
            let bytes = std::fs::read(path).unwrap();
            assert_eq!(
                &bytes[..4],
                b"fLaC",
                "an explicit format needs no known extension"
            );
        });
    }

    /// An extension outside the recognised set, or a path with none, is
    /// refused with the format identity and no file is created. Catches a
    /// silent default container.
    #[test]
    fn save_refuses_an_unrecognised_extension() {
        let input = grid_samples(400);
        with_save_path("refused.mp3", |path| {
            let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
            let err = file.save(path, SaveOptions::default()).unwrap_err();
            match err {
                DecibriError::AudioFormatUnsupported { reason } => {
                    assert!(
                        reason.contains("'.mp3'"),
                        "the extension is named: {reason}"
                    );
                }
                other => panic!("expected AudioFormatUnsupported, got {other:?}"),
            }
            assert!(!path.exists(), "no file appears for a refused save");
        });
        with_save_path("refused-no-extension", |path| {
            let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
            let err = file.save(path, SaveOptions::default()).unwrap_err();
            assert!(matches!(err, DecibriError::AudioFormatUnsupported { .. }));
            assert!(!path.exists());
        });
    }

    /// Extension matching is ASCII case-insensitive, and `.aif` and `.aifc`
    /// select the AIFF writer. Catches a case-sensitive extension table.
    #[test]
    fn save_extension_matching_is_case_insensitive() {
        let input = grid_samples(400);
        for name in ["upper.WAV", "mixed.Flac", "short.aif", "compressed.aifc"] {
            with_save_path(name, |path| {
                let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
                file.save(path, SaveOptions::default()).unwrap();
                let back = collect(File::open(path, FileConfig::default()).unwrap());
                assert_eq!(back, input, "{name}: saved and read back");
            });
        }
    }

    /// Every FLAC compression level in range is accepted, both boundaries
    /// included, decodes back to the identical audio, and a level past the
    /// range is refused with its own identity before anything is written.
    /// Catches the writer's own rejection leaking through as a malformed
    /// file error, and a lossy level.
    #[test]
    fn save_flac_compression_range_and_losslessness() {
        let reference = collect(File::open(libflac_clip_path(), FileConfig::default()).unwrap());
        for level in [0u8, 5, 8] {
            with_save_path(&format!("level{level}.flac"), |path| {
                let file = File::open(libflac_clip_path(), FileConfig::default()).unwrap();
                let options = SaveOptions {
                    compression: Some(level),
                    ..Default::default()
                };
                file.save(path, options).unwrap();
                let back = collect(File::open(path, FileConfig::default()).unwrap());
                assert_eq!(back, reference, "level {level} is lossless");
            });
        }
        with_save_path("level9.flac", |path| {
            let file = File::open(libflac_clip_path(), FileConfig::default()).unwrap();
            let options = SaveOptions {
                compression: Some(9),
                ..Default::default()
            };
            let err = file.save(path, options).unwrap_err();
            assert!(matches!(err, DecibriError::FlacCompressionOutOfRange));
            assert!(!path.exists(), "nothing is written for a refused level");
        });
    }

    /// A signal above full scale saves, the overshoot clamps to full scale,
    /// and the report counts exactly the clamped samples. Catches a silent
    /// clamp and a count that includes in-range samples.
    #[test]
    fn save_clips_overscale_and_counts_it() {
        let mut input = grid_samples(1600);
        input[10] = 1.85;
        input[20] = -1.85;
        input[30] = 2.5;
        with_save_path("clipped.wav", |path| {
            let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
            let report = file.save(path, SaveOptions::default()).unwrap();
            assert_eq!(
                report.clipped_samples, 3,
                "exactly the three overscale samples"
            );
            assert_eq!(report.non_finite_samples, 0);
            let back = collect(File::open(path, FileConfig::default()).unwrap());
            // Positive full scale quantises to 32767; negative to -32768.
            assert_eq!(back[10], 32767.0 / 32768.0);
            assert_eq!(back[20], -1.0);
            assert_eq!(back[30], 32767.0 / 32768.0);
            assert_eq!(back[40], input[40], "in-range neighbours are untouched");
        });
    }

    /// A non-finite sample reaches the save on the direct path (the probe
    /// pinned by `buffer_passthrough_is_byte_identical`), never reaches the
    /// file, and is counted: NaN becomes silence, an infinity becomes full
    /// scale, and the clip count stays zero because the repair is not a
    /// clamp. Catches the guard being dropped, folded into the clamp, or
    /// left to the writer.
    #[test]
    fn save_replaces_non_finite_and_counts_it() {
        let mut input = grid_samples(1600);
        input[10] = f32::NAN;
        input[20] = f32::INFINITY;
        input[30] = f32::NEG_INFINITY;
        with_save_path("nonfinite.wav", |path| {
            let file = File::buffer(input.clone(), 16000, 1, FileConfig::default()).unwrap();
            let report = file.save(path, SaveOptions::default()).unwrap();
            assert_eq!(
                report.non_finite_samples, 3,
                "exactly the three non-finite samples"
            );
            assert_eq!(report.clipped_samples, 0, "a repair is not a clip");
            let back = collect(File::open(path, FileConfig::default()).unwrap());
            assert_eq!(back[10], 0.0, "NaN became silence");
            assert_eq!(
                back[20],
                32767.0 / 32768.0,
                "positive infinity became full scale"
            );
            assert_eq!(back[30], -1.0, "negative infinity became full scale");
        });
    }

    /// The same repair applies when the non-finite input arrives from disk:
    /// a float WAV carrying NaN and infinity opens on the direct path and
    /// saves as the repaired audio. Pins the reachable path from `File::open`
    /// as well as from `File::buffer`.
    #[test]
    fn save_repairs_non_finite_input_read_from_a_float_file() {
        let mut input = grid_samples(1600);
        input[10] = f32::NAN;
        input[20] = f32::INFINITY;
        let wav = wav_bytes(3, 1, 16000, &input);
        let report = with_file("nonfinite-input.wav", &wav, |source| {
            with_save_path("nonfinite-from-open.wav", |dest| {
                let file = File::open(source, FileConfig::default()).unwrap();
                file.save(dest, SaveOptions::default()).unwrap()
            })
        });
        assert_eq!(report.non_finite_samples, 2);
        assert_eq!(report.clipped_samples, 0);
    }

    /// The guard and the clamp are separate steps: the guard repairs
    /// non-finite samples and leaves finite overshoot alone, so a float
    /// target given the guarded samples preserves 1.85 verbatim and its
    /// clip count is zero; the clamp alone accounts for clipping. Catches
    /// overscale clamping being folded into the non-finite repair, which
    /// would silently change what a float target writes.
    #[test]
    fn save_guard_leaves_overscale_for_the_float_target_to_preserve() {
        let mut samples = vec![0.5, 1.85, f32::NAN, -1.85];
        let replaced = replace_non_finite(&mut samples);
        assert_eq!(replaced, 1, "only the NaN is repaired");
        assert_eq!(
            samples,
            [0.5, 1.85, 0.0, -1.85],
            "the overshoot is untouched"
        );

        // A float target writes the guarded samples through verbatim: the
        // overshoot survives into the file, so its clip count is zero.
        let file = decibri_decode::WavWriter::new(
            decibri_decode::AudioSpec::mono(16000),
            decibri_decode::WavCodec::Float32,
        )
        .to_bytes(&samples)
        .expect("the float writer accepts guarded samples");
        let audio = decibri_decode::decode(&file).expect("the float file reads back");
        assert_eq!(audio.samples(), &[0.5, 1.85, 0.0, -1.85]);

        // The clamp is where clipping happens and is counted.
        let clipped = clamp_overscale(&mut samples);
        assert_eq!(clipped, 2);
        assert_eq!(samples, [0.5, 1.0, 0.0, -1.0]);
    }

    /// A destination the filesystem refuses reports `FileWriteFailed` with
    /// the path and a walkable I/O cause. Catches the write failure
    /// surfacing as a decode identity or losing the path.
    #[test]
    fn save_write_failure_reports_file_write_failed() {
        use std::error::Error as _;
        let dest = std::env::temp_dir()
            .join("decibri-save-no-such-directory")
            .join("out.wav");
        let file = File::buffer(grid_samples(400), 16000, 1, FileConfig::default()).unwrap();
        let err = file.save(&dest, SaveOptions::default()).unwrap_err();
        match &err {
            DecibriError::FileWriteFailed { path, .. } => {
                assert_eq!(path, &dest, "the offending path is carried");
            }
            other => panic!("expected FileWriteFailed, got {other:?}"),
        }
        assert!(err.source().is_some(), "the I/O cause is walkable");
    }

    /// An empty source saves as a valid, empty file in every container, and
    /// each reads back as zero samples. Catches a writer that refuses or
    /// corrupts a zero-length payload.
    #[test]
    fn save_empty_source_writes_a_valid_empty_file() {
        for ext in ["wav", "aiff", "flac"] {
            with_save_path(&format!("empty.{ext}"), |path| {
                let file = File::buffer(Vec::new(), 16000, 1, FileConfig::default()).unwrap();
                let report = file.save(path, SaveOptions::default()).unwrap();
                assert_eq!(report.clipped_samples, 0);
                assert_eq!(report.non_finite_samples, 0);
                let back = collect(File::open(path, FileConfig::default()).unwrap());
                assert!(back.is_empty(), "{ext}: zero samples back");
            });
        }
    }

    // ── Channels: delivery, the map, the feed, and the writers ──────────

    /// Collect every conditioned sample a `File` iteration delivers,
    /// asserting each chunk carries the target rate and the expected
    /// delivered channel count. The multichannel counterpart of `collect`.
    fn collect_at(file: File, channels: u16) -> Vec<f32> {
        let rate = file.sample_rate();
        let mut out = Vec::new();
        for chunk in file {
            let chunk = chunk.expect("iteration should not error");
            assert_eq!(chunk.sample_rate, rate);
            assert_eq!(chunk.channels, channels);
            out.extend(chunk.data);
        }
        out
    }

    /// A `FileConfig` naming a delivered channel count, everything else at
    /// the defaults.
    fn channels_config(channels: u16) -> FileConfig {
        FileConfig {
            channels,
            ..Default::default()
        }
    }

    /// A `FileConfig` naming a channel map, with `channels` set to the map's
    /// own length as the map contract requires.
    fn map_config(map: &[u16]) -> FileConfig {
        FileConfig {
            channels: map.len() as u16,
            channel_map: Some(map.to_vec()),
            ..Default::default()
        }
    }

    /// A buffer that is not a whole number of frames is refused before the
    /// source is accepted, with the offending sizes named, and a zero input
    /// channel count is refused as describing no buffer at all. Catches a
    /// buffer form that truncates or rotates instead of refusing.
    #[test]
    fn a_misaligned_buffer_is_refused_and_consumes_nothing() {
        let err = File::buffer(vec![0.0; 7], 16000, 2, channels_config(2))
            .err()
            .expect("the construction should be refused");
        assert!(
            matches!(
                err,
                DecibriError::BlockSizeNotFrameAligned {
                    samples: 7,
                    channels: 2
                }
            ),
            "got {err:?}"
        );
        let err = File::buffer(vec![0.0; 4], 16000, 0, FileConfig::default())
            .err()
            .expect("the construction should be refused");
        assert!(
            matches!(err, DecibriError::ChannelsOutOfRange),
            "got {err:?}"
        );
    }

    /// Without a map, a delivered count above the source's own is refused
    /// with both counts named: decibri delivers source channels and cannot
    /// manufacture one the source does not have.
    #[test]
    fn an_unmapped_over_ask_is_refused_by_the_source_count() {
        let err = File::buffer(grid_samples(8), 16000, 2, channels_config(4))
            .err()
            .expect("the construction should be refused");
        assert!(
            matches!(
                err,
                DecibriError::FileChannelsUnsupported {
                    requested: 4,
                    available: 2
                }
            ),
            "got {err:?}"
        );
    }

    /// Without a map, a strict subset above one is refused, naming the
    /// channel map as the answer: which source channels a subset means has
    /// no single reading, so decibri declines to guess, exactly as capture
    /// does.
    #[test]
    fn an_unmapped_strict_subset_requires_the_map() {
        let err = File::buffer(grid_samples(16), 16000, 8, channels_config(2))
            .err()
            .expect("the construction should be refused");
        assert!(
            matches!(
                err,
                DecibriError::FileChannelSelectionAmbiguous {
                    requested: 2,
                    available: 8
                }
            ),
            "got {err:?}"
        );
        assert!(
            err.to_string().contains("channel map"),
            "the refusal names the channel map: {err}"
        );
    }

    /// The file-side map is validated exactly as capture's: the length must
    /// match the delivered count, and every entry must name a channel the
    /// source has. The length error is the shared variant; the range error
    /// is the file surface's own.
    #[test]
    fn the_file_map_is_validated_like_captures() {
        let config = FileConfig {
            channels: 2,
            channel_map: Some(vec![0]),
            ..Default::default()
        };
        let err = File::buffer(grid_samples(8), 16000, 2, config)
            .err()
            .expect("the construction should be refused");
        assert!(
            matches!(
                err,
                DecibriError::ChannelMapLengthMismatch {
                    entries: 1,
                    channels: 2
                }
            ),
            "got {err:?}"
        );

        let err = File::buffer(grid_samples(8), 16000, 2, map_config(&[0, 5]))
            .err()
            .expect("the construction should be refused");
        assert!(
            matches!(
                err,
                DecibriError::FileChannelMapOutOfRange {
                    index: 5,
                    available: 2
                }
            ),
            "got {err:?}"
        );
    }

    /// A map gathers, permutes and duplicates on the file path exactly as on
    /// the capture path: `[1, 0]` swaps a stereo source, `[1, 1]` duplicates
    /// one channel, `[1]` selects one alone, and `[0, 1, 0, 1]` delivers
    /// more channels than the source has. Sample values prove the gather;
    /// nothing is averaged.
    #[test]
    fn a_map_gathers_permutes_and_duplicates_on_the_file_path() {
        let frames = 500;
        let source = grid_samples(frames * 2);
        let left: Vec<f32> = source.iter().copied().step_by(2).collect();
        let right: Vec<f32> = source.iter().copied().skip(1).step_by(2).collect();

        let swapped = collect_at(
            File::buffer(source.clone(), 16000, 2, map_config(&[1, 0])).unwrap(),
            2,
        );
        let expected: Vec<f32> = right
            .iter()
            .zip(&left)
            .flat_map(|(&r, &l)| [r, l])
            .collect();
        assert_eq!(swapped, expected, "[1, 0] permutes");

        let doubled = collect_at(
            File::buffer(source.clone(), 16000, 2, map_config(&[1, 1])).unwrap(),
            2,
        );
        let expected: Vec<f32> = right.iter().flat_map(|&r| [r, r]).collect();
        assert_eq!(doubled, expected, "[1, 1] duplicates");

        let selected = collect_at(
            File::buffer(source.clone(), 16000, 2, map_config(&[1])).unwrap(),
            1,
        );
        assert_eq!(selected, right, "[1] selects one channel alone");

        let widened = collect_at(
            File::buffer(source.clone(), 16000, 2, map_config(&[0, 1, 0, 1])).unwrap(),
            4,
        );
        let expected: Vec<f32> = left
            .iter()
            .zip(&right)
            .flat_map(|(&l, &r)| [l, r, l, r])
            .collect();
        assert_eq!(
            widened, expected,
            "a map may deliver more than the source has"
        );
    }

    /// Multichannel delivery chunks in whole frames: at three channels a
    /// full chunk is 1600 frames (4800 interleaved samples), the final chunk
    /// carries the remainder, and the concatenation equals the source
    /// exactly, so no chunk boundary rotates the channel identities.
    #[test]
    fn multichannel_delivery_chunks_whole_frames() {
        let frames = 2000;
        let source = grid_samples(frames * 3);
        let file = File::buffer(source.clone(), 16000, 3, channels_config(3)).unwrap();
        let mut sizes = Vec::new();
        let mut out = Vec::new();
        for chunk in file {
            let chunk = chunk.expect("iteration should not error");
            assert_eq!(chunk.channels, 3);
            sizes.push(chunk.data.len());
            out.extend(chunk.data);
        }
        assert_eq!(sizes, vec![4800, 1200], "whole frames per chunk");
        assert_eq!(out, source, "the delivery is exact and unrotated");
    }

    /// At one channel the detector feed takes the untouched path: the drained
    /// feed equals the source byte for byte, so adding the collapse changed
    /// nothing on the mono path.
    #[cfg(feature = "vad")]
    #[test]
    fn the_feed_collapse_is_identity_at_one_channel() {
        let input = sine(16000, 0.3);
        let mut file = File::buffer(input.clone(), 16000, 1, vad_file_config()).unwrap();
        let mut fed = Vec::new();
        loop {
            let Some(chunk) = file.next() else { break };
            chunk.expect("iteration should not error");
            fed.extend(file.vad_input().expect("vad configured"));
        }
        assert_eq!(fed, input);
    }

    /// Above one channel the detector feed is the frame average, the live
    /// path's own collapse: a stereo delivery feeds the detector mono, and
    /// the whole-recording analysis of the stereo source equals the analysis
    /// of its own average. Catches a feed that hands the detector
    /// interleaved multichannel.
    #[cfg(feature = "vad")]
    #[test]
    fn the_feed_collapse_averages_above_one() {
        // Stereo carrying the golden speech on both channels, slightly
        // offset in level so the average is a genuine function of both.
        let mono = golden_samples();
        let stereo: Vec<f32> = mono.iter().flat_map(|&s| [s * 0.8, s * 1.2]).collect();
        let average = crate::sample::downmix_to_mono(&stereo, 2);

        let config = FileConfig {
            channels: 2,
            ..vad_file_config()
        };
        let mut file = File::buffer(stereo.clone(), 16000, 2, config).unwrap();
        let mut fed = Vec::new();
        loop {
            let Some(chunk) = file.next() else { break };
            chunk.expect("iteration should not error");
            fed.extend(file.vad_input().expect("vad configured"));
        }
        assert_eq!(fed, average, "the feed is the frame average");

        let config = FileConfig {
            channels: 2,
            ..vad_file_config()
        };
        let stereo_report = File::buffer(stereo, 16000, 2, config)
            .unwrap()
            .analyze()
            .unwrap();
        let mono_report = File::buffer(average, 16000, 1, vad_file_config())
            .unwrap()
            .analyze()
            .unwrap();
        assert_eq!(stereo_report.scores, mono_report.scores);
        assert_eq!(stereo_report.segments, mono_report.segments);
    }

    /// Collect the detector feed a `File` iteration drains, chunk by chunk.
    #[cfg(feature = "vad")]
    fn collect_feed(mut file: File) -> Vec<f32> {
        let mut fed = Vec::new();
        loop {
            let Some(chunk) = file.next() else { break };
            chunk.expect("iteration should not error");
            fed.extend(file.vad_input().expect("the feed is active"));
        }
        fed
    }

    /// A `DetectorSource::Channel` feeds the named delivered channel alone on
    /// the file surface: a silent channel against a ramp, so a wrong
    /// selection is visible rather than plausible. Naming the silent channel
    /// feeds silence; naming the loud one feeds the ramp, byte for byte. The
    /// same feed `analyze` consumes, so the whole-recording analysis reads
    /// the named channel too.
    #[cfg(feature = "vad")]
    #[test]
    fn a_channel_source_feeds_the_named_channel_on_the_file_surface() {
        let frames = 3200;
        let loud: Vec<f32> = (0..frames).map(|i| (i % 16) as f32 * 0.05 + 0.1).collect();
        // Delivered channel 0 silent, delivered channel 1 loud.
        let stereo: Vec<f32> = loud.iter().flat_map(|&s| [0.0, s]).collect();

        for (source, expected) in [
            (DetectorSource::Channel(0), vec![0.0_f32; frames]),
            (DetectorSource::Channel(1), loud.clone()),
        ] {
            let config = FileConfig {
                channels: 2,
                detector_source: source,
                ..vad_file_config()
            };
            let fed = collect_feed(File::buffer(stereo.clone(), 16000, 2, config).unwrap());
            assert_eq!(fed, expected, "the feed carries {source:?} alone");
        }

        // The default remains the frame average, byte for byte, beside the
        // selecting configurations above.
        let config = FileConfig {
            channels: 2,
            ..vad_file_config()
        };
        let fed = collect_feed(File::buffer(stereo.clone(), 16000, 2, config).unwrap());
        assert_eq!(
            fed,
            crate::sample::downmix_to_mono(&stereo, 2),
            "no source set: the feed is the frame average"
        );
    }

    /// A multichannel delivery keeps the feed active even with no detector
    /// configured and no transform: `vad_input` hands out the mono collapse
    /// of the pre-conditioning signal, so a binding never falls back to the
    /// interleaved delivered chunk. The same three-condition contract as the
    /// live path's `detector_feed`. Regression: an interleaved multichannel
    /// signal reaching a detector as consecutive mono samples.
    #[cfg(feature = "vad")]
    #[test]
    fn a_multichannel_delivery_keeps_the_feed_active_without_vad() {
        let frames = 3200;
        let loud: Vec<f32> = (0..frames).map(|i| (i % 16) as f32 * 0.05 + 0.1).collect();
        let stereo: Vec<f32> = loud.iter().flat_map(|&s| [0.0, s]).collect();

        let config = FileConfig {
            channels: 2,
            ..FileConfig::default()
        };
        let fed = collect_feed(File::buffer(stereo.clone(), 16000, 2, config).unwrap());
        assert_eq!(
            fed,
            crate::sample::downmix_to_mono(&stereo, 2),
            "no VAD, no transform: the feed is the delivery's mono collapse"
        );
    }

    /// The detector source names DELIVERED channels, the position within the
    /// frames a consumer receives, not source channels: under a permuting map
    /// `[1, 0]`, `Channel(0)` feeds source channel 1, the one the map placed
    /// at delivered position 0. Regression: a source index resolved in the
    /// source's own channel space, which reads the other channel here while
    /// every length still agrees.
    #[cfg(feature = "vad")]
    #[test]
    fn a_channel_source_resolves_in_delivered_space_under_a_permuting_map() {
        let frames = 3200;
        let a: Vec<f32> = (0..frames).map(|i| (i % 8) as f32 * 0.1 - 0.35).collect();
        let b: Vec<f32> = a.iter().map(|&s| s + 0.25).collect();
        let stereo: Vec<f32> = a.iter().zip(&b).flat_map(|(&x, &y)| [x, y]).collect();

        let config = FileConfig {
            channels: 2,
            channel_map: Some(vec![1, 0]),
            detector_source: DetectorSource::Channel(0),
            ..vad_file_config()
        };
        let fed = collect_feed(File::buffer(stereo, 16000, 2, config).unwrap());
        assert_eq!(
            fed, b,
            "delivered channel 0 carries source channel 1 under the map, and the feed reads it"
        );
    }

    /// Under a duplicating map `[0, 0]` every delivered position carries
    /// source channel 0, and the detector source names a position: naming
    /// delivered channel 1 feeds source channel 0's samples, never source
    /// channel 1's. The delivered space makes a duplicated channel
    /// unambiguous by position, which is what the index space is for.
    #[cfg(feature = "vad")]
    #[test]
    fn a_channel_source_under_a_duplicating_map_reads_the_delivered_position() {
        let frames = 3200;
        let a: Vec<f32> = (0..frames).map(|i| (i % 8) as f32 * 0.1 - 0.35).collect();
        let b: Vec<f32> = a.iter().map(|&s| s + 0.25).collect();
        let stereo: Vec<f32> = a.iter().zip(&b).flat_map(|(&x, &y)| [x, y]).collect();

        let config = FileConfig {
            channels: 2,
            channel_map: Some(vec![0, 0]),
            detector_source: DetectorSource::Channel(1),
            ..vad_file_config()
        };
        let fed = collect_feed(File::buffer(stereo, 16000, 2, config).unwrap());
        assert_eq!(
            fed, a,
            "delivered channel 1 carries source channel 0 under the duplicating map"
        );
    }

    /// A detector source at or above the configured delivered count is
    /// refused at construction, against that count alone: a large delivered
    /// count admits a correspondingly large index. The negative control
    /// against a fixed maximum reintroduced on the index.
    #[test]
    fn a_detector_source_out_of_range_is_refused_at_construction() {
        let config = FileConfig {
            channels: 2,
            detector_source: DetectorSource::Channel(2),
            ..FileConfig::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0; 64], 16000, 2, config),
            Err(DecibriError::DetectorSourceOutOfRange {
                index: 2,
                channels: 2
            })
        ));

        // No fixed maximum: 1024 delivered channels admit index 1023.
        let config = FileConfig {
            channels: 1024,
            detector_source: DetectorSource::Channel(1023),
            ..FileConfig::default()
        };
        assert!(
            File::buffer(vec![0.0; 1024], 16000, 1024, config).is_ok(),
            "the delivered count is the only ceiling"
        );
    }

    /// THE round trip: a three-channel source saved and re-read reproduces
    /// its channels in order, exactly, through WAV and FLAC alike. The first
    /// end-to-end proof that what goes in multichannel comes back
    /// multichannel, unaveraged and unrotated.
    #[test]
    fn save_round_trip_preserves_channels_in_order() {
        let source = grid_samples(2000 * 3);
        for ext in ["wav", "flac"] {
            with_save_path(&format!("channels.{ext}"), |path| {
                let file = File::buffer(source.clone(), 16000, 3, channels_config(3)).unwrap();
                let report = file.save(path, SaveOptions::default()).unwrap();
                assert_eq!(report.clipped_samples, 0, "{ext}: nothing to clip");
                let back = collect_at(File::open(path, channels_config(3)).unwrap(), 3);
                assert_eq!(back, source, "{ext}: the channels survive in order");
            });
        }
    }

    /// FLAC's channel ceiling is the container's own: nine channels are
    /// refused with the container layer's exact text forwarded, and eight,
    /// the format's limit itself, are written and read back exactly.
    #[test]
    fn flac_limit_is_the_containers_own_at_eight() {
        with_save_path("nine.flac", |path| {
            let file = File::buffer(grid_samples(9), 16000, 9, channels_config(9)).unwrap();
            let err = file
                .save(path, SaveOptions::default())
                .expect_err("the save should be refused");
            match &err {
                DecibriError::AudioFormatUnsupported { reason } => {
                    assert_eq!(
                        reason, "9-channel audio is not a supported layout",
                        "the refusal is the container layer's own text"
                    );
                }
                other => panic!("expected AudioFormatUnsupported, got {other:?}"),
            }
        });

        let source = grid_samples(200 * 8);
        with_save_path("eight.flac", |path| {
            let file = File::buffer(source.clone(), 16000, 8, channels_config(8)).unwrap();
            file.save(path, SaveOptions::default()).unwrap();
            let back = collect_at(File::open(path, channels_config(8)).unwrap(), 8);
            assert_eq!(back, source, "the limit itself is accepted");
        });
    }

    /// WAV's ceiling is `nBlockAlign`, a 16-bit field holding the frame's
    /// byte size: at the 16-bit samples decibri writes, 32767 channels fit
    /// and 32768 do not. Both sides of the boundary are the container
    /// layer's answer, not decibri's.
    #[test]
    fn wav_limit_is_block_align_not_decibris() {
        with_save_path("over.wav", |path| {
            let file =
                File::buffer(vec![0.0; 32768], 16000, 32768, channels_config(32768)).unwrap();
            let err = file
                .save(path, SaveOptions::default())
                .expect_err("the save should be refused");
            assert!(
                matches!(err, DecibriError::AudioFormatUnsupported { .. }),
                "got {err:?}"
            );
        });
        with_save_path("wide.wav", |path| {
            let file =
                File::buffer(grid_samples(32767), 16000, 32767, channels_config(32767)).unwrap();
            file.save(path, SaveOptions::default()).unwrap();
            let back = collect_at(File::open(path, channels_config(32767)).unwrap(), 32767);
            assert_eq!(back.len(), 32767, "the boundary itself is accepted");
        });
    }

    /// The negative control on the channel count: where no container limit
    /// applies, the whole path serves `u16::MAX` channels, delivery and AIFF
    /// save alike. An invented decibri-side maximum added anywhere on the
    /// file path fails here loudly.
    #[test]
    fn no_invented_channel_maximum_on_the_file_path() {
        let channels = u16::MAX;
        let source = grid_samples(channels as usize);
        let delivered = collect_at(
            File::buffer(source.clone(), 16000, channels, channels_config(channels)).unwrap(),
            channels,
        );
        assert_eq!(delivered, source, "delivery serves u16::MAX channels");

        with_save_path("widest.aiff", |path| {
            let file =
                File::buffer(source.clone(), 16000, channels, channels_config(channels)).unwrap();
            file.save(path, SaveOptions::default()).unwrap();
            let back = collect_at(
                File::open(path, channels_config(channels)).unwrap(),
                channels,
            );
            assert_eq!(back, source, "AIFF carries the count unbounded");
        });
    }
}
