//! Offline audio source.
//!
//! A [`File`] conditions audio you already have, a WAV recording on disk or
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
//! [`FileConfig`], then open a WAV path with [`File::open`] (or the
//! [`File::new`] alias) or wrap in-memory samples with [`File::buffer`]. The
//! path form reads the input rate and channel count from the WAV header; the
//! buffer form takes the input rate explicitly because raw samples carry no
//! header.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};

use crate::error::DecibriError;
use crate::microphone::{AudioChunk, DenoiseModel, HighpassFilter};
use crate::sample;
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

/// Number of output samples per delivered chunk (mono at the target rate),
/// matching the microphone's default `frames_per_buffer`. The final chunk may
/// be shorter once the chain's end-of-stream tail has been drained.
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
    /// Validate the configuration: the target sample rate, the AGC target, and
    /// the limiter ceiling (each when set) must fall within the supported
    /// ranges, matching the live capture path's validation exactly. With VAD
    /// configured, the detector threshold is validated fail-fast as well.
    ///
    /// # Errors
    /// - [`DecibriError::SampleRateOutOfRange`] when `sample_rate` is outside
    ///   1000-384000.
    /// - [`DecibriError::AgcTargetOutOfRange`] when `agc` is outside -40..=-3.
    /// - [`DecibriError::LimiterCeilingOutOfRange`] when `limiter` is outside
    ///   -3.0..=0.0.
    /// - [`DecibriError::VadThresholdOutOfRange`] when the VAD threshold is
    ///   outside 0.0..=1.0.
    pub fn validate(&self) -> Result<(), DecibriError> {
        if !(1000..=384000).contains(&self.sample_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
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
    /// The source's native rate in Hz (from the WAV header, or the explicit
    /// `input_rate` of [`File::buffer`]).
    input_rate: u32,
    /// The source's channel count (from the WAV header; 1 for buffers).
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
    /// Delivered-output re-block buffer: conditioned mono samples at the
    /// target rate, drained in [`DELIVERY_FRAMES`] chunks.
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
    /// Segment-merge holdoff in milliseconds of file time.
    #[cfg(feature = "vad")]
    vad_holdoff_ms: u32,
}

impl File {
    /// Open a WAV file and prepare it for conditioning (and, with VAD
    /// configured, analysis). Reads the whole file; the input rate and
    /// channel count come from the WAV header. Supports 16-bit PCM and 32-bit
    /// float WAV encodings; other formats are owned by a separate conversion
    /// feature, keeping this path dependency-free.
    ///
    /// # Errors
    /// - [`DecibriError::FileReadFailed`] when the file cannot be read.
    /// - [`DecibriError::WavInvalid`] when the bytes are not a supported WAV.
    /// - Configuration errors exactly as [`FileConfig::validate`] reports.
    pub fn open(path: impl AsRef<Path>, config: FileConfig) -> Result<Self, DecibriError> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|source| DecibriError::FileReadFailed {
            path: path.to_path_buf(),
            source,
        })?;
        let wav = parse_wav(&bytes)?;
        Self::from_source(wav.samples, wav.sample_rate, wav.channels, config)
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
    /// VAD configured, analysis). `samples` are mono f32 in [-1.0, 1.0] at
    /// `input_rate`; raw samples carry no header, so the rate is explicit.
    ///
    /// # Errors
    /// - [`DecibriError::SampleRateOutOfRange`] when `input_rate` is outside
    ///   1000-384000.
    /// - Configuration errors exactly as [`FileConfig::validate`] reports.
    pub fn buffer(
        samples: Vec<f32>,
        input_rate: u32,
        config: FileConfig,
    ) -> Result<Self, DecibriError> {
        if !(1000..=384000).contains(&input_rate) {
            return Err(DecibriError::SampleRateOutOfRange);
        }
        Self::from_source(samples, input_rate, 1, config)
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

        // Denoise is enabled only when a model AND its path are both set,
        // exactly as on the live path.
        let denoise = config
            .denoise
            .zip(config.denoise_model_path.as_deref())
            .map(|(model, path)| (model, path, config.ort_library_path.as_deref()));
        let stage = build_capture_stage(
            input_channels,
            1,
            input_rate,
            target_rate,
            Transforms {
                dc_removal: config.dc_removal,
                denoise,
                highpass: config.highpass,
                agc: config.agc,
                limiter: config.limiter,
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
            vad_holdoff_ms: config.vad_holdoff_ms,
        })
    }

    /// The target output rate every delivered chunk carries.
    pub fn sample_rate(&self) -> u32 {
        self.target_rate
    }

    /// The source's native rate, from the WAV header or the explicit
    /// [`File::buffer`] input rate.
    pub fn input_rate(&self) -> u32 {
        self.input_rate
    }

    /// Whether the detector feed is maintained: when VAD is configured, or
    /// when the chain has a conditioning transform (the delivered output then
    /// differs from the pre-conditioning signal a detector should read),
    /// exactly as the live capture path keeps its tap.
    fn feed_active(&self) -> bool {
        #[cfg(feature = "vad")]
        {
            self.vad.is_some() || self.stage.as_ref().is_some_and(CaptureStage::has_transform)
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
    /// pre-conditioning signal, in file order. With VAD configured the feed
    /// is at [`vad_rate`](Self::vad_rate); with only a conditioning transform
    /// it stays at the target rate. Returns `None` when the feed is inactive
    /// (no VAD and no transform), in which case the delivered chunk already
    /// is the pre-conditioning signal, exactly as
    /// [`crate::microphone::MicrophoneStream::vad_input`] contracts.
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
    /// Runs the conditioning pass once, scoring the pre-conditioning signal
    /// window by window (512 samples at 16 kHz, 256 at 8 kHz), then merges
    /// consecutive speech windows whose silence gaps are within
    /// [`FileConfig::vad_holdoff_ms`] of file time. Consumes the `File`: the
    /// analysis is a single pass, separate from iteration.
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

        // Drive the whole source through the chain, scoring the detector feed
        // one exact window at a time so each call yields exactly one score.
        // The delivered audio is discarded as it accumulates: analysis
        // returns the report, not the conditioned chunks.
        let mut probabilities: Vec<f32> = Vec::new();
        let mut pending: Vec<f32> = Vec::new();
        while !self.flushed {
            self.advance()?;
            self.reblock.clear();
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

    /// Feed source blocks through the chain until the re-block buffer can
    /// deliver one chunk or the source is exhausted and flushed. One step of
    /// the single pass; both iteration and analysis drive it.
    fn advance(&mut self) -> Result<(), DecibriError> {
        while self.reblock.len() < DELIVERY_FRAMES && !self.flushed {
            if self.pos < self.source.len() {
                let feed = FEED_FRAMES * self.input_channels as usize;
                let end = (self.pos + feed).min(self.source.len());
                let range = self.pos..end;
                self.pos = end;
                self.ingest(range)?;
            } else {
                self.flush_chain()?;
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
            self.push_vad_feed();
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
            self.push_vad_feed();
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
    /// the detector feed, resampling to the detector rate when the two
    /// differ, and enforce the feed's memory bound.
    #[cfg(feature = "vad")]
    fn push_vad_feed(&mut self) {
        if self.scratch.is_empty() {
            return;
        }
        match &mut self.vad_resampler {
            Some(resampler) => {
                let mut out = Vec::new();
                resampler.process(&self.scratch, &mut out);
                self.vad_queue.extend(out);
            }
            None => self.vad_queue.extend(self.scratch.iter().copied()),
        }
        self.cap_vad_queue();
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
}

impl Iterator for File {
    type Item = Result<AudioChunk, DecibriError>;

    /// Deliver the next conditioned chunk: [`DELIVERY_FRAMES`] mono samples
    /// at the target rate, with a possibly shorter final chunk once the
    /// chain's end-of-stream tail has been drained. Returns `None` after the
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
        let take = self.reblock.len().min(DELIVERY_FRAMES);
        if take == 0 {
            self.finished = true;
            return None;
        }
        let data: Vec<f32> = self.reblock.drain(..take).collect();
        if self.flushed && self.reblock.is_empty() {
            self.finished = true;
        }
        Some(Ok(AudioChunk {
            data,
            sample_rate: self.target_rate,
            channels: 1,
        }))
    }
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

/// A parsed WAV source: interleaved f32 samples plus the header's format.
struct WavSource {
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
}

/// Parse a WAV file's bytes: RIFF/WAVE with a `fmt ` chunk describing 16-bit
/// PCM (format 1) or 32-bit IEEE float (format 3), any channel count, any
/// rate, followed by a `data` chunk. The minimal reader keeps this path free
/// of any decode dependency; other encodings are owned by a separate
/// conversion feature.
fn parse_wav(bytes: &[u8]) -> Result<WavSource, DecibriError> {
    fn u16_at(bytes: &[u8], at: usize) -> u16 {
        u16::from_le_bytes([bytes[at], bytes[at + 1]])
    }
    fn u32_at(bytes: &[u8], at: usize) -> u32 {
        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
    }
    let invalid = |reason: &'static str| DecibriError::WavInvalid { reason };

    if bytes.len() < 12 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err(invalid("not a RIFF/WAVE file"));
    }

    // Walk the chunk list for `fmt ` and `data`. Chunk bodies are padded to
    // an even byte count in the file.
    let mut fmt: Option<(u16, u16, u32, u16)> = None; // format, channels, rate, bits
    let mut data: Option<(usize, usize)> = None; // offset, length
    let mut offset = 12usize;
    while offset + 8 <= bytes.len() {
        let id = &bytes[offset..offset + 4];
        let size = u32_at(bytes, offset + 4) as usize;
        let body = offset + 8;
        if body + size > bytes.len() {
            return Err(invalid("chunk extends past end of file"));
        }
        if id == b"fmt " {
            if size < 16 {
                return Err(invalid("fmt chunk too short"));
            }
            fmt = Some((
                u16_at(bytes, body),
                u16_at(bytes, body + 2),
                u32_at(bytes, body + 4),
                u16_at(bytes, body + 14),
            ));
        } else if id == b"data" {
            data = Some((body, size));
            break;
        }
        offset = body + size + (size & 1);
    }

    let (format, channels, sample_rate, bits) = fmt.ok_or(invalid("missing fmt chunk"))?;
    let (data_offset, data_len) = data.ok_or(invalid("missing data chunk"))?;
    if channels == 0 {
        return Err(invalid("channel count is zero"));
    }
    if !(1000..=384000).contains(&sample_rate) {
        return Err(DecibriError::SampleRateOutOfRange);
    }

    let payload = &bytes[data_offset..data_offset + data_len];
    let samples = match (format, bits) {
        (1, 16) => sample::i16_le_bytes_to_f32(payload),
        (3, 32) => sample::f32_le_bytes_to_f32(payload),
        _ => {
            return Err(invalid(
                "unsupported encoding; 16-bit PCM or 32-bit float required",
            ))
        }
    };

    Ok(WavSource {
        samples,
        sample_rate,
        channels,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Interleave a mono signal into a synthetic WAV byte vector: 16-bit PCM
    /// (format 1) or 32-bit float (format 3), any rate and channel count.
    fn wav_bytes(format: u16, channels: u16, rate: u32, samples: &[f32]) -> Vec<u8> {
        let payload = match format {
            1 => sample::f32_to_i16_le_bytes(samples),
            3 => sample::f32_to_f32_le_bytes(samples),
            _ => unreachable!("test formats are 1 and 3"),
        };
        let bits: u16 = if format == 1 { 16 } else { 32 };
        let block_align = channels * bits / 8;
        let byte_rate = rate * u32::from(block_align);
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"RIFF");
        bytes.extend_from_slice(&(36 + payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(b"WAVE");
        bytes.extend_from_slice(b"fmt ");
        bytes.extend_from_slice(&16u32.to_le_bytes());
        bytes.extend_from_slice(&format.to_le_bytes());
        bytes.extend_from_slice(&channels.to_le_bytes());
        bytes.extend_from_slice(&rate.to_le_bytes());
        bytes.extend_from_slice(&byte_rate.to_le_bytes());
        bytes.extend_from_slice(&block_align.to_le_bytes());
        bytes.extend_from_slice(&bits.to_le_bytes());
        bytes.extend_from_slice(b"data");
        bytes.extend_from_slice(&(payload.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&payload);
        bytes
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
        let file = File::buffer(input.clone(), 16000, FileConfig::default()).unwrap();
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
        let file = File::buffer(input.clone(), 48000, FileConfig::default()).unwrap();
        let out = collect(file);

        let mut resampler = PolyphaseResampler::new(48000, 16000).unwrap();
        let mut expected = Vec::new();
        for block in input.chunks(FEED_FRAMES) {
            resampler.process(block, &mut expected);
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
        let file = File::buffer(input.clone(), 16000, config).unwrap();
        let out = collect(file);

        let mut chain = build_capture_stage(
            1,
            1,
            16000,
            16000,
            Transforms {
                dc_removal: true,
                denoise: None,
                highpass: None,
                agc: None,
                limiter: None,
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
        let file = File::buffer(Vec::new(), 16000, FileConfig::default()).unwrap();
        assert_eq!(collect(file).len(), 0);
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

    /// Bytes that are not RIFF/WAVE, a truncated chunk, and an unsupported
    /// encoding each report the typed WAV parse failure.
    #[test]
    fn parse_rejects_invalid_wav() {
        assert!(matches!(
            parse_wav(b"not a wav file at all"),
            Err(DecibriError::WavInvalid { .. })
        ));

        let mut truncated = wav_bytes(1, 1, 16000, &sine(16000, 0.1));
        truncated.truncate(60);
        assert!(matches!(
            parse_wav(&truncated),
            Err(DecibriError::WavInvalid { .. })
        ));

        // 8-bit PCM is unsupported: rewrite the bits-per-sample field.
        let mut eight_bit = wav_bytes(1, 1, 16000, &sine(16000, 0.1));
        eight_bit[34] = 8;
        eight_bit[35] = 0;
        assert!(matches!(
            parse_wav(&eight_bit),
            Err(DecibriError::WavInvalid { .. })
        ));
    }

    /// Configuration validation matches the live path: rate, AGC target, and
    /// limiter ceiling ranges are enforced at construction.
    #[test]
    fn config_validation_matches_live_ranges() {
        let config = FileConfig {
            sample_rate: 999,
            ..Default::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0], 16000, config),
            Err(DecibriError::SampleRateOutOfRange)
        ));

        let config = FileConfig {
            agc: Some(-2),
            ..Default::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0], 16000, config),
            Err(DecibriError::AgcTargetOutOfRange)
        ));

        let config = FileConfig {
            limiter: Some(0.5),
            ..Default::default()
        };
        assert!(matches!(
            File::buffer(vec![0.0], 16000, config),
            Err(DecibriError::LimiterCeilingOutOfRange)
        ));

        assert!(matches!(
            File::buffer(vec![0.0], 999, FileConfig::default()),
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

    #[cfg(feature = "vad")]
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
        let file = File::buffer(sine(16000, 0.1), 16000, FileConfig::default()).unwrap();
        assert!(matches!(
            file.analyze(),
            Err(DecibriError::VadNotConfigured)
        ));
        let file = File::buffer(sine(16000, 0.1), 16000, FileConfig::default()).unwrap();
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
        let mut file = File::buffer(sine(16000, 0.5), 16000, FileConfig::default()).unwrap();
        let first = file.next();
        assert!(matches!(first, Some(Ok(_))), "the first chunk is delivered");
        assert!(matches!(file.analyze(), Err(DecibriError::FileEngaged)));

        let mut file = File::buffer(sine(16000, 0.5), 16000, FileConfig::default()).unwrap();
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
            let mut file = File::buffer(sine(16000, 0.5), 16000, FileConfig::default()).unwrap();
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
                let _ = f.by_ref().peekable().peek().is_some();
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
        let mut file = File::buffer(Vec::new(), 16000, FileConfig::default()).unwrap();
        assert!(file.next().is_none(), "an empty source delivers no chunk");
        assert!(matches!(file.analyze(), Err(DecibriError::FileEngaged)));

        // A drained source: iteration ran to its end, which is the shape a
        // caller reaches by streaming the whole recording first.
        let mut file = File::buffer(sine(16000, 0.5), 16000, FileConfig::default()).unwrap();
        file.by_ref().for_each(drop);
        assert!(matches!(file.analyze(), Err(DecibriError::FileEngaged)));
    }

    /// A drained `File` keeps ending quietly: `next()` goes on returning
    /// `None` rather than reporting the engaged state, which belongs to
    /// analysis alone.
    #[test]
    fn iteration_past_exhaustion_stays_quiet() {
        let mut file = File::buffer(sine(16000, 0.5), 16000, FileConfig::default()).unwrap();
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
        let samples = parse_wav(&bytes).unwrap().samples;
        let mut detector = SileroVad::new(VadConfig {
            model_path: silero_model_path(),
            ..VadConfig::default()
        })
        .unwrap();
        let mut expected: Vec<f32> = Vec::new();
        for window in samples.chunks_exact(512) {
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
        let file = File::buffer(sine(8000, 0.1), 8000, config).unwrap();
        assert_eq!(file.vad_rate(), Some(8000));

        let file = File::buffer(sine(16000, 0.1), 16000, FileConfig::default()).unwrap();
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
        let mut file = File::buffer(input.clone(), 16000, config).unwrap();
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

        let mut plain = File::buffer(input, 16000, FileConfig::default()).unwrap();
        assert!(plain.vad_input().is_none());
    }
}
