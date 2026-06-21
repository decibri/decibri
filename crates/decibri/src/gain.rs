//! Adaptive level control for the capture chain.
//!
//! [`LevelControl`] is one internal engine that drives a captured signal's level
//! toward a target by applying a smoothed, rate-limited gain. It is a same-length,
//! sample-in-sample-out stage: it reads one sample and writes one sample, carrying
//! its level estimate and current gain across blocks, so it wraps through the
//! chain's `InPlace` adapter exactly like the DC blocker and the high-pass biquad.
//! It adds no algorithmic delay (no look-ahead), so it declares zero latency.
//!
//! The engine is built around a [`LevelMode`] seam that selects what it measures
//! and how fast its gain envelope moves. Today the only mode is [`LevelMode::Agc`]
//! (automatic gain control: a broadband RMS level driven toward a dBFS target with
//! a fast envelope). The seam is the extension point for a future loudness mode
//! (a K-weighted loudness target with a slower envelope); adding it is a new
//! variant plus its measurement and rate constants, with no change to the control
//! loop, the chain wiring, or the byte-identical off path.

use crate::stage::InPlaceDsp;

/// Convert a level in decibels relative to full scale to a linear amplitude
/// (`1.0` = full scale). `0 dBFS` maps to `1.0`, `-20 dBFS` to `0.1`.
fn db_to_linear(db: f32) -> f32 {
    10.0_f32.powf(db / 20.0)
}

/// The measurement-and-envelope strategy a [`LevelControl`] engine runs.
///
/// One internal engine backs every adaptive level capability; the mode selects
/// what the engine measures per sample and how fast its gain envelope moves.
/// Today the only mode is [`LevelMode::Agc`]. A future loudness mode (a
/// K-weighted loudness target with a slower envelope) is a new variant here plus
/// its per-sample weighting and rate constants, leaving the control loop, the
/// chain wiring, and the off path untouched.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LevelMode {
    /// Automatic gain control: the broadband RMS level is driven toward a dBFS
    /// target with a fast gain envelope (fast attack, rate-limited release).
    Agc,
}

/// Adaptive gain control: measures the running input level and moves a smoothed
/// gain toward the level that brings the output to the target, with no look-ahead.
///
/// # Measurement
/// The level estimate is a running mean of the squared input, a momentary RMS
/// measure smoothed with a roughly 400 ms time constant. A fast-start blend
/// (running average until the exponential time constant takes over) primes the
/// estimate from the opening samples, so the engine reaches a correct gain
/// decision within tens of milliseconds rather than over the full window. The
/// estimate is per-sample, so processing is independent of how the stream is
/// split into blocks.
///
/// # Gain envelope
/// The gain moves toward the target asymmetrically: downward corrections (the
/// input got louder, the gain must drop) use a fast attack so a loud onset is
/// pulled down promptly, while upward corrections (the input is quiet, the gain
/// must rise) are rate-limited to a gentle steady-state slope so the engine does
/// not pump or chase quiet gaps. During a brief opening window the upward slope is
/// faster (warm-up), so the opening reaches the target without a long ramp.
///
/// # Cold start
/// At stream open the gain is unity, so the first sample is delivered at its
/// natural level (never boosted, never silent). The level estimate primes from
/// the opening samples; the warm-up window then lets the gain reach the target
/// within tens of milliseconds of the first present signal. The warm-up is spent
/// only while signal is present, so an opening pause does not burn it before the
/// first speech arrives. The upward gain rise is gated on signal presence (held
/// while the level sits below the noise floor) so a silent or noise-only opening
/// is not boosted, with a short hangover so the gate does not flutter on brief
/// inter-word gaps. Downward correction is never gated: a loud onset always pulls
/// the gain down.
///
/// This engine manages average level, not peaks. It can wind the gain up on a
/// quiet passage, so a loud transient that follows can exit above full scale;
/// catching that peak is the limiter stage's job. Unity start, the fast attack,
/// the gain cap, and the noise-floor gate bound the gain, but they do not by
/// themselves keep every output sample within full scale.
///
/// # State lifecycle
/// Holds its level estimate, current gain, sample count, warm-up countdown, and
/// hangover countdown across blocks; initialised sensibly at construction; no
/// end-of-stream tail (same-length, so it keeps the default no-op flush); built
/// fresh per stream open with no reset path.
pub(crate) struct LevelControl {
    /// The active measurement-and-envelope strategy. Read per sample to select
    /// the weighting; the seam a future loudness mode extends.
    mode: LevelMode,
    /// Target level as a linear RMS amplitude (`1.0` = full scale), from the
    /// requested dBFS target.
    target_linear: f32,
    /// Presence threshold as a linear amplitude: the noise floor below which the
    /// upward gain rise is gated, so noise and silence are not boosted.
    noise_floor_linear: f32,
    /// Largest gain the engine may apply (a linear bound from the max-gain cap),
    /// so the loop cannot wind gain up without limit on a near-silent input.
    max_gain_linear: f32,
    /// Smallest gain the engine may apply (a linear bound), so a hot input is
    /// attenuated only down to a sane floor.
    min_gain_linear: f32,
    /// Per-sample smoothing coefficient for the level estimate (the steady-state
    /// roughly 400 ms time constant).
    level_coeff: f32,
    /// Per-sample multiplicative cap for the steady-state upward gain rise (the
    /// gentle release slope).
    rise_step: f32,
    /// Per-sample smoothing coefficient for the downward gain correction (the
    /// fast attack).
    attack_coeff: f32,
    /// Per-sample smoothing coefficient for the upward gain rise during the
    /// opening warm-up window (faster than the steady-state slope).
    warmup_coeff: f32,
    /// Number of opening samples over which the warm-up upward slope applies,
    /// counted down per sample.
    warmup_remaining: u64,
    /// Number of samples the presence gate stays open after the level last
    /// crossed the noise floor (the hangover), counted down per sample.
    hangover_samples: u64,
    /// Remaining hangover after the level last sat above the noise floor.
    hangover_remaining: u64,
    /// Running mean of the squared input: the momentary level estimate.
    level_sq: f32,
    /// Samples processed so far, used by the fast-start blend to prime the level
    /// estimate from the opening samples. Saturates; only the opening matters.
    sample_count: u64,
    /// Current applied gain (linear). Unity at construction; the first sample is
    /// delivered at this gain before the gain moves, so the opening is unboosted.
    gain: f32,
}

impl LevelControl {
    /// Steady-state level-measurement time constant in seconds (the momentary
    /// RMS window the envelope rides on).
    const LEVEL_TC_SECS: f32 = 0.4;
    /// Steady-state maximum upward gain change, in decibels per second (the
    /// gentle release slope that avoids pumping).
    const RISE_DB_PER_SEC: f32 = 6.0;
    /// Downward-correction (attack) gain-smoothing time constant in seconds. The
    /// gain follows a change in the desired gain with this constant; because the
    /// desired gain tracks the slower level estimate, the end-to-end pull-down of
    /// a loud onset takes tens of ms, not this constant.
    const ATTACK_TC_SECS: f32 = 0.010;
    /// Upward-rise time constant during the opening warm-up window in seconds,
    /// faster than the steady-state slope so the opening reaches target quickly.
    const WARMUP_TC_SECS: f32 = 0.020;
    /// Length of the opening warm-up window in seconds.
    const WARMUP_SECS: f32 = 0.1;
    /// Presence-gate hangover in seconds: how long the gate stays open after the
    /// level last crossed the noise floor, so it does not flutter on brief gaps.
    const HANGOVER_SECS: f32 = 0.2;
    /// Noise floor in dBFS: the level below which the upward gain rise is gated,
    /// so noise and silence are never boosted toward the target.
    const NOISE_FLOOR_DBFS: f32 = -50.0;
    /// Maximum gain in decibels: the cap on amplification (and, mirrored, the
    /// floor on attenuation) that bounds the control loop.
    const MAX_GAIN_DB: f32 = 30.0;

    /// Build an AGC-mode engine: drive the broadband RMS level toward `target_db`
    /// dBFS for a stream at `sample_rate` Hz. The attack, release, warm-up,
    /// max-gain cap, and noise-floor gate are fixed internal defaults; the target
    /// is the only caller-chosen value. The caller is responsible for keeping
    /// `target_db` in the validated range; it is used as the target level and
    /// does not change which defaults apply.
    pub(crate) fn agc(target_db: i8, sample_rate: u32) -> Self {
        let sr = sample_rate as f32;
        // Per-sample smoothing coefficient for a one-pole filter at `tc` seconds.
        let coeff = |tc: f32| 1.0 - (-1.0 / (tc * sr)).exp();
        Self {
            mode: LevelMode::Agc,
            target_linear: db_to_linear(target_db as f32),
            noise_floor_linear: db_to_linear(Self::NOISE_FLOOR_DBFS),
            max_gain_linear: db_to_linear(Self::MAX_GAIN_DB),
            min_gain_linear: db_to_linear(-Self::MAX_GAIN_DB),
            level_coeff: coeff(Self::LEVEL_TC_SECS),
            // A per-sample multiplicative step that compounds to RISE_DB_PER_SEC
            // over one second.
            rise_step: db_to_linear(Self::RISE_DB_PER_SEC / sr),
            attack_coeff: coeff(Self::ATTACK_TC_SECS),
            warmup_coeff: coeff(Self::WARMUP_TC_SECS),
            warmup_remaining: (Self::WARMUP_SECS * sr) as u64,
            hangover_samples: (Self::HANGOVER_SECS * sr) as u64,
            hangover_remaining: 0,
            level_sq: 0.0,
            sample_count: 0,
            gain: 1.0,
        }
    }

    /// Per-sample input weighting, the mode seam. AGC measures the broadband
    /// sample unchanged; a future loudness mode would K-weight it here.
    fn weigh(&self, sample: f32) -> f32 {
        match self.mode {
            LevelMode::Agc => sample,
        }
    }
}

impl InPlaceDsp for LevelControl {
    fn process_in_place(&mut self, samples: &mut [f32]) {
        for sample in samples.iter_mut() {
            let x = *sample;
            // Apply the current gain first, then update the gain for the next
            // sample. So the first sample is delivered at the unity gain set at
            // construction (unboosted), and the gain lags the measurement by one
            // sample (a feedback loop, no look-ahead, zero added latency).
            *sample = x * self.gain;

            // Advance the level estimate. A fast-start blend (a running average
            // until the steady-state coefficient takes over) primes the estimate
            // from the opening samples, so the gain decision is correct within
            // tens of milliseconds rather than over the full window.
            let weighted = self.weigh(x);
            let coeff = self.level_coeff.max(1.0 / (self.sample_count + 1) as f32);
            self.level_sq += coeff * (weighted * weighted - self.level_sq);
            self.sample_count = self.sample_count.saturating_add(1);
            let level = self.level_sq.sqrt();

            // Presence gate (the noise-floor guard) with a hangover, so a silent
            // or noise-only stretch holds the upward rise while a brief inter-word
            // gap does not.
            let present = if level > self.noise_floor_linear {
                self.hangover_remaining = self.hangover_samples;
                true
            } else if self.hangover_remaining > 0 {
                self.hangover_remaining -= 1;
                true
            } else {
                false
            };

            if present {
                let desired =
                    (self.target_linear / level).clamp(self.min_gain_linear, self.max_gain_linear);
                if desired > self.gain {
                    // Upward: gated on presence (true here), rate-limited. Fast
                    // during the opening warm-up, then the gentle steady slope.
                    if self.warmup_remaining > 0 {
                        self.gain += (desired - self.gain) * self.warmup_coeff;
                    } else {
                        self.gain = (self.gain * self.rise_step).min(desired);
                    }
                } else {
                    // Downward (attack): fast, never gated, so a loud onset is
                    // always pulled down promptly.
                    self.gain += (desired - self.gain) * self.attack_coeff;
                }

                // Spend the warm-up window only while signal is present, so a
                // silent or noise-only opening does not burn the fast-start
                // before the first real signal arrives. The fast convergence the
                // warm-up buys is then applied to that first signal, not wasted
                // on an opening pause.
                if self.warmup_remaining > 0 {
                    self.warmup_remaining -= 1;
                }
            }
            // When not present the gain is frozen and the warm-up is preserved:
            // neither rise nor fall, so the level is held through silence and the
            // fast-start is saved for the first present signal.
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: u32 = 16_000;
    const TARGET_DB: i8 = -18;

    /// Root-mean-square of a slice.
    fn rms(samples: &[f32]) -> f32 {
        if samples.is_empty() {
            return 0.0;
        }
        (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt()
    }

    /// A mono sine of `seconds` at `freq_hz` with linear amplitude `amp`.
    fn tone(amp: f32, freq_hz: f32, seconds: f32) -> Vec<f32> {
        let n = (seconds * SR as f32) as usize;
        (0..n)
            .map(|i| amp * (2.0 * std::f32::consts::PI * freq_hz * i as f32 / SR as f32).sin())
            .collect()
    }

    /// The linear amplitude of a sine whose RMS sits at `db` dBFS.
    fn amp_for_dbfs(db: f32) -> f32 {
        // RMS of a sine is amp / sqrt(2); solve amp so 20*log10(rms) == db.
        db_to_linear(db) * std::f32::consts::SQRT_2
    }

    /// Process `block` and return a copy of the conditioned output.
    fn run(lc: &mut LevelControl, block: &[f32]) -> Vec<f32> {
        let mut buf = block.to_vec();
        lc.process_in_place(&mut buf);
        buf
    }

    /// Cold start (a): the first sample is delivered at unity gain. With the gain
    /// applied before it moves, the opening sample is byte-identical to the input
    /// for any opening level, hot or quiet, so the open is never a boost.
    #[test]
    fn cold_start_first_sample_is_unity() {
        for amp in [0.01_f32, 0.2, 0.9] {
            let mut lc = LevelControl::agc(TARGET_DB, SR);
            let input = tone(amp, 220.0, 0.05);
            let out = run(&mut lc, &input);
            assert_eq!(
                out[0], input[0],
                "the first sample passes at unity (amp {amp}): {} vs {}",
                out[0], input[0]
            );
        }
    }

    /// Cold start (b): a quiet-but-present opening (below target, above the noise
    /// floor) reaches approximately the target within tens of milliseconds, not
    /// over the full ~400 ms window. Priming plus the warm-up slope converge fast.
    #[test]
    fn cold_start_converges_to_target_within_tens_of_ms() {
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        // Quiet input ~20 dB below the -18 dBFS target, but above the -50 dBFS
        // noise floor, so it is genuine signal to be boosted.
        let input = tone(amp_for_dbfs(-38.0), 220.0, 0.2);
        let target = db_to_linear(TARGET_DB as f32);

        // 20 ms blocks: inspect the output level block by block.
        let block = (0.02 * SR as f32) as usize;
        let mut levels = Vec::new();
        for chunk in input.chunks(block) {
            levels.push(rms(&run(&mut lc, chunk)));
        }

        // By ~100 ms (the fifth 20 ms block) the output is within ~3 dB of target.
        let early = levels[4];
        assert!(
            early > target * 0.7 && early < target * 1.4,
            "output {early} should be near target {target} within ~100 ms"
        );
        // The settled tail sits close to target.
        let settled = *levels.last().unwrap();
        assert!(
            settled > target * 0.8 && settled < target * 1.25,
            "settled output {settled} should track target {target}"
        );
    }

    /// Cold start (b'): the warm-up is spent on the first present signal, not
    /// burned by leading silence. An opening of 150 ms of silence (longer than
    /// the 100 ms warm-up window) followed by a quiet but present signal still
    /// reaches approximately the target within tens of ms of the signal onset,
    /// not over the multi-second steady-state ramp. This is the case the warm-up
    /// exists for: a stream opens, a pause, then speech. With the warm-up counted
    /// down only while signal is present, the opening silence does not consume it.
    #[test]
    fn cold_start_after_silence_converges_fast() {
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let target = db_to_linear(TARGET_DB as f32);
        let block = (0.02 * SR as f32) as usize;

        // 150 ms of opening silence: longer than the 100 ms warm-up window, so an
        // unconditional warm-up countdown would fully burn it before any signal.
        let _ = run(&mut lc, &vec![0.0_f32; (0.15 * SR as f32) as usize]);

        // Then a quiet -38 dBFS signal (above the -50 dBFS noise floor), needing
        // ~20 dB of gain to reach the -18 dBFS target.
        let signal = tone(amp_for_dbfs(-38.0), 220.0, 1.0);
        let levels: Vec<f32> = signal
            .chunks(block)
            .map(|c| rms(&run(&mut lc, c)))
            .collect();

        // By ~60 ms after the signal onset (the third 20 ms block) the output is
        // already past 70% of target: the warm-up was preserved through the
        // silence and applied to the first real signal. The slow ~6 dB/s steady
        // ramp would be ~14 dB short here (it needs seconds), so this assertion
        // fails if the warm-up is burned by the opening silence.
        assert!(
            levels[2] > target * 0.7,
            "first signal after opening silence converges fast (got {} at ~60 ms, target {target}); \
             the warm-up must be spent on signal, not on leading silence",
            levels[2]
        );

        // It then settles at target (the priming overshoot decays back down).
        let tail: f32 = levels[levels.len() - 5..].iter().sum::<f32>() / 5.0;
        assert!(
            tail > target * 0.8 && tail < target * 1.25,
            "the level settles at target after the silent opening (tail {tail}, target {target})"
        );
    }

    /// Cold start (c): a silent (and a sub-noise-floor noise-only) opening is not
    /// boosted. The presence gate holds the upward rise, so the gain stays at
    /// unity and the quiet opening is delivered unamplified, with no first-speech
    /// noise blast.
    #[test]
    fn cold_start_silent_opening_is_not_boosted() {
        // Pure silence.
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let silence = vec![0.0_f32; (0.3 * SR as f32) as usize];
        let out = run(&mut lc, &silence);
        assert!(
            out.iter().all(|&s| s == 0.0),
            "silence stays silent: the gate holds the rise, no boost"
        );

        // Sub-noise-floor noise (~-60 dBFS, below the -50 dBFS gate) is left
        // essentially at unity rather than amplified toward target.
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let noise = tone(amp_for_dbfs(-60.0), 300.0, 0.3);
        let out = run(&mut lc, &noise);
        let in_rms = rms(&noise);
        let out_rms = rms(&out);
        assert!(
            out_rms < in_rms * 2.0,
            "sub-noise-floor input is not boosted toward target: in {in_rms}, out {out_rms}"
        );
    }

    /// Cold start (d): a hot opening (near full scale, above target) does not
    /// overshoot into clipping. From unity start the gain drives down toward
    /// target (a hot opening is attenuated, not amplified), so the opening itself
    /// stays within full scale. This is the opening case only: mid-stream, a gain
    /// wound up on a quiet passage can let a later loud transient exceed full
    /// scale, which the limiter stage catches.
    #[test]
    fn cold_start_hot_opening_does_not_overshoot() {
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let input = tone(0.9, 220.0, 0.3);
        let out = run(&mut lc, &input);
        assert!(
            out.iter().all(|&s| s.abs() <= 1.0),
            "no output sample clips full scale on a hot opening"
        );
        assert!(
            out.iter().all(|&s| s.abs() <= 0.9 + 1e-6),
            "a hot input is never boosted above its own peak"
        );
        // It settles toward the target by attenuating.
        let block = (0.02 * SR as f32) as usize;
        let last: Vec<f32> = input.chunks(block).map(|c| rms(&run(&mut lc, c))).collect();
        let _ = last; // exercised; the no-clip assertions above are the guarantee
        let settled = rms(&out[out.len() - block..]);
        assert!(
            settled < rms(&input[..block]),
            "the hot input is attenuated toward target (settled {settled})"
        );
    }

    /// Cold start (e): downward correction is never gated. A loud onset following
    /// a gated (silent) opening is pulled down promptly, even though the upward
    /// rise was being held.
    #[test]
    fn loud_onset_pulls_gain_down_even_while_rise_gated() {
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        // A silent opening keeps the rise gated and the gain at unity.
        let _ = run(&mut lc, &vec![0.0_f32; (0.1 * SR as f32) as usize]);
        // Then a hot onset well above target.
        let loud = tone(0.8, 220.0, 0.2);
        let out = run(&mut lc, &loud);

        let block = (0.02 * SR as f32) as usize;
        let onset = rms(&out[..block]);
        let after = rms(&out[out.len() - block..]);
        assert!(
            after < onset,
            "the loud onset is pulled down (onset {onset} -> settled {after})"
        );
        let target = db_to_linear(TARGET_DB as f32);
        assert!(
            after < target * 2.0,
            "the level is brought down toward target ({after} vs target {target})"
        );
    }

    /// Steady state: a quiet, present signal converges up to the target and holds
    /// there without pumping. A loud signal is pulled down to the same target.
    /// Both settle near the target RMS.
    #[test]
    fn steady_state_converges_and_holds() {
        let target = db_to_linear(TARGET_DB as f32);
        let block = (0.02 * SR as f32) as usize;

        // Quiet (-30 dBFS) converges up to target and holds.
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let quiet = tone(amp_for_dbfs(-30.0), 220.0, 1.0);
        let levels: Vec<f32> = quiet.chunks(block).map(|c| rms(&run(&mut lc, c))).collect();
        let tail = &levels[levels.len() - 5..];
        for &l in tail {
            assert!(
                l > target * 0.8 && l < target * 1.25,
                "quiet input settles near target (got {l}, target {target})"
            );
        }
        // Held, not pumping: the tail blocks vary little.
        let tail_min = tail.iter().cloned().fold(f32::INFINITY, f32::min);
        let tail_max = tail.iter().cloned().fold(0.0_f32, f32::max);
        assert!(
            tail_max / tail_min < 1.2,
            "the settled level holds steady (min {tail_min}, max {tail_max})"
        );

        // Loud (-6 dBFS) is pulled down to the same target.
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let loud = tone(amp_for_dbfs(-6.0), 220.0, 1.0);
        let levels: Vec<f32> = loud.chunks(block).map(|c| rms(&run(&mut lc, c))).collect();
        let settled = *levels.last().unwrap();
        assert!(
            settled > target * 0.8 && settled < target * 1.25,
            "loud input is pulled down to target (got {settled}, target {target})"
        );
    }

    /// State continuity: the engine carries its level estimate and gain across
    /// blocks, so processing one signal in a single call equals processing it in
    /// irregular chunks, bit for bit. The control is purely per-sample, so block
    /// boundaries never change the result.
    #[test]
    fn state_carries_across_blocks() {
        let input = tone(amp_for_dbfs(-30.0), 180.0, 0.5);

        let one_shot = {
            let mut lc = LevelControl::agc(TARGET_DB, SR);
            run(&mut lc, &input)
        };

        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let mut chunked = Vec::with_capacity(input.len());
        for chunk in [
            &input[..1000],
            &input[1000..1001],
            &input[1001..4321],
            &input[4321..],
        ] {
            chunked.extend(run(&mut lc, chunk));
        }

        assert_eq!(
            chunked, one_shot,
            "per-sample control: chunked processing equals one-shot, bit for bit"
        );
    }

    /// An empty block is a no-op and leaves the engine state untouched, so the
    /// next real block behaves as if the empty call never happened.
    #[test]
    fn empty_block_is_a_noop() {
        let mut lc = LevelControl::agc(TARGET_DB, SR);
        let out = run(&mut lc, &[]);
        assert!(out.is_empty(), "an empty block yields no output");
        // The opening sample of the next block is still unity.
        let input = tone(0.3, 220.0, 0.05);
        let out = run(&mut lc, &input);
        assert_eq!(out[0], input[0], "the empty call did not advance state");
    }
}
