//! Cadence pins for the `decibri-resampler` dependency.
//!
//! An instance's output cadence, meaning the number of samples each `process`
//! call produces, the reported latency, and the length of the `flush` tail,
//! is a function of the rate pair and the sequence of input lengths alone.
//! Sample values play no part: identically constructed instances fed the same
//! sequence of input lengths with different content emit the same number of
//! samples from every individual call, report the same latency after every
//! call, and produce flush tails of equal length. Chunking moves samples
//! between calls and the flush, never the total: however the input is
//! chunked, the stream yields the same total sample count. These tests pin
//! both properties on every engine path the resampler dispatches to, at the
//! rate pairs the capture chain constructs and at pairs that reach the paths
//! those production pairs do not.

#![cfg(feature = "capture")]

use decibri_resampler::{PolyphaseResampler, Resampler};

/// Rate pairs covering every engine path. The constructor reduces `out/in` to
/// lowest terms and dispatches on the reduced numerator L: equal rates are the
/// identity passthrough, L of 1024 or less drives the exact rational engine,
/// and a larger L drives the general engine. 48000 -> 16000 and
/// 44100 -> 16000 are the pairs the capture chain builds for common devices;
/// 44101 and 16000 are coprime, so those two pairs reduce to L past the cap
/// and drive the general engine in both directions.
const PAIRS: &[(u32, u32)] = &[
    (16_000, 16_000), // identity
    (48_000, 16_000), // exact, L = 1
    (44_100, 16_000), // exact, L = 160
    (16_000, 48_000), // exact upsample, more than one output per input
    (44_101, 16_000), // general
    (16_000, 44_101), // general upsample
];

/// Samples fed per instance. Prime, so the stream ends mid-phase for every
/// pair above and the flush tail carries a partial-frame position.
const INPUT_LEN: usize = 997;

/// Deterministic broadband signal in [-1, 1) from a linear congruential
/// generator.
fn broadband(n: usize) -> Vec<f32> {
    let mut state = 0x1234_5678_u32;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((state >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

/// Full-scale alternating signal.
fn alternating(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
        .collect()
}

/// The three contents every pair is measured with. Silence first: an all-zero
/// buffer is the content most likely to take a value-dependent shortcut, so it
/// is paired against energy across the band and against a full-scale extreme.
fn contents(n: usize) -> [(&'static str, Vec<f32>); 3] {
    [
        ("silence", vec![0.0; n]),
        ("broadband", broadband(n)),
        ("full-scale alternating", alternating(n)),
    ]
}

/// Output cadence of one freshly constructed instance fed `input` in a single
/// call: samples produced by `process`, samples appended by `flush`, and the
/// reported latency.
fn single_call_cadence(in_rate: u32, out_rate: u32, input: &[f32]) -> (usize, usize, usize) {
    let mut r = PolyphaseResampler::new(in_rate, out_rate).unwrap();
    let mut out = Vec::new();
    r.process(input, &mut out).unwrap();
    let produced = out.len();
    r.flush(&mut out);
    (produced, out.len() - produced, r.latency_samples())
}

/// Per-call cadence of a freshly constructed instance fed `input` in chunks
/// whose sizes cycle through `sizes`, empty calls included: for each `process`
/// call, the number of samples that call appended and the latency reported
/// after it, in call order, then the length of the `flush` tail.
fn chunked_cadence(
    in_rate: u32,
    out_rate: u32,
    input: &[f32],
    sizes: &[usize],
) -> (Vec<(usize, usize)>, usize) {
    let mut r = PolyphaseResampler::new(in_rate, out_rate).unwrap();
    let mut out = Vec::new();
    let mut calls = Vec::new();
    let mut fed = 0usize;
    for size in sizes.iter().cycle() {
        if fed == input.len() {
            break;
        }
        let end = (fed + size).min(input.len());
        let before = out.len();
        r.process(&input[fed..end], &mut out).unwrap();
        calls.push((out.len() - before, r.latency_samples()));
        fed = end;
    }
    let before = out.len();
    r.flush(&mut out);
    (calls, out.len() - before)
}

#[test]
fn cadence_is_content_independent() {
    for &(in_rate, out_rate) in PAIRS {
        let contents = contents(INPUT_LEN);
        let (name0, first) = &contents[0];
        let baseline = single_call_cadence(in_rate, out_rate, first);
        for (name, input) in &contents[1..] {
            let cadence = single_call_cadence(in_rate, out_rate, input);
            assert_eq!(
                cadence, baseline,
                "{in_rate}->{out_rate}: (produced, tail, latency) for {name} \
                 differs from {name0}"
            );
        }

        // The rates-match case is a passthrough: the instance reports
        // identity, every input sample comes back out, the flush appends
        // nothing, and the latency is zero.
        let r = PolyphaseResampler::new(in_rate, out_rate).unwrap();
        if in_rate == out_rate {
            assert!(r.is_identity());
            assert_eq!(baseline, (INPUT_LEN, 0, 0));
        } else {
            assert!(!r.is_identity());
        }
    }
}

#[test]
fn cadence_is_chunking_and_content_independent() {
    // Mismatched chunk sizes, empty calls included, none dividing INPUT_LEN.
    let strategies: &[&[usize]] = &[&[1], &[7], &[64], &[3, 1, 64, 0, 7, 129, 13]];
    for &(in_rate, out_rate) in PAIRS {
        let contents = contents(INPUT_LEN);
        let (name0, first) = &contents[0];
        let (produced, tail, _) = single_call_cadence(in_rate, out_rate, first);
        let single_total = produced + tail;
        for sizes in strategies {
            let baseline = chunked_cadence(in_rate, out_rate, first, sizes);

            // Chunking moves samples between process calls and the flush,
            // never the total. The call sequences differ across chunkings,
            // so this comparison is deliberately a total, not per call.
            let (baseline_calls, baseline_tail) = &baseline;
            let total: usize =
                baseline_calls.iter().map(|&(n, _)| n).sum::<usize>() + baseline_tail;
            assert_eq!(
                total, single_total,
                "{in_rate}->{out_rate}: total output for {name0} in chunks \
                 {sizes:?} differs from {name0} fed in one call"
            );

            // Fed the same sequence of input lengths with different content,
            // every individual call emits the same number of samples, reports
            // the same latency, and the flush tails match. The whole per-call
            // sequence is compared, so a difference at one call is caught
            // even if a later call compensates.
            for (name, input) in &contents[1..] {
                let cadence = chunked_cadence(in_rate, out_rate, input, sizes);
                assert_eq!(
                    cadence, baseline,
                    "{in_rate}->{out_rate}: per-call cadence for {name} in \
                     chunks {sizes:?} differs from {name0}"
                );
            }
        }
    }
}
