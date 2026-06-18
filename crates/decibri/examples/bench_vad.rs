//! VAD micro-benchmark: SileroVad inference latency.
//!
//! Measures wall-time per `SileroVad::process()` call on a deterministic
//! 16 kHz audio chunk.
//!
//! # Run
//!
//! ```bash
//! cargo run --release --example bench_vad \
//!     -p decibri \
//!     --no-default-features \
//!     --features capture,playback,vad,denoise,gain,ort-download-binaries
//! ```
//!
//! Output is two lines: `p50 = <ns>` and `p99 = <ns>`. For a fair comparison,
//! run the builds being compared on the same hardware back-to-back.

use std::path::Path;
use std::time::Instant;

use decibri::vad::{SileroVad, VadConfig};

const ITERATIONS: usize = 10_000;
const WINDOW_SAMPLES: usize = 512;
const SAMPLE_RATE: u32 = 16_000;

fn main() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let model = Path::new(manifest_dir)
        .join("..")
        .join("..")
        .join("models")
        .join("silero_vad.onnx");

    if !model.is_file() {
        panic!(
            "bench_vad: silero model not found at {}. Cannot benchmark.",
            model.display()
        );
    }

    // `VadConfig` is `#[non_exhaustive]`: default-construct then assign the
    // public fields rather than using a struct literal.
    let mut config = VadConfig::default();
    config.model_path = model;
    config.sample_rate = SAMPLE_RATE;
    config.threshold = 0.5;
    config.ort_library_path = None;

    let mut vad = SileroVad::new(config).expect("SileroVad construction should succeed");

    // Deterministic input: a 16 kHz sine wave at 440 Hz, normalized to [-1, 1].
    // Same waveform across runs so timing differences reflect inference cost
    // alone, not input-data variability.
    let chunk: Vec<f32> = (0..WINDOW_SAMPLES)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE as f32;
            (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
        })
        .collect();

    // Warm-up to stabilize CPU caches and ORT internal allocations. Not
    // included in the timing measurements.
    for _ in 0..100 {
        vad.process(&chunk).expect("warmup process should succeed");
    }

    let mut samples_ns = Vec::with_capacity(ITERATIONS);
    for _ in 0..ITERATIONS {
        let start = Instant::now();
        vad.process(&chunk).expect("bench process should succeed");
        samples_ns.push(start.elapsed().as_nanos() as u64);
    }
    samples_ns.sort_unstable();

    let p50 = samples_ns[samples_ns.len() / 2];
    let p99 = samples_ns[(samples_ns.len() * 99) / 100];

    println!("iterations = {ITERATIONS}");
    println!("window_samples = {WINDOW_SAMPLES}");
    println!("sample_rate = {SAMPLE_RATE}");
    println!("p50 = {p50} ns");
    println!("p99 = {p99} ns");
}
