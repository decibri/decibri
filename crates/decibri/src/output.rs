// Audio output (speaker playback) — stub for Phase 2.
//
// Will provide cpal-based speaker output with the same crossbeam channel
// pattern used by capture. The public API will mirror AudioCapture:
//
//   let output = AudioOutput::new(OutputConfig { ... })?;
//   output.write(&chunk)?;
//   output.stop();
