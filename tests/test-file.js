'use strict';

/**
 * File: the offline source (conditioning + whole-file VAD analysis).
 *
 * CI-safe: no microphone or speaker required. Covers the File surface (path
 * and buffer constructors, bare-vs-open equivalence, conditioning through the
 * shared chain, the end-of-file flush tail, Float32Array enforcement), the
 * per-chunk VAD events in FILE time, and the whole-file analyze()/analyse()
 * report. Silero-driven cases read the repo's golden speech fixture and are
 * skipped when it is absent.
 *
 * Also pins the live Microphone's CURRENT wall-clock speech/silence state
 * machine (a characterization of `_processVadValue`), so the mic's timing
 * behavior is protected by a test alongside the File's file-time machine.
 *
 * Runs standalone (`node tests/test-file.js`) or from test-ci.js, which
 * chains `fileTests` into its CI run with its own counters.
 */

const path = require('path');
const fs = require('fs');
const os = require('os');
const {
  Microphone,
  File,
  DecibriError,
} = require(path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js'));

const GOLDEN_WAV = path.join(
  __dirname, '..', 'crates', 'decibri', 'tests', 'assets', 'vad-golden-tts-speech-16k.wav'
);

// ─── Fixture helpers ─────────────────────────────────────────────────────────

/** A mono sine at the given rate, amplitude 0.5. */
function sineSamples(rate, seconds, amplitude = 0.5) {
  const count = Math.floor(rate * seconds);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i++) {
    out[i] = amplitude * Math.sin((2 * Math.PI * 440 * i) / rate);
  }
  return out;
}

/** Write mono 16-bit PCM WAV bytes for the given samples. */
function writeWav(filePath, samples, rate) {
  const payload = Buffer.alloc(samples.length * 2);
  for (let i = 0; i < samples.length; i++) {
    const v = Math.max(-32768, Math.min(32767, Math.round(samples[i] * 32768)));
    payload.writeInt16LE(v, i * 2);
  }
  const header = Buffer.alloc(44);
  header.write('RIFF', 0);
  header.writeUInt32LE(36 + payload.length, 4);
  header.write('WAVE', 8);
  header.write('fmt ', 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20); // PCM
  header.writeUInt16LE(1, 22); // mono
  header.writeUInt32LE(rate, 24);
  header.writeUInt32LE(rate * 2, 28);
  header.writeUInt16LE(2, 32);
  header.writeUInt16LE(16, 34);
  header.write('data', 36);
  header.writeUInt32LE(payload.length, 40);
  fs.writeFileSync(filePath, Buffer.concat([header, payload]));
}

/** Concatenate every conditioned chunk a File delivers. */
async function readAll(file) {
  const chunks = [];
  for await (const chunk of file) {
    chunks.push(chunk);
  }
  return Buffer.concat(chunks);
}

// ─── The test suite ─────────────────────────────────────────────────────────

/**
 * Run the File test cases against the provided harness
 * ({ assert, assertThrows, assertRejects } sharing the caller's counters).
 */
async function fileTests(h) {
  const { assert, assertThrows, assertRejects } = h;
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'decibri-file-'));
  const wavPath = path.join(tmp, 'clip.wav');
  const samples = sineSamples(16000, 0.5); // 8000 samples
  writeWav(wavPath, samples, 16000);

  console.log('File: constructors and conditioning');

  // Bare `new File(path)` and `await File.open(path)` produce identical output.
  const bareOut = await readAll(new File(wavPath));
  const openedOut = await readAll(await File.open(wavPath));
  assert(bareOut.equals(openedOut), 'bare and open constructors deliver identical output');
  assert(bareOut.length === 16000, `passthrough delivers all samples (got ${bareOut.length})`);

  // A mono 16 kHz WAV with no conditioning passes through byte-identically.
  const original = fs.readFileSync(wavPath).subarray(44);
  assert(bareOut.equals(original), 'no-conditioning path is byte-identical to the recording');

  // The buffer path resamples inputRate to sampleRate, flush tail included.
  const at48k = sineSamples(48000, 0.25); // 12000 samples
  const resampled = await readAll(File.buffer(at48k, { inputRate: 48000, sampleRate: 16000 }));
  const outSamples = resampled.length / 2;
  assert(
    outSamples >= 4000 && outSamples <= 4512,
    `48k to 16k resample delivers the full recording plus the flushed tail (got ${outSamples})`
  );

  // dtype float32 delivers 4 bytes per sample, mirroring Microphone.
  const f32Out = await readAll(new File(wavPath, { dtype: 'float32' }));
  assert(f32Out.length === samples.length * 4, 'float32 dtype delivers 4 bytes per sample');

  // Conditioning applies: dcRemoval pulls a constant offset toward zero.
  const offset = new Float32Array(8000);
  for (let i = 0; i < offset.length; i++) offset[i] = 0.25;
  const conditioned = await readAll(File.buffer(offset, { inputRate: 16000, dcRemoval: true }));
  let sum = 0;
  for (let i = 0; i < conditioned.length; i += 2) sum += conditioned.readInt16LE(i);
  const mean = sum / (conditioned.length / 2) / 32768;
  assert(Math.abs(mean) < 0.05, `dcRemoval conditions the delivered audio (mean ${mean.toFixed(3)})`);

  console.log('File: buffer input enforcement');

  // A raw Buffer of bytes is ambiguous and rejected; so is anything that is
  // not a Float32Array, and a missing inputRate.
  assertThrows(
    () => File.buffer(Buffer.from([0, 0, 0, 0]), { inputRate: 16000 }),
    TypeError,
    'not a Buffer of bytes'
  );
  assertThrows(
    () => File.buffer([0.0, 0.1], { inputRate: 16000 }),
    TypeError,
    'requires a Float32Array'
  );
  assertThrows(
    () => File.buffer(new Float32Array(16), {}),
    TypeError,
    'inputRate is required'
  );
  assertThrows(
    () => File.buffer(new Float32Array(16), { inputRate: 999 }),
    RangeError,
    'inputRate must be between 1000 and 384000'
  );

  console.log('File: option validation mirrors Microphone');

  assertThrows(() => new File(wavPath, { sampleRate: 999 }), RangeError, 'sample rate must be between');
  assertThrows(() => new File(wavPath, { vad: true }), TypeError, 'vad: true is no longer supported');
  assertThrows(() => new File(wavPath, { vad: 'bogus' }), TypeError, 'Invalid vad value');
  assertThrows(() => new File(wavPath, { highpass: 60 }), RangeError, 'highpass must be one of: 80, 100');
  assertThrows(() => new File(42), TypeError, 'path must be a string');

  // Constructor errors from the native layer: a missing file and junk bytes.
  assertThrows(() => new File(path.join(tmp, 'missing.wav')), DecibriError, 'Failed to read audio file');
  const junkPath = path.join(tmp, 'junk.wav');
  fs.writeFileSync(junkPath, 'this is not a wav file');
  assertThrows(() => new File(junkPath), DecibriError, 'invalid WAV file');

  console.log('File: per-chunk VAD events in file time');

  // Half a second of loud signal then 0.6 s of silence, energy mode: the
  // speech event fires on the loud region and, decisively, the silence event
  // fires BEFORE the stream ends even though the whole file processes in
  // milliseconds. A wall-clock 300 ms holdoff could not elapse in that time;
  // only a holdoff measured in FILE time can.
  const loud = sineSamples(16000, 0.5);
  const speech = new Float32Array(loud.length + 9600);
  speech.set(loud, 0);
  const vadFile = File.buffer(speech, { inputRate: 16000, vad: 'energy' });
  const events = [];
  vadFile.on('speech', () => events.push('speech'));
  vadFile.on('silence', () => events.push('silence'));
  let sawScore = 0;
  for await (const chunk of vadFile) {
    sawScore = Math.max(sawScore, vadFile.vadScore);
  }
  assert(
    events.length === 2 && events[0] === 'speech' && events[1] === 'silence',
    `speech then silence fire within the pass (got ${JSON.stringify(events)})`
  );
  assert(sawScore > 0.01, `vadScore reports the energy score during the pass (got ${sawScore})`);

  // With no vad option there are no scores and no speech events.
  const plain = File.buffer(speech, { inputRate: 16000 });
  let plainEvents = 0;
  plain.on('speech', () => plainEvents++);
  await readAll(plain);
  assert(plainEvents === 0 && plain.vadScore === 0, 'no vad option means no VAD surface');

  console.log('File: whole-file analysis');

  // analyze() requires VAD: never a silently constructed detector.
  await assertRejects(
    () => File.buffer(speech, { inputRate: 16000 }).analyze(),
    RangeError,
    'analysis requires VAD'
  );
  await assertRejects(
    () => File.buffer(speech, { inputRate: 16000, vad: 'energy' }).analyze(),
    RangeError,
    "analyze() requires vad: 'silero'"
  );

  // The Silero cases need the golden fixture AND a loadable ONNX Runtime
  // (the bundled platform dylib, or ORT_DYLIB_PATH). A dev checkout without
  // a staged runtime skips them, exactly as the Python suite gates its
  // ORT-loading tests on the bundled dylib.
  let goldenReport = null;
  if (fs.existsSync(GOLDEN_WAV)) {
    try {
      goldenReport = await (await File.open(GOLDEN_WAV, { vad: 'silero' })).analyze();
    } catch (e) {
      if (e.message && e.message.includes('failed to load ONNX Runtime')) {
        console.log('  SKIP: ONNX Runtime not staged; silero analysis cases skipped');
      } else {
        throw e;
      }
    }
  } else {
    console.log('  SKIP: golden speech fixture not available; silero analysis cases skipped');
  }
  if (goldenReport !== null) {
    const report = goldenReport;
    assert(report.scores.length > 0, 'analysis returns per-window scores');
    let tiled = true;
    let maxScore = 0;
    let thresholdConsistent = true;
    for (let i = 0; i < report.scores.length; i++) {
      const w = report.scores[i];
      if (Math.abs(w.start - (i * 512) / 16000) > 1e-9) tiled = false;
      if (Math.abs(w.end - w.start - 512 / 16000) > 1e-9) tiled = false;
      if (w.isSpeech !== w.vadScore >= 0.5) thresholdConsistent = false;
      maxScore = Math.max(maxScore, w.vadScore);
    }
    assert(tiled, 'windows tile the recording in 32 ms steps of file time');
    assert(thresholdConsistent, 'isSpeech is the raw threshold test per window');
    assert(maxScore >= 0.5, `real speech crosses the threshold (max ${maxScore.toFixed(2)})`);
    assert(report.segments.length > 0, 'speech windows merge into segments');
    const duration = report.scores[report.scores.length - 1].end;
    assert(
      report.segments.every((s) => s.start < s.end && s.end <= duration + 1e-9),
      'segments sit inside the recording'
    );

    // Both spellings return the same analysis.
    const analysed = await (await File.open(GOLDEN_WAV, { vad: 'silero' })).analyse();
    assert(
      JSON.stringify(analysed) === JSON.stringify(report),
      'analyse() equals analyze()'
    );

    // A non-detector target rate still analyzes: the detector feed is
    // resampled internally, with no setting and no error.
    const resampledReport = await (
      await File.open(GOLDEN_WAV, { vad: 'silero', sampleRate: 22050 })
    ).analyze();
    assert(
      resampledReport.segments.length > 0,
      'a non-16k target rate analyzes through the internal feed resample'
    );

    // A File is a single pass: a second analysis on the same object rejects.
    // A consumed source is a lifecycle error, so it surfaces as the decibri
    // error (like a closed stream), not the RangeError reserved for argument
    // validation.
    const oneShot = await File.open(GOLDEN_WAV, { vad: 'silero' });
    await oneShot.analyze();
    await assertRejects(() => oneShot.analyze(), DecibriError, 'File already consumed');
  }

  console.log('Microphone: wall-clock VAD state machine characterization');

  // Pin the live mic's CURRENT speech/silence semantics (wall-clock holdoff
  // via setTimeout) at the prototype level, with no device or native handle:
  // the File's file-time machine above is a deliberate offline divergence,
  // and this characterization keeps the mic's own behavior test-protected.
  const fake = {
    _vadScore: 0,
    _vadThreshold: 0.5,
    _vadHoldoff: 50,
    _isSpeaking: false,
    _silenceTimer: null,
    events: [],
    emit(name) {
      this.events.push(name);
    },
  };
  Microphone.prototype._processVadValue.call(fake, 0.9);
  assert(fake._isSpeaking === true && fake.events.join(',') === 'speech',
    'mic machine: above threshold enters speaking and emits speech');
  Microphone.prototype._processVadValue.call(fake, 0.1);
  assert(fake._isSpeaking === true && fake._silenceTimer !== null,
    'mic machine: below threshold arms the wall-clock silence timer');
  await new Promise((resolve) => setTimeout(resolve, 120));
  assert(fake._isSpeaking === false && fake.events.join(',') === 'speech,silence',
    'mic machine: the silence event fires after the wall-clock holdoff elapses');

  fs.rmSync(tmp, { recursive: true, force: true });
}

module.exports = { fileTests };

// Standalone entry point: run with local counters and a summary.
if (require.main === module) {
  let passed = 0;
  let failed = 0;
  const harness = {
    assert(condition, label) {
      if (condition) {
        passed++;
      } else {
        console.log(`  FAIL: ${label}`);
        failed++;
      }
    },
    assertThrows(fn, errorType, messagePart) {
      try {
        fn();
        console.log(`  FAIL: expected ${errorType.name} but no error thrown`);
        failed++;
      } catch (e) {
        if (!(e instanceof errorType)) {
          console.log(`  FAIL: expected ${errorType.name}, got ${e.constructor.name}: ${e.message}`);
          failed++;
        } else if (!e.message.includes(messagePart)) {
          console.log(`  FAIL: message mismatch`);
          console.log(`    expected to contain: ${messagePart}`);
          console.log(`    actual: ${e.message}`);
          failed++;
        } else {
          passed++;
        }
      }
    },
    async assertRejects(fn, errorType, messagePart) {
      try {
        await fn();
        console.log(`  FAIL: expected ${errorType.name} rejection but the promise resolved`);
        failed++;
      } catch (e) {
        if (!(e instanceof errorType)) {
          console.log(`  FAIL: expected ${errorType.name} rejection, got ${e.constructor.name}: ${e.message}`);
          failed++;
        } else if (!e.message.includes(messagePart)) {
          console.log(`  FAIL: rejection message mismatch`);
          console.log(`    expected to contain: ${messagePart}`);
          console.log(`    actual: ${e.message}`);
          failed++;
        } else {
          passed++;
        }
      }
    },
  };

  fileTests(harness)
    .then(() => {
      console.log('═══════════════════════════════════════');
      console.log(`  Passed:  ${passed}`);
      console.log(`  Failed:  ${failed}`);
      console.log('═══════════════════════════════════════');
      if (failed > 0) {
        process.exit(1);
      }
    })
    .catch((err) => {
      console.error('  FATAL: File tests threw unexpectedly:', err);
      process.exit(1);
    });
}
