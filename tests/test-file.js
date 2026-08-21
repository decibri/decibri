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
const { spawnSync } = require('child_process');
const { Readable } = require('stream');
const { pipeline } = require('stream/promises');

const DECIBRI_ENTRY = path.join(__dirname, '..', 'npm', 'decibri', 'src', 'decibri.js');
const {
  Microphone,
  File,
  AudioWriter,
  DecibriError,
} = require(DECIBRI_ENTRY);

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

  console.log('File: rate getters');

  // sampleRate is the rate every delivered chunk carries; inputRate is the
  // source's own rate. They match when nothing was resampled.
  const rateDefault = new File(wavPath);
  assert(rateDefault.sampleRate === 16000, `sampleRate defaults to 16000 (got ${rateDefault.sampleRate})`);
  assert(rateDefault.inputRate === 16000, `inputRate reads the WAV header (got ${rateDefault.inputRate})`);
  rateDefault.close();

  // A WAV read at a different engine rate: the two getters disagree, which is
  // the only way a caller learns the source was resampled.
  const wav48Path = path.join(tmp, 'clip-48k.wav');
  writeWav(wav48Path, sineSamples(48000, 0.25), 48000);
  const rateResampled = await File.open(wav48Path, { sampleRate: 16000 });
  assert(rateResampled.sampleRate === 16000, `sampleRate reports the engine rate (got ${rateResampled.sampleRate})`);
  assert(rateResampled.inputRate === 48000, `inputRate reports the source rate (got ${rateResampled.inputRate})`);

  // Both survive a full pass and a close, the same as vadScore: the File keeps
  // answering what it was configured with after the source is gone.
  await readAll(rateResampled);
  assert(
    rateResampled.sampleRate === 16000 && rateResampled.inputRate === 48000,
    'the rate getters still read after the source is consumed'
  );
  rateResampled.close();
  assert(
    rateResampled.sampleRate === 16000 && rateResampled.inputRate === 48000,
    'the rate getters still read after close()'
  );

  // The buffer path reports the explicit inputRate it was given.
  const rateBuffered = File.buffer(sineSamples(44100, 0.1), { inputRate: 44100, sampleRate: 22050 });
  assert(rateBuffered.sampleRate === 22050, `buffer sampleRate is the requested rate (got ${rateBuffered.sampleRate})`);
  assert(rateBuffered.inputRate === 44100, `buffer inputRate is the declared rate (got ${rateBuffered.inputRate})`);
  rateBuffered.close();

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

  // inputChannels is the channel counterpart of inputRate: shape-checked on
  // the buffer path, and refused on the open path, where the header answers.
  assertThrows(
    () => File.buffer(new Float32Array(16), { inputRate: 16000, inputChannels: 1.5 }),
    TypeError,
    'inputChannels must be an integer'
  );
  assertThrows(
    () => File.buffer(new Float32Array(16), { inputRate: 16000, inputChannels: 0 }),
    RangeError,
    'inputChannels must be between 1 and 65535'
  );
  assertThrows(
    () => new File(wavPath, { inputChannels: 2 }),
    TypeError,
    'inputChannels applies only to File.buffer'
  );
  await assertRejects(
    () => File.open(wavPath, { inputChannels: 2 }),
    TypeError,
    'inputChannels applies only to File.buffer'
  );
  // A sample count that is not a whole number of frames at the declared
  // interleave is the core's refusal, surfaced as the same RangeError class
  // the live block-size refusal carries.
  assertThrows(
    () => File.buffer(new Float32Array(7), { inputRate: 16000, inputChannels: 2, channels: 2 }),
    RangeError,
    'the requested block size of 7 samples is not a whole number of 2-channel frames'
  );

  console.log('File: option validation mirrors Microphone');

  assertThrows(() => new File(wavPath, { sampleRate: 999 }), RangeError, 'sample rate must be between');
  // The channel options carry the Microphone's own checks and messages.
  assertThrows(() => new File(wavPath, { channels: 0 }), RangeError, 'channels must be at least 1');
  assertThrows(() => new File(wavPath, { channelMap: 'left' }), TypeError, 'Invalid channelMap value');
  assertThrows(() => new File(wavPath, { channels: 1, channelMap: [0.5] }), TypeError, 'channelMap entries must be integers');
  assertThrows(() => new File(wavPath, { channels: 1, channelMap: [-1] }), RangeError, 'channelMap entries must be between 0 and 65535');
  assertThrows(() => new File(wavPath, { channels: 2, channelMap: [0] }), RangeError, 'channelMap must have exactly one entry per channel');
  assertThrows(() => new File(wavPath, { vad: true }), TypeError, 'vad: true is no longer supported');
  assertThrows(() => new File(wavPath, { vad: 'bogus' }), TypeError, 'Invalid vad value');
  // The flat vadThreshold/vadHoldoff forms raise the same migration error a
  // Microphone raises, instead of being silently ignored on File.
  assertThrows(() => new File(wavPath, { vadThreshold: 0.7 }), TypeError, 'vadThreshold and vadHoldoff are no longer supported');
  assertThrows(() => new File(wavPath, { vadHoldoff: 100 }), TypeError, 'vadThreshold and vadHoldoff are no longer supported');
  assertThrows(() => new File(wavPath, { highpass: 60 }), RangeError, 'highpass must be one of: 80, 100');
  assertThrows(() => new File(42), TypeError, 'path must be a string');

  // Constructor errors from the native layer: a missing file and junk bytes.
  assertThrows(() => new File(path.join(tmp, 'missing.wav')), DecibriError, 'Failed to read audio file');
  const junkPath = path.join(tmp, 'junk.wav');
  fs.writeFileSync(junkPath, 'this is not a wav file');
  assertThrows(() => new File(junkPath), DecibriError, 'unsupported audio format');

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
    // The whole analysis, pinned to exact counts. A change to either number
    // means the analysis itself changed.
    assert(
      report.scores.length === 208 && report.segments.length === 2,
      `the golden recording analyzes to 208 scores and 2 segments ` +
        `(got ${report.scores.length} and ${report.segments.length})`
    );
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

  console.log('File: analysis of an engaged stream is refused at the call');

  // Engaging the stream at all refuses a later analysis, rather than
  // reporting on the part not yet read. Each of these engages by a different
  // route, and a bare resume()/pause() pair with no data listener is enough:
  // the read it schedules runs on a later tick.
  //
  // These Files carry no vad, so the error also pins the check order: the
  // engaged state is reported before the missing detector configuration.
  // Keeping them model-free is what lets these cases run without a staged
  // ONNX Runtime.
  const engageRoutes = {
    'resume(); pause()': (f) => { f.resume(); f.pause(); },
    "on('data')": (f) => { f.on('data', () => {}); f.pause(); },
    "on('readable')": (f) => { f.on('readable', () => {}); },
    "once('readable')": (f) => { f.once('readable', () => {}); },
    'read()': (f) => { f.read(); },
  };
  for (const [label, engage] of Object.entries(engageRoutes)) {
    const engaged = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000 });
    engage(engaged);
    await assertRejects(() => engaged.analyze(), DecibriError, 'File iteration has begun');
    engaged.close();
  }

  // The refusal carries the dedicated FILE_ENGAGED code, and analyse()
  // refuses on the same terms.
  const engagedFile = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000 });
  engagedFile.resume();
  engagedFile.pause();
  let engagedErr = null;
  try {
    await engagedFile.analyze();
  } catch (e) {
    engagedErr = e;
  }
  assert(
    engagedErr instanceof DecibriError && engagedErr.code === 'FILE_ENGAGED',
    `an engaged File carries the FILE_ENGAGED code (got ${engagedErr && engagedErr.code})`
  );
  await assertRejects(() => engagedFile.analyse(), DecibriError, 'File iteration has begun');
  engagedFile.close();

  // An engaged energy-mode File reports the engaged state, not the mode, so
  // the check order matches the core and the Python package.
  const engagedEnergy = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000, vad: 'energy' });
  engagedEnergy.resume();
  engagedEnergy.pause();
  await assertRejects(() => engagedEnergy.analyze(), DecibriError, 'File iteration has begun');
  engagedEnergy.close();

  // A fully read File refuses analysis on the same terms as a partly read one.
  const drained = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000 });
  await readAll(drained);
  await assertRejects(() => drained.analyze(), DecibriError, 'File iteration has begun');

  // The refusal has to keep the process alive, not just reject: an 'error'
  // event from the stream's own read machinery, with no listener attached,
  // terminates the process. Run it in a child so the exit status is the
  // assertion.
  const survivorPath = path.join(tmp, 'survivor.js');
  fs.writeFileSync(
    survivorPath,
    [
      "'use strict';",
      `const { File } = require(${JSON.stringify(DECIBRI_ENTRY)});`,
      'const f = File.buffer(new Float32Array(3200), { inputRate: 16000 });',
      'f.resume();',
      'f.pause();',
      "f.analyze().then(() => process.exit(3)).catch(() => {});",
      '',
    ].join('\n')
  );
  const survivor = spawnSync(process.execPath, [survivorPath], { encoding: 'utf8' });
  assert(
    survivor.status === 0,
    `analysing an engaged File leaves the process alive ` +
      `(exit ${survivor.status}, stderr: ${(survivor.stderr || '').split('\n')[0]})`
  );

  console.log('File: a refused analysis leaves the File usable');

  // Every rejection reachable before the pass begins leaves the source in
  // place, so fixing the call and streaming the recording still works.
  const refusedNoVad = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000 });
  await assertRejects(() => refusedNoVad.analyze(), RangeError, 'analysis requires VAD');
  assert(
    (await readAll(refusedNoVad)).length === 6400,
    'a File rejected for the missing VAD still delivers its audio'
  );

  const refusedEnergy = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000, vad: 'energy' });
  await assertRejects(() => refusedEnergy.analyze(), RangeError, "analyze() requires vad: 'silero'");
  assert(
    (await readAll(refusedEnergy)).length === 6400,
    'a File rejected for the energy mode still delivers its audio'
  );

  const refusedEngaged = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000 });
  refusedEngaged.resume();
  refusedEngaged.pause();
  await assertRejects(() => refusedEngaged.analyze(), DecibriError, 'File iteration has begun');
  assert(
    (await readAll(refusedEngaged)).length === 6400,
    'a File refused for being engaged still delivers its audio'
  );

  console.log('File: consumed source is a loud lifecycle failure');

  // A successful analysis takes the source, so a later iteration fails loud
  // instead of yielding an empty stream that reads like a silent recording,
  // and it carries the dedicated FILE_CONSUMED code. The detector may fail to
  // load without a staged ONNX Runtime; the source is taken either way, which
  // is the state under test here.
  const coded = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000, vad: 'silero' });
  await coded.analyze().catch(() => {});
  let consumedErr = null;
  try {
    await readAll(coded);
  } catch (e) {
    consumedErr = e;
  }
  assert(
    consumedErr instanceof DecibriError && consumedErr.code === 'FILE_CONSUMED',
    `a consumed File carries the FILE_CONSUMED code (got ${consumedErr && consumedErr.code})`
  );

  // The detector's runtime failing to load is a typed failure on every
  // attempt in a process, never an abort: a second analysis after a failed
  // load reports what the first did. Run it in a child so an abort is an
  // exit status, not the end of this suite. With a loadable runtime both
  // analyses succeed; without one both reject with ORT_LOAD_FAILED.
  const retryPath = path.join(tmp, 'retry.js');
  fs.writeFileSync(
    retryPath,
    [
      "'use strict';",
      `const { File, DecibriError } = require(${JSON.stringify(DECIBRI_ENTRY)});`,
      'const outcomes = [];',
      '(async () => {',
      '  for (let i = 0; i < 2; i++) {',
      '    try {',
      "      await File.buffer(new Float32Array(3200), { inputRate: 16000, vad: 'silero' }).analyze();",
      "      outcomes.push('ok');",
      '    } catch (e) {',
      "      outcomes.push(e instanceof DecibriError ? e.code : 'untyped');",
      '    }',
      '  }',
      '  process.stdout.write(JSON.stringify(outcomes));',
      '})();',
      '',
    ].join('\n')
  );
  const retry = spawnSync(process.execPath, [retryPath], { encoding: 'utf8' });
  assert(
    retry.status === 0,
    `two silero analyses in one process leave the process alive ` +
      `(exit ${retry.status}, signal ${retry.signal}, stderr: ${(retry.stderr || '').split('\n')[0]})`
  );
  let retryOutcomes = [];
  try {
    retryOutcomes = JSON.parse(retry.stdout);
  } catch (_) {
    retryOutcomes = [];
  }
  assert(
    retryOutcomes.length === 2 && retryOutcomes.every((o) => o === 'ok' || o === 'ORT_LOAD_FAILED'),
    `each silero analysis either succeeds or rejects with ORT_LOAD_FAILED (got ${retry.stdout})`
  );
  assert(
    retryOutcomes.length === 2 && retryOutcomes[0] === retryOutcomes[1],
    `a second silero analysis reports what the first did (got ${retry.stdout})`
  );

  // An exhausted File yields nothing on a second pass, no error.
  const exhausted = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000 });
  assert((await readAll(exhausted)).length > 0, 'first pass over a File delivers audio');
  assert(
    (await readAll(exhausted)).length === 0,
    'a second pass over an exhausted File yields nothing, no error'
  );

  // A closed File reads as an ended stream, no error.
  const closedFile = File.buffer(sineSamples(16000, 0.2), { inputRate: 16000 });
  closedFile.close();
  assert(
    (await readAll(closedFile)).length === 0,
    'reading a closed File yields nothing, no error'
  );

  console.log('File: save writes the conditioned recording');

  // Samples on the exact 16-bit grid (k / 32768), so the quantisation into a
  // 16-bit file is the identity and round trips assert sample for sample.
  const gridSamples = (count) => {
    const out = new Float32Array(count);
    for (let i = 0; i < count; i++) out[i] = ((i % 1201) - 600) / 32768;
    return out;
  };
  const grid = gridSamples(4000);

  // Round trip in every container the extension rule names: the saved bytes
  // carry the right magic and read back exactly. Catches a writer wired to
  // the wrong container and a container that loses samples.
  for (const [ext, magic] of [['wav', 'RIFF'], ['aiff', 'FORM'], ['flac', 'fLaC']]) {
    const dest = path.join(tmp, `roundtrip.${ext}`);
    const report = await File.buffer(grid, { inputRate: 16000 }).save(dest);
    assert(
      report.clippedSamples === 0 && report.nonFiniteSamples === 0,
      `${ext}: a clean save reports zero counts`
    );
    const bytes = fs.readFileSync(dest);
    assert(bytes.subarray(0, 4).toString('latin1') === magic, `${ext}: the extension picks the container`);
    const back = await readAll(new File(dest));
    let exact = back.length === grid.length * 2;
    for (let i = 0; exact && i < grid.length; i++) {
      if (back.readInt16LE(i * 2) !== Math.trunc(grid[i] * 32768)) exact = false;
    }
    assert(exact, `${ext}: the round trip is exact on the 16-bit grid`);
  }

  // A conditioned save writes the conditioned audio, not the input. Catches
  // a save path that bypasses the conditioning chain.
  const dcOffset = new Float32Array(4000).fill(0.25);
  const dcDest = path.join(tmp, 'conditioned.wav');
  await File.buffer(dcOffset, { inputRate: 16000, dcRemoval: true }).save(dcDest);
  const dcBack = await readAll(new File(dcDest));
  let dcSum = 0;
  for (let i = 0; i < dcBack.length; i += 2) dcSum += dcBack.readInt16LE(i);
  const dcMean = dcSum / (dcBack.length / 2) / 32768;
  assert(
    Math.abs(dcMean) < 0.05,
    `a save carries the conditioned audio, not the input (mean ${dcMean.toFixed(3)})`
  );

  // The denoise round trip: a denoised save differs from its source and is
  // readable. Needs the bundled model and a loadable ONNX Runtime, gated
  // exactly as the silero cases above.
  let denoisedReport = null;
  const denoiseDest = path.join(tmp, 'denoised.flac');
  try {
    denoisedReport = await (await File.open(wavPath, { denoise: 'fastenhancer-t' })).save(denoiseDest);
  } catch (e) {
    if (e.message && (e.message.includes('failed to load ONNX Runtime') || e.message.includes('model not found'))) {
      console.log('  SKIP: ONNX Runtime not staged; denoise save case skipped');
    } else {
      throw e;
    }
  }
  if (denoisedReport !== null) {
    const denoisedBack = await readAll(new File(denoiseDest));
    assert(denoisedBack.length > 0, 'a denoised save reads back');
    assert(
      !denoisedBack.subarray(0, Math.min(denoisedBack.length, samples.length * 2))
        .equals(fs.readFileSync(wavPath).subarray(44)),
      'a denoised save differs from its source'
    );
  }

  // Saving is refused once the stream is engaged, with the FILE_ENGAGED
  // code, and the engaged state is reported before the arguments are
  // interpreted. Catches a save of the unread remainder.
  const engagedSave = File.buffer(gridSamples(3200), { inputRate: 16000 });
  engagedSave.resume();
  engagedSave.pause();
  let engagedSaveErr = null;
  try {
    await engagedSave.save(path.join(tmp, 'engaged.wav'));
  } catch (e) {
    engagedSaveErr = e;
  }
  assert(
    engagedSaveErr instanceof DecibriError && engagedSaveErr.code === 'FILE_ENGAGED',
    `saving an engaged File carries the FILE_ENGAGED code (got ${engagedSaveErr && engagedSaveErr.code})`
  );
  await assertRejects(
    () => engagedSave.save(path.join(tmp, 'engaged.flac'), { compression: 99 }),
    DecibriError,
    'File iteration has begun'
  );
  engagedSave.close();

  // A save consumes the source exactly as analyze does: a second save, a
  // later read, and a later analysis all fail loud with FILE_CONSUMED.
  const oneSave = File.buffer(gridSamples(3200), { inputRate: 16000 });
  await oneSave.save(path.join(tmp, 'first.wav'));
  await assertRejects(
    () => oneSave.save(path.join(tmp, 'second.wav')),
    DecibriError,
    'File already consumed'
  );
  let savedThenRead = null;
  try {
    await readAll(oneSave);
  } catch (e) {
    savedThenRead = e;
  }
  assert(
    savedThenRead instanceof DecibriError && savedThenRead.code === 'FILE_CONSUMED',
    `reading a saved File carries the FILE_CONSUMED code (got ${savedThenRead && savedThenRead.code})`
  );

  // Every failure detected before the pass begins leaves the File usable:
  // fix the argument and the same File still saves. Catches a pre-check
  // that takes the source.
  const refusedSave = File.buffer(grid, { inputRate: 16000 });
  await assertRejects(
    () => refusedSave.save(path.join(tmp, 'refused.mp3')),
    DecibriError,
    'unsupported audio format'
  );
  await assertRejects(
    () => refusedSave.save(path.join(tmp, 'refused.flac'), { compression: 9 }),
    RangeError,
    'flac compression level must be between 0 and 8'
  );
  const retriedReport = await refusedSave.save(path.join(tmp, 'retried.wav'));
  assert(
    retriedReport.clippedSamples === 0,
    'a File refused for its arguments still saves once they are fixed'
  );

  // The container comes from the name, an explicit format beats it, and an
  // unrecognised extension is an error rather than a silent default.
  const overrideDest = path.join(tmp, 'override.flac');
  await File.buffer(grid, { inputRate: 16000 }).save(overrideDest, { format: 'wav' });
  assert(
    fs.readFileSync(overrideDest).subarray(0, 4).toString('latin1') === 'RIFF',
    'an explicit format override beats the extension'
  );
  let extErr = null;
  try {
    await File.buffer(grid, { inputRate: 16000 }).save(path.join(tmp, 'noformat.mp3'));
  } catch (e) {
    extErr = e;
  }
  assert(
    extErr instanceof DecibriError &&
      extErr.code === 'AUDIO_FORMAT_UNSUPPORTED' &&
      extErr.message.includes("'.mp3'"),
    `an unrecognised extension is refused by name (got ${extErr && extErr.code}: ${extErr && extErr.message})`
  );
  await assertRejects(
    () => File.buffer(grid, { inputRate: 16000 }).save(path.join(tmp, 'x.wav'), { format: 'mp3' }),
    TypeError,
    'Invalid format value'
  );

  // FLAC compression levels: both range boundaries save and read back the
  // identical audio; past the boundary is a range error; a non-number is a
  // type error. Catches the writer's own rejection leaking through as a
  // malformed-file error.
  for (const level of [0, 8]) {
    const levelDest = path.join(tmp, `level${level}.flac`);
    await File.buffer(grid, { inputRate: 16000 }).save(levelDest, { compression: level });
    const levelBack = await readAll(new File(levelDest));
    let levelExact = levelBack.length === grid.length * 2;
    for (let i = 0; levelExact && i < grid.length; i++) {
      if (levelBack.readInt16LE(i * 2) !== Math.trunc(grid[i] * 32768)) levelExact = false;
    }
    assert(levelExact, `flac level ${level} saves and reads back losslessly`);
  }
  await assertRejects(
    () => File.buffer(grid, { inputRate: 16000 }).save(path.join(tmp, 'bad.flac'), { compression: -1 }),
    RangeError,
    'flac compression level must be between 0 and 8'
  );
  await assertRejects(
    () => File.buffer(grid, { inputRate: 16000 }).save(path.join(tmp, 'bad.flac'), { compression: 'high' }),
    TypeError,
    'compression must be a number'
  );

  // Clipping: a signal above full scale saves, the overshoot clamps to full
  // scale, and the count is exactly the clamped samples. Catches a silent
  // clamp and a count that includes in-range samples.
  const hot = gridSamples(1600);
  hot[10] = 1.85;
  hot[20] = -1.85;
  hot[30] = 2.5;
  const hotDest = path.join(tmp, 'clipped.wav');
  const hotReport = await File.buffer(hot, { inputRate: 16000 }).save(hotDest);
  assert(
    hotReport.clippedSamples === 3 && hotReport.nonFiniteSamples === 0,
    `an overscale save counts exactly the clamped samples (got ${hotReport.clippedSamples})`
  );
  const hotBack = await readAll(new File(hotDest));
  assert(
    hotBack.readInt16LE(10 * 2) === 32767 &&
      hotBack.readInt16LE(20 * 2) === -32768 &&
      hotBack.readInt16LE(30 * 2) === 32767,
    'the overshoot lands at full scale, never wrapped'
  );

  // Non-finite samples never reach the file: NaN becomes silence, an
  // infinity becomes full scale, each counted, and the clip count stays
  // zero because a repair is not a clamp. Catches the guard being dropped.
  const glitched = gridSamples(1600);
  glitched[10] = Number.NaN;
  glitched[20] = Number.POSITIVE_INFINITY;
  glitched[30] = Number.NEGATIVE_INFINITY;
  const glitchedDest = path.join(tmp, 'nonfinite.wav');
  const glitchedReport = await File.buffer(glitched, { inputRate: 16000 }).save(glitchedDest);
  assert(
    glitchedReport.nonFiniteSamples === 3 && glitchedReport.clippedSamples === 0,
    `a non-finite save counts the repairs and no clips (got ${glitchedReport.nonFiniteSamples} and ${glitchedReport.clippedSamples})`
  );
  const glitchedBack = await readAll(new File(glitchedDest));
  assert(
    glitchedBack.readInt16LE(10 * 2) === 0 &&
      glitchedBack.readInt16LE(20 * 2) === 32767 &&
      glitchedBack.readInt16LE(30 * 2) === -32768,
    'NaN saved as silence and the infinities as full scale'
  );

  // An unwritable destination reports the write identity, not a decode one.
  let writeErr = null;
  try {
    await File.buffer(grid, { inputRate: 16000 }).save(
      path.join(tmp, 'no-such-directory', 'out.wav')
    );
  } catch (e) {
    writeErr = e;
  }
  assert(
    writeErr instanceof DecibriError && writeErr.code === 'FILE_WRITE_FAILED',
    `an unwritable destination carries FILE_WRITE_FAILED (got ${writeErr && writeErr.code})`
  );
  assert(
    writeErr && writeErr.message.startsWith('Failed to write audio file'),
    'the write failure names the operation'
  );

  console.log('AudioWriter: the Writable produces the same bytes as save()');

  // A File piped into an AudioWriter writes the identical file save()
  // writes, in WAV and in FLAC. Catches a second encode path drifting from
  // the first, and an int16 dequantisation that is not the inverse of the
  // delivery quantisation.
  const saveDest = path.join(tmp, 'via-save.wav');
  await File.buffer(grid, { inputRate: 16000 }).save(saveDest);
  const pipeDest = path.join(tmp, 'via-pipe.wav');
  const wavWriter = new AudioWriter(pipeDest, { sampleRate: 16000 });
  await pipeline(File.buffer(grid, { inputRate: 16000 }), wavWriter);
  assert(
    fs.readFileSync(saveDest).equals(fs.readFileSync(pipeDest)),
    'the pipe and save() produce byte-identical WAV files'
  );
  assert(
    wavWriter.report !== null &&
      wavWriter.report.clippedSamples === 0 &&
      wavWriter.report.nonFiniteSamples === 0,
    'the writer carries the SaveReport after finish'
  );

  const saveFlac = path.join(tmp, 'via-save.flac');
  await File.buffer(grid, { inputRate: 16000 }).save(saveFlac, { compression: 5 });
  const pipeFlac = path.join(tmp, 'via-pipe.flac');
  await pipeline(
    File.buffer(grid, { inputRate: 16000 }),
    new AudioWriter(pipeFlac, { sampleRate: 16000, compression: 5 })
  );
  assert(
    fs.readFileSync(saveFlac).equals(fs.readFileSync(pipeFlac)),
    'the pipe and save() produce byte-identical FLAC files'
  );

  // The float32 pipe carries the samples verbatim and lands on the same
  // file again.
  const pipeF32 = path.join(tmp, 'via-pipe-f32.wav');
  await pipeline(
    File.buffer(grid, { inputRate: 16000, dtype: 'float32' }),
    new AudioWriter(pipeF32, { sampleRate: 16000, dtype: 'float32' })
  );
  assert(
    fs.readFileSync(saveDest).equals(fs.readFileSync(pipeF32)),
    'the float32 pipe produces the same bytes as save()'
  );

  // A non-decibri source: plain Buffers of PCM pipe into a valid file, the
  // case the writer exists for beyond File.
  const ttsDest = path.join(tmp, 'tts.wav');
  const pcm = Buffer.alloc(3200);
  for (let i = 0; i < 1600; i++) pcm.writeInt16LE((i % 1201) - 600, i * 2);
  await pipeline(
    Readable.from([pcm.subarray(0, 1000), pcm.subarray(1000)]),
    new AudioWriter(ttsDest, { sampleRate: 24000 })
  );
  const ttsBack = await readAll(new File(ttsDest, { sampleRate: 24000 }));
  assert(
    ttsBack.equals(pcm),
    'a plain PCM stream writes a file that reads back sample for sample'
  );

  // The writer's own validation: the rate is required and ranged, the
  // channel count is floor-bounded only (each container's own ceiling
  // answers at the write), the dtype is one of the two encodings, and a
  // byte stream that does not divide into whole samples is refused rather
  // than truncated.
  assertThrows(() => new AudioWriter(42, { sampleRate: 16000 }), TypeError, 'path must be a string');
  assertThrows(() => new AudioWriter(path.join(tmp, 'w.wav'), {}), TypeError, 'sampleRate is required');
  assertThrows(
    () => new AudioWriter(path.join(tmp, 'w.wav'), { sampleRate: 999 }),
    RangeError,
    'sample rate must be between 1000 and 384000'
  );
  assertThrows(
    () => new AudioWriter(path.join(tmp, 'w.wav'), { sampleRate: 16000, channels: 0 }),
    RangeError,
    'channels must be at least 1'
  );
  assertThrows(
    () => new AudioWriter(path.join(tmp, 'w.wav'), { sampleRate: 16000, dtype: 'int8' }),
    TypeError,
    "dtype must be 'int16' or 'float32'"
  );
  assertThrows(
    () => new AudioWriter(path.join(tmp, 'w.wav'), { sampleRate: 16000, compression: 9 }),
    RangeError,
    'flac compression level must be between 0 and 8'
  );
  await assertRejects(
    () =>
      pipeline(
        Readable.from([Buffer.alloc(3)]),
        new AudioWriter(path.join(tmp, 'odd.wav'), { sampleRate: 16000 })
      ),
    RangeError,
    'do not divide into whole int16 samples'
  );

  console.log('File: channels and the map reach the file surface');

  // A stereo source with distinct channels, on the 16-bit grid so every
  // comparison is exact: L a bounded ramp, R its negation.
  const stereoFrames = 4000;
  const stereo = new Float32Array(stereoFrames * 2);
  for (let f = 0; f < stereoFrames; f++) {
    const v = ((f % 1201) - 600) / 32768;
    stereo[f * 2] = v;
    stereo[f * 2 + 1] = -v;
  }

  // channels: 2 delivers both channels interleaved, unaveraged, unrotated.
  const bothOut = await readAll(
    File.buffer(stereo, { inputRate: 16000, inputChannels: 2, channels: 2, dtype: 'float32' })
  );
  assert(
    bothOut.length === stereo.length * 4,
    `two channels deliver every interleaved sample (got ${bothOut.length / 4})`
  );
  let stereoIdentical = true;
  for (let i = 0; i < stereo.length; i++) {
    if (bothOut.readFloatLE(i * 4) !== stereo[i]) {
      stereoIdentical = false;
      break;
    }
  }
  assert(stereoIdentical, 'the delivered stream is the source, channel for channel');

  // A map selects and permutes: [1, 0] swaps the two channels.
  const swappedOut = await readAll(
    File.buffer(stereo, {
      inputRate: 16000,
      inputChannels: 2,
      channels: 2,
      channelMap: [1, 0],
      dtype: 'float32',
    })
  );
  let swapOk = true;
  for (let f = 0; f < stereoFrames; f++) {
    if (
      swappedOut.readFloatLE(f * 8) !== stereo[f * 2 + 1] ||
      swappedOut.readFloatLE(f * 8 + 4) !== stereo[f * 2]
    ) {
      swapOk = false;
      break;
    }
  }
  assert(swapOk, 'channelMap [1, 0] permutes the delivered channels');

  // The refusals carry the file surface's own identities: an over-ask, an
  // unmapped strict subset, and a map entry the source does not have.
  let overAsk = null;
  try {
    File.buffer(stereo, { inputRate: 16000, inputChannels: 2, channels: 4 });
  } catch (e) {
    overAsk = e;
  }
  assert(
    overAsk instanceof DecibriError &&
      overAsk.code === 'FILE_CHANNELS_UNSUPPORTED' &&
      overAsk.message === 'the file does not have 4 channels to deliver; it has 2',
    `an unmapped over-ask carries FILE_CHANNELS_UNSUPPORTED (got ${overAsk && overAsk.code}: ${overAsk && overAsk.message})`
  );
  const eight = new Float32Array(64); // 8 frames of 8 channels
  let subset = null;
  try {
    File.buffer(eight, { inputRate: 16000, inputChannels: 8, channels: 2 });
  } catch (e) {
    subset = e;
  }
  assert(
    subset instanceof DecibriError &&
      subset.code === 'FILE_CHANNEL_SELECTION_AMBIGUOUS' &&
      subset.message === "delivering 2 of the file's 8 channels requires a channel map",
    `an unmapped strict subset carries FILE_CHANNEL_SELECTION_AMBIGUOUS (got ${subset && subset.code}: ${subset && subset.message})`
  );
  let outOfRange = null;
  try {
    File.buffer(stereo, { inputRate: 16000, inputChannels: 2, channels: 2, channelMap: [0, 5] });
  } catch (e) {
    outOfRange = e;
  }
  assert(
    outOfRange instanceof DecibriError &&
      outOfRange.code === 'FILE_CHANNEL_MAP_OUT_OF_RANGE' &&
      outOfRange.message === 'the file channel map names channel 5; the file has 2 channels',
    `a map entry the source lacks carries FILE_CHANNEL_MAP_OUT_OF_RANGE (got ${outOfRange && outOfRange.code}: ${outOfRange && outOfRange.message})`
  );

  console.log('File: the detector source names one delivered channel');

  // A stereo source whose channels are distinguishable by level alone,
  // delivered channel 0 silent and delivered channel 1 loud, so a wrong
  // selection is visible rather than plausible: the energy score is the RMS
  // of exactly the samples the detector was fed.
  const srcFrames = 4000;
  const silentLoud = new Float32Array(srcFrames * 2);
  for (let f = 0; f < srcFrames; f++) {
    silentLoud[f * 2] = 0;
    silentLoud[f * 2 + 1] = 0.5;
  }
  const scoreOf = async (vad) => {
    const scored = File.buffer(silentLoud, {
      inputRate: 16000,
      inputChannels: 2,
      channels: 2,
      vad,
    });
    let top = 0;
    for await (const chunk of scored) {
      top = Math.max(top, scored.vadScore);
    }
    return top;
  };
  const silentScore = await scoreOf({ model: 'energy', source: 0 });
  assert(silentScore < 0.001, `naming the silent channel scores its silence (got ${silentScore})`);
  const loudScore = await scoreOf({ model: 'energy', source: 1 });
  assert(
    Math.abs(loudScore - 0.5) < 0.01,
    `naming the loud channel scores its own level (got ${loudScore})`
  );
  // No source set: the score is the frame average's, half the loud level
  // here, so the default is the established collapse beside the selections.
  const averagedScore = await scoreOf({ model: 'energy' });
  assert(
    Math.abs(averagedScore - 0.25) < 0.01,
    `no source: the score is the frame average's (got ${averagedScore})`
  );

  // The refusal is the wrapper's own RangeError, the same class and message
  // the Microphone raises for the same input, checked against the delivered
  // count alone.
  assertThrows(
    () =>
      File.buffer(silentLoud, {
        inputRate: 16000,
        inputChannels: 2,
        channels: 2,
        vad: { model: 'energy', source: 2 },
      }),
    RangeError,
    'the detector source names delivered channel 2; the delivered channel count is 2'
  );

  // The round trip: a stereo save re-read delivers the identical stream,
  // and the written header carries the channel count.
  const stereoDest = path.join(tmp, 'stereo.wav');
  await File.buffer(stereo, {
    inputRate: 16000,
    inputChannels: 2,
    channels: 2,
  }).save(stereoDest);
  const stereoHeader = fs.readFileSync(stereoDest);
  assert(
    stereoHeader.readUInt16LE(22) === 2,
    `the saved WAV header carries 2 channels (got ${stereoHeader.readUInt16LE(22)})`
  );
  const stereoBack = await readAll(new File(stereoDest, { channels: 2 }));
  const stereoSaved = await readAll(
    File.buffer(stereo, { inputRate: 16000, inputChannels: 2, channels: 2 })
  );
  assert(
    stereoBack.equals(stereoSaved),
    'the stereo round trip reproduces the delivered stream exactly'
  );

  // The multichannel writer: a stereo pipe writes the identical file the
  // stereo save writes.
  const stereoPipe = path.join(tmp, 'stereo-pipe.wav');
  await pipeline(
    File.buffer(stereo, { inputRate: 16000, inputChannels: 2, channels: 2 }),
    new AudioWriter(stereoPipe, { sampleRate: 16000, channels: 2 })
  );
  assert(
    fs.readFileSync(stereoDest).equals(fs.readFileSync(stereoPipe)),
    'the stereo pipe and save() produce byte-identical files'
  );

  // FLAC's ceiling is the container's own, surfaced with its own text.
  await assertRejects(
    () =>
      File.buffer(new Float32Array(9), {
        inputRate: 16000,
        inputChannels: 9,
        channels: 9,
      }).save(path.join(tmp, 'nine.flac')),
    DecibriError,
    '9-channel audio is not a supported layout'
  );

  // File-time VAD advances by frames, not by interleaved samples: a stereo
  // tail of 0.25 s of silence sits inside the 300 ms holdoff, so no silence
  // event fires; counted in raw samples it would read as 0.5 s and fire.
  const loudStereo = new Float32Array(16000 * 2 * 0.5 + 16000 * 2 * 0.25);
  for (let f = 0; f < 8000; f++) {
    const v = 0.5 * Math.sin((2 * Math.PI * 440 * f) / 16000);
    loudStereo[f * 2] = v;
    loudStereo[f * 2 + 1] = v;
  }
  const stereoVad = File.buffer(loudStereo, {
    inputRate: 16000,
    inputChannels: 2,
    channels: 2,
    vad: 'energy',
  });
  const stereoEvents = [];
  stereoVad.on('speech', () => stereoEvents.push('speech'));
  stereoVad.on('silence', () => stereoEvents.push('silence'));
  await readAll(stereoVad);
  assert(
    stereoEvents.length === 1 && stereoEvents[0] === 'speech',
    `file time advances by frames: a 0.25 s stereo tail stays inside the holdoff (got ${JSON.stringify(stereoEvents)})`
  );

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
