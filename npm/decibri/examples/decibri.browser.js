// decibri browser bundle. GENERATED from src/browser/index.js. Do not edit by hand; see examples/README.md to regenerate.
"use strict";
var decibri = (function() {
	//#region \0rolldown/runtime.js
	var __commonJSMin = (cb, mod) => () => (mod || cb((mod = { exports: {} }).exports, mod), mod.exports);
	//#endregion
	//#region npm/decibri/src/browser/emitter.js
	var require_emitter = /* @__PURE__ */ __commonJSMin(((exports, module) => {
		/**
		* Minimal event emitter for browsers.
		*
		* Provides .on() / .off() / .once() / .emit() with the same API pattern
		* as Node.js EventEmitter. Used instead of browser-native EventTarget to
		* preserve API parity with the Node.js decibri (.on('data', cb) pattern).
		*
		* Ported from decibri-web emitter.ts. Logic identical, types removed.
		*/
		var Emitter = class {
			constructor() {
				this._listeners = /* @__PURE__ */ new Map();
			}
			on(event, fn) {
				let set = this._listeners.get(event);
				if (!set) {
					set = /* @__PURE__ */ new Set();
					this._listeners.set(event, set);
				}
				set.add(fn);
				return this;
			}
			off(event, fn) {
				const set = this._listeners.get(event);
				if (!set) return this;
				if (set.delete(fn)) return this;
				for (const listener of set) if (listener._original === fn) {
					set.delete(listener);
					return this;
				}
				return this;
			}
			once(event, fn) {
				const wrapper = (...args) => {
					this.off(event, wrapper);
					fn(...args);
				};
				wrapper._original = fn;
				return this.on(event, wrapper);
			}
			emit(event, ...args) {
				const set = this._listeners.get(event);
				if (!set || set.size === 0) return false;
				for (const fn of set) fn(...args);
				return true;
			}
			removeAllListeners(event) {
				if (event !== void 0) this._listeners.delete(event);
				else this._listeners.clear();
				return this;
			}
		};
		module.exports = { Emitter };
	}));
	//#endregion
	//#region npm/decibri/src/browser/worklet-inline.js
	var require_worklet_inline = /* @__PURE__ */ __commonJSMin(((exports, module) => {
		module.exports = { WORKLET_SOURCE: "var u=class extends AudioWorkletProcessor{constructor(r){super();let e=r.processorOptions;this.framesPerBuffer=e.framesPerBuffer,this.format=e.format,this.ratio=e.nativeSampleRate/e.targetSampleRate,this.needsResample=e.nativeSampleRate!==e.targetSampleRate,this.position=0,this.buffer=new Float32Array(this.framesPerBuffer),this.bufferIndex=0}process(r,e,s){let t=r[0]?.[0];if(!t||t.length===0)return!0;let f;this.needsResample?f=this.resample(t):f=t;let a=0;for(;a<f.length;){let i=this.framesPerBuffer-this.bufferIndex,o=f.length-a,n=Math.min(i,o);this.buffer.set(f.subarray(a,a+n),this.bufferIndex),this.bufferIndex+=n,a+=n,this.bufferIndex>=this.framesPerBuffer&&this.flush()}return!0}resample(r){let e=r.length,s=0,t=this.position;for(;t<e-1;)s++,t+=this.ratio;let f=new Float32Array(s);t=this.position;for(let a=0;a<s;a++){let i=Math.floor(t),o=t-i;f[a]=r[i]*(1-o)+r[i+1]*o,t+=this.ratio}return this.position=Math.max(0,t-e),f}flush(){let r;if(this.format===\"int16\"){let e=new Int16Array(this.framesPerBuffer);for(let s=0;s<this.framesPerBuffer;s++)e[s]=Math.max(-32768,Math.min(32767,Math.round(this.buffer[s]*32768)));r=e.buffer}else r=this.buffer.slice(0,this.framesPerBuffer).buffer;this.port.postMessage(r,[r]),this.buffer=new Float32Array(this.framesPerBuffer),this.bufferIndex=0}};registerProcessor(\"decibri-processor\",u);\n" };
	}));
	//#endregion
	//#region npm/decibri/src/browser/decibri-browser.js
	var require_decibri_browser = /* @__PURE__ */ __commonJSMin(((exports, module) => {
		const { Emitter } = require_emitter();
		const { WORKLET_SOURCE } = require_worklet_inline();
		const VERSION = "5.0.0";
		/**
		* Browser microphone capture.
		*
		* Uses getUserMedia + AudioWorklet for real-time audio capture in browsers.
		* Emits 'data' events with Int16Array or Float32Array chunks.
		*
		* Ported from decibri-web decibri.ts. Logic identical, types removed.
		*
		* @example
		* const { Microphone } = require('decibri'); // browser entry via conditional export
		* const mic = new Microphone({ sampleRate: 16000 });
		* mic.on('data', (chunk) => { // chunk is Int16Array });
		* await mic.start();
		* // later...
		* mic.stop();
		*/
		var Microphone = class extends Emitter {
			constructor(options = {}) {
				super();
				this._audioContext = null;
				this._stream = null;
				this._sourceNode = null;
				this._workletNode = null;
				this._started = false;
				this._starting = null;
				this._stopRequested = false;
				if (options.vadThreshold !== void 0 || options.vadHoldoff !== void 0) throw new TypeError("vadThreshold and vadHoldoff are no longer supported. Pass them on the vad config object: vad: { model: 'energy', threshold: 0.01, holdoffMs: 300 }.");
				const vad = options.vad ?? false;
				let vadThreshold = .01;
				let vadHoldoff = 300;
				if (vad === false) this._vad = false;
				else if (vad === true) throw new TypeError("vad: true is no longer supported. Specify the mode explicitly: vad: 'energy'.");
				else if (vad === "energy") this._vad = true;
				else if (vad !== null && typeof vad === "object" && !Array.isArray(vad)) {
					if (vad.model !== "energy") throw new TypeError(`Invalid vad model: ${JSON.stringify(vad.model)}. Expected 'energy'.`);
					this._vad = true;
					if (vad.threshold !== void 0) {
						if (vad.threshold < 0 || vad.threshold > 1) throw new TypeError(`threshold must be between 0 and 1, got ${vad.threshold}`);
						vadThreshold = vad.threshold;
					}
					if (vad.holdoffMs !== void 0) {
						if (vad.holdoffMs < 0) throw new TypeError(`holdoffMs must be >= 0, got ${vad.holdoffMs}`);
						vadHoldoff = vad.holdoffMs;
					}
				} else throw new TypeError(`Invalid vad value: ${JSON.stringify(vad)}. Expected false, 'energy', or a config object { model, threshold, holdoffMs }.`);
				this._vadThreshold = vadThreshold;
				this._vadHoldoff = vadHoldoff;
				this._vadScore = 0;
				this._isSpeaking = false;
				this._silenceTimer = null;
				this._sampleRate = options.sampleRate ?? 16e3;
				this._channels = options.channels ?? 1;
				this._framesPerBuffer = options.framesPerBuffer ?? 1600;
				this._device = options.device;
				this._dtype = options.dtype ?? "int16";
				this._echoCancellation = options.echoCancellation ?? true;
				this._noiseSuppression = options.noiseSuppression ?? true;
				this._workletUrl = options.workletUrl;
				if (this._sampleRate < 1e3 || this._sampleRate > 384e3) throw new TypeError(`sample rate must be between 1000 and 384000, got ${this._sampleRate}`);
				if (this._channels < 1 || this._channels > 32) throw new TypeError(`channels must be between 1 and 32, got ${this._channels}`);
				if (this._framesPerBuffer < 64 || this._framesPerBuffer > 65536) throw new TypeError(`frames per buffer must be between 64 and 65536, got ${this._framesPerBuffer}`);
				if (this._dtype !== "int16" && this._dtype !== "float32") throw new TypeError("dtype must be 'int16' or 'float32'");
			}
			/**
			* Start microphone capture.
			* Requests microphone permission and sets up the audio pipeline.
			* Must be called from a user gesture context in Safari.
			* No-op if already started. Returns the existing promise if a start
			* is already in progress.
			*/
			start() {
				if (this._started) return Promise.resolve();
				if (this._starting) return this._starting;
				this._starting = this._doStart().finally(() => {
					this._starting = null;
				});
				return this._starting;
			}
			/**
			* Stop microphone capture and release all resources.
			* Safe to call multiple times or before start().
			* After stop(), calling start() again creates a fresh session.
			*/
			stop() {
				if (!this._started) {
					if (this._starting) this._stopRequested = true;
					return;
				}
				this._started = false;
				if (this._stream) this._stream.getTracks().forEach((t) => t.stop());
				if (this._sourceNode) this._sourceNode.disconnect();
				if (this._workletNode) {
					this._workletNode.disconnect();
					this._workletNode.port.close();
				}
				if (this._audioContext) this._audioContext.close();
				if (this._silenceTimer !== null) {
					clearTimeout(this._silenceTimer);
					this._silenceTimer = null;
				}
				this._isSpeaking = false;
				this._audioContext = null;
				this._stream = null;
				this._sourceNode = null;
				this._workletNode = null;
				this.emit("end");
				this.emit("close");
			}
			/** Whether the microphone is currently capturing. */
			get isOpen() {
				return this._started;
			}
			/**
			* Most recent VAD score: the normalized RMS of the last chunk in `'energy'`
			* mode, or 0 when VAD is disabled or before the first chunk is processed.
			* @returns {number}
			*/
			get vadScore() {
				return this._vadScore;
			}
			/**
			* List available audio input devices.
			* Device labels may be empty until microphone permission is granted.
			*/
			static async devices() {
				return (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput").map((d) => ({
					deviceId: d.deviceId,
					label: d.label,
					groupId: d.groupId
				}));
			}
			/** Version information. */
			static version() {
				return { decibri: VERSION };
			}
			async _doStart() {
				this._audioContext = new AudioContext();
				const nativeSampleRate = this._audioContext.sampleRate;
				await this._audioContext.resume();
				const audioConstraints = {
					channelCount: this._channels,
					echoCancellation: this._echoCancellation,
					noiseSuppression: this._noiseSuppression
				};
				if (this._device) audioConstraints.deviceId = { exact: this._device };
				try {
					this._stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
				} catch (err) {
					await this._audioContext.close();
					this._audioContext = null;
					const error = this._mapError(err);
					this.emit("error", error);
					throw error;
				}
				let blobUrl = null;
				const workletUrl = this._workletUrl ?? (blobUrl = this._createBlobUrl());
				try {
					await this._audioContext.audioWorklet.addModule(workletUrl);
				} catch (err) {
					if (blobUrl) URL.revokeObjectURL(blobUrl);
					this._stream.getTracks().forEach((t) => t.stop());
					this._stream = null;
					await this._audioContext.close();
					this._audioContext = null;
					const error = /* @__PURE__ */ new Error("Failed to load audio worklet: " + (err instanceof Error ? err.message : String(err)));
					this.emit("error", error);
					throw error;
				}
				if (blobUrl) URL.revokeObjectURL(blobUrl);
				this._sourceNode = this._audioContext.createMediaStreamSource(this._stream);
				this._workletNode = new AudioWorkletNode(this._audioContext, "decibri-processor", { processorOptions: {
					framesPerBuffer: this._framesPerBuffer,
					format: this._dtype,
					nativeSampleRate,
					targetSampleRate: this._sampleRate
				} });
				this._workletNode.port.onmessage = (event) => {
					const buffer = event.data;
					const chunk = this._dtype === "int16" ? new Int16Array(buffer) : new Float32Array(buffer);
					this.emit("data", chunk);
					if (this._vad) this._processVad(chunk);
				};
				this._sourceNode.connect(this._workletNode);
				this._started = true;
				if (this._stopRequested) {
					this._stopRequested = false;
					this.stop();
				}
			}
			_createBlobUrl() {
				const blob = new Blob([WORKLET_SOURCE], { type: "application/javascript" });
				return URL.createObjectURL(blob);
			}
			_mapError(err) {
				if (err instanceof DOMException) switch (err.name) {
					case "NotAllowedError": return /* @__PURE__ */ new Error("Microphone permission denied");
					case "NotFoundError": return /* @__PURE__ */ new Error("No microphone found");
					default: return /* @__PURE__ */ new Error("Microphone access failed: " + err.message);
				}
				return err instanceof Error ? err : new Error(String(err));
			}
			_processVad(chunk) {
				const rms = this._computeRms(chunk);
				this._vadScore = rms;
				if (rms >= this._vadThreshold) {
					if (this._silenceTimer !== null) {
						clearTimeout(this._silenceTimer);
						this._silenceTimer = null;
					}
					if (!this._isSpeaking) {
						this._isSpeaking = true;
						this.emit("speech");
					}
				} else if (this._isSpeaking && this._silenceTimer === null) this._silenceTimer = setTimeout(() => {
					this._isSpeaking = false;
					this._silenceTimer = null;
					this.emit("silence");
				}, this._vadHoldoff);
			}
			_computeRms(chunk) {
				let sum = 0;
				const n = chunk.length;
				if (n === 0) return 0;
				if (chunk instanceof Float32Array) for (let i = 0; i < n; i++) sum += chunk[i] * chunk[i];
				else for (let i = 0; i < n; i++) {
					const s = chunk[i] / 32768;
					sum += s * s;
				}
				return Math.sqrt(sum / n);
			}
		};
		module.exports = { Microphone };
	}));
	//#endregion
	//#region npm/decibri/src/browser/output-worklet-inline.js
	var require_output_worklet_inline = /* @__PURE__ */ __commonJSMin(((exports, module) => {
		module.exports = { OUTPUT_WORKLET_SOURCE: "class R{constructor(t){this.capacity=t,this.buffer=new Float32Array(t),this.writeIndex=0,this.readIndex=0,this.size=0}get availableRead(){return this.size}get availableWrite(){return this.capacity-this.size}get isEmpty(){return this.size===0}get isFull(){return this.size===this.capacity}write(t){let e=Math.min(t.length,this.availableWrite);for(let s=0;s<e;s++)this.buffer[this.writeIndex]=t[s],this.writeIndex=this.writeIndex+1===this.capacity?0:this.writeIndex+1;return this.size+=e,e}readInto(t,e,s){let i=Math.min(s,this.size);for(let f=0;f<i;f++)t[e+f]=this.buffer[this.readIndex],this.readIndex=this.readIndex+1===this.capacity?0:this.readIndex+1;return this.size-=i,i}clear(){this.writeIndex=0,this.readIndex=0,this.size=0}}class P extends AudioWorkletProcessor{constructor(t){super();let e=t&&t.processorOptions||{},s=e.ringCapacity??96000;this.ring=new R(s),this.hadData=!1,this.port.onmessage=i=>this._onmessage(i)}_onmessage(t){let e=t.data;if(e&&e.type===\"flush\"){this.ring.clear(),this.hadData=!1;return}if(e instanceof ArrayBuffer){let s=new Float32Array(e),i=this.ring.write(s);i>0&&(this.hadData=!0),this.port.postMessage({type:\"level\",queued:this.ring.availableRead,capacity:this.ring.capacity,accepted:i,requested:s.length})}}process(t,e,s){let i=e[0];if(!i||i.length===0)return!0;let f=i[0].length,a=this.ring.readInto(i[0],0,f);for(let o=a;o<f;o++)i[0][o]=0;for(let o=1;o<i.length;o++)i[o].set(i[0]);return this.hadData&&this.ring.isEmpty&&(this.hadData=!1,this.port.postMessage({type:\"drained\"})),!0}}P.RingBuffer=R;registerProcessor(\"decibri-output-processor\",P);\n" };
	}));
	//#endregion
	//#region npm/decibri/src/browser/decibri-output-browser.js
	var require_decibri_output_browser = /* @__PURE__ */ __commonJSMin(((exports, module) => {
		const { OUTPUT_WORKLET_SOURCE } = require_output_worklet_inline();
		const BUFFER_SECONDS = 2;
		/**
		* Linear-interpolation resampler, the inverse of the capture worklet's: it
		* takes samples at the user's rate and produces samples at the context rate.
		* Carries a fractional position across calls so successive writes stay
		* continuous. Logic mirrors worklet-processor.js resample(), with from and to
		* swapped (from = user rate, to = context rate).
		*/
		var Resampler = class {
			constructor(fromRate, toRate) {
				this.ratio = fromRate / toRate;
				this.position = 0;
			}
			process(input) {
				if (this.ratio === 1) return input;
				const inputLength = input.length;
				let count = 0;
				let pos = this.position;
				while (pos < inputLength - 1) {
					count++;
					pos += this.ratio;
				}
				const output = new Float32Array(count);
				pos = this.position;
				for (let i = 0; i < count; i++) {
					const idx = Math.floor(pos);
					const frac = pos - idx;
					output[i] = input[idx] * (1 - frac) + input[idx + 1] * frac;
					pos += this.ratio;
				}
				this.position = Math.max(0, pos - inputLength);
				return output;
			}
		};
		/**
		* Browser audio playback.
		*
		* Plays PCM audio through the Web Audio API. Samples are converted to float32
		* and resampled to the context rate on the main thread, then fed to an output
		* AudioWorklet that drains them to the speakers. This is the inverse of the
		* browser Microphone: the Microphone captures input and emits chunks; the
		* Speaker accepts chunks and plays them.
		*
		* Playback is async throughout, as the browser requires: write() resolves when
		* the samples are queued (after backpressure when the queue is full) and
		* drain() resolves when the queued audio has finished playing. The first
		* write() (or start()) must run in a user gesture so the browser allows audio.
		*
		* @example
		* const { Speaker } = require('decibri'); // browser entry via conditional export
		* const speaker = new Speaker({ sampleRate: 16000 });
		* button.onclick = async () => {
		*   await speaker.write(int16Chunk); // Int16Array of PCM samples
		*   await speaker.drain();
		*   speaker.stop();
		* };
		*/
		var Speaker = class {
			constructor(options = {}) {
				this._audioContext = null;
				this._workletNode = null;
				this._resampler = null;
				this._started = false;
				this._starting = null;
				this._stopRequested = false;
				this._lastAck = null;
				this._capacity = 0;
				this._contextRate = 0;
				this._unplayed = false;
				this._pendingDrains = [];
				this._sampleRate = options.sampleRate ?? 16e3;
				this._channels = options.channels ?? 1;
				this._dtype = options.dtype ?? "int16";
				this._workletUrl = options.workletUrl;
				if (this._sampleRate < 1e3 || this._sampleRate > 384e3) throw new TypeError(`sample rate must be between 1000 and 384000, got ${this._sampleRate}`);
				if (this._channels < 1 || this._channels > 32) throw new TypeError(`channels must be between 1 and 32, got ${this._channels}`);
				if (this._dtype !== "int16" && this._dtype !== "float32") throw new TypeError("dtype must be 'int16' or 'float32'");
			}
			/**
			* Create and resume the AudioContext and load the output worklet.
			*
			* Must be called from a user gesture context so the browser allows audio.
			* Optional: write() starts playback on its own if start() was not called, but
			* calling start() from a click handler is the reliable way to unlock audio
			* before samples are ready. No-op if already started; returns the in-flight
			* promise if a start is already in progress.
			*
			* @returns {Promise<void>}
			*/
			start() {
				if (this._started) return Promise.resolve();
				if (this._starting) return this._starting;
				this._starting = this._doStart().finally(() => {
					this._starting = null;
				});
				return this._starting;
			}
			/**
			* Play PCM audio. Resolves when the samples are queued.
			*
			* Converts the chunk to context-rate float32 (int16 to float32 if needed,
			* then resample), and feeds it to the output worklet. When the queue is full
			* it applies backpressure: the returned promise does not resolve until there
			* is room, so a caller that awaits write() is paced to playback. Starts
			* playback on the first call (gesture-sensitive).
			*
			* Await calls sequentially to preserve sample order. An empty chunk resolves
			* immediately. Writing after stop() starts a fresh playback session.
			*
			* @param {Int16Array|Float32Array|ArrayBuffer} chunk PCM samples in the
			*   configured dtype.
			* @returns {Promise<void>}
			*/
			async write(chunk) {
				await this.start();
				if (!this._started) throw new Error("Speaker is stopped");
				const float32 = this._convert(chunk);
				if (float32.length === 0) return;
				let offset = 0;
				while (offset < float32.length) {
					const sliceLen = Math.min(float32.length - offset, this._capacity);
					await this._reserve(sliceLen);
					if (!this._started) throw new Error("Speaker is stopped");
					const slice = float32.slice(offset, offset + sliceLen);
					this._workletNode.port.postMessage(slice.buffer, [slice.buffer]);
					this._lastAck = {
						queued: this._estimateQueue() + sliceLen,
						time: this._now()
					};
					this._unplayed = true;
					offset += sliceLen;
				}
			}
			/**
			* Wait for all queued audio to finish playing.
			*
			* Resolves when the worklet reports its queue drained. Resolves immediately
			* if nothing is queued (nothing written, or playback already caught up).
			*
			* @returns {Promise<void>}
			*/
			drain() {
				if (!this._started || !this._unplayed) return Promise.resolve();
				return new Promise((resolve) => {
					this._pendingDrains.push(resolve);
				});
			}
			/**
			* Immediate stop. Discards queued audio and releases all resources.
			* Safe to call multiple times or before start(). After stop(), write() or
			* start() begins a fresh session.
			*/
			stop() {
				if (!this._started) {
					if (this._starting) this._stopRequested = true;
					return;
				}
				this._started = false;
				if (this._workletNode) {
					try {
						this._workletNode.port.postMessage({ type: "flush" });
					} catch {}
					this._workletNode.disconnect();
					this._workletNode.port.onmessage = null;
					this._workletNode.port.close();
				}
				if (this._audioContext) this._audioContext.close();
				const drains = this._pendingDrains;
				this._pendingDrains = [];
				for (const resolve of drains) resolve();
				this._audioContext = null;
				this._workletNode = null;
				this._resampler = null;
				this._lastAck = null;
				this._unplayed = false;
			}
			/** Whether audio is currently queued and playing. */
			get isPlaying() {
				return this._started && this._unplayed;
			}
			async _doStart() {
				this._audioContext = new AudioContext();
				this._contextRate = this._audioContext.sampleRate;
				await this._audioContext.resume();
				if (this._audioContext.state === "suspended") {
					await this._audioContext.close();
					this._audioContext = null;
					throw new Error("Audio playback is blocked until a user gesture resumes the audio context");
				}
				let blobUrl = null;
				const workletUrl = this._workletUrl ?? (blobUrl = this._createBlobUrl());
				try {
					await this._audioContext.audioWorklet.addModule(workletUrl);
				} catch (err) {
					if (blobUrl) URL.revokeObjectURL(blobUrl);
					await this._audioContext.close();
					this._audioContext = null;
					throw new Error("Failed to load audio worklet: " + (err instanceof Error ? err.message : String(err)));
				}
				if (blobUrl) URL.revokeObjectURL(blobUrl);
				this._capacity = Math.round(this._contextRate * BUFFER_SECONDS);
				this._resampler = new Resampler(this._sampleRate, this._contextRate);
				this._lastAck = null;
				this._unplayed = false;
				this._workletNode = new AudioWorkletNode(this._audioContext, "decibri-output-processor", {
					numberOfInputs: 0,
					numberOfOutputs: 1,
					outputChannelCount: [this._channels],
					processorOptions: { ringCapacity: this._capacity }
				});
				this._workletNode.port.onmessage = (event) => this._onMessage(event);
				this._workletNode.connect(this._audioContext.destination);
				this._started = true;
				if (this._stopRequested) {
					this._stopRequested = false;
					this.stop();
				}
			}
			_onMessage(event) {
				const msg = event.data;
				if (!msg) return;
				if (msg.type === "level") {
					this._lastAck = {
						queued: msg.queued,
						time: this._now()
					};
					this._capacity = msg.capacity;
				} else if (msg.type === "drained") {
					this._unplayed = false;
					const drains = this._pendingDrains;
					this._pendingDrains = [];
					for (const resolve of drains) resolve();
				}
			}
			_createBlobUrl() {
				const blob = new Blob([OUTPUT_WORKLET_SOURCE], { type: "application/javascript" });
				return URL.createObjectURL(blob);
			}
			_now() {
				return this._audioContext ? this._audioContext.currentTime : 0;
			}
			_estimateQueue() {
				if (!this._lastAck) return 0;
				const elapsed = this._now() - this._lastAck.time;
				const played = Math.max(0, elapsed) * this._contextRate;
				return Math.max(0, this._lastAck.queued - played);
			}
			async _reserve(samples) {
				while (this._started) {
					const overflow = this._estimateQueue() + samples - this._capacity;
					if (overflow <= 0) return;
					const waitMs = Math.max(1, overflow / this._contextRate * 1e3);
					await new Promise((resolve) => setTimeout(resolve, waitMs));
				}
			}
			_convert(chunk) {
				let float32;
				if (this._dtype === "int16") {
					const int16 = this._asInt16(chunk);
					float32 = new Float32Array(int16.length);
					for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768;
				} else float32 = this._asFloat32(chunk);
				return this._resampler.process(float32);
			}
			_asInt16(chunk) {
				if (chunk instanceof Int16Array) return chunk;
				if (chunk instanceof ArrayBuffer) return new Int16Array(chunk);
				if (ArrayBuffer.isView(chunk)) return new Int16Array(chunk.buffer, chunk.byteOffset, Math.floor(chunk.byteLength / 2));
				throw new TypeError("write() expects an Int16Array, ArrayBuffer, or typed array for int16 dtype");
			}
			_asFloat32(chunk) {
				if (chunk instanceof Float32Array) return chunk;
				if (chunk instanceof ArrayBuffer) return new Float32Array(chunk);
				if (ArrayBuffer.isView(chunk)) return new Float32Array(chunk.buffer, chunk.byteOffset, Math.floor(chunk.byteLength / 4));
				throw new TypeError("write() expects a Float32Array, ArrayBuffer, or typed array for float32 dtype");
			}
		};
		module.exports = { Speaker };
	}));
	//#endregion
	return (/* @__PURE__ */ __commonJSMin(((exports, module) => {
		const { Microphone } = require_decibri_browser();
		const { Speaker } = require_decibri_output_browser();
		const { Emitter } = require_emitter();
		module.exports = {
			Microphone,
			Speaker,
			Emitter
		};
	})))();
})();
