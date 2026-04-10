'use strict';

/**
 * Minified AudioWorklet processor source, embedded as a string for Blob URL loading.
 *
 * THIS IS THE CODE THAT ACTUALLY RUNS IN THE BROWSER.
 * The readable version is in worklet-processor.js (documentation/reference only).
 * If worklet-processor.js logic changes, this string MUST be regenerated.
 *
 * Generated from decibri-web v0.1.1 build output — logic identical to worklet-processor.js.
 */
const WORKLET_SOURCE = "var u=class extends AudioWorkletProcessor{constructor(r){super();let e=r.processorOptions;this.framesPerBuffer=e.framesPerBuffer,this.format=e.format,this.ratio=e.nativeSampleRate/e.targetSampleRate,this.needsResample=e.nativeSampleRate!==e.targetSampleRate,this.position=0,this.buffer=new Float32Array(this.framesPerBuffer),this.bufferIndex=0}process(r,e,s){let t=r[0]?.[0];if(!t||t.length===0)return!0;let f;this.needsResample?f=this.resample(t):f=t;let a=0;for(;a<f.length;){let i=this.framesPerBuffer-this.bufferIndex,o=f.length-a,n=Math.min(i,o);this.buffer.set(f.subarray(a,a+n),this.bufferIndex),this.bufferIndex+=n,a+=n,this.bufferIndex>=this.framesPerBuffer&&this.flush()}return!0}resample(r){let e=r.length,s=0,t=this.position;for(;t<e-1;)s++,t+=this.ratio;let f=new Float32Array(s);t=this.position;for(let a=0;a<s;a++){let i=Math.floor(t),o=t-i;f[a]=r[i]*(1-o)+r[i+1]*o,t+=this.ratio}return this.position=Math.max(0,t-e),f}flush(){let r;if(this.format===\"int16\"){let e=new Int16Array(this.framesPerBuffer);for(let s=0;s<this.framesPerBuffer;s++)e[s]=Math.max(-32768,Math.min(32767,Math.round(this.buffer[s]*32768)));r=e.buffer}else r=this.buffer.slice(0,this.framesPerBuffer).buffer;this.port.postMessage(r,[r]),this.buffer=new Float32Array(this.framesPerBuffer),this.bufferIndex=0}};registerProcessor(\"decibri-processor\",u);\n";

module.exports = { WORKLET_SOURCE };
