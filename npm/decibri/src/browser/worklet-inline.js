'use strict';

/**
 * Minified AudioWorklet processor source, embedded as a string for Blob URL loading.
 *
 * THIS IS THE CODE THAT ACTUALLY RUNS IN THE BROWSER.
 * The readable version is in worklet-processor.js (documentation/reference only).
 * If worklet-processor.js logic changes, this string MUST be regenerated.
 *
 * Logic identical to worklet-processor.js.
 */
const WORKLET_SOURCE = "var e=class extends AudioWorkletProcessor{constructor(e){super();let t=e.processorOptions;this.framesPerBuffer=t.framesPerBuffer,this.format=t.format,this.ratio=t.nativeSampleRate/t.targetSampleRate,this.needsResample=t.nativeSampleRate!==t.targetSampleRate,this.channelMap=t.channelMap??null,this.mapError=!1,this.position=0,this.buffer=new Float32Array(this.framesPerBuffer),this.bufferIndex=0}process(e,t,n){let r=e[0];if(!r||r.length===0||!r[0]||r[0].length===0)return!0;let i=r.length,a;if(this.channelMap){if(this.mapError)return!1;for(let e=0;e<this.channelMap.length;e++)if(this.channelMap[e]>=i)return this.mapError=!0,this.port.postMessage({type:`error`,message:`the channel map names device channel `+this.channelMap[e]+`; the device reports `+i+` input channels`}),!1;a=r[this.channelMap[0]]}else if(i===1)a=r[0];else{let e=r[0].length;a=new Float32Array(e);for(let t=0;t<e;t++){let e=0;for(let n=0;n<i;n++)e=Math.fround(e+r[n][t]);a[t]=e/i}}let o;o=this.needsResample?this.resample(a):a;let s=0;for(;s<o.length;){let e=this.framesPerBuffer-this.bufferIndex,t=o.length-s,n=Math.min(e,t);this.buffer.set(o.subarray(s,s+n),this.bufferIndex),this.bufferIndex+=n,s+=n,this.bufferIndex>=this.framesPerBuffer&&this.flush()}return!0}resample(e){let t=e.length,n=0,r=this.position;for(;r<t-1;)n++,r+=this.ratio;let i=new Float32Array(n);r=this.position;for(let t=0;t<n;t++){let n=Math.floor(r),a=r-n;i[t]=e[n]*(1-a)+e[n+1]*a,r+=this.ratio}return this.position=Math.max(0,r-t),i}flush(){let e;if(this.format===`int16`){let t=new Int16Array(this.framesPerBuffer);for(let e=0;e<this.framesPerBuffer;e++)t[e]=Math.max(-32768,Math.min(32767,Math.round(this.buffer[e]*32768)));e=t.buffer}else e=this.buffer.slice(0,this.framesPerBuffer).buffer;this.port.postMessage(e,[e]),this.buffer=new Float32Array(this.framesPerBuffer),this.bufferIndex=0}};registerProcessor(`decibri-processor`,e);";

module.exports = { WORKLET_SOURCE };
