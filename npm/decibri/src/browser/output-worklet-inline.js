'use strict';

/**
 * Minified output AudioWorklet processor source, embedded as a string for Blob
 * URL loading.
 *
 * THIS IS THE CODE THAT ACTUALLY RUNS IN THE BROWSER.
 * The readable version is in output-worklet-processor.js (documentation /
 * reference only). If output-worklet-processor.js logic changes, this string
 * MUST be regenerated.
 *
 * Logic identical to output-worklet-processor.js.
 */
const OUTPUT_WORKLET_SOURCE = "class R{constructor(t){this.capacity=t,this.buffer=new Float32Array(t),this.writeIndex=0,this.readIndex=0,this.size=0}get availableRead(){return this.size}get availableWrite(){return this.capacity-this.size}get isEmpty(){return this.size===0}get isFull(){return this.size===this.capacity}write(t){let e=Math.min(t.length,this.availableWrite);for(let s=0;s<e;s++)this.buffer[this.writeIndex]=t[s],this.writeIndex=this.writeIndex+1===this.capacity?0:this.writeIndex+1;return this.size+=e,e}readInto(t,e,s){let i=Math.min(s,this.size);for(let f=0;f<i;f++)t[e+f]=this.buffer[this.readIndex],this.readIndex=this.readIndex+1===this.capacity?0:this.readIndex+1;return this.size-=i,i}clear(){this.writeIndex=0,this.readIndex=0,this.size=0}}class P extends AudioWorkletProcessor{constructor(t){super();let e=t&&t.processorOptions||{},s=e.ringCapacity??96000;this.ring=new R(s),this.hadData=!1,this.port.onmessage=i=>this._onmessage(i)}_onmessage(t){let e=t.data;if(e&&e.type===\"flush\"){this.ring.clear(),this.hadData=!1;return}if(e instanceof ArrayBuffer){let s=new Float32Array(e),i=this.ring.write(s);i>0&&(this.hadData=!0),this.port.postMessage({type:\"level\",queued:this.ring.availableRead,capacity:this.ring.capacity,accepted:i,requested:s.length})}}process(t,e,s){let i=e[0];if(!i||i.length===0)return!0;let f=i[0].length,a=this.ring.readInto(i[0],0,f);for(let o=a;o<f;o++)i[0][o]=0;for(let o=1;o<i.length;o++)i[o].set(i[0]);return this.hadData&&this.ring.isEmpty&&(this.hadData=!1,this.port.postMessage({type:\"drained\"})),!0}}P.RingBuffer=R;registerProcessor(\"decibri-output-processor\",P);\n";

module.exports = { OUTPUT_WORKLET_SOURCE };
