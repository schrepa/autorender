import type { Runtime } from "../generated/runtime";

export type SoundOrder =
  | { open: true; latency: number | null }
  | { decode: number; bytes: Uint8Array }
  | { start: number; sound: number; gain: number; pan: number; rate: number; looping: boolean; offset: number }
  | { tune: number; gain: number; pan: number; rate: number; seconds: number }
  | { stop: number; seconds: number }
  | { master: number; seconds: number };

type SoundGame = Pick<Runtime, "take_sound" | "sound_opened" | "sound_open_failed" | "sound_decoded" | "sound_failed" | "voice_ended" | "voice_dropped">;

interface Voice { source: AudioBufferSourceNode; gain: GainNode; panner: StereoPannerNode }

function ramp(param: AudioParam, value: number, at: number, seconds: number) {
  param.cancelScheduledValues(at);
  param.setValueAtTime(param.value, at);
  if (seconds > 0) param.linearRampToValueAtTime(value, at + seconds);
  else param.setValueAtTime(value, at);
}

export class PageSound {
  private context: AudioContext | undefined;
  private master: GainNode | undefined;
  private readonly buffers = new Map<number, AudioBuffer>();
  private readonly voices = new Map<number, Voice>();
  private replies: ((game: SoundGame) => void)[] = [];
  private told = "";
  private readonly abort = new AbortController();

  constructor(canvas: HTMLCanvasElement) {
    const wake = () => { if (this.context?.state === "suspended") void this.context.resume().catch(() => undefined); };
    canvas.addEventListener("pointerdown", wake, { signal: this.abort.signal });
    canvas.addEventListener("keydown", wake, { signal: this.abort.signal });
  }

  exchange(game: SoundGame) {
    for (const reply of this.replies.splice(0)) reply(game);
    const context = this.context;
    if (context) {
      const latency = context.baseLatency + (context.outputLatency || 0);
      const state = `${context.state} ${context.sampleRate} ${context.destination.channelCount} ${latency}`;
      if (state !== this.told) {
        this.told = state;
        game.sound_opened(context.state === "running", context.sampleRate, context.destination.channelCount, latency);
      }
    }
    for (const order of game.take_sound()) this.apply(order);
  }

  dispose() {
    this.abort.abort();
    for (const voice of this.voices.values()) voice.source.onended = null;
    this.voices.clear();
    void this.context?.close().catch(() => undefined);
    this.context = undefined;
  }

  private apply(order: SoundOrder) {
    if ("open" in order) this.open(order.latency);
    else if ("decode" in order) this.decode(order.decode, order.bytes);
    else if ("start" in order) this.start(order);
    else if ("tune" in order) this.tune(order);
    else if ("stop" in order) this.stop(order.stop, order.seconds);
    else if (this.context && this.master) ramp(this.master.gain, order.master, this.context.currentTime, order.seconds);
  }

  private open(latency: number | null) {
    if (this.context) return;
    try {
      const context = new AudioContext(latency === null ? {} : { latencyHint: latency });
      const master = context.createGain();
      master.connect(context.destination);
      this.context = context;
      this.master = master;
    } catch (error) {
      this.replies.push(game => game.sound_open_failed(String(error)));
    }
  }

  private decode(sound: number, bytes: Uint8Array) {
    const context = this.context;
    if (!context) { this.replies.push(game => game.sound_failed(sound, "no_output")); return; }
    const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    context.decodeAudioData(copy).then(
      buffer => {
        this.buffers.set(sound, buffer);
        this.replies.push(game => game.sound_decoded(sound, buffer.length, buffer.sampleRate, buffer.numberOfChannels));
      },
      (error: unknown) => { this.replies.push(game => game.sound_failed(sound, String(error))); },
    );
  }

  private start(order: Extract<SoundOrder, { start: number }>) {
    const voice = order.start;
    const context = this.context;
    const master = this.master;
    const buffer = this.buffers.get(order.sound);
    if (!context || !master || !buffer) { this.replies.push(game => game.voice_dropped(voice, "no_output_or_sound")); return; }
    try {
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.loop = order.looping;
      source.playbackRate.value = order.rate;
      const gain = context.createGain();
      gain.gain.value = order.gain;
      const panner = context.createStereoPanner();
      panner.pan.value = order.pan;
      source.connect(gain).connect(panner).connect(master);
      source.onended = () => {
        if (this.voices.get(voice)?.source !== source) return;
        this.voices.delete(voice);
        panner.disconnect();
        this.replies.push(game => game.voice_ended(voice));
      };
      source.start(0, order.offset);
      this.voices.set(voice, { source, gain, panner });
    } catch (error) {
      this.replies.push(game => game.voice_dropped(voice, String(error)));
    }
  }

  private tune(order: Extract<SoundOrder, { tune: number }>) {
    const voice = this.voices.get(order.tune);
    const context = this.context;
    if (!voice || !context) return;
    const at = context.currentTime;
    ramp(voice.gain.gain, order.gain, at, order.seconds);
    ramp(voice.panner.pan, order.pan, at, order.seconds);
    ramp(voice.source.playbackRate, order.rate, at, order.seconds);
  }

  private stop(id: number, seconds: number) {
    const voice = this.voices.get(id);
    const context = this.context;
    if (!voice || !context) return;
    const at = context.currentTime;
    ramp(voice.gain.gain, 0, at, seconds);
    voice.source.stop(at + seconds);
  }
}
