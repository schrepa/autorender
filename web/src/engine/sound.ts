import { SOUND_ORDERS as O } from "../generated/host";
import type { Runtime } from "../generated/runtime";

type SoundGame = Pick<Runtime, "take_sound" | "take_sound_bytes" | "sound_opened" | "sound_open_failed" | "sound_decoded" | "sound_failed" | "voice_ended" | "voice_dropped">;

interface Voice { source: AudioBufferSourceNode; gain: GainNode; panner: StereoPannerNode; rate: number }

function text(bytes: Uint8Array, start: number, length: number): string {
  return String.fromCharCode(...bytes.subarray(start, start + length));
}

function oggVorbis(bytes: Uint8Array): boolean {
  return text(bytes, 0, 4) === "OggS" && text(bytes, 27 + (bytes[26] ?? 0), 7) === "\x01vorbis";
}

function ramp(param: AudioParam, value: number, at: number, seconds: number) {
  param.cancelScheduledValues(at);
  param.setValueAtTime(param.value, at);
  if (seconds > 0) param.linearRampToValueAtTime(value, at + seconds);
  else param.setValueAtTime(value, at);
}

export class PageSound {
  private static readonly MAX_DECODING = 4;
  private context: AudioContext | undefined;
  private master: GainNode | undefined;
  private readonly buffers = new Map<number, AudioBuffer>();
  private readonly voices = new Map<number, Voice>();
  private readonly pendingDecodes: { sound: number; buffer: ArrayBuffer }[] = [];
  private decoding = 0;
  private replies: ((game: SoundGame) => void)[] = [];
  private readonly told = { state: "", rate: 0, channels: 0, latency: 0 };
  private readonly abort = new AbortController();

  constructor(canvas: HTMLCanvasElement) {
    const wake = () => { if (this.context?.state === "suspended") void this.context.resume().catch(() => undefined); };
    canvas.addEventListener("pointerdown", wake, { signal: this.abort.signal });
    canvas.addEventListener("keydown", wake, { signal: this.abort.signal });
  }

  exchange(game: SoundGame) {
    if (this.replies.length > 0) {
      const replies = this.replies;
      this.replies = [];
      for (const reply of replies) reply(game);
    }
    const context = this.context;
    if (context) {
      const latency = context.baseLatency + (context.outputLatency || 0);
      const channels = context.destination.channelCount;
      const told = this.told;
      if (context.state !== told.state || context.sampleRate !== told.rate || channels !== told.channels || !Object.is(latency, told.latency)) {
        Object.assign(told, { state: context.state, rate: context.sampleRate, channels, latency });
        game.sound_opened(context.state === "running", context.sampleRate, channels, latency);
      }
    }
    this.apply(game.take_sound(), game);
  }

  dispose() {
    this.abort.abort();
    this.pendingDecodes.length = 0;
    for (const voice of this.voices.values()) voice.source.onended = null;
    this.voices.clear();
    void this.context?.close().catch(() => undefined);
    this.context = undefined;
  }

  private apply(orders: Float64Array, game: SoundGame) {
    let bytes: Uint8Array | undefined;
    let read = 0;
    let at = 0;
    const field = (offset: number) => orders[at + offset];
    while (at < orders.length) {
      const code = field(0);
      if (code === O.open.code) {
        this.open(field(O.open.latency_given) ? field(O.open.latency_seconds) : null);
        at += O.open.size;
      } else if (code === O.decode.code) {
        bytes ??= game.take_sound_bytes();
        const length = field(O.decode.length);
        this.decode(field(O.decode.sound), bytes.subarray(read, read + length));
        read += length;
        at += O.decode.size;
      } else if (code === O.start.code) {
        this.start(field(O.start.voice), field(O.start.sound), field(O.start.gain), field(O.start.pan), field(O.start.rate), field(O.start.looping) !== 0, field(O.start.offset_seconds));
        at += O.start.size;
      } else if (code === O.tune.code) {
        this.tune(field(O.tune.voice), field(O.tune.gain), field(O.tune.pan), field(O.tune.rate), field(O.tune.seconds));
        at += O.tune.size;
      } else if (code === O.stop.code) {
        this.stop(field(O.stop.voice), field(O.stop.seconds));
        at += O.stop.size;
      } else {
        if (this.context && this.master) ramp(this.master.gain, field(O.master.gain), this.context.currentTime, field(O.master.seconds));
        at += O.master.size;
      }
    }
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
    if (!oggVorbis(bytes)) { this.replies.push(game => game.sound_failed(sound, "not_ogg_vorbis")); return; }
    if (!this.context) { this.replies.push(game => game.sound_failed(sound, "no_output")); return; }
    const copy = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength) as ArrayBuffer;
    this.pendingDecodes.push({ sound, buffer: copy });
    this.pumpDecodes();
  }

  private pumpDecodes() {
    while (this.decoding < PageSound.MAX_DECODING && this.pendingDecodes.length > 0 && this.context) {
      const next = this.pendingDecodes.shift();
      if (!next) return;
      const context = this.context;
      this.decoding += 1;
      context.decodeAudioData(next.buffer).then(
        buffer => {
          this.buffers.set(next.sound, buffer);
          this.replies.push(game => game.sound_decoded(next.sound, buffer.length, buffer.sampleRate, buffer.numberOfChannels));
        },
        (error: unknown) => { this.replies.push(game => game.sound_failed(next.sound, String(error))); },
      ).finally(() => {
        this.decoding -= 1;
        this.pumpDecodes();
      });
    }
  }

  private start(voice: number, sound: number, level: number, pan: number, rate: number, looping: boolean, offset: number) {
    const context = this.context;
    const master = this.master;
    const buffer = this.buffers.get(sound);
    if (!context || !master || !buffer) { this.replies.push(game => game.voice_dropped(voice, "no_output_or_sound")); return; }
    try {
      const source = context.createBufferSource();
      source.buffer = buffer;
      source.loop = looping;
      source.playbackRate.value = rate;
      const gain = context.createGain();
      gain.gain.value = level;
      const panner = context.createStereoPanner();
      panner.pan.value = pan;
      source.connect(gain).connect(panner).connect(master);
      source.onended = () => {
        if (this.voices.get(voice)?.source !== source) return;
        this.voices.delete(voice);
        panner.disconnect();
        this.replies.push(game => game.voice_ended(voice));
      };
      source.start(0, offset);
      this.voices.set(voice, { source, gain, panner, rate });
    } catch (error) {
      this.replies.push(game => game.voice_dropped(voice, String(error)));
    }
  }

  private tune(id: number, gain: number, pan: number, rate: number, seconds: number) {
    const voice = this.voices.get(id);
    const context = this.context;
    if (!voice || !context) return;
    const at = context.currentTime;
    ramp(voice.gain.gain, gain, at, seconds);
    ramp(voice.panner.pan, pan, at, seconds);
    if (rate === voice.rate) return;
    ramp(voice.source.playbackRate, rate, at, seconds);
    voice.rate = rate;
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
