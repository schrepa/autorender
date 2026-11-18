import { FRAME_INTERVAL_MS, FRAME_PHASES, HOST_PAUSES } from "../generated/host";
import { engineError, engineFailure } from "../generated/errors";
import { parseJson, type Runtime } from "../generated/runtime";
import type { Development } from "./dev";
import { BrowserInput } from "./input";
import { type JobPool, jobPool } from "./jobs";
import { fatal, panicReport } from "./panic";
import "./style.css";

export type { Runtime };
const PAGE_ELEMENTS = ["scene", "status", "message", "pause", "reset", "fps", "simulation", "extraction", "draws", "uploads", "render-time"];

function ensurePage() {
  if (!document.getElementById("scene")) {
    document.body.classList.add("engine-page");
    document.body.insertAdjacentHTML("afterbegin", `<canvas id="scene" tabindex="0" aria-label="Game view"></canvas>
<div class="engine-hud"><span id="status" role="status"><i></i>Initializing</span><strong id="fps">—</strong>
<span>simulation <span id="simulation">—</span></span><span>extraction <span id="extraction">—</span></span>
<span>draws <span id="draws">—</span></span><span id="uploads"></span><span id="render-time"></span>
<button id="pause" type="button">Pause</button><button id="reset" type="button">Reset view</button></div>
<div id="message" class="message" role="status">Loading…</div>`);
  }
  const missing = PAGE_ELEMENTS.filter(id => !document.getElementById(id));
  if (missing.length) {
    throw engineFailure({ code: "missing_resource", path: "$", expected: "present", got: missing.map(id => `#${id}`) });
  }
}

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw engineFailure({ code: "missing_resource", path: "$", expected: "present", got: `#${id}` });
  return found as T;
}

function metric(id: string, value: string, unit = "") {
  const node = element(id);
  node.replaceChildren(document.createTextNode(value));
  if (unit) {
    const small = document.createElement("small");
    small.textContent = ` ${unit}`;
    node.append(small);
  }
}

class FrameHud {
  private span: { start: number; frames: number } | undefined;
  private intervals = 0;
  private elapsed = 0;
  private simulation = 0;
  private extraction = 0;
  private submission = 0;
  private gpu = 0;
  private timed = 0;
  private samples = 0;
  private marked = { at: -Infinity, phase: -1 };

  preparing(runtime: Runtime, phase: number) {
    const at = performance.now();
    if (phase === this.marked.phase && at - this.marked.at < 250) return;
    this.marked = { at, phase };
    const stats = parseJson(runtime.stats());
    performance.mark("engine-preparing", { detail: {
      phase, tick: stats.tick, preparation_ms: stats.preparation_ms, readiness: stats.readiness,
      pipelines: stats.pipelines, pipelines_created: stats.pipelines_created, jobs: stats.jobs, upload_bytes: stats.upload_bytes,
    } });
  }

  reset() {
    this.span = undefined;
    this.intervals = 0; this.elapsed = 0;
    this.simulation = 0; this.extraction = 0; this.submission = 0; this.gpu = 0; this.timed = 0; this.samples = 0;
  }

  frame(runtime: Runtime, timestamp: number) {
    const span = this.span;
    if (!span) { this.span = { start: timestamp, frames: 0 }; return; }
    span.frames++;
    if (timestamp - span.start < 250) return;
    this.elapsed += timestamp - span.start; this.intervals += span.frames;
    this.span = { start: timestamp, frames: 0 };
    const stats = parseJson(runtime.stats());
    const gpu = stats.resolution.gpu_ms;
    this.simulation += stats.simulation_ms;
    this.extraction += stats.extraction_ms;
    this.submission += stats.render_ms;
    if (typeof gpu === "number") { this.gpu += gpu; this.timed++; }
    this.samples++;
    metric("fps", this.elapsed > 0 ? (this.intervals * 1000 / this.elapsed).toFixed(0) : "—", "FPS");
    metric("simulation", (this.simulation / this.samples).toFixed(2), "ms");
    metric("extraction", (this.extraction / this.samples).toFixed(2), "ms");
    metric("draws", String(stats.draw_calls));
    element("uploads").textContent = `${(stats.upload_bytes / 1024).toFixed(0)} KiB / frame`;
    const gpuTime = typeof gpu === "number" ? `GPU ${(this.gpu / this.timed).toFixed(2)} ms` : `GPU time unavailable: ${gpu.unavailable}`;
    element("render-time").textContent = `${stats.instances.toLocaleString("en-US")} instances · ${stats.indirect_draws} indirect draws · CPU render ${(this.submission / this.samples).toFixed(2)} ms · ${gpuTime}`;
    if (this.samples >= 4) {
      this.simulation = 0; this.extraction = 0; this.submission = 0; this.gpu = 0; this.timed = 0; this.samples = 0; this.elapsed = 0; this.intervals = 0;
    }
  }
}

class GpuPace {
  private drawn = 0;
  private since = 0;
  private done = false;
  private shown = true;
  private refresh = FRAME_INTERVAL_MS;
  private animated: number | undefined;
  private stamp = -Infinity;
  private posted = 0;
  private tick = 0;
  private readonly channel = new MessageChannel();

  constructor(private readonly canvas: HTMLCanvasElement, early: () => void) {
    this.channel.port1.onmessage = (event: MessageEvent<number>) => {
      if (event.data !== this.posted) return;
      this.posted = 0;
      early();
    };
  }

  animation(timestamp: number): boolean {
    if (this.animated !== undefined && timestamp > this.animated) {
      this.refresh = Math.min(this.refresh, timestamp - this.animated);
    }
    this.animated = timestamp;
    if (this.shown) return false;
    this.shown = true;
    this.offer();
    return true;
  }

  time(timestamp: number): number {
    this.stamp = Math.max(this.stamp, timestamp);
    return this.stamp;
  }

  submitted(since: number, early: boolean) {
    const drawn = ++this.drawn;
    this.since = since;
    this.done = false;
    this.shown = !early;
    this.posted = 0;
    const queue = (this.canvas.getContext("webgpu") as GPUCanvasContext | null)?.getConfiguration()?.device.queue;
    void queue?.onSubmittedWorkDone().then(() => {
      if (drawn !== this.drawn) return;
      this.done = true;
      this.offer();
    }, () => undefined);
  }

  stop() {
    this.drawn++;
    this.done = false;
    this.shown = true;
    this.posted = 0;
  }

  dispose() {
    this.stop();
    this.channel.port1.close();
  }

  private offer() {
    if (!this.done || !this.shown || this.posted || performance.now() - this.since < this.refresh) return;
    this.posted = ++this.tick;
    this.channel.port2.postMessage(this.posted);
  }
}

export interface GameOptions<T extends Runtime> {
  create(canvas: HTMLCanvasElement, signal: AbortSignal): Promise<T>;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files(source: string, signal: AbortSignal): Promise<unknown> };
}

interface Live<T> { kind: "preparing" | "playing"; game: T; paused: boolean; shown: boolean }
type HostState<T> = { kind: "starting" } | Live<T> | { kind: "failed" } | { kind: "disposed" };

export function startGame<T extends Runtime>(options: GameOptions<T>) {
ensurePage();

const lifecycle = new AbortController();
const listenerOptions = { signal: lifecycle.signal };

const canvas = element<HTMLCanvasElement>("scene");
const status = element("status");
const message = element("message");
const pause = element<HTMLButtonElement>("pause");
const reset = element<HTMLButtonElement>("reset");
const input = new BrowserInput(canvas);
const instrumented = import.meta.env.MODE === "profiling";
const development = (import.meta.env.DEV || instrumented) && new URLSearchParams(location.search).get("engine_dev") === "1";
canvas.style.visibility = "hidden";
let state: HostState<T> = { kind: "starting" };
let dev: Development | undefined;
let request = 0;
const hud = new FrameHud();
const pace = new GpuPace(canvas, () => {
  const current = live();
  if (current?.kind === "playing" && current.shown && !halted(current)) run(current, performance.now(), true);
});
let jobs: JobPool | undefined;
const ticks = new MessageChannel();
let tickCount = 0;
ticks.port1.onmessage = (event: MessageEvent<number>) => {
  if (event.data === request) frame(performance.now());
};
let pixelWidth = 1;
let pixelHeight = 1;

function setStatus(text: string, shown: string) {
  status.replaceChildren(document.createElement("i"), document.createTextNode(text));
  status.dataset.state = shown;
}

function live(): Live<T> | undefined {
  return state.kind === "preparing" || state.kind === "playing" ? state : undefined;
}

function halted(current: Live<T>) { return current.paused || document.hidden; }

function disposed() { return state.kind === "disposed"; }

function clearInput() { input.clear(); }

function connectInput() {
  input.game = live()?.game;
}

function suspendGame() {
  live()?.game.suspend();
}

function pauseGame(reason: keyof typeof HOST_PAUSES) {
  clearInput();
  live()?.game.pause(HOST_PAUSES[reason]);
}

function resumeGame(reason: keyof typeof HOST_PAUSES) {
  live()?.game.resume(HOST_PAUSES[reason]);
}

function resize() {
  const current = live();
  if (!current) return;
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(bounds.width * ratio);
  const height = Math.round(bounds.height * ratio);
  const idle = pixelWidth === 0 || pixelHeight === 0;
  pixelWidth = width;
  pixelHeight = height;
  current.game.set_pixel_ratio(ratio);
  if (idle && width > 0 && height > 0) resume();
}
const observer = new ResizeObserver(resize);
observer.observe(canvas);
window.addEventListener("resize", resize, listenerOptions);

function halt() {
  if (request > 0) cancelAnimationFrame(request);
  request = 0;
  pace.stop();
  hud.reset();
}

function stopClock() {
  halt();
  suspendGame();
}

function wake() {
  if (request || !live()) return;
  request = -++tickCount;
  ticks.port2.postMessage(request);
}

function fail(error: unknown) {
  if (state.kind === "disposed" || state.kind === "failed") return;
  const stopped = fatal(error) ? panicReport(error)?.message ?? String(error) : undefined;
  state = { kind: "failed" };
  connectInput();
  const refused = engineError(error);
  setStatus(stopped ? "Engine stopped" : refused ? "Engine error" : "Renderer unavailable", "error");
  message.hidden = false;
  message.textContent = stopped ? `The engine stopped.\n${stopped}`
    : refused ? `The engine could not start or continue.\n${refused.message}`
    : `Unable to start or continue the engine.\n${String(error)}\nWebGPU requires a supported browser on localhost or HTTPS.`;
  pause.disabled = true;
  reset.disabled = true;
  halt();
  dev?.close(stopped ? `runtime_panicked ${stopped}` : `runtime_failed ${refused?.message ?? String(error)}`);
  (jobs ?? jobPool())?.terminate();
  jobs = undefined;
  console.error(error);
}

function guarded(work: () => void) {
  try { work(); } catch (error) { fail(error); }
}

function starting(current: Live<T>) {
  current.kind = "playing";
  const queue = (canvas.getContext("webgpu") as GPUCanvasContext | null)?.getConfiguration()?.device.queue;
  if (!queue) { started(current); return; }
  void queue.onSubmittedWorkDone().then(() => {
    if (live() !== current || current.shown) return;
    started(current);
    if (!request && !halted(current)) request = requestAnimationFrame(frame);
  }, () => undefined);
}

function started(current: Live<T>) {
  current.shown = true;
  jobs?.started();
  hud.reset();
  canvas.style.visibility = "";
  message.hidden = true;
  if (!halted(current)) setStatus("WebGPU active", "running");
}

function frame(timestamp: number) {
  request = 0;
  const current = live();
  if (!current) return;
  if (current.kind === "playing" && !current.shown) return;
  if (halted(current)) {
    if (dev?.pending()) run(current, timestamp, false);
    return;
  }
  if (current.kind === "playing" && pace.animation(timestamp)) {
    request = requestAnimationFrame(frame);
    return;
  }
  run(current, timestamp, false);
}

function run(current: Live<T>, at: number, early: boolean) {
  if (panicReport()) { fail(engineFailure({ code: "runtime_panicked", path: "$" })); return; }
  const game = current.game;
  const since = performance.now();
  const timestamp = current.kind === "playing" ? pace.time(at) : at;
  try {
    jobs?.exchange();
    input.pollPads();
    const phase = game.animation_frame(timestamp, pixelWidth, pixelHeight);
    if (phase === FRAME_PHASES.playing) pace.submitted(since, early);
    input.setCursorMode(game.cursor_mode());
    jobs?.exchange();
    dev?.turned();
    if (current.kind === "preparing") hud.preparing(game, phase);
    if (phase === FRAME_PHASES.started) {
      starting(current);
    } else if (phase === FRAME_PHASES.playing) {
      hud.frame(game, timestamp);
    } else if (phase === FRAME_PHASES.idle || phase === FRAME_PHASES.paused) {
      pace.stop();
      hud.reset();
      if (dev?.pending()) wake();
      return;
    }
    if (!live() || request || (current.kind === "playing" && !current.shown)) return;
    if (!halted(current)) schedule(phase);
    else if (dev?.pending()) wake();
  } catch (error) { fail(error); }
}

function schedule(phase: number) {
  if (phase === FRAME_PHASES.collecting) {
    request = -++tickCount;
    ticks.port2.postMessage(request);
  } else {
    request = requestAnimationFrame(frame);
  }
}

function resume() {
  const current = live();
  if (!current || document.hidden) return;
  if (current.paused) { setStatus("Paused", "paused"); return; }
  if (request) return;
  if (!current.shown) {
    message.hidden = false;
    message.textContent = "Loading…";
    setStatus("Loading", "preparing");
  } else {
    setStatus("WebGPU active", "running");
  }
  if (current.kind === "playing" && !current.shown) return;
  request = requestAnimationFrame(frame);
}

window.addEventListener("blur", () => pauseGame("focus"), listenerOptions);
window.addEventListener("focus", () => resumeGame("focus"), listenerOptions);
window.addEventListener("error", event => { if (fatal(event.error)) fail(event.error); }, listenerOptions);
window.addEventListener("unhandledrejection", event => { if (fatal(event.reason)) fail(event.reason); }, listenerOptions);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    halt();
    pauseGame("hidden");
    if (state.kind !== "failed") setStatus("Suspended", "suspended");
    return;
  }
  resumeGame("hidden");
  if (state.kind !== "failed") resume();
}, listenerOptions);
pause.addEventListener("click", () => {
  const current = live();
  if (!current) return;
  current.paused = !current.paused;
  pause.textContent = current.paused ? "Resume" : "Pause";
  pause.setAttribute("aria-pressed", String(current.paused));
  stopClock();
  if (current.paused) setStatus("Paused", "paused");
  else resume();
}, listenerOptions);
reset.addEventListener("click", () => {
  const current = live();
  if (current) guarded(() => current.game.reset_camera());
}, listenerOptions);
async function start() {
  if (disposed()) return;
  pause.disabled = true;
  reset.disabled = true;
  if (!("gpu" in navigator)) throw engineFailure({ code: "runtime_unavailable", path: "$", expected: "gpu" });
  const created = await options.create(canvas, lifecycle.signal);
  if (disposed()) { created.free(); return; }
  state = { kind: "preparing", game: created, paused: false, shown: false };
  created.apply_host_settings(location.search);
  connectInput();
  jobs = jobPool();
  jobs?.attach(created, fail);
  resize();
  message.hidden = true;
  pause.disabled = false;
  reset.disabled = false;
  if (development) {
    const { connectDevelopment } = await import("./dev");
    if (disposed()) return;
    dev = connectDevelopment({ game: created, signal: lifecycle.signal, source: options.source, wake, fail });
  }
  resume();
}

window.addEventListener("pagehide", () => { halt(); pauseGame("suspended"); }, listenerOptions);
window.addEventListener("pageshow", () => { resumeGame("suspended"); resume(); }, listenerOptions);
if (import.meta.hot) import.meta.hot.dispose(() => {
  const current = live();
  jobs?.detach();
  lifecycle.abort();
  stopClock(); observer.disconnect(); input.dispose(); ticks.port1.close(); pace.dispose();
  state = { kind: "disposed" };
  current?.game.free();
});
const source = options.source;
const followed = import.meta.hot && source && URL.parse(source.url, location.href)?.href;
if (import.meta.hot && source && followed) {
  let edited = false;
  let following = false;
  let shown = false;
  const reloadSource = async () => {
    edited = false;
    const current = live();
    if (!current) return;
    try {
      const text = await source.load(lifecycle.signal);
      const files = await source.files(text, lifecycle.signal);
      if (live() !== current) return;
      current.game.reload_scene(text, files);
      hud.reset();
      if (shown) message.hidden = true;
      shown = false;
    } catch (error) {
      if (fatal(error)) { fail(error); return; }
      if (disposed()) return;
      const refused = engineError(error);
      shown = true;
      message.hidden = false;
      message.textContent = `The scene edit was not applied; the previous scene is still running.\n${refused?.message ?? String(error)}`;
      console.error(`{"event":"source_changed","error":${refused?.message ?? JSON.stringify(String(error))}}`);
    }
  };
  const follow = async () => {
    following = true;
    while (edited && (state.kind === "starting" || live())) {
      await (live() ? reloadSource() : new Promise(resolve => setTimeout(resolve, 100)));
    }
    following = false;
  };
  import.meta.hot.on("engine:scene-changed", ({ urls }: { urls: string[] }) => {
    if (!urls.some(url => URL.parse(url, location.href)?.href === followed)) return;
    edited = true;
    if (!following) void follow();
  });
}
void start().catch(fail);
}
