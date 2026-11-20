import { FRAME_PHASES } from "../generated/host";
import { engineError, engineFailure } from "../generated/errors";
import type { Error as Failure } from "../generated/reply";
import { parseJson, type Runtime } from "../generated/runtime";
import type { Readiness } from "../generated/stats";
import type { Development } from "./dev";
import { BrowserInput } from "./input";
import { fatal, panicReport } from "./panic";
import { PageSound } from "./sound";
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
    if (!element("fps").checkVisibility()) { this.reset(); return; }
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
    element("render-time").textContent = `${stats.instances.toLocaleString("en-US")} instances · CPU render${(this.submission / this.samples).toFixed(2)} ms · ${gpuTime}`;
    if (this.samples >= 4) {
      this.simulation = 0; this.extraction = 0; this.submission = 0; this.gpu = 0; this.timed = 0; this.samples = 0; this.elapsed = 0; this.intervals = 0;
    }
  }
}

export interface Loading { readiness: Readiness; pipelines: number }
export interface Stopped { title: string; summary: string; detail: string; error?: Failure }

export interface StartWatch {
  shown?(): void;
  failed?(stopped: Stopped): void;
  progress?(loading: Loading): void;
}

export interface GameOptions<T extends Runtime> extends StartWatch {
  create(canvas: HTMLCanvasElement, signal: AbortSignal): Promise<T>;
  source: { url: string; load(signal: AbortSignal): Promise<string>; feed(game: Runtime, signal: AbortSignal): Promise<void> };
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
const sound = new PageSound(canvas);
const instrumented = import.meta.env.MODE === "profiling";
const development = (import.meta.env.DEV || instrumented) && new URLSearchParams(location.search).get("engine_dev") === "1";
canvas.style.visibility = "hidden";
let state: HostState<T> = { kind: "starting" };
let dev: Development | undefined;
let request = 0;
const hud = new FrameHud();
const ticks = new MessageChannel();
let tickCount = 0;
ticks.port1.onmessage = (event: MessageEvent<number>) => {
  if (event.data === request) frame(performance.now());
};
let pixelWidth = 1;
let pixelHeight = 1;
let sampled = { at: -Infinity, phase: -1 };
let feeding: Promise<unknown> = Promise.resolve();

function prepared(game: T, phase: number) {
  const at = performance.now();
  if (phase === sampled.phase && at - sampled.at < 250) return;
  sampled = { at, phase };
  const { readiness, pipelines } = parseJson(game.stats());
  options.progress?.({ readiness, pipelines });
}

function handOver(game: T, hand: () => void): Promise<void> {
  const fed = feeding.then(() => {
    lifecycle.signal.throwIfAborted();
    hand();
    return options.source.feed(game, lifecycle.signal);
  });
  feeding = fed.catch(() => undefined);
  return fed;
}

function setStatus(text: string, shown: string) {
  status.replaceChildren(document.createElement("i"), document.createTextNode(text));
  status.dataset.state = shown;
}

function live(): Live<T> | undefined {
  return state.kind === "preparing" || state.kind === "playing" ? state : undefined;
}

function halted(current: Live<T>) { return current.paused || document.hidden; }

function disposed() { return state.kind === "disposed"; }

function connectInput() {
  input.game = live()?.game;
}

function suspendGame() {
  live()?.game.suspend();
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
  hud.reset();
  if (dev?.pending()) wake();
}

function stopClock() {
  halt();
  suspendGame();
}

function wake() {
  if (request || !live()) return;
  if (!document.hidden) {
    request = requestAnimationFrame(frame);
    return;
  }
  request = -++tickCount;
  ticks.port2.postMessage(request);
}

function fail(error: unknown) {
  if (state.kind === "disposed" || state.kind === "failed") return;
  const stopped = fatal(error) ? panicReport(error)?.message ?? String(error) : undefined;
  state = { kind: "failed" };
  connectInput();
  const refused = engineError(error);
  const title = stopped ? "Engine stopped" : refused ? "Engine error" : "Renderer unavailable";
  const summary = stopped ? "The engine stopped."
    : refused ? "The engine could not start or continue."
    : "Unable to start or continue the engine.";
  const detail = stopped ? stopped
    : refused ? refused.message
    : `${String(error)}\nWebGPU requires a supported browser on localhost or HTTPS.`;
  setStatus(title, "error");
  message.hidden = false;
  message.textContent = `${summary}\n${detail}`;
  pause.disabled = true;
  reset.disabled = true;
  halt();
  dev?.close(stopped ? `runtime_panicked ${stopped}` : `runtime_failed ${refused?.message ?? String(error)}`);
  console.error(error);
  const failure: Failure | undefined = !stopped && refused
    ? { code: refused.code, path: refused.path, allowed: refused.allowed, expected: refused.expected, got: refused.got, nearest: refused.nearest }
    : undefined;
  options.failed?.({ title, summary, detail, error: failure });
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
    if (halted(current)) { if (dev?.pending()) wake(); }
    else if (!request) request = requestAnimationFrame(frame);
  }, () => undefined);
}

function started(current: Live<T>) {
  current.shown = true;
  hud.reset();
  canvas.style.visibility = "";
  message.hidden = true;
  if (!halted(current)) setStatus("WebGPU active", "running");
  options.shown?.();
}

function frame(timestamp: number) {
  request = 0;
  const current = live();
  if (!current) return;
  if (current.kind === "playing" && !current.shown) return;
  if (halted(current)) {
    if (dev?.pending()) run(current, timestamp);
    return;
  }
  run(current, timestamp);
}

function run(current: Live<T>, timestamp: number) {
  if (panicReport()) { fail(engineFailure({ code: "runtime_panicked", path: "$" })); return; }
  const game = current.game;
  try {
    input.pollPads();
    const phase = game.animation_frame(timestamp, pixelWidth, pixelHeight);
    input.setCursorMode(game.cursor_mode());
    sound.exchange(game);
    dev?.turned();
    if (current.kind === "preparing" && options.progress) prepared(game, phase);
    if (phase === FRAME_PHASES.started) {
      starting(current);
    } else if (phase === FRAME_PHASES.playing) {
      hud.frame(game, timestamp);
    } else if (phase === FRAME_PHASES.idle || phase === FRAME_PHASES.paused) {
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

window.addEventListener("blur", () => live()?.game.focused(false), listenerOptions);
window.addEventListener("focus", () => live()?.game.focused(true), listenerOptions);
window.addEventListener("error", event => { if (fatal(event.error)) fail(event.error); }, listenerOptions);
window.addEventListener("unhandledrejection", event => { if (fatal(event.reason)) fail(event.reason); }, listenerOptions);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    halt();
    input.clear();
    live()?.game.hidden(true);
    if (state.kind !== "failed") setStatus("Suspended", "suspended");
    return;
  }
  live()?.game.hidden(false);
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
  handOver(created, () => undefined).catch(fail);
  created.apply_host_settings(location.search);
  connectInput();
  resize();
  message.hidden = true;
  pause.disabled = false;
  reset.disabled = false;
  if (development) {
    const { connectDevelopment } = await import("./dev");
    if (disposed()) return;
    dev = connectDevelopment({ game: created, signal: lifecycle.signal, source: options.source, handOver: hand => handOver(created, hand), wake, fail });
  }
  resume();
}

if (import.meta.hot) import.meta.hot.dispose(() => {
  const current = live();
  lifecycle.abort();
  stopClock(); observer.disconnect(); input.dispose(); sound.dispose(); ticks.port1.close();
  state = { kind: "disposed" };
  current?.game.free();
});
const source = options.source;
const followed = import.meta.hot && URL.parse(source.url, location.href)?.href;
if (import.meta.hot && followed) {
  let edited = false;
  let following = false;
  let shown = false;
  const reloadSource = async () => {
    edited = false;
    const current = live();
    if (!current) return;
    try {
      const text = await source.load(lifecycle.signal);
      if (live() !== current) return;
      await handOver(current.game, () => current.game.reload_scene(text));
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
