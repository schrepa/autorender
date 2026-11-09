import { type Distribution, FRAME_PHASES, HOST_PAUSES } from "../generated/host";
import { engineError, engineFailure } from "../generated/errors";
import { parseJson, type Runtime } from "../generated/runtime";
import type { FrameStats } from "../generated/stats";
import { BrowserInput } from "./input";
import { type JobPool, jobPool } from "./jobs";
import { fatal, panicReport } from "./panic";
import { HOST_PARAMETERS } from "./parameters";
import "./style.css";

export type { Distribution, FrameStats, Runtime };
export function readStats(runtime: Pick<Runtime, "stats">): FrameStats {
  return parseJson(runtime.stats());
}
export function readDistribution(runtime: Pick<Runtime, "distribution">, values: readonly number[]): Distribution | null {
  return parseJson(runtime.distribution(Float64Array.from(values)));
}
export interface HostControls<T extends Runtime> {
  mutate(action: (game: T) => void): void;
  setResolutionScale(scale: number): void;
  setInputEnabled(enabled: boolean): void;
  distribution(values: readonly number[]): Distribution | null;
  size(): [number, number];
  paused(): boolean;
  busy(): boolean;
}
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
  private frames = 0;
  private span: { start: number; frames: number } | undefined;
  private intervals = 0;
  private elapsed = 0;
  private simulation = 0;
  private extraction = 0;
  private submission = 0;
  private samples = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {}

  reset() {
    this.span = undefined;
    this.intervals = 0; this.elapsed = 0;
    this.simulation = 0; this.extraction = 0; this.submission = 0; this.samples = 0;
  }

  frame(runtime: Runtime, timestamp: number): boolean {
    this.frames++;
    const span = this.span;
    if (!span) { this.span = { start: timestamp, frames: 0 }; return false; }
    span.frames++;
    if (timestamp - span.start < 250) return false;
    this.elapsed += timestamp - span.start; this.intervals += span.frames;
    this.span = { start: timestamp, frames: 0 };
    const stats = readStats(runtime);
    this.simulation += stats.simulation_ms;
    this.extraction += stats.extraction_ms;
    this.submission += stats.render_ms;
    this.samples++;
    metric("fps", this.elapsed > 0 ? (this.intervals * 1000 / this.elapsed).toFixed(0) : "—", "FPS");
    metric("simulation", (this.simulation / this.samples).toFixed(2), "ms");
    metric("extraction", (this.extraction / this.samples).toFixed(2), "ms");
    metric("draws", String(stats.draw_calls));
    element("uploads").textContent = `${(stats.upload_bytes / 1024).toFixed(0)} KiB / frame`;
    const triangles = stats.triangles === null ? "" : ` · ${stats.triangles.toLocaleString("en-US")} triangles`;
    element("render-time").textContent = `${stats.instances} drawn${triangles} · ${stats.meshes_stats.culled_instances} culled · CPU render ${(this.submission / this.samples).toFixed(2)} ms · GPU time not measured`;
    const dataset = this.canvas.dataset;
    dataset.frames = String(this.frames);
    dataset.instances = String(stats.instances);
    dataset.renderCandidates = String(stats.meshes_stats.render_candidates);
    dataset.residentMeshes = String(stats.resident_meshes);
    dataset.meshUploadBytes = String(stats.mesh_upload_bytes);
    dataset.residentMeshBytes = String(stats.resident_mesh_bytes);
    dataset.residentTextures = String(stats.resident_textures);
    dataset.textureUploadBytes = String(stats.texture_upload_bytes);
    if (this.samples >= 4) {
      this.simulation = 0; this.extraction = 0; this.submission = 0; this.samples = 0; this.elapsed = 0; this.intervals = 0;
    }
    return true;
  }
}

export interface GameOptions<T extends Runtime> {
  create(canvas: HTMLCanvasElement, signal: AbortSignal): Promise<T>;
  controls?: Record<string, { values: string; run(game: T, args: Record<string, unknown>): unknown }>;
  parameters?: Record<string, string>;
  poll?(game: T): void;
  level?(game: T, path: string, signal: AbortSignal, progress: (done: number, total: number) => void): Promise<void>;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files(source: string, signal: AbortSignal): Promise<unknown> };
  ready?(game: T, controls: HostControls<T>): void;
  beforeFrame?(game: T, timestamp: number): void;
  afterFrame?(game: T, timestamp: number): void;
  afterHostFrame?(game: T, timestamp: number): void;
  interrupted?(reason: string): void;
}

interface Live<T> { kind: "preparing" | "playing"; game: T; paused: boolean; busy: boolean; level: boolean }
type HostState<T> = { kind: "starting" } | Live<T> | { kind: "failed"; stopped: string | undefined; busy: boolean } | { kind: "disposed" };

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
const preparedDevelopment = development && new URLSearchParams(location.search).get("prepare") === "1";
const prepares = !development || preparedDevelopment;
if (prepares) canvas.style.visibility = "hidden";
let state: HostState<T> = { kind: "starting" };
let disposeDevelopment: (() => void) | undefined;
let request = 0;
const hud = new FrameHud(canvas);
let jobs: JobPool | undefined;
let pixelWidth = 1;
let pixelHeight = 1;
let inputEnabled = true;

function setStatus(text: string, shown: string) {
  status.replaceChildren(document.createElement("i"), document.createTextNode(text));
  status.dataset.state = shown;
}

function live(): Live<T> | undefined {
  return state.kind === "preparing" || state.kind === "playing" ? state : undefined;
}

function disposed() { return state.kind === "disposed"; }

function exclusive<R>(work: () => Promise<R>): Promise<R> | undefined {
  const current = state;
  if (current.kind === "starting" || current.kind === "disposed" || current.busy) return undefined;
  current.busy = true;
  return work().finally(() => {
    if (state.kind !== "starting" && state.kind !== "disposed") state.busy = false;
  });
}

function clearInput() { input.clear(); }

function connectInput() {
  input.game = inputEnabled ? live()?.game : undefined;
}

function interrupt(reason: string) {
  if (state.kind !== "failed") options.interrupted?.(reason);
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
  if (width !== pixelWidth || height !== pixelHeight) interrupt("resize");
  pixelWidth = width;
  pixelHeight = height;
  current.game.set_pixel_ratio(ratio);
  if (idle && width > 0 && height > 0) resume();
}
const observer = new ResizeObserver(resize);
observer.observe(canvas);
window.addEventListener("resize", resize, listenerOptions);

function halt() {
  cancelAnimationFrame(request);
  request = 0;
  hud.reset();
}

function stopClock() {
  interrupt("suspension");
  halt();
  suspendGame();
}

function fail(error: unknown) {
  if (state.kind === "disposed" || state.kind === "failed") return;
  const stopped = fatal(error) ? panicReport(error)?.message ?? String(error) : undefined;
  state = { kind: "failed", stopped, busy: state.kind !== "starting" && state.busy };
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
  jobs?.terminate();
  jobs = undefined;
  console.error(error);
}

function guarded(work: () => void) {
  try { work(); } catch (error) { fail(error); }
}

function started(current: Live<T>) {
  current.kind = "playing";
  hud.reset();
  canvas.style.visibility = "";
  message.hidden = true;
  setStatus("WebGPU active", "running");
}

function pollLevel(current: Live<T>) {
  if (current.level || !options.level) return;
  const game = current.game;
  const path = game.take_level_request();
  if (path === undefined) return;
  current.level = true;
  game.report_level(0, 0);
  void options.level(game, path, lifecycle.signal, (done, total) => game.report_level(done, total))
    .catch(error => {
      if (fatal(error)) fail(error);
      else if (live() === current) game.fail_level(String(error));
    })
    .finally(() => { current.level = false; });
}

function frame(timestamp: number) {
  request = 0;
  const current = live();
  if (!current || current.paused || document.hidden) return;
  if (panicReport()) { fail(engineFailure({ code: "runtime_panicked", path: "$" })); return; }
  const game = current.game;
  try {
    if (current.kind === "playing") options.beforeFrame?.(game, timestamp);
    input.pollPads();
    const phase = game.animation_frame(timestamp, pixelWidth, pixelHeight);
    input.setCursorMode(game.cursor_mode());
    jobs?.exchange();
    if (phase === FRAME_PHASES.started) {
      started(current);
    } else if (phase === FRAME_PHASES.playing) {
      pollLevel(current);
      options.afterFrame?.(game, timestamp);
      if (hud.frame(game, timestamp)) options.poll?.(game);
      options.afterHostFrame?.(game, timestamp);
    } else if (phase === FRAME_PHASES.idle) {
      hud.reset();
      return;
    }
    if (live()) request = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}

function resume() {
  const current = live();
  if (!current || document.hidden) return;
  if (current.paused) { setStatus("Paused", "paused"); return; }
  if (request) return;
  if (current.kind === "preparing") {
    message.hidden = false;
    message.textContent = "Loading…";
    setStatus("Loading", "preparing");
  } else {
    setStatus("WebGPU active", "running");
  }
  request = requestAnimationFrame(frame);
}

window.addEventListener("blur", () => { interrupt("focus_lost"); pauseGame("focus"); }, listenerOptions);
window.addEventListener("focus", () => resumeGame("focus"), listenerOptions);
canvas.addEventListener("blur", () => interrupt("focus_lost"), listenerOptions);
window.addEventListener("error", event => { if (fatal(event.error)) fail(event.error); }, listenerOptions);
window.addEventListener("unhandledrejection", event => { if (fatal(event.reason)) fail(event.reason); }, listenerOptions);
document.addEventListener("visibilitychange", () => {
  if (document.hidden) {
    interrupt("suspension");
    halt();
    pauseGame("hidden");
    if (state.kind !== "failed") setStatus("Suspended", "suspended");
    return;
  }
  resumeGame("hidden");
  if (state.kind !== "failed") resume();
}, listenerOptions);
function setPaused(value: boolean) {
  const current = live();
  if (!current || current.paused === value) return;
  current.paused = value;
  pause.textContent = value ? "Resume" : "Pause";
  pause.setAttribute("aria-pressed", String(value));
  stopClock();
  if (value) setStatus("Paused", "paused");
  else resume();
}
pause.addEventListener("click", () => {
  const current = live();
  if (current && !current.busy) setPaused(!current.paused);
}, listenerOptions);
reset.addEventListener("click", () => {
  const current = live();
  if (current && !current.busy) guarded(() => current.game.reset_camera());
}, listenerOptions);
async function start() {
  if (disposed()) return;
  pause.disabled = true;
  reset.disabled = true;
  if (!("gpu" in navigator)) throw engineFailure({ code: "runtime_unavailable", path: "$", expected: "gpu" });
  const created = await options.create(canvas, lifecycle.signal);
  if (disposed()) { created.free(); return; }
  state = { kind: prepares ? "preparing" : "playing", game: created, paused: development && !preparedDevelopment, busy: false, level: false };
  created.apply_host_settings(location.search);
  created.set_preparation(prepares);
  connectInput();
  jobs = jobPool();
  jobs?.attach(created, fail);
  resize();
  message.hidden = true;
  pause.disabled = false;
  reset.disabled = false;
  options.ready?.(created, {
    mutate(action) {
      const current = live();
      if (!current || current.busy) throw engineFailure({ code: "runtime_busy", path: "$" });
      const game = current.game;
      stopClock();
      try {
        action(game);
        game.suspend();
        game.frame(performance.now(), pixelWidth, pixelHeight);
        game.suspend();
        options.poll?.(game);
      } catch (error) {
        if (fatal(error)) fail(error);
        throw error;
      } finally { resume(); }
    },
    setResolutionScale(scale) {
      if (!Number.isFinite(scale) || scale < 0.25 || scale > 2) {
        throw engineFailure({ code: "out_of_range", path: "$.scale", allowed: { kind: "range", min: 0.25, max: 2 }, got: scale });
      }
      live()?.game.set_setting("resolution.scale", JSON.stringify(scale));
    },
    setInputEnabled(enabled) {
      inputEnabled = enabled; clearInput();
      live()?.game.clear_input();
      connectInput();
    },
    distribution: values => {
      const current = live();
      return current ? readDistribution(current.game, values) : null;
    },
    size: () => [pixelWidth, pixelHeight], paused: () => live()?.paused ?? false,
    busy: () => live()?.busy ?? true,
  });
  if (development) {
    const { connectDevelopment } = await import("./dev");
    if (disposed()) return;
    disposeDevelopment = connectDevelopment({
      game: created, signal: lifecycle.signal, paused: () => live()?.paused ?? false,
      setPaused, exclusive,
      size: () => [pixelWidth, pixelHeight],
      poll: () => {
        input.setCursorMode(created.cursor_mode());
        options.poll?.(created);
      },
      frame(timestamp) {
        created.frame(timestamp, pixelWidth, pixelHeight);
        jobs?.exchange();
      },
      source: options.source,
      gpuWorkDone: () => (canvas.getContext("webgpu") as GPUCanvasContext | null)
        ?.getConfiguration()?.device.queue.onSubmittedWorkDone(),
      fail, stopped: () => state.kind === "failed" ? state.stopped : undefined,
      controls: Object.fromEntries(Object.entries(options.controls ?? {}).map(([name, control]) => [name, {
        values: control.values, run: (args: Record<string, unknown>) => control.run(created, args),
      }])),
      parameters: { ...HOST_PARAMETERS, ...options.parameters },
    });
    pause.textContent = "Resume";
    pause.setAttribute("aria-pressed", "true");
    await window.__engineDev!.request({ verb: "control", name: "render" });
  }
  resume();
}

window.addEventListener("pagehide", () => { interrupt("suspension"); halt(); pauseGame("suspended"); }, listenerOptions);
window.addEventListener("pageshow", () => { resumeGame("suspended"); resume(); }, listenerOptions);
if (import.meta.hot) import.meta.hot.dispose(() => {
  const current = live();
  jobs?.detach();
  disposeDevelopment?.();
  lifecycle.abort();
  stopClock(); observer.disconnect(); input.dispose();
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
      await (exclusive(reloadSource) ?? new Promise(resolve => setTimeout(resolve, 100)));
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
