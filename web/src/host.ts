import { BrowserInput } from "./input";
import "./style.css";

/**
 * The engine's frame statistics (Rust `FrameStats` with the renderer's
 * `RenderStats`), parsed from the JSON `Runtime.stats` returns. Only the fields
 * this host code reads are typed; agents receive every field.
 */
export interface RuntimeStats {
  simulation_ms: number; extraction_ms: number; render_ms: number; dropped_seconds: number;
  instances: number; draw_calls: number; upload_bytes: number; invalid_transforms: number; fixed_steps: number;
  render_candidates: number; visible_instances: number; culled_instances: number; dropped_instances: number;
  missing_assets: number; culling_enabled: boolean; triangles: number | null;
  resident_meshes: number; mesh_upload_bytes: number; resident_mesh_bytes: number;
  resident_textures: number; texture_upload_bytes: number; resident_texture_bytes: number;
  pipelines_created: number; pipelines_compiling: boolean;
  [field: string]: unknown;
}
/** The last frame's statistics of `runtime`. */
export function readStats(runtime: { stats(): string }): RuntimeStats {
  return JSON.parse(runtime.stats()) as RuntimeStats;
}
export interface Runtime {
  /** Games without action buttons (Orbit, the Stress Lab) declare only the axes; JavaScript drops the rest. */
  frame(timestamp: number, width: number, height: number, forward: number, right: number, up: number, yaw: number, pitch: number, held: number, pressed: number, released: number): void;
  stats(): string;
  suspend(): void;
  reset_camera(): void;
  set_count?(count: number): void;
  dev_command?(request: string, width: number, height: number): string;
  dev_reload?(source: string, expectedRevision: string, revision: string, files?: unknown): string;
  profiling_start?(maxEvents: number): void;
  profiling_stop?(): string;
  profiling_status?(): string;
  profiling_poll?(): void;
  profiling_report?(): string;
  free(): void;
}
/** Optional controls for example-specific panels; mutations run between frames. */
export interface HostControls<T extends Runtime> {
  mutate(action: (game: T) => void): void;
  setResolutionScale(scale: number): void;
  setInputEnabled(enabled: boolean): void;
  size(): [number, number];
  paused(): boolean;
  busy(): boolean;
}
function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element #${id}`);
  return found as T;
}

/** Shows `value` with `unit` in metric `id`. */
function metric(id: string, value: string, unit = "") {
  const node = element(id);
  node.replaceChildren(document.createTextNode(value));
  if (unit) {
    const small = document.createElement("small");
    small.textContent = ` ${unit}`;
    node.append(small);
  }
}

/**
 * The page's frame rate and CPU timing readouts. Statistics are polled four
 * times a second, outside entity loops, and averaged over bounded windows so
 * the readouts follow workload changes.
 */
class FrameHud {
  private frames = 0;
  private previous: number | undefined;
  private lastSample = 0;
  private intervals = 0;
  private elapsed = 0;
  private simulation = 0;
  private extraction = 0;
  private submission = 0;
  private samples = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {}

  /** Leaves the time until the next frame out of the frame rate. */
  skipInterval() { this.previous = undefined; }

  /** Restarts every average, e.g. after a suspension. */
  reset() {
    this.previous = undefined;
    this.intervals = 0; this.elapsed = 0;
    this.simulation = 0; this.extraction = 0; this.submission = 0; this.samples = 0;
  }

  /** Counts a frame of `runtime` at `timestamp`; returns whether it sampled
   * the statistics, four times a second. */
  frame(runtime: Runtime, timestamp: number): boolean {
    this.frames++;
    if (this.previous !== undefined) { this.elapsed += timestamp - this.previous; this.intervals++; }
    this.previous = timestamp;
    if (timestamp - this.lastSample < 250) return false;
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
    element("render-time").textContent = `${stats.instances} drawn${triangles} · ${stats.culled_instances} culled · CPU render ${(this.submission / this.samples).toFixed(2)} ms · GPU time not measured`;
    const dataset = this.canvas.dataset;
    dataset.frames = String(this.frames);
    dataset.instances = String(stats.instances);
    dataset.renderCandidates = String(stats.render_candidates);
    dataset.residentMeshes = String(stats.resident_meshes);
    dataset.meshUploadBytes = String(stats.mesh_upload_bytes);
    dataset.residentMeshBytes = String(stats.resident_mesh_bytes);
    dataset.residentTextures = String(stats.resident_textures);
    dataset.textureUploadBytes = String(stats.texture_upload_bytes);
    this.lastSample = timestamp;
    if (this.samples >= 4) {
      this.simulation = 0; this.extraction = 0; this.submission = 0; this.samples = 0; this.elapsed = 0; this.intervals = 0;
    }
    return true;
  }
}

export function boot<T extends Runtime>(options: {
  create(canvas: HTMLCanvasElement, count: number, signal: AbortSignal): Promise<T>;
  poll?(game: T): void;
  reload?(game: T, signal: AbortSignal): Promise<void>;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files?(source: string, signal: AbortSignal): Promise<unknown> };
  ready?(game: T, controls: HostControls<T>): void;
  beforeFrame?(game: T, timestamp: number): void;
  afterFrame?(game: T, timestamp: number): void;
  afterHostFrame?(game: T, timestamp: number): void;
  interrupted?(reason: string): void;
}) {

const lifecycle = new AbortController();
const listenerOptions = { signal: lifecycle.signal };
let disposed = false;

const canvas = element<HTMLCanvasElement>("scene");
const status = element("status");
const message = element("message");
const count = document.getElementById("count") as HTMLSelectElement | null;
const pause = element<HTMLButtonElement>("pause");
const reset = element<HTMLButtonElement>("reset");
const input = new BrowserInput(canvas);
let game: T | undefined;
const instrumented = import.meta.env.MODE === "profiling" || import.meta.env.MODE === "attribution";
const development = (import.meta.env.DEV || instrumented) && new URLSearchParams(location.search).get("engine_dev") === "1";
let paused = development;
// Pipelines compile when a frame first needs them (decision 049). Ordinary
// sessions first render their starting state without advancing time behind a
// hidden canvas (Firefox paints a visible WebGPU canvas by waiting on its GPU
// process), until rendered frames stop creating pipelines and the GPU has
// compiled them; then play starts. Development sessions start paused and
// compile with the frames their commands render.
let preparing = !development;
// Hidden from the start: creating the renderer already compiles pipelines.
if (preparing) canvas.style.visibility = "hidden";
let preparedSince: number | undefined;
let quietFrames = 0;
const QUIET_FRAMES = 3;
const PREPARE_LIMIT_MS = 60000;
let developmentBusy = false;
let disposeDevelopment: (() => void) | undefined;
let failed = false;
let request = 0;
const hud = new FrameHud(canvas);
let pixelWidth = 1;
let pixelHeight = 1;
let resolutionScale = 1;
let inputEnabled = true;

function setStatus(text: string, state: string) {
  status.replaceChildren(document.createElement("i"), document.createTextNode(text));
  status.dataset.state = state;
}

function clearInput() { input.clear(); }

function resize() {
  if (disposed) return;
  const bounds = canvas.getBoundingClientRect();
  if (bounds.width <= 0 || bounds.height <= 0) { options.interrupted?.("resize"); pixelWidth = 0; pixelHeight = 0; return; }
  const dpr = Math.min(window.devicePixelRatio || 1, 1.5) * resolutionScale;
  // A resolution budget controls fill cost independently of entity count.
  const scale = Math.min(dpr, 2560 / Math.max(bounds.width, 1), 1440 / Math.max(bounds.height, 1));
  const width = Math.max(1, Math.round(bounds.width * scale));
  const height = Math.max(1, Math.round(bounds.height * scale));
  if (width !== pixelWidth || height !== pixelHeight) options.interrupted?.("resize");
  pixelWidth = width;
  pixelHeight = height;
}
const observer = new ResizeObserver(resize);
observer.observe(canvas);
window.addEventListener("resize", resize, listenerOptions);

function stopClock() {
  options.interrupted?.("suspension");
  cancelAnimationFrame(request);
  request = 0;
  hud.reset();
  clearInput();
  game?.suspend();
}

function fail(error: unknown) {
  if (disposed) return;
  failed = true;
  stopClock();
  setStatus("Renderer unavailable", "error");
  message.hidden = false;
  message.textContent = `Unable to start or continue the engine.\n${String(error)}\nWebGPU requires a supported browser on localhost or HTTPS.`;
  pause.disabled = true;
  if (count) count.disabled = true;
  reset.disabled = true;
  console.error(error);
}

function prepareFrame(current: T, timestamp: number) {
  preparedSince ??= performance.now();
  const late = performance.now() - preparedSince > PREPARE_LIMIT_MS;
  const compiling = readStats(current).pipelines_compiling;
  // Frames rendered while compiles run would only queue behind them.
  if (compiling && !late) return;
  current.suspend();
  current.frame(timestamp, pixelWidth, pixelHeight, 0, 0, 0, 0, 0, 0, 0, 0);
  const created = readStats(current).pipelines_created;
  quietFrames = created === 0 ? quietFrames + 1 : 0;
  if (quietFrames < QUIET_FRAMES && !late) return;
  preparing = false;
  current.suspend();
  hud.skipInterval();
  canvas.style.visibility = "";
  message.hidden = true;
  setStatus("WebGPU active", "running");
}

function frame(timestamp: number) {
  request = 0;
  if (disposed || !game || paused || document.hidden || failed) return;
  try {
    if (preparing) {
      prepareFrame(game, timestamp);
      if (!disposed) request = requestAnimationFrame(frame);
      return;
    }
    options.beforeFrame?.(game, timestamp);
    game.frame(timestamp, pixelWidth, pixelHeight, inputEnabled ? input.forward : 0, inputEnabled ? input.right : 0, inputEnabled ? input.upAxis : 0,
      inputEnabled ? input.yaw : 0, inputEnabled ? input.pitch : 0, inputEnabled ? input.held : 0, inputEnabled ? input.pressed : 0, inputEnabled ? input.released : 0);
    input.consumed();
    options.afterFrame?.(game, timestamp);
    if (hud.frame(game, timestamp)) options.poll?.(game);
    options.afterHostFrame?.(game, timestamp);
    if (!disposed) request = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}

function resume() {
  if (disposed) return;
  if (game && paused && !document.hidden && !failed) setStatus("Paused", "paused");
  if (game && !paused && !document.hidden && !failed && !request) {
    if (preparing) {
      message.hidden = false;
      message.textContent = "Compiling shaders…";
      setStatus("Compiling shaders", "preparing");
    } else {
      setStatus("WebGPU active", "running");
    }
    request = requestAnimationFrame(frame);
  }
}

window.addEventListener("blur", () => { options.interrupted?.("focus_lost"); clearInput(); game?.suspend(); hud.skipInterval(); }, listenerOptions);
canvas.addEventListener("blur", () => { options.interrupted?.("focus_lost"); game?.suspend(); hud.skipInterval(); }, listenerOptions);
document.addEventListener("visibilitychange", () => {
  stopClock();
  if (failed) return;
  if (document.hidden) setStatus("Suspended", "suspended");
  else resume();
}, listenerOptions);
function setPaused(value: boolean) {
  if (paused === value) return;
  paused = value;
  pause.textContent = paused ? "Resume" : "Pause";
  pause.setAttribute("aria-pressed", String(paused));
  stopClock();
  if (paused) setStatus("Paused", "paused");
  else resume();
}
pause.addEventListener("click", () => {
  if (!developmentBusy) setPaused(!paused);
}, listenerOptions);
count?.addEventListener("change", () => {
  if (developmentBusy) return;
  try { game?.set_count?.(Number(count.value)); } catch (error) { fail(error); }
}, listenerOptions);
reset.addEventListener("click", () => { if (!developmentBusy) game?.reset_camera(); }, listenerOptions);
const reload = document.getElementById("reload") as HTMLButtonElement | null;
reload?.addEventListener("click", async () => {
  if (disposed || failed || developmentBusy || reload.disabled || !game || !options.reload) return;
  reload.disabled = true;
  developmentBusy = true;
  const notice = element("notice");
  const current = game;
  try {
    await options.reload(current, lifecycle.signal);
    if (disposed || game !== current) return;
    current.suspend(); clearInput(); hud.skipInterval();
    notice.textContent = "Scene reloaded from its source file.";
  } catch (error) {
    if (!disposed) notice.textContent = `Reload failed; the current scene is still running. ${String(error)}`;
  } finally { developmentBusy = false; if (!disposed) reload.disabled = false; }
}, listenerOptions);

async function start() {
  if (disposed) return;
  pause.disabled = true;
  if (count) count.disabled = true;
  reset.disabled = true;
  if (!("gpu" in navigator)) throw new Error("WebGPU is not available in this browser or execution context.");
  const created = await options.create(canvas, Number(count?.value ?? 10000), lifecycle.signal);
  // GPU initialization may finish after teardown even when fetch was aborted.
  if (disposed) { created.free(); return; }
  game = created;
  resize();
  message.hidden = true;
  pause.disabled = false;
  if (count) count.disabled = false;
  reset.disabled = false;
  options.ready?.(created, {
    mutate(action) {
      if (disposed || failed || developmentBusy || !game) throw new Error("The runtime is busy or unavailable.");
      stopClock();
      try {
        action(game);
        game.suspend();
        // Reset timing before an ordinary zero-delta frame: this also refreshes
        // the image in production while the host is paused.
        game.frame(performance.now(), pixelWidth, pixelHeight, 0, 0, 0, 0, 0, 0, 0, 0);
        game.suspend();
        options.poll?.(game);
      } finally { resume(); }
    },
    setResolutionScale(scale) {
      if (!Number.isFinite(scale) || scale < 0.25 || scale > 2) throw new Error("Resolution scale must be 0.25–2.");
      resolutionScale = scale; resize();
    },
    setInputEnabled(enabled) { inputEnabled = enabled; clearInput(); },
    size: () => [pixelWidth, pixelHeight], paused: () => paused,
    busy: () => disposed || failed || developmentBusy,
  });
  if (development) {
    const { connectDevelopment } = await import("./dev");
    if (disposed) return;
    disposeDevelopment = connectDevelopment({
      game: created, signal: lifecycle.signal, paused: () => paused,
      setPaused, setBusy: value => { developmentBusy = value; },
      isBusy: () => developmentBusy,
      size: () => [pixelWidth, pixelHeight], poll: () => options.poll?.(created),
      source: options.source,
    });
    pause.textContent = "Resume";
    pause.setAttribute("aria-pressed", "true");
    await window.__engineDev!.request({ op: "render" });
  }
  resume();
}

window.addEventListener("pagehide", () => { stopClock(); }, listenerOptions);
window.addEventListener("pageshow", () => resume(), listenerOptions);
if (import.meta.hot) import.meta.hot.dispose(() => {
  disposed = true;
  disposeDevelopment?.();
  lifecycle.abort();
  stopClock(); observer.disconnect(); input.dispose(); game?.free(); game = undefined;
});
void start().catch(fail);
}
