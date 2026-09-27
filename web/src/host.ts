import { BrowserInput } from "./input";
import "./style.css";

export interface RuntimeStats {
  simulation_ms: number; extraction_ms: number; render_ms: number;
  instances: number; draw_calls: number; upload_bytes: number;
  invalid_transforms: number;
  render_candidates: number; visible_instances: number; culled_instances: number;
  dropped_instances: number; missing_assets: number; culling_enabled: boolean;
  fixed_steps: number; dropped_seconds: number;
  resident_meshes: number; mesh_upload_bytes: number; resident_mesh_bytes: number;
  resident_textures: number; texture_upload_bytes: number; resident_texture_bytes: number;
  shadows_enabled: boolean; shadow_draws: number; shadow_kept_redraw: number;
  shadow_near_casters: number | undefined; shadow_moving_casters: number | undefined; shadow_kept_casters: number | undefined;
  ambient_occlusion: boolean; bloom: boolean;
  local_lights: number; dropped_lights: number;
  lights_visible: number | undefined; light_cluster_overflow: number | undefined;
  sun_shafts: boolean; reflections: boolean; transparent_draws: number; particles: number; dropped_particles: number;
  free(): void;
}
export interface Runtime {
  frame(timestamp: number, width: number, height: number, forward: number, right: number, up: number, yaw: number, pitch: number, held: number, pressed: number, released: number): void;
  stats(): RuntimeStats;
  suspend(): void;
  reset_camera(): void;
  set_count?(count: number): void;
  dev_command?(request: string, width: number, height: number): string;
  dev_reload?(source: string, expectedRevision: string, revision: string): string;
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
export function boot<T extends Runtime>(options: {
  create(canvas: HTMLCanvasElement, count: number, signal: AbortSignal): Promise<T>;
  poll?(game: T): void;
  reload?(game: T, signal: AbortSignal): Promise<void>;
  source?: { url: string; load(signal: AbortSignal): Promise<string> };
  ready?(game: T, controls: HostControls<T>): void;
  beforeFrame?(game: T, timestamp: number): void;
  afterFrame?(game: T, timestamp: number): void;
  afterHostFrame?(game: T, timestamp: number): void;
  interrupted?(reason: string): void;
}) {

const lifecycle = new AbortController();
const listenerOptions = { signal: lifecycle.signal };
let disposed = false;

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`Missing element #${id}`);
  return found as T;
}

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
let developmentBusy = false;
let disposeDevelopment: (() => void) | undefined;
let failed = false;
let request = 0;
let previousFrame: number | undefined;
let lastStats = 0;
let intervals = 0;
let elapsed = 0;
let frameCount = 0;
let simulation = 0;
let extraction = 0;
let submission = 0;
let samples = 0;
let droppedTime = 0;
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
  previousFrame = undefined;
  intervals = 0;
  elapsed = 0;
  simulation = 0;
  extraction = 0;
  submission = 0;
  samples = 0;
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

function metric(id: string, value: string, unit = "") {
  const node = element(id);
  node.replaceChildren(document.createTextNode(value));
  if (unit) {
    const small = document.createElement("small");
    small.textContent = ` ${unit}`;
    node.append(small);
  }
}

function frame(timestamp: number) {
  request = 0;
  if (disposed || !game || paused || document.hidden || failed) return;
  try {
    options.beforeFrame?.(game, timestamp);
    game.frame(timestamp, pixelWidth, pixelHeight, inputEnabled ? input.forward : 0, inputEnabled ? input.right : 0, inputEnabled ? input.upAxis : 0,
      inputEnabled ? input.yaw : 0, inputEnabled ? input.pitch : 0, inputEnabled ? input.held : 0, inputEnabled ? input.pressed : 0, inputEnabled ? input.released : 0);
    input.consumed();
    options.afterFrame?.(game, timestamp);
    frameCount++;
    if (previousFrame !== undefined) { elapsed += timestamp - previousFrame; intervals++; }
    previousFrame = timestamp;
    // Poll aggregate observations only four times a second, outside entity loops.
    if (timestamp - lastStats >= 250) {
      const stats = game.stats();
      try {
        simulation += stats.simulation_ms;
        extraction += stats.extraction_ms;
        submission += stats.render_ms;
        samples++;
        droppedTime = stats.dropped_seconds;
        metric("fps", elapsed > 0 ? (intervals * 1000 / elapsed).toFixed(0) : "—", "FPS");
        metric("simulation", (simulation / samples).toFixed(2), "ms");
        metric("extraction", (extraction / samples).toFixed(2), "ms");
        metric("draws", String(stats.draw_calls));
        element("uploads").textContent = `${(stats.upload_bytes / 1024).toFixed(0)} KiB / frame`;
        element("render-time").textContent = `${stats.instances} drawn · ${stats.culled_instances} culled · CPU render ${(submission / samples).toFixed(2)} ms · GPU time not measured`;
        canvas.dataset.frames = String(frameCount);
        canvas.dataset.instances = String(stats.instances);
        canvas.dataset.renderCandidates = String(stats.render_candidates);
        canvas.dataset.visibleInstances = String(stats.visible_instances);
        canvas.dataset.culledInstances = String(stats.culled_instances);
        canvas.dataset.cullingEnabled = String(stats.culling_enabled);
        canvas.dataset.fixedSteps = String(stats.fixed_steps);
        canvas.dataset.droppedSeconds = String(droppedTime);
        canvas.dataset.residentMeshes = String(stats.resident_meshes);
        canvas.dataset.meshUploadBytes = String(stats.mesh_upload_bytes);
        canvas.dataset.residentMeshBytes = String(stats.resident_mesh_bytes);
        canvas.dataset.residentTextures = String(stats.resident_textures);
        canvas.dataset.textureUploadBytes = String(stats.texture_upload_bytes);
        canvas.dataset.residentTextureBytes = String(stats.resident_texture_bytes);
        canvas.dataset.shadowsEnabled = String(stats.shadows_enabled);
        canvas.dataset.shadowDraws = String(stats.shadow_draws);
        canvas.dataset.ambientOcclusion = String(stats.ambient_occlusion);
        canvas.dataset.bloom = String(stats.bloom);
        options.poll?.(game);
      } finally { stats.free(); }
      lastStats = timestamp;
      // Bounded windows keep the HUD responsive to workload changes.
      if (samples >= 4) { simulation = 0; extraction = 0; submission = 0; samples = 0; elapsed = 0; intervals = 0; }
    }
    options.afterHostFrame?.(game, timestamp);
    if (!disposed) request = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}

function resume() {
  if (disposed) return;
  if (game && paused && !document.hidden && !failed) setStatus("Paused", "paused");
  if (game && !paused && !document.hidden && !failed && !request) {
    setStatus("WebGPU active", "running");
    request = requestAnimationFrame(frame);
  }
}

window.addEventListener("blur", () => { options.interrupted?.("focus_lost"); clearInput(); game?.suspend(); previousFrame = undefined; }, listenerOptions);
canvas.addEventListener("blur", () => { options.interrupted?.("focus_lost"); game?.suspend(); previousFrame = undefined; }, listenerOptions);
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
    current.suspend(); clearInput(); previousFrame = undefined;
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
