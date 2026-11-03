import { type Distribution, FRAME_PHASES } from "../generated/host";
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
  private previous: number | undefined;
  private lastSample = 0;
  private intervals = 0;
  private elapsed = 0;
  private simulation = 0;
  private extraction = 0;
  private submission = 0;
  private samples = 0;

  constructor(private readonly canvas: HTMLCanvasElement) {}

  skipInterval() { this.previous = undefined; }

  reset() {
    this.previous = undefined;
    this.intervals = 0; this.elapsed = 0;
    this.simulation = 0; this.extraction = 0; this.submission = 0; this.samples = 0;
  }

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
    this.lastSample = timestamp;
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
  reload?(game: T, signal: AbortSignal): Promise<void>;
  level?(game: T, path: string, signal: AbortSignal, progress: (done: number, total: number) => void): Promise<void>;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files?(source: string, signal: AbortSignal): Promise<unknown> };
  ready?(game: T, controls: HostControls<T>): void;
  beforeFrame?(game: T, timestamp: number): void;
  afterFrame?(game: T, timestamp: number): void;
  afterHostFrame?(game: T, timestamp: number): void;
  interrupted?(reason: string): void;
}

export function startGame<T extends Runtime>(options: GameOptions<T>) {
ensurePage();

const lifecycle = new AbortController();
const listenerOptions = { signal: lifecycle.signal };
let disposed = false;

const canvas = element<HTMLCanvasElement>("scene");
const status = element("status");
const message = element("message");
const pause = element<HTMLButtonElement>("pause");
const reset = element<HTMLButtonElement>("reset");
const input = new BrowserInput(canvas);
let game: T | undefined;
const instrumented = import.meta.env.MODE === "profiling" || import.meta.env.MODE === "attribution";
const development = (import.meta.env.DEV || instrumented) && new URLSearchParams(location.search).get("engine_dev") === "1";
const preparedDevelopment = development && new URLSearchParams(location.search).get("prepare") === "1";
let paused = development && !preparedDevelopment;
let preparing = !development || preparedDevelopment;
if (preparing) canvas.style.visibility = "hidden";
let developmentBusy = false;
let loadingLevel = false;
let disposeDevelopment: (() => void) | undefined;
let failed = false;
let stopped: string | undefined;
let request = 0;
let working = 0;
let between = 0;
const hud = new FrameHud(canvas);
let jobs: JobPool | undefined;
let pixelWidth = 1;
let pixelHeight = 1;
let inputEnabled = true;

function setStatus(text: string, state: string) {
  status.replaceChildren(document.createElement("i"), document.createTextNode(text));
  status.dataset.state = state;
}

function clearInput() { input.clear(); }

function connectInput() {
  input.game = inputEnabled && !failed ? game : undefined;
}

function interrupt(reason: string) {
  if (!failed) options.interrupted?.(reason);
}

function suspendGame() {
  if (!failed) game?.suspend();
}

function resize() {
  if (disposed || failed || !game) return;
  const bounds = canvas.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  const width = Math.round(bounds.width * ratio);
  const height = Math.round(bounds.height * ratio);
  const idle = pixelWidth === 0 || pixelHeight === 0;
  if (width !== pixelWidth || height !== pixelHeight) interrupt("resize");
  pixelWidth = width;
  pixelHeight = height;
  game.set_pixel_ratio(ratio);
  if (idle && width > 0 && height > 0) resume();
}
const observer = new ResizeObserver(resize);
observer.observe(canvas);
window.addEventListener("resize", resize, listenerOptions);

function halt() {
  cancelAnimationFrame(request);
  request = 0;
  clearTimeout(working);
  working = 0;
  clearTimeout(between);
  between = 0;
  hud.reset();
  clearInput();
}

function stopClock() {
  interrupt("suspension");
  halt();
  suspendGame();
}

function fail(error: unknown) {
  if (disposed || failed) return;
  failed = true;
  connectInput();
  stopped = fatal(error) ? panicReport(error)?.message ?? String(error) : undefined;
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

function workBetweenFrames() {
  between = 0;
  if (disposed || !game || paused || document.hidden || failed || preparing) return;
  const current = game;
  guarded(() => {
    current.load_between_frames();
    jobs?.exchange();
  });
}

function workWhileCompiling() {
  working = 0;
  if (disposed || !game || paused || document.hidden || failed || !preparing) return;
  const current = game;
  guarded(() => {
    const waiting = current.work_while_compiling(pixelWidth, pixelHeight);
    jobs?.exchange();
    if (waiting) working = window.setTimeout(workWhileCompiling, 0);
  });
}

function started() {
  preparing = false;
  hud.skipInterval();
  canvas.style.visibility = "";
  message.hidden = true;
  setStatus("WebGPU active", "running");
}

function pollLevel(current: T) {
  if (loadingLevel || !options.level) return;
  const path = current.take_level_request();
  if (path === undefined) return;
  loadingLevel = true;
  current.report_level(0, 0);
  void options.level(current, path, lifecycle.signal, (done, total) => current.report_level(done, total))
    .catch(error => {
      if (fatal(error)) fail(error);
      else if (!disposed && !failed) current.fail_level(String(error));
    })
    .finally(() => { loadingLevel = false; });
}

function frame(timestamp: number) {
  request = 0;
  if (disposed || !game || paused || document.hidden || failed) return;
  if (panicReport()) { fail(engineFailure({ code: "runtime_panicked", path: "$" })); return; }
  try {
    if (!preparing) options.beforeFrame?.(game, timestamp);
    input.pollPads();
    const phase = game.animation_frame(timestamp, pixelWidth, pixelHeight);
    input.setCursorMode(game.cursor_mode());
    jobs?.exchange();
    if (phase === FRAME_PHASES.compiling) {
      if (!working) working = window.setTimeout(workWhileCompiling, 0);
    } else if (phase === FRAME_PHASES.started) {
      started();
    } else if (phase === FRAME_PHASES.playing) {
      pollLevel(game);
      options.afterFrame?.(game, timestamp);
      if (hud.frame(game, timestamp)) options.poll?.(game);
      options.afterHostFrame?.(game, timestamp);
      if (!between) between = window.setTimeout(workBetweenFrames, 0);
    } else if (phase === FRAME_PHASES.idle) {
      hud.skipInterval();
      return;
    }
    if (!disposed) request = requestAnimationFrame(frame);
  } catch (error) { fail(error); }
}

function resume() {
  if (disposed) return;
  if (game && paused && !document.hidden && !failed) setStatus("Paused", "paused");
  if (game && !paused && !document.hidden && !failed && !request) {
    if (preparing) {
      message.hidden = false;
      message.textContent = "Loading…";
      setStatus("Loading", "preparing");
    } else {
      setStatus("WebGPU active", "running");
    }
    request = requestAnimationFrame(frame);
  }
}

window.addEventListener("blur", () => { interrupt("focus_lost"); clearInput(); suspendGame(); hud.skipInterval(); }, listenerOptions);
canvas.addEventListener("blur", () => { interrupt("focus_lost"); suspendGame(); hud.skipInterval(); }, listenerOptions);
window.addEventListener("error", event => { if (fatal(event.error)) fail(event.error); }, listenerOptions);
window.addEventListener("unhandledrejection", event => { if (fatal(event.reason)) fail(event.reason); }, listenerOptions);
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
reset.addEventListener("click", () => {
  if (!developmentBusy && !failed && game) guarded(() => game?.reset_camera());
}, listenerOptions);
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
    if (fatal(error)) fail(error);
    else if (!disposed) notice.textContent = `Reload failed; the current scene is still running. ${String(error)}`;
  } finally { developmentBusy = false; if (!disposed) reload.disabled = false; }
}, listenerOptions);

async function start() {
  if (disposed) return;
  pause.disabled = true;
  reset.disabled = true;
  if (!("gpu" in navigator)) throw engineFailure({ code: "runtime_unavailable", path: "$", expected: "gpu" });
  const created = await options.create(canvas, lifecycle.signal);
  if (disposed) { created.free(); return; }
  game = created;
  created.set_setting("dynamic_resolution.enabled", "true");
  created.apply_host_settings(location.search);
  created.set_preparation(preparing);
  connectInput();
  jobs = jobPool();
  jobs?.attach(created, fail);
  resize();
  message.hidden = true;
  pause.disabled = false;
  reset.disabled = false;
  options.ready?.(created, {
    mutate(action) {
      if (disposed || failed || developmentBusy || !game) throw engineFailure({ code: "runtime_busy", path: "$" });
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
      if (!failed) game?.set_setting("resolution.scale", JSON.stringify(scale));
    },
    setInputEnabled(enabled) {
      inputEnabled = enabled; clearInput();
      if (!failed) game?.clear_input();
      connectInput();
    },
    distribution: values => disposed || failed || !game ? null : readDistribution(game, values),
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
      frame(timestamp) {
        created.frame(timestamp, pixelWidth, pixelHeight);
        jobs?.exchange();
      },
      source: options.source,
      gpuWorkDone: () => (canvas.getContext("webgpu") as GPUCanvasContext | null)
        ?.getConfiguration()?.device.queue.onSubmittedWorkDone(),
      fail, stopped: () => stopped,
      controls: Object.fromEntries(Object.entries(options.controls ?? {}).map(([name, control]) => [name, {
        values: control.values, run: (args: Record<string, unknown>) => control.run(created, args),
      }])),
      parameters: { ...HOST_PARAMETERS, ...options.parameters },
    });
    pause.textContent = "Resume";
    pause.setAttribute("aria-pressed", "true");
    await window.__engineDev!.request({ verb: "control", name: "render" });
    const environment = new URLSearchParams(location.search).get("environment");
    if (environment) {
      const playing = !paused;
      setPaused(true);
      const reply = await window.__engineDev!.request({
        verb: "control", name: "environment", version: window.__engineDev!.version, args: { merge: true, ...JSON.parse(environment) as Record<string, unknown> },
      });
      if (playing) setPaused(false);
      if (!reply.ok) {
        const refused = reply.error ?? { code: "invalid_environment", path: "$" };
        throw engineFailure({ ...refused, path: `?environment${refused.path.replace(/^\$(\.args)?/, "")}` });
      }
    }
  }
  resume();
}

window.addEventListener("pagehide", () => { stopClock(); }, listenerOptions);
window.addEventListener("pageshow", () => resume(), listenerOptions);
if (import.meta.hot) import.meta.hot.dispose(() => {
  disposed = true;
  jobs?.detach();
  disposeDevelopment?.();
  lifecycle.abort();
  stopClock(); observer.disconnect(); input.dispose();
  if (!failed) game?.free();
  game = undefined;
});
void start().catch(fail);
}
