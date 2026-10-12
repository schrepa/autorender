import { type Distribution, FRAME_PHASES } from "../generated/host";
import { engineError } from "../generated/errors";
import { parseJson, type Runtime } from "../generated/runtime";
import type { FrameStats } from "../generated/stats";
import { BrowserInput } from "./input";
import { type JobPool, jobPool } from "./jobs";
import { fatal, panicReport } from "./panic";
import { HOST_PARAMETERS } from "./parameters";
import "./style.css";

export type { Distribution, FrameStats, Runtime };
/** The last frame's statistics of `runtime`, typed from Rust's
 * `FrameStats`. */
export function readStats(runtime: Pick<Runtime, "stats">): FrameStats {
  return parseJson(runtime.stats());
}
/** The spread of measured `values` as the engine computes it on every host
 * (`Runtime.distribution`, typed from Rust's `Distribution`): null without
 * values, or with one that is not a finite number of at least 0. */
export function readDistribution(runtime: Pick<Runtime, "distribution">, values: readonly number[]): Distribution | null {
  return parseJson(runtime.distribution(Float64Array.from(values)));
}
/** Controls for a game's own panels; mutations run between frames. */
export interface HostControls<T extends Runtime> {
  mutate(action: (game: T) => void): void;
  setResolutionScale(scale: number): void;
  setInputEnabled(enabled: boolean): void;
  /** The spread of measured `values` (`readDistribution`); null without
   * values or once the engine stopped. */
  distribution(values: readonly number[]): Distribution | null;
  size(): [number, number];
  paused(): boolean;
  busy(): boolean;
}
/** The elements the host fills, by ID. */
const PAGE_ELEMENTS = ["scene", "status", "message", "pause", "reset", "fps", "simulation", "extraction", "draws", "uploads", "render-time"];

/** A page without a canvas `#scene` gets the engine's own page; a page with
 * one must have every element the host fills. */
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
    throw new Error(`The page lacks elements the host fills: ${missing.map(id => `#${id}`).join(", ")}; leave out #scene to use the engine's own page.`);
  }
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

/** What a game's page starts it with. */
export interface GameOptions<T extends Runtime> {
  /** Creates the game on `canvas`. */
  create(canvas: HTMLCanvasElement, signal: AbortSignal): Promise<T>;
  /** The game's own development controls, which agents run with the
   * `game` control by name; each refreshes the image after. */
  controls?: Record<string, { description: string; run(game: T, args: Record<string, unknown>): unknown }>;
  /** The page parameters the game reads, with what each does. */
  parameters?: Record<string, string>;
  poll?(game: T): void;
  reload?(game: T, signal: AbortSignal): Promise<void>;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files?(source: string, signal: AbortSignal): Promise<unknown> };
  ready?(game: T, controls: HostControls<T>): void;
  beforeFrame?(game: T, timestamp: number): void;
  afterFrame?(game: T, timestamp: number): void;
  afterHostFrame?(game: T, timestamp: number): void;
  interrupted?(reason: string): void;
}

/** Runs the game `options.create` makes on the page: frames, input, job
 * workers, loading, the page's switches, statistics and, in development
 * builds with `?engine_dev=1`, the development tools. */
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
// ?prepare=1 starts a development session as ordinary sessions start, so
// agents can inspect loading as players see it.
const preparedDevelopment = development && new URLSearchParams(location.search).get("prepare") === "1";
let paused = development && !preparedDevelopment;
// Pipelines compile when a frame first needs them. Ordinary
// sessions first render their starting state without advancing time behind a
// hidden canvas (Firefox paints a visible WebGPU canvas by waiting on its GPU
// process), until rendered frames stop creating pipelines, the GPU has
// compiled them and terrains have loaded the camera's view;
// then play starts. Development sessions start paused and compile with the
// frames their commands render.
let preparing = !development || preparedDevelopment;
// Hidden from the start: creating the renderer already compiles pipelines.
if (preparing) canvas.style.visibility = "hidden";
let developmentBusy = false;
let disposeDevelopment: (() => void) | undefined;
// Once the engine failed, nothing calls into its module again; `stopped`
// says why when its module stopped (a panic's message, or the trap).
let failed = false;
let stopped: string | undefined;
let request = 0;
// Loading work while pipelines compile: browsers stop calling
// animation frames while their GPU process compiles, so it runs in tasks.
let working = 0;
// Terrain loading between frames: after a frame of play, a task loads
// terrain until shortly before the next frame is due, by the engine's pace
// of recent frames. Frames waiting on the GPU leave that time free; frames
// waiting on the CPU leave none.
let between = 0;
const hud = new FrameHud(canvas);
// Job workers, for games that hand out jobs.
let jobs: JobPool | undefined;
let pixelWidth = 1;
let pixelHeight = 1;
let resolutionScale = 1;
let inputEnabled = true;

function setStatus(text: string, state: string) {
  status.replaceChildren(document.createElement("i"), document.createTextNode(text));
  status.dataset.state = state;
}

function clearInput() { input.clear(); }

/** Sends the page's input to the game while it takes input, and the
 * module runs. */
function connectInput() {
  input.game = inputEnabled && !failed ? game : undefined;
}

/** Tells the game's page `reason` interrupted its timing, while the engine
 * runs: its handlers may call into the module. */
function interrupt(reason: string) {
  if (!failed) options.interrupted?.(reason);
}

/** Drops the game's suspension time and held input, while its module runs. */
function suspendGame() {
  if (!failed) game?.suspend();
}

/** Sizes the view as the engine's resolution policy does on every host;
 * frames need no size before the game exists, which sizes it at start. */
function resize() {
  if (disposed || failed || !game) return;
  const bounds = canvas.getBoundingClientRect();
  const [width = 0, height = 0] = game.view_size(bounds.width, bounds.height, window.devicePixelRatio, resolutionScale);
  const idle = pixelWidth === 0 || pixelHeight === 0;
  if (width !== pixelWidth || height !== pixelHeight) interrupt("resize");
  pixelWidth = width;
  pixelHeight = height;
  // A view that regains area asks for frames again.
  if (idle && width > 0 && height > 0) resume();
}
const observer = new ResizeObserver(resize);
observer.observe(canvas);
window.addEventListener("resize", resize, listenerOptions);

/** Stops frames and the loading tasks between them; calls nothing in the
 * module. */
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

/** Stops the page for good after `error`: the engine's panic or its
 * module's trap, or a failure to start or run a frame. The page says so
 * first; then frames, loading and the job workers stop, and nothing calls
 * into the module again. */
function fail(error: unknown) {
  if (disposed || failed) return;
  failed = true;
  // Nothing calls into the module again.
  connectInput();
  stopped = fatal(error) ? panicReport(error)?.message ?? String(error) : undefined;
  // The engine's own refusal (a page parameter it does not take, a scene it
  // cannot run) says what failed; anything else is likely WebGPU's.
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

/** Runs `work`, which calls into the module; a module that stopped stops
 * the page. */
function guarded(work: () => void) {
  try { work(); } catch (error) { fail(error); }
}

/** Terrain loading until shortly before the next frame of play is due. */
function workBetweenFrames() {
  between = 0;
  if (disposed || !game || paused || document.hidden || failed || preparing) return;
  const current = game;
  guarded(() => {
    current.load_between_frames();
    jobs?.exchange();
  });
}

/** Loading work in tasks while preparation waits on compiles. */
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

/** Play starts: the canvas shows what preparation drew. */
function started() {
  preparing = false;
  hud.skipInterval();
  canvas.style.visibility = "";
  message.hidden = true;
  setStatus("WebGPU active", "running");
}

/** The animation frame at `timestamp`: preparation's or one of play, as
 * the engine's frame loop runs them. */
function frame(timestamp: number) {
  request = 0;
  if (disposed || !game || paused || document.hidden || failed) return;
  // The module stopped where the page's own code caught its error.
  if (panicReport()) { fail(new Error("the engine panicked")); return; }
  try {
    if (!preparing) options.beforeFrame?.(game, timestamp);
    const phase = game.animation_frame(timestamp, pixelWidth, pixelHeight);
    jobs?.exchange();
    if (phase === FRAME_PHASES.compiling) {
      // Browsers stop animation frames while their GPU process compiles.
      if (!working) working = window.setTimeout(workWhileCompiling, 0);
    } else if (phase === FRAME_PHASES.started) {
      started();
    } else if (phase === FRAME_PHASES.playing) {
      options.afterFrame?.(game, timestamp);
      if (hud.frame(game, timestamp)) options.poll?.(game);
      options.afterHostFrame?.(game, timestamp);
      if (!between) between = window.setTimeout(workBetweenFrames, 0);
    } else if (phase === FRAME_PHASES.idle) {
      // A view without area asks for no frames until it has area again.
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
// A module that stops where nothing caught its error stops the page too.
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
  if (!("gpu" in navigator)) throw new Error("WebGPU is not available in this browser or execution context.");
  const created = await options.create(canvas, lifecycle.signal);
  // GPU initialization may finish after teardown even when fetch was aborted.
  if (disposed) { created.free(); return; }
  game = created;
  // The engine switches the page's parameters name.
  created.apply_host_settings(location.search);
  created.set_preparation(preparing);
  connectInput();
  // The game's jobs run on the page's workers; a module of theirs or the
  // game's that stops stops the page.
  jobs = jobPool();
  jobs?.attach(created, fail);
  resize();
  message.hidden = true;
  pause.disabled = false;
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
        game.frame(performance.now(), pixelWidth, pixelHeight);
        game.suspend();
        options.poll?.(game);
      } catch (error) {
        if (fatal(error)) fail(error);
        throw error;
      } finally { resume(); }
    },
    setResolutionScale(scale) {
      if (!Number.isFinite(scale) || scale < 0.25 || scale > 2) throw new Error("Resolution scale must be 0.25–2.");
      resolutionScale = scale; resize();
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
      fail, stopped: () => stopped,
      controls: Object.fromEntries(Object.entries(options.controls ?? {}).map(([name, control]) => [name, {
        description: control.description, run: (args: Record<string, unknown>) => control.run(created, args),
      }])),
      parameters: { ...HOST_PARAMETERS, ...options.parameters },
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
  jobs?.detach();
  disposeDevelopment?.();
  lifecycle.abort();
  stopClock(); observer.disconnect(); input.dispose();
  if (!failed) game?.free();
  game = undefined;
});
void start().catch(fail);
}
