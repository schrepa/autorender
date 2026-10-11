import { stressBindings } from "./bindings";
import { type FrameStats, type HostControls, readStats, startGame } from "../engine/host";
import { Recorder, type RecordingReport } from "../engine/recording";
import type { DetailedProfiler, DetailedProfileReport } from "../engine/profiling";
import type { StressFiles, StressGame } from "../../pkg/stress_lab";
import { fetchVirtualPrefix } from "../engine/virtual-mesh";
import { isKtx2, readRgba8Ktx2 } from "../engine/ktx2";
import { adapterFeatures } from "../engine/adapter";
import { jobPool } from "../engine/jobs";
import {
  STRESS_CATALOG, STRESS_CONFIG_LIMITS, type StressConfig, type StressSampleField, type StressScene, type StressSnapshot,
} from "../generated/stress";
import { STRESS_PARAMETERS } from "./parameters";
import "./stress.css";

/** The scanned rock maps under public/stress/rocks (vite.config.ts). */
declare const __STRESS_ROCKS__: readonly string[];

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const choice = (id: string) => el<HTMLSelectElement>(id);
const number = (id: string) => Number(el<HTMLInputElement>(id).value);
const checked = (id: string) => el<HTMLInputElement>(id).checked;
/** The values select `id` offers. */
const options = (id: string) => [...choice(id).options].map(option => option.value);
/** Fills select `id` with `choices`, which the generated catalog holds. */
function offer(id: string, choices: readonly { value: string | number; label?: string }[]) {
  choice(id).replaceChildren(...choices.map(({ value, label }) => new Option(label ?? String(value), String(value))));
}
offer("preset", STRESS_CATALOG.presets);
offer("scene-select", STRESS_CATALOG.scenes);
offer("look", STRESS_CATALOG.looks);
offer("effects", STRESS_CATALOG.effects);
offer("view", STRESS_CATALOG.views);
offer("meshes", STRESS_CATALOG.mesh_varieties.map(value => ({ value })));
// Number inputs take the bounds validation keeps.
for (const [id, field] of [["colliders", "colliders"], ["sensors", "sensor_percent"], ["seed", "seed"]] as const) {
  const limits = STRESS_CONFIG_LIMITS[field];
  el<HTMLInputElement>(id).min = String(limits.minimum);
  el<HTMLInputElement>(id).max = String(limits.maximum);
}
const scenes: StressScene[] = STRESS_CATALOG.scenes.map(scene => scene.value);
const format = (value: number) => value.toLocaleString("en-US");
const ms = (value: number) => `${value.toFixed(2)} ms`;
const bytes = (value: number) => `${(value / 1048576).toFixed(2)} MiB`;
// The presets the Stress Lab declares, the first the default; native hosts
// offer the same.
const presets: Record<string, { title: string; note: string; config: StressConfig }> = Object.fromEntries(
  STRESS_CATALOG.presets.map(({ value, title, note, config }) => [value, { title, note, config: { ...config } }]));
const baseline: StressConfig = { ...STRESS_CATALOG.presets[0].config };
let activeConfig = { ...baseline }, activeTitle: string = STRESS_CATALOG.presets[0].title;
let game: StressGame | undefined, controls: HostControls<StressGame> | undefined;
let latest: StressSnapshot | undefined, report: RecordingReport | undefined;
let previousTimestamp: number | undefined, recordStarted = 0;
let profiler: DetailedProfiler | undefined, detailedReport: DetailedProfileReport | undefined;
let wasmMemory: WebAssembly.Memory | undefined;
// The dense scene's virtual meshes, fetched once: the game
// keeps them for later workloads.
let filesClass: typeof StressFiles | undefined, denseSupplied = false;
let prefixBytes: ((header: Uint8Array) => number | undefined) | undefined;
async function denseFiles(): Promise<StressFiles | undefined> {
  if (denseSupplied || !filesClass || !prefixBytes) return undefined;
  const files = new filesClass();
  try {
    // Each file's prefix, fetched together: the renderer streams the other
    // pages from its URL.
    const names = JSON.parse(filesClass.dense_names()) as string[];
    const urls = names.map(name => new URL(`${import.meta.env.BASE_URL}stress/dense/${name}`, location.href).href);
    const prefix = prefixBytes;
    const prefixes = await Promise.all(urls.map(url => fetchVirtualPrefix(url, prefix)));
    names.forEach((name, index) => files.add_streamed(name, prefixes[index], urls[index]));
  } catch (error) {
    files.free();
    throw new Error(`${String(error)} Build the files with \`cargo run -p engine_vgbuild --release --bin dense_assets\`.`);
  }
  denseSupplied = true;
  return files;
}
// The highlands' terrain material maps, fetched once: the
// game keeps them for later workloads. Without them its ground shows plain
// colors.
let highlandsSupplied = false;
async function highlandsFiles(): Promise<StressFiles | undefined> {
  if (highlandsSupplied || !filesClass) return undefined;
  const files = new filesClass();
  try {
    const names = JSON.parse(filesClass.highlands_names()) as string[];
    const images = await Promise.all(names.map(async name => {
      const response = await fetch(new URL(`${import.meta.env.BASE_URL}stress/terrain/${name}`, location.href));
      const bytes = new Uint8Array(await response.arrayBuffer());
      const image = response.ok && isKtx2(bytes) ? readRgba8Ktx2(name, bytes) : undefined;
      if (!image) throw new Error(`stress/terrain/${name} is missing or not an uncompressed RGBA8 KTX2 image.`);
      return image;
    }));
    names.forEach((name, index) => files.add_image(name, images[index].width, images[index].height, images[index].pixels));
  } catch (error) {
    files.free();
    notice(`The ground shows plain colors: ${String(error)} Build its materials with \`cargo run -p engine_assetc --release --bin terrain_materials\`.`);
    return undefined;
  }
  // Scanned rock maps, kept locally and optional: without them rocks wear
  // the terrain rock's maps. Only those the server found are requested.
  const rocks = (JSON.parse(filesClass.rock_names()) as string[]).filter(name => __STRESS_ROCKS__.includes(name));
  await Promise.all(rocks.map(async name => {
    try {
      const response = await fetch(new URL(`${import.meta.env.BASE_URL}stress/rocks/${name}`, location.href));
      const bytes = new Uint8Array(await response.arrayBuffer());
      const image = response.ok && isKtx2(bytes) ? readRgba8Ktx2(name, bytes) : undefined;
      if (image) files.add_image(`rocks/${name}`, image.width, image.height, image.pixels);
    } catch { /* Absent or unreadable: the terrain rock's maps stand in. */ }
  }));
  highlandsSupplied = true;
  return files;
}
/** The files `config`'s scene reads that the game lacks. */
async function sceneFiles(config: StressConfig): Promise<StressFiles | undefined> {
  if (config.scene === "dense") return denseFiles();
  if (config.scene === "highlands") return highlandsFiles();
  return undefined;
}
let lastDiagnostic = -Infinity;
let wasmIdentity: { url: string | null; sha256: string | null } = { url: null, sha256: null };
const listeners = new AbortController();
const listenerOptions = { signal: listeners.signal };
const intervals: number[] = [], chart: [number, number, number][] = [];

function snapshot(runtime: StressGame): StressSnapshot { return JSON.parse(runtime.stress_stats()) as StressSnapshot; }
/** What a recording or a detailed profile notes about the run it measures, on a `canvas` sized canvas. */
function runMetadata(data: StressSnapshot, canvas: [number, number]) {
  return { config: data.config, seed: data.config.seed, generation: data.generation, camera: data.camera,
    culling: data.config.culling, resolution_scale: number("resolution"), canvas,
    presentation: game?.gpu_presentation() === false ? "cpu" : "gpu", pipeline: game?.tick_pipelining() === false ? "off" : "on",
    taa: game?.temporal_antialiasing() ? "on" : "off", occlusion: game?.occlusion_culling() ? "on" : "off",
    subpixel: game?.subpixel_culling() ? "on" : "off", visibility, lod, impostors,
    look: choice("look").value, effects: choice("effects").value,
    build: params.get("engine_build"), runtime: "stress-lab-0.1", mode: import.meta.env.MODE,
    start_frame: data.frame, start_tick: data.tick, workload: activeTitle, wasm: wasmIdentity, scroll: [scrollX, scrollY] };
}
/** Offers JSON `text` as the download `name`. */
function download(text: string, name: string) {
  const url = URL.createObjectURL(new Blob([text], { type: "application/json" }));
  const anchor = document.createElement("a"); anchor.href = url; anchor.download = name;
  anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function notice(text: string, error = false) {
  el("notice").textContent = text; el("notice").dataset.error = String(error);
}
function lock() {
  const value = recorder.active || !!profiler?.busy;
  el<HTMLFieldSetElement>("workload-fields").disabled = value;
  for (const id of ["preset", "view", "resolution", "look", "effects", "culling", "reset", "share"]) {
    (el(id) as HTMLButtonElement | HTMLSelectElement | HTMLInputElement).disabled = value;
  }
  controls?.setInputEnabled(!value);
  el("record").textContent = recorder.active ? "Stop recording" : "Record 10 seconds";
  el("quick-record").textContent = recorder.active ? "Stop" : "Record";
  el<HTMLButtonElement>("record").disabled = !game || !!profiler?.busy;
  el<HTMLButtonElement>("quick-record").disabled = !game || !!profiler?.busy;
  el<HTMLButtonElement>("profile").disabled = !profiler || recorder.active || (profiler.busy && !profiler.active);
  el<HTMLButtonElement>("quick-profile").disabled = el<HTMLButtonElement>("profile").disabled;
  el("quick-profile").hidden = import.meta.env.MODE !== "profiling";
  el("quick-profile").textContent = profiler?.active ? "Stop profile" : "Profile 5s";
  el("profile").textContent = profiler?.active ? "Stop profile" : profiler?.busy ? "Collecting GPU results…" : "Detailed profile · 5 seconds";
  el<HTMLButtonElement>("profile-export").disabled = !detailedReport || value;
}
const recorder = new Recorder(value => {
  report = value; lock();
  el<HTMLButtonElement>("export").disabled = false;
  const total = value.cpu_ms.total, cadence = value.frame_interval_ms;
  const state = value.complete ? "Recorded" : `Incomplete (${value.reason})`;
  el("record-status").textContent = `${state} · ${value.samples} frames · CPU median ${total?.median.toFixed(2) ?? "—"} ms / p95 ${total?.p95.toFixed(2) ?? "—"} ms · frame p95 ${cadence?.p95.toFixed(2) ?? "—"} ms. Export includes diagnostics.`;
});

function fill(config: StressConfig) {
  for (const [id, field] of [["objects", "renderables"], ["moving", "moving_percent"], ["colliders", "colliders"],
    ["sensors", "sensor_percent"], ["queries", "queries_per_tick"], ["meshes", "mesh_variety"], ["seed", "seed"]] as const) {
    const input = el<HTMLInputElement | HTMLSelectElement>(id);
    if (input instanceof HTMLSelectElement && ![...input.options].some(option => Number(option.value) === config[field])) {
      input.add(new Option(String(config[field]), String(config[field])));
    }
    input.value = String(config[field]);
  }
  for (const [id, field] of [["hierarchy", "hierarchy"], ["interpolation", "interpolation"], ["textures", "textured"], ["dense", "dense_colliders"], ["culling", "culling"],
    ["compressed", "compressed_textures"]] as const) {
    el<HTMLInputElement>(id).checked = config[field];
  }
  choice("scene-select").value = config.scene;
}
function readConfig(): StressConfig {
  return { seed: number("seed"), renderables: number("objects"), moving_percent: number("moving"), colliders: number("colliders"),
    sensor_percent: number("sensors"), queries_per_tick: number("queries"), mesh_variety: number("meshes"), hierarchy: checked("hierarchy"),
    interpolation: checked("interpolation"), textured: checked("textures"), dense_colliders: checked("dense"), culling: checked("culling"),
    scene: choice("scene-select").value as StressScene, compressed_textures: checked("compressed") };
}
/** One fingerprint of the module, taken in the background at startup,
 * outside measurement windows: exports made before it ends carry none.
 * Production filenames and managed generation URLs are immutable; do not
 * rebuild during a run. */
async function fingerprintWasm() {
  const wasmUrl = performance.getEntriesByType("resource").map(entry => entry.name)
    .find(name => name.includes("stress_lab") && new URL(name).pathname.endsWith(".wasm"));
  if (!wasmUrl) return;
  wasmIdentity.url = wasmUrl;
  try {
    const maximumBytes = (import.meta.env.MODE === "profiling" ? 64 : 8) * 1024 * 1024;
    const response = await fetch(wasmUrl, { cache: "force-cache" });
    if (response.ok && Number(response.headers.get("content-length") ?? 0) <= maximumBytes) {
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength <= maximumBytes) {
        const hash = await crypto.subtle.digest("SHA-256", buffer);
        wasmIdentity.sha256 = Array.from(new Uint8Array(hash), byte => byte.toString(16).padStart(2, "0")).join("");
      }
    }
  } catch { /* A missing optional fingerprint remains explicit in the export. */ }
}
/** Whether the adapter samples BC7, which compressed metro textures need. */
let blockCompression: boolean | undefined;
async function detectBlockCompression(): Promise<boolean> {
  return (await adapterFeatures()).has("texture-compression-bc");
}
/** `config` with compressed textures off where BC7 is unavailable. */
function supported(config: StressConfig): StressConfig {
  if (!config.compressed_textures || blockCompression !== false) return config;
  notice("This GPU lacks texture-compression-bc: textures stay RGBA8.", true);
  const fallback = { ...config, compressed_textures: false };
  fill(fallback);
  return fallback;
}
async function apply(requested: StressConfig, title: string) {
  if (!controls) return;
  const config = supported(requested);
  if (profiler?.busy || recorder.active) { notice("Finish the active capture before changing the workload.", true); return; }
  let files: StressFiles | undefined;
  try { files = await sceneFiles(config); } catch (error) { notice(`Workload unchanged: ${String(error)}`, true); return; }
  try {
    controls.mutate(runtime => {
      runtime.configure(JSON.stringify(config), files);
    });
    activeConfig = config; activeTitle = title;
    choice("view").value = "overview";
    chart.length = 0; intervals.length = 0; previousTimestamp = undefined;
    el("active-preset").textContent = title;
    notice(`Applied seed ${config.seed}. Compare one setting at a time for a useful baseline.`);
  } catch (error) { notice(`Workload unchanged: ${String(error)}`, true); }
}
function mutate(action: (runtime: StressGame) => void) {
  if (profiler?.busy || recorder.active) { notice("Finish the active capture before changing the view.", true); return; }
  try { controls?.mutate(action); } catch (error) { notice(String(error), true); }
}

function drawChart(stats: FrameStats) {
  chart.push([stats.simulation_ms, stats.extraction_ms, stats.render_ms]);
  if (chart.length > 120) chart.shift();
  const max = Math.max(33.34, ...chart.flat()) * 1.08;
  const y = (value: number) => 110 - value / max * 106;
  for (const [index, id] of ["sim-path", "extract-path", "render-path"].entries()) {
    const path = chart.map((values, i) => `${i ? "L" : "M"}${(i / 119 * 640).toFixed(1)},${y(values[index]).toFixed(1)}`).join(" ");
    el(id).setAttribute("d", path);
  }
  el("budget-line").setAttribute("y1", String(y(16.67))); el("budget-line").setAttribute("y2", String(y(16.67)));
  el("chart-scale").textContent = `Scale ${max.toFixed(1)} ms · 120 recent samples`;
}

function poll(runtime: StressGame) {
  latest = snapshot(runtime);
  const stats = readStats(runtime), data = latest;
  el("active-summary").textContent = `${format(data.renderables)} objects · ${format(data.moving_nodes)} moving nodes · seed ${data.config.seed}`;
  el("scene-count").textContent = `${format(data.renderables)} OBJECTS`;
  el("pixel-size").textContent = controls ? controls.size().join(" × ") : "";
  el("render-cpu").textContent = ms(stats.render_ms);
  const cpu = stats.simulation_ms + stats.extraction_ms + stats.render_ms;
  el("frame-budget").textContent = `${(cpu / 16.67 * 100).toFixed(0)}%`;
  el("drawn").textContent = `${format(stats.instances)} / ${format(stats.render_candidates)}`;
  el("nodes").textContent = format(data.transform_nodes);
  el("sync-time").textContent = ms(data.timings.sync_pre_ms + data.timings.sync_post_ms);
  el("sensor-time").textContent = ms(data.timings.sensor_ms);
  el("query-time").textContent = `${ms(data.timings.queries_ms)} / ${format(data.queries.calls)}`;
  el("pairs").textContent = `${format(data.collision.sensor_pairs)} / ${format(data.collision.sensor_candidates)}`;
  el("gpu-bytes").textContent = bytes(stats.resident_mesh_bytes + stats.resident_texture_bytes);
  el("cache-bytes").textContent = bytes(data.collision.cache_bytes);
  el("query-errors").textContent = `${format(data.queries.errors)} / ${format(data.queries.hits)}`;
  el("dropped").textContent = `${stats.dropped_seconds.toFixed(2)} s`;
  const limited = !data.collision.complete || (data.collision.sensor_attempted_tick !== null && !data.collision.sensor_complete) || data.queries.errors > 0;
  el("collision-state").dataset.state = limited ? "limited" : "ready";
  el("collision-state").textContent = limited ? (data.collision.sensor_error ?? data.collision.error ?? data.queries.last_error ?? "Incomplete") : `${format(data.collision.active_colliders)} active colliders`;
  const labels = ["Simulation", "Extraction", "Render preparation/submission"];
  const stages = [stats.simulation_ms, stats.extraction_ms, stats.render_ms];
  const largest = labels[stages.indexOf(Math.max(...stages))];
  el("pressure").textContent = limited ? "Collision work reached a limit or failed. Inspect the exported counters before comparing timings."
    : `${largest} is the largest sampled CPU stage. ${cpu > 16.67 ? "CPU work exceeds a 60 Hz frame budget." : "CPU work is currently within a 60 Hz frame budget."}`;
  if (intervals.length) {
    const sorted = [...intervals].sort((a, b) => a - b);
    el("live-cadence").textContent = `${sorted[Math.floor(sorted.length / 2)].toFixed(1)} ms median interval`;
  }
  drawChart(stats);
  if (recorder.active) el("record-status").textContent = `Recording · ${((performance.now() - recordStarted) / 1000).toFixed(1)} s including warm-up · camera locked`;
  if (profiler?.active) el("profile-status").textContent = `Capturing CPU, collision, allocator and browser events · ${((Number(profiler.status().elapsed_ms) || 0) / 1000).toFixed(1)} / 5 seconds · camera locked`;
}

// The page's parameters, which STRESS_PARAMETERS describes.
const params = new URLSearchParams(location.search);
const visibility = params.get("visibility") === "on" ? "on" : "off";
const lod = params.get("lod") === "off" ? "off" : "on";
const impostors = params.get("impostors") === "off" ? "off" : "on";
// The engine refuses unknown looks and effects: the page fails with why.
const initialLook = params.get("look") ?? "scene";
const initialEffects = params.get("effects") ?? "all";
if (params.has("config")) {
  try {
    const source = params.get("config")!;
    if (source.length > 2000) throw new Error("Workload link is too large.");
    const parsed = JSON.parse(source) as Partial<StressConfig>;
    // Links from before scenes existed build the tower field.
    // and before compressed textures, without them.
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      parsed.scene ??= "field";
      parsed.compressed_textures ??= false;
    }
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object" || Object.keys(parsed).length !== Object.keys(baseline).length ||
        Object.entries(baseline).some(([key, value]) => typeof parsed[key as keyof StressConfig] !== typeof value) || !scenes.includes(parsed.scene as StressScene)) {
      throw new Error("Expected the complete workload configuration.");
    }
    activeConfig = parsed as StressConfig;
    activeTitle = "SHARED WORKLOAD";
  } catch (error) { notice(`Invalid workload link; using visual preset. ${String(error)}`, true); activeConfig = { ...baseline }; }
}
const requestedPreset = params.get("preset");
if (!params.has("config") && requestedPreset && requestedPreset in presets) {
  activeConfig = { ...presets[requestedPreset].config };
  activeTitle = presets[requestedPreset].title;
  choice("preset").value = requestedPreset;
  el("preset-note").textContent = presets[requestedPreset].note;
}
fill(activeConfig);
el("active-preset").textContent = activeTitle;
el("preset-note").textContent = presets[choice("preset").value].note;

startGame({
  parameters: STRESS_PARAMETERS,
  // What agents change through the game control; each refreshes the image.
  controls: {
    workload: {
      description: "Runs preset {preset: NAME} or configuration {config: {...}} (fields over the current one) in place of the workload",
      async run(runtime, args) {
        const preset = typeof args.preset === "string" ? presets[args.preset] : undefined;
        if (args.preset !== undefined && !preset) throw new Error(`no preset ${String(args.preset)}; presets are ${Object.keys(presets).join(", ")}`);
        const config: StressConfig = preset ? { ...preset.config } : { ...activeConfig, ...(args.config as Partial<StressConfig> | undefined) };
        runtime.configure(JSON.stringify(config), await sceneFiles(config));
        activeConfig = config; activeTitle = preset?.title ?? "AGENT WORKLOAD";
        fill(config); el("active-preset").textContent = activeTitle;
        return { config, title: activeTitle };
      },
    },
    look: {
      description: `Lighting preset {name}: ${STRESS_CATALOG.looks.map(look => look.value).join(", ")}`,
      run(runtime, args) { runtime.set_look(String(args.name)); choice("look").value = String(args.name); return { look: args.name }; },
    },
    effects: {
      description: `Rendering effects {name}: all, none or comma-separated ${STRESS_CATALOG.effects.filter(effect => effect.value.startsWith("no-")).map(effect => effect.value).join(", ")}`,
      run(runtime, args) { runtime.set_effects(String(args.name)); return { effects: args.name }; },
    },
    view: {
      description: `Camera view {name}: ${STRESS_CATALOG.views.map(view => view.value).join(", ")}`,
      run(runtime, args) { runtime.set_view(String(args.name)); choice("view").value = String(args.name); return { view: args.name }; },
    },
  },
  async create(canvas) {
    // Job workers start while the game's module loads.
    jobPool();
    const { default: init, StressGame, StressFiles: files, virtual_prefix_bytes } = await stressBindings();
    filesClass = files;
    prefixBytes = virtual_prefix_bytes;
    // The adapter's features arrive while the module compiles.
    const compression = detectBlockCompression();
    const initialized = await init();
    wasmMemory = (initialized as { memory?: WebAssembly.Memory }).memory;
    void fingerprintWasm();
    blockCompression = await compression;
    activeConfig = supported(activeConfig);
    return StressGame.create(canvas, JSON.stringify(activeConfig), await sceneFiles(activeConfig));
  },
  ready(runtime, host) {
    game = runtime; controls = host;
    runtime.set_visibility_buffer(visibility === "on");
    runtime.set_level_of_detail(lod === "on");
    runtime.set_impostors(impostors === "on");
    runtime.set_look(initialLook);
    choice("look").value = initialLook;
    runtime.set_effects(initialEffects);
    // Combinations without an option of their own keep the select on the
    // closest single choice; recordings report the actual setting.
    if ([...choice("effects").options].some(option => option.value === initialEffects)) choice("effects").value = initialEffects;
    lock();
    el("active-preset").textContent = activeTitle;
    if (import.meta.env.MODE === "profiling") {
      void import("../engine/profiling").then(({ DetailedProfiler, supportsProfiling }) => {
        if (listeners.signal.aborted || !supportsProfiling(runtime)) {
          if (!listeners.signal.aborted) el("profile-status").textContent = "These bindings do not expose detailed profiling. Build the profiling target to enable capture.";
          return;
        }
        profiler = new DetailedProfiler(runtime, {
          allowed() {
            if (recorder.active) throw new Error("Stop the baseline recording before profiling");
            if (host.busy() || host.paused() || document.hidden) throw new Error("Resume a visible, available scene before profiling");
          },
          metadata() {
            const data = snapshot(runtime);
            return { ...runMetadata(data, host.size()), viewport: [innerWidth, innerHeight], device_pixel_ratio: devicePixelRatio,
              diagnostics_at_start: data };
          },
          memory: () => wasmMemory,
          changed: lock,
          completed(value) {
            detailedReport = value; lock();
            const gpu = value.summary.gpu as Record<string, unknown>;
            el("profile-status").textContent = `${value.complete ? "Captured" : `Incomplete (${value.reason})`} · ${value.summary.cpu_events} CPU events · ${value.summary.host_frames} frames · ${value.summary.browser_entries} browser entries. Export opens in Perfetto or Chrome tracing.`;
            el("profile-cpu").textContent = `${format(Number(value.summary.cpu_events))} events · ${format(Number(value.summary.dropped_cpu_events))} dropped`;
            el("profile-gpu").textContent = gpu.supported ? `${gpu.completed ?? 0} pass samples · ${gpu.pending ?? 0} pending · ${gpu.dropped ?? 0} dropped` : String(gpu.reason ?? "GPU timestamps unavailable");
            const memory = value.summary.memory_end as { wasm_linear_bytes: number | null; js_heap: { used_bytes: number } | null } | null;
            el("profile-memory").textContent = `${memory?.wasm_linear_bytes === null || memory?.wasm_linear_bytes === undefined ? "Wasm unavailable" : `${bytes(memory.wasm_linear_bytes)} Wasm capacity`} · ${memory?.js_heap ? `${bytes(memory.js_heap.used_bytes)} JS heap` : "JS heap unavailable"}`;
            el("profile-browser").textContent = `${value.summary.browser_entries} browser entries · ${value.browser.dropped_entries} dropped`;
          },
        });
        el("profile-status").textContent = "Ready · five seconds of CPU scopes, collision details, allocation counters, supported GPU timestamps and browser stalls.";
        lock();
      }).catch(error => { el("profile-status").textContent = `Profiler unavailable: ${String(error)}`; });
    }
  },
  poll,
  interrupted(reason) { recorder.abort(reason); profiler?.abort(reason); previousTimestamp = undefined; },
  beforeFrame: import.meta.env.MODE === "profiling" ? (_runtime, timestamp) => profiler?.beforeFrame(timestamp) : undefined,
  afterHostFrame: import.meta.env.MODE === "profiling" ? () => profiler?.afterHostFrame() : undefined,
  afterFrame(runtime, timestamp) {
    if (import.meta.env.MODE === "profiling") profiler?.afterFrame();
    if (previousTimestamp !== undefined) {
      intervals.push(timestamp - previousTimestamp); if (intervals.length > 120) intervals.shift();
    }
    previousTimestamp = timestamp;
    if (!recorder.active) return;
    // The workload's numbers without its full snapshot, read every 250 ms.
    const stats = readStats(runtime);
    const values = runtime.frame_sample();
    const sample = Object.fromEntries(STRESS_CATALOG.sample.map((field, index) => [field, values[index]])) as Record<StressSampleField, number>;
    recorder.sample({ timestamp, simulation_ms: stats.simulation_ms, extraction_ms: stats.extraction_ms, render_ms: stats.render_ms,
      dropped_seconds: stats.dropped_seconds,
      phases_ms: { collision_sync: sample.sync_ms, sensors: sample.sensor_ms, queries: sample.queries_ms, animation: sample.animation_ms },
      counters: { instances: stats.instances, candidates: stats.render_candidates, visible: stats.visible_instances, culled: stats.culled_instances,
        draws: stats.draw_calls, upload_bytes: stats.upload_bytes, fixed_steps: stats.fixed_steps, missing_assets: stats.missing_assets,
        invalid_transforms: stats.invalid_transforms, dropped_instances: stats.dropped_instances, culling: stats.culling_enabled,
        gpu_mesh_bytes: stats.resident_mesh_bytes, gpu_texture_bytes: stats.resident_texture_bytes,
        mesh_upload_bytes: stats.mesh_upload_bytes, texture_upload_bytes: stats.texture_upload_bytes,
        colliders: sample.active_colliders, collision_complete: sample.collision_complete === 1, invalid_colliders: sample.invalid_colliders,
        overflow_colliders: sample.overflow_colliders, sensor_complete: sample.sensor_complete === 1, sensor_pairs: sample.sensor_pairs,
        sensor_candidates: sample.sensor_candidates, query_calls: sample.query_calls, query_hits: sample.query_hits, query_errors: sample.query_errors },
    });
    if (timestamp - lastDiagnostic >= 250) { recorder.diagnostic(timestamp, snapshot(runtime)); lastDiagnostic = timestamp; }
  },
});

el("preset").addEventListener("change", () => {
  const preset = presets[choice("preset").value];
  fill(preset.config); el("preset-note").textContent = preset.note;
  apply({ ...preset.config }, preset.title);
}, listenerOptions);
el("apply").addEventListener("click", () => apply(readConfig(), "CUSTOM WORKLOAD"), listenerOptions);
el("view").addEventListener("change", () => mutate(runtime => runtime.set_view(choice("view").value)), listenerOptions);
el("reset").addEventListener("click", () => { recorder.abort("camera_changed"); choice("view").value = "overview"; mutate(runtime => runtime.reset_camera()); }, listenerOptions);
el("culling").addEventListener("change", () => mutate(runtime => { runtime.set_culling(checked("culling")); activeConfig.culling = checked("culling"); }), listenerOptions);
el("resolution").addEventListener("change", () => mutate(() => controls!.setResolutionScale(number("resolution"))), listenerOptions);
el("look").addEventListener("change", () => mutate(runtime => runtime.set_look(choice("look").value)), listenerOptions);
el("effects").addEventListener("change", () => mutate(runtime => runtime.set_effects(choice("effects").value)), listenerOptions);
el("record").addEventListener("click", () => {
  if (recorder.active) { recorder.abort("stopped_by_user"); return; }
  if (profiler?.busy) { notice("Finish the detailed profile before recording a baseline.", true); return; }
  if (!game || !controls || controls.busy() || controls.paused()) { notice("Resume the scene before recording.", true); return; }
  try {
    recorder.start({ ...runMetadata(snapshot(game), controls.size()),
      instrumentation: "Demo phase timers are enabled. While recording, small aggregate Rust observations are serialized once per frame. CPU stage timers exclude this host work; RAF intervals include it. GPU execution time is not measured." });
    recordStarted = performance.now(); lastDiagnostic = -Infinity; lock();
    notice("Recording the current workload. Camera input is temporarily locked.");
  } catch (error) { notice(String(error), true); }
}, listenerOptions);
el("quick-record").addEventListener("click", () => el("record").click(), listenerOptions);
el("profile").addEventListener("click", () => {
  if (!profiler) return;
  try {
    if (profiler.active) { void profiler.stop().catch(error => notice(String(error), true)); return; }
    profiler.start(); detailedReport = undefined; lock();
    el("profile-status").textContent = "Capturing for five seconds. Camera and workload controls are locked.";
    notice("Detailed profiling adds measurement overhead. Keep this capture separate from your ordinary baseline.");
  } catch (error) { notice(String(error), true); }
}, listenerOptions);
el("quick-profile").addEventListener("click", () => el("profile").click(), listenerOptions);
el("profile-export").addEventListener("click", () => {
  if (!profiler || profiler.busy) return;
  try {
    const value = profiler.export();
    const serialized = JSON.stringify(value);
    if (new TextEncoder().encode(serialized).byteLength > 32 * 1024 * 1024) {
      el("profile-status").textContent = "Export exceeds the 32 MiB limit. The capture is retained; start a shorter capture or request fewer events through the agent.";
      return;
    }
    download(serialized, `stress-profile-${value.complete ? "complete" : "incomplete"}-${value.started_at.replaceAll(":", "-")}.json`);
  } catch (error) { notice(String(error), true); }
}, listenerOptions);
el("export").addEventListener("click", () => {
  if (!report) return;
  download(JSON.stringify(report, null, 2), `stress-${report.complete ? "complete" : "incomplete"}-${report.started_at.replaceAll(":", "-")}.json`);
}, listenerOptions);
el("share").addEventListener("click", async () => {
  if (!game) return;
  const url = new URL(location.href); url.search = "";
  url.searchParams.set("config", JSON.stringify(snapshot(game).config));
  try { await navigator.clipboard.writeText(url.href); notice("Workload link copied. It opens at the overview camera with the same configuration and seed."); }
  catch { notice(`Workload link: ${url.href}`); }
}, listenerOptions);
if (import.meta.hot) import.meta.hot.dispose(() => { recorder.abort("host_reload"); profiler?.dispose(); listeners.abort(); });
