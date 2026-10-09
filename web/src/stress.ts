import { stressBindings } from "./bindings";
import { boot, readStats, type HostControls, type RuntimeStats } from "./host";
import { StressRecorder, type StressRecordingReport } from "./stress-recording";
import type { DetailedProfiler, DetailedProfileReport } from "./profiling";
import type { StressFiles, StressGame } from "../pkg/stress_lab";
import { fetchVirtualPrefix } from "./virtual-mesh";
import { isKtx2, readRgba8Ktx2 } from "./ktx2";
import { adapterFeatures } from "./adapter";
import { jobPool } from "./jobs";
import "./stress.css";

type Scene = "field" | "city" | "metro" | "dense" | "highlands";
type Config = {
  seed: number; renderables: number; moving_percent: number; hierarchy: boolean;
  interpolation: boolean; colliders: number; sensor_percent: number; dense_colliders: boolean;
  queries_per_tick: number; mesh_variety: number; textured: boolean; culling: boolean; scene: Scene;
  compressed_textures: boolean;
};
type Snapshot = {
  version: number; generation: number; frame: string; tick: string; config: Config;
  renderables: number; transform_nodes: number; moving_nodes: number; sensor_nodes: number;
  timings: { sync_pre_ms: number; sync_post_ms: number; sensor_ms: number; queries_ms: number; animation_ms: number; scope: string };
  collision: { colliders: number; active_colliders: number; invalid_colliders: number; overflow_colliders: number; complete: boolean;
    bvh_nodes: number; cache_bytes: number; error: string | null; sensor_pairs: number; sensor_events: number; sensor_complete: boolean;
    sensor_candidates: number; sensor_error: string | null; sensor_attempted_tick: string | null };
  queries: { calls: number; hits: number; errors: number; total_errors: string; last_error: string | null };
  camera: { position: number[]; target: number[]; vertical_fov_radians: number; near: number; far: number };
};
const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const choice = (id: string) => el<HTMLSelectElement>(id);
const number = (id: string) => Number(el<HTMLInputElement>(id).value);
const checked = (id: string) => el<HTMLInputElement>(id).checked;
/** The values select `id` offers: the page's one list of scenes, looks and effects. */
const options = (id: string) => [...choice(id).options].map(option => option.value);
const scenes = options("scene-select") as Scene[];
const format = (value: number) => value.toLocaleString("en-US");
const ms = (value: number) => `${value.toFixed(2)} ms`;
const bytes = (value: number) => `${(value / 1048576).toFixed(2)} MiB`;
const baseline: Config = { seed: 1337, renderables: 90000, moving_percent: 0, hierarchy: false, interpolation: false,
  colliders: 0, sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 32, textured: true, culling: true, scene: "field",
  compressed_textures: false };
const presets: Record<string, { title: string; note: string; config: Config }> = {
  visual: { title: "MAXIMUM VISUAL LOAD", note: "Rendering, extraction, batching and texture sampling at maximum object count.", config: baseline },
  city: { title: "WILD CITY", note: "52 building, vehicle and machine shapes: skyscrapers, traffic, trains, drones, cranes and Ferris wheels. Moving objects are the traffic.",
    config: { ...baseline, scene: "city", moving_percent: 10, hierarchy: true, interpolation: true, mesh_variety: 64 } },
  metro: { title: "METRO", note: "The city with every engine feature: glTF maps over procedural textures (BC7 where supported), refracting glass, masked foliage, tints and rotating ads, skinned people, keyframed props, thousands of lights and particles.",
    config: { ...baseline, scene: "metro", moving_percent: 10, hierarchy: true, interpolation: true, mesh_variety: 64, compressed_textures: true } },
  dense: { title: "DENSE GEOMETRY", note: "Boulders and fluted columns of 330,000 to 590,000 triangles each on terrain tiles, drawn as virtual geometry: the GPU picks the clusters each view needs. Build the files with `cargo run -p engine_vgbuild --release --bin dense_assets`.",
    config: { ...baseline, scene: "dense", renderables: 2000, moving_percent: 5, hierarchy: false, interpolation: true, mesh_variety: 8 } },
  highlands: { title: "HIGHLANDS", note: "An 8 km terrain the engine generates as the camera moves: eroded mountains from noise, then hand-placed stamps level a plateau for a hamlet of towers, flatten a valley floor and cut a cave mouth. Coarse levels keep the whole world visible; finer tiles stream around the camera.",
    config: { ...baseline, scene: "highlands", renderables: 4000, moving_percent: 10, hierarchy: true, interpolation: true, mesh_variety: 32 } },
  mixed: { title: "MIXED WORKLOAD", note: "A moving hierarchy, interpolation, filtered queries and sensors share the frame budget.",
    config: { ...baseline, renderables: 10000, moving_percent: 25, hierarchy: true, interpolation: true, colliders: 1000, sensor_percent: 10, queries_per_tick: 32, mesh_variety: 8 } },
  motion: { title: "TRANSFORM PRESSURE", note: "50,000 moving objects with parents and interpolation. Toggle each feature to isolate its cost.",
    config: { ...baseline, renderables: 50000, moving_percent: 100, hierarchy: true, interpolation: true, mesh_variety: 1, textured: false } },
  queries: { title: "SPATIAL QUERY PRESSURE", note: "10,000 colliders with moving bounds and 256 ray, overlap and sweep queries per fixed tick.",
    config: { ...baseline, renderables: 10000, moving_percent: 10, colliders: 10000, queries_per_tick: 256, mesh_variety: 8, textured: false } },
  sensors: { title: "DENSE SENSOR PRESSURE", note: "Clusters deliberately exceed pair budgets. Incomplete detection is visible and retained in recordings.",
    config: { ...baseline, renderables: 10000, moving_percent: 25, colliders: 2500, sensor_percent: 100, dense_colliders: true, mesh_variety: 8 } },
};
let activeConfig = { ...baseline }, activeTitle = presets.visual.title;
let game: StressGame | undefined, controls: HostControls<StressGame> | undefined;
let latest: Snapshot | undefined, report: StressRecordingReport | undefined;
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
  // the terrain rock's maps.
  const rocks = JSON.parse(filesClass.rock_names()) as string[];
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
async function sceneFiles(config: Config): Promise<StressFiles | undefined> {
  if (config.scene === "dense") return denseFiles();
  if (config.scene === "highlands") return highlandsFiles();
  return undefined;
}
let lastDiagnostic = -Infinity;
let wasmIdentity: { url: string | null; sha256: string | null } = { url: null, sha256: null };
const listeners = new AbortController();
const listenerOptions = { signal: listeners.signal };
const intervals: number[] = [], chart: [number, number, number][] = [];

function snapshot(runtime: StressGame): Snapshot { return JSON.parse(runtime.stress_stats()) as Snapshot; }
/** What a recording or a detailed profile notes about the run it measures, on a `canvas` sized canvas. */
function runMetadata(data: Snapshot, canvas: [number, number]) {
  return { config: data.config, seed: data.config.seed, generation: data.generation, camera: data.camera,
    culling: data.config.culling, resolution_scale: number("resolution"), canvas, presentation, pipeline,
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
const recorder = new StressRecorder(value => {
  report = value; lock();
  el<HTMLButtonElement>("export").disabled = false;
  const total = value.cpu_ms.total, cadence = value.frame_interval_ms;
  const state = value.complete ? "Recorded" : `Incomplete (${value.reason})`;
  el("record-status").textContent = `${state} · ${value.samples} frames · CPU median ${total?.median.toFixed(2) ?? "—"} ms / p95 ${total?.p95.toFixed(2) ?? "—"} ms · frame p95 ${cadence?.p95.toFixed(2) ?? "—"} ms. Export includes diagnostics.`;
});

function fill(config: Config) {
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
function readConfig(): Config {
  return { seed: number("seed"), renderables: number("objects"), moving_percent: number("moving"), colliders: number("colliders"),
    sensor_percent: number("sensors"), queries_per_tick: number("queries"), mesh_variety: number("meshes"), hierarchy: checked("hierarchy"),
    interpolation: checked("interpolation"), textured: checked("textures"), dense_colliders: checked("dense"), culling: checked("culling"),
    scene: choice("scene-select").value as Scene, compressed_textures: checked("compressed") };
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
function supported(config: Config): Config {
  if (!config.compressed_textures || blockCompression !== false) return config;
  notice("This GPU lacks texture-compression-bc: textures stay RGBA8.", true);
  const fallback = { ...config, compressed_textures: false };
  fill(fallback);
  return fallback;
}
async function apply(requested: Config, title: string) {
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

function drawChart(stats: RuntimeStats) {
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

const params = new URLSearchParams(location.search);
/** A ?name=on|off choice over each workload's own setting, or null. */
function toggle(name: string): "on" | "off" | null {
  const value = params.get(name);
  return value === "on" || value === "off" ? value : null;
}
// Ordinary frames blend, compose and cull on the GPU unless ?presentation=cpu,
// and GPU-presented frames split fixed ticks across frames unless
// ?pipeline=off. ?taa=on|off temporally antialiases them or not (the
// highlands' do), ?occlusion=off draws every frustum-visible object, and
// ?subpixel=off also those covering no pixel center, for A/B; each sets the
// world's settings over every workload's own.
const presentation = params.get("presentation") === "cpu" ? "cpu" : "gpu";
const pipeline = params.get("pipeline") === "off" ? "off" : "on";
const taa = toggle("taa");
const occlusion = toggle("occlusion");
const subpixel = toggle("subpixel");
// ?visibility=on shades virtual geometry through the visibility buffer
// instead of as it draws, for A/B.
const visibility = params.get("visibility") === "on" ? "on" : "off";
// ?lod=off draws every instance as its mesh instead of at its level of
// detail, for A/B.
const lod = params.get("lod") === "off" ? "off" : "on";
// ?impostors=off draws no impostors, only mesh levels.
const impostors = params.get("impostors") === "off" ? "off" : "on";
// ?look=golden|dusk|night|day|sky picks a lighting preset over the scene's own;
// sky is the physically based sky at a time of day.
const initialLook = options("look").includes(params.get("look") ?? "") ? params.get("look")! : "scene";
// ?effects=none, or comma-separated no-shadows, no-occlusion, no-bloom,
// no-lights, no-shafts, no-reflections, no-particles, no-refraction and
// no-probes, turns effects off.
const effectsOff = options("effects").filter(value => value.startsWith("no-"));
const requestedEffects = params.get("effects") ?? "all";
const initialEffects = requestedEffects === "none" || requestedEffects.split(",").every(token => effectsOff.includes(token))
  ? requestedEffects : "all";
if (params.has("config")) {
  try {
    const source = params.get("config")!;
    if (source.length > 2000) throw new Error("Workload link is too large.");
    const parsed = JSON.parse(source) as Partial<Config>;
    // Links from before scenes existed build the tower field.
    // and before compressed textures, without them.
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
      parsed.scene ??= "field";
      parsed.compressed_textures ??= false;
    }
    if (!parsed || Array.isArray(parsed) || typeof parsed !== "object" || Object.keys(parsed).length !== Object.keys(baseline).length ||
        Object.entries(baseline).some(([key, value]) => typeof parsed[key as keyof Config] !== typeof value) || !scenes.includes(parsed.scene as Scene)) {
      throw new Error("Expected the complete workload configuration.");
    }
    activeConfig = parsed as Config;
    activeTitle = "SHARED WORKLOAD";
  } catch (error) { notice(`Invalid workload link; using visual preset. ${String(error)}`, true); activeConfig = { ...baseline }; }
}
// ?preset=NAME starts from that preset instead of the maximum visual load.
const requestedPreset = params.get("preset");
if (!params.has("config") && requestedPreset && requestedPreset in presets) {
  activeConfig = { ...presets[requestedPreset].config };
  activeTitle = presets[requestedPreset].title;
  choice("preset").value = requestedPreset;
  el("preset-note").textContent = presets[requestedPreset].note;
}
fill(activeConfig);

boot({
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
    runtime.set_gpu_presentation(presentation === "gpu");
    runtime.set_tick_pipelining(pipeline === "on");
    if (taa) runtime.set_temporal_antialiasing(taa === "on");
    if (occlusion) runtime.set_occlusion_culling(occlusion === "on");
    if (subpixel) runtime.set_subpixel_culling(subpixel === "on");
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
      void import("./profiling").then(({ DetailedProfiler, supportsProfiling }) => {
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
    const [syncMs, sensorMs, queriesMs, animationMs, activeColliders, invalidColliders, overflowColliders, collisionComplete,
      sensorPairs, sensorCandidates, sensorComplete, queryCalls, queryHits, queryErrors] = runtime.frame_sample();
    recorder.sample({ timestamp, simulation_ms: stats.simulation_ms, extraction_ms: stats.extraction_ms, render_ms: stats.render_ms,
      dropped_seconds: stats.dropped_seconds,
      phases_ms: { collision_sync: syncMs, sensors: sensorMs, queries: queriesMs, animation: animationMs },
      counters: { instances: stats.instances, candidates: stats.render_candidates, visible: stats.visible_instances, culled: stats.culled_instances,
        draws: stats.draw_calls, upload_bytes: stats.upload_bytes, fixed_steps: stats.fixed_steps, missing_assets: stats.missing_assets,
        invalid_transforms: stats.invalid_transforms, dropped_instances: stats.dropped_instances, culling: stats.culling_enabled,
        gpu_mesh_bytes: stats.resident_mesh_bytes, gpu_texture_bytes: stats.resident_texture_bytes,
        mesh_upload_bytes: stats.mesh_upload_bytes, texture_upload_bytes: stats.texture_upload_bytes,
        colliders: activeColliders, collision_complete: collisionComplete === 1, invalid_colliders: invalidColliders,
        overflow_colliders: overflowColliders, sensor_complete: sensorComplete === 1, sensor_pairs: sensorPairs,
        sensor_candidates: sensorCandidates, query_calls: queryCalls, query_hits: queryHits, query_errors: queryErrors },
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
