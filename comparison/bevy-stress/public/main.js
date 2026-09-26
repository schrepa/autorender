import init from "./pkg/bevy_stress_comparison.js";

const baseline = {
  seed: 1337, renderables: 90000, moving_percent: 0, hierarchy: false,
  interpolation: false, colliders: 0, sensor_percent: 0,
  dense_colliders: false, queries_per_tick: 0, mesh_variety: 32,
  textured: true, culling: true,
};
const presets = {
  visual: baseline,
  transforms: { ...baseline, renderables: 50000, moving_percent: 100, hierarchy: true, interpolation: true, mesh_variety: 1, textured: false },
  moving90k: { ...baseline, moving_percent: 100, hierarchy: true, interpolation: true, mesh_variety: 8 },
};
const $ = (id) => document.getElementById(id);
const notice = (message, error = false) => {
  $("notice").textContent = message;
  $("notice").classList.toggle("error", error);
};

let config = baseline;
try {
  const raw = new URLSearchParams(location.search).get("config");
  if (raw) config = { ...baseline, ...JSON.parse(raw) };
  if (config.colliders || config.queries_per_tick || config.dense_colliders) {
    throw new Error("Bevy-only comparison requires colliders=0, queries_per_tick=0, dense_colliders=false.");
  }
} catch (error) {
  config = baseline;
  history.replaceState(null, "", location.pathname);
  notice(`Invalid configuration: ${error}`, true);
}
$("details").textContent = `${config.renderables.toLocaleString()} objects · ${config.moving_percent}% moving · ${config.mesh_variety} meshes/textures · hierarchy ${config.hierarchy ? "on" : "off"} · interpolation ${config.interpolation ? "on" : "off"}`;
for (const button of document.querySelectorAll("[data-preset]")) {
  button.addEventListener("click", () => {
    const next = presets[button.dataset.preset];
    location.search = `?config=${encodeURIComponent(JSON.stringify(next))}`;
  });
}
for (const button of document.querySelectorAll("[data-view]")) {
  button.addEventListener("click", () => {
    const url = new URL(location.href);
    url.searchParams.set("view", button.dataset.view);
    location.href = url.href;
  });
}

const canvas = $("bevy-canvas");
canvas.addEventListener("pointerdown", () => canvas.focus());
function updateViewport() {
  $("viewport").textContent = `${canvas.width}×${canvas.height} drawing buffer · DPR ${devicePixelRatio.toFixed(2)}`;
}
setInterval(updateViewport, 1000);
updateViewport();

const lastIntervals = [];
let previous = null;
let recording = null;
function percentile(sorted, p) {
  const x = (sorted.length - 1) * p;
  const lo = Math.floor(x);
  return sorted[lo] + (sorted[Math.ceil(x)] - sorted[lo]) * (x - lo);
}
function finishRecording(rec) {
  recording = null;
  $("record").disabled = false;
  const sorted = [...rec.intervals].sort((a, b) => a - b);
  const elapsed = rec.endedAt - rec.startedAt;
  const fps = rec.intervals.length * 1000 / elapsed;
  const report = {
    kind: "bevy-stress-comparison-v1",
    engine: "Bevy 0.19.1",
    config,
    browser: navigator.userAgent,
    viewport: { canvasWidth: canvas.width, canvasHeight: canvas.height, dpr: devicePixelRatio },
    view: new URLSearchParams(location.search).get("view") || "overview",
    startedAt: new Date(rec.startedWall).toISOString(),
    durationMs: elapsed,
    frames: rec.intervals.length,
    fps,
    intervalMs: { median: percentile(sorted, .5), p95: percentile(sorted, .95), p99: percentile(sorted, .99) },
    intervalsMs: rec.intervals,
    scope: "browser requestAnimationFrame cadence; no CPU/GPU phase instrumentation",
  };
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `bevy-stress-${new Date().toISOString().replaceAll(":", "-")}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
  notice(`Recorded ${fps.toFixed(1)} FPS over ${(elapsed / 1000).toFixed(1)} s. JSON downloaded.`);
}
$("record").addEventListener("click", () => {
  if (recording) return;
  recording = { warmup: 30, intervals: [], startedAt: 0, startedWall: 0, endedAt: 0 };
  $("record").disabled = true;
  notice("Warming up 30 frames…");
});
document.addEventListener("visibilitychange", () => {
  if (document.hidden && recording) {
    recording = null;
    $("record").disabled = false;
    notice("Recording aborted because the tab was hidden.", true);
  }
  previous = null;
});
function sample(now) {
  requestAnimationFrame(sample);
  if (previous !== null) {
    const interval = now - previous;
    if (interval > 0 && interval < 2000) {
      lastIntervals.push(interval);
      if (lastIntervals.length > 120) lastIntervals.shift();
      if (recording) {
        if (recording.warmup > 0) {
          recording.warmup--;
          if (recording.warmup === 0) {
            recording.startedAt = now;
            recording.startedWall = Date.now();
            notice("Recording 10 seconds…");
          }
        } else {
          recording.intervals.push(interval);
          recording.endedAt = now;
          if (now - recording.startedAt >= 10000) finishRecording(recording);
        }
      }
    }
  }
  previous = now;
}
requestAnimationFrame(sample);
setInterval(() => {
  if (!lastIntervals.length) return;
  const total = lastIntervals.reduce((sum, ms) => sum + ms, 0);
  const mean = total / lastIntervals.length;
  $("fps").textContent = `${(1000 / mean).toFixed(1)} FPS`;
  $("ms").textContent = `${mean.toFixed(1)} ms / frame`;
}, 500);

try {
  await init();
  $("record").disabled = false;
  notice("Ready. Wait for the scene to settle, then record.");
} catch (error) {
  notice(`Bevy failed to start: ${error}`, true);
  console.error(error);
}
