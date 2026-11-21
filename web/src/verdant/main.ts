import { adapterFound } from "../engine/adapter";
import { playGame } from "../engine/game";
import type { Loading, Stopped } from "../engine/host";

function part<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`verdant.html has no #${id}`);
  return found as T;
}

const layers = { loading: part("loading"), gpu: part("gpu"), failure: part("failure") };
const stage = part("loading-stage");
const track = part("loading-track");
const bar = part("loading-bar");
const detail = part("loading-detail");
let layer: "loading" | "gpu" | "failure" | "playing" = "loading";

part("failure-reload").addEventListener("click", () => location.reload());

function show(next: "loading" | "gpu" | "failure" | "playing") {
  if (layer === "gpu" || layer === next) return;
  const fading = layer === "loading" && next === "playing";
  layer = next;
  for (const [name, element] of Object.entries(layers)) {
    if (!(fading && name === "loading")) element.hidden = name !== next;
  }
  if (!fading) return;
  layers.loading.classList.add("gone");
  setTimeout(() => { if (layer === "playing") layers.loading.hidden = true; }, 300);
}

function progress({ readiness, pipelines }: Loading) {
  if (layer !== "loading") return;
  if (readiness.compiling > 0) {
    stage.textContent = "Compiling shaders";
    track.toggleAttribute("data-unknown", false);
    bar.style.setProperty("--done", String(pipelines / Math.max(1, pipelines + readiness.compiling)));
    detail.textContent = `${pipelines} / ${pipelines + readiness.compiling}`;
  } else {
    stage.textContent = "Preparing meadow";
    track.toggleAttribute("data-unknown", true);
    detail.textContent = "";
  }
}

function shown() {
  if (layer === "loading") show("playing");
}

function failed({ title, summary, detail: text, error }: Stopped) {
  part("failure-title").textContent = `Verdant stopped: ${title}`;
  part("failure-summary").textContent = summary;
  part("failure-detail").textContent = error ? JSON.stringify(error, null, 2) : text;
  show("failure");
}

function refuseGpu(reason: string) {
  part("gpu-reason").textContent = reason;
  show("gpu");
}

const dev = new URLSearchParams(location.search).get("engine_dev") === "1";
if (dev) document.body.dataset.dev = "";

if (!window.isSecureContext) {
  refuseGpu("This page requires HTTPS or localhost for WebGPU.");
} else if (!("gpu" in navigator)) {
  refuseGpu("This browser does not offer WebGPU.");
} else {
  void adapterFound().then(found => found
    ? playGame({
      wasm: "verdant",
      scene: "verdant.scene.json",
      load: () => import("../../pkg/verdant"),
      game: bindings => bindings.VerdantGame,
      shown,
      failed,
      progress,
    })
    : refuseGpu("No WebGPU graphics adapter was returned."));
}
