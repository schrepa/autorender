import { adapterFound } from "../engine/adapter";
import { playGame } from "../engine/game";
import type { Loading, Stopped } from "../engine/host";

function el<T extends HTMLElement = HTMLElement>(id: string): T {
  return document.getElementById(id) as T;
}

function shown() {
  const loading = el("loading");
  loading.classList.add("gone");
  setTimeout(() => { loading.hidden = true; }, 240);
}

function progress({ readiness, pipelines }: Loading) {
  const track = el("loading-track");
  if (readiness.compiling > 0) {
    el("loading-stage").textContent = "COMPILING SHADERS";
    el("loading-detail").textContent = `${pipelines} / ${pipelines + readiness.compiling}`;
    delete track.dataset.unknown;
    el("loading-bar").style.setProperty("--done", String(pipelines / Math.max(1, pipelines + readiness.compiling)));
  } else {
    el("loading-stage").textContent = "QUICKSILVER";
    el("loading-detail").textContent = "";
    track.dataset.unknown = "";
  }
}

function refuseGpu(reason: string) {
  el("loading").hidden = true;
  el("gpu-reason").textContent = reason;
  el("gpu").hidden = false;
}

function failed({ title, summary, detail, error }: Stopped) {
  el("loading").hidden = true;
  el("failure-title").textContent = title;
  el("failure-summary").textContent = summary;
  el("failure-code").textContent = error?.code ?? "";
  el("failure-path").textContent = error?.path ?? "";
  el("failure-detail").textContent = detail;
  el("failure").hidden = false;
}

el("failure-reload").addEventListener("click", () => location.reload());

const dev = new URLSearchParams(location.search).get("engine_dev") === "1";
if (dev) document.body.dataset.dev = "";

if (!window.isSecureContext) {
  refuseGpu("WebGPU requires HTTPS or localhost.");
} else if (!("gpu" in navigator)) {
  refuseGpu("This browser does not offer WebGPU.");
} else {
  adapterFound().then(found => found
    ? playGame({
      wasm: "quicksilver",
      scene: "quicksilver.scene.json",
      load: () => import("../../pkg/quicksilver"),
      game: bindings => bindings.QuicksilverGame,
      shown,
      failed,
      progress,
    })
    : refuseGpu("No WebGPU graphics adapter was returned."));
}
