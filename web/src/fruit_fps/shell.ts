import type { Loading, Stopped } from "../engine/host";

type Layer = "loading" | "gpu" | "failure" | "playing";

export function part<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`fruit_fps.html has no #${id}`);
  return found as T;
}

const layers = { loading: part("loading"), gpu: part("gpu"), failure: part("failure") };
const stage = part("loading-stage");
const track = part("loading-track");
const bar = part("loading-bar");
const detail = part("loading-detail");
let layer: Layer = "loading";

part("failure-reload").addEventListener("click", () => location.reload());

function show(next: Layer) {
  if (layer === "gpu" || layer === next) return;
  const fading = layer === "loading" && next === "playing";
  layer = next;
  for (const [name, element] of Object.entries(layers)) {
    if (!(fading && name === "loading")) element.hidden = name !== next;
  }
  if (!fading) return;
  layers.loading.classList.add("gone");
  setTimeout(() => { if (layer === "playing") layers.loading.hidden = true; }, 320);
}

function paint(title: string, done: number, total: number, text: string) {
  stage.textContent = title;
  track.toggleAttribute("data-unknown", total <= 0);
  bar.style.setProperty("--done", String(total > 0 ? Math.min(1, done / total) : 0));
  detail.textContent = text;
}

export function progress({ readiness, pipelines }: Loading) {
  if (layer !== "loading") return;
  const first = readiness.groups[0];
  if (first?.stage === "arriving") {
    paint("Downloading", first.files_in, first.files, `${first.files_in} of ${first.files} files · ${(first.bytes_in / 1048576).toFixed(1)} MB`);
  } else if (readiness.loading.total > 0) {
    paint("Loading the world", readiness.loading.done, readiness.loading.total, `${readiness.loading.done} of ${readiness.loading.total} loads`);
  } else if (readiness.compiling > 0) {
    paint("Compiling shaders", pipelines, pipelines + readiness.compiling, `${readiness.compiling} pipelines compiling · ${pipelines} ready`);
  } else {
    paint("Preparing the first frame", 0, 0, readiness.unsettled.length > 0 ? `${readiness.unsettled.length} parts settling` : "");
  }
}

export function shown() {
  if (layer === "loading") show("playing");
}

export function failed({ title, summary, detail, error }: Stopped) {
  const heading = part("failure-title");
  heading.textContent = `${heading.dataset.name} stopped: ${title}`;
  part("failure-summary").textContent = summary;
  part("failure-code").textContent = error?.code ?? "—";
  part("failure-path").textContent = error?.path ?? "—";
  const rest = error && Object.entries(error).filter(([key, value]) => key !== "code" && key !== "path" && value !== null && value !== undefined);
  part("failure-detail").textContent = rest ? (rest.length > 0 ? JSON.stringify(Object.fromEntries(rest), null, 2) : "") : detail;
  show("failure");
}

export function refuseGpu(reason: string) {
  part("gpu-reason").textContent = reason;
  show("gpu");
}
