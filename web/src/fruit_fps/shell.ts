import type { Readiness } from "../generated/stats";

interface Preparing { readiness: Readiness; pipelines: number }
interface Failure { code: string; path: string; rest: Record<string, unknown> }
type Layer = "loading" | "gpu" | "failure" | "playing";

function part<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`fruit_fps.html has no #${id}`);
  return found as T;
}

const layers = { loading: part("loading"), gpu: part("gpu"), failure: part("failure") };
const stage = part("loading-stage");
const track = part("loading-track");
const bar = part("loading-bar");
const detail = part("loading-detail");
const asked = new Set<string>();
const fetched = new Set<string>();
let bytes = 0;
let preparing: Preparing | undefined;
let layer: Layer = "loading";

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

function progress(title: string, done: number, total: number, text: string) {
  stage.textContent = title;
  track.toggleAttribute("data-unknown", total <= 0);
  bar.style.setProperty("--done", String(total > 0 ? Math.min(1, done / total) : 0));
  detail.textContent = text;
}

function paint() {
  if (layer !== "loading") return;
  const ready = preparing?.readiness;
  const files = `${fetched.size} of ${asked.size} files · ${(bytes / 1048576).toFixed(1)} MB`;
  if (ready && ready.loading.total > 0) {
    progress("Loading the world", ready.loading.done, ready.loading.total, `${ready.loading.done} of ${ready.loading.total} loads`);
  } else if (ready && ready.compiling > 0) {
    const made = preparing?.pipelines ?? 0;
    progress("Compiling shaders", made, made + ready.compiling, `${ready.compiling} pipelines compiling · ${made} ready`);
  } else if (ready) {
    progress("Preparing the first frame", 0, 0, ready.unsettled.length > 0 ? `${ready.unsettled.length} parts settling` : "");
  } else if (fetched.size < asked.size) {
    progress("Downloading", fetched.size, asked.size, files);
  } else if (asked.size > 0) {
    progress("Starting the engine", 0, 0, files);
  } else {
    progress("Opening", 0, 0, "");
  }
}

function countFiles() {
  const fetching = window.fetch.bind(window);
  window.fetch = (input: RequestInfo | URL, init?: RequestInit) => {
    asked.add(input instanceof Request ? input.url : new URL(input, location.href).href);
    paint();
    return fetching(input, init);
  };
  new PerformanceObserver(list => {
    for (const entry of list.getEntries() as PerformanceResourceTiming[]) {
      if (!asked.has(entry.name) || fetched.has(entry.name)) continue;
      fetched.add(entry.name);
      bytes += entry.encodedBodySize;
    }
    paint();
  }).observe({ type: "resource" });
  new PerformanceObserver(list => {
    for (const entry of list.getEntriesByName("engine-preparing")) {
      const found = (entry as PerformanceMark).detail as Partial<Preparing> | null;
      if (found?.readiness) preparing = { readiness: found.readiness, pipelines: found.pipelines ?? 0 };
    }
    paint();
  }).observe({ type: "mark" });
}

function failure(line: string | undefined): Failure | undefined {
  if (!line) return undefined;
  try {
    const parsed: unknown = JSON.parse(line);
    if (typeof parsed !== "object" || parsed === null || !("code" in parsed)) return undefined;
    const { code, path, ...rest } = parsed as Record<string, unknown>;
    return { code: String(code), path: typeof path === "string" ? path : "—", rest };
  } catch {
    return undefined;
  }
}

function failed(status: HTMLElement, message: HTMLElement) {
  const [summary = "", ...lines] = (message.textContent ?? "").split("\n");
  const found = failure(lines[0]);
  const title = part("failure-title");
  title.textContent = `${title.dataset.name} stopped: ${(status.textContent ?? "").trim() || "error"}`;
  part("failure-summary").textContent = summary;
  part("failure-code").textContent = found?.code ?? "—";
  part("failure-path").textContent = found?.path ?? "—";
  const rest = found ? Object.fromEntries(Object.entries(found.rest).filter(([, value]) => value !== null && value !== undefined)) : undefined;
  part("failure-detail").textContent = rest ? (Object.keys(rest).length > 0 ? JSON.stringify(rest, null, 2) : "") : lines.join("\n");
  show("failure");
}

export function refuseGpu(reason: string) {
  part("gpu-reason").textContent = reason;
  show("gpu");
}

export function watchStart() {
  countFiles();
  paint();
  const canvas = part<HTMLCanvasElement>("scene");
  const status = part("status");
  const message = part("message");
  new MutationObserver(() => {
    if (canvas.style.visibility !== "hidden" && layer === "loading") show("playing");
  }).observe(canvas, { attributes: true, attributeFilter: ["style"] });
  new MutationObserver(() => {
    if (status.dataset.state === "error") failed(status, message);
  }).observe(status, { attributes: true, attributeFilter: ["data-state"] });
  part("failure-reload").addEventListener("click", () => location.reload());
}
