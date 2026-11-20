import { adapterFound } from "../engine/adapter";
import { playGame } from "../engine/game";
import { failed, part, progress, refuseGpu, shown } from "./shell";

interface KeyboardLock { lock(codes?: string[]): Promise<void>; unlock(): void }

const keyboard = (navigator as Navigator & { keyboard?: KeyboardLock }).keyboard;

function immerse() {
  if (document.fullscreenElement) return;
  const entered = document.documentElement.requestFullscreen({ navigationUI: "hide" });
  void entered.then(() => keyboard?.lock()).catch(() => undefined);
}

function followRatio() {
  matchMedia(`(resolution: ${window.devicePixelRatio}dppx)`).addEventListener("change", () => {
    window.dispatchEvent(new Event("resize"));
    followRatio();
  }, { once: true });
}

function views() {
  const player = part("player");
  const scene = part("scene");
  const theater = part("view-theater");
  theater.addEventListener("click", () => {
    const wide = player.dataset.view !== "theater";
    player.dataset.view = wide ? "theater" : "window";
    theater.setAttribute("aria-pressed", String(wide));
    scene.focus();
  });
  part("view-full").addEventListener("click", () => {
    immerse();
    scene.focus();
  });
}

const BUCKET_MS = 100;
const BUCKETS = 300;
const SPIKE_MS = 25;
const CEILING_MS = 50;

function frameRate() {
  const shown = part("rate");
  const scene = part<HTMLCanvasElement>("scene");
  const pace = part<HTMLCanvasElement>("pace");
  const ink = pace.getContext("2d");
  const worsts = new Float32Array(BUCKETS);
  const spikes = new Uint16Array(BUCKETS);
  let card = "";
  void navigator.gpu.requestAdapter({ powerPreference: "high-performance" }).then(adapter => {
    const info = adapter?.info;
    card = [info?.vendor, info?.architecture].filter(Boolean).join(" ") || info?.description || "";
  }, () => undefined);
  let since = 0;
  let last = 0;
  let frames = 0;
  let worst = 0;
  let bucket = 0;
  const draw = () => {
    if (!ink) return;
    const { width, height } = pace;
    ink.clearRect(0, 0, width, height);
    ink.fillStyle = "#FFFFFF40";
    for (const line of [1000 / 60, 2000 / 60]) ink.fillRect(0, height - Math.round(line / CEILING_MS * height), width, 1);
    for (let column = 0; column < BUCKETS; column += 1) {
      const gap = worsts[(bucket + 1 + column) % BUCKETS];
      if (gap <= 0) continue;
      const tall = Math.max(1, Math.round(Math.min(gap, CEILING_MS) / CEILING_MS * height));
      ink.fillStyle = gap >= SPIKE_MS ? "#FF6F59" : "#FFD23F";
      ink.fillRect(column * width / BUCKETS, height - tall, width / BUCKETS, tall);
    }
  };
  const frame = (now: number) => {
    const at = Math.floor(now / BUCKET_MS);
    if (last === 0) {
      since = now; frames = 0; worst = 0; bucket = at;
    } else {
      const gap = now - last;
      frames += 1;
      worst = Math.max(worst, gap);
      if (at !== bucket) {
        for (let fresh = Math.max(bucket + 1, at - BUCKETS + 1); fresh <= at; fresh += 1) { worsts[fresh % BUCKETS] = 0; spikes[fresh % BUCKETS] = 0; }
        bucket = at;
        draw();
      }
      worsts[at % BUCKETS] = Math.max(worsts[at % BUCKETS], gap);
      if (gap >= SPIKE_MS) spikes[at % BUCKETS] += 1;
    }
    last = now;
    if (now - since >= 1000) {
      const longest = worsts.reduce((most, gap) => Math.max(most, gap), 0);
      const count = spikes.reduce((sum, some) => sum + some, 0);
      shown.textContent = `${Math.round(frames * 1000 / (now - since))} fps · worst ${worst.toFixed(1)} ms · 30 s: worst ${Math.round(longest)} ms, ${count} over ${SPIKE_MS} ms · ${scene.width}×${scene.height}${card ? ` · ${card}` : ""}`;
      since = now; frames = 0; worst = 0;
    }
    requestAnimationFrame(frame);
  };
  document.addEventListener("visibilitychange", () => { last = 0; });
  requestAnimationFrame(frame);
}

function interruptions() {
  document.addEventListener("fullscreenchange", () => {
    if (document.fullscreenElement) return;
    keyboard?.unlock();
    if (document.pointerLockElement) document.exitPointerLock();
  });
  window.addEventListener("beforeunload", event => {
    if (document.pointerLockElement || document.fullscreenElement) event.preventDefault();
  });
  followRatio();
}

if (new URLSearchParams(location.search).get("engine_dev") === "1") document.body.dataset.dev = "";

if (!window.isSecureContext) {
  refuseGpu("This page is not served over HTTPS or from localhost, and browsers only offer WebGPU to secure pages.");
} else if (!("gpu" in navigator)) {
  refuseGpu("This browser does not offer WebGPU, or it is switched off.");
} else {
  views();
  frameRate();
  interruptions();
  adapterFound().then(found => found
    ? playGame({ wasm: "fruit_fps", scene: "fruit_fps.scene.json", load: () => import("../../pkg/fruit_fps"), game: bindings => bindings.FruitFpsGame, shown, failed, progress })
    : refuseGpu("This browser has WebGPU but gave no graphics adapter: hardware acceleration may be off, or the GPU or its driver is blocked. chrome://gpu (Chrome, Edge) or about:support (Firefox) shows why."));
}
