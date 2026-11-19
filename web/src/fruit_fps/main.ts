import { adapterFeatures } from "../engine/adapter";
import { playGame } from "../engine/game";
import { failed, progress, refuseGpu, shown } from "./shell";

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

function interruptions() {
  document.addEventListener("pointerdown", event => {
    if (event.button === 0 && event.target instanceof HTMLCanvasElement) immerse();
  });
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
  interruptions();
  adapterFeatures().then(
    features => playGame({ wasm: "fruit_fps", scene: features.includes("texture-compression-bc") ? "fruit_fps.scene.json" : "fruit_fps.rgba8.scene.json", load: () => import("../../pkg/fruit_fps"), game: bindings => bindings.FruitFpsGame, shown, failed, progress }),
    () => refuseGpu("This browser has WebGPU but gave no graphics adapter: hardware acceleration may be off, or the GPU or its driver is blocked. chrome://gpu (Chrome, Edge) or about:support (Firefox) shows why."),
  );
}
