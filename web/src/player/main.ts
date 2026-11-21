import { adapterFound, warmAdapter } from "../engine/adapter";
import { requireWasmSimd } from "../engine/bindings";
import { type Runtime, startGame } from "../engine/host";
import { feedSceneFiles, loadSceneSource } from "../engine/source";
import { failed, part, progress, refuseGpu, setGameName, shown } from "./shell";

interface KeyboardLock { lock(codes?: string[]): Promise<void>; unlock(): void }

interface OrientationLock {
  lock?(orientation: "landscape" | "portrait"): Promise<void>;
  unlock?(): void;
}

type PreferredOrientation = "landscape" | "portrait" | "any";

interface GameEntry {
  id: string;
  title: string;
  module: string;
  scene: string;
  export?: string;
  wasm?: string;
  preferred_orientation?: PreferredOrientation;
}

interface GameCatalog {
  default?: string;
  games: GameEntry[];
}

interface PlayableClass {
  create(canvas: HTMLCanvasElement, scene: string): Promise<Runtime>;
}

interface GameModule extends Record<string, unknown> {
  default(input?: unknown): Promise<unknown>;
}

const SCENE_SUFFIX = ".scene.json";
const keyboard = (navigator as Navigator & { keyboard?: KeyboardLock }).keyboard;
const orientationApi = (screen as Screen & { orientation?: OrientationLock }).orientation;
let preferredOrientation: PreferredOrientation = "any";
let orientationLocked = false;

function lockPreferredOrientation() {
  if (preferredOrientation !== "landscape" && preferredOrientation !== "portrait") return;
  if (typeof orientationApi?.lock !== "function") return;
  void orientationApi.lock(preferredOrientation).then(
    () => { orientationLocked = true; },
    () => undefined,
  );
}

function unlockOrientation() {
  if (!orientationLocked) return;
  orientationLocked = false;
  try { orientationApi?.unlock?.(); } catch {}
}

function immerse(player: HTMLElement) {
  if (document.fullscreenElement) return;
  const root = document.documentElement;
  if (typeof root.requestFullscreen !== "function") {
    player.dataset.view = player.dataset.view === "immersive" ? "window" : "immersive";
    return;
  }
  const entered = root.requestFullscreen({ navigationUI: "hide" });
  void entered.then(
    () => {
      lockPreferredOrientation();
      void keyboard?.lock().catch(() => undefined);
    },
    () => {
      player.dataset.view = "immersive";
    },
  );
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
    immerse(player);
    scene.focus();
  });
  part("view-exit").addEventListener("click", () => {
    player.dataset.view = "window";
    theater.setAttribute("aria-pressed", "false");
    scene.focus();
  });
}

function interruptions() {
  document.addEventListener("fullscreenchange", () => {
    if (document.fullscreenElement) return;
    unlockOrientation();
    keyboard?.unlock();
    if (document.pointerLockElement && typeof document.exitPointerLock === "function") {
      document.exitPointerLock();
    }
  });
  window.addEventListener("beforeunload", event => {
    if (document.pointerLockElement || document.fullscreenElement) event.preventDefault();
  });
  followRatio();
}

function sameOrigin(path: string, suffix?: string): string {
  const url = new URL(path, location.href);
  if (url.origin !== location.origin) {
    throw new Error(`${path}: expected a same-origin URL under ${location.origin}`);
  }
  if (suffix && !url.pathname.endsWith(suffix)) {
    throw new Error(`${path}: expected a ${suffix} URL`);
  }
  return url.href;
}

function isPlayable(value: unknown): value is PlayableClass {
  return (typeof value === "function" || (typeof value === "object" && value !== null))
    && typeof (value as { create?: unknown }).create === "function";
}

function findPlayable(bindings: GameModule, explicit?: string): PlayableClass {
  if (explicit) {
    const named = bindings[explicit];
    if (isPlayable(named)) return named;
    throw new Error(`game module exports no playable class named ${explicit}`);
  }
  for (const [key, value] of Object.entries(bindings)) {
    if (key !== "default" && key !== "initSync" && isPlayable(value)) return value;
  }
  throw new Error("game module exports no class with static create(canvas, scene)");
}

function loadModule(url: string): Promise<GameModule> {
  return import(url) as Promise<GameModule>;
}

async function loadCatalog(): Promise<GameCatalog> {
  const url = new URL("games.json", location.href).href;
  let lastError: unknown;
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      const response = await fetch(url, { cache: "no-store" });
      if (!response.ok) throw new Error(`games.json returned HTTP ${response.status}`);
      const data = await response.json() as Partial<GameCatalog>;
      if (!Array.isArray(data.games) || data.games.length === 0) {
        throw new Error("games.json lists no games");
      }
      return { default: data.default, games: data.games };
    } catch (error) {
      lastError = error;
      if (attempt === 0) await new Promise(resolve => setTimeout(resolve, 350));
    }
  }
  throw lastError;
}

function setupCatalogUi(games: GameEntry[], active: GameEntry) {
  preferredOrientation = active.preferred_orientation ?? "any";
  setGameName(active.title);
  if (games.length <= 1) return;
  const title = part("game-title");
  const select = part<HTMLSelectElement>("game-select");
  select.replaceChildren(...games.map(entry => {
    const option = document.createElement("option");
    option.value = entry.id;
    option.textContent = entry.title;
    option.selected = entry.id === active.id;
    return option;
  }));
  title.hidden = true;
  select.hidden = false;
  select.addEventListener("change", () => {
    const next = new URL(location.href);
    next.searchParams.set("game", select.value);
    next.searchParams.delete("scene");
    location.assign(next.href);
  });
}

function settled<T>(promise: Promise<T>): Promise<T> {
  promise.catch(() => undefined);
  return promise;
}

function launch(entry: GameEntry, sceneOverride: string | null) {
  const moduleUrl = sameOrigin(entry.module);
  const wasmUrl = entry.wasm ? sameOrigin(entry.wasm) : undefined;
  const sceneUrl = sameOrigin(sceneOverride ?? entry.scene, SCENE_SUFFIX);
  const loaded = settled((async () => {
    requireWasmSimd();
    const bindings = await loadModule(moduleUrl);
    if (typeof bindings.default !== "function") throw new Error(`${entry.module} has no default initializer`);
    if (wasmUrl) await bindings.default({ module_or_path: wasmUrl });
    else await bindings.default();
    return findPlayable(bindings, entry.export);
  })());
  const source = {
    url: sceneUrl,
    load: (signal: AbortSignal) => loadSceneSource(sceneUrl, signal),
    feed: (running: Runtime, signal: AbortSignal, shown?: Promise<void>) => feedSceneFiles(running, sceneUrl, signal, shown),
  };
  const opening = new AbortController();
  const firstText = settled(Promise.resolve().then(() => source.load(opening.signal)));
  startGame<Runtime>({
    source,
    shown,
    failed,
    progress,
    async create(canvas, signal) {
      signal.addEventListener("abort", () => opening.abort(signal.reason), { once: true });
      const played = await loaded;
      const text = await firstText;
      signal.throwIfAborted();
      const running = await played.create(canvas, text);
      running.set_file_root(new URL(".", sceneUrl).href);
      return running;
    },
  });
}

async function boot() {
  const params = new URLSearchParams(location.search);
  if (params.get("engine_dev") === "1") document.body.dataset.dev = "";
  views();
  interruptions();
  warmAdapter();
  let catalog: GameCatalog;
  try {
    catalog = await loadCatalog();
  } catch (error) {
    failed({ title: "Catalog error", summary: "Could not load games.json.", detail: String(error) });
    return;
  }
  const wanted = params.get("game");
  const entry = wanted
    ? catalog.games.find(item => item.id === wanted)
    : catalog.games.find(item => item.id === catalog.default) ?? catalog.games[0];
  if (!entry) {
    failed({ title: "Unknown game", summary: `No game with id "${wanted}" in games.json.`, detail: catalog.games.map(item => item.id).join(", ") });
    return;
  }
  setupCatalogUi(catalog.games, entry);
  if (!window.isSecureContext) {
    refuseGpu("This page is not served over HTTPS or from localhost, and browsers only offer WebGPU to secure pages.");
    return;
  }
  if (!("gpu" in navigator)) {
    refuseGpu("This browser does not offer WebGPU, or it is switched off.");
    return;
  }
  const found = await adapterFound();
  if (!found) {
    refuseGpu("This browser has WebGPU but gave no graphics adapter: hardware acceleration may be off, or the GPU or its driver is blocked.");
    return;
  }
  try {
    launch(entry, params.get("scene"));
  } catch (error) {
    failed({ title: "Launch error", summary: "Could not start the selected game.", detail: String(error) });
  }
}

void boot();
