import clients from "../generated/clients.json";
import { type GameBindings, loadGame } from "./bindings";
import { type Runtime, startGame } from "./host";
import type { JobWorkerClass } from "./jobs";
import { fetchBytes, loadSceneFiles, loadSceneSource } from "./source";

declare const __GAME_FILES__: string[];

const SCENE_SUFFIX = ".scene.json";

interface SceneLoader {
  take_reads(): { uri: string }[];
  read(uri: string, bytes: Uint8Array): boolean;
  image(uri: string, width: number, height: number, pixels: Uint8Array): void;
  decode(uri: string, bytes: Uint8Array): void;
  finish(): unknown;
  free(): void;
}

interface FileSet {
  add_buffer(uri: string, bytes: Uint8Array): void;
}

export interface PlayedBindings extends GameBindings {
  SceneFileSet: new () => FileSet;
  SceneLoad: new (source: string) => SceneLoader;
}

export interface Playable<T extends Runtime> {
  create(canvas: HTMLCanvasElement, query: string, scene?: string, files?: unknown, gameFiles?: unknown): Promise<T>;
  files(): string[];
}

export interface PlayedGame<T extends Runtime, B extends PlayedBindings> {
  wasm: string;
  scene?: string;
  load(): Promise<B>;
  worker: JobWorkerClass;
  game(bindings: B): Playable<T>;
}

function checkedScene(path: string, base: string): string {
  const url = new URL(path, base);
  if (url.origin !== location.origin || !url.pathname.endsWith(SCENE_SUFFIX)) {
    throw new Error(`${path}: expected a ${SCENE_SUFFIX} URL of ${location.origin}`);
  }
  return url.href;
}

async function gameFiles(set: FileSet, paths: string[], signal: AbortSignal): Promise<FileSet> {
  await Promise.all(paths.filter(path => __GAME_FILES__.includes(path)).map(async path => {
    set.add_buffer(path, await fetchBytes(new URL(`${import.meta.env.BASE_URL}${path}`, location.href).href, signal));
  }));
  return set;
}

export function playGame<T extends Runtime, B extends PlayedBindings>(game: PlayedGame<T, B>) {
  const bindings = () => loadGame(game.load, game.worker);
  const folder = clients.find(client => client.wasm === game.wasm)?.folder;
  const declared = import.meta.env.DEV && game.scene && folder ? `/_engine/source/${folder}/${game.scene}` : undefined;
  const url = new URLSearchParams(location.search).get("scene") ?? declared;
  const sceneUrl = () => checkedScene(url ?? "", location.href);
  const source = url === undefined ? undefined : {
    url,
    load: (signal: AbortSignal) => loadSceneSource(sceneUrl(), signal),
    files: async (text: string, signal: AbortSignal) => loadSceneFiles(sceneUrl(), text, await bindings(), signal),
  };
  startGame<T>({
    parameters: { scene: "PATH", without: "BRICK,..." },
    source,
    async create(canvas, signal) {
      const loaded = await bindings();
      const played = game.game(loaded);
      const text = source && await source.load(signal);
      const files = source && text !== undefined ? await source.files(text, signal) : undefined;
      const found = await gameFiles(new loaded.SceneFileSet(), played.files(), signal);
      signal.throwIfAborted();
      return played.create(canvas, location.search, text, files, found);
    },
    async level(running, path, signal, progress) {
      const target = checkedScene(path, url === undefined ? location.href : sceneUrl());
      const text = await loadSceneSource(target, signal);
      const files = await loadSceneFiles(target, text, await bindings(), signal, progress);
      signal.throwIfAborted();
      running.reload_scene(text, files);
    },
  });
}
