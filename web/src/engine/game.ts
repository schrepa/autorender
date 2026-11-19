import clients from "../generated/clients.json";
import { warmAdapter } from "./adapter";
import { type GameBindings, loadGame } from "./bindings";
import { type Fetched, type Runtime, type StartWatch, startGame } from "./host";
import { type SceneFileBindings, loadSceneFiles, loadSceneSource } from "./source";

const SCENE_SUFFIX = ".scene.json";

type PlayedBindings = GameBindings & SceneFileBindings;

export interface Playable<T extends Runtime> {
  create(canvas: HTMLCanvasElement, scene?: string, files?: unknown): Promise<T>;
}

export interface PlayedGame<T extends Runtime, B extends PlayedBindings> extends StartWatch {
  wasm: string;
  scene?: string;
  load(): Promise<B>;
  game(bindings: B): Playable<T>;
}

function checkedScene(path: string, base: string): string {
  const url = new URL(path, base);
  if (url.origin !== location.origin || !url.pathname.endsWith(SCENE_SUFFIX)) {
    throw new Error(`${path}: expected a ${SCENE_SUFFIX} URL of ${location.origin}`);
  }
  return url.href;
}

function settled<T>(promise: Promise<T>): Promise<T> {
  promise.catch(() => undefined);
  return promise;
}

export function playGame<T extends Runtime, B extends PlayedBindings>(game: PlayedGame<T, B>) {
  warmAdapter();
  const loading = settled(loadGame(game.load));
  const bindings = () => loading;
  const client = clients.find(client => client.wasm === game.wasm);
  const declared = game.scene && client ? import.meta.env.DEV ? `/_engine/source/${client.folder}/${game.scene}` : `${client.game}/${game.scene}` : undefined;
  const url = new URLSearchParams(location.search).get("scene") ?? declared;
  const sceneUrl = () => checkedScene(url ?? "", location.href);
  const source = url === undefined ? undefined : {
    url,
    load: (signal: AbortSignal) => loadSceneSource(sceneUrl(), signal),
    files: async (text: string, signal: AbortSignal, fetched?: (fetched: Fetched) => void) => loadSceneFiles(sceneUrl(), text, await bindings(), signal, fetched),
  };
  const opening = new AbortController();
  const firstText = source && settled(Promise.resolve().then(() => source.load(opening.signal)));
  startGame<T>({
    source,
    shown: game.shown,
    failed: game.failed,
    progress: game.progress,
    async create(canvas, signal, fetched) {
      signal.addEventListener("abort", () => opening.abort(signal.reason), { once: true });
      const played = game.game(await bindings());
      const text = await firstText;
      const files = source && text !== undefined ? await source.files(text, signal, fetched) : undefined;
      signal.throwIfAborted();
      const running = await played.create(canvas, text, files);
      if (url !== undefined) running.set_file_root(new URL(".", sceneUrl()).href);
      return running;
    },
  });
}
