import clients from "../generated/clients.json";
import { warmAdapter } from "./adapter";
import { type GameBindings, loadGame } from "./bindings";
import { type Runtime, type StartWatch, startGame } from "./host";
import { feedSceneFiles, loadSceneSource } from "./source";

const SCENE_SUFFIX = ".scene.json";

export interface Playable<T extends Runtime> {
  create(canvas: HTMLCanvasElement, scene: string): Promise<T>;
}

export interface PlayedGame<T extends Runtime, B extends GameBindings> extends StartWatch {
  wasm: string;
  scene: string;
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

export function playGame<T extends Runtime, B extends GameBindings>(game: PlayedGame<T, B>) {
  warmAdapter();
  const loading = settled(loadGame(game.load));
  const client = clients.find(client => client.wasm === game.wasm);
  const declared = client ? import.meta.env.DEV ? `/_engine/source/${client.folder}/${game.scene}` : `${client.game}/${game.scene}` : "";
  const url = new URLSearchParams(location.search).get("scene") ?? declared;
  const sceneUrl = () => checkedScene(url, location.href);
  const source = {
    url,
    load: (signal: AbortSignal) => loadSceneSource(sceneUrl(), signal),
    feed: (running: Runtime, signal: AbortSignal, shown?: Promise<void>) => feedSceneFiles(running, sceneUrl(), signal, shown),
  };
  const opening = new AbortController();
  const firstText = settled(Promise.resolve().then(() => source.load(opening.signal)));
  startGame<T>({
    source,
    shown: game.shown,
    failed: game.failed,
    progress: game.progress,
    async create(canvas, signal) {
      signal.addEventListener("abort", () => opening.abort(signal.reason), { once: true });
      const played = game.game(await loading);
      const text = await firstText;
      signal.throwIfAborted();
      const running = await played.create(canvas, text);
      running.set_file_root(new URL(".", sceneUrl()).href);
      return running;
    },
  });
}
