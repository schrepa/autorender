import { loadGame } from "../engine/bindings";
import { startGame } from "../engine/host";
import { runSmokes, smokeSelection, type SmokeRun } from "../engine/smokes";
import { loadSceneFiles, loadSceneSource } from "../engine/source";
import { PLAYER_PARAMETERS } from "./parameters";
import JobWorker from "./worker.ts?worker";

const params = new URLSearchParams(location.search);
const sourceUrl = params.get("source") ?? "";
const playerBindings = () => loadGame("engine_player", () => import("../../pkg/engine_player"), JobWorker);

function checkedScene(path: string, base: string): string {
  const url = new URL(path, base);
  if (url.origin !== location.origin || !url.pathname.endsWith(".scene.json")) {
    throw new Error(`${path}: expected a .scene.json URL of ${location.origin}`);
  }
  return url.href;
}

const checkedSource = () => checkedScene(sourceUrl, location.href);

const loadScene = (signal: AbortSignal) => loadSceneSource(checkedSource(), signal);
const loadFiles = async (source: string, signal: AbortSignal) => loadSceneFiles(checkedSource(), source, await playerBindings(), signal);

const smoke = smokeSelection();
if (smoke !== null) {
  void runSmokes(smoke, async () => {
    const { PlayerGame } = await playerBindings();
    const exported = PlayerGame as unknown as { smokes?: SmokeRun };
    return exported.smokes?.bind(PlayerGame);
  });
} else {
  startGame({
    parameters: PLAYER_PARAMETERS,
    source: { url: sourceUrl, load: loadScene, files: loadFiles },
    async create(canvas, signal) {
      const { PlayerGame } = await playerBindings();
      const source = await loadScene(signal);
      const files = await loadFiles(source, signal);
      signal.throwIfAborted();
      return PlayerGame.create(canvas, source, files, params.get("without") ?? "");
    },
    async reload(game, signal) {
      const source = await loadScene(signal);
      const files = await loadFiles(source, signal);
      signal.throwIfAborted();
      game.reload_scene(source, files);
    },
    async level(game, path, signal, progress) {
      const url = checkedScene(path, checkedSource());
      const source = await loadSceneSource(url, signal);
      const files = await loadSceneFiles(url, source, await playerBindings(), signal, progress);
      signal.throwIfAborted();
      game.reload_scene(source, files);
    },
    poll(game) {
      document.getElementById("scene")!.dataset.progress = game.scene_progress();
    },
  });
}
