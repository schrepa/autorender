import { loadGame } from "../engine/bindings";
import { startGame } from "../engine/host";
import { loadSceneFiles, loadSceneSource } from "../engine/source";
import { PLAYER_PARAMETERS } from "./parameters";

const params = new URLSearchParams(location.search);
const sourceUrl = params.get("source") ?? "";
const playerBindings = () => loadGame("engine_player", () => import("../../pkg/engine_player"));

function checkedSource(): string {
  const url = new URL(sourceUrl, location.href);
  if (url.origin !== location.origin || !url.pathname.endsWith(".scene.json")) {
    throw new Error(`?source=${sourceUrl}: expected a .scene.json URL of ${location.origin}`);
  }
  return url.href;
}

const loadScene = (signal: AbortSignal) => loadSceneSource(checkedSource(), signal);
const loadFiles = async (source: string, signal: AbortSignal) => loadSceneFiles(checkedSource(), source, await playerBindings(), signal);

startGame({
  parameters: PLAYER_PARAMETERS,
  source: { url: sourceUrl, load: loadScene, files: loadFiles },
  async create(canvas, signal) {
    const { default: init, PlayerGame } = await playerBindings();
    await init();
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
  poll(game) {
    document.getElementById("scene")!.dataset.progress = game.scene_progress();
  },
});
