import { playgroundBindings } from "./bindings";
import { boot } from "./host";
import { loadSceneFiles, loadSceneSource } from "./source";

const params = new URLSearchParams(location.search);
const selected = import.meta.env.DEV ? params.get("scene") : null;
// ?taa=on antialiases ordinary frames temporally.
const taa = params.get("taa") === "on";
if (selected && !/^\/scenes\/[A-Za-z0-9_-]+\.scene\.json$/.test(selected)) throw new Error("Development scene must be a /scenes/<name>.scene.json path.");
const sourceUrl = selected ?? "/scenes/playground.scene.json";
const loadScene = (signal: AbortSignal) => loadSceneSource(sourceUrl, signal);
const loadFiles = async (source: string, signal: AbortSignal) => loadSceneFiles(sourceUrl, source, await playgroundBindings(), signal);

boot({
  source: { url: sourceUrl, load: loadScene, files: loadFiles },
  async create(canvas, _count, signal) {
    const { default: init, PlaygroundGame } = await playgroundBindings();
    await init();
    const source = await loadScene(signal);
    const files = await loadFiles(source, signal);
    signal.throwIfAborted();
    const game = await PlaygroundGame.create(canvas, source, files);
    game.set_temporal_antialiasing(taa);
    return game;
  },
  async reload(game, signal) {
    const source = await loadScene(signal);
    const files = await loadFiles(source, signal);
    signal.throwIfAborted();
    game.reload_scene(source, files);
  },
  poll(game) {
    const progress = JSON.parse(game.progress());
    const score = document.getElementById("score")!;
    score.textContent = `${progress.collected} / ${progress.total}`;
    document.getElementById("objective")!.textContent = progress.collected === progress.total ? "Garden cleared. Reload to play again." : "Collect every crystal in the garden.";
    const canvas = document.getElementById("scene")!;
    canvas.dataset.progress = JSON.stringify(progress);
    document.getElementById("revision")!.textContent = `SCENE REVISION ${progress.revision}`;
  },
});
