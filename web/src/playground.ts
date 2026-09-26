import { playgroundBindings } from "./bindings";
import { boot } from "./host";
import { loadSceneSource } from "./source";

const selected = import.meta.env.DEV ? new URLSearchParams(location.search).get("scene") : null;
if (selected && !/^\/scenes\/[A-Za-z0-9_-]+\.scene\.json$/.test(selected)) throw new Error("Development scene must be a /scenes/<name>.scene.json path.");
const sourceUrl = selected ?? "/scenes/playground.scene.json";
const loadScene = (signal: AbortSignal) => loadSceneSource(sourceUrl, signal);

boot({
  source: { url: sourceUrl, load: loadScene },
  async create(canvas, _count, signal) {
    const { default: init, PlaygroundGame } = await playgroundBindings();
    await init();
    const source = await loadScene(signal);
    signal.throwIfAborted();
    return PlaygroundGame.create(canvas, source);
  },
  async reload(game, signal) {
    const source = await loadScene(signal);
    signal.throwIfAborted();
    game.reload_scene(source);
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
