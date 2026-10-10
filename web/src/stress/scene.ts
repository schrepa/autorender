import { stressBindings } from "./bindings";
import { startGame } from "../engine/host";
import { loadSceneFiles, loadSceneSource } from "../engine/source";
import "./stress.css";

// An authored scene in place of a workload, reloaded by revision through the
// development tools.
const params = new URLSearchParams(location.search);
const sourceUrl = params.get("source") ?? "";
if (!/^\/scenes\/[A-Za-z0-9_-]+\.scene\.json$/.test(sourceUrl)) throw new Error("The scene source must be a /scenes/<name>.scene.json path.");
document.body.dataset.mode = "scene";
const loadScene = (signal: AbortSignal) => loadSceneSource(sourceUrl, signal);
const loadFiles = async (source: string, signal: AbortSignal) => loadSceneFiles(sourceUrl, source, await stressBindings(), signal);

startGame({
  source: { url: sourceUrl, load: loadScene, files: loadFiles },
  async create(canvas, signal) {
    const { default: init, StressGame } = await stressBindings();
    await init();
    const source = await loadScene(signal);
    const files = await loadFiles(source, signal);
    signal.throwIfAborted();
    return StressGame.create_scene(canvas, source, files);
  },
  async reload(game, signal) {
    const source = await loadScene(signal);
    const files = await loadFiles(source, signal);
    signal.throwIfAborted();
    game.reload_scene(source, files);
  },
  poll(game) {
    const progress = JSON.parse(game.scene_progress());
    document.getElementById("scene")!.dataset.progress = JSON.stringify(progress);
    document.getElementById("active-preset")!.textContent = progress.scene_id.toUpperCase();
    document.getElementById("active-summary")!.textContent = `${sourceUrl} · revision ${progress.revision}`;
  },
});
