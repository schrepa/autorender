import { loadGame } from "../engine/bindings";
import { type GameOptions, startGame } from "../engine/host";
import { loadSceneSource } from "../engine/source";
import clients from "../generated/clients.json";
import type { HighlandsGame, HighlandsRocks } from "../../pkg/highlands";
import JobWorker from "./worker.ts?worker";

declare const __PUBLIC_KTX2__: readonly string[];

const FOLDER = "highlands/";
const parameters: Record<string, string> = {};
const controls: NonNullable<GameOptions<HighlandsGame>["controls"]> = {};
const bindings = () => loadGame(() => import("../../pkg/highlands"), JobWorker);
const source = import.meta.env.DEV ? sceneSource() : undefined;

function sceneSource() {
  const folder = clients.find(client => client.game === "highlands")?.folder;
  const url = new URLSearchParams(location.search).get("source") ?? `/_engine/source/${folder}/highlands.scene.json`;
  return { url, load: (signal: AbortSignal) => loadSceneSource(url, signal), files: async () => new (await bindings()).SceneFileSet() };
}

async function rocks(found: HighlandsRocks, signal: AbortSignal): Promise<HighlandsRocks> {
  await Promise.all(__PUBLIC_KTX2__.filter(path => path.startsWith(`${FOLDER}rocks/`)).map(async path => {
    try {
      const response = await fetch(new URL(`${import.meta.env.BASE_URL}${path}`, location.href), { signal });
      if (response.ok) found.add(path.slice(FOLDER.length), new Uint8Array(await response.arrayBuffer()));
    } catch {}
  }));
  return found;
}

startGame<HighlandsGame>({
  parameters,
  controls,
  source,
  async create(canvas, signal) {
    const { HighlandsGame, HighlandsRocks } = await bindings();
    Object.assign(parameters, JSON.parse(HighlandsGame.parameters()) as Record<string, string>);
    for (const [name, values] of Object.entries(JSON.parse(HighlandsGame.controls()) as Record<string, string>)) {
      controls[name] = { values, run(game, args) { game.control(name, String(args.name)); return { [name]: args.name }; } };
    }
    const scene = await source?.load(signal);
    return HighlandsGame.create(canvas, await rocks(new HighlandsRocks(), signal), location.search, scene);
  },
});
