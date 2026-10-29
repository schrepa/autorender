import { loadGame } from "../engine/bindings";
import { type GameOptions, startGame } from "../engine/host";
import type { HighlandsGame, HighlandsRocks } from "../../pkg/highlands";

declare const __PUBLIC_KTX2__: readonly string[];

const FOLDER = "highlands/";
const parameters: Record<string, string> = {};
const controls: NonNullable<GameOptions<HighlandsGame>["controls"]> = {};

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
  async create(canvas, signal) {
    const { default: init, HighlandsGame, HighlandsRocks } = await loadGame("highlands", () => import("../../pkg/highlands"));
    await init();
    Object.assign(parameters, JSON.parse(HighlandsGame.parameters()) as Record<string, string>);
    for (const [name, values] of Object.entries(JSON.parse(HighlandsGame.controls()) as Record<string, string>)) {
      controls[name] = { values, run(game, args) { game.control(name, String(args.name)); return { [name]: args.name }; } };
    }
    return HighlandsGame.create(canvas, await rocks(new HighlandsRocks(), signal), location.search);
  },
});
