import { loadGame } from "../engine/bindings";
import { type GameOptions, startGame } from "../engine/host";
import { runSmokes, smokeSelection, type SmokeRun } from "../engine/smokes";
import type { HighlandsGame, HighlandsRocks } from "../../pkg/highlands";
import JobWorker from "./worker.ts?worker";

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

type HighlandsSmokes = (canvas: HTMLCanvasElement, names: string, report: (line: string) => void, query: string) => Promise<boolean>;

const smoke = smokeSelection();
if (smoke !== null) {
  void runSmokes(smoke, async (): Promise<SmokeRun | undefined> => {
    const { HighlandsGame } = await loadGame("highlands", () => import("../../pkg/highlands"), JobWorker);
    const smokes = (HighlandsGame as unknown as { smokes?: HighlandsSmokes }).smokes;
    return smokes && ((canvas, names, report) => smokes.call(HighlandsGame, canvas, names, report, location.search));
  });
} else startGame<HighlandsGame>({
  parameters,
  controls,
  async create(canvas, signal) {
    const { HighlandsGame, HighlandsRocks } = await loadGame("highlands", () => import("../../pkg/highlands"), JobWorker);
    Object.assign(parameters, JSON.parse(HighlandsGame.parameters()) as Record<string, string>);
    for (const [name, values] of Object.entries(JSON.parse(HighlandsGame.controls()) as Record<string, string>)) {
      controls[name] = { values, run(game, args) { game.control(name, String(args.name)); return { [name]: args.name }; } };
    }
    return HighlandsGame.create(canvas, await rocks(new HighlandsRocks(), signal), location.search);
  },
});
