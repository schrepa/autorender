import { playGame } from "../engine/game";
import JobWorker from "./worker.ts?worker";

playGame({
  wasm: "engine_player",
  load: () => import("../../pkg/engine_player"),
  worker: JobWorker,
  game: bindings => bindings.PlayerGame,
});
