import { playGame } from "../engine/game";
import JobWorker from "./worker.ts?worker";

playGame({
  wasm: "bare_game",
  load: () => import("../../pkg/bare_game"),
  worker: JobWorker,
  game: bindings => bindings.BareGame,
});
