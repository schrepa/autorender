import { playGame } from "../engine/game";
import JobWorker from "./worker.ts?worker";

playGame({
  wasm: "highlands",
  scene: "highlands.scene.json",
  load: () => import("../../pkg/highlands"),
  worker: JobWorker,
  game: bindings => bindings.HighlandsGame,
});
