import { loadGame } from "../engine/bindings";

export const stressBindings = () => loadGame("stress_lab", () => import("../../pkg/stress_lab"));
