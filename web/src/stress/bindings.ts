import { loadGame } from "../engine/bindings";

/** The Stress Lab's bindings. */
export const stressBindings = () => loadGame("stress_lab", () => import("../../pkg/stress_lab"));
