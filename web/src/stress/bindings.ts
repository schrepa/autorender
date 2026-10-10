import { generationModule, requireWasmSimd } from "../engine/bindings";

/** The Stress Lab's bindings: the build a watched browser is pinned to, else
 * the served one. */
export async function stressBindings(): Promise<typeof import("../../pkg/stress_lab")> {
  requireWasmSimd();
  const module = generationModule("stress_lab");
  return module ? import(/* @vite-ignore */ module) : import("../../pkg/stress_lab");
}
