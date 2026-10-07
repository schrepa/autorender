// (module (func (drop (v128.const i32x4 0 0 0 0))))
// Validate one standard SIMD instruction before importing/instantiating a game.
// No memory, imports, exports or execution are needed for this capability check.
const simdProbe = new Uint8Array([
  0, 97, 115, 109, 1, 0, 0, 0,
  1, 4, 1, 96, 0, 0,
  3, 2, 1, 0,
  10, 23, 1, 21, 0, 253, 12,
  0, 0, 0, 0, 0, 0, 0, 0,
  0, 0, 0, 0, 0, 0, 0, 0,
  26, 11,
]);
let simdSupported: boolean | undefined;

function requireWasmSimd(): void {
  if (simdSupported === undefined) {
    try {
      simdSupported = typeof WebAssembly !== "undefined" && WebAssembly.validate(simdProbe);
    } catch {
      simdSupported = false;
    }
  }
  if (!simdSupported) {
    throw new Error("This game build requires WebAssembly SIMD (simd128). Use an updated browser with WebAssembly SIMD and WebGPU enabled.");
  }
}

/** Pin a watched browser to one complete immutable Rust build. */
function generationModule(name: string): string | undefined {
  if (!import.meta.env.DEV) return;
  const generation = new URLSearchParams(location.search).get("engine_build");
  if (!generation) return;
  if (!/^[A-Za-z0-9_-]{1,100}$/.test(generation)) throw new Error("Invalid development build identifier.");
  return `/_engine/builds/${generation}/${name}.js`;
}

/** The job worker module (decision 110) of the build a watched browser is
 * pinned to, for workers to import; none outside watched builds. */
export function engineWorkerModule(): string | undefined {
  return generationModule("engine_worker");
}

export async function playgroundBindings(): Promise<typeof import("../pkg/playground")> {
  requireWasmSimd();
  const module = generationModule("playground");
  return module ? import(/* @vite-ignore */ module) : import("../pkg/playground");
}

export async function orbitBindings(): Promise<typeof import("../pkg/orbit_demo")> {
  requireWasmSimd();
  const module = generationModule("orbit_demo");
  return module ? import(/* @vite-ignore */ module) : import("../pkg/orbit_demo");
}

export async function stressBindings(): Promise<typeof import("../pkg/stress_lab")> {
  requireWasmSimd();
  const module = generationModule("stress_lab");
  return module ? import(/* @vite-ignore */ module) : import("../pkg/stress_lab");
}
