import { engineFailure } from "../generated/errors";

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

export function requireWasmSimd(): void {
  if (simdSupported === undefined) {
    try {
      simdSupported = typeof WebAssembly !== "undefined" && WebAssembly.validate(simdProbe);
    } catch {
      simdSupported = false;
    }
  }
  if (!simdSupported) {
    throw engineFailure({ code: "runtime_unavailable", path: "$", expected: "supported", got: "simd128" });
  }
}

export async function loadGame<T>(wasm: string, load: () => Promise<T>): Promise<T> {
  requireWasmSimd();
  const pinned = generationModule(wasm);
  return pinned ? import(/* @vite-ignore */ pinned) as Promise<T> : load();
}

export function generationModule(name: string): string | undefined {
  if (!import.meta.env.DEV) return;
  const generation = new URLSearchParams(location.search).get("engine_build");
  if (!generation) return;
  if (!/^[A-Za-z0-9_-]{1,100}$/.test(generation)) throw engineFailure({ code: "invalid_id", path: "?engine_build", expected: "id", got: generation });
  return `/_engine/builds/${generation}/${name}.js`;
}

export function engineWorkerModule(): string | undefined {
  return generationModule("engine_worker");
}
