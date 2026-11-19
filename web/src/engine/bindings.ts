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

export interface GameBindings {
  default(): Promise<unknown>;
}

export async function loadGame<T extends GameBindings>(load: () => Promise<T>): Promise<T> {
  requireWasmSimd();
  const bindings = await load();
  await bindings.default();
  return bindings;
}
