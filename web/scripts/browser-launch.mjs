/**
 * How every script launches Chrome: the executable `BROWSER_PATH` names, else
 * the `BROWSER_CHANNEL` channel (Chrome by default), headless, with unsafe
 * WebGPU, and SwiftShader with `SOFTWARE_GPU=1`.
 */
import { resolve } from "node:path";

/** The flags every launch passes. */
export const launchFlags = ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])];

/** The browser a launch uses, as reports name it. */
export const browserChannel = process.env.BROWSER_PATH ? "explicit-executable" : process.env.BROWSER_CHANNEL ?? "chrome";

/** Playwright launch options with `args` after the shared flags and `extra`
 * beside them; `executablePath` overrides `BROWSER_PATH`. */
export function launchOptions({ executablePath, args = [], ...extra } = {}) {
  const path = executablePath ?? process.env.BROWSER_PATH;
  return {
    ...(path ? { executablePath: resolve(path) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
    headless: true,
    args: [...launchFlags, ...args],
    ...extra,
  };
}
