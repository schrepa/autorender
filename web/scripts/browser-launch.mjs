/**
 * How every script launches Chrome: the executable `BROWSER_PATH` names, else
 * the `BROWSER_CHANNEL` channel (Chrome by default), headless, with unsafe
 * WebGPU, and SwiftShader with `SOFTWARE_GPU=1`; and how checks and agent
 * sessions keep the shaders it compiled from one launch to the next.
 */
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

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

/** What a profile folder keeps from one launch to the next: Chrome's
 * compiled-shader caches (WebGPU's, through Dawn, and the compositor's), by
 * folder name at the profile's top and in its Default profile. Everything
 * else (storage, the HTTP cache, settings) goes before each launch, so a page
 * starts as in a fresh profile but for the shaders compiled before; a
 * changed shader misses and compiles. Unknown names go too: a browser that
 * keeps its shaders elsewhere starts cold. */
const SHADER_CACHES = new Set(["DawnWebGPUCache", "DawnGraphiteCache", "GraphiteDawnCache", "GPUCache", "GrShaderCache", "ShaderCache"]);

/** Launches at the same time each take a profile folder of their own. */
const FOLDERS = 16;

/** The lock files of the folders this process holds, freed when it exits. */
const held = new Set();
process.once("exit", () => { for (const lock of held) rmSync(lock, { force: true }); });

/** Whether process `pid` runs. */
function running(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return error.code === "EPERM"; }
}

/** Takes `lock` for this process: free, or held by a process that ended. */
function take(lock) {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      writeFileSync(lock, String(process.pid), { flag: "wx" });
      held.add(lock);
      return true;
    } catch (error) {
      if (error.code !== "EEXIST") return false;
    }
    let owner;
    try { owner = Number(readFileSync(lock, "utf8")); } catch { continue; }
    if (running(owner)) return false;
    rmSync(lock, { force: true });
  }
  return false;
}

/** A profile folder of the shader cache (`SMOKE_SHADER_CACHE_DIR`, else one
 * in the system's temporary folder), this process's until `release`, with
 * all but the shader caches removed; none with `SMOKE_SHADER_CACHE=off`, or
 * when every folder is in use. */
function claimFolder() {
  if (process.env.SMOKE_SHADER_CACHE === "off") return null;
  const root = process.env.SMOKE_SHADER_CACHE_DIR ?? join(tmpdir(), "engine-chrome-shader-cache");
  try { mkdirSync(root, { recursive: true }); } catch { return null; }
  for (let index = 0; index < FOLDERS; index++) {
    const folder = join(root, String(index));
    const lock = `${folder}.lock`;
    if (!take(lock)) continue;
    const release = () => { held.delete(lock); rmSync(lock, { force: true }); };
    try {
      mkdirSync(folder, { recursive: true });
      for (const entry of readdirSync(folder, { withFileTypes: true })) {
        const path = join(folder, entry.name);
        if (entry.name === "Default" && entry.isDirectory()) {
          for (const inner of readdirSync(path)) {
            if (!SHADER_CACHES.has(inner)) rmSync(join(path, inner), { recursive: true, force: true });
          }
        } else if (!SHADER_CACHES.has(entry.name)) {
          rmSync(path, { recursive: true, force: true });
        }
      }
      return { folder, release };
    } catch {
      // Still in use by a browser that outlived its process: the next folder.
      release();
    }
  }
  return null;
}

/** Chrome as `launchOptions` describes it with one context of
 * `contextOptions` (viewport, scale): in a profile folder that keeps the
 * shaders it compiled for the next launch, else (`SMOKE_SHADER_CACHE=off`,
 * every folder in use, or a launch there failing) in a temporary profile.
 * Resolves to the context, its first page, Chrome's version and `close`,
 * which frees the folder too. */
export async function launchContext({ executablePath, args = [], ...contextOptions } = {}) {
  const { chromium } = await import("@playwright/test");
  const options = launchOptions({ executablePath, args });
  const claimed = claimFolder();
  if (claimed) {
    try {
      const context = await chromium.launchPersistentContext(claimed.folder, { ...options, ...contextOptions });
      const page = context.pages()[0] ?? await context.newPage();
      // A persistent context has no browser object to ask.
      const session = await context.newCDPSession(page);
      const { product } = await session.send("Browser.getVersion").finally(() => session.detach());
      return { context, page, version: product.split("/").pop(), close: () => context.close().finally(claimed.release) };
    } catch {
      claimed.release();
    }
  }
  const browser = await chromium.launch(options);
  const context = await browser.newContext(contextOptions);
  return { context, page: await context.newPage(), version: browser.version(), close: () => browser.close() };
}
