// @ts-check
import { mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";

export const launchFlags = ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])];

export const browserChannel = process.env.BROWSER_PATH ? "explicit-executable" : process.env.BROWSER_CHANNEL ?? "chrome";

/** @param {{ executablePath?: string, args?: string[], [option: string]: any }} [options] */
export function launchOptions({ executablePath, args = [], ...extra } = {}) {
  const path = executablePath ?? process.env.BROWSER_PATH;
  return {
    ...(path ? { executablePath: resolve(path) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
    headless: true,
    args: [...launchFlags, ...args],
    ...extra,
  };
}

const SHADER_CACHES = new Set(["DawnWebGPUCache", "DawnGraphiteCache", "GraphiteDawnCache", "GPUCache", "GrShaderCache", "ShaderCache"]);

const FOLDERS = 16;

const held = new Set();
process.once("exit", () => { for (const lock of held) rmSync(lock, { force: true }); });

export function running(pid) {
  if (!Number.isSafeInteger(pid) || pid <= 0) return false;
  try { process.kill(pid, 0); return true; } catch (error) { return error.code === "EPERM"; }
}

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
      release();
    }
  }
  return null;
}

/** @param {{ executablePath?: string, args?: string[], [option: string]: any }} [options] */
export async function launchContext({ executablePath, args = [], ...contextOptions } = {}) {
  const { chromium } = await import("@playwright/test");
  const options = launchOptions({ executablePath, args });
  const claimed = claimFolder();
  if (claimed) {
    try {
      const context = await chromium.launchPersistentContext(claimed.folder, { ...options, ...contextOptions });
      const uncached = async page => {
        const session = await context.newCDPSession(page);
        await session.send("Network.enable");
        await session.send("Network.setCacheDisabled", { cacheDisabled: true });
      };
      context.on("page", page => { uncached(page).catch(() => {}); });
      const page = context.pages()[0] ?? await context.newPage();
      await uncached(page);
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
