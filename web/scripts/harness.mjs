/**
 * What the browser smoke checks share: Chrome with WebGPU and a page that
 * records its errors, development commands under one retry rule and
 * deadline, pixel comparisons with explicit tolerances, the development
 * server's pages, and the authored fixture scenes the Stress Lab loads in
 * place of a workload.
 */
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { mkdir, rm } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";
import { launchOptions } from "./browser-launch.mjs";

/** The development server the checks open: `DEMO_URL`, else Vite's default. */
export const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;

/** `page` of the development server with development tools, and `query`
 * (objects as JSON, such as the Stress Lab's `config`). */
export function devUrl(page, query = {}) {
  const url = new URL(page, base);
  url.searchParams.set("engine_dev", "1");
  for (const [key, value] of Object.entries(query)) {
    url.searchParams.set(key, typeof value === "string" ? value : JSON.stringify(value));
  }
  return url;
}

/** The scene URL fixtures are served at, which no file on disk holds. */
export const fixtureScene = "/scenes/smoke-fixture.scene.json";

/** The Stress Lab with development tools, loading the fixture scene in
 * place of a workload, and `query`. */
export function sceneUrl(query = {}) {
  return devUrl("stress.html", { source: fixtureScene, ...query });
}

/** The garden fixture: a version 4 scene with a player, solids,
 * pickups, a textured prefab and parents, as a fresh object. */
export function gardenScene() {
  return JSON.parse(readFileSync(new URL("fixtures/garden.scene.json", import.meta.url), "utf8"));
}

/** `artifacts/NAME/`, emptied, for `name`'s captures: each run replaces
 * the last one's. */
export async function artifacts(name) {
  const output = fileURLToPath(new URL(`../../artifacts/${name}/`, import.meta.url));
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  return output;
}

/** Headless Chrome (`BROWSER_PATH`, or the `BROWSER_CHANNEL` channel) with
 * unsafe WebGPU, SwiftShader with `SOFTWARE_GPU=1`, and one page of `width`
 * by `height` whose first errors collect in `errors`. */
export async function launch({ width = 1200, height = 900, timeout = 30_000, args = [] } = {}) {
  const browser = await chromium.launch(launchOptions({ args }));
  const page = await browser.newPage({ viewport: { width, height }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(timeout);
  const errors = [];
  const record = value => { if (errors.length < 20) errors.push(String(value).slice(0, 2000)); };
  page.on("pageerror", record);
  page.on("console", entry => { if (entry.type() === "error") record(entry.text()); });
  return { browser, page, errors };
}

/** Sends development commands to `page`, each answering within `deadline`
 * milliseconds with `ok` equal to `expectedOk`. Only a `command_busy` reply
 * is retried, until the deadline: it rejects a command before it runs, so
 * no command is ever repeated after it ran. */
export function commander(page, { deadline = 30_000 } = {}) {
  return async function command(op, fields = {}, expectedOk = true) {
    const until = Date.now() + deadline;
    for (;;) {
      let timer;
      const reply = await Promise.race([
        page.evaluate(request => window.__engineDev.request(request), { op, ...fields }),
        new Promise((_, reject) => {
          timer = setTimeout(() => reject(new Error(`${op} timed out; it was not retried`)), Math.max(1, until - Date.now()));
        }),
      ]).finally(() => clearTimeout(timer));
      if (reply?.error?.code === "command_busy" && Date.now() < until) { await delay(25); continue; }
      assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
      if (reply.result?.rendering) reply.result.rendering = strictStats(reply.result.rendering);
      return reply;
    }
  };
}

/** A frame's stats (a reply's `rendering`) whose reads of a field they lack
 * throw, at any depth: every field is always present, so a renamed or
 * misspelt one fails the check instead of reading `undefined`. */
export function strictStats(stats) {
  return new Proxy(stats, {
    get(target, field, receiver) {
      if (typeof field === "string" && !(field in target) && field !== "toJSON" && field !== "then") {
        throw new Error(`frame stats have no field ${JSON.stringify(field)}; describe FrameStats lists them`);
      }
      const value = Reflect.get(target, field, receiver);
      return value && typeof value === "object" && !Array.isArray(value) ? strictStats(value) : value;
    },
  });
}

/** The development protocol's version, request limit and ops, generated
 * from the engine (`cargo xtask schema`). */
export const protocol = JSON.parse(readFileSync(new URL("../src/generated/protocol.json", import.meta.url), "utf8"));

/** Waits up to `timeout` milliseconds for the page's development host to
 * answer this protocol's version. */
export async function developmentReady(page, { timeout = 60_000 } = {}) {
  await page.waitForFunction(version => window.__engineDev?.version === version, protocol.version, { timeout });
}

/** Opens `url` and pauses it once its development host answers, waiting up
 * to `timeout` milliseconds while the page starts and builds its workload. */
export async function openPaused(page, url, { timeout = 60_000 } = {}) {
  await page.goto(String(url), { waitUntil: "domcontentloaded" });
  await developmentReady(page, { timeout });
  return commander(page, { deadline: timeout })("pause");
}

/** How many pixels of two PNG captures (base64) differ by more than
 * `tolerance` in a color channel, or in alpha too with `alpha`; the
 * captures must have one size. */
export function differentPixels(page, images, { tolerance = 0, alpha = false } = {}) {
  return page.evaluate(async ({ images, tolerance, channels }) => {
    const pixels = await Promise.all(images.map(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d");
      context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height);
    }));
    if (pixels[0].width !== pixels[1].width || pixels[0].height !== pixels[1].height) throw new Error("Capture dimensions changed.");
    let changed = 0;
    for (let offset = 0; offset < pixels[0].data.length; offset += 4) {
      for (let channel = 0; channel < channels; channel++) {
        if (Math.abs(pixels[0].data[offset + channel] - pixels[1].data[offset + channel]) > tolerance) { changed++; break; }
      }
    }
    return { changed, total: pixels[0].width * pixels[0].height };
  }, { images, tolerance, channels: alpha ? 4 : 3 });
}

/** Asserts finite `actual` within `epsilon` of `expected`, element by element. */
export function near(actual, expected, message, epsilon = 0.00005) {
  assert.equal(actual?.length, expected.length, message);
  actual.forEach((value, index) => assert.ok(Number.isFinite(value) && Math.abs(value - expected[index]) <= epsilon,
    `${message}[${index}]: ${value} != ${expected[index]}`));
}

/** CPU and GPU residency of an `assets` reply, for comparisons. */
export function residency(reply) {
  const { result } = reply;
  return {
    meshes: result.meshes, materials: result.materials, textures: result.textures,
    cpu: result.resident_cpu_bytes, textureCpu: result.texture_cpu_bytes,
    gpuMeshes: result.rendering.resident_meshes, meshGpu: result.rendering.resident_mesh_bytes,
    gpuTextures: result.rendering.resident_textures, textureGpu: result.rendering.resident_texture_bytes,
  };
}

/** A disposable scene the Stress Lab loads from `fixtureScene`:
 * `route()` serves it, `reload(command, next)` replaces it with `next` as
 * the following revision through the reload control, and `serve(document)`
 * serves `document` as it is, e.g. for a reload that must be refused. */
export function routedScene(page, document) {
  let served = structuredClone(document);
  let revision = document.revision;
  return {
    route: () => page.route(`**${fixtureScene}*`, route => route.fulfill({ json: served })),
    get revision() { return revision; },
    serve(next) { served = structuredClone(next); },
    async reload(command, next) {
      served = structuredClone(next);
      served.revision = revision + 1;
      const reply = await command("reload", { expected_revision: String(revision), revision: String(served.revision) });
      revision = served.revision;
      assert.equal(reply.observation.scene.revision, String(revision));
      return reply;
    },
  };
}
