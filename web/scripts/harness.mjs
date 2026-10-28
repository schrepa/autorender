// @ts-check
import "./exit-codes.mjs";
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { readFileSync } from "node:fs";
import { mkdir, rm } from "node:fs/promises";
import { constants, setPriority } from "node:os";
import { basename, sep } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { launchContext } from "./browser-launch.mjs";
import { recordReceived, recordSent } from "./exercised.mjs";
import { holdHeavyRun } from "./heavy-run.mjs";
import { workspace, workspacePath } from "./workspace.mjs";

await holdHeavyRun(`smoke ${basename(process.argv[1] ?? "")}`);

export const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;

export function devUrl(page, query = {}) {
  const url = new URL(page, base);
  url.searchParams.set("engine_dev", "1");
  for (const [key, value] of Object.entries(query)) {
    url.searchParams.set(key, typeof value === "string" ? value : JSON.stringify(value));
  }
  return url;
}

export const fixtureScene = "/scenes/smoke-fixture.scene.json";

export function sceneUrl(query = {}) {
  return devUrl("player.html", { source: fixtureScene, ...query });
}

export function gardenScene() {
  return JSON.parse(readFileSync(new URL("fixtures/garden.scene.json", import.meta.url), "utf8"));
}

export function gridScene({ side = 24, moving = false } = {}) {
  const size = 32;
  const pixels = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const shade = 150 + ((x * 7 + y * 13) % 48);
      const light = ((x >> 2) + (y >> 2)) % 2 === 0;
      pixels.set(light ? [shade + 40, shade + 30, shade, 255] : [shade - 80, shade - 70, shade - 50, 255], (y * size + x) * 4);
    }
  }
  const paints = [[0.85, 0.35, 0.2, 1], [0.25, 0.6, 0.85, 1], [0.4, 0.75, 0.3, 1], [0.9, 0.75, 0.25, 1]];
  const cube = { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] };
  const entities = [{
    id: "ground", mesh: "box", material: "ground", transform: { translation: [0, -0.5, 0], scale: [side * 6, 1, side * 6] },
    ...(moving ? { collider: { shape: cube } } : {}),
  }];
  for (let index = 0; index < side * side; index++) {
    const [row, column] = [Math.floor(index / side), index % side];
    const scale = 0.8 + (index % 5) * 0.3;
    entities.push({
      id: `piece-${index}`, mesh: index % 3 === 0 ? "gem" : "box", material: `paint-${index % paints.length}`,
      transform: {
        translation: [(column - (side - 1) / 2) * 4, scale / 2, (row - (side - 1) / 2) * 4],
        rotation_degrees: [0, (index * 37) % 360, 0], scale: [scale, scale, scale],
      },
      ...(moving ? { collider: { shape: cube }, rigid_body: { velocity: [0, 4 + (index % 7), 0], spin: [0, 1 + (index % 3), 0] } } : {}),
    });
    if (index % 4 === 0) {
      entities.push({
        id: `cap-${index}`, parent: `piece-${index}`, mesh: "box", material: `paint-${(index + 1) % paints.length}`,
        transform: { translation: [0, 0.75, 0], scale: [0.5, 0.5, 0.5] },
      });
    }
  }
  return {
    id: "grid", revision: 1,
    camera: { position: [0, 30, side * 2.6], target: [0, 0, 0], fov_degrees: 60, near: 0.1, far: 1000 },
    textures: [{ id: "checker", width: size, height: size, rgba8: pixels.toString("base64") }],
    materials: [
      { id: "ground", color: [0.55, 0.58, 0.52, 1], texture: "checker" },
      ...paints.map((color, index) => ({ id: `paint-${index}`, color, texture: "checker" })),
    ],
    meshes: [{ id: "box", source: { kind: "cube" } }, { id: "gem", source: { kind: "octahedron" } }],
    entities,
  };
}

export async function nativeBinary(cargoPackage, binary, { profiling = false } = {}) {
  const [chosen, features] = profiling ? ["profiling", "dev,profiling"] : ["release", "dev"];
  const profile = process.env.ENGINE_NATIVE_PROFILE || chosen;
  const env = profiling && process.platform === "linux"
    ? { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ""} -Cforce-frame-pointers=yes`.trim() }
    : process.env;
  const binaries = await cargoBinaries(["-p", cargoPackage, "--bin", binary, "--profile", profile, "--features", features], env);
  return binaries[binary];
}

export function playerProgram(options = {}) {
  return nativeBinary("engine_player", "engine-player-native", options);
}

export function cargoBinaries(cargoArguments, env = process.env) {
  const built = new Promise((resolveBuild, reject) => {
    const child = spawn(process.env.CARGO ?? "cargo", ["build", "-q", "--locked", "--message-format=json", ...cargoArguments],
      { cwd: workspace, env, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
    try { if (child.pid !== undefined) setPriority(child.pid, constants.priority.PRIORITY_BELOW_NORMAL); } catch {}
    let stdout = "", stderr = "";
    child.stdout.on("data", bytes => { stdout += bytes; });
    child.stderr.on("data", bytes => { stderr = (stderr + bytes).slice(-16_384); });
    child.on("error", reject);
    child.on("close", code => {
      if (code !== 0) { reject(new Error(`cargo build ${cargoArguments.join(" ")} exited ${code}: ${stderr}`)); return; }
      const binaries = {};
      for (const line of stdout.split(/\r?\n/)) {
        const message = line.startsWith("{") ? JSON.parse(line) : null;
        if (message?.reason === "compiler-artifact" && message.executable) binaries[message.target.name] = message.executable;
      }
      resolveBuild(binaries);
    });
  });
  built.catch(() => {});
  return built;
}

export async function artifacts(name) {
  const output = `${workspacePath("artifacts", name)}${sep}`;
  await rm(output, { recursive: true, force: true });
  await mkdir(output, { recursive: true });
  return output;
}

export async function launch({ width = 1200, height = 900, timeout = 30_000, args = [] } = {}) {
  const { page, version, close } = await launchContext({ args, viewport: { width, height }, deviceScaleFactor: 1 });
  const browser = { close, version: () => version };
  page.setDefaultTimeout(timeout);
  const errors = [];
  const record = value => { if (errors.length < 20) errors.push(String(value).slice(0, 2000)); };
  page.on("pageerror", record);
  page.on("console", entry => { if (entry.type() === "error") record(entry.text()); });
  return { browser, page, errors };
}

export function commander(page, { deadline = 30_000 } = {}) {
  return async function command(op, fields = {}, expectedOk = true) {
    recordSent({ op, ...fields });
    let until = Date.now() + deadline;
    for (;;) {
      let timer;
      let answered = false;
      const reply = await Promise.race([
        page.evaluate(request => window.__engineDev.request(request), { op, ...fields }),
        new Promise((_, reject) => {
          const expire = async () => {
            const waiting = await page.evaluate(() => window.__engineDev?.waitingOnGpu?.() ?? false).catch(() => false);
            if (answered) return;
            if (waiting) until = Date.now() + deadline;
            if (Date.now() >= until) reject(new Error(`${op} timed out; it was not retried`));
            else timer = setTimeout(expire, until - Date.now());
          };
          timer = setTimeout(expire, Math.max(1, until - Date.now()));
        }),
      ]).finally(() => { answered = true; clearTimeout(timer); });
      if (reply?.error?.code === "command_busy" && Date.now() < until) { await delay(25); continue; }
      recordReceived(reply);
      assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
      if (reply.result?.rendering) reply.result.rendering = strictStats(reply.result.rendering);
      return reply;
    }
  };
}

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

/**
 * @param {(op: string) => Promise<any>} command
 * @param {number} frames
 * @param {{ done?: (status: any) => boolean, deadline?: number }} [options]
 */
export async function waitFrames(command, frames, { done = () => true, deadline = 120_000 } = {}) {
  const first = Number((await command("status")).observation.frame);
  const until = Date.now() + deadline;
  for (;;) {
    const status = await command("status");
    if (Number(status.observation.frame) - first >= frames && done(status)) return status;
    if (Date.now() > until) {
      throw new Error(`${frames} frames did not run to the state wanted within ${deadline / 1000} s: ${JSON.stringify(status.observation)}`);
    }
    await delay(10);
  }
}

export const protocol = JSON.parse(readFileSync(new URL("../src/generated/protocol.json", import.meta.url), "utf8"));

export async function developmentReady(page, { timeout = 60_000 } = {}) {
  await page.waitForFunction(version => window.__engineDev?.version === version, protocol.version, { timeout });
}

export async function openPaused(page, url, { timeout = 60_000 } = {}) {
  await page.goto(String(url), { waitUntil: "domcontentloaded" });
  await developmentReady(page, { timeout });
  return commander(page, { deadline: timeout })("pause");
}

export function differentPixels(page, images, { tolerance = 0, alpha = false } = {}) {
  return page.evaluate(async ({ images, tolerance, channels }) => {
    const pixels = await Promise.all(images.map(async png => {
      const image = new Image();
      image.src = `data:image/png;base64,${png}`;
      await image.decode();
      const canvas = document.createElement("canvas");
      canvas.width = image.width; canvas.height = image.height;
      const context = /** @type {CanvasRenderingContext2D} */ (canvas.getContext("2d"));
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

export function near(actual, expected, message, epsilon = 0.00005) {
  assert.equal(actual?.length, expected.length, message);
  actual.forEach((value, index) => assert.ok(Number.isFinite(value) && Math.abs(value - expected[index]) <= epsilon,
    `${message}[${index}]: ${value} != ${expected[index]}`));
}

export function residency(reply) {
  const { result } = reply;
  return {
    meshes: result.meshes, materials: result.materials, textures: result.textures,
    cpu: result.resident_cpu_bytes, textureCpu: result.texture_cpu_bytes,
    gpuMeshes: result.rendering.resident_meshes, meshGpu: result.rendering.resident_mesh_bytes,
    gpuTextures: result.rendering.resident_textures, textureGpu: result.rendering.resident_texture_bytes,
  };
}

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
