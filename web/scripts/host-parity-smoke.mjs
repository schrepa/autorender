// @ts-check
import "./exit-codes.mjs";
import assert from "node:assert/strict";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { crc32, deflateSync } from "node:zlib";
import { launchBrowserSession } from "./agent-browser.mjs";
import { launchNativeSession } from "./agent-native.mjs";
import { expectDisabledFeatures } from "./exercised.mjs";
import { cargoBinaries, devUrl, protocol } from "./harness.mjs";
import { workspacePath } from "./workspace.mjs";

const binaries = cargoBinaries(["-p", "engine_player", "--bin", "engine-player-native", "--profile", "quick", "--features", "dev"]);

const KNOWN = [
  ["describe ProfileCapture: native only *", "host", "native profiles take hardware counters and PIX"],
  ["describe PageParameters: web only engine_dev", "host", "the page's development tools are a URL parameter"],
  ["describe PageParameters: web only prepare", "host", "a page's development session may start as players' do"],
  ["describe PageParameters: web only engine_build", "host", "a watched page is pinned to a build"],
  ["describe PageParameters: native only {size,gpu}", "host", "a native host picks its view's size and GPU; a page has its canvas and browser"],
  ["describe PageParameters: native only {counters,vendor}", "host", "native profiling tools"],
];

function pattern(text) {
  const escaped = text.replace(/[.+?^$()|[\]\\]/g, "\\$&")
    .replace(/\*/g, ".*")
    .replace(/\{([^}]+)\}/g, (_, words) => `(?:${words.split(",").join("|")})`);
  return new RegExp(`^${escaped}$`);
}

const CASES = [
  ["status", { verb: "query", name: "status" }],
  ["describe", { verb: "describe" }],
  ["describe FrameStats", { verb: "describe", name: "FrameStats" }],
  ["describe ProfileCapture", { verb: "describe", name: "ProfileCapture" }],
  ["inspect environment", { verb: "inspect", component: "Environment" }],
  ["assets", { verb: "query", name: "assets" }],
  ["input", { verb: "control", name: "input", args: { forward: 1, yaw: 0.25 } }],
  ["step", { verb: "control", name: "step", args: { ticks: 3 } }],
  ["render", { verb: "control", name: "render" }],
  ["present", { verb: "control", name: "present", args: { alpha: 0.5 } }],
  ["capture", { verb: "capture" }],
  ["capture presented", { verb: "capture", alpha: 0.5 }],
  ["frames", { verb: "control", name: "frames", args: { count: 3, interval_ms: 1000 / 60, capture: true } }],
  ["benchmark", { verb: "control", name: "benchmark", args: { mode: "render", warmup: 1, samples: 4 } }],
  ["benchmark steps", { verb: "control", name: "benchmark", args: { mode: "step", warmup: 0, samples: 2 } }],
  ["culling", { verb: "control", name: "culling", args: { enabled: true } }],
  ["environment", { verb: "control", name: "environment", args: { bloom: { intensity: 0.5 } } }],
  ["camera", { verb: "control", name: "camera", args: { position: [0, 30, 60], target: [0, 0, 0], vertical_fov_radians: 1, near: 0.1, far: 2000 } }],
  ["terrain height", { verb: "query", name: "terrain_height", args: { points: [[0, 0]] } }],
  ["distribution", { verb: "query", name: "distribution", args: { values: [4, 1, 3, 2] } }],
  ["inspect", { verb: "inspect", limit: 3 }],
  ["unknown op", { verb: "control", name: "weather" }, "error"],
  ["unknown field", { verb: "control", name: "render", args: { sky: 1 } }, "error"],
  ["missing verb", { verbs: "query", name: "status" }, "error"],
  ["too large", { verb: "query", name: "status", args: { padding: "x".repeat(protocol.max_request_bytes) } }, "error"],
  ["raycast without collision", { verb: "query", name: "raycast", args: { origin: [0, 50, 0], direction: [0, -1, 0], max_distance: 100 } }, "error"],
  ["invalid step", { verb: "control", name: "step", args: { ticks: 0 } }, "error"],
  ["invalid input", { verb: "control", name: "input", args: { forward: 2 } }, "error"],
  ["invalid benchmark", { verb: "control", name: "benchmark", args: { samples: 0 } }, "error"],
  ["invalid frames", { verb: "control", name: "frames", args: { count: 0, interval_ms: 1000 / 60 } }, "error"],
  ["unknown game control", { verb: "control", name: "game", args: { name: "weather", args: {} } }, "error"],
  ["reload to a revision the source lacks", { verb: "control", name: "reload", args: { expected_revision: "1", revision: "2" } }, "error"],
  ["deep profile", { verb: "profile", deep: { duration_ms: 100 } }, "error"],
  ["profile", { verb: "profile", frames: 2 }],
  ["explain frame", { verb: "explain", about: "frame" }],
  ["explain scene", { verb: "explain", about: "scene" }],
  ["watch", { verb: "watch", stats: ["frame", "tick"], entities: [{ entity: "box", component: "Transform" }], frames: 2 }],
  ["resume", { verb: "control", name: "resume" }],
  ["step while running", { verb: "control", name: "step", args: { ticks: 1 } }, "error"],
  ["capture while running", { verb: "capture" }, "error"],
  ["frames while running", { verb: "control", name: "frames", args: { count: 1, interval_ms: 1000 / 60 } }, "error"],
  ["pause", { verb: "control", name: "pause" }],
  ["bloom before the fault", { verb: "control", name: "environment", args: { bloom: { intensity: 0.5 } } }],
  ["debug fault", { verb: "control", name: "debug_fault", args: { feature: "post" } }],
  ["frames after the fault", { verb: "control", name: "frames", args: { count: 30, interval_ms: 1000 / 60 } }],
  ["status after the fault", { verb: "query", name: "status" }],
  ["fault in no lazy feature", { verb: "control", name: "debug_fault", args: { feature: "water" } }, "error"],
  ["missing asset of no entity", { verb: "control", name: "debug_missing_asset", args: { entity: "nothing" } }, "error"],
  ["device loss", { verb: "control", name: "debug_device_loss" }, "error"],
  ["render after the loss", { verb: "control", name: "render" }, "error"],
  ["status after the loss", { verb: "query", name: "status" }],
  ["debug panic", { verb: "control", name: "debug_panic" }, "error"],
  ["status after the panic", { verb: "query", name: "status" }, "error"],
];

function shape(value, prefix = "", out = new Set()) {
  if (Array.isArray(value)) {
    for (const item of value) shape(item, `${prefix}[]`, out);
  } else if (value && typeof value === "object") {
    for (const [key, item] of Object.entries(value)) {
      const path = prefix ? `${prefix}.${key}` : key;
      out.add(path);
      if (path !== "result.types[].parameters" && path !== "result.types[].controls") shape(item, path, out);
    }
  }
  return out;
}

function described(reply, name, field) {
  return new Set(Object.keys(reply.result?.types?.find(type => type.name === name)?.[field] ?? {}));
}

function divergences(name, expectError, web, native) {
  const found = [];
  for (const [host, reply] of [["web", web], ["native", native]]) {
    if (reply.ok === expectError) found.push(`${name}: ${host} ${reply.ok ? "accepts" : `refuses with ${reply.error?.code}`}`);
  }
  if (web.ok !== native.ok) return found;
  if (!web.ok) {
    const fields = web.error?.code === "runtime_panicked" ? ["code", "path", "message"] : ["code", "path"];
    for (const field of fields) {
      if (web.error?.[field] !== native.error?.[field]) found.push(`${name}: error ${field} ${web.error?.[field]} on web, ${native.error?.[field]} natively`);
    }
  }
  const [webPaths, nativePaths] = [shape(web), shape(native)];
  for (const path of webPaths) if (!nativePaths.has(path)) found.push(`${name}: web only ${path}`);
  for (const path of nativePaths) if (!webPaths.has(path)) found.push(`${name}: native only ${path}`);
  return found;
}

async function answer(session, command) {
  try { return await session.execute(structuredClone(command)); }
  catch (error) { return { ok: false, error: { code: error.code ?? "thrown", path: error.path ?? "$", message: String(error.message ?? error) } }; }
}

const commandsPath = workspacePath("web/public/scenes/host-parity-commands.scene.json");
await mkdir(workspacePath("web/public/scenes"), { recursive: true });
await writeFile(commandsPath, JSON.stringify({
  id: "host-parity-commands", revision: 1,
  camera: { position: [0, 2, 4], target: [0, 0, 0], fov_degrees: 60, near: 0.1, far: 100 },
  materials: [{ id: "plain", color: [0.8, 0.3, 0.2, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [{ id: "box", mesh: "box", material: "plain", transform: { translation: [0, 0, 0] } }],
}));
const native = await launchNativeSession({
  program: (await binaries)["engine-player-native"],
  args: ["--source", commandsPath, "--without", "collision,physics", "--size=640x400", "--workers=2"],
}).catch(async error => { await rm(commandsPath, { force: true }); throw error; });
let web;
try {
  web = await launchBrowserSession({
    url: devUrl("player.html", { source: "/scenes/host-parity-commands.scene.json", without: "collision,physics", workers: "2" }).href,
  });
  const found = [];
  const replies = {};
  for (const [name, command, expected] of CASES) {
    if (name === "debug fault") expectDisabledFeatures();
    const pair = { web: await answer(web, command), native: await answer(native, command) };
    replies[name] = pair;
    found.push(...divergences(name, expected === "error", pair.web, pair.native));
  }
  for (const [type, field] of [["PageParameters", "parameters"], ["GameControls", "controls"]]) {
    const [webNames, nativeNames] = [described(replies.describe.web, type, field), described(replies.describe.native, type, field)];
    for (const parameter of webNames) if (!nativeNames.has(parameter)) found.push(`describe ${type}: web only ${parameter}`);
    for (const parameter of nativeNames) if (!webNames.has(parameter)) found.push(`describe ${type}: native only ${parameter}`);
  }
  const known = KNOWN.map(([text, step, what]) => ({ text, step, what, matches: pattern(text) }));
  const unknown = found.filter(divergence => !known.some(entry => entry.matches.test(divergence)));
  const gone = known.filter(entry => !found.some(divergence => entry.matches.test(divergence))).map(entry => entry.text);
  console.log(JSON.stringify({
    known: known.map(({ text, step, what }) => ({ divergence: text, removed_by: step, what })),
    unknown,
  }, null, 1));
  assert.deepEqual(unknown, [], `the hosts diverge where no known divergence says so:\n${unknown.join("\n")}`);
  assert.deepEqual(gone, [], `known divergences that no longer show; delete their entries:\n${gone.join("\n")}`);
  const fault = replies["status after the fault"];
  for (const reply of [fault.web, fault.native]) {
    assert.deepEqual(reply.result?.rendering?.disabled_features, ["post"], JSON.stringify(reply.result?.rendering));
  }
  for (const name of ["device loss", "render after the loss"]) {
    for (const reply of [replies[name].web, replies[name].native]) {
      assert.equal(reply.error?.code, "render_failed", JSON.stringify(reply));
      assert.match(reply.error.message, /GPU device lost/, reply.error.message);
    }
  }
  const panic = replies["debug panic"];
  for (const reply of [panic.web, panic.native]) {
    assert.equal(reply.error?.code, "runtime_panicked", JSON.stringify(reply));
    assert.match(reply.error.message, /^panicked at crates[\\/]engine[\\/]src[\\/]dev_host[\\/]controls[\\/]mod\.rs:\d+:\d+: debug_panic: /, reply.error.message);
  }
  const errors = [];
  web.page.on("pageerror", error => errors.push(String(error)));
  web.page.on("console", entry => { if (entry.type() === "error") errors.push(entry.text()); });
  await web.page.evaluate(() => new Promise(/** @param {(value?: unknown) => void} resolve */ resolve => {
    let frames = 0;
    const next = () => (++frames >= 90 ? resolve() : requestAnimationFrame(next));
    requestAnimationFrame(next);
  }));
  assert.equal(await web.page.locator("#status").getAttribute("data-state"), "error");
  assert.match(await web.page.locator("#status").textContent(), /Engine stopped/);
  assert.deepEqual(errors, [], "the page calls into the stopped module no more");
  console.log(`host parity smoke passed: ${CASES.length} commands, ${found.length} divergences, all known`);
} finally {
  await web?.close();
  await native.close();
  await rm(commandsPath, { force: true });
}

function png(width, pixels) {
  const chunk = (type, data) => {
    const head = Buffer.alloc(8);
    head.writeUInt32BE(data.length, 0);
    head.write(type, 4, "latin1");
    const crc = Buffer.alloc(4);
    crc.writeUInt32BE(crc32(Buffer.concat([head.subarray(4), data])), 0);
    return Buffer.concat([head, data, crc]);
  };
  const height = pixels.length / 4 / width;
  const header = Buffer.alloc(13);
  header.writeUInt32BE(width, 0);
  header.writeUInt32BE(height, 4);
  header.set([8, 6, 0, 0, 0], 8);
  const rows = Buffer.concat(Array.from({ length: height }, (_, row) =>
    Buffer.from([0, ...pixels.slice(row * width * 4, (row + 1) * width * 4)])));
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header), chunk("IDAT", deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]);
}

const folder = workspacePath("web/public/scenes/host-parity");
const scenePath = workspacePath("web/public/scenes/host-parity.scene.json");
await mkdir(folder, { recursive: true });
await writeFile(`${folder}/alpha.png`, png(2, [201, 77, 13, 3, 10, 250, 128, 128, 255, 255, 255, 0, 33, 66, 99, 254]));
await writeFile(scenePath, JSON.stringify({
  id: "host-parity", revision: 1,
  camera: { position: [0, 2, 4], target: [0, 0, 0], fov_degrees: 60, near: 0.1, far: 100 },
  textures: [{ id: "alpha", uri: "host-parity/alpha.png" }],
  materials: [{ id: "alpha", color: [1, 1, 1, 1], texture: "alpha" }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [{ id: "box", mesh: "box", material: "alpha", transform: { translation: [0, 0, 0] } }],
}));
let imagesWeb, imagesNative;
try {
  imagesNative = await launchNativeSession({ program: (await binaries)["engine-player-native"], args: ["--source", scenePath, "--size=320x200"] });
  imagesWeb = await launchBrowserSession({ url: devUrl("player.html", { source: "/scenes/host-parity.scene.json" }).href });
  const [web, native] = [await imagesWeb.execute({ verb: "query", name: "digest" }), await imagesNative.execute({ verb: "query", name: "digest" })];
  assert.ok(web.ok && native.ok, JSON.stringify({ web: web.error, native: native.error }));
  assert.equal(web.result.assets.textures.count, 1, JSON.stringify(web.result.assets));
  assert.deepEqual(web.result.assets.textures, native.result.assets.textures, "the PNG decodes to the same texture on both hosts");
  const missing = {};
  for (const [host, session] of [["web", imagesWeb], ["native", imagesNative]]) {
    await session.execute({ verb: "control", name: "pause" });
    const removed = await session.execute({ verb: "control", name: "debug_missing_asset", args: { entity: "box" } });
    assert.ok(removed.ok, JSON.stringify(removed.error));
    assert.deepEqual([removed.result.temporary, removed.result.entities], [true, 1], JSON.stringify(removed.result));
    assert.equal(removed.result.rendering.missing_assets, 1, JSON.stringify(removed.result.rendering));
    const status = await session.execute({ verb: "query", name: "status" });
    assert.equal(status.result.rendering.missing_assets, 1, JSON.stringify(status.result.rendering));
    const listed = await session.execute({ verb: "query", name: "assets" });
    assert.equal(listed.result.missing_total, 1);
    missing[host] = listed.result.missing;
  }
  assert.deepEqual(missing.web, missing.native, "both hosts list the missing mesh alike");
  const [entry] = missing.web;
  assert.deepEqual([entry.entity.id, entry.entity.scene, entry.draws, entry.missing], ["box", "host-parity", "mesh", "mesh"], JSON.stringify(entry));
  assert.deepEqual(entry.position, [0, 0, 0]);
  const unknown = await answer(imagesNative, { verb: "control", name: "debug_missing_asset", args: { entity: "nothing" } });
  assert.deepEqual([unknown.error?.code, unknown.error?.path], ["unknown_entity", "$.args.entity"]);
} finally {
  await imagesWeb?.close();
  await imagesNative?.close();
  await rm(folder, { recursive: true, force: true });
  await rm(scenePath, { force: true });
}
console.log("host parity smoke passed: a partially transparent PNG decodes to the same texture on both hosts, and both list a missing mesh alike");
