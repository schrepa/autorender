/** The web and native hosts answer the development protocol alike: the same
 * commands sent to the Stress Lab on the development server's page and
 * natively give replies with the same fields, the same error codes and
 * paths, and describe lists the same page parameters and game controls,
 * and an authored scene's partially transparent PNG decodes to a texture
 * of the same content hash on both,
 * but for the known divergences listed here, each tagged with the hosts
 * step that removes it (`host`: a difference of the transports, which
 * stays). A divergence that no longer shows fails the check until its entry
 * goes. A fault injected into the screen effects (`debug_fault`) disables
 * them alone on both, which render on without them. Last, `debug_panic`
 * stops the engine on both: each answers
 * `runtime_panicked` with the same report, and the page shows it stopped
 * and calls into its module no more. Builds `stress-lab` natively
 * (`--profile quick --features dev`) and opens the development server's
 * stress.html (`DEMO_URL`). */
// @ts-check
import "./exit-codes.mjs";
import assert from "node:assert/strict";
import { mkdir, rm, writeFile } from "node:fs/promises";
import { crc32, deflateSync } from "node:zlib";
import { launchBrowserSession } from "./agent-browser.mjs";
import { launchNativeSession } from "./agent-native.mjs";
import { cargoBinaries, devUrl, protocol } from "./harness.mjs";
import { workspacePath } from "./workspace.mjs";

// The native Stress Lab with its development tools, built in the background
// while the page starts: nothing here is timed.
const binaries = cargoBinaries(["-p", "stress-lab", "--bin", "stress-lab-native", "--profile", "quick", "--features", "dev"]);

/** What differs between the hosts today: the divergences each pattern
 * names (`*` stands for any text), the hosts step that removes them, and
 * what differs. */
const KNOWN = [
  ["describe ProfileCapture: native only *", "host", "native profiles take hardware counters and PIX"],
  ["describe PageParameters: web only engine_dev", "host", "the page's development tools are a URL parameter"],
  ["describe PageParameters: web only prepare", "host", "a page's development session may start as players' do"],
  ["describe PageParameters: web only engine_build", "host", "a watched page is pinned to a build"],
  ["describe PageParameters: native only {size,gpu}", "host", "a native host picks its view's size and GPU; a page has its canvas and browser"],
  ["describe PageParameters: native only {counters,vendor}", "host", "native profiling tools"],
  ["describe PageParameters: native only {view,assets,rocks}", "app", "the Stress Lab's own native options"],
];

/** A pattern of `KNOWN` as a regular expression: `*` any text, `{a,b}`
 * either word. */
function pattern(text) {
  const escaped = text.replace(/[.+?^$()|[\]\\]/g, "\\$&")
    .replace(/\*/g, ".*")
    .replace(/\{([^}]+)\}/g, (_, words) => `(?:${words.split(",").join("|")})`);
  return new RegExp(`^${escaped}$`);
}

/** The commands both hosts run, in order, by name; `error` where both must
 * refuse it. Commands only compare while the hosts start alike: each sets
 * what it reads. */
const CASES = [
  ["status", { op: "status" }],
  ["describe", { op: "describe" }],
  ["describe FrameStats", { op: "describe", name: "FrameStats" }],
  ["describe ProfileCapture", { op: "describe", name: "ProfileCapture" }],
  ["inspect environment", { op: "inspect", component: "Environment" }],
  ["assets", { op: "assets" }],
  ["input", { op: "input", forward: 1, yaw: 0.25 }],
  ["step", { op: "step", ticks: 3 }],
  ["render", { op: "render" }],
  ["present", { op: "present", alpha: 0.5 }],
  ["capture", { op: "capture" }],
  ["capture presented", { op: "capture", alpha: 0.5 }],
  ["frames", { op: "frames", count: 3, interval_ms: 1000 / 60, capture: true }],
  ["benchmark", { op: "benchmark", mode: "render", warmup: 1, samples: 4 }],
  ["benchmark steps", { op: "benchmark", mode: "step", warmup: 0, samples: 2 }],
  ["culling", { op: "culling", enabled: true }],
  ["environment", { op: "environment", bloom: { intensity: 0.5 } }],
  ["camera", { op: "camera", position: [0, 30, 60], target: [0, 0, 0], vertical_fov_radians: 1, near: 0.1, far: 2000 }],
  ["game look", { op: "game", name: "look", args: { name: "night" } }],
  ["game workload", { op: "game", name: "workload", args: { preset: "visual" } }],
  ["terrain height", { op: "terrain_height", points: [[0, 0]] }],
  ["distribution", { op: "distribution", values: [4, 1, 3, 2] }],
  // Refusals: the same code at the same path.
  ["unknown op", { op: "weather" }, "error"],
  ["unknown field", { op: "render", sky: 1 }, "error"],
  ["missing op", { ops: "status" }, "error"],
  ["too large", { op: "status", padding: "x".repeat(protocol.max_request_bytes) }, "error"],
  ["inspect without scene", { op: "inspect", limit: 3 }, "error"],
  ["raycast without collision", { op: "raycast", origin: [0, 50, 0], direction: [0, -1, 0], max_distance: 100 }, "error"],
  ["invalid step", { op: "step", ticks: 0 }, "error"],
  ["invalid input", { op: "input", forward: 2 }, "error"],
  ["invalid benchmark", { op: "benchmark", samples: 0 }, "error"],
  ["invalid frames", { op: "frames", count: 0, interval_ms: 1000 / 60 }, "error"],
  ["unknown game control", { op: "game", name: "weather", args: {} }, "error"],
  ["unknown look", { op: "game", name: "look", args: { name: "noon" } }, "error"],
  ["reload without source", { op: "reload", expected_revision: "1", revision: "2" }, "error"],
  ["profile", { op: "profile", duration_ms: 100 }, "error"],
  ["resume", { op: "resume" }],
  ["step while running", { op: "step", ticks: 1 }, "error"],
  ["capture while running", { op: "capture" }, "error"],
  ["frames while running", { op: "frames", count: 1, interval_ms: 1000 / 60 }, "error"],
  ["pause", { op: "pause" }],
  // A fault in the screen effects, which bloom needs, disables them alone:
  // the next frames find it, as browsers report GPU errors later.
  ["bloom before the fault", { op: "environment", bloom: { intensity: 0.5 } }],
  ["debug fault", { op: "debug_fault", feature: "post" }],
  ["frames after the fault", { op: "frames", count: 30, interval_ms: 1000 / 60 }],
  ["status after the fault", { op: "status" }],
  ["fault in no lazy feature", { op: "debug_fault", feature: "water" }, "error"],
  // A panic stops the engine on both hosts, which then answer everything
  // with the panic's report: last, as nothing runs after it.
  ["debug panic", { op: "debug_panic" }, "error"],
  ["status after the panic", { op: "status" }, "error"],
];

/** Every field path of `value`: object fields by name, array elements as
 * `[]` (their fields merged). The parameters and controls describe lists
 * are compared by name instead. */
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

/** The names in field `field` of type `name` that a describe reply lists. */
function described(reply, name, field) {
  return new Set(Object.keys(reply.result?.types?.find(type => type.name === name)?.[field] ?? {}));
}

/** Each divergence between the web's reply and the native one to a case,
 * as the check names them. */
function divergences(name, expectError, web, native) {
  const found = [];
  for (const [host, reply] of [["web", web], ["native", native]]) {
    if (reply.ok === expectError) found.push(`${name}: ${host} ${reply.ok ? "accepts" : `refuses with ${reply.error?.code}`}`);
  }
  if (web.ok !== native.ok) return found;
  if (!web.ok) {
    // A panic's message is the same report on both hosts.
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

/** `session`'s reply to `command`, a driver's refusal included. */
async function answer(session, command) {
  try { return await session.execute(structuredClone(command)); }
  catch (error) { return { ok: false, error: { code: error.code ?? "thrown", path: error.path ?? "$", message: String(error.message ?? error) } }; }
}

const native = await launchNativeSession({
  program: (await binaries)["stress-lab-native"],
  args: ["--preset=visual", "--size=640x400", "--workers=2"],
});
let web;
try {
  web = await launchBrowserSession({ url: devUrl("stress.html", { preset: "visual", workers: "2" }).href });
  const found = [];
  const replies = {};
  for (const [name, command, expected] of CASES) {
    const pair = { web: await answer(web, command), native: await answer(native, command) };
    replies[name] = pair;
    found.push(...divergences(name, expected === "error", pair.web, pair.native));
  }
  // What describe lists beside the engine's types: the page's parameters
  // and the game's controls.
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
  // The fault disabled the screen effects alone, on both hosts.
  const fault = replies["status after the fault"];
  for (const reply of [fault.web, fault.native]) {
    assert.deepEqual(reply.result?.rendering?.disabled_features, ["post"], JSON.stringify(reply.result?.rendering));
  }
  // Both report the engine's own panic, where it happened; the page says the
  // engine stopped and calls into its module no more (nothing fails after).
  const panic = replies["debug panic"];
  for (const reply of [panic.web, panic.native]) {
    assert.equal(reply.error?.code, "runtime_panicked", JSON.stringify(reply));
    assert.match(reply.error.message, /^panicked at crates[\\/]engine[\\/]src[\\/]dev_host[\\/]controls[\\/]mod\.rs:\d+:\d+: debug_panic: /, reply.error.message);
  }
  const errors = [];
  web.page.on("pageerror", error => errors.push(String(error)));
  web.page.on("console", entry => { if (entry.type() === "error") errors.push(entry.text()); });
  // Ninety of the browser's animation frames, in which the stopped page
  // must call into its module no more.
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
}

/** An RGBA8 PNG `width` pixels wide of `pixels`, rows from the top. */
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
  // Each row after its filter byte, none.
  const rows = Buffer.concat(Array.from({ length: height }, (_, row) =>
    Buffer.from([0, ...pixels.slice(row * width * 4, (row + 1) * width * 4)])));
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", header), chunk("IDAT", deflateSync(rows)), chunk("IEND", Buffer.alloc(0))]);
}

// A scene whose texture is a PNG with partial alpha: a decoder that
// premultiplies changes its colors (the faintest pixel's most).
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
  imagesNative = await launchNativeSession({ program: (await binaries)["stress-lab-native"], args: ["--source", scenePath, "--size=320x200"] });
  imagesWeb = await launchBrowserSession({ url: devUrl("stress.html", { source: "/scenes/host-parity.scene.json" }).href });
  const [web, native] = [await imagesWeb.execute({ op: "digest" }), await imagesNative.execute({ op: "digest" })];
  assert.ok(web.ok && native.ok, JSON.stringify({ web: web.error, native: native.error }));
  assert.equal(web.result.assets.textures.count, 1, JSON.stringify(web.result.assets));
  assert.deepEqual(web.result.assets.textures, native.result.assets.textures, "the PNG decodes to the same texture on both hosts");
} finally {
  await imagesWeb?.close();
  await imagesNative?.close();
  await rm(folder, { recursive: true, force: true });
  await rm(scenePath, { force: true });
}
console.log("host parity smoke passed: a partially transparent PNG decodes to the same texture on both hosts");
