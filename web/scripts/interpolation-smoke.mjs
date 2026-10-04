/** Real Chrome interpolation contracts. The disposable scene exists only in routed responses. */
import assert from "node:assert/strict";
import { createHash, randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";

const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;
const url = new URL("playground.html", base);
url.searchParams.set("engine_dev", "1");
if (process.env.ENGINE_BUILD) url.searchParams.set("engine_build", process.env.ENGINE_BUILD);
const output = fileURLToPath(new URL(`../../artifacts/interpolation-smoke-${randomUUID()}/`, import.meta.url));
const camera = { position: [1.3, 0.8, 2.4], target: [0, 0.3, 0], vertical_fov_radians: Math.PI / 3, near: 0.1, far: 30 };
const fixture = {
  version: 4, id: "interpolation-fixture", revision: 1,
  camera: { position: camera.position, target: camera.target, fov_degrees: 60, near: 0.1, far: 30 },
  materials: [{ id: "player-color", color: [0.08, 0.65, 0.85, 1] }, { id: "accessory-color", color: [0.95, 0.35, 0.06, 1] }],
  meshes: [{ id: "box", source: { kind: "cube" } }],
  entities: [
    { id: "player", mesh: "box", material: "player-color", tags: ["player"], collider: { shape: { kind: "cuboid", half_extents: [0.5, 0.5, 0.5] }, memberships: 2, filter: 1, sensor: false }, transform: { translation: [0, 0.15, 0], scale: [0.16, 0.3, 0.16] } },
    { id: "accessory", parent: "player", mesh: "box", material: "accessory-color", transform: { translation: [0, 1.6, 0], scale: [0.5, 0.5, 0.5] } },
    { id: "floor", transform: { translation: [0, -0.5, 0] }, collider: { shape: { kind: "cuboid", half_extents: [500, 0.5, 500] }, memberships: 1, filter: 2, sensor: false } },
  ],
};
let served = structuredClone(fixture), revision = fixture.revision;
const browser = await chromium.launch({
  ...(process.env.BROWSER_PATH ? { executablePath: resolve(process.env.BROWSER_PATH) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
  headless: true, args: ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(20_000);
const errors = [], checks = [];
const bindingUrls = new Set();
const recordError = value => { if (errors.length < 20) errors.push(String(value).slice(0, 2000)); };
page.on("pageerror", recordError);
page.on("console", entry => { if (entry.type() === "error") recordError(entry.text()); });
page.on("response", response => {
  if (response.request().resourceType() === "script" && new URL(response.url()).pathname.endsWith("/playground.js")) bindingUrls.add(response.url());
});
let suiteTimer;

async function command(op, fields = {}, expectedOk = true) {
  let timer;
  try {
    const reply = await Promise.race([
      page.evaluate(command => window.__engineDev.request(command), { op, ...fields }),
      new Promise((_, reject) => { timer = setTimeout(() => reject(new Error(`${op} timed out; its mutation was not retried`)), 10_000); }),
    ]);
    assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
    return reply;
  } finally { clearTimeout(timer); }
}

async function inspect(id, component) {
  const reply = await command("inspect", { entity: id, component });
  return reply.result.entity.components[component];
}

function near(actual, expected, message, epsilon = 0.00005) {
  assert.equal(actual?.length, expected.length, message);
  actual.forEach((value, index) => assert.ok(Number.isFinite(value) && Math.abs(value - expected[index]) <= epsilon,
    `${message}[${index}]: ${value} != ${expected[index]}`));
}

async function authority() {
  const observation = (await command("status")).observation;
  return {
    player: await inspect("player", "Transform"), child: await inspect("accessory", "Transform"),
    playerGlobal: await inspect("player", "GlobalTransform"), childGlobal: await inspect("accessory", "GlobalTransform"),
    progress: (await command("inspect", { component: "GameProgress" })).result.value,
    tick: observation.tick, frame: observation.frame,
  };
}

async function presentedMatches(id, matrix, label) {
  const value = await inspect(id, "PresentedTransform");
  assert.equal(value.valid, true, label);
  near(value.matrix, matrix, `${label} matrix`);
  near(value.translation, matrix.slice(12, 15), `${label} translation`);
  return value;
}

function presentation(reply, mode, alpha) {
  const value = reply.observation.presentation;
  assert.equal(value.mode, mode);
  assert.ok(Math.abs(value.alpha - alpha) < 0.000001);
  assert.equal(value.frame, reply.observation.frame, "Extraction must identify the actual observed frame.");
  assert.equal(value.tick, reply.observation.tick, "Extraction must identify the actual observed fixed tick.");
  for (const key of ["history_nodes", "history_capacity", "history_bytes", "presented_nodes", "presentation_scratch_bytes"]) assert.ok(Number.isSafeInteger(value[key]) && value[key] >= 0, key);
  return value;
}

async function reload(document) {
  served = structuredClone(document);
  served.revision = revision + 1;
  const reply = await command("reload", { expected_revision: String(revision), revision: String(served.revision) });
  revision = served.revision;
  assert.equal(reply.observation.scene.revision, String(revision));
  assert.equal(reply.result.rendering.invalid_transforms, 0);
  presentation(reply, "current", 1);
  return reply;
}

async function moveOne() {
  const before = await authority();
  await command("input", { forward: 1 });
  const stepped = await command("step", { ticks: 1 });
  await command("input");
  assert.equal(BigInt(stepped.observation.tick), BigInt(before.tick) + 1n);
  presentation(stepped, "current", 1);
  const after = await authority();
  assert.ok(before.playerGlobal.translation[2] - after.playerGlobal.translation[2] > 0.1);
  assert.deepEqual(before.child, after.child, "Inherited motion must not mutate a child's local transform.");
  await presentedMatches("player", after.playerGlobal.matrix, "exact-step player");
  await presentedMatches("accessory", after.childGlobal.matrix, "exact-step child");
  return { before, after };
}

async function capture(name, alpha) {
  const reply = await command("capture", alpha === undefined ? {} : { alpha });
  presentation(reply, alpha === undefined ? "current" : "interpolated", alpha ?? 1);
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  assert.ok(bytes.length > 1000);
  return { png: bytes.toString("base64"), hash: createHash("sha256").update(bytes).digest("hex"), reply };
}

async function changedPixels(first, second) {
  return page.evaluate(async images => {
    const pixels = await Promise.all(images.map(async png => {
      const image = new Image(); image.src = `data:image/png;base64,${png}`; await image.decode();
      const canvas = document.createElement("canvas"); canvas.width = image.width; canvas.height = image.height;
      const context = canvas.getContext("2d"); context.drawImage(image, 0, 0);
      return context.getImageData(0, 0, image.width, image.height);
    }));
    if (pixels[0].width !== pixels[1].width || pixels[0].height !== pixels[1].height) throw new Error("Canvas dimensions changed");
    let changed = 0;
    for (let offset = 0; offset < pixels[0].data.length; offset += 4) {
      if ([0, 1, 2, 3].some(channel => pixels[0].data[offset + channel] !== pixels[1].data[offset + channel])) changed++;
    }
    return changed;
  }, [first.png, second.png]);
}

try {
  suiteTimer = setTimeout(() => { void browser.close(); }, 120_000);
  await page.route("**/scenes/playground.scene.json", route => route.fulfill({ json: served }));
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__engineDev?.version === 1, null, { timeout: 30_000 });
  const deadline = Date.now() + 5000;
  while (true) {
    const reply = await page.evaluate(() => window.__engineDev.request({ op: "pause" }));
    // This rejection occurs before dispatch; uncertain mutations are never retried.
    if (reply.error?.code === "command_busy" && Date.now() < deadline) { await delay(25); continue; }
    assert.equal(reply.ok, true, JSON.stringify(reply));
    assert.equal(reply.observation.tick, "0");
    break;
  }
  await mkdir(output, { recursive: true });
  assert.equal(bindingUrls.size, 1, "The ordinary-frame check must use the exact loaded game bindings.");
  const ordinaryFrames = await page.evaluate(async ({ moduleUrl, fixture }) => {
    const { PlaygroundGame, SceneFileSet } = await import(moduleUrl);
    const canvas = document.createElement("canvas");
    canvas.width = 320; canvas.height = 240;
    canvas.style.cssText = "position:fixed;left:-10000px;top:0;width:320px;height:240px";
    document.body.append(canvas);
    let game;
    try {
      // The fixture references no files (decision 030).
      game = await PlaygroundGame.create(canvas, JSON.stringify(fixture), new SceneFileSet());
      // Development commands between these frames would complete a pipelined
      // tick early (decision 015); this check covers interpolation alone.
      game.set_tick_pipelining(false);
      const request = command => {
        const reply = JSON.parse(game.dev_command(JSON.stringify({ version: 1, ...command }), 320, 240));
        if (!reply.ok) throw new Error(JSON.stringify(reply));
        return reply;
      };
      const inspect = (entity, component) => request({ op: "inspect", entity, component }).result.entity.components[component];
      const state = () => ({ observation: request({ op: "status" }).observation,
        player: inspect("player", "GlobalTransform"), child: inspect("accessory", "GlobalTransform"),
        presentedPlayer: inspect("player", "PresentedTransform"), presentedChild: inspect("accessory", "PresentedTransform") });
      const frame = timestamp => game.frame(timestamp, 320, 240, 1, 0, 0, 0, 0, 0, 0, 0);
      frame(0);
      const before = state();
      frame(1000 / 60 + 0.00001);
      const afterTick = state();
      frame(25);
      const halfTick = state();
      game.suspend();
      frame(5000);
      const resumed = state();
      return { before, afterTick, halfTick, resumed };
    } finally { game?.free(); canvas.remove(); }
  }, { moduleUrl: [...bindingUrls][0], fixture });
  assert.equal(ordinaryFrames.before.observation.tick, "0");
  assert.equal(ordinaryFrames.afterTick.observation.tick, "1");
  assert.equal(ordinaryFrames.halfTick.observation.tick, "1");
  presentation(ordinaryFrames.halfTick, "interpolated", 0.5);
  assert.ok(ordinaryFrames.before.player.translation[2] - ordinaryFrames.afterTick.player.translation[2] > 0.1);
  for (const [authoritative, presented] of [["player", "presentedPlayer"], ["child", "presentedChild"]]) {
    assert.deepEqual(ordinaryFrames.halfTick[authoritative], ordinaryFrames.afterTick[authoritative]);
    const expected = ordinaryFrames.before[authoritative].matrix.map((value, index) => (value + ordinaryFrames.afterTick[authoritative].matrix[index]) / 2);
    near(ordinaryFrames.halfTick[presented].matrix, expected, `ordinary frame ${presented}`);
    near(ordinaryFrames.resumed[presented].matrix, ordinaryFrames.afterTick[authoritative].matrix, `ordinary resume ${presented}`);
  }
  assert.equal(ordinaryFrames.resumed.observation.tick, "1", "Suspended wall time must not advance gameplay.");
  checks.push("ordinary_frame_half_tick_motion", "ordinary_frame_suspension_baseline");
  for (const name of ["PresentedTransform", "InterpolateTransform", "PresentationState"]) assert.ok((await command("describe", { name })).result.descriptor.schema);
  assert.deepEqual(await inspect("player", "InterpolateTransform"), { enabled: true });
  checks.push("descriptor_discovery", "explicit_game_opt_in");

  const { before, after } = await moveOne();
  const fractions = [];
  for (const alpha of [0, 0.25, 0.5, 0.75, 1]) {
    const reply = await command("present", { alpha });
    presentation(reply, "interpolated", alpha);
    for (const [id, key] of [["player", "playerGlobal"], ["accessory", "childGlobal"]]) {
      const matrix = before[key].matrix.map((value, index) => value + (after[key].matrix[index] - value) * alpha);
      await presentedMatches(id, matrix, `${id} alpha ${alpha}`);
    }
    assert.deepEqual(await authority(), after, "Preview must preserve gameplay, local/world authority and tick.");
    fractions.push({ alpha, player: (await inspect("player", "PresentedTransform")).translation });
  }
  const half = await capture("half-tick", 0.5);
  const exact = await capture("exact-current");
  const visualDifference = await changedPixels(half, exact);
  assert.ok(visualDifference > 20, "The fractional pose must change visible pixels.");
  await presentedMatches("player", after.playerGlobal.matrix, "exact capture");
  await command("present", { alpha: 0.5 });
  presentation(await command("render"), "current", 1);
  await presentedMatches("player", after.playerGlobal.matrix, "exact render");
  checks.push("fractional_parent_child_pose", "unchanged_authority_and_gameplay", "fractional_capture_pixels", "exact_step_render_capture");

  await command("present", { alpha: 0.5 });
  const preview = await inspect("player", "PresentedTransform");
  const invalid = [];
  for (const op of ["present", "capture"]) for (const alpha of [-0.01, 1.01, null, "0.5"]) {
    const reply = await command(op, { alpha }, false);
    assert.equal(reply.observation.tick, after.tick);
    assert.ok(reply.error.code);
    assert.deepEqual(await inspect("player", "PresentedTransform"), preview);
    invalid.push({ op, alpha, code: reply.error.code });
  }
  await command("pause");
  assert.deepEqual(await inspect("player", "PresentedTransform"), preview, "Repeated pause is a no-op.");
  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  await command("present", { alpha: 0 });
  await presentedMatches("player", after.playerGlobal.matrix, "focus loss resets history");
  await presentedMatches("accessory", after.childGlobal.matrix, "focus loss resets inherited history");
  const movedAgain = await moveOne();
  await command("present", { alpha: 0.5 });
  await page.evaluate(async () => {
    const resumed = await window.__engineDev.request({ op: "resume" });
    const paused = await window.__engineDev.request({ op: "pause" });
    if (!resumed.ok || !paused.ok) throw new Error("resume/pause failed");
  });
  await command("present", { alpha: 0 });
  await presentedMatches("player", movedAgain.after.playerGlobal.matrix, "pause transition resets history");
  await page.locator("#reset").click();
  await command("present", { alpha: 0 });
  const reset = await authority();
  near(reset.playerGlobal.translation, before.playerGlobal.translation, "reset teleports to spawn");
  await presentedMatches("player", reset.playerGlobal.matrix, "reset has no stale pose");
  await presentedMatches("accessory", reset.childGlobal.matrix, "reset snaps descendants");
  checks.push("invalid_fraction_preserves_state", "repeated_pause_noop", "focus_and_pause_reset_history", "teleport_reset_snaps_subtree");

  const teleported = structuredClone(fixture);
  teleported.entities[0].transform.translation = [0.7, 0.15, -0.4];
  await reload(teleported);
  await command("present", { alpha: 0 });
  const replacement = await authority();
  near(replacement.playerGlobal.translation, teleported.entities[0].transform.translation, "reload accepts teleported source");
  await presentedMatches("player", replacement.playerGlobal.matrix, "reload discards former history");
  await presentedMatches("accessory", replacement.childGlobal.matrix, "reload snaps child");
  const assetBytes = (await command("assets")).result.resident_cpu_bytes;
  const history = [];
  for (let repeat = 0; repeat < 3; repeat++) {
    await reload(fixture);
    await moveOne();
    const reply = await command("present", { alpha: 0.5 });
    history.push(presentation(reply, "interpolated", 0.5));
    assert.equal((await command("assets")).result.resident_cpu_bytes, assetBytes);
  }
  assert.ok(history.every(value => ["history_nodes", "history_capacity", "history_bytes", "presentation_scratch_bytes"].every(field => value[field] === history[0][field])), "Repeated replacements must not grow retained presentation storage.");
  const preserved = await authority(), preservedPreview = await inspect("player", "PresentedTransform");
  served = structuredClone(fixture); served.revision = revision + 1; served.entities[1].parent = "missing";
  const rejected = await command("reload", { expected_revision: String(revision), revision: String(served.revision) }, false);
  assert.equal(rejected.error.code, "unknown_parent");
  assert.equal(rejected.observation.scene.revision, String(revision));
  assert.deepEqual(await authority(), preserved);
  assert.deepEqual(await inspect("player", "PresentedTransform"), preservedPreview);
  checks.push("reload_resets_history", "repeated_reload_bounded_storage", "rejected_reload_preserves_preview");

  await reload(fixture);
  await command("camera", { position: [0, 0.15, 2], target: [0, 0.15, 0], vertical_fov_radians: Math.PI / 3, near: 0.1, far: 2 });
  await moveOne();
  const edgeCases = [];
  for (const [alpha, visible] of [[0, 2], [0.5, 1], [1, 0]]) {
    await command("culling", { enabled: true });
    const on = await capture(`far-edge-${alpha}-on`, alpha);
    assert.equal(on.reply.result.rendering.render_candidates, 2);
    assert.equal(on.reply.result.rendering.visible_instances, visible, `alpha ${alpha} culling must use the presented pose`);
    assert.equal(on.reply.result.rendering.culled_instances, 2 - visible);
    await command("culling", { enabled: false });
    const off = await capture(`far-edge-${alpha}-off`, alpha);
    assert.equal(off.reply.result.rendering.instances, 2);
    const pixels = await changedPixels(on, off);
    assert.equal(pixels, 0, "Presented-pose culling must preserve rendered pixels.");
    edgeCases.push({ alpha, visible, culled: 2 - visible, changedPixels: pixels });
  }
  checks.push("culling_uses_presented_pose", "culling_pixel_equivalence");
  const removed = structuredClone(fixture); removed.entities = removed.entities.filter(entity => entity.id !== "accessory");
  await reload(removed);
  const removal = await command("present", { alpha: 0.5 });
  assert.ok(removal.observation.presentation.presented_nodes < history[0].presented_nodes);
  assert.equal((await command("inspect", { entity: "accessory" }, false)).error.code, "unknown_entity");
  assert.deepEqual(errors, []);
  checks.push("removed_entities_retire_presented_components", "clean_browser_diagnostics");
  console.log(JSON.stringify({ ok: true, url: url.href, browser: browser.version(), artifact: output, fractions, visualDifference,
    ordinaryFrame: { tick: ordinaryFrames.halfTick.observation.tick, alpha: ordinaryFrames.halfTick.observation.presentation.alpha,
      authoritativePlayer: ordinaryFrames.halfTick.player.translation, presentedPlayer: ordinaryFrames.halfTick.presentedPlayer.translation },
    history: history[0], afterRemoval: removal.observation.presentation, edgeCases, invalid, checks }));
} finally {
  clearTimeout(suiteTimer);
  await browser.close();
}
