// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, differentPixels, gridScene, launch, openPaused, routedScene, sceneUrl } from "./harness.mjs";
import { openTerrain } from "./terrain-fixture.mjs";

const url = sceneUrl();
const output = await artifacts("debug-view-smoke");
const { browser, page, errors } = await launch({ timeout: 180_000 });

const command = commander(page);

async function settle() {
  for (let frame = 0; frame < 600; frame++) {
    if (!(await command("present", { alpha: 0 })).result.rendering.pipelines_compiling) break;
    await delay(50);
  }
  let rendering, previous, stable = 0;
  for (let frame = 1; frame <= 8 || stable < 5; frame++) {
    assert.ok(frame <= 600, `read-back counts settle: ${JSON.stringify(rendering)}`);
    rendering = (await command("present", { alpha: 0 })).result.rendering;
    const counts = JSON.stringify([rendering.level_of_detail.lod0_instances, rendering.level_of_detail.lod1_instances,
      rendering.level_of_detail.lod2_instances, rendering.level_of_detail.impostor_instances]);
    const current = rendering.read_back_age != null && rendering.read_back_age < frame;
    stable = current && counts === previous ? stable + 1 : 0;
    previous = current ? counts : undefined;
    await delay(20);
  }
  return rendering;
}

async function capture(name) {
  await settle();
  const reply = await command("capture", { alpha: 0 });
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { rendering: reply.result.rendering, png: bytes.toString("base64") };
}

async function exact(name) {
  let previous;
  for (let attempt = 0; attempt < 20; attempt++) {
    const reply = await command("capture", { alpha: 0 });
    const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
    const frame = { rendering: reply.result.rendering, png: bytes.toString("base64") };
    if (previous && (await differentPixels(page, [previous.png, frame.png])).changed === 0) return frame;
    previous = frame;
  }
  assert.fail(`${name}: two exact captures running agree`);
}

async function shownView() {
  return (await command("inspect", { component: "DebugView" })).result.value.view;
}

const views = ["kinds", "levels", "cards", "shadows", "casters", "overdraw", "normals"];
const terrainViews = ["levels", "tiles", "materials", "sky", "detail"];
const waterViews = ["water_depth", "water_speed", "water_froude", "water_turbulence", "water_white", "water_reach",
  "water_span", "water_eddies", "water_roughness"];
const report = {};
const kinds = ["mesh", "placed", "card", "placed_card", "terrain", "cover", "water", "falls", "sky"];
const slotted = ["mesh", "card"];

try {
  await routedScene(page, gridScene()).route();
  await openPaused(page, url, { timeout: 120_000 });
  await command("environment", { reflection_probes: { enabled: false } });

  const listing = await command("describe");
  assert.ok(listing.result.types.some(type => type.name === "DebugView"));
  const look = await command("inspect", { component: "Environment" });
  assert.deepEqual(look.result.value.debug_view, { view: "off", shade: true });

  const shaded = await capture("shaded");
  assert.ok(!shaded.rendering.render_passes.includes("debug_view"), JSON.stringify(shaded.rendering.render_passes));
  const pipelines = shaded.rendering.pipelines;
  const before = await exact("exact");

  const shown = {};
  for (const view of views) {
    const changed = await command("environment", { debug_view: { view } });
    assert.deepEqual(changed.result.environment_changed, ["debug_view"]);
    assert.equal(await shownView(), view);
    const frame = await capture(view);
    assert.ok(frame.rendering.render_passes.includes("debug_view"), JSON.stringify(frame.rendering.render_passes));
    assert.ok(frame.rendering.draw_calls > 0, JSON.stringify(frame.rendering.draw_calls));
    const pixels = await differentPixels(page, [shaded.png, frame.png], { tolerance: 8 });
    assert.ok(pixels.changed > pixels.total / 4, `${view} shows other colors than the shaded image: ${JSON.stringify(pixels)}`);
    shown[view] = { ...frame, pixels };
  }
  const levels = await differentPixels(page, [shown.kinds.png, shown.levels.png], { tolerance: 8 });
  assert.ok(levels.changed > 0, "levels differ from kinds");
  assert.ok(shown.overdraw.rendering.pipelines > pipelines, "views create their pipelines when shown");

  await command("environment", { debug_view: { view: "off" } });
  assert.equal(await shownView(), "off");
  const again = await capture("shaded-again");
  assert.ok(!again.rendering.render_passes.includes("debug_view"), JSON.stringify(again.rendering.render_passes));
  assert.equal(again.rendering.pipelines_created, 0);
  const same = await differentPixels(page, [before.png, (await exact("exact-again")).png]);
  assert.equal(same.changed, 0, `the shaded image returns exactly: ${JSON.stringify(same)}`);

  const unknown = await command("environment", { debug_view: { view: "everything" } }, false);
  assert.equal(unknown.error.path.startsWith("$.debug_view"), true, JSON.stringify(unknown.error));

  const { display_width: width, display_height: height } = again.rendering.resolution;
  const probes = [];
  for (const fraction of [0.3, 0.4, 0.5, 0.6, 0.7]) {
    const point = { x: Math.floor(width * fraction), y: Math.floor(height * 0.55) };
    const probed = (await command("probe", { ...point, alpha: 0 })).result;
    assert.deepEqual(probed.pixel, [point.x, point.y]);
    assert.ok(kinds.includes(probed.kind), JSON.stringify(probed));
    assert.equal(probed.shaded.length, 3, JSON.stringify(probed));
    if (probed.kind !== "sky") {
      assert.ok(probed.depth > 0 && probed.depth <= 1, JSON.stringify(probed));
      assert.ok(probed.world.every(Number.isFinite), JSON.stringify(probed));
      assert.ok(probed.fragments >= 1, JSON.stringify(probed));
      assert.ok(Math.abs(Math.hypot(...probed.normal) - 1) < 0.01, JSON.stringify(probed));
    }
    if (slotted.includes(probed.kind)) {
      assert.equal(typeof probed.object, "number", JSON.stringify(probed));
      assert.ok(probed.entity, JSON.stringify(probed));
    }
    probes.push(probed);
  }
  assert.ok(probes.some(probed => slotted.includes(probed.kind)), `an entity drew one of the points: ${JSON.stringify(probes.map(probed => probed.kind))}`);
  const sky = (await command("probe", { x: Math.floor(width / 2), y: 2, alpha: 0 })).result;
  assert.ok(kinds.includes(sky.kind), JSON.stringify(sky));
  const outside = await command("probe", { x: width, y: 0 }, false);
  assert.equal(outside.error.code, "out_of_range");
  assert.equal(outside.error.path, "$.x");
  const after = await exact("exact-after-probe");
  assert.equal((await differentPixels(page, [before.png, after.png])).changed, 0);

  await openPaused(page, sceneUrl({ "debug-view": "cards" }), { timeout: 120_000 });
  assert.equal(await shownView(), "cards");

  assert.deepEqual(errors, []);
  report.views = Object.fromEntries(Object.entries(shown).map(([view, frame]) =>
    [view, { draws: frame.rendering.draw_calls, changed_pixels: frame.pixels.changed, pipelines: frame.rendering.pipelines }]));
  report.probes = probes;
  report.sky = sky;
} finally {
  await browser.close();
}

const terrain = await openTerrain("debug-view-terrain");
try {
  const shaded = await terrain.capture("terrain-shaded");
  report.terrain_views = {};
  for (const view of [...waterViews, ...terrainViews]) {
    await terrain.command("environment", { debug_view: { view } });
    const png = await terrain.capture(`terrain-${view}`);
    const pixels = await differentPixels(terrain.page, [shaded, png], { tolerance: 8 });
    assert.ok(pixels.changed > pixels.total / 4, `${view} shows the fixture otherwise: ${JSON.stringify(pixels)}`);
    report.terrain_views[view] = pixels.changed;
  }
  const { display_width: width, display_height: height } = (await terrain.command("render")).result.rendering.resolution;
  let ground;
  for (const [x, y] of [[0.5, 0.75], [0.4, 0.8], [0.6, 0.8], [0.5, 0.9], [0.3, 0.7], [0.7, 0.7]]) {
    const probed = (await terrain.command("probe", { x: Math.floor(width * x), y: Math.floor(height * y) })).result;
    if (probed.kind === "terrain") {
      ground = probed;
      break;
    }
  }
  assert.ok(ground, "the probe finds the fixture's ground");
  const made = ground.terrain;
  assert.ok(made && made.tile.every(Number.isInteger) && made.patch.every(value => value >= 0 && value < 16), JSON.stringify(ground));
  assert.ok(ground.level >= 0 && ground.level < 8, JSON.stringify(ground));
  assert.ok(made.materials.length > 0 && made.materials.every(share => share.share > 0 && share.share <= 1), JSON.stringify(made));
  const total = made.materials.reduce((sum, share) => sum + share.share, 0);
  assert.ok(Math.abs(total - 1) < 0.05, `the materials' shares make the whole: ${JSON.stringify(made.materials)}`);
  assert.ok(made.sky >= 0 && made.sky <= 1, JSON.stringify(made));
  assert.ok(made.detail.detailed >= made.detail.averaged - 0.01, JSON.stringify(made.detail));
  assert.equal(ground.water, null, JSON.stringify(ground));
  const above = (await terrain.command("probe", { x: Math.floor(width / 2), y: 1 })).result;
  assert.equal(above.terrain, null, JSON.stringify(above));
  assert.equal(above.water, null, JSON.stringify(above));
  await terrain.command("environment", { debug_view: { view: "off" } });
  assert.deepEqual(terrain.errors, []);
  report.ground = ground;
} finally {
  await terrain.close();
}

console.log(JSON.stringify({
  ok: true, url: url.href, artifact: output, ...report,
  checks: ["setting_discoverable", "views_draw", "views_create_pipelines_when_shown", "off_is_exact", "unknown_view_refused",
    "probe_answers", "probe_refuses_outside", "probe_keeps_image", "host_switch", "clean_browser_diagnostics",
    "water_views_draw", "terrain_views_draw", "terrain_probe"],
}));
