/** Levels of detail and impostors in the Stress Lab's metro, in real Chrome. */
// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused } from "./harness.mjs";

const config = {
  seed: 1337, renderables: 20000, moving_percent: 10, hierarchy: true, interpolation: true, colliders: 0,
  sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 64, textured: true, culling: true,
  scene: "metro", compressed_textures: false,
};
const url = devUrl("stress.html", { config });
const output = await artifacts("lod-smoke");
const { browser, page, errors } = await launch({ timeout: 180_000 });

const command = commander(page);

// Counts of instances per level are read back a few frames late, later
// while the GPU falls behind, and the first GPU frames compile pipelines
// and bake impostors: present frames of the same view until counts read
// back from frames presented here (fewer frames late than were presented)
// hold for several frames running.
async function present(gpu) {
  let rendering, previous, stable = 0;
  for (let frame = 1; frame <= 8 || stable < 5; frame++) {
    assert.ok(frame <= 600, `read-back counts settle: ${JSON.stringify(rendering)}`);
    rendering = (await command("present", { alpha: 0, gpu })).result.rendering;
    const counts = JSON.stringify([rendering.occlusion_culling.instances, rendering.level_of_detail.lod0_instances,
      rendering.level_of_detail.lod1_instances, rendering.level_of_detail.lod2_instances, rendering.level_of_detail.impostor_instances]);
    const current = !gpu || (rendering.occlusion_culling.instances != null && rendering.read_back_age < frame);
    stable = current && counts === previous ? stable + 1 : 0;
    previous = current ? counts : undefined;
    await delay(20);
  }
  return rendering;
}

async function capture(name, gpu) {
  // Pipelines go on compiling for seconds after the frames that first
  // need them, and a capture presents only once they have; the frames
  // after them settle what the view draws.
  for (let frame = 0; frame < 600; frame++) {
    if (!(await command("present", { alpha: 0, gpu })).result.rendering.pipelines_compiling) break;
    await delay(50);
  }
  const rendering = await present(gpu);
  await command("capture", { alpha: 0, gpu });
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { rendering, png: bytes.toString("base64") };
}

const coarser = rendering => (rendering.level_of_detail.lod1_instances ?? 0) + (rendering.level_of_detail.lod2_instances ?? 0)
  + (rendering.level_of_detail.impostor_instances ?? 0);


try {
  await openPaused(page, url, { timeout: 120_000 });

  // Reflection probes capture their cubes over many frames, which would
  // change reflections between captures taken apart.
  await command("environment", { reflection_probes: { enabled: false } });

  // The settings are discoverable and read as their scene section.
  const listing = await command("describe");
  assert.ok(listing.result.types.some(type => type.name === "LevelOfDetail"));
  const look = await command("inspect", { component: "Environment" });
  assert.deepEqual(look.result.value.level_of_detail, { enabled: true, pixel_error: 1, impostors: true, fade_seconds: 0.5 });

  // The Stress Lab's own controls run through the game control, refusing
  // what they do not know; the page lists them for the bridge's describe.
  const controls = await page.evaluate(() => window.__engineDev.controls);
  assert.deepEqual(Object.keys(controls).sort(), ["effects", "look", "view", "workload"]);
  const viewed = await command("game", { name: "view", args: { name: "overview" } });
  assert.deepEqual(viewed.result.result, { view: "overview" });
  const unknownPreset = await command("game", { name: "workload", args: { preset: "nowhere" } }, false);
  assert.equal(unknownPreset.error.code, "game_rule");

  // From the overview, GPU-presented frames draw distant props at coarser
  // levels and the smallest as impostors, baked the first frame their draw
  // groups draw. How many pixels each way of drawing changes against the
  // exact CPU reference is measured, not judged: small objects shift by up
  // to a pixel where impostors take over, so differences run long, and
  // shading gone wrong would change far more.
  const measurements = [];
  const differs = (name, pixels, note) => measurements.push({
    name, value: pixels.changed / pixels.total * 100, unit: "%",
    note: `pixels whose largest channel differs by more than 8 levels from the CPU reference, ${note}`,
  });
  const cpu = await capture("overview-cpu", false);
  const gpu = await capture("overview-gpu", true);
  assert.ok(coarser(gpu.rendering) > 0, `coarser levels draw: ${JSON.stringify(gpu.rendering)}`);
  assert.ok(gpu.rendering.level_of_detail.impostors > 0 && gpu.rendering.level_of_detail.impostor_instances > 0,
    `impostors draw: ${JSON.stringify(gpu.rendering)}`);
  assert.equal(gpu.rendering.level_of_detail.dropped_impostors, 0);
  const pixels = await differentPixels(page, [cpu.png, gpu.png], { tolerance: 8 });
  differs("impostor difference", pixels, "with mesh levels and impostors");

  // Without impostors, the mesh levels alone.
  await command("environment", { level_of_detail: { impostors: false } });
  const levels = await capture("overview-levels", true);
  assert.equal(levels.rendering.level_of_detail.impostor_instances ?? 0, 0, JSON.stringify(levels.rendering));
  assert.ok(coarser(levels.rendering) > 0, `coarser levels draw: ${JSON.stringify(levels.rendering)}`);
  const levelPixels = await differentPixels(page, [cpu.png, levels.png], { tolerance: 8 });
  differs("mesh level difference", levelPixels, "with mesh levels alone");

  // Disabled, every instance draws its mesh, as the CPU reference does.
  const disabled = await command("environment", { level_of_detail: { enabled: false } });
  assert.deepEqual(disabled.result.environment_changed, ["level_of_detail"]);
  const exact = await capture("overview-exact", true);
  assert.equal(coarser(exact.rendering), 0, JSON.stringify(exact.rendering));
  const exactPixels = await differentPixels(page, [cpu.png, exact.png], { tolerance: 8 });
  differs("disabled difference", exactPixels, "with levels of detail off");

  // A budget keeps that many impostors baked and reports the draw groups it
  // leaves drawing their meshes.
  await command("environment", { level_of_detail: { max_impostors: 2 } });
  const budget = await present(true);
  assert.ok(budget.level_of_detail.impostors > 0 && budget.level_of_detail.impostors <= 2 && budget.level_of_detail.dropped_impostors > 0,
    `budget: ${JSON.stringify(budget)}`);

  // A larger pixel error draws more instances coarser; invalid settings are
  // refused. The control changes nothing in source.
  await command("environment", { level_of_detail: { enabled: true, pixel_error: 4 } });
  const coarse = await present(true);
  assert.ok(coarser(coarse) > coarser(gpu.rendering),
    `pixel error 4 draws more coarsely (${coarser(coarse)} against ${coarser(gpu.rendering)})`);
  const invalid = await command("environment", { level_of_detail: { pixel_error: 0.1 } }, false);
  assert.equal(invalid.error.code, "invalid_environment");
  assert.equal(invalid.error.path, "$.level_of_detail.pixel_error");
  await command("environment", { level_of_detail: { enabled: true, pixel_error: 1 } });

  assert.deepEqual(errors, []);
  console.log(JSON.stringify({
    ok: true, url: url.href, artifact: output,
    overview: {
      changed_pixels: pixels.changed, level_changed_pixels: levelPixels.changed, total_pixels: pixels.total,
      exact_changed_pixels: exactPixels.changed,
      levels: [gpu.rendering.level_of_detail.lod0_instances, gpu.rendering.level_of_detail.lod1_instances, gpu.rendering.level_of_detail.lod2_instances,
        gpu.rendering.level_of_detail.impostor_instances],
      impostors: gpu.rendering.level_of_detail.impostors,
      coarse_levels: [coarse.level_of_detail.lod0_instances, coarse.level_of_detail.lod1_instances, coarse.level_of_detail.lod2_instances, coarse.level_of_detail.impostor_instances],
      budget: { impostors: budget.level_of_detail.impostors, dropped: budget.level_of_detail.dropped_impostors },
    },
    checks: ["settings_discoverable", "coarser_levels_draw", "impostors_draw", "disabled_draws_meshes", "impostor_budget", "pixel_error_control", "invalid_settings_refused", "clean_browser_diagnostics"],
    measurements,
  }));
} finally {
  await browser.close();
}
