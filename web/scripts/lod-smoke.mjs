/** Levels of detail and impostors (decisions 047 and 048) in the Stress Lab's metro, in real Chrome. */
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

// Counts of instances per level are read back a few frames late, and the
// first GPU frames compile pipelines and bake impostors: present frames of
// the same view until the counts hold for several frames running.
async function present(gpu) {
  let rendering, previous, stable = 0;
  for (let frame = 0; frame < 8 || stable < 5; frame++) {
    assert.ok(frame < 600, `read-back counts settle: ${JSON.stringify(rendering)}`);
    rendering = (await command("present", { alpha: 0, gpu })).result.rendering;
    const counts = JSON.stringify([rendering.occluded_instances, rendering.lod0_instances,
      rendering.lod1_instances, rendering.lod2_instances, rendering.impostor_instances]);
    stable = counts === previous && (!gpu || rendering.occluded_instances != null) ? stable + 1 : 0;
    previous = counts;
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

const coarser = rendering => (rendering.lod1_instances ?? 0) + (rendering.lod2_instances ?? 0)
  + (rendering.impostor_instances ?? 0);


try {
  await openPaused(page, url, { timeout: 120_000 });

  // Reflection probes capture their cubes over many frames, which would
  // change reflections between captures taken apart.
  await command("environment", { reflection_probes: { enabled: false } });

  // The settings are discoverable and read as their scene section.
  const listing = await command("describe");
  assert.ok(listing.result.types.some(type => type.name === "LevelOfDetail"));
  const look = await command("inspect", { component: "Environment" });
  assert.deepEqual(look.result.value.level_of_detail, { enabled: true, pixel_error: 1, impostors: true });

  // From the overview, GPU-presented frames draw distant props at coarser
  // levels and the smallest as impostors, baked the first frame their draw
  // groups draw; both stay within a pixel of the exact CPU reference, so
  // only a small share of pixels changes.
  const cpu = await capture("overview-cpu", false);
  const gpu = await capture("overview-gpu", true);
  assert.ok(coarser(gpu.rendering) > 0, `coarser levels draw: ${JSON.stringify(gpu.rendering)}`);
  assert.ok(gpu.rendering.impostors > 0 && gpu.rendering.impostor_instances > 0,
    `impostors draw: ${JSON.stringify(gpu.rendering)}`);
  assert.equal(gpu.rendering.dropped_impostors, 0);
  const pixels = await differentPixels(page, [cpu.png, gpu.png], { tolerance: 8 });
  // Small objects shift by up to a pixel where impostors take over, so
  // differences run long; shading gone wrong would change far more.
  assert.ok(pixels.changed <= pixels.total * 0.05, `impostors: ${pixels.changed} of ${pixels.total} pixels differ`);

  // Without impostors, the mesh levels alone.
  await command("environment", { level_of_detail: { impostors: false } });
  const levels = await capture("overview-levels", true);
  assert.equal(levels.rendering.impostor_instances ?? 0, 0, JSON.stringify(levels.rendering));
  assert.ok(coarser(levels.rendering) > 0, `coarser levels draw: ${JSON.stringify(levels.rendering)}`);
  const levelPixels = await differentPixels(page, [cpu.png, levels.png], { tolerance: 8 });
  assert.ok(levelPixels.changed <= levelPixels.total * 0.01, `levels: ${levelPixels.changed} of ${levelPixels.total} pixels differ`);

  // Disabled, every instance draws its mesh and the frames match exactly.
  const disabled = await command("environment", { level_of_detail: { enabled: false } });
  assert.deepEqual(disabled.result.environment_changed, ["level_of_detail"]);
  const exact = await capture("overview-exact", true);
  assert.equal(coarser(exact.rendering), 0, JSON.stringify(exact.rendering));
  const exactPixels = await differentPixels(page, [cpu.png, exact.png], { tolerance: 8 });
  assert.ok(exactPixels.changed <= exactPixels.total * 0.001, `disabled: ${exactPixels.changed} pixels differ`);

  // A budget keeps that many impostors baked and reports the draw groups it
  // leaves drawing their meshes.
  await command("environment", { level_of_detail: { max_impostors: 2 } });
  const budget = await present(true);
  assert.ok(budget.impostors > 0 && budget.impostors <= 2 && budget.dropped_impostors > 0,
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
      levels: [gpu.rendering.lod0_instances, gpu.rendering.lod1_instances, gpu.rendering.lod2_instances,
        gpu.rendering.impostor_instances],
      impostors: gpu.rendering.impostors,
      coarse_levels: [coarse.lod0_instances, coarse.lod1_instances, coarse.lod2_instances, coarse.impostor_instances],
      budget: { impostors: budget.impostors, dropped: budget.dropped_impostors },
    },
    checks: ["settings_discoverable", "coarser_levels_draw", "impostors_draw", "impostors_within_pixels", "levels_within_pixels", "disabled_exact", "impostor_budget", "pixel_error_control", "invalid_settings_refused", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
