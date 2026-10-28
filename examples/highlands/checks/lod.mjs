// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { artifacts, commander, devUrl, differentPixels, launch, openPaused } from "../../../web/scripts/harness.mjs";

const url = devUrl("highlands.html");
const output = await artifacts("highlands-lod");
const { browser, page, errors } = await launch({ timeout: 180_000 });

const command = commander(page);

async function present(gpu) {
  let rendering, previous, stable = 0;
  for (let frame = 1; frame <= 8 || stable < 5; frame++) {
    assert.ok(frame <= 600, `read-back counts settle: ${JSON.stringify(rendering)}`);
    rendering = (await command("present", { alpha: 0 })).result.rendering;
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
  for (let frame = 0; frame < 600; frame++) {
    if (!(await command("present", { alpha: 0 })).result.rendering.pipelines_compiling) break;
    await delay(50);
  }
  const rendering = await present(gpu);
  await command("capture", { alpha: 0 });
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { rendering, png: bytes.toString("base64") };
}

const coarser = rendering => (rendering.level_of_detail.lod1_instances ?? 0) + (rendering.level_of_detail.lod2_instances ?? 0)
  + (rendering.level_of_detail.impostor_instances ?? 0);


try {
  await openPaused(page, url, { timeout: 120_000 });

  await command("environment", { reflection_probes: { enabled: false } });

  const listing = await command("describe");
  assert.ok(listing.result.types.some(type => type.name === "LevelOfDetail"));
  const look = await command("inspect", { component: "Environment" });
  assert.deepEqual(look.result.value.level_of_detail, { enabled: true, pixel_error: 1, impostors: true, fade_seconds: 0.5 });

  const viewed = await command("game", { name: "view", args: { name: "overview" } });
  assert.deepEqual(viewed.result.result, { view: "overview" });

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
  assert.equal(gpu.rendering.level_of_detail.meshes_without_levels, 0);
  const pixels = await differentPixels(page, [cpu.png, gpu.png], { tolerance: 8 });
  differs("impostor difference", pixels, "with mesh levels and impostors");

  await command("environment", { level_of_detail: { impostors: false } });
  const levels = await capture("overview-levels", true);
  assert.equal(levels.rendering.level_of_detail.impostor_instances ?? 0, 0, JSON.stringify(levels.rendering));
  assert.ok(coarser(levels.rendering) > 0, `coarser levels draw: ${JSON.stringify(levels.rendering)}`);
  const levelPixels = await differentPixels(page, [cpu.png, levels.png], { tolerance: 8 });
  differs("mesh level difference", levelPixels, "with mesh levels alone");

  const disabled = await command("environment", { level_of_detail: { enabled: false } });
  assert.deepEqual(disabled.result.environment_changed, ["level_of_detail"]);
  const exact = await capture("overview-exact", true);
  assert.equal(coarser(exact.rendering), 0, JSON.stringify(exact.rendering));
  const exactPixels = await differentPixels(page, [cpu.png, exact.png], { tolerance: 8 });
  differs("disabled difference", exactPixels, "with levels of detail off");

  await command("environment", { level_of_detail: { max_impostors: 2 } });
  const budget = await present(true);
  assert.ok(budget.level_of_detail.impostors > 0 && budget.level_of_detail.impostors <= 2 && budget.level_of_detail.dropped_impostors > 0,
    `budget: ${JSON.stringify(budget)}`);

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
