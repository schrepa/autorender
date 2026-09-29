/** Levels of detail (decision 047) in the Stress Lab's metro, in real Chrome. */
import assert from "node:assert/strict";
import { randomUUID } from "node:crypto";
import { mkdir } from "node:fs/promises";
import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { setTimeout as delay } from "node:timers/promises";
import { chromium } from "@playwright/test";

const base = `${(process.env.DEMO_URL ?? "http://127.0.0.1:5173").replace(/\/$/, "")}/`;
const config = {
  seed: 1337, renderables: 20000, moving_percent: 10, hierarchy: true, interpolation: true, colliders: 0,
  sensor_percent: 0, dense_colliders: false, queries_per_tick: 0, mesh_variety: 64, textured: true, culling: true,
  scene: "metro", compressed_textures: false,
};
const url = new URL("stress.html", base);
url.searchParams.set("engine_dev", "1");
url.searchParams.set("config", JSON.stringify(config));
const output = fileURLToPath(new URL(`../../artifacts/lod-smoke-${randomUUID()}/`, import.meta.url));
await mkdir(output, { recursive: true });
const browser = await chromium.launch({
  ...(process.env.BROWSER_PATH ? { executablePath: resolve(process.env.BROWSER_PATH) } : { channel: process.env.BROWSER_CHANNEL ?? "chrome" }),
  headless: true,
  args: ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])],
});
const page = await browser.newPage({ viewport: { width: 1200, height: 900 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(180_000);
const errors = [];
const recordError = value => { if (errors.length < 20) errors.push(String(value).slice(0, 2000)); };
page.on("pageerror", recordError);
page.on("console", entry => { if (entry.type() === "error") recordError(entry.text()); });

async function command(op, fields = {}, expectedOk = true) {
  for (;;) {
    const reply = await page.evaluate(command => window.__engineDev.request(command), { op, ...fields });
    if (reply?.error?.code === "command_busy") { await delay(50); continue; }
    assert.equal(reply.ok, expectedOk, `${op}: ${JSON.stringify(reply).slice(0, 4000)}`);
    return reply;
  }
}

// Counts of instances per level are read back a few frames late: present a
// few frames of the same view before reading them.
async function present(gpu) {
  let rendering;
  for (let frame = 0; frame < 8; frame++) {
    rendering = (await command("present", { alpha: 0, gpu })).result.rendering;
    await delay(20);
  }
  return rendering;
}

async function capture(name, gpu) {
  const rendering = await present(gpu);
  await command("capture", { alpha: 0, gpu });
  const bytes = await page.locator("canvas").screenshot({ path: `${output}/${name}.png` });
  return { rendering, png: bytes.toString("base64") };
}

const coarser = rendering => (rendering.lod1_instances ?? 0) + (rendering.lod2_instances ?? 0);

// Pixels whose largest channel difference exceeds the tolerance.
async function differentPixels(images, tolerance) {
  return page.evaluate(async ({ images, tolerance }) => {
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
    let changed = 0;
    for (let offset = 0; offset < pixels[0].data.length; offset += 4) {
      if ([0, 1, 2].some(channel => Math.abs(pixels[0].data[offset + channel] - pixels[1].data[offset + channel]) > tolerance)) changed++;
    }
    return { changed, total: pixels[0].width * pixels[0].height };
  }, { images, tolerance });
}

try {
  await page.goto(url.href, { waitUntil: "domcontentloaded" });
  await page.waitForFunction(() => window.__engineDev?.version === 1, null, { timeout: 120_000 });
  for (let attempt = 0; ; attempt++) {
    const reply = await page.evaluate(() => window.__engineDev.request({ op: "pause" }));
    if (reply.ok) break;
    assert.ok(attempt < 200, `pause: ${JSON.stringify(reply)}`);
    await delay(100);
  }

  // The settings are discoverable and read as their scene section.
  const listing = await command("describe");
  assert.ok(listing.result.types.some(type => type.name === "LevelOfDetail"));
  const look = await command("inspect", { component: "Environment" });
  assert.deepEqual(look.result.value.level_of_detail, { enabled: true, pixel_error: 1 });

  // From the overview, GPU-presented frames draw distant props at coarser
  // levels, which stay within a pixel of the exact CPU reference: only a
  // small share of pixels changes.
  const cpu = await capture("overview-cpu", false);
  const gpu = await capture("overview-gpu", true);
  assert.ok(coarser(gpu.rendering) > 0, `coarser levels draw: ${JSON.stringify(gpu.rendering)}`);
  const pixels = await differentPixels([cpu.png, gpu.png], 8);
  assert.ok(pixels.changed <= pixels.total * 0.01, `levels: ${pixels.changed} of ${pixels.total} pixels differ`);

  // Disabled, every instance draws its mesh and the frames match exactly.
  const disabled = await command("environment", { level_of_detail: { enabled: false } });
  assert.deepEqual(disabled.result.environment_changed, ["level_of_detail"]);
  const exact = await capture("overview-exact", true);
  assert.equal(coarser(exact.rendering), 0, JSON.stringify(exact.rendering));
  const exactPixels = await differentPixels([cpu.png, exact.png], 8);
  assert.ok(exactPixels.changed <= exactPixels.total * 0.001, `disabled: ${exactPixels.changed} pixels differ`);

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
      changed_pixels: pixels.changed, total_pixels: pixels.total, exact_changed_pixels: exactPixels.changed,
      levels: [gpu.rendering.lod0_instances, gpu.rendering.lod1_instances, gpu.rendering.lod2_instances],
      coarse_levels: [coarse.lod0_instances, coarse.lod1_instances, coarse.lod2_instances],
    },
    checks: ["settings_discoverable", "coarser_levels_draw", "levels_within_pixels", "disabled_exact", "pixel_error_control", "invalid_settings_refused", "clean_browser_diagnostics"],
  }));
} finally {
  await browser.close();
}
