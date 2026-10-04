import { chromium } from "@playwright/test";
import assert from "node:assert/strict";
import { mkdir } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const url = process.env.DEMO_URL ?? "http://127.0.0.1:5173";
const browser = await chromium.launch({
  channel: process.env.BROWSER_CHANNEL ?? "chrome",
  headless: true,
  args: ["--enable-unsafe-webgpu", ...(process.env.SOFTWARE_GPU === "1" ? ["--use-angle=swiftshader"] : [])],
});
const errors = [];
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 }, deviceScaleFactor: 1 });
page.on("pageerror", error => errors.push(String(error)));
page.on("console", entry => { if (entry.type() === "error") errors.push(entry.text()); });
try {
  await page.goto(url);
  await page.waitForFunction(() => document.querySelector("#status")?.dataset.state === "running", null, { timeout: 60_000 });
  await page.waitForFunction(() => Number(document.querySelector("canvas")?.dataset.frames) >= 30);
  assert.equal(await page.locator("#scene").getAttribute("data-render-candidates"), "10000");
  assert.ok(Number(await page.locator("#scene").getAttribute("data-instances")) < 10000);
  assert.equal(await page.locator("#draws").textContent(), "1");
  await page.locator("#count").selectOption("1000");
  await page.waitForFunction(() => document.querySelector("canvas")?.dataset.renderCandidates === "1000");
  await page.locator("#count").selectOption("50000");
  await page.waitForFunction(() => document.querySelector("canvas")?.dataset.renderCandidates === "50000");
  await page.locator("#count").selectOption("10000");
  await page.waitForFunction(() => document.querySelector("canvas")?.dataset.renderCandidates === "10000");

  await page.locator("#pause").click();
  const frozen = await page.locator("#scene").getAttribute("data-frames");
  await page.waitForTimeout(350);
  assert.equal(await page.locator("#scene").getAttribute("data-frames"), frozen);
  assert.equal(await page.locator("#status").getAttribute("data-state"), "paused");
  await page.evaluate(() => {
    Object.defineProperty(document, "hidden", { configurable: true, value: true });
    document.dispatchEvent(new Event("visibilitychange"));
    delete document.hidden;
    document.dispatchEvent(new Event("visibilitychange"));
  });
  assert.equal(await page.locator("#status").getAttribute("data-state"), "paused");
  await page.locator("#pause").click();
  await page.waitForFunction(value => Number(document.querySelector("canvas")?.dataset.frames) > Number(value), frozen);
  await page.locator("#scene").focus();
  await page.keyboard.down("KeyW");
  await page.waitForTimeout(200);
  await page.keyboard.up("KeyW");
  const bounds = await page.locator("#scene").boundingBox();
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.move(bounds.x + bounds.width / 2 + 100, bounds.y + bounds.height / 2 + 30, { steps: 5 });
  await page.mouse.up();
  await page.locator("#reset").click();
  await page.locator(".viewport").evaluate(element => { element.style.display = "none"; });
  await page.waitForFunction(() => document.querySelector("#draws")?.textContent === "0");
  await page.locator(".viewport").evaluate(element => { element.style.display = ""; });
  await page.waitForFunction(() => document.querySelector("#draws")?.textContent === "1");

  const output = fileURLToPath(new URL("../../artifacts/", import.meta.url));
  await mkdir(output, { recursive: true });
  await page.screenshot({ path: `${output}/orbit-demo-desktop.png`, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(600);
  assert.equal(await page.locator("#status").getAttribute("data-state"), "running");
  const extent = await page.locator("canvas").evaluate(canvas => ({ width: canvas.width, height: canvas.height }));
  assert.ok(extent.width > 0 && extent.width <= 390);
  assert.ok(extent.height > 0);
  await page.screenshot({ path: `${output}/orbit-demo-mobile.png`, fullPage: true });

  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto(`${url}/playground.html`);
  await page.waitForFunction(() => !!document.querySelector("canvas")?.dataset.progress);
  const progress = () => page.locator("canvas").evaluate(canvas => JSON.parse(canvas.dataset.progress));
  const initial = await progress();
  assert.equal(initial.total, 6);
  assert.equal(initial.collected, 0);
  // Six draw groups, each drawing its mesh's nonempty levels of detail
  // (decision 047) on GPU-presented frames.
  assert.equal(await page.locator("#draws").textContent(), "11");
  assert.equal(await page.locator("canvas").getAttribute("data-resident-textures"), "1");
  const residency = await page.locator("canvas").getAttribute("data-resident-mesh-bytes");
  await page.screenshot({ path: `${output}/playground-desktop.png`, fullPage: true });
  await page.locator("canvas").focus();
  await page.keyboard.down("KeyW");
  await page.waitForFunction(() => JSON.parse(document.querySelector("canvas").dataset.progress).collected >= 1);
  await page.keyboard.up("KeyW");
  assert.ok((await progress()).player[2] < initial.player[2]);
  await page.keyboard.press("Space");
  await page.waitForFunction(() => JSON.parse(document.querySelector("canvas").dataset.progress).jumps === 1);

  const source = await (await page.request.get(`${url}/scenes/playground.scene.json`)).json();
  const invalid = structuredClone(source);
  invalid.entities[0].material = "does-not-exist";
  await page.route("**/scenes/playground.scene.json", route => route.fulfill({ json: invalid }));
  const beforeFailure = await progress();
  await page.locator("#reload").click();
  await page.waitForFunction(() => document.querySelector("#notice").textContent.startsWith("Reload failed"));
  assert.equal((await progress()).revision, beforeFailure.revision);
  assert.equal((await progress()).collected, beforeFailure.collected);
  assert.equal(await page.locator("#status").getAttribute("data-state"), "running");
  await page.unroute("**/scenes/playground.scene.json");
  for (let revision = source.revision + 1; revision <= source.revision + 3; revision++) {
    const revised = { ...source, revision };
    await page.route("**/scenes/playground.scene.json", route => route.fulfill({ json: revised }));
    await page.locator("#reload").click();
    await page.waitForFunction(value => JSON.parse(document.querySelector("canvas").dataset.progress).revision === value, revision);
    const next = await progress();
    assert.equal(next.collected, 0);
    assert.equal(next.cpu_meshes, initial.cpu_meshes);
    assert.equal(next.cpu_materials, initial.cpu_materials);
    assert.equal(next.cpu_asset_bytes, initial.cpu_asset_bytes);
    assert.equal(await page.locator("canvas").getAttribute("data-resident-meshes"), "6");
    assert.equal(next.cpu_textures, initial.cpu_textures);
    assert.equal(next.cpu_texture_bytes, initial.cpu_texture_bytes);
    assert.equal(await page.locator("canvas").getAttribute("data-resident-textures"), "1");
    assert.equal(await page.locator("canvas").getAttribute("data-resident-mesh-bytes"), residency);
    await page.unroute("**/scenes/playground.scene.json");
  }
  await page.waitForFunction(() => document.querySelector("canvas").dataset.meshUploadBytes === "0");
  await page.waitForFunction(() => document.querySelector("canvas").dataset.textureUploadBytes === "0");
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForTimeout(400);
  await page.screenshot({ path: `${output}/playground-mobile.png`, fullPage: true });
  assert.deepEqual(errors, []);

  // Validate the actual unsupported-device user flow independently of hardware.
  const unsupported = await browser.newPage();
  await unsupported.addInitScript(() => { delete Object.getPrototypeOf(navigator).gpu; });
  await unsupported.goto(url);
  await unsupported.waitForFunction(() => document.querySelector("#status")?.dataset.state === "error");
  assert.match(await unsupported.locator("#message").textContent(), /WebGPU/);
  console.log(JSON.stringify({ result: "passed", checks: ["WebGPU initialization", "10k/1k/50k workload", "per-mesh instanced draws", "pause/resume", "keyboard/pointer controls", "resize", "playable scene collection and jumping", "failed reload preservation", "successful revision reload", "CPU/GPU asset retirement", "unsupported WebGPU"], screenshots: output, browser: browser.version() }, null, 2));
} finally { await browser.close(); }
