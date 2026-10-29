// @ts-check
import assert from "node:assert/strict";
import { commander, gridScene, launch, openPaused, routedScene, sceneUrl, waitFrames } from "./harness.mjs";

const { browser, page, errors } = await launch({ width: 800, height: 600 });
try {
  await routedScene(page, gridScene()).route();
  await openPaused(page, sceneUrl(), { timeout: 120_000 });
  const command = commander(page, { deadline: 120_000 });
  /** @returns {Promise<{ position: number[], target: number[] }>} */
  const camera = async () => (await command("inspect", { component: "Camera" })).result.value;
  const direction = (/** @type {{ position: number[], target: number[] }} */ view) => {
    const delta = view.target.map((value, axis) => value - view.position[axis]);
    const length = Math.hypot(...delta);
    return delta.map(value => value / length);
  };
  const moved = (/** @type {number[]} */ from, /** @type {number[]} */ to) => Math.hypot(...to.map((value, axis) => value - from[axis]));
  const canvas = page.locator("canvas").first();
  const box = await canvas.boundingBox();
  assert.ok(box, "the page has its canvas");
  await canvas.focus();
  await command("control", "resume");
  const start = await camera();

  await page.keyboard.down("KeyW");
  await waitFrames(command, 20);
  await page.keyboard.up("KeyW");
  const forward = await camera();
  const step = forward.position.map((value, axis) => value - start.position[axis]);
  const along = step.reduce((sum, value, axis) => sum + value * direction(start)[axis], 0);
  assert.ok(along > 0.5, `W moved the camera forward: ${JSON.stringify({ start, forward })}`);

  await waitFrames(command, 10);
  const still = await camera();
  assert.ok(moved(forward.position, still.position) < 1e-3, `released, the camera stops: ${JSON.stringify({ forward, still })}`);

  const [x, y] = [box.x + box.width / 2, box.y + box.height / 2];
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let step = 1; step <= 10; step++) await page.mouse.move(x + step * 12, y + step * 4);
  await page.mouse.up();
  await waitFrames(command, 10);
  const turned = await camera();
  const [before, after] = [direction(still), direction(turned)];
  const turn = Math.acos(Math.min(1, before.reduce((sum, value, axis) => sum + value * after[axis], 0)));
  assert.ok(turn > 0.05, `the drag turned the view by ${turn} radians`);

  const kept = await page.evaluate(() => {
    const target = document.querySelector("canvas");
    if (!target) throw new Error("no canvas");
    const press = (/** @type {string} */ code) => {
      const event = new KeyboardEvent("keydown", { code, key: code, bubbles: true, cancelable: true });
      target.dispatchEvent(event);
      window.dispatchEvent(new KeyboardEvent("keyup", { code, key: code, bubbles: true }));
      return event.defaultPrevented;
    };
    return { used: press("KeyW"), unused: press("KeyF") };
  });
  assert.deepEqual(kept, { used: true, unused: false });

  await canvas.focus();
  await page.keyboard.down("KeyW");
  await waitFrames(command, 5);
  await page.evaluate(() => window.dispatchEvent(new Event("blur")));
  const released = await camera();
  await waitFrames(command, 10);
  const after_blur = await camera();
  await page.keyboard.up("KeyW");
  assert.ok(moved(released.position, after_blur.position) < 1e-3, `focus lost, nothing is held: ${JSON.stringify({ released, after_blur })}`);

  await command("control", "pause");
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, forward: along, turn, checks: ["key_moves", "release_stops", "drag_turns", "unused_keys_stay", "blur_releases", "clean_browser_diagnostics"] }));
} finally {
  await browser.close();
}
