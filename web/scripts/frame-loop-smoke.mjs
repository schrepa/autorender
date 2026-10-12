/**
 * The page's frames run through the engine's frame loop: a session that
 * starts as players' do prepares behind a hidden canvas, then plays; a
 * canvas without area goes idle, so no frame runs and its ticks stand
 * still, and frames run again from where they stopped once it has area,
 * without catching up the time it spent idle.
 */
// @ts-check
import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import { commander, developmentReady, devUrl, launch, waitFrames } from "./harness.mjs";

const { browser, page, errors } = await launch({ width: 800, height: 600 });
try {
  // ?prepare=1: the development session starts as an ordinary one.
  await page.goto(String(devUrl("stress.html", { preset: "visual", prepare: "1" })), { waitUntil: "domcontentloaded" });
  await developmentReady(page, { timeout: 120_000 });
  const command = commander(page, { deadline: 120_000 });
  await page.waitForFunction(() => document.querySelector("#status")?.getAttribute("data-state") === "running", null, { timeout: 120_000 });
  assert.equal(await page.locator("#scene").evaluate(canvas => canvas.style.visibility), "", "play shows the canvas preparation hid");
  await waitFrames(command, 10);

  // A canvas without area: no frame runs and ticks stand still.
  const resize = (/** @type {string} */ size) => page.locator("#scene").evaluate((canvas, size) => {
    canvas.style.width = size; canvas.style.height = size;
  }, size);
  await resize("0px");
  await delay(300);
  const idle = (await command("status")).observation;
  await delay(500);
  const still = (await command("status")).observation;
  assert.deepEqual([still.frame, still.tick], [idle.frame, idle.tick], `an idle view runs no frames: ${JSON.stringify({ idle, still })}`);

  // Area again: frames run on, without catching up the idle time: the
  // ticks they ran fit the time since, not the idle half second too.
  const tickSeconds = (await command("status")).result.fixed_delta_seconds;
  const resumed = Date.now();
  await resize("");
  const running = await waitFrames(command, 10);
  const seconds = (Date.now() - resumed) / 1000;
  const ticks = Number(running.observation.tick) - Number(still.tick);
  const frames = Number(running.observation.frame) - Number(still.frame);
  assert.ok(frames >= 10, `frames run again: ${JSON.stringify(running.observation)}`);
  assert.ok(ticks <= seconds / tickSeconds + 3, `the idle time was not caught up: ${ticks} ticks in ${seconds} s`);
  assert.deepEqual(errors, []);
  console.log(JSON.stringify({ ok: true, frames, ticks, seconds, checks: ["prepares_then_plays", "idle_runs_no_frames", "area_runs_again", "idle_time_dropped", "clean_browser_diagnostics"] }));
} finally {
  await browser.close();
}
