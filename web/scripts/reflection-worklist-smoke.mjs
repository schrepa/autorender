/** Reflection worklists on the GPU: one stable spatial queue, hierarchical
 * prefixes, exact stored trace bytes against unchanged main, and reflection
 * lifetime through ordinary Stress Lab frames. Assertion failures exit 3. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { artifacts, cargoBinaries, commander, devUrl, launch, openPaused, waitFrames } from "./harness.mjs";
import { workspace, workspacePath } from "./workspace.mjs";
import { checkReflectionGpu } from "./reflection-worklist-gpu.mjs";
import { reflectionStateSource } from "./reflection-worklist-state.mjs";
import { installReflectionProbe, inspectLiveReflections, verifyLiveReflectionTargets } from "./reflection-worklist-live.mjs";
import { compareReflectionOrders } from "./reflection-worklist-orders.mjs";

const output = await artifacts("reflection-worklist-smoke");
const source = name => readFile(workspacePath("crates/engine_render/src", name), "utf8");
const catalog = await source("shaders.rs");
async function moduleSource(name) {
  const files = [...catalog.match(new RegExp(`${name} = \\[([\\s\\S]*?)\\];`))[1].matchAll(/"([^"]+\.wgsl)"/g)].map(match => match[1]);
  return (await Promise.all(files.map(source))).join("\n");
}
// Independent fragment source, pinned before extracting reflection_at.
const baseCommit = "96a06f7bf50d6e75c64e4135087679d9418167c3";
const baseSource = ["color.wgsl", "fullscreen.wgsl", "effects.wgsl", "post.wgsl"].map(name =>
  execFileSync("git", ["show", `${baseCommit}:crates/engine_render/src/${name}`], { cwd: workspace, encoding: "utf8", windowsHide: true })).join("\n");
const sources = { work: await moduleSource("REFLECTION_WORK_SHADER"), trace: await moduleSource("REFLECTION_TRACE_SHADER"),
  export: await moduleSource("REFLECTION_EXPORT_SHADER"), post: await moduleSource("POST_SHADER"), base: baseSource,
  depth: await moduleSource("DEPTH_PYRAMID_SHADER"), fullscreen: await source("fullscreen.wgsl") };
// Read the rejected sibling as an independent diagnostic variant, never merge it.
const siblingCommit = "39bb68c5532af9094a5979adf991e7d2582a25a7";
const siblingCatalog = execFileSync("git", ["show", `${siblingCommit}:crates/engine_render/src/shaders.rs`], { cwd: workspace, encoding: "utf8", windowsHide: true });
function siblingModule(name) {
  const files = [...siblingCatalog.match(new RegExp(`${name} = \\[([\\s\\S]*?)\\];`))[1].matchAll(/"([^"]+\.wgsl)"/g)].map(match => match[1]);
  return files.map(name => execFileSync("git", ["show", `${siblingCommit}:crates/engine_render/src/${name}`], { cwd: workspace, encoding: "utf8", windowsHide: true })).join("\n");
}
sources.sibling = { commit: siblingCommit, work: siblingModule("REFLECTION_WORK_SHADER"), trace: siblingModule("REFLECTION_TRACE_SHADER"), export: siblingModule("REFLECTION_EXPORT_SHADER") };
sources.baseState = reflectionStateSource(sources.base, true);
sources.fallbackState = reflectionStateSource(sources.post, true, true);
sources.traceState = reflectionStateSource(sources.trace, false);
const binaries = await cargoBinaries(["-p", "engine_render", "--example", "reflection_queue_plan", "--profile", "quick"]);
let host;
try { host = JSON.parse(execFileSync(binaries.reflection_queue_plan, [], { encoding: "utf8", windowsHide: true })); }
catch (error) { if (error.status === 101) assert.fail(String(error.stderr)); throw error; }
assert.ok(host.rejections >= 19, "the actual host planner rejects unsupported requests");
const { browser, page, errors } = await launch({ timeout: 120_000 });
const clearCaptures = () => page.evaluate(async () => {
  await Promise.all(window.__reflectionProbe.pending);
  for (const capture of window.__reflectionProbe.captures) for (const object of capture.owned) object.destroy();
  window.__reflectionProbe.captures = []; window.__reflectionProbe.pending = [];
});
try {
  const url = "http://127.0.0.1:18667/reflection-gpu-diagnostic";
  await page.route(url, route => route.fulfill({ contentType: "text/html", body: "<!doctype html><title>Reflection GPU check</title>" }));
  await page.goto(url);
  const gpu = process.argv.includes("--live-only") ? { ok: true, diagnostic_only: true }
    : await page.evaluate(checkReflectionGpu, { sources, plans: host.plans, layouts: host.layouts });
  await writeFile(`${output}/gpu.json`, JSON.stringify(gpu, null, 2));
  assert.equal(gpu.ok, true, gpu.error);
  const live = [];
  if (!process.argv.includes("--gpu-only")) {
    await page.addInitScript(installReflectionProbe);
    const command = commander(page);
    const settings = JSON.parse(await readFile(workspacePath("bench/settings/default.json"), "utf8"));
    const settle = async workload => {
      await command("resume");
      const result = await waitFrames(command, 60, { deadline: 240_000, done: ({ result }) => {
        const frame = result.rendering;
        return !frame.loading && !frame.pipelines_compiling && (workload !== "highlands"
          || (frame.terrain.tiles > 0 && frame.terrain.loading_tiles === 0 && frame.terrain.fallback_patches === 0));
      } });
      await command("pause");
      return result.result.rendering;
    };
    for (const workload of ["metro", "highlands"]) {
      const scene = JSON.parse(await readFile(workspacePath("bench/scenes", `${workload}.json`), "utf8"));
      const options = { ...settings.options, ...scene.options };
      await openPaused(page, devUrl("stress.html", { ...options, effects: "no-reflections" }));
      assert.equal(await page.evaluate(() => window.__reflectionProbe.created), 0, "initially off allocates no queue scratch");
      await openPaused(page, devUrl("stress.html", options));
      const settled = await settle(workload);
      const ordinaryTemporal = (await command("inspect", { component: "TemporalAntialiasing" })).result.value.enabled;
      await page.evaluate(() => window.__reflectionProbe.armed = 4);
      await command("resume");
      await waitFrames(command, 8);
      await command("pause");
      await page.locator("canvas").screenshot({ path: `${output}/${workload}-default.png` });
      const frames = await page.evaluate(inspectLiveReflections, { sources });
      const targets = await page.evaluate(verifyLiveReflectionTargets, { sources, layouts: host.layouts });
      assert.equal(frames.length, 4, `${workload}: four ordinary frames captured`);
      assert.deepEqual([...new Set(frames.map(f => f.phase))].sort(), ordinaryTemporal ? [0, 1, 2, 3] : [0], `${workload}: preserve its ordinary temporal setting`);
      if (workload === "metro") assert.ok(frames.every(f => f.active > 0), "unchanged Metro has nonzero original ray work in every phase");
      const orders = workload === "metro" ? await page.evaluate(compareReflectionOrders, { sources }) : [];
      await writeFile(`${output}/${workload}-orders.json`, JSON.stringify(orders, null, 2));
      await clearCaptures();
      await command("environment", { temporal_antialiasing: { enabled: true } });
      await page.evaluate(() => window.__reflectionProbe.armed = 4);
      await command("resume"); await waitFrames(command, 8); await command("pause");
      const temporal = await page.evaluate(inspectLiveReflections, { sources });
      assert.deepEqual([...new Set(temporal.map(f => f.phase))].sort(), [0, 1, 2, 3], "ordinary temporal four-phase cadence");
      live.push({ workload, frames, orders, temporal, targets, settled });
      await clearCaptures();
      // Force the real planner to reject the next enabled frame, then compare
      // its complete original trace target to unchanged main on those inputs.
      await command("environment", { merge: true, reflections: { intensity: 0 } });
      await page.evaluate(() => window.__reflectionProbe.forceFallback = true);
      const fallbackBefore = await page.evaluate(() => window.__reflectionProbe.fallback);
      await command("environment", { merge: true, reflections: { intensity: 1 } });
      await page.evaluate(() => window.__reflectionProbe.armed = 1);
      await command("capture");
      assert.equal(await page.evaluate(() => window.__reflectionProbe.live.size), 0, "unsupported plan allocates no scratch");
      assert.ok(await page.evaluate(before => window.__reflectionProbe.fallback > before, fallbackBefore), "unsupported plan executes original full frame");
      const fallbackTargets = await page.evaluate(verifyLiveReflectionTargets, { sources, layouts: host.layouts });
      assert.equal(fallbackTargets.length, 1, "one full original frame was compared");
      assert.equal(fallbackTargets[0].path, "original");
      live.at(-1).fallbackTargets = fallbackTargets;
      await clearCaptures();
      await command("environment", { merge: true, reflections: { intensity: 0 } });
      await page.evaluate(() => window.__reflectionProbe.forceFallback = false);
      await command("environment", { merge: true, reflections: { intensity: 1 } });
      for (const enabled of [false, true, false, true]) {
        await command("environment", { merge: true, reflections: { intensity: enabled ? 1 : 0 } });
        await command("render");
        const liveBuffers = await page.evaluate(() => window.__reflectionProbe.live.size);
        assert.equal(liveBuffers, enabled ? 7 : 0, "scratch is released while reflections are disabled");
        if (!enabled) {
          const before = await page.evaluate(() => window.__reflectionProbe.traces);
          await command("resume"); await waitFrames(command, 3); await command("pause");
          assert.equal(await page.evaluate(() => window.__reflectionProbe.traces), before, "ordinary disabled frames do no queue work");
        }
      }
      const camera = (await command("inspect", { component: "Camera" })).result.value;
      const moved = structuredClone(camera);
      moved.position[0] += workload === "metro" ? 5 : 25;
      moved.target[0] += workload === "metro" ? 5 : 25;
      await command("camera", moved);
      await settle(workload);
      await page.evaluate(() => window.__reflectionProbe.armed = 4);
      await command("resume"); await waitFrames(command, 8); await command("pause");
      const motion = await page.evaluate(inspectLiveReflections, { sources });
      assert.equal(motion.length, 4, "camera motion preserves complete queues across four frames");
      live.at(-1).motion = motion;
      live.at(-1).motionTargets = await page.evaluate(verifyLiveReflectionTargets, { sources, layouts: host.layouts });
      await page.locator("canvas").screenshot({ path: `${output}/${workload}-motion.png` });
      await clearCaptures();
      await command("camera", camera);
      await command("game", { name: "workload", args: { config: scene.options.config } });
      await settle(workload);
      await page.evaluate(() => window.__reflectionProbe.armed = 4);
      await command("resume"); await waitFrames(command, 8); await command("pause");
      const reloaded = await page.evaluate(inspectLiveReflections, { sources });
      assert.equal(reloaded.length, 4, "same-page workload reload rebinds valid reflection inputs");
      assert.equal(await page.evaluate(() => window.__reflectionProbe.live.size), 7, "scene reload retains one complete scratch bundle");
      live.at(-1).reloaded = reloaded;
      live.at(-1).reloadedTargets = await page.evaluate(verifyLiveReflectionTargets, { sources, layouts: host.layouts });
      await clearCaptures();
      const beforeResize = await page.evaluate(() => window.__reflectionProbe.created);
      await page.setViewportSize(workload === "metro" ? { width: 1027, height: 769 } : { width: 1201, height: 901 });
      await command("resume"); await waitFrames(command, 5); await command("pause");
      assert.ok(await page.evaluate(before => window.__reflectionProbe.created > before && window.__reflectionProbe.live.size === 7, beforeResize), `${workload}: resize rebuilds one complete scratch bundle`);
      await page.locator("canvas").screenshot({ path: `${output}/${workload}-resized.png` });
      await page.reload();
      await page.waitForFunction(() => !!window.__engineDev);
      await command("render");
    }
  }
  assert.deepEqual(errors, [], "no GPU validation or page errors");
  await writeFile(`${output}/live.json`, JSON.stringify(live, null, 2));
  console.log(JSON.stringify({ ok: true, output, base: baseCommit, host, gpu, live }));
} catch (error) {
  if (error.message?.includes("REFLECTION_ASSERTION:")) assert.fail(error.message);
  throw error;
} finally { await browser.close(); }
