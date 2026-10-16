/** Reflection classification on the GPU: stable four-bin queues, hierarchical
 * prefixes, exact stored trace bytes against unchanged main, and reflection
 * lifetime through ordinary Stress Lab frames. Assertion failures exit 3. */
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { readFile, writeFile } from "node:fs/promises";
import { artifacts, cargoBinaries, commander, devUrl, launch, openPaused, waitFrames } from "./harness.mjs";
import { workspace, workspacePath } from "./workspace.mjs";
import { checkReflectionGpu } from "./reflection-classification-gpu.mjs";
import { reflectionStateSource } from "./reflection-classification-state.mjs";

const output = await artifacts("reflection-classification-smoke");
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
sources.baseState = reflectionStateSource(sources.base, true);
sources.traceState = reflectionStateSource(sources.trace, false);
const binaries = await cargoBinaries(["-p", "engine_render", "--example", "reflection_queue_plan", "--profile", "quick"]);
const host = JSON.parse(execFileSync(binaries.reflection_queue_plan, [], { encoding: "utf8", windowsHide: true }));
assert.ok(host.rejections >= 19, "the actual host planner rejects unsupported requests");
const { browser, page, errors } = await launch({ timeout: 120_000 });
try {
  const url = "http://127.0.0.1:18667/reflection-gpu-diagnostic";
  await page.route(url, route => route.fulfill({ contentType: "text/html", body: "<!doctype html><title>Reflection GPU check</title>" }));
  await page.goto(url);
  const gpu = await page.evaluate(checkReflectionGpu, { sources, plans: host.plans });
  await writeFile(`${output}/gpu.json`, JSON.stringify(gpu, null, 2));
  assert.equal(gpu.ok, true, gpu.error);
  if (!process.argv.includes("--gpu-only")) {
    const command = commander(page);
    for (const workload of ["metro", "highlands"]) {
      await openPaused(page, devUrl("stress.html", { workload }));
      await command("resume");
      await waitFrames(command, 8);
      await command("pause");
      for (const enabled of [false, true, false, true]) {
        await command("environment", { merge: true, reflections: { enabled } });
        await command("render");
      }
      await page.setViewportSize({ width: 1027, height: 769 });
      await command("render");
      await page.reload();
      await page.waitForFunction(() => !!window.__engineDev);
      await command("render");
    }
  }
  assert.deepEqual(errors, [], "no GPU validation or page errors");
  console.log(JSON.stringify({ ok: true, output, base: baseCommit, host, gpu }));
} finally { await browser.close(); }
