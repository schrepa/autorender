/**
 * The browser leg of `cargo xtask equivalence`: runs equivalence cells in
 * headless Chrome against the development servers of a base and a
 * candidate, as the native leg runs them in the native host, and writes what
 * each run captured for xtask to judge.
 *
 * `node web/scripts/equivalence-browser.mjs PLAN.json RESULT.json`: the plan
 * holds the page size and a list of runs, each a cell on one side: the
 * side's server (`url`) and tree (`tree`, whose fixtures authored scenes are
 * read from), the Stress Lab's page parameters, an authored `source` scene,
 * environment sections merged over the scene's, the ticks to step, the views
 * and the captures (`exact`, `cpu:ALPHA`, `gpu:ALPHA`, `display:FRAMES`).
 * Each run opens the page paused, steps, and for each view settles every
 * capture with the frame kind it draws until nothing loads or compiles,
 * read-backs describe frames drawn since, and the work counters hold still,
 * then 30 frames more; the capture is the canvas after the development
 * capture presented it, pinned to the plan's size without the page's
 * overlays. Scanned rock maps are left out, as the native leg's
 * `--rocks off` leaves them. The result lists each run's capture files and
 * frame statistics, its digests, the views that did not settle, and why a
 * run stopped where one did.
 */
// @ts-check
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { dirname, join, resolve } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import { launchContext } from "./browser-launch.mjs";
import { holdHeavyRun } from "./heavy-run.mjs";

await holdHeavyRun("equivalence browser leg");

const [planPath, resultPath] = process.argv.slice(2);
if (!planPath || !resultPath) {
  console.error("usage: node web/scripts/equivalence-browser.mjs PLAN.json RESULT.json");
  process.exit(1);
}
const plan = JSON.parse(await readFile(planPath, "utf8"));
/** Frames in a row whose work counters hold still before a view counts as settled. */
const QUIET = 3;
/** Frames after settling, so time-sliced work completes its rounds. */
const TAIL = 30;
/** How long a capture may take to settle. */
const SETTLE_MS = 300_000;
/** The URL authored scenes are served at, which no file holds. */
const FIXTURE = "/scenes/equivalence-fixture.scene.json";
/** Frame statistics that are not work (bench.rs NOT_WORK). */
const NOT_WORK = new Set(["frame", "tick", "read_back_age", "simulation_ms", "extraction_ms", "render_ms"]);

/** The work counters of frame statistics `stats`: dotted paths to numbers
 * and booleans, lists as their lengths. */
function work(stats, prefix = "", out = {}) {
  for (const [key, value] of Object.entries(stats ?? {})) {
    const path = prefix ? `${prefix}.${key}` : key;
    if (Array.isArray(value)) out[path] = value.length;
    else if (value && typeof value === "object") work(value, path, out);
    else if ((typeof value === "number" || typeof value === "boolean") && !NOT_WORK.has(path)) out[path] = value;
  }
  return out;
}

const { page, close } = await launchContext({ viewport: { width: plan.size[0] + 100, height: plan.size[1] + 100 }, deviceScaleFactor: 1 });
page.setDefaultTimeout(600_000);

/** The page's reply to `command`, retried only while busy. */
async function request(command) {
  for (;;) {
    const reply = await page.evaluate(command => window.__engineDev.request(command), command);
    if (reply?.error?.code === "command_busy") { await delay(25); continue; }
    return reply;
  }
}

/** The result of `command`, which must succeed. */
async function ok(command) {
  const reply = await request(command);
  if (!reply?.ok) throw new Error(`${command.op}: ${JSON.stringify(reply?.error ?? reply).slice(0, 2000)}`);
  return reply.result;
}

/** The frame a capture settles with, none for display frames. */
function settling(capture) {
  if (capture === "exact") return { op: "render" };
  const [kind, value] = capture.split(":");
  if (kind === "cpu") return { op: "present", alpha: Number(value) };
  if (kind === "gpu") return { op: "present", alpha: Number(value), gpu: true };
  return undefined;
}

/** Draws `frame` until settled, then 30 more; what kept changing where it
 * did not settle in time. */
async function settle(frame) {
  const started = Date.now();
  let previous, quiet = 0;
  for (let drawn = 1; ; drawn++) {
    const stats = (await ok(frame)).rendering;
    const busy = stats.loading === true || stats.pipelines_compiling === true;
    const lag = stats.read_back_age;
    const stale = typeof lag === "number" && lag >= drawn;
    const counters = JSON.stringify(work(stats));
    quiet = !busy && !stale && counters === previous ? quiet + 1 : 0;
    if (quiet >= Math.max(QUIET, (typeof lag === "number" ? lag : 0) + 1)) break;
    if (Date.now() - started > SETTLE_MS) return `${busy ? "loading or compiling; " : ""}${stale ? "read-backs older than settling; " : ""}counters changing`;
    previous = counters;
    if (busy) await delay(5);
  }
  for (let frame_ = 0; frame_ < TAIL; frame_++) await ok(frame);
  return undefined;
}

/** Takes `capture` of the settled view into `file`. */
async function shoot(capture, file) {
  let fields;
  if (capture === "exact") fields = {};
  else if (capture.startsWith("display:")) {
    const reply = await request({ op: "frames", count: Number(capture.split(":")[1]), interval_ms: 1000 / 60, capture: true });
    if (!reply?.ok) return { unavailable: `the host ran no display frames: ${reply?.error?.message ?? ""}` };
    await page.locator("canvas").screenshot({ path: file });
    return { path: file, rendering: reply.result.rendering };
  } else {
    const [kind, value] = capture.split(":");
    fields = kind === "gpu" ? { alpha: Number(value), gpu: true } : { alpha: Number(value) };
  }
  const result = await ok({ op: "capture", ...fields });
  await page.locator("canvas").screenshot({ path: file });
  return { path: file, rendering: result.rendering };
}

/** One run: the cell on one side, its files into `folder`. */
async function run(entry, folder) {
  /** @type {{ cell: any, side: any, index: any, shots: Record<string, any>, digests: Record<string, any>, unsettled: string[], error: string | null }} */
  const result = { cell: entry.cell, side: entry.side, index: entry.index, shots: {}, digests: {}, unsettled: [], error: null };
  await mkdir(folder, { recursive: true });
  await page.unrouteAll({ behavior: "ignoreErrors" });
  // As the native leg's --rocks off: the page wears the terrain rock's maps.
  await page.route("**/stress/rocks/**", route => route.fulfill({ status: 404, body: "" }));
  const query = new URLSearchParams({ engine_dev: "1", ...Object.fromEntries(Object.entries(entry.page).map(([key, value]) =>
    [key, typeof value === "string" ? value : JSON.stringify(value)])) });
  if (entry.source) {
    // The authored scene and the files beside it, from the side's tree.
    const sourcePath = resolve(entry.tree, entry.source);
    const document = JSON.parse(await readFile(sourcePath, "utf8"));
    const folderOfScene = dirname(sourcePath);
    await page.route(`**${FIXTURE}*`, route => route.fulfill({ json: document }));
    await page.route("**/scenes/**", async route => {
      const path = new URL(route.request().url()).pathname;
      if (path === FIXTURE) return route.fallback();
      try { await route.fulfill({ body: await readFile(join(folderOfScene, decodeURIComponent(path.slice("/scenes/".length)))) }); }
      catch { await route.fulfill({ status: 404, body: "" }); }
    });
    query.set("source", FIXTURE);
  }
  try {
    await page.goto(new URL(`stress.html?${query}`, `${entry.url.replace(/\/$/, "")}/`).href, { waitUntil: "domcontentloaded" });
    // The canvas alone at the plan's size, as the native host draws it:
    // the page's layout and overlays (labels, frame rate) stay out.
    await page.addStyleTag({ content: `.viewport { position: fixed !important; left: 0 !important; top: 0 !important;
      width: ${plan.size[0]}px !important; height: ${plan.size[1]}px !important; border: 0 !important; border-radius: 0 !important; z-index: 10000; }
      .viewport > :not(canvas) { visibility: hidden !important; }` });
    await page.waitForFunction(() => window.__engineDev?.version !== undefined, null, { timeout: 600_000 });
    await ok({ op: "pause" });
    if (entry.environment) await ok({ op: "environment", merge: true, ...entry.environment });
    if (entry.ticks > 0) await ok({ op: "step", ticks: entry.ticks });
    for (const view of entry.views) {
      if (view.camera) await ok({ op: "camera", ...view.camera });
      else if (view.name !== "scene") await ok({ op: "game", name: "view", args: { name: view.name } });
      for (const capture of entry.captures) {
        const key = `${view.name} ${capture}`;
        const frame = settling(capture);
        const unsettled = frame && await settle(frame);
        if (unsettled) result.unsettled.push(`${key}: ${unsettled}`);
        result.shots[key] = await shoot(capture, join(folder, `${key.replace(/[ :]/g, "-")}.png`));
      }
      const digest = await request({ op: "digest" });
      result.digests[view.name] = digest?.ok
        ? { hash: digest.result.hash, entities: digest.result.entities, components: digest.result.components, assets: digest.result.assets }
        : null;
    }
  } catch (error) {
    result.error = String(error?.message ?? error).slice(0, 2000);
  }
  return result;
}

const results = [];
try {
  for (const entry of plan.runs) {
    console.error(`${entry.cell}: ${entry.side} run ${entry.index} (browser).`);
    results.push(await run(entry, join(plan.output, entry.cell, `${entry.side}-browser-${entry.index}`)));
  }
} finally {
  await close();
}
await writeFile(resultPath, JSON.stringify({ runs: results }, null, 1));
