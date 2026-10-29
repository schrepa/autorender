// @ts-check
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { join } from "node:path";
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
const QUIET = 3;
const TAIL = 30;
const SETTLE_MS = 300_000;
const NOT_WORK = new Set(["frame", "tick", "read_back_age", "simulation_ms", "extraction_ms", "render_ms"]);

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

let speaksVerbs = true;

function legacy(command) {
  const { verb, name, args, ...fields } = command;
  return verb === "query" || verb === "control" ? { op: name, ...args } : { op: verb, ...fields };
}

async function request(command) {
  for (;;) {
    const sent = speaksVerbs ? command : legacy(command);
    const reply = await page.evaluate(command => window.__engineDev.request(command), sent);
    if (reply?.error?.code === "command_busy") { await delay(25); continue; }
    return reply;
  }
}

async function hear() {
  const reply = await request({ verb: "query", name: "status" });
  speaksVerbs = reply?.ok === true || reply?.error?.path !== "$.command";
}

async function ok(command) {
  const reply = await request(command);
  if (!reply?.ok) throw new Error(`${command.verb} ${command.name ?? ""}: ${JSON.stringify(reply?.error ?? reply).slice(0, 2000)}`);
  return reply.result;
}

function settling(capture) {
  if (capture === "exact") return { verb: "control", name: "render" };
  const [kind, value] = capture.split(":");
  if (kind === "gpu") return { verb: "control", name: "present", args: { alpha: Number(value) } };
  return undefined;
}

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

async function shoot(capture, file) {
  let fields;
  if (capture === "exact") fields = {};
  else if (capture.startsWith("display:")) {
    const reply = await request({ verb: "control", name: "frames", args: { count: Number(capture.split(":")[1]), interval_ms: 1000 / 60, capture: true } });
    if (!reply?.ok) return { unavailable: `the host ran no display frames: ${reply?.error?.code ?? ""}` };
    await page.locator("canvas").screenshot({ path: file });
    return { path: file, rendering: reply.result.rendering };
  } else {
    fields = { alpha: Number(capture.split(":")[1]) };
  }
  const result = await ok({ verb: "capture", ...fields });
  await page.locator("canvas").screenshot({ path: file });
  return { path: file, rendering: result.rendering };
}

async function run(entry, folder) {
  /** @type {{ cell: any, side: any, index: any, shots: Record<string, any>, digests: Record<string, any>, unsettled: string[], error: string | null }} */
  const result = { cell: entry.cell, side: entry.side, index: entry.index, shots: {}, digests: {}, unsettled: [], error: null };
  await mkdir(folder, { recursive: true });
  await page.unrouteAll({ behavior: "ignoreErrors" });
  /** @type {string[]} */
  const withheld = entry.withheld;
  await page.route(url => withheld.some(path => {
    const requested = decodeURIComponent(url.pathname);
    return path.endsWith("/") ? requested.startsWith(path) : requested === path;
  }), route => route.fulfill({ status: 404, body: "" }));
  const query = new URLSearchParams({ engine_dev: "1", ...Object.fromEntries(Object.entries(entry.page).map(([key, value]) =>
    [key, typeof value === "string" ? value : JSON.stringify(value)])) });
  try {
    await page.goto(new URL(`${entry.html}?${query}`, `${entry.url.replace(/\/$/, "")}/`).href, { waitUntil: "domcontentloaded" });
    await page.addStyleTag({ content: `.engine-page #scene { position: fixed !important; inset: auto !important; left: 0 !important; top: 0 !important;
      width: ${plan.size[0]}px !important; height: ${plan.size[1]}px !important; z-index: 10000; }
      .engine-page .engine-hud, .engine-page .message { visibility: hidden !important; }` });
    await page.waitForFunction(() => window.__engineDev?.version !== undefined, null, { timeout: 600_000 });
    speaksVerbs = true;
    await hear();
    await ok({ verb: "control", name: "pause" });
    if (entry.environment) await ok({ verb: "control", name: "environment", args: { merge: true, ...entry.environment } });
    if (entry.ticks > 0) await ok({ verb: "control", name: "step", args: { ticks: entry.ticks } });
    for (const view of entry.views) {
      if (view.camera) await ok({ verb: "control", name: "camera", args: { ...view.camera } });
      else if (view.name !== "scene") await ok({ verb: "control", name: "game", args: { name: "view", args: { name: view.name } } });
      for (const capture of entry.captures) {
        const key = `${view.name} ${capture}`;
        const frame = settling(capture);
        const unsettled = frame && await settle(frame);
        if (unsettled) result.unsettled.push(`${key}: ${unsettled}`);
        result.shots[key] = await shoot(capture, join(folder, `${key.replace(/[ :]/g, "-")}.png`));
      }
      const digest = await request({ verb: "query", name: "digest" });
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
