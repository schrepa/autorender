// @ts-check
import { copyFile, mkdir, readFile } from "node:fs/promises";
import { resolve } from "node:path";
import { AgentError, VERSION, atMost, checkOptions, failure, options } from "./agent-protocol.mjs";
import { launchBrowserSession } from "./agent-browser.mjs";
import { launchNativeSession } from "./agent-native.mjs";
import { workspace } from "./workspace.mjs";

const OFF = "off";
const COUNTED = ["stderr", "spawn", "error", "pageerror", "http"];
const SHADER_WARNING = /shadermodule|wgsl|\[invalid |validation/i;

function line(record) {
  console.log(JSON.stringify(record));
}

function poses(plan) {
  if (!Array.isArray(plan?.poses) || plan.poses.length === 0) throw new AgentError("missing_field", "$.poses", { expected: "present" });
  let moved = false;
  return plan.poses.map((pose, at) => {
    const path = `$.poses[${at}]`;
    if (typeof pose?.name !== "string" || !pose.name) throw new AgentError("invalid_value", `${path}.name`, { expected: "id", got: pose?.name ?? null });
    if (pose.camera !== undefined && (typeof pose.camera !== "object" || pose.camera === null)) throw new AgentError("invalid_value", `${path}.camera`, { expected: "object", got: pose.camera });
    if (pose.camera === undefined && moved) throw new AgentError("missing_field", `${path}.camera`, { expected: "present" });
    moved ||= pose.camera !== undefined;
    const views = pose.views ?? [];
    if (!Array.isArray(views) || views.some(view => typeof view !== "string" || view === OFF)) throw new AgentError("invalid_value", `${path}.views`, { expected: "array", got: views });
    const ticks = pose.ticks ?? 0;
    if (!Number.isInteger(ticks) || ticks < 0) throw new AgentError("invalid_value", `${path}.ticks`, { expected: "integer", got: ticks });
    return { name: pose.name, camera: pose.camera, ticks, views };
  });
}

async function listedViews(ask) {
  const described = await ask({ verb: "describe", name: "debug_view" });
  const views = described.result?.descriptor?.schema?.properties?.view?.enum;
  if (!described.ok || !Array.isArray(views)) throw new AgentError(described.error?.code ?? "missing_field", described.error?.path ?? "$.descriptor.schema.properties.view.enum", { got: described.error ?? null });
  return { poses: [{ name: "start", views: views.filter(view => view !== OFF) }] };
}

/** @param {{ execute: (command: object) => Promise<any> }} owned @param {Record<string, string>} values */
async function look(owned, values) {
  const limitMs = Number(values["limit-ms"]);
  if (!Number.isInteger(limitMs) || limitMs <= 0) throw new AgentError("invalid_arguments", "--limit-ms", { expected: "integer", got: values["limit-ms"] ?? null });
  if (!values.out) throw new AgentError("invalid_arguments", "--out", { expected: "present" });
  const out = resolve(values.out);
  await mkdir(out, { recursive: true });
  const ask = async command => {
    let timer;
    try {
      return await Promise.race([
        owned.execute(command),
        new Promise((_, reject) => { timer = setTimeout(() => reject(new AgentError("command_timeout", "$", { allowed: atMost(limitMs), got: command })), limitMs); }),
      ]);
    } finally { clearTimeout(timer); }
  };
  let failures = 0;
  const failed = (view, record) => { failures++; line({ event: "error", view, ...record }); };
  const seen = new Set();
  const failedLoads = new Set();
  const disabled = new Set();
  async function diagnosed(view) {
    const reply = await ask({ verb: "query", name: "diagnostics" });
    if (!reply.ok) { failed(view, { request: "diagnostics", error: reply.error }); return; }
    for (const entry of reply.result?.entries ?? []) {
      const key = `${entry.at} ${entry.kind} ${entry.message}`;
      if (seen.has(key)) continue;
      seen.add(key);
      if (COUNTED.includes(entry.kind) || (entry.kind === "warning" && SHADER_WARNING.test(entry.message))) failed(view, { diagnostic: entry.kind, message: entry.message });
    }
  }
  function judged(view, rendering) {
    for (const failure of rendering.readiness?.failed ?? []) {
      const key = JSON.stringify(failure);
      if (!failedLoads.has(key)) { failedLoads.add(key); failed(view, { readiness_failed: failure }); }
    }
    for (const feature of [].concat(rendering.disabled_features ?? [])) {
      if (!disabled.has(feature)) { disabled.add(feature); failed(view, { disabled_feature: feature }); }
    }
  }
  const plan = values.plan ? JSON.parse(await readFile(values.plan, "utf8")) : {};
  const planned = poses(plan.poses === undefined ? await listedViews(ask) : plan);
  await diagnosed(null);
  let shot = 0, ticked = 0;
  for (const pose of planned) {
    if (pose.camera) {
      const moved = await ask({ verb: "control", name: "camera", args: pose.camera });
      if (!moved.ok) { failed(pose.name, { request: "camera", error: moved.error }); continue; }
    }
    if (pose.ticks > 0) {
      const stepped = await ask({ verb: "control", name: "step", args: { ticks: pose.ticks } });
      if (stepped.ok) ticked += pose.ticks;
      else failed(pose.name, { request: "step", error: stepped.error });
    }
    for (const view of [OFF, ...pose.views]) {
      const name = view === OFF ? pose.name : `${pose.name}/${view}`;
      const shown = await ask({ verb: "control", name: "environment", args: { debug_view: { view } } });
      if (!shown.ok) { failed(name, { request: "environment", error: shown.error }); continue; }
      const started = Date.now();
      let rendering = {}, settled = false, rendered = true;
      while (true) {
        const reply = await ask({ verb: "control", name: "render" });
        if (!reply.ok) { failed(name, { request: "render", error: reply.error }); rendered = false; break; }
        rendering = reply.result?.rendering ?? {};
        judged(name, rendering);
        const readiness = rendering.readiness ?? {};
        if (readiness.loading?.total === 0 && readiness.compiling === 0 && readiness.settling === 0) { settled = true; break; }
        if (Date.now() - started >= limitMs) break;
      }
      const seconds = (Date.now() - started) / 1000;
      if (!rendered) continue;
      const captured = await ask({ verb: "capture" });
      if (!captured.ok) { failed(name, { request: "capture", error: captured.error }); continue; }
      const drawn = captured.result?.rendering ?? rendering;
      judged(name, drawn);
      const meshes = drawn.bricks?.meshes_stats ?? {};
      const png = resolve(out, `${String(++shot).padStart(2, "0")}-${name.replace(/[^A-Za-z0-9_]+/g, "-")}.png`);
      await copyFile(resolve(workspace, captured.result.artifact.path), png);
      if (!settled) failures++;
      line({
        event: "view", view: name, png, ticks: ticked,
        ...(settled ? { settle_s: seconds } : { unsettled_after_s: seconds, readiness: rendering.readiness ?? null }),
        drawn: meshes.visible_instances ?? null, culled: meshes.culled_instances ?? null, read_back_age: meshes.read_back_age ?? null,
      });
      await diagnosed(name);
    }
  }
  return failures === 0 ? 0 : 3;
}

async function main() {
  const values = options(process.argv.slice(2));
  checkOptions(values, ["native", "source", "url", "plan", "out", "limit-ms"]);
  if (values.native && values.url) throw new AgentError("invalid_arguments", "--url", { expected: "absent" });
  if (!values.native && !values.url) throw new AgentError("invalid_arguments", "--url", { expected: "present" });
  if (values.source && !values.native) throw new AgentError("invalid_arguments", "--native", { expected: "present" });
  const owned = values.native
    ? await launchNativeSession({ program: resolve(values.native), args: values.source ? ["--scene", values.source] : [] })
    : await launchBrowserSession({ url: values.url });
  try { process.exitCode = await look(owned, values); } finally { await owned.close(); }
}

await main().catch(error => {
  console.log(JSON.stringify({ version: VERSION, ...failure(error) }));
  process.exitCode = 1;
});
