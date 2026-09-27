/** End-to-end agent workflow using an isolated, disposable authored scene. */
import assert from "node:assert/strict";
import { spawn, execFile } from "node:child_process";
import { readFile, writeFile, mkdir, unlink } from "node:fs/promises";
import { createServer } from "node:net";
import { createInterface } from "node:readline";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { randomUUID } from "node:crypto";

const executeFile = promisify(execFile);
const workspace = fileURLToPath(new URL("../../", import.meta.url));
const script = join(workspace, "web/scripts/agent.mjs");
const fixtureName = `agent-smoke-${randomUUID()}.scene.json`;
const sourcePath = join(workspace, "web/public/scenes", fixtureName);
const sourceBytes = await readFile(join(workspace, "web/public/scenes/playground.scene.json"));
const source = JSON.parse(sourceBytes.toString("utf8"));
assert.ok(Number.isSafeInteger(source.revision), "The smoke fixture revision must fit a JavaScript integer.");
const playerId = source.entities.find(entity => entity.tags?.includes("player"))?.id;
assert.ok(playerId, "The authored sample needs a player.");
const output = join(workspace, "artifacts/agent", `smoke-${randomUUID()}`);
await mkdir(output, { recursive: true });
await writeFile(sourcePath, sourceBytes, { flag: "wx" });

const reservation = createServer();
await new Promise(resolve => reservation.listen(0, "127.0.0.1", resolve));
const port = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const url = new URL(process.env.AGENT_URL ?? new URL("playground.html", `${process.env.DEMO_URL ?? "http://127.0.0.1:5173"}/`).href);
url.searchParams.set("scene", `/scenes/${fixtureName}`);
const child = spawn(process.execPath, [script, "serve", "--url", url.href, "--port", String(port)], { cwd: workspace, windowsHide: true, stdio: ["ignore", "pipe", "pipe"] });
let stderr = "";
child.stderr.on("data", bytes => { stderr = (stderr + bytes.toString()).slice(-65_536); });
const lines = createInterface({ input: child.stdout });
let session;

async function cli(verb, fields = {}, expectOk = true) {
  const args = [script, verb, "--port", String(port)];
  if (verb !== "sessions") args.push("--session", session, "--json", JSON.stringify(fields));
  let stdout;
  let code = 0;
  try { ({ stdout } = await executeFile(process.execPath, args, { cwd: workspace, windowsHide: true, timeout: 25_000 })); }
  catch (error) { stdout = error.stdout; code = error.code; }
  const response = JSON.parse(stdout.trim());
  if (verb !== "sessions") {
    assert.equal(response.ok, expectOk, JSON.stringify(response));
    assert.equal(response.session, session);
    assert.equal(typeof response.id, "string");
    assert.equal(code === 0, expectOk);
  }
  return response;
}

async function sceneCommand(args, expectOk = true) {
  let stdout;
  let code = 0;
  try { ({ stdout } = await executeFile(process.env.CARGO ?? "cargo", ["xtask", "scene", ...args], { cwd: workspace, windowsHide: true, timeout: 120_000, maxBuffer: 1024 * 1024 })); }
  catch (error) { stdout = error.stdout; code = error.code; }
  assert.equal(code === 0, expectOk, stdout);
  const response = JSON.parse(stdout.trim().split(/\r?\n/).at(-1));
  assert.equal(response.ok, expectOk, JSON.stringify(response));
  return response;
}

try {
  const readiness = await new Promise((resolve, reject) => {
    const timer = setTimeout(() => reject(new Error(`Bridge startup timed out: ${stderr}`)), 90_000);
    child.once("error", error => { clearTimeout(timer); reject(error); });
    child.once("exit", code => { clearTimeout(timer); reject(new Error(`Bridge exited ${code}: ${stderr}`)); });
    lines.once("line", line => {
      clearTimeout(timer);
      try { resolve(JSON.parse(line)); } catch (error) { reject(error); }
    });
  });
  assert.equal(readiness.ok, true, JSON.stringify(readiness));
  session = readiness.session;
  const discovery = await cli("sessions");
  assert.equal(discovery.sessions[0].session, session);

  const status = await cli("status");
  assert.equal(status.result.paused, true);
  assert.equal(status.observation.scene.revision, String(source.revision));
  const descriptor = await cli("describe", { name: "Transform" });
  assert.ok(descriptor.result.descriptor.schema);
  const sceneDescriptor = await cli("describe", { name: "SceneDocument" });
  assert.ok(sceneDescriptor.result.schema);
  const inspectPlayer = () => cli("inspect", { entity: playerId, component: "Transform" });
  const before = await inspectPlayer();
  const invalid = await cli("inspect", { limit: 101 }, false);
  assert.equal(invalid.observation.tick, before.observation.tick);
  await cli("input", { forward: 1 });
  const stepped = await cli("step", { ticks: 30 });
  assert.equal(BigInt(stepped.observation.tick) - BigInt(before.observation.tick), 30n);
  const after = await inspectPlayer();
  assert.ok(after.result.entity.components.Transform.translation[2] < before.result.entity.components.Transform.translation[2]);
  await cli("input", { forward: 0 });
  const badStep = await cli("step", { ticks: 0 }, false);
  assert.equal(badStep.observation.tick, after.observation.tick);
  const capture = await cli("capture");
  assert.equal(capture.observation.tick, after.observation.tick);
  const png = await readFile(join(workspace, capture.result.artifact.path));
  assert.equal(png.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
  assert.ok(png.length > 1000);
  const effects = rendering => [rendering.shadows_enabled, rendering.ambient_occlusion, rendering.bloom];
  assert.deepEqual(effects(capture.result.rendering), [true, true, true]);

  // The look: discovered, inspected in scene-section shape, changed temporarily.
  const listing = await cli("describe");
  for (const name of ["Atmosphere", "Shadows", "AmbientOcclusion", "Bloom", "Environment"]) {
    assert.ok(listing.result.types.some(type => type.name === name), name);
  }
  const look = await cli("inspect", { component: "Environment" });
  assert.deepEqual(Object.keys(look.result.value).sort(), ["ambient_occlusion", "atmosphere", "bloom", "shadows"]);
  const mesh = await cli("inspect", { entity: playerId, component: "Mesh3d" });
  assert.equal(typeof mesh.result.entity.components.Mesh3d.surface.roughness, "number");
  const invalidLook = await cli("environment", { shadows: { resolution: 3000 } }, false);
  assert.equal(invalidLook.error.code, "invalid_environment");
  assert.equal(invalidLook.error.path, "$.shadows.resolution");
  assert.equal(invalidLook.observation.tick, after.observation.tick);
  assert.deepEqual((await cli("inspect", { component: "Environment" })).result.value, look.result.value);
  const night = await cli("environment", { atmosphere: { preset: "night" }, bloom: { intensity: 0 } });
  assert.equal(night.result.temporary, true);
  assert.deepEqual(night.result.environment_changed, ["atmosphere", "bloom"]);
  assert.deepEqual(effects(night.result.rendering), [true, true, false]);
  assert.equal(night.observation.tick, after.observation.tick);
  const nightLook = await cli("inspect", { component: "Environment" });
  assert.notDeepEqual(nightLook.result.value.atmosphere, look.result.value.atmosphere);
  assert.deepEqual(nightLook.result.value.shadows, look.result.value.shadows);
  const nightCapture = await cli("capture");
  assert.notDeepEqual(await readFile(join(workspace, nightCapture.result.artifact.path)), png);

  const rejectedReload = await cli("reload", { expected_revision: "0", revision: String(source.revision + 1) }, false);
  assert.equal(rejectedReload.observation.scene.revision, String(source.revision));
  const invalidCandidate = structuredClone(source);
  invalidCandidate.revision++;
  invalidCandidate.entities[0].material = "agent-missing-material";
  const invalidPath = join(output, "invalid.scene.json");
  await writeFile(invalidPath, JSON.stringify(invalidCandidate));
  await sceneCommand(["validate", "--file", invalidPath], false);
  await sceneCommand(["save", "--file", sourcePath, "--candidate", invalidPath, "--expected-revision", String(source.revision)], false);
  assert.deepEqual(await readFile(sourcePath), sourceBytes);
  const legacyLook = structuredClone(source);
  legacyLook.revision++;
  legacyLook.environment = { bloom: { intensity: 1 } };
  const legacyPath = join(output, "legacy-environment.scene.json");
  await writeFile(legacyPath, JSON.stringify(legacyLook));
  const legacyError = await sceneCommand(["validate", "--file", legacyPath], false);
  assert.deepEqual([legacyError.error.code, legacyError.error.path], ["version_required", "$.environment"]);

  const candidate = structuredClone(source);
  candidate.revision++;
  candidate.version = 5;
  candidate.environment = { atmosphere: { preset: "dusk" }, ambient_occlusion: { intensity: 0 } };
  candidate.entities.find(entity => entity.id === playerId).name = "Agent verified player";
  const candidatePath = join(output, "candidate.scene.json");
  await writeFile(candidatePath, JSON.stringify(candidate, null, 2));
  await sceneCommand(["validate", "--file", candidatePath]);
  await sceneCommand(["save", "--file", sourcePath, "--candidate", candidatePath, "--expected-revision", String(source.revision)]);
  await sceneCommand(["save", "--file", sourcePath, "--candidate", candidatePath, "--expected-revision", String(source.revision)], false);
  const reloaded = await cli("reload", { expected_revision: String(source.revision), revision: String(candidate.revision) });
  assert.equal(reloaded.observation.scene.revision, String(candidate.revision));
  assert.equal((await inspectPlayer()).result.entity.name, "Agent verified player");
  // A saved environment owns the whole look: omitted sections take defaults,
  // replacing the temporary night control.
  const savedLook = (await cli("inspect", { component: "Environment" })).result.value;
  assert.equal(savedLook.ambient_occlusion.intensity, 0);
  assert.deepEqual(savedLook.bloom, look.result.value.bloom);
  assert.deepEqual(savedLook.shadows, look.result.value.shadows);
  assert.notDeepEqual(savedLook.atmosphere, nightLook.result.value.atmosphere);
  assert.notDeepEqual(savedLook.atmosphere, look.result.value.atmosphere);
  assert.deepEqual(effects(reloaded.result.rendering), [true, false, true]);
  const assets = await cli("request", { op: "assets" });
  assert.ok(assets.result.meshes >= 4);
  const resumed = await cli("resume");
  assert.equal(resumed.result.paused, false);
  await cli("step", { ticks: 1 }, false);
  await cli("pause");
  const diagnostics = await cli("diagnostics");
  assert.deepEqual(diagnostics.result.entries.filter(entry => entry.kind === "pageerror" || entry.kind === "error"), []);
  console.log(JSON.stringify({ ok: true, session, artifact: capture.result.artifact.path, checks: ["discovery", "description", "inspection", "precise_step", "input", "capture", "environment_inspection", "environment_control", "failed_reload", "source_validation", "guarded_save", "reload_revision", "saved_environment", "resume_pause", "diagnostics"] }));
} finally {
  lines.close();
  try {
    if (child.exitCode === null) {
      const exited = new Promise(resolve => child.once("exit", resolve));
      if (session) await cli("close");
      else child.kill("SIGTERM");
      await exited;
    }
  } finally { await unlink(sourcePath); }
}
