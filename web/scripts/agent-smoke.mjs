/** End-to-end agent workflow using an isolated, disposable authored scene. */
import assert from "node:assert/strict";
import { spawn, execFile } from "node:child_process";
import { readFile, writeFile, mkdir, rm, unlink } from "node:fs/promises";
import { createServer } from "node:net";
import { createInterface } from "node:readline";
import { promisify } from "node:util";
import { fileURLToPath } from "node:url";
import { join } from "node:path";
import { createHash, randomUUID } from "node:crypto";
import { strictStats } from "./harness.mjs";

const executeFile = promisify(execFile);
const workspace = fileURLToPath(new URL("../../", import.meta.url));
const script = join(workspace, "web/scripts/agent.mjs");
const fixtureName = `agent-smoke-${randomUUID()}.scene.json`;
// Local scenes are not tracked, so a fresh checkout has no folder for them.
const scenes = join(workspace, "web/public/scenes");
const sourcePath = join(scenes, fixtureName);
const sourceBytes = await readFile(join(workspace, "web/scripts/fixtures/garden.scene.json"));
const source = JSON.parse(sourceBytes.toString("utf8"));
assert.ok(Number.isSafeInteger(source.revision), "The smoke fixture revision must fit a JavaScript integer.");
const playerId = source.entities.find(entity => entity.tags?.includes("player"))?.id;
assert.ok(playerId, "The authored sample needs a player.");
// Each run replaces the last one's output.
const output = join(workspace, "artifacts/agent-smoke");
await rm(output, { recursive: true, force: true });
await mkdir(output, { recursive: true });
await mkdir(scenes, { recursive: true });
await writeFile(sourcePath, sourceBytes, { flag: "wx" });

const reservation = createServer();
await new Promise(resolve => reservation.listen(0, "127.0.0.1", resolve));
const port = reservation.address().port;
await new Promise(resolve => reservation.close(resolve));
const url = new URL(process.env.AGENT_URL ?? new URL("stress.html", `${process.env.DEMO_URL ?? "http://127.0.0.1:5173"}/`).href);
url.searchParams.set("source", `/scenes/${fixtureName}`);
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
  if (response.result?.rendering) response.result.rendering = strictStats(response.result.rendering);
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
  // The scene schema arrives in parts: its top level, then any definition.
  const sceneDescriptor = await cli("describe", { name: "SceneDocument" });
  assert.ok(sceneDescriptor.result.schema.definitions.includes("SceneEnvironment"));
  const environmentSchema = await cli("describe", { name: "SceneDocument", definition: "SceneEnvironment" });
  assert.ok(environmentSchema.result.schema.properties.terrain);
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
  // Captures name their bytes, hash, frame and tick.
  assert.equal(capture.result.artifact.bytes, png.length);
  assert.equal(capture.result.artifact.sha256, createHash("sha256").update(png).digest("hex"));
  assert.equal(capture.result.artifact.tick, after.observation.tick);
  const effects = rendering => [rendering.shadows.active, rendering.ambient_occlusion.active, rendering.bloom.active];
  assert.deepEqual(effects(capture.result.rendering), [true, true, true]);
  // Every reply's stats hold exactly the fields FrameStats' schema describes.
  const statsSchema = (await cli("describe", { name: "FrameStats" })).result.schema;
  const described = (schema, stats, path) => {
    assert.deepEqual(Object.keys(stats).sort(), Object.keys(schema.properties).sort(), path);
    for (const [field, property] of Object.entries(schema.properties)) {
      const definition = property.$ref?.split("/").at(-1);
      if (definition) described(statsSchema.$defs[definition], stats[field], `${path}.${field}`);
    }
  };
  described(statsSchema, capture.result.rendering, "rendering");

  // The look: discovered, inspected in scene-section shape, changed temporarily.
  const listing = await cli("describe");
  // The bridge adds the page's parameters and controls, and status names the build.
  const parameters = listing.result.types.find(type => type.name === "PageParameters")?.parameters ?? {};
  assert.ok(parameters.source && parameters.engine_dev && parameters.taa, `page parameters: ${Object.keys(parameters)}`);
  assert.ok(listing.result.types.some(type => type.name === "GameControls"));
  const statusReply = (await cli("status")).result;
  assert.ok(statusReply.build.features.includes("dev") && typeof statusReply.build.version === "string", JSON.stringify(statusReply.build));
  assert.ok("adapter" in statusReply);
  for (const name of ["Atmosphere", "Shadows", "AmbientOcclusion", "Bloom", "Environment"]) {
    assert.ok(listing.result.types.some(type => type.name === name), name);
  }
  // The inspection holds every section the environment's schema names.
  const look = await cli("inspect", { component: "Environment" });
  const sections = (await cli("describe", { name: "Environment" })).result.descriptor.schema.properties;
  assert.deepEqual(Object.keys(look.result.value).sort(), Object.keys(sections).sort());
  const mesh = await cli("inspect", { entity: playerId, component: "Mesh3d" });
  assert.equal(typeof mesh.result.entity.components.Mesh3d.material.surface.roughness, "number");
  const invalidLook = await cli("environment", { shadows: { resolution: 3000 } }, false);
  assert.equal(invalidLook.error.code, "invalid_environment");
  assert.equal(invalidLook.error.path, "$.shadows.resolution");
  assert.equal(invalidLook.observation.tick, after.observation.tick);
  assert.deepEqual((await cli("inspect", { component: "Environment" })).result.value, look.result.value);
  // Opt-in effects start off; an empty section turns one on with its defaults.
  assert.deepEqual([capture.result.rendering.sun_shafts.active, capture.result.rendering.reflections.active], [false, false]);
  const night = await cli("environment", { atmosphere: { preset: "night" }, bloom: { intensity: 0 }, sun_shafts: {}, reflections: {} });
  assert.equal(night.result.temporary, true);
  assert.deepEqual(night.result.environment_changed, ["atmosphere", "bloom", "sun_shafts", "reflections"]);
  assert.deepEqual(effects(night.result.rendering), [true, true, false]);
  assert.deepEqual([night.result.rendering.sun_shafts.active, night.result.rendering.reflections.active], [true, true]);
  assert.equal(night.observation.tick, after.observation.tick);
  const nightLook = await cli("inspect", { component: "Environment" });
  assert.notDeepEqual(nightLook.result.value.atmosphere, look.result.value.atmosphere);
  assert.deepEqual(nightLook.result.value.shadows, look.result.value.shadows);
  const nightCapture = await cli("capture");
  assert.notDeepEqual(await readFile(join(workspace, nightCapture.result.artifact.path)), png);
  // Captures compare: the same one to itself, the night to the day.
  const same = (await cli("compare", { a: capture.result.artifact.path, b: capture.result.artifact.path })).result.compare;
  assert.equal(same.different, 0);
  const changed = (await cli("compare", { a: capture.result.artifact.path, b: nightCapture.result.artifact.path, tolerance: 8 })).result.compare;
  assert.ok(changed.different > 1000 && changed.different <= changed.total, `the night differs: ${JSON.stringify(changed)}`);

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
  candidate.version = 6;
  candidate.environment = { atmosphere: { preset: "dusk" }, ambient_occlusion: { intensity: 0 } };
  candidate.entities.find(entity => entity.id === playerId).name = "Agent verified player";
  // Version 6: a light and a particle emitter on a transform-only node, and a
  // glass box where the camera looks.
  candidate.materials.push({ id: "agent-glass", color: [0.6, 0.85, 1.0, 1.0], opacity: 0.4 });
  candidate.entities.push(
    { id: "agent-lamp", transform: { translation: [0, 3, 0] }, light: { intensity: 20, range: 8 }, particles: { rate: 20, lifetime: 1 } },
    { id: "agent-glass-box", mesh: source.meshes[0].id, material: "agent-glass", transform: { translation: [0, 1, 0] } },
  );
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
  // The saved environment omits the opt-in effects: they are off again.
  assert.deepEqual([savedLook.sun_shafts.intensity, savedLook.reflections.intensity], [0, 0]);
  assert.deepEqual([reloaded.result.rendering.sun_shafts.active, reloaded.result.rendering.reflections.active], [false, false]);
  const lamp = await cli("inspect", { entity: "agent-lamp", component: "LocalLight" });
  assert.equal(lamp.result.entity.components.LocalLight.intensity, 20);
  const emitter = await cli("inspect", { entity: "agent-lamp", component: "ParticleEmitter" });
  assert.equal(emitter.result.entity.components.ParticleEmitter.rate, 20);
  const glass = await cli("inspect", { entity: "agent-glass-box", component: "Mesh3d" });
  assert.ok(Math.abs(glass.result.entity.components.Mesh3d.material.opacity - 0.4) < 1e-6);
  assert.ok(reloaded.result.rendering.local_lights.shaded >= 1);
  assert.ok(reloaded.result.rendering.particles.drawn > 0);
  assert.ok(reloaded.result.rendering.transparent_draws >= 1);
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
  console.log(JSON.stringify({ ok: true, session, artifact: capture.result.artifact.path, checks: ["discovery", "description", "inspection", "precise_step", "input", "capture", "environment_inspection", "environment_control", "failed_reload", "source_validation", "guarded_save", "reload_revision", "saved_environment", "version_six_lights_particles_glass", "resume_pause", "diagnostics"] }));
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
