/** Real Rust edit/build/browser verification. Restores only source bytes it owns. */
import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { randomUUID } from "node:crypto";
import { readFile, writeFile, mkdir } from "node:fs/promises";
import { createServer } from "node:net";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { setTimeout as delay } from "node:timers/promises";
import { fileURLToPath } from "node:url";
import { stopProcessTree } from "./iteration-process.mjs";

const workspace = fileURLToPath(new URL("../../", import.meta.url));
const sourcePath = join(workspace, "examples/playground/src/gameplay.rs");
const original = await readFile(sourcePath);
const source = original.toString("utf8");
const movement = "Vec3::new(input.move_right, 0.0, -input.move_forward).normalize_or_zero() * dt * 6.5;";
assert.equal(source.split(movement).length, 2, "Expected exactly one Crystal Garden movement speed expression.");
const slower = Buffer.from(source.replace(movement, movement.replace("6.5;", "3.25;")));
const marker = `iteration-smoke-${randomUUID()}`;
const broken = Buffer.concat([original, Buffer.from(`\ncompile_error!("${marker}");\n`)]);
const output = join(workspace, "artifacts/agent", marker);
await mkdir(output, { recursive: true });
await writeFile(join(output, "gameplay.original.rs"), original, { flag: "wx" });
let lastWritten = original;
let edited = false;

// Hold both reservations together so the assigned ports cannot be identical.
const reservations = [createServer(), createServer()];
await Promise.all(reservations.map(server => new Promise(resolve => server.listen(0, "127.0.0.1", resolve))));
const [webPort, bridgePort] = reservations.map(server => server.address().port);
await Promise.all(reservations.map(server => new Promise(resolve => server.close(resolve))));
const endpoint = `http://127.0.0.1:${bridgePort}`;
const child = spawn(process.execPath, [join(workspace, "web/scripts/agent.mjs"), "watch", "--web-port", String(webPort), "--port", String(bridgePort), "--game", "playground"], {
  cwd: workspace, windowsHide: true, detached: process.platform !== "win32", stdio: ["ignore", "pipe", "pipe"],
});
let stderr = "";
let lastEvent;
let watch;
let spawnError;
let exited = false;
child.stderr.on("data", bytes => { stderr = (stderr + bytes.toString()).slice(-65_536); });
const exit = new Promise(resolve => {
  child.once("error", error => { spawnError = error; exited = true; resolve({ error }); });
  child.once("exit", (code, signal) => { exited = true; resolve({ code, signal }); });
});
const lines = createInterface({ input: child.stdout });
lines.on("line", line => {
  lastEvent = line.slice(-16_384);
  try {
    const event = JSON.parse(line);
    if (event.event === "watch_ready") watch = event.watch;
  } catch { /* A malformed event is included in any subsequent timeout report. */ }
});

async function development() {
  const response = await fetch(`${endpoint}/development`, { signal: AbortSignal.timeout(5000) });
  const envelope = await response.json();
  assert.equal(envelope.ok, true, JSON.stringify(envelope));
  assert.equal(envelope.result.watch, watch, "The endpoint must belong to this test's watcher.");
  return envelope.result;
}

async function waitFor(label, predicate, { ignoreFailureBuild } = {}) {
  const deadline = Date.now() + 660_000;
  let last;
  while (Date.now() < deadline) {
    if (exited) throw new Error(`Watcher exited during ${label}: ${spawnError ?? stderr}\n${lastEvent ?? ""}`);
    if (watch) {
      last = await development();
      if (predicate(last)) {
        process.stderr.write(`${JSON.stringify({ phase: label, state: last.state, build: last.latest?.build })}\n`);
        return last;
      }
      if (last.watcher_error) throw new Error(`Watcher failed during ${label}: ${JSON.stringify(last.watcher_error)}`);
      if (last.latest?.state === "failed" && label !== "compiler_failure" && last.latest.build !== ignoreFailureBuild) {
        throw new Error(`Build failed during ${label}: ${JSON.stringify(last.latest)}`);
      }
    }
    await delay(250);
  }
  throw new Error(`Timed out during ${label}: ${JSON.stringify(last)}\n${stderr}\n${lastEvent ?? ""}`);
}

async function request(session, command, expected = true) {
  const response = await fetch(`${endpoint}/request`, {
    method: "POST", headers: { "Content-Type": "application/json" }, signal: AbortSignal.timeout(20_000),
    body: JSON.stringify({ version: 1, id: randomUUID(), session, command }),
  });
  const envelope = await response.json();
  assert.equal(envelope.ok, expected, JSON.stringify(envelope));
  if (expected) assert.equal(envelope.session, session);
  return envelope;
}

async function inspect(session) {
  return request(session, { op: "inspect", entity: "player", component: "Transform" });
}

async function distance(session) {
  const before = await inspect(session);
  await request(session, { op: "input", right: 1 });
  const stepped = await request(session, { op: "step", ticks: 6 });
  assert.equal(BigInt(stepped.observation.tick) - BigInt(before.observation.tick), 6n);
  const after = await inspect(session);
  await request(session, { op: "input" });
  return after.result.entity.components.Transform.translation[0] - before.result.entity.components.Transform.translation[0];
}

async function edit(bytes) {
  assert.deepEqual(await readFile(sourcePath), lastWritten, `Source changed outside this smoke test; refusing to overwrite ${sourcePath}. Backup: ${join(output, "gameplay.original.rs")}`);
  await writeFile(sourcePath, bytes);
  lastWritten = bytes;
  edited = true;
}

let report;
let failure;
try {
  const first = await waitFor("initial_ready", status => status.latest?.state === "ready" && Boolean(status.active));
  const firstSession = first.active.session;
  const baseline = await distance(firstSession);
  assert.ok(Math.abs(baseline - 0.65) < 0.0001, `Expected speed 6.5 over six ticks, observed ${baseline}.`);
  const preserved = await inspect(firstSession);

  await edit(broken);
  const failed = await waitFor("compiler_failure", status => status.latest?.state === "failed" && status.latest.build !== first.active.build);
  assert.deepEqual(failed.active, first.active, "Failed compilation must preserve the active build and session.");
  const diagnostic = failed.latest.result?.diagnostics?.find(item => item.message?.includes(marker));
  assert.ok(diagnostic, `Compiler error must be structured: ${JSON.stringify(failed.latest)}`);
  assert.ok(diagnostic.spans.some(span => span.file.replaceAll("\\", "/").endsWith("examples/playground/src/gameplay.rs") && span.line_start > 0 && span.column_start > 0), "Compiler error must identify its source location.");
  await writeFile(join(output, "compiler-failure.json"), JSON.stringify(failed, null, 2));
  const unchanged = await inspect(firstSession);
  assert.deepEqual(unchanged.result.entity.components.Transform, preserved.result.entity.components.Transform);
  assert.equal(unchanged.observation.tick, preserved.observation.tick);
  const stillRunning = await request(firstSession, { op: "step", ticks: 1 });
  assert.equal(BigInt(stillRunning.observation.tick) - BigInt(preserved.observation.tick), 1n);

  await edit(slower);
  const second = await waitFor("replacement_ready", status => status.latest?.state === "ready" && status.active?.build !== first.active.build, { ignoreFailureBuild: failed.latest.build });
  assert.notEqual(second.active.session, firstSession, "A changed Rust runtime needs a fresh session identity.");
  assert.notEqual(second.active.fingerprint, first.active.fingerprint);
  const stale = await request(firstSession, { op: "step", ticks: 1 }, false);
  assert.equal(stale.error.code, "session_mismatch");
  const changedDistance = await distance(second.active.session);
  assert.ok(Math.abs(changedDistance - baseline / 2) < 0.0001, `Changed Rust speed must halve movement: ${baseline} → ${changedDistance}.`);
  const capture = await request(second.active.session, { op: "capture" });
  const png = await readFile(join(workspace, capture.result.artifact.path));
  assert.equal(png.subarray(0, 8).toString("hex"), "89504e470d0a1a0a");
  assert.ok(png.length > 1000);
  assert.equal(capture.session, second.active.session);
  assert.equal(capture.build, second.active.build);
  assert.equal((await development()).active.build, second.active.build);
  const diagnostics = await request(second.active.session, { op: "diagnostics" });
  assert.deepEqual(diagnostics.result.entries.filter(entry => ["pageerror", "error"].includes(entry.kind)), []);
  report = {
    ok: true, previous_build: first.active.build, build: second.active.build,
    previous_session: firstSession, session: second.active.session,
    movement: { before: baseline, after: changedDistance }, artifact: capture.result.artifact.path,
    checks: ["automatic_rust_detection", "compiler_error_spans", "failed_build_preserves_runtime", "fresh_session", "stale_session_rejected", "changed_gameplay", "capture", "diagnostics"],
  };
} catch (error) { failure = error; }
finally {
  const cleanupErrors = [];
  if (edited) {
    try {
      assert.deepEqual(await readFile(sourcePath), lastWritten, `Source changed outside the test. Refusing restoration; original backup: ${join(output, "gameplay.original.rs")}`);
      await writeFile(sourcePath, original);
    } catch (error) { cleanupErrors.push(error); }
  }
  if (!exited && watch) {
    try {
      await development(); // Verify watcher ownership before sending close.
      const discovery = await (await fetch(`${endpoint}/sessions`, { signal: AbortSignal.timeout(5000) })).json();
      await request(discovery.sessions[0].session, { op: "close" });
    } catch (error) { cleanupErrors.push(error); }
  }
  if (!exited) {
    let timer;
    const stopped = await Promise.race([exit.then(() => true), new Promise(resolve => { timer = setTimeout(() => resolve(false), 30_000); })]);
    clearTimeout(timer);
    if (!stopped) {
      cleanupErrors.push(new Error("Watcher did not stop after close; terminating only its owned process tree."));
      await stopProcessTree(child);
      await exit;
    }
  }
  lines.close();
  // Closed endpoints prove the owned listeners were released, rather than merely hidden.
  for (const port of [webPort, bridgePort]) {
    try {
      const probe = createServer();
      await new Promise((resolve, reject) => { probe.once("error", reject); probe.listen(port, "127.0.0.1", resolve); });
      await new Promise(resolve => probe.close(resolve));
    } catch (error) { cleanupErrors.push(new Error(`Owned port ${port} was not released: ${error}`)); }
  }
  if (cleanupErrors.length) failure = new AggregateError([...(failure ? [failure] : []), ...cleanupErrors], "Iteration smoke cleanup failed.");
}
if (failure) throw failure;
report.checks.push("source_restored", "owned_services_stopped");
await writeFile(join(output, "result.json"), JSON.stringify(report, null, 2));
console.log(JSON.stringify(report));
