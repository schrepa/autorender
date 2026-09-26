import assert from "node:assert/strict";
import { setTimeout as delay } from "node:timers/promises";
import test from "node:test";
import { IterationController } from "./iteration-controller.mjs";

const snapshot = (value = "a") => ({ fingerprint: value.repeat(64), files: 1 });
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
}

async function until(condition) {
  const deadline = Date.now() + 2000;
  while (!condition()) {
    if (Date.now() > deadline) throw new Error("Expected controller progress.");
    await delay(1);
  }
}

function harness(overrides = {}) {
  const calls = { build: [], prepare: [], promote: [], discard: [], events: [] };
  let current = snapshot();
  const controller = new IterationController({
    build: async (attempt, source) => { calls.build.push(attempt); return overrides.build?.(attempt, source) ?? { ok: true }; },
    prepare: async (attempt, result) => { calls.prepare.push(attempt); return overrides.prepare?.(attempt, result) ?? { session: `session-${attempt.id}` }; },
    promote: async (candidate, source, commit) => {
      calls.promote.push(candidate);
      if (overrides.promote) await overrides.promote(candidate, source, commit);
      else commit();
    },
    discard: async (attempt, candidate) => { calls.discard.push({ attempt, candidate }); await overrides.discard?.(attempt, candidate); },
    fingerprint: async () => overrides.fingerprint?.() ?? current,
    onEvent: value => { calls.events.push(value); return overrides.onEvent?.(value); },
    signal: overrides.signal,
  });
  return { controller, calls, setCurrent(value, enqueue = true) {
    current = snapshot(value);
    if (enqueue) controller.enqueue(current);
  } };
}

test("successful iterations expose identified generations and immutable status values", async () => {
  const { controller, calls, setCurrent } = harness();
  assert.deepEqual(controller.status(), { state: "idle" });
  setCurrent("a");
  await controller.whenIdle();
  const status = controller.status();
  assert.equal(status.state, "ready");
  assert.equal(status.active.fingerprint, snapshot().fingerprint);
  assert.equal(status.active.build, calls.build[0].id);
  assert.match(status.active.build, /^[a-f0-9-]{36}$/);
  assert.equal(status.latest.build, status.active.build);
  assert.equal(status.latest.result.ok, true);
  assert.deepEqual(calls.events.map(value => value.state), ["building", "preparing", "activating", "ready"]);
  status.active.session = "tampered";
  status.latest.result.ok = false;
  assert.notEqual(controller.status().active.session, "tampered");
  assert.equal(controller.status().latest.result.ok, true);
  assert.equal(calls.discard.length, 0);
  await controller.stop();
});

test("compilation failure preserves the active browser and waits for a new edit", async () => {
  const { controller, calls, setCurrent } = harness({ build: async attempt => attempt.fingerprint === snapshot("b").fingerprint
    ? { ok: false, error: { code: "compile_failed", message: "unknown variable" }, diagnostics: [{ level: "error", line: 4 }] }
    : { ok: true } });
  setCurrent("a");
  await controller.whenIdle();
  const active = controller.status().active;
  setCurrent("b");
  await controller.whenIdle();
  assert.equal(controller.status().state, "failed");
  assert.deepEqual(controller.status().active, active);
  assert.equal(controller.status().latest.error.code, "compile_failed");
  assert.equal(controller.status().latest.result.diagnostics[0].line, 4);
  setCurrent("b");
  await controller.whenIdle();
  assert.equal(calls.build.length, 2);
  assert.equal(calls.discard.length, 1);
  setCurrent("c");
  await controller.whenIdle();
  assert.equal(controller.status().active.fingerprint, snapshot("c").fingerprint);
  await controller.stop();
});

test("an active build keeps only the latest pending edit and never overlaps builds", async () => {
  const gate = deferred();
  let running = 0;
  let maximum = 0;
  const { controller, calls, setCurrent } = harness({ build: async attempt => {
    maximum = Math.max(maximum, ++running);
    if (attempt.fingerprint === snapshot("a").fingerprint) await gate.promise;
    running--;
    return { ok: true };
  } });
  setCurrent("a");
  await until(() => calls.build.length === 1);
  setCurrent("b");
  setCurrent("c");
  setCurrent("d");
  gate.resolve();
  await controller.whenIdle();
  assert.deepEqual(calls.build.map(value => value.fingerprint), [snapshot("a").fingerprint, snapshot("d").fingerprint]);
  assert.equal(maximum, 1);
  assert.equal(calls.prepare.length, 1);
  assert.equal(calls.discard[0].attempt.id, calls.build[0].id);
  assert.equal(controller.status().active.fingerprint, snapshot("d").fingerprint);
  assert.ok(calls.events.some(value => value.state === "superseded"));
  await controller.stop();
});

test("a fresh disk scan discovers an edit before its watcher notification", async () => {
  const state = harness({ build: async attempt => {
    if (attempt.fingerprint === snapshot("a").fingerprint) state.setCurrent("b", false);
    return { ok: true };
  } });
  state.setCurrent("a");
  await state.controller.whenIdle();
  assert.deepEqual(state.calls.build.map(value => value.fingerprint), [snapshot("a").fingerprint, snapshot("b").fingerprint]);
  assert.equal(state.calls.prepare.length, 1);
  assert.equal(state.controller.status().active.fingerprint, snapshot("b").fingerprint);
  await state.controller.stop();
});

test("source changes during browser preparation discard that candidate", async () => {
  const gate = deferred();
  const { controller, calls, setCurrent } = harness({ prepare: async attempt => {
    if (attempt.fingerprint === snapshot("a").fingerprint) await gate.promise;
    return { session: `session-${attempt.id}` };
  } });
  setCurrent("a");
  await until(() => calls.prepare.length === 1);
  setCurrent("b");
  gate.resolve();
  await controller.whenIdle();
  assert.equal(calls.discard.length, 1);
  assert.equal(calls.discard[0].candidate.session, `session-${calls.build[0].id}`);
  assert.equal(calls.promote.length, 1);
  assert.equal(controller.status().active.fingerprint, snapshot("b").fingerprint);
  await controller.stop();
});

test("the synchronous commit rejects a newer edit during promotion", async () => {
  const state = harness({ promote: async (_candidate, source, commit) => {
    if (source.fingerprint === snapshot("a").fingerprint) state.setCurrent("b");
    commit();
  } });
  state.setCurrent("a");
  await state.controller.whenIdle();
  assert.equal(state.calls.discard.length, 1);
  assert.equal(state.controller.status().active.fingerprint, snapshot("b").fingerprint);
  assert.ok(state.calls.events.some(value => value.latest.error?.code === "build_superseded"));
  await state.controller.stop();
});

test("restoring the active source discards a pending generation without rebuilding the active one", async () => {
  const gate = deferred();
  const { controller, calls, setCurrent } = harness({ build: async attempt => {
    if (attempt.fingerprint === snapshot("b").fingerprint) await gate.promise;
    return { ok: true };
  } });
  setCurrent("a");
  await controller.whenIdle();
  const active = controller.status().active;
  setCurrent("b");
  await until(() => calls.build.length === 2);
  setCurrent("a");
  gate.resolve();
  await controller.whenIdle();
  assert.equal(calls.build.length, 2);
  assert.deepEqual(controller.status().active, active);
  assert.equal(calls.discard.length, 1);
  await controller.stop();
});

test("stop waits for a running build and prevents late preparation or activation", async () => {
  const gate = deferred();
  const { controller, calls, setCurrent } = harness({ build: async () => { await gate.promise; return { ok: true }; } });
  setCurrent("a");
  await until(() => calls.build.length === 1);
  let stopped = false;
  const stop = controller.stop().then(() => { stopped = true; });
  await delay(5);
  assert.equal(stopped, false);
  assert.equal(controller.status().state, "stopped");
  gate.resolve();
  await stop;
  assert.equal(calls.prepare.length, 0);
  assert.equal(calls.discard.length, 1);
  assert.equal(controller.status().active, undefined);
  setCurrent("b");
  await controller.whenIdle();
  assert.equal(calls.build.length, 1);
});

test("abort during promotion rejects its commit and preserves the previous active session", async () => {
  const lifecycle = new AbortController();
  const gate = deferred();
  const { controller, calls, setCurrent } = harness({ signal: lifecycle.signal, promote: async (_candidate, source, commit) => {
    if (source.fingerprint === snapshot("b").fingerprint) await gate.promise;
    commit();
  } });
  setCurrent("a");
  await controller.whenIdle();
  const active = controller.status().active;
  setCurrent("b");
  await until(() => calls.promote.length === 2);
  lifecycle.abort();
  gate.resolve();
  await controller.whenIdle();
  assert.equal(controller.status().state, "stopped");
  assert.deepEqual(controller.status().active, active);
  assert.equal(calls.discard.length, 1);
  assert.equal(controller.status().latest.error.code, "iteration_stopped");
});

test("pre-aborted controllers never start work", async () => {
  const lifecycle = new AbortController();
  lifecycle.abort();
  const { controller, calls, setCurrent } = harness({ signal: lifecycle.signal });
  setCurrent("a");
  await controller.whenIdle();
  assert.equal(controller.status().state, "stopped");
  assert.equal(calls.build.length, 0);
});

test("retirement failures after commit preserve the committed session", async () => {
  const { controller, calls, setCurrent } = harness({ promote: async (_candidate, _source, commit) => {
    commit();
    throw Object.assign(new Error("old browser did not close"), { code: "retirement_failed" });
  } });
  setCurrent("a");
  await controller.whenIdle();
  assert.equal(controller.status().state, "ready");
  assert.equal(controller.status().active.build, calls.build[0].id);
  assert.equal(controller.status().latest.retirement_error.code, "retirement_failed");
  assert.equal(calls.discard.length, 0);
  await controller.stop();
});

test("prepare and activation errors preserve the active session and dispose failed candidates", async () => {
  const { controller, calls, setCurrent } = harness({
    prepare: async attempt => {
      if (attempt.fingerprint === snapshot("b").fingerprint) throw new Error("browser failed to start");
      return { session: `session-${attempt.id}` };
    },
    promote: async (_candidate, source, commit) => {
      if (source.fingerprint === snapshot("c").fingerprint) throw new Error("bridge rejected candidate");
      commit();
    },
  });
  setCurrent("a");
  await controller.whenIdle();
  const active = controller.status().active;
  for (const value of ["b", "c"]) {
    setCurrent(value);
    await controller.whenIdle();
    assert.equal(controller.status().state, "failed");
    assert.deepEqual(controller.status().active, active);
  }
  assert.equal(calls.discard.length, 2);
  assert.equal(calls.discard[0].candidate, undefined);
  assert.equal(calls.discard[1].candidate.session, `session-${calls.build[2].id}`);
  await controller.stop();
});

test("promotion must commit once before returning, and saved callbacks expire", async () => {
  let saved;
  const { controller, calls, setCurrent } = harness({ promote: async (_candidate, _source, commit) => { saved = commit; } });
  setCurrent("a");
  await controller.whenIdle();
  assert.equal(controller.status().latest.error.code, "activation_not_committed");
  assert.equal(calls.discard.length, 1);
  assert.throws(saved, { code: "activation_expired" });
  assert.equal(controller.status().active, undefined);
  await controller.stop();
});

test("cleanup failures and bounded adapter errors are observable without retry loops", async () => {
  const { controller, calls, setCurrent } = harness({
    build: async () => { throw Object.assign(new Error("x".repeat(8000)), { code: "compile_failed", path: "y".repeat(500) }); },
    discard: async () => { throw new Error("cleanup failed"); },
    onEvent: () => { throw new Error("observer failed"); },
  });
  setCurrent("a");
  await controller.whenIdle();
  const status = controller.status();
  assert.equal(status.state, "failed");
  assert.equal(status.latest.error.message.length, 4000);
  assert.equal(status.latest.error.path.length, 256);
  assert.equal(status.latest.cleanup_error.message, "cleanup failed");
  assert.equal(calls.build.length, 1);
  await controller.stop();
});

test("oversized or malformed build results never reach browser preparation", async () => {
  for (const result of [{ ok: true, diagnostics: "x".repeat(512 * 1024) }, { success: true }]) {
    const { controller, calls, setCurrent } = harness({ build: async () => result });
    setCurrent("a");
    await controller.whenIdle();
    assert.equal(controller.status().state, "failed");
    assert.equal(calls.prepare.length, 0);
    assert.equal(calls.discard.length, 1);
    assert.ok(["invalid_build_result", "build_result_too_large"].includes(controller.status().latest.error.code));
    await controller.stop();
  }
});

test("invalid fingerprints are rejected before starting a build", async () => {
  const { controller, calls } = harness();
  assert.throws(() => controller.enqueue({ fingerprint: "arbitrary", files: 1 }), { code: "invalid_fingerprint" });
  assert.throws(() => controller.enqueue({ ...snapshot(), files: 20_001 }), { code: "invalid_fingerprint" });
  await controller.whenIdle();
  assert.equal(calls.build.length, 0);
  await controller.stop();
});
