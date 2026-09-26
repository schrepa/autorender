import test from "node:test";
import assert from "node:assert/strict";
import { createBridge } from "./agent-protocol.mjs";

const request = (session, id, op = "step") => ({ version: 1, session, id, command: { op } });
const deferred = () => { let resolve; const promise = new Promise(r => { resolve = r; }); return { promise, resolve }; };

test("replacement drains accepted old work and rejects arrivals with stale session identity", async () => {
  const held = deferred();
  const calls = [];
  const bridge = createBridge({ session: "old", url: "old", build: "old-build", execute: async command => { calls.push(command.op); if (calls.length === 1) await held.promise; return { ok: true }; } });
  const first = bridge.submit(request("old", "first"));
  const second = bridge.submit(request("old", "second", "inspect"));
  let committed = false;
  const swapped = bridge.replaceSession({ session: "new", url: "new", build: "new-build", execute: async () => ({ ok: true }), commit: () => { committed = true; } });
  assert.throws(() => bridge.submit(request("old", "late")), { code: "session_replacing" });
  assert.equal(committed, false);
  held.resolve();
  assert.equal((await first).session, "old");
  assert.equal((await first).build, "old-build");
  assert.equal((await second).session, "old");
  await swapped;
  assert.deepEqual(calls, ["step", "inspect"]);
  assert.equal(committed, true);
  assert.throws(() => bridge.submit(request("old", "stale")), { code: "session_mismatch" });
  const current = await bridge.submit(request("new", "first"));
  assert.equal(current.session, "new");
  assert.equal(current.build, "new-build");
});

test("superseded or stopped activation preserves old identity and its replay ledger", async () => {
  const bridge = createBridge({ session: "old", url: "old", execute: async () => ({ ok: true }) });
  await bridge.submit(request("old", "used"));
  await assert.rejects(bridge.replaceSession({ session: "new", validate: async () => false }), { code: "build_superseded" });
  assert.throws(() => bridge.submit(request("old", "used")), { code: "duplicate_request" });
  assert.equal((await bridge.submit(request("old", "unused"))).ok, true);
  await assert.rejects(bridge.replaceSession({ session: "new", commit: () => { throw new Error("stopped"); } }), /stopped/);
  assert.equal((await bridge.submit(request("old", "still"))).session, "old");
});

test("an initially unavailable bridge can recover on first successful activation", async () => {
  const bridge = createBridge({ session: "pending", url: "", initiallyUnavailable: true, execute: async () => ({ ok: false }) });
  assert.throws(() => bridge.submit(request("pending", "first")), { code: "session_unavailable" });
  await bridge.replaceSession({ session: "ready", url: "game", execute: async () => ({ ok: true }) });
  assert.equal((await bridge.submit(request("ready", "first"))).ok, true);
});

test("close accepted before replacement prevents a late session from being published", async () => {
  const held = deferred();
  const bridge = createBridge({ session: "old", url: "", onClose() {}, execute: async () => { await held.promise; return { ok: true }; } });
  const first = bridge.submit(request("old", "first"));
  const close = bridge.submit(request("old", "close", "close"));
  const swap = bridge.replaceSession({ session: "new", execute: async () => ({ ok: true }) });
  const rejected = assert.rejects(swap, { code: "session_closed" });
  held.resolve();
  await first;
  assert.equal((await close).ok, true);
  await rejected;
});
