import assert from "node:assert/strict";
import { test } from "node:test";
import { request as httpRequest } from "node:http";
import { createBridge, MAX_BYTES, loopbackUrl } from "./agent-protocol.mjs";

const session = "test-session";
const request = (id, command = { op: "status" }) => ({ version: 1, id, session, command });
const deferred = () => {
  let resolve;
  const promise = new Promise(done => { resolve = done; });
  return { promise, resolve };
};

test("only explicit loopback HTTP targets can own a development browser", () => {
  assert.equal(loopbackUrl("http://127.0.0.1:5173/playground.html?x=1").searchParams.get("engine_dev"), "1");
  for (const url of ["https://localhost/", "http://example.com/", "file:///game.html", "http://localhost.example.com/", "http://user@127.0.0.1/"]) {
    assert.throws(() => loopbackUrl(url), { code: "invalid_url" });
  }
});

test("protocol and session validation happen before execution", async () => {
  let calls = 0;
  const bridge = createBridge({ session, execute: async () => { calls++; return { ok: true }; } });
  assert.throws(() => bridge.submit({ ...request("one"), version: 2 }), { code: "protocol_version" });
  assert.throws(() => bridge.submit({ ...request("one"), session: "old-session" }), { code: "session_mismatch" });
  assert.throws(() => bridge.submit({ ...request("one"), extra: true }), { code: "invalid_request" });
  assert.throws(() => bridge.submit({ ...request("one"), id: ["one"] }), { code: "invalid_request_id" });
  assert.equal(calls, 0);
  assert.deepEqual(await bridge.submit(request("one")), { version: 1, id: "one", session, ok: true });
  assert.equal(calls, 1);
});

test("mutations remain serialized and accepted request IDs never replay", async () => {
  const first = deferred();
  const order = [];
  const bridge = createBridge({ session, execute: async command => {
    order.push(command.op);
    if (command.op === "capture") await first.promise;
    return { ok: true };
  } });
  const capture = bridge.submit(request("one", { op: "capture" }));
  const resume = bridge.submit(request("two", { op: "resume" }));
  await Promise.resolve();
  assert.deepEqual(order, ["capture"]);
  assert.throws(() => bridge.submit(request("one", { op: "step" })), { code: "duplicate_request" });
  first.resolve();
  await Promise.all([capture, resume]);
  assert.deepEqual(order, ["capture", "resume"]);
  assert.throws(() => bridge.submit(request("one")), { code: "duplicate_request" });
});

test("queue capacity includes the executing operation", async () => {
  const block = deferred();
  const bridge = createBridge({ session, queueLimit: 2, execute: async () => { await block.promise; return { ok: true }; } });
  const one = bridge.submit(request("one"));
  const two = bridge.submit(request("two"));
  assert.throws(() => bridge.submit(request("three")), { code: "queue_full" });
  block.resolve();
  await Promise.all([one, two]);
  assert.equal((await bridge.submit(request("three"))).ok, true);
});

test("uncertain timed-out mutation invalidates the session and prevents overlap", async () => {
  let calls = 0;
  let closed = 0;
  const block = deferred();
  const bridge = createBridge({ session, deadlineMs: 20, onTimeout: () => { closed++; }, execute: async () => {
    calls++;
    await block.promise;
    return { ok: true };
  } });
  const one = bridge.submit(request("one"));
  const two = bridge.submit(request("two"));
  assert.equal((await one).error.code, "command_timeout");
  assert.equal((await two).error.code, "session_unavailable");
  assert.equal(calls, 1);
  assert.equal(closed, 1);
  assert.throws(() => bridge.submit(request("three")), { code: "session_unavailable" });
  block.resolve();
});

test("ordinary command errors preserve the session; response bytes are bounded", async () => {
  const bridge = createBridge({ session, execute: async command => {
    if (command.op === "fail") throw new Error("test failure");
    if (command.op === "large") return { ok: true, result: "x".repeat(MAX_BYTES) };
    return { ok: true, result: { observation: { frame: "9007199254740993" } } };
  } });
  assert.equal((await bridge.submit(request("one", { op: "fail" }))).error.code, "bridge_error");
  assert.equal((await bridge.submit(request("two", { op: "large" }))).error.code, "response_too_large");
  assert.equal((await bridge.submit(request("three"))).result.observation.frame, "9007199254740993");
});

test("browser payloads cannot override the transport identity", async () => {
  const bridge = createBridge({ session, execute: async () => ({ ok: true, version: 42, session: "other", id: "other" }) });
  assert.deepEqual(await bridge.submit(request("one")), { ok: true, version: 1, session, id: "one" });
});

test("rejected execution errors cannot exceed the response budget", async () => {
  const bridge = createBridge({ session, execute: async () => { throw Object.assign(new Error("x".repeat(MAX_BYTES * 2)), { code: "c".repeat(MAX_BYTES), path: "p".repeat(MAX_BYTES) }); } });
  const response = await bridge.submit(request("one"));
  assert.equal(response.ok, false);
  assert.ok(Buffer.byteLength(JSON.stringify(response)) < 8192);
});

test("null and undefined rejections remain structured and leave the queue usable", async () => {
  const bridge = createBridge({ session, execute: async command => {
    if (command.op === "null") throw null;
    if (command.op === "undefined") throw undefined;
    return { ok: true };
  } });
  for (const op of ["null", "undefined"]) {
    const response = await bridge.submit(request(op, { op }));
    assert.equal(response.error.code, "bridge_error");
    assert.equal(response.error.message, op);
  }
  assert.equal((await bridge.submit(request("three"))).ok, true);
});

test("unavailable sessions still permit orderly close", async () => {
  const bridge = createBridge({ session, onClose: () => {}, execute: async () => { throw new Error("close must not reach the runtime"); } });
  bridge.markUnavailable();
  assert.throws(() => bridge.submit(request("one")), { code: "session_unavailable" });
  assert.equal((await bridge.submit(request("close", { op: "close" }))).ok, true);
});

test("HTTP discovery, routing, request bounds, and browser-origin rejection", async t => {
  const bridge = createBridge({ session, url: "http://127.0.0.1:5173/?engine_dev=1", execute: async () => ({ ok: true }) });
  await new Promise(resolve => bridge.server.listen(0, "127.0.0.1", resolve));
  t.after(() => new Promise(resolve => { bridge.server.closeAllConnections(); bridge.server.close(resolve); }));
  const endpoint = `http://127.0.0.1:${bridge.server.address().port}`;
  const sessions = await (await fetch(`${endpoint}/sessions`)).json();
  assert.equal(sessions.sessions[0].session, session);
  const denied = await fetch(`${endpoint}/sessions`, { headers: { Origin: "http://remote.example" } });
  assert.equal(denied.status, 403);
  assert.equal((await denied.json()).error.code, "origin_rejected");
  const wrongHost = await new Promise(resolve => {
    httpRequest(`${endpoint}/sessions`, { headers: { Host: "remote.example" } }, response => {
      response.resume();
      resolve(response.statusCode);
    }).end();
  });
  assert.equal(wrongHost, 403);
  const missing = await fetch(`${endpoint}/unknown`);
  assert.equal(missing.status, 404);
  const plain = await fetch(`${endpoint}/request`, { method: "POST", body: "{}" });
  assert.equal((await plain.json()).error.code, "content_type");
  const malformed = await fetch(`${endpoint}/request`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "{" });
  assert.equal((await malformed.json()).error.code, "invalid_json");
  const large = await fetch(`${endpoint}/request`, { method: "POST", headers: { "Content-Type": "application/json" }, body: "x".repeat(MAX_BYTES + 1) });
  assert.equal(large.status, 413);
  const chunkedLarge = await new Promise((resolve, reject) => {
    const outgoing = httpRequest(`${endpoint}/request`, { method: "POST", headers: { "Content-Type": "application/json", "Transfer-Encoding": "chunked" } }, response => {
      let body = "";
      response.on("data", bytes => { body += bytes.toString(); });
      response.on("end", () => resolve({ status: response.statusCode, body: JSON.parse(body) }));
    });
    outgoing.on("error", reject);
    outgoing.write("x".repeat(MAX_BYTES));
    outgoing.end("x");
  });
  assert.equal(chunkedLarge.status, 413);
  assert.equal(chunkedLarge.body.error.code, "request_too_large");
  const accepted = await fetch(`${endpoint}/request`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(request("accepted")) });
  assert.equal((await accepted.json()).ok, true);
});
