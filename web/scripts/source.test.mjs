import assert from "node:assert/strict";
import test from "node:test";
import { loadSceneSource } from "../src/source.ts";

const LIMIT = 64 * 1024 * 1024;

function streamed(chunks, options = {}) {
  let pulls = 0;
  let canceled = 0;
  const body = new ReadableStream({
    pull(controller) {
      const chunk = chunks[pulls++];
      if (chunk) controller.enqueue(chunk);
      else controller.close();
    },
    cancel() { canceled++; },
  }, { highWaterMark: 0 });
  const response = new Response(body, { status: options.status ?? 200, headers: options.headers });
  return { response, body, get pulls() { return pulls; }, get canceled() { return canceled; } };
}

function fetchResponse(t, response) {
  const previousFetch = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return response;
  };
  t.after(() => { globalThis.fetch = previousFetch; });
  return calls;
}

test("source loading preserves multibyte UTF-8 split across transport chunks", async t => {
  const source = '{"name":"garden 🙂 café 漢字"}\n';
  const bytes = new TextEncoder().encode(source);
  // Single-byte chunks split every multibyte sequence and its continuation bytes.
  const stream = streamed([...bytes].map(byte => Uint8Array.of(byte)));
  const calls = fetchResponse(t, stream.response);
  const lifecycle = new AbortController();
  assert.equal(await loadSceneSource("/scenes/garden.json", lifecycle.signal), source);
  assert.deepEqual(calls, [{ url: "/scenes/garden.json", options: { cache: "no-store", signal: lifecycle.signal } }]);
  assert.equal(stream.canceled, 0);
  assert.equal(stream.body.locked, false);
});

test("declared oversized responses are canceled without reading their body", async t => {
  const stream = streamed([Uint8Array.of(65)], { headers: { "Content-Length": String(LIMIT + 1) } });
  fetchResponse(t, stream.response);
  await assert.rejects(loadSceneSource("/scene.json", new AbortController().signal), /64 MiB source budget/);
  assert.equal(stream.pulls, 0);
  assert.equal(stream.canceled, 1);
  assert.equal(stream.body.locked, false);
});

test("chunked responses cannot exceed the byte budget and stop at the first overflow", async t => {
  const stream = streamed([new Uint8Array(LIMIT).fill(65), Uint8Array.of(66), Uint8Array.of(67)]);
  fetchResponse(t, stream.response);
  await assert.rejects(loadSceneSource("/scene.json", new AbortController().signal), /64 MiB source budget/);
  assert.equal(stream.pulls, 2, "overflow must not read the next chunk");
  assert.equal(stream.canceled, 1);
  assert.equal(stream.body.locked, false);
});

test("the exact byte limit succeeds even when Content-Length understates the body", async t => {
  const stream = streamed([new Uint8Array(LIMIT).fill(65)], { headers: { "Content-Length": "1" } });
  fetchResponse(t, stream.response);
  const source = await loadSceneSource("/scene.json", new AbortController().signal);
  assert.equal(source.length, LIMIT);
  assert.equal(source[0], "A");
  assert.equal(source.at(-1), "A");
  assert.equal(stream.canceled, 0);
  assert.equal(stream.body.locked, false);
});

test("malformed UTF-8 cancels transport instead of replacing invalid bytes", async t => {
  const stream = streamed([Uint8Array.of(0xc3, 0x28), Uint8Array.of(65)]);
  fetchResponse(t, stream.response);
  await assert.rejects(loadSceneSource("/scene.json", new AbortController().signal), TypeError);
  assert.equal(stream.pulls, 1);
  assert.equal(stream.canceled, 1);
  assert.equal(stream.body.locked, false);
});

test("an incomplete UTF-8 sequence at EOF is rejected and its reader released", async t => {
  const stream = streamed([Uint8Array.of(0xf0, 0x9f)]);
  fetchResponse(t, stream.response);
  await assert.rejects(loadSceneSource("/scene.json", new AbortController().signal), TypeError);
  assert.equal(stream.body.locked, false);
});

test("HTTP failures report the status and cancel unread response bodies", async t => {
  const stream = streamed([Uint8Array.of(65)], { status: 503 });
  fetchResponse(t, stream.response);
  await assert.rejects(loadSceneSource("/scene.json", new AbortController().signal), /HTTP 503/);
  assert.equal(stream.pulls, 0);
  assert.equal(stream.canceled, 1);
  assert.equal(stream.body.locked, false);
});

test("network failures propagate without replacing the original cause", async t => {
  const previousFetch = globalThis.fetch;
  const failure = new TypeError("network disconnected");
  globalThis.fetch = async () => { throw failure; };
  t.after(() => { globalThis.fetch = previousFetch; });
  await assert.rejects(loadSceneSource("/scene.json", new AbortController().signal), error => error === failure);
});

test("successful responses without a body fail clearly", async t => {
  fetchResponse(t, new Response(null, { status: 204 }));
  await assert.rejects(loadSceneSource("/scene.json", new AbortController().signal), /no body/);
});
