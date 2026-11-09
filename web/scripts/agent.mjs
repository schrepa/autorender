#!/usr/bin/env node
// @ts-check
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { open } from "node:fs/promises";
import { resolve } from "node:path";
import { AgentError, VERSION, MAX_BYTES, atMost, checkOptions, checkTimeout, createBridge, failure, options, range, variants } from "./agent-protocol.mjs";
import { launchBrowserSession } from "./agent-browser.mjs";
import { launchNativeSession } from "./agent-native.mjs";

const protocol = JSON.parse(readFileSync(new URL("../src/generated/protocol.json", import.meta.url), "utf8"));
const verbs = new Set(protocol.verbs ?? []);
const reading = new Set(["describe", "inspect", "query"]);
const verbOptions = ["port", "session", "file", "json", "id", "timeout-ms", "brief"];
const commands = {
  serve: { options: ["url", "port", "browser", "native", "args", "profiling"] },
  sessions: { options: ["port"] },
};

function checkSwitch(values, name) {
  const value = values[name] ?? "off";
  if (value !== "on" && value !== "off") throw new AgentError("invalid_arguments", `--${name}`, { allowed: variants(["on", "off"]), got: value });
}

function pageUrl(url) {
  if (!url) throw new AgentError("invalid_arguments", "--url", { expected: "present" });
  return url;
}

function portValue(value = "7331") {
  if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > 65_535) throw new AgentError("invalid_arguments", "--port", { allowed: range(1, 65_535), got: value });
  return Number(value);
}

async function readJson(path) {
  const file = await open(path, "r");
  try {
    const buffer = Buffer.alloc(MAX_BYTES + 1);
    let size = 0;
    while (size < buffer.length) {
      const { bytesRead } = await file.read(buffer, size, buffer.length - size, null);
      if (!bytesRead) break;
      size += bytesRead;
    }
    if (size > MAX_BYTES) throw new AgentError("request_too_large", "--file", { allowed: atMost(MAX_BYTES), got: size });
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer.subarray(0, size)));
  } finally { await file.close(); }
}

async function serve(values) {
  checkOptions(values, commands.serve.options);
  const port = portValue(values.port);
  checkSwitch(values, "profiling");
  let owned;
  let bridge;
  let closing;
  const shutdown = (answering = false) => {
    if (!closing) {
      process.removeListener("SIGINT", onSignal);
      process.removeListener("SIGTERM", onSignal);
      bridge?.markUnavailable();
      if (!answering) bridge?.server.closeAllConnections();
      closing = Promise.allSettled([
        owned?.close(),
        bridge?.server.listening ? new Promise(resolveClose => bridge.server.close(resolveClose)) : Promise.resolve(),
      ]);
    }
    return closing;
  };
  function onSignal() { void shutdown(); }
  process.once("SIGINT", onSignal);
  process.once("SIGTERM", onSignal);
  try {
    if (values.native && (values.url || values.browser)) throw new AgentError("invalid_arguments", values.url ? "--url" : "--browser", { expected: "absent" });
    if ((values.args || values.profiling) && !values.native) throw new AgentError("invalid_arguments", "--native", { expected: "present" });
    owned = values.native
      ? await launchNativeSession({ program: resolve(values.native), args: values.args?.split(" ").filter(Boolean).map(option => `--${option}`) ?? [] })
      : await launchBrowserSession({
        url: pageUrl(values.url), browserPath: values.browser,
      });
    if (closing) { await owned.close(); return; }
    bridge = createBridge({
      session: owned.session, url: owned.url,
      onClose: () => shutdown().then(() => process.exit()),
      execute: command => owned.execute(command),
      activity: () => owned.activity?.(),
    });
    owned.setInvalidated(() => { void shutdown(true); });
    await new Promise((resolveListen, reject) => {
      bridge.server.once("error", reject);
      bridge.server.listen(port, "127.0.0.1", resolveListen);
    });
    if (closing) { bridge.server.closeAllConnections(); await new Promise(resolveClose => bridge.server.close(resolveClose)); return; }
    console.log(JSON.stringify({ version: VERSION, ok: true, session: owned.session, url: owned.url, endpoint: `http://127.0.0.1:${port}`, state: "ready" }));
  } catch (error) { await shutdown(); throw error; }
}

async function onlySession(endpoint) {
  const listed = await fetchJson(endpoint, "/sessions", { signal: AbortSignal.timeout(5_000) });
  const first = listed?.sessions?.[0]?.session;
  if (typeof first !== "string") throw new AgentError("invalid_arguments", "--session", { expected: "present" });
  return first;
}

function requestOptions(values) {
  return {
    ...(values["timeout-ms"] === undefined ? {} : { timeout_ms: Number(values["timeout-ms"]) }),
    ...(values.brief ? { brief: true } : {}),
  };
}

async function client(verb, values) {
  checkOptions(values, verb in commands ? commands[verb].options : verbOptions);
  const endpoint = `http://127.0.0.1:${portValue(values.port)}`;
  if (verb === "sessions") {
    return fetchJson(endpoint, "/sessions", { signal: AbortSignal.timeout(5_000) });
  }
  if (values.file && values.json) throw new AgentError("invalid_arguments", "--json", { expected: "absent" });
  const data = values.file ? await readJson(values.file) : values.json ? JSON.parse(values.json) : {};
  const session = values.session ?? await onlySession(endpoint);
  const request = { version: VERSION, id: values.id ?? randomUUID(), session, command: { ...data, verb }, ...requestOptions(values) };
  checkTimeout(request.timeout_ms);
  const body = JSON.stringify(request);
  if (Buffer.byteLength(body) > MAX_BYTES) throw new AgentError("request_too_large", "--json", { allowed: atMost(MAX_BYTES), got: Buffer.byteLength(body) });
  const waited = (request.timeout_ms ?? 0) + 5_000;
  const post = { method: "POST", headers: { "Content-Type": "application/json" }, body, signal: AbortSignal.timeout(Math.max(120_000, waited)) };
  return fetchJson(endpoint, "/request", post, !reading.has(verb));
}

function unreachable(endpoint, error, mutation) {
  const connectionRefused = ["ECONNREFUSED", "ENOTFOUND", "EHOSTUNREACH", "ENETUNREACH"].includes(error?.cause?.code);
  if (mutation && !connectionRefused) {
    return new AgentError("request_outcome_unknown", "request", { got: String(error?.cause?.code ?? error?.message ?? error) });
  }
  return new AgentError("bridge_unreachable", "--port", { got: endpoint });
}

async function fetchJson(endpoint, path, options, mutation = false) {
  try {
    const response = await fetch(`${endpoint}${path}`, options);
    return await response.json();
  } catch (error) {
    throw unreachable(endpoint, error, mutation);
  }
}

async function main() {
  const [verb, ...args] = process.argv.slice(2);
  const values = options(args, ["brief"]);
  if (verb === "serve") { await serve(values); return; }
  if (!(verb in commands) && !verbs.has(verb)) {
    throw new AgentError("unknown_variant", "verb", { allowed: variants([...Object.keys(commands), ...verbs]), got: verb });
  }
  const response = await client(verb, values);
  console.log(JSON.stringify(response));
  if (response.ok === false) process.exitCode = 1;
}

await main().catch(error => {
  console.log(JSON.stringify({ version: VERSION, ...failure(error) }));
  process.exitCode = 1;
});
