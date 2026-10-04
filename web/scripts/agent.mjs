#!/usr/bin/env node
/** Machine-readable local agent client and development browser owner. */
import { randomUUID } from "node:crypto";
import { open } from "node:fs/promises";
import { AgentError, VERSION, MAX_BYTES, createBridge, failure } from "./agent-protocol.mjs";
import { launchBrowserSession } from "./agent-browser.mjs";

const verbs = new Set(["status", "describe", "inspect", "assets", "raycast", "overlap", "shape_cast", "collision_events", "terrain_height", "terrain_sweep", "pause", "resume", "step", "input", "camera", "culling", "environment", "terrain", "present", "benchmark", "profile", "reload", "capture", "diagnostics", "close"]);

function options(args) {
  const result = {};
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i];
    if (!key.startsWith("--") || !args[i + 1] || args[i + 1].startsWith("--") || key.slice(2) in result) throw new AgentError("arguments", `Expected unique --name value options; received ${key}.`);
    result[key.slice(2)] = args[i + 1];
  }
  return result;
}

function portValue(value = "7331") {
  if (!/^\d+$/.test(value) || Number(value) < 1 || Number(value) > 65_535) throw new AgentError("arguments", "Port must be an integer from 1 to 65535.", "port");
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
    if (size > MAX_BYTES) throw new AgentError("request_too_large", "Request file exceeds the 1 MiB limit.", "file");
    return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(buffer.subarray(0, size)));
  } finally { await file.close(); }
}

function checkOptions(values, names) {
  for (const key of Object.keys(values)) if (!names.includes(key)) throw new AgentError("arguments", `Unknown option --${key}.`);
}

async function serve(values) {
  checkOptions(values, ["url", "port", "browser"]);
  const port = portValue(values.port);
  let owned;
  let bridge;
  let closing;
  const shutdown = () => {
    if (!closing) {
      process.removeListener("SIGINT", onSignal);
      process.removeListener("SIGTERM", onSignal);
      bridge?.markUnavailable();
      bridge?.server.closeAllConnections();
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
    owned = await launchBrowserSession({
      url: values.url ?? "http://127.0.0.1:5173/playground.html", browserPath: values.browser,
    });
    if (closing) { await owned.close(); return; }
    bridge = createBridge({
      session: owned.session, url: owned.url,
      onTimeout: () => owned.close(),
      onClose: shutdown,
      execute: command => owned.execute(command),
    });
    owned.setInvalidated(() => bridge.markUnavailable());
    await new Promise((resolveListen, reject) => {
      bridge.server.once("error", reject);
      bridge.server.listen(port, "127.0.0.1", resolveListen);
    });
    if (closing) { bridge.server.closeAllConnections(); await new Promise(resolveClose => bridge.server.close(resolveClose)); return; }
    console.log(JSON.stringify({ version: VERSION, ok: true, session: owned.session, url: owned.url, build: owned.build, endpoint: `http://127.0.0.1:${port}`, state: "ready" }));
  } catch (error) { await shutdown(); throw error; }
}

async function client(verb, values) {
  checkOptions(values, ["port", "session", "file", "json", "id"]);
  const endpoint = `http://127.0.0.1:${portValue(values.port)}`;
  if (verb === "sessions" || verb === "development") {
    return fetchJson(endpoint, `/${verb}`, { signal: AbortSignal.timeout(5_000) });
  }
  if (values.file && values.json) throw new AgentError("arguments", "Use either --file or --json, not both.");
  const data = values.file ? await readJson(values.file) : values.json ? JSON.parse(values.json) : {};
  let request;
  if (verb === "request" && data.version !== undefined) {
    request = data;
    if (values.session && values.session !== request.session) throw new AgentError("session_mismatch", "The --session option must match the request file.");
    if (values.id && values.id !== request.id) throw new AgentError("invalid_request_id", "The --id option must match the request file.");
  } else {
    if (!values.session) throw new AgentError("arguments", "Provide --session from the serve or sessions response.");
    const command = verb === "request" ? data : { ...data, op: verb };
    request = { version: VERSION, id: values.id ?? randomUUID(), session: values.session, command };
  }
  const body = JSON.stringify(request);
  if (Buffer.byteLength(body) > MAX_BYTES) throw new AgentError("request_too_large", "Serialized request exceeds the 1 MiB limit.");
  return fetchJson(endpoint, "/request", {
    method: "POST", headers: { "Content-Type": "application/json" }, body, signal: AbortSignal.timeout(20_000),
  }, !new Set(["status", "describe", "inspect", "assets", "diagnostics", "terrain_height", "terrain_sweep"]).has(request.command?.op));
}

async function fetchJson(endpoint, path, options, mutation = false) {
  try {
    const response = await fetch(`${endpoint}${path}`, options);
    return await response.json();
  } catch (error) {
    const connectionRefused = ["ECONNREFUSED", "ENOTFOUND", "EHOSTUNREACH", "ENETUNREACH"].includes(error?.cause?.code);
    if (mutation && !connectionRefused) {
      throw new AgentError("request_outcome_unknown", `The bridge at ${endpoint} did not return a complete response. The command may have executed; do not retry it. Discover sessions and inspect state with a new request ID. ${error?.message ?? error}`, "request");
    }
    throw new AgentError("bridge_unreachable", `Cannot reach the development bridge at ${endpoint}. Start cargo xtask agent watch (or agent serve), and check its --port. ${error?.message ?? error}`, "endpoint");
  }
}

async function main() {
  const [verb, ...args] = process.argv.slice(2);
  const values = options(args);
  if (verb === "serve") { await serve(values); return; }
  if (verb === "watch") {
    checkOptions(values, ["web-port", "port", "game", "browser"]);
    const { startIteration } = await import("./iteration.mjs");
    await startIteration(values);
    return;
  }
  if (verb !== "sessions" && verb !== "development" && verb !== "request" && !verbs.has(verb)) throw new AgentError("arguments", "Use watch, serve, development, sessions, request, status, describe, inspect, assets, raycast, overlap, shape_cast, collision_events, terrain_height, terrain_sweep, pause, resume, step, input, camera, culling, environment, terrain, present, benchmark, reload, capture, diagnostics, or close.");
  const response = await client(verb, values);
  console.log(JSON.stringify(response));
  if (response.ok === false) process.exitCode = 1;
}

await main().catch(error => {
  console.log(JSON.stringify({ version: VERSION, ...failure(error) }));
  process.exitCode = 1;
});
