#!/usr/bin/env node
// @ts-check
import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { open } from "node:fs/promises";
import { AgentError, VERSION, MAX_BYTES, atMost, checkTimeout, createBridge, failure, range, variants } from "./agent-protocol.mjs";
import { launchBrowserSession } from "./agent-browser.mjs";
import { launchNativeSession, nativeProgram } from "./agent-native.mjs";

const protocol = JSON.parse(readFileSync(new URL("../src/generated/protocol.json", import.meta.url), "utf8"));
const verbs = new Set((protocol.verbs ?? []).filter(verb => verb !== "check"));
const reading = new Set(["describe", "inspect", "query"]);
const verbOptions = ["port", "session", "file", "json", "id", "timeout-ms", "rendering"];
const commands = {
  iterate: { options: ["web-port", "port", "game", "browser"] },
  serve: { options: ["url", "port", "browser", "native", "args", "profiling"] },
  ab: { options: ["native", "args", "url", "browser", "a", "b", "rounds", "samples", "warmup", "mode", "profile", "profile-ms"] },
  sessions: { options: ["port"] },
  development: { options: ["port"] },
  request: { options: verbOptions },
  help: { options: ["schema"] },
};

function options(args) {
  const result = {};
  for (let i = 0; i < args.length; i += 2) {
    const key = args[i];
    if (!key.startsWith("--") || !args[i + 1] || args[i + 1].startsWith("--") || key.slice(2) in result) throw new AgentError("invalid_arguments", key, { got: key });
    result[key.slice(2)] = args[i + 1];
  }
  return result;
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

function checkOptions(values, names) {
  for (const key of Object.keys(values)) if (!names.includes(key)) throw new AgentError("invalid_arguments", `--${key}`, { allowed: variants(names.map(name => `--${name}`)), got: `--${key}` });
}

async function serve(values) {
  checkOptions(values, commands.serve.options);
  const port = portValue(values.port);
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
      ? await launchNativeSession({ program: nativeProgram(values.native, { profiling: values.profiling === "on" }), args: values.args?.split(" ").filter(Boolean).map(option => `--${option}`) ?? [] })
      : await launchBrowserSession({
        url: pageUrl(values.url), browserPath: values.browser,
      });
    if (closing) { await owned.close(); return; }
    bridge = createBridge({
      session: owned.session, url: owned.url,
      onClose: () => shutdown().then(() => process.exit()),
      execute: (command, onRecord) => owned.execute(command, onRecord),
      activity: () => owned.activity?.(),
    });
    owned.setInvalidated(() => { void shutdown(true); });
    await new Promise((resolveListen, reject) => {
      bridge.server.once("error", reject);
      bridge.server.listen(port, "127.0.0.1", resolveListen);
    });
    if (closing) { bridge.server.closeAllConnections(); await new Promise(resolveClose => bridge.server.close(resolveClose)); return; }
    console.log(JSON.stringify({ version: VERSION, ok: true, session: owned.session, url: owned.url, build: owned.build, endpoint: `http://127.0.0.1:${port}`, state: "ready" }));
  } catch (error) { await shutdown(); throw error; }
}

async function onlySession(endpoint) {
  const listed = await fetchJson(endpoint, "/sessions", { signal: AbortSignal.timeout(5_000) });
  const first = listed?.sessions?.[0]?.session;
  if (typeof first !== "string") throw new AgentError("invalid_arguments", "--session", { expected: "present" });
  return first;
}

function requestOptions(values) {
  if (values.rendering !== undefined && values.rendering !== "off") throw new AgentError("invalid_arguments", "--rendering", { allowed: variants(["off"]), got: values.rendering });
  return {
    ...(values["timeout-ms"] === undefined ? {} : { timeout_ms: Number(values["timeout-ms"]) }),
    ...(values.rendering === undefined ? {} : { rendering: false }),
  };
}

async function client(verb, values) {
  checkOptions(values, verb in commands ? commands[verb].options : verbOptions);
  const endpoint = `http://127.0.0.1:${portValue(values.port)}`;
  if (verb === "sessions" || verb === "development") {
    return fetchJson(endpoint, `/${verb}`, { signal: AbortSignal.timeout(5_000) });
  }
  if (values.file && values.json) throw new AgentError("invalid_arguments", "--json", { expected: "absent" });
  const data = values.file ? await readJson(values.file) : values.json ? JSON.parse(values.json) : {};
  let request;
  if (verb === "request" && data.version !== undefined) {
    request = { ...data, ...requestOptions(values) };
    if (values.session && values.session !== request.session) throw new AgentError("session_mismatch", "--session", { expected: "matching", got: request.session });
    if (values.id && values.id !== request.id) throw new AgentError("invalid_request_id", "--id", { expected: "matching", got: request.id });
  } else {
    const session = values.session ?? await onlySession(endpoint);
    const command = verb === "request" ? data : { ...data, verb };
    request = { version: VERSION, id: values.id ?? randomUUID(), session, command, ...requestOptions(values) };
  }
  checkTimeout(request.timeout_ms);
  const body = JSON.stringify(request);
  if (Buffer.byteLength(body) > MAX_BYTES) throw new AgentError("request_too_large", "--json", { allowed: atMost(MAX_BYTES), got: Buffer.byteLength(body) });
  const mutation = !reading.has(request.command?.verb);
  const post = { method: "POST", headers: { "Content-Type": "application/json" }, body };
  const waited = (request.timeout_ms ?? 0) + 5_000;
  return request.command?.verb === "watch"
    ? stream(endpoint, { ...post, signal: AbortSignal.timeout(Math.max(600_000, waited)) }, mutation)
    : fetchJson(endpoint, "/request", { ...post, signal: AbortSignal.timeout(Math.max(120_000, waited)) }, mutation);
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

async function stream(endpoint, options, mutation) {
  let last;
  try {
    const response = await fetch(`${endpoint}/request`, options);
    if (!response.body) return await response.json();
    const decoder = new TextDecoder();
    let pending = "";
    const take = line => {
      if (!line.trim()) return;
      const value = JSON.parse(line);
      if (value && "record" in value && !("ok" in value)) console.log(line);
      else last = value;
    };
    for await (const chunk of response.body) {
      pending += decoder.decode(chunk, { stream: true });
      for (let at = pending.indexOf("\n"); at >= 0; at = pending.indexOf("\n")) {
        take(pending.slice(0, at));
        pending = pending.slice(at + 1);
      }
    }
    take(pending);
  } catch (error) {
    throw unreachable(endpoint, error, mutation);
  }
  return last ?? failure(new AgentError("invalid_browser_response", "$", { expected: "present" }));
}

function help([asked, ...rest]) {
  if (rest.length) throw new AgentError("invalid_arguments", rest[0], { expected: "absent", got: rest[0] });
  const reply = { version: VERSION, ok: true };
  if (asked === undefined) return { ...reply, verbs: protocol.verbs, commands: Object.keys(commands), more: ["help VERB", "help COMMAND", "help --schema"] };
  if (asked === "--schema") {
    return {
      ...reply,
      verbs: protocol.verbs, queries: protocol.queries, controls: [...protocol.controls, "close"],
      commands: Object.fromEntries(Object.entries(commands).map(([name, { options }]) => [name, options])),
      verb_options: verbOptions,
    };
  }
  if (asked in commands) return { ...reply, command: asked, options: commands[asked].options };
  if (!protocol.verbs.includes(asked)) throw new AgentError("unknown_variant", "help", { allowed: variants([...protocol.verbs, ...Object.keys(commands), "--schema"]), got: asked });
  const names = { query: protocol.queries, control: [...protocol.controls, "close"] };
  return { ...reply, verb: asked, options: asked === "check" ? ["json", "file"] : verbOptions, ...(asked in names ? { names: names[asked] } : {}) };
}

async function main() {
  const [verb, ...args] = process.argv.slice(2);
  if (verb === "help") { console.log(JSON.stringify(help(args))); return; }
  const values = options(args);
  if (verb === "serve") { await serve(values); return; }
  if (verb === "ab") {
    checkOptions(values, commands.ab.options);
    const { runAb } = await import("./agent-ab.mjs");
    if (values.native && (values.url || values.browser)) throw new AgentError("invalid_arguments", values.url ? "--url" : "--browser", { expected: "absent" });
    const owned = values.native
      ? await launchNativeSession({ program: nativeProgram(values.native, { profiling: values.profile === "on" }), args: values.args?.split(" ").filter(Boolean).map(option => `--${option}`) ?? [] })
      : await launchBrowserSession({ url: pageUrl(values.url), browserPath: values.browser });
    try {
      console.log(JSON.stringify({ version: VERSION, ok: true, session: owned.session, url: owned.url, result: await runAb(owned, values) }));
    } finally { await owned.close(); }
    return;
  }
  if (verb === "iterate") {
    checkOptions(values, commands.iterate.options);
    const { startIteration } = await import("./iteration.mjs");
    await startIteration(values);
    return;
  }
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
