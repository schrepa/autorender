// @ts-check
import { execFileSync, spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdir, readFile, readdir, rm, stat } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import { inflateSync } from "node:zlib";
import { setTimeout as delay } from "node:timers/promises";
import { PANIC_PREFIX } from "../src/generated/host.ts";
import { AgentError, atMost, detailOf, range, variants } from "./agent-protocol.mjs";
import { recordReceived, recordSent } from "./exercised.mjs";
import { workspace } from "./workspace.mjs";

const SESSIONS_KEPT = 10;
const READY_TIMEOUT_MS = 300_000;

const clients = JSON.parse(readFileSync(new URL("../src/generated/clients.json", import.meta.url), "utf8"));

export function nativeProgram(name, { profiling = false } = {}) {
  const client = clients.find(client => client.native && (client.package === name || client.native === name));
  if (!client) return resolve(name);
  const [chosen, features] = profiling ? ["profiling", "dev,profiling"] : ["release", "dev"];
  const profile = process.env.ENGINE_NATIVE_PROFILE || chosen;
  const env = profiling && process.platform === "linux"
    ? { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ""} -Cforce-frame-pointers=yes`.trim() }
    : process.env;
  execFileSync("cargo", ["build", "-q", "-p", client.package, "--bin", client.native, "--profile", profile, "--features", features],
    { cwd: workspace, env, stdio: ["ignore", "ignore", "inherit"] });
  const folder = profile === "dev" ? "debug" : profile;
  return resolve(workspace, "target", folder, process.platform === "win32" ? `${client.native}.exe` : client.native);
}

async function pruneSessions(root) {
  const entries = await readdir(root, { withFileTypes: true }).catch(() => []);
  const sessions = await Promise.all(entries.filter(entry => entry.isDirectory()).map(async entry => {
    const path = resolve(root, entry.name);
    return { path, modified: (await stat(path).catch(() => ({ mtimeMs: 0 }))).mtimeMs };
  }));
  sessions.sort((a, b) => b.modified - a.modified);
  await Promise.all(sessions.slice(SESSIONS_KEPT).map(({ path }) => rm(path, { recursive: true, force: true })));
}

function decodePng(bytes) {
  let offset = 8, width = 0, height = 0;
  const data = [];
  while (offset < bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const type = bytes.toString("latin1", offset + 4, offset + 8);
    const body = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = body.readUInt32BE(0); height = body.readUInt32BE(4);
      if (body[8] !== 8 || body[9] !== 6 || body[12] !== 0) throw new AgentError("invalid_texture", "$.compare", { expected: "image", got: { bit_depth: body[8], color_type: body[9], interlace: body[12] } });
    } else if (type === "IDAT") data.push(body);
    offset += 12 + length;
  }
  const raw = inflateSync(Buffer.concat(data));
  const stride = width * 4, pixels = Buffer.alloc(stride * height);
  for (let row = 0; row < height; row++) {
    const filter = raw[row * (stride + 1)];
    const line = raw.subarray(row * (stride + 1) + 1, (row + 1) * (stride + 1));
    const out = row * stride;
    for (let x = 0; x < stride; x++) {
      const left = x >= 4 ? pixels[out + x - 4] : 0;
      const up = row > 0 ? pixels[out - stride + x] : 0;
      const corner = row > 0 && x >= 4 ? pixels[out - stride + x - 4] : 0;
      const predictor = filter === 1 ? left : filter === 2 ? up : filter === 3 ? (left + up) >> 1
        : filter === 4 ? paeth(left, up, corner) : 0;
      pixels[out + x] = (line[x] + predictor) & 255;
    }
  }
  return { width, height, data: pixels };
}

function paeth(a, b, c) {
  const p = a + b - c, pa = Math.abs(p - a), pb = Math.abs(p - b), pc = Math.abs(p - c);
  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

const status = { verb: "query", name: "status" };

async function compareCaptures(command, request) {
  if (!command || typeof command !== "object" || Object.keys(command).some(key => !["a", "b", "tolerance"].includes(key))) {
    throw new AgentError("unknown_field", "$.compare", { allowed: variants(["a", "b", "tolerance"]) });
  }
  const tolerance = command.tolerance ?? 0;
  if (!Number.isInteger(tolerance) || tolerance < 0 || tolerance > 255) throw new AgentError("out_of_range", "$.compare.tolerance", { allowed: range(0, 255), got: tolerance });
  const root = resolve(workspace, "artifacts", "agent");
  const images = [];
  for (const name of ["a", "b"]) {
    const path = typeof command[name] === "string" ? resolve(workspace, command[name]) : "";
    if (!path.startsWith(root + sep) || !path.endsWith(".png")) throw new AgentError("invalid_path", `$.compare.${name}`, { expected: "path", got: command[name] });
    const bytes = await readFile(path).catch(() => { throw new AgentError("invalid_path", `$.compare.${name}`, { expected: "readable", got: command[name] }); });
    images.push(decodePng(bytes));
  }
  const response = await request(status);
  if (!response.ok) return response;
  let result;
  if (images[0].width !== images[1].width || images[0].height !== images[1].height) {
    result = { same_size: false, sizes: images.map(({ width, height }) => [width, height]) };
  } else {
    let different = 0, largest = 0;
    for (let offset = 0; offset < images[0].data.length; offset += 4) {
      let pixel = 0;
      for (let channel = 0; channel < 3; channel++) pixel = Math.max(pixel, Math.abs(images[0].data[offset + channel] - images[1].data[offset + channel]));
      if (pixel > tolerance) different++;
      largest = Math.max(largest, pixel);
    }
    result = { same_size: true, width: images[0].width, height: images[0].height, different, total: images[0].width * images[0].height, largest_difference: largest };
  }
  return { ...response, result: { compare: result, a: command.a, b: command.b, tolerance } };
}

/** @param {{ program: string, args?: string[], session?: string, signal?: AbortSignal }} options */
export async function launchNativeSession({ program, args = [], session = randomUUID(), signal }) {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(session)) throw new AgentError("invalid_session", "session", { expected: "id", got: session });
  const artifacts = resolve(workspace, "artifacts", "agent", session);
  await pruneSessions(resolve(workspace, "artifacts", "agent"));
  await mkdir(artifacts, { recursive: true });
  const child = spawn(program, [...args, "--dev", "--dev-artifacts", artifacts], { cwd: workspace, stdio: ["pipe", "pipe", "pipe"] });
  const diagnostics = [];
  const waiting = [];
  let closing, invalidated = false, invalidationCallback, ready;
  let panic;
  const panicked = () => ({ ok: false, error: { code: "runtime_panicked", path: "$", got: panic } });
  function diagnostic(kind, message) {
    diagnostics.push({ kind, message: message.slice(0, 2000), at: new Date().toISOString() });
    if (diagnostics.length > 100) diagnostics.shift();
  }
  const readyLine = new Promise((resolveReady, rejectReady) => { ready = { resolve: resolveReady, reject: rejectReady }; });
  createInterface({ input: child.stderr }).on("line", line => {
    if (line.startsWith(PANIC_PREFIX)) {
      try { panic = JSON.parse(line.slice(PANIC_PREFIX.length)); } catch {}
    }
    diagnostic("stderr", line);
  });
  createInterface({ input: child.stdout }).on("line", line => {
    let value;
    try { value = JSON.parse(line); } catch { diagnostic("stdout", line); return; }
    if (ready) { const first = ready; ready = undefined; first.resolve(value); return; }
    if (value && typeof value === "object" && "record" in value && !("ok" in value)) {
      waiting[0]?.onRecord?.(value.record);
      return;
    }
    waiting.shift()?.resolve(value);
  });
  function invalidate() {
    if (invalidated) return;
    invalidated = true;
    const error = new AgentError("session_unavailable", "$", { expected: "live", got: { diagnostics: diagnostics.slice(-5) } });
    ready?.reject(error);
    for (const pending of waiting.splice(0)) {
      if (panic) pending.resolve(panicked());
      else pending.reject(error);
    }
    try { invalidationCallback?.(); } finally { void close(); }
  }
  child.on("error", error => { diagnostic("spawn", String(error)); invalidate(); });
  child.on("close", invalidate);
  function close() {
    if (!closing) {
      closing = new Promise(/** @param {(value?: unknown) => void} resolveClose */ resolveClose => {
        if (child.exitCode !== null || child.signalCode !== null) { resolveClose(); return; }
        child.once("exit", () => resolveClose());
        child.stdin.end();
        setTimeout(() => child.kill(), 5000).unref();
      });
    }
    return closing;
  }
  function setInvalidated(callback) {
    invalidationCallback = callback;
    if (invalidated) callback();
  }
  const aborted = () => { void close(); };
  signal?.addEventListener("abort", aborted, { once: true });
  function request(command, onRecord) {
    recordSent(command);
    if (panic && invalidated) return Promise.resolve(panicked());
    if (invalidated || closing) return Promise.reject(new AgentError("session_unavailable", "$", { expected: "live" }));
    return new Promise((resolveReply, rejectReply) => {
      waiting.push({ resolve: resolveReply, reject: rejectReply, onRecord });
      child.stdin.write(`${JSON.stringify(command)}\n`);
    }).then(reply => {
      recordReceived(reply);
      return reply;
    });
  }
  try {
    let timer;
    const identity = await Promise.race([readyLine, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new AgentError("runtime_unavailable", "$", { allowed: atMost(READY_TIMEOUT_MS), got: { diagnostics: diagnostics.slice(-5) } })), READY_TIMEOUT_MS);
    })]).finally(() => clearTimeout(timer));
    if (identity?.ready !== true || typeof identity.instance !== "string") throw new AgentError("missing_runtime_identity", "$", { expected: "id", got: identity?.instance ?? null });
    for (const command of [{ verb: "control", name: "render" }, status]) {
      const response = await request(command);
      if (!response?.ok) throw new AgentError(response?.error?.code ?? "runtime_unavailable", response?.error?.path, detailOf(response?.error));
      if (response.result?.paused !== true) throw new AgentError("runtime_not_paused", "$", { expected: "paused" });
    }
    const owned = {
      session, url: `native:${relative(workspace, program).split(sep).join("/")}`, instance: identity.instance, build: identity.build ?? null,
      adapter: identity.adapter, close, setInvalidated,
      async execute(command, onRecord) {
        if (command.verb === "describe" && command.name === "ProfileCapture") {
          if (Object.keys(command).some(key => !["verb", "name"].includes(key))) throw new AgentError("unknown_field", "$", { allowed: variants(["verb", "name"]) });
          const response = await request(status);
          if (!response.ok) return response;
          const { nativeProfileDescriptor } = await import("./agent-profile.mjs");
          response.result = { ...nativeProfileDescriptor, status: { available: identity.profiling === true, native: true, counters: identity.counters === true, pix: identity.pix === true } };
          return response;
        }
        if (command.verb === "capture" && command.compare !== undefined) {
          const { compare, ...capture } = command;
          if (typeof compare?.b === "string") return compareCaptures(compare, request);
          const captured = await owned.execute(capture);
          if (!captured.ok) return captured;
          const compared = await compareCaptures({ ...compare, b: captured.result.artifact.path }, request);
          if (compared.ok) captured.result.compare = compared.result.compare;
          return compared.ok ? captured : compared;
        }
        if (command.verb === "query" && command.name === "diagnostics") {
          const response = await request(status);
          if (response.ok) response.result = { observation: response.observation ?? response.result?.observation, entries: [...diagnostics] };
          return response;
        }
        const response = await request(command, onRecord);
        if (command.verb === "describe" && command.name === undefined && response.ok && Array.isArray(response.result?.types)) {
          response.result.types.push({ name: "GameControls", kind: "tooling", controls: identity.controls ?? {} });
          response.result.types.push({ name: "PageParameters", kind: "tooling", parameters: identity.parameters ?? {} });
        }
        if (command.verb === "profile" && response.ok && response.result?.artifacts) {
          for (const [kind, path] of Object.entries(response.result.artifacts)) {
            if (typeof path === "string") response.result.artifacts[kind] = relative(workspace, path).split(sep).join("/");
          }
        }
        if (response.ok && typeof response.result?.artifact?.path === "string") {
          const artifact = response.result.artifact;
          artifact.path = relative(workspace, artifact.path).split(sep).join("/");
          const bytes = await readFile(resolve(workspace, artifact.path));
          if (createHash("sha256").update(bytes).digest("hex") !== artifact.sha256) {
            return { ok: false, observation: response.observation, error: { code: "capture_failed", path: "$", expected: "matching", got: artifact.path } };
          }
        }
        return response;
      },
    };
    return owned;
  } catch (error) {
    await close();
    await delay(0);
    throw error;
  }
}
