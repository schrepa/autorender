/** Own one native development runtime: a game's program serving the
 * development protocol over standard input and output (`--dev`), drawing
 * offscreen, as pages serve it to browser sessions. */
import { execFileSync, spawn } from "node:child_process";
import { createHash, randomUUID } from "node:crypto";
import { mkdir, readFile, readdir, rm, stat } from "node:fs/promises";
import { relative, resolve, sep } from "node:path";
import { createInterface } from "node:readline";
import { fileURLToPath } from "node:url";
import { inflateSync } from "node:zlib";
import { setTimeout as delay } from "node:timers/promises";
import { AgentError } from "./agent-protocol.mjs";

const workspace = fileURLToPath(new URL("../../", import.meta.url));
// Sessions' captures stay under artifacts/agent/SESSION, as browser
// sessions' do; past the most recent SESSIONS_KEPT, the oldest go.
const SESSIONS_KEPT = 10;
const READY_TIMEOUT_MS = 300_000;

/** The program `native` names: a workspace package's native binary
 * (`NAME-native`), built with its development tools (and with `profiling`,
 * its profiling captures), or a path to one built so. */
export function nativeProgram(native, { profiling = false } = {}) {
  if (!/^[a-z][a-z0-9-]*$/.test(native)) return resolve(native);
  // Profiling builds keep line tables (the `profiling` profile) and, on
  // Linux, frame pointers, which the in-process sampler follows there.
  const [profile, features] = profiling ? ["profiling", "dev,profiling"] : ["release", "dev"];
  const env = profiling && process.platform === "linux"
    ? { ...process.env, RUSTFLAGS: `${process.env.RUSTFLAGS ?? ""} -Cforce-frame-pointers=yes`.trim() }
    : process.env;
  execFileSync("cargo", ["build", "-q", "-p", native, "--profile", profile, "--features", features], { cwd: workspace, env, stdio: ["ignore", "ignore", "inherit"] });
  return resolve(workspace, "target", profile, process.platform === "win32" ? `${native}-native.exe` : `${native}-native`);
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

/** RGBA8 pixels of a PNG the native host wrote: 8-bit RGBA, not
 * interlaced, any row filters. */
function decodePng(bytes) {
  let offset = 8, width = 0, height = 0;
  const data = [];
  while (offset < bytes.length) {
    const length = bytes.readUInt32BE(offset);
    const type = bytes.toString("latin1", offset + 4, offset + 8);
    const body = bytes.subarray(offset + 8, offset + 8 + length);
    if (type === "IHDR") {
      width = body.readUInt32BE(0); height = body.readUInt32BE(4);
      if (body[8] !== 8 || body[9] !== 6 || body[12] !== 0) throw new Error("only 8-bit RGBA PNGs without interlacing compare");
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

/** How many pixels of two captures under artifacts/agent differ by more
 * than `tolerance` (0 to 255) in a color channel: `{a, b, tolerance}`. */
async function compareCaptures(command, request) {
  if (Object.keys(command).some(key => !["op", "a", "b", "tolerance"].includes(key))) {
    throw new AgentError("unknown_field", "compare takes a, b and tolerance.", "$");
  }
  const tolerance = command.tolerance ?? 0;
  if (!Number.isInteger(tolerance) || tolerance < 0 || tolerance > 255) throw new AgentError("out_of_range", "tolerance is 0 to 255.", "tolerance");
  const root = resolve(workspace, "artifacts", "agent");
  const images = [];
  for (const name of ["a", "b"]) {
    const path = typeof command[name] === "string" ? resolve(workspace, command[name]) : "";
    if (!path.startsWith(root + sep) || !path.endsWith(".png")) throw new AgentError("invalid_path", `${name} must name a capture under artifacts/agent.`, name);
    const bytes = await readFile(path).catch(() => { throw new AgentError("invalid_path", `${command[name]} cannot be read.`, name); });
    images.push(decodePng(bytes));
  }
  const response = await request({ op: "status" });
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

export async function launchNativeSession({ program, args = [], session = randomUUID(), signal }) {
  if (!/^[A-Za-z0-9][A-Za-z0-9_-]{0,95}$/.test(session)) throw new AgentError("invalid_session", "Session identity must be a bounded identifier.", "session");
  const artifacts = resolve(workspace, "artifacts", "agent", session);
  await pruneSessions(resolve(workspace, "artifacts", "agent"));
  await mkdir(artifacts, { recursive: true });
  const child = spawn(program, [...args, "--dev", "--dev-artifacts", artifacts], { cwd: workspace, stdio: ["pipe", "pipe", "pipe"] });
  const diagnostics = [];
  const waiting = [];
  let closing, invalidated = false, invalidationCallback, ready;
  function diagnostic(kind, message) {
    diagnostics.push({ kind, message: message.slice(0, 2000), at: new Date().toISOString() });
    if (diagnostics.length > 100) diagnostics.shift();
  }
  const readyLine = new Promise((resolveReady, rejectReady) => { ready = { resolve: resolveReady, reject: rejectReady }; });
  createInterface({ input: child.stderr }).on("line", line => diagnostic("stderr", line));
  createInterface({ input: child.stdout }).on("line", line => {
    let value;
    try { value = JSON.parse(line); } catch { diagnostic("stdout", line); return; }
    if (ready) { const first = ready; ready = undefined; first.resolve(value); return; }
    waiting.shift()?.resolve(value);
  });
  function invalidate() {
    if (invalidated) return;
    invalidated = true;
    const error = new AgentError("session_unavailable", `The native runtime exited. Diagnostics: ${JSON.stringify(diagnostics.slice(-5))}`);
    ready?.reject(error);
    for (const pending of waiting.splice(0)) pending.reject(error);
    try { invalidationCallback?.(); } finally { void close(); }
  }
  child.on("error", error => { diagnostic("spawn", String(error)); invalidate(); });
  child.on("exit", invalidate);
  function close() {
    if (!closing) {
      closing = new Promise(resolveClose => {
        if (child.exitCode !== null || child.signalCode !== null) { resolveClose(); return; }
        child.once("exit", () => resolveClose());
        child.stdin.end();
        // A runtime that does not end with its input is stopped.
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
  function request(command) {
    if (invalidated || closing) return Promise.reject(new AgentError("session_unavailable", "The native runtime is no longer available."));
    return new Promise((resolveReply, rejectReply) => {
      waiting.push({ resolve: resolveReply, reject: rejectReply });
      child.stdin.write(`${JSON.stringify(command)}\n`);
    });
  }
  try {
    let timer;
    const identity = await Promise.race([readyLine, new Promise((_, reject) => {
      timer = setTimeout(() => reject(new AgentError("runtime_unavailable", `The native runtime did not start. Diagnostics: ${JSON.stringify(diagnostics.slice(-5))}`)), READY_TIMEOUT_MS);
    })]).finally(() => clearTimeout(timer));
    if (identity?.ready !== true || typeof identity.instance !== "string") throw new AgentError("missing_runtime_identity", "The native runtime must announce its instance identity.");
    for (const op of ["render", "status"]) {
      const response = await request({ op });
      if (!response?.ok) throw new AgentError(response?.error?.code ?? "runtime_unavailable", response?.error?.message ?? "The native runtime rejected initialization.", response?.error?.path);
      if (response.result?.paused !== true) throw new AgentError("runtime_not_paused", "A new development runtime must start paused.");
    }
    return {
      session, url: `native:${relative(workspace, program).split(sep).join("/")}`, instance: identity.instance, build: identity.build ?? null,
      adapter: identity.adapter, close, setInvalidated,
      async execute(command) {
        if (command.op === "describe" && command.name === "ProfileCapture") {
          if (Object.keys(command).some(key => !["op", "name"].includes(key))) throw new AgentError("invalid_request", "describe accepts only name.");
          const response = await request({ op: "status" });
          if (!response.ok) return response;
          const { profileDescriptor } = await import("./agent-profile.mjs");
          response.result = { ...profileDescriptor, status: { available: identity.profiling === true, native: true } };
          return response;
        }
        if (command.op === "compare") return compareCaptures(command, request);
        if (command.op === "diagnostics") {
          const response = await request({ op: "status" });
          if (response.ok) response.result = { observation: response.observation ?? response.result?.observation, entries: [...diagnostics] };
          return response;
        }
        const response = await request(command);
        if (command.op === "describe" && command.name === undefined && response.ok && Array.isArray(response.result?.types)) {
          response.result.types.push({ name: "GameControls", kind: "tooling", description: "The game's own controls, which the game control runs by name with args", controls: identity.controls ?? {} });
          response.result.types.push({ name: "PageParameters", kind: "tooling", description: "The native program's options the host and game read", parameters: identity.parameters ?? {} });
        }
        if (command.op === "profile" && response.ok && response.result?.artifacts) {
          for (const [kind, path] of Object.entries(response.result.artifacts)) {
            if (typeof path === "string") response.result.artifacts[kind] = relative(workspace, path).split(sep).join("/");
          }
        }
        if (command.op === "capture" && response.ok && response.result?.artifact) {
          const artifact = response.result.artifact;
          artifact.path = relative(workspace, artifact.path).split(sep).join("/");
          const bytes = await readFile(resolve(workspace, artifact.path));
          if (createHash("sha256").update(bytes).digest("hex") !== artifact.sha256) {
            return { ok: false, observation: response.observation, error: { code: "capture_failed", path: "$", message: "the capture changed after it was written" } };
          }
        }
        return response;
      },
    };
  } catch (error) {
    await close();
    await delay(0);
    throw error;
  }
}
