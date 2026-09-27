/** On-demand adapter, dynamically imported only by the Vite development host. */
import type { Runtime } from "./host";
import { BenchmarkError, benchmarkOptions, readRendering, sampleBenchmark } from "./benchmark.ts";

type Command = { op: string; [field: string]: unknown };
type Observation = { frame: string; tick: string; scene: { id: string; revision: string } | null };
type Reply = {
  version?: number; ok: boolean; observation?: Observation;
  result?: Record<string, unknown>;
  error?: { code: string; path: string; message: string };
};
interface Host {
  game: Runtime;
  signal: AbortSignal;
  paused(): boolean;
  setPaused(value: boolean): void;
  setBusy(value: boolean): void;
  isBusy?(): boolean;
  size(): [number, number];
  poll(): void;
  source?: { url: string; load(signal: AbortSignal): Promise<string>; files?(source: string, signal: AbortSignal): Promise<unknown> };
}

declare global {
  interface Window {
    __engineDev?: { version: 1; instance: string; build: string | null; request(command: Command): Promise<Reply> };
  }
}

/** Commands run between frames; no borrowed Wasm views escape. */
export function connectDevelopment(host: Host): () => void {
  let disposed = false;
  let busy = false;
  const runtime = host.game;
  if (!runtime.dev_command) throw new Error("Development Wasm is missing; run cargo xtask dev.");

  function invoke(command: Command): Reply {
    if (disposed || host.signal.aborted) throw new Error("Development session has been disposed.");
    const [width, height] = host.size();
    return JSON.parse(runtime.dev_command!(JSON.stringify({ ...command, version: 1 }), width, height));
  }
  function error(code: string, path: string, message: string): Reply {
    const reply: Reply = { ok: false, error: { code, path, message } };
    if (!disposed && !host.signal.aborted) {
      try { reply.observation = invoke({ op: "status" }).observation; } catch { /* Preserve the original runtime failure. */ }
    }
    return reply;
  }
  function finish(reply: Reply): Reply {
    if (reply.ok) {
      const rendering = readRendering(runtime.stats());
        reply.result = {
          ...reply.result, observation: reply.observation, paused: host.paused(),
          source: host.source?.url ?? null,
          rendering,
        };
      host.poll();
    }
    return reply;
  }
  function only(command: Command, fields: string[]) {
    return Object.keys(command).every(key => key === "op" || fields.includes(key));
  }
  const api = {
    version: 1 as const,
    instance: crypto.randomUUID(),
    build: new URLSearchParams(globalThis.location?.search ?? "").get("engine_build"),
    async request(command: Command): Promise<Reply> {
      if (disposed || host.signal.aborted) return error("session_closed", "$", "the browser runtime was disposed");
      if (busy || host.isBusy?.()) return error("command_busy", "$", "one development command is already running");
      if (!command || typeof command !== "object" || Array.isArray(command) || typeof command.op !== "string") {
        return error("invalid_request", "$.command", "expected an object with an op string");
      }
      try {
        if (new TextEncoder().encode(JSON.stringify(command)).byteLength > 16 * 1024) {
          return error("request_too_large", "$.command", "command exceeds 16 KiB");
        }
      } catch { return error("invalid_request", "$.command", "command must be serializable JSON"); }
      busy = true;
      host.setBusy(true);
      try {
        let reply: Reply;
        switch (command.op) {
          case "pause":
          case "resume":
            if (!only(command, [])) return error("invalid_request", "$", "pause/resume accept no arguments");
            host.setPaused(command.op === "pause");
            reply = invoke({ op: "status" });
            break;
          case "input":
          case "camera":
          case "culling":
          case "environment":
          case "present":
          case "benchmark":
          case "step":
          case "render":
          case "capture":
          case "reload": {
            if (!host.paused()) return error("pause_required", "$", "pause before input, camera/culling/environment edits, presentation previews, benchmarking, stepping, rendering, capture, or reload");
            if (command.op === "benchmark") {
              const options = benchmarkOptions(command);
              const benchmark = await sampleBenchmark({
                signal: host.signal, invoke, stats: () => runtime.stats(), size: host.size,
                interrupted: () => disposed || !host.paused() || (globalThis.document?.hidden ?? false),
              }, options);
              reply = { ok: true, version: 1, observation: benchmark.observation.end, result: { benchmark, temporary: true } };
            } else if (command.op === "capture") {
              if (!only(command, ["alpha", "gpu"]) || ("gpu" in command && (typeof command.gpu !== "boolean" || !("alpha" in command)))) {
                return error("invalid_request", "$", "capture accepts only an optional interpolation alpha, and gpu with alpha");
              }
              reply = invoke("alpha" in command ? { op: "present", alpha: command.alpha, ...(command.gpu ? { gpu: true } : {}) } : { op: "render" });
              if (!reply.ok) return reply;
              await presented(host.signal);
            } else if (command.op === "reload") {
              if (!only(command, ["expected_revision", "revision"]) ||
                  typeof command.expected_revision !== "string" || !/^(0|[1-9][0-9]*)$/.test(command.expected_revision) ||
                  typeof command.revision !== "string" || !/^(0|[1-9][0-9]*)$/.test(command.revision)) {
                return error("invalid_revision", "$.revision", "reload requires expected_revision and revision as unsigned decimal strings");
              }
              if (!host.source || !runtime.dev_reload) return error("unsupported_command", "$.op", "this game has no authored scene reload adapter");
              const source = await host.source.load(host.signal);
              const files = await host.source.files?.(source, host.signal);
              host.signal.throwIfAborted();
              if (disposed) return error("session_closed", "$", "runtime was disposed during scene fetch");
              reply = JSON.parse(runtime.dev_reload(source, command.expected_revision, command.revision, files));
              if (reply.ok) {
                const rendered = invoke({ op: "render" });
                if (!rendered.ok) return finish(rendered);
                reply.observation = rendered.observation;
              }
            } else {
              reply = invoke(command);
            }
            break;
          }
          default:
            reply = invoke(command);
        }
        if (disposed || host.signal.aborted) return error("session_closed", "$", "runtime was disposed during command");
        return finish(reply);
      } catch (cause) {
        if (disposed || host.signal.aborted) return error("session_closed", "$", "runtime was disposed during command");
        if (cause instanceof BenchmarkError) return error(cause.code, "$", cause.message);
        return error("browser_command_failed", "$", String(cause));
      } finally {
        busy = false;
        host.setBusy(false);
      }
    },
  };
  window.__engineDev = api;
  host.signal.addEventListener("abort", dispose, { once: true });
  function dispose() {
    disposed = true;
    if (window.__engineDev === api) delete window.__engineDev;
  }
  return dispose;
}

function presented(signal: AbortSignal): Promise<void> {
  return new Promise((resolve, reject) => {
    let first = 0;
    let second = 0;
    const timeout = setTimeout(() => finish(new Error("browser presentation timed out")), 5000);
    function finish(error?: Error) {
      clearTimeout(timeout);
      cancelAnimationFrame(first);
      cancelAnimationFrame(second);
      signal.removeEventListener("abort", aborted);
      if (error) reject(error); else resolve();
    }
    function aborted() { finish(new Error("session closed")); }
    signal.addEventListener("abort", aborted, { once: true });
    if (signal.aborted) { aborted(); return; }
    first = requestAnimationFrame(() => { second = requestAnimationFrame(() => finish()); });
  });
}
