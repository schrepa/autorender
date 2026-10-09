/** On-demand adapter, dynamically imported only by the Vite development host. */
import { readStats, type Runtime } from "./host";
import { BenchmarkError, type BenchmarkOptions, type Reply, sampleBenchmark } from "./benchmark.ts";
// The protocol's version, request limit and ops, generated from the engine
// (`cargo xtask schema`).
import protocol from "./generated/protocol.json";

type Command = { op: string; [field: string]: unknown };
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
    __engineDev?: { version: number; instance: string; build: string | null; request(command: Command): Promise<Reply> };
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
    return JSON.parse(runtime.dev_command!(JSON.stringify({ ...command, version: protocol.version }), width, height, host.paused()));
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
      const rendering = readStats(runtime);
        reply.result = {
          ...reply.result, observation: reply.observation, paused: host.paused(),
          source: host.source?.url ?? null,
          rendering,
        };
      host.poll();
    }
    return reply;
  }
  /** The controls the engine hands back for the page to carry out. */
  const hostControls: Record<string, (control: Command) => Promise<Reply>> = {
    async pause() {
      host.setPaused(true);
      return invoke({ op: "status" });
    },
    async resume() {
      host.setPaused(false);
      return invoke({ op: "status" });
    },
    async benchmark(control) {
      const options = control as unknown as BenchmarkOptions;
      const benchmark = await sampleBenchmark({
        signal: host.signal, invoke, stats: () => readStats(runtime), size: host.size,
        interrupted: () => disposed || !host.paused() || (globalThis.document?.hidden ?? false),
      }, options);
      return { ok: true, version: protocol.version, observation: benchmark.observation.end, result: { benchmark, temporary: true } };
    },
    async capture(control) {
      const reply = invoke("alpha" in control ? { op: "present", alpha: control.alpha, ...(control.gpu ? { gpu: true } : {}) } : { op: "render" });
      if (reply.ok) await presented(host.signal);
      return reply;
    },
    async reload(control) {
      if (!host.source || !runtime.dev_reload) return error("unsupported_command", "$.op", "this game has no authored scene reload adapter");
      const source = await host.source.load(host.signal);
      const files = await host.source.files?.(source, host.signal);
      host.signal.throwIfAborted();
      if (disposed) return error("session_closed", "$", "runtime was disposed during scene fetch");
      const reply = JSON.parse(runtime.dev_reload(source, control.expected_revision as string, control.revision as string, files)) as Reply;
      if (reply.ok) {
        const rendered = invoke({ op: "render" });
        if (!rendered.ok) return rendered;
        reply.observation = rendered.observation;
      }
      return reply;
    },
  };
  const api = {
    version: protocol.version,
    instance: crypto.randomUUID(),
    build: new URLSearchParams(globalThis.location?.search ?? "").get("engine_build"),
    async request(command: Command): Promise<Reply> {
      if (disposed || host.signal.aborted) return error("session_closed", "$", "the browser runtime was disposed");
      if (busy || host.isBusy?.()) return error("command_busy", "$", "one development command is already running");
      if (!command || typeof command !== "object" || Array.isArray(command) || typeof command.op !== "string") {
        return error("invalid_request", "$.command", "expected an object with an op string");
      }
      try {
        if (new TextEncoder().encode(JSON.stringify(command)).byteLength > protocol.max_request_bytes) {
          return error("request_too_large", "$.command", `command exceeds ${protocol.max_request_bytes} bytes`);
        }
      } catch { return error("invalid_request", "$.command", "command must be serializable JSON"); }
      busy = true;
      host.setBusy(true);
      try {
        // The engine checks every command, pause rules included, and runs
        // it, or returns the control the page carries out, validated with
        // its defaults.
        let reply = invoke(command);
        const control = reply.ok ? reply.result?.control as Command | undefined : undefined;
        if (control) {
          const carry = hostControls[command.op];
          if (!carry) return error("unsupported_command", "$.op", `this host cannot carry out ${command.op}`);
          reply = await carry(control);
          if (!reply.ok) return finish(reply);
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
