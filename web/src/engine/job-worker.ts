import type { JobWorker } from "../../pkg/engine_worker";

type Bindings = typeof import("../../pkg/engine_worker");
import { fatal, panicReport } from "./panic";

type Request =
  | { type: "init"; bindings?: string; module?: WebAssembly.Module }
  | { type: "list"; hash: Uint8Array }
  | { type: "clear" }
  | { type: "share"; key: string; blob: Blob }
  | { type: "unshare"; key: string }
  | { type: "run"; id: number; kind: string; source: string; input: Uint8Array; priority: number }
  | { type: "cancel"; id: number }
  | { type: "image"; task: number; uri: string; bytes: Uint8Array }
  | { type: "unpack"; task: number; bytes: Uint8Array; raw: number };

type Report = { type: "done"; id: number; output: Uint8Array } | { type: "failed"; id: number; error: string };

const scope = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<Request>) => void) | null;
};

let bindings: Bindings | undefined;
let worker: JobWorker | undefined;
let broken = false;
let queue: Promise<void> = Promise.resolve();
let running = false;

scope.onmessage = (event) => {
  queue = queue.then(() => handle(event.data)).catch(error => fatal(error) ? stopped(error) : fail(error));
};

function fail(error: unknown) {
  if (broken) return;
  broken = true;
  scope.postMessage({ type: "broken", error: String(error) });
}

function stopped(error: unknown) {
  if (broken) return;
  broken = true;
  scope.postMessage({ type: "panicked", report: panicReport(error) ?? null, error: String(error) });
}

const channel = new MessageChannel();
const yielded: (() => void)[] = [];
channel.port1.onmessage = () => yielded.shift()?.();
function yieldToMessages(): Promise<void> {
  return new Promise(resolve => {
    yielded.push(resolve);
    channel.port2.postMessage(0);
  });
}

async function work() {
  if (running) return;
  running = true;
  try {
    while (worker && !broken && worker.busy()) {
      try {
        const report = worker.slice() as Report | undefined;
        if (report) scope.postMessage(report, "output" in report ? [report.output.buffer as ArrayBuffer] : []);
      } catch (error) {
        if (fatal(error)) stopped(error); else fail(error);
        return;
      }
      await yieldToMessages();
    }
  } finally {
    running = false;
  }
}

async function start(pinned?: string, compiled?: WebAssembly.Module) {
  const module = (pinned
    ? await import(/* @vite-ignore */ pinned)
    : await import("../../pkg/engine_worker")) as Bindings;
  await module.default(compiled ? { module_or_path: compiled } : undefined);
  bindings = module;
}

function loaded(): Bindings {
  if (!bindings) throw new Error("worker unavailable");
  return bindings;
}

async function handle(request: Request) {
  switch (request.type) {
    case "init":
      await start(request.bindings, request.module);
      return;
    case "list":
      worker?.free();
      worker = undefined;
      worker = new (loaded().JobWorker)(request.hash);
      return;
    case "clear":
      worker?.clear();
      return;
    case "share": {
      const bytes = new Uint8Array(await request.blob.arrayBuffer());
      try { worker?.share(request.key, bytes); } catch (error) { console.warn(`Share ${request.key} refused: ${String(error)}`); }
      return;
    }
    case "unshare":
      worker?.unshare(request.key);
      return;
    case "run":
      if (!worker || broken) {
        scope.postMessage({ type: "failed", id: request.id, error: "worker unavailable" });
        return;
      }
      try {
        worker.hold(request.id, request.kind, request.source, request.input, request.priority);
      } catch (error) {
        if (fatal(error)) throw error;
        scope.postMessage({ type: "failed", id: request.id, error: String(error) });
        return;
      }
      void work();
      return;
    case "cancel":
      worker?.cancel(request.id);
      return;
    case "unpack":
      try {
        const record = loaded().JobWorker.unpack(request.bytes, request.raw);
        scope.postMessage({ type: "task", task: request.task, result: record }, [record.buffer as ArrayBuffer]);
      } catch (error) {
        scope.postMessage({ type: "task_failed", task: request.task, error: String(error) });
      }
      return;
    case "image":
      try {
        const image = loaded().JobWorker.decode_image(request.uri, request.bytes) as { width: number; height: number; pixels: Uint8Array };
        scope.postMessage({ type: "task", task: request.task, result: image }, [image.pixels.buffer as ArrayBuffer]);
      } catch (error) {
        scope.postMessage({ type: "task_failed", task: request.task, error: String(error) });
      }
      return;
  }
}
