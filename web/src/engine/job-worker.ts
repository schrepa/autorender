// A job worker: runs in a Web Worker the page's pool starts,
// holds what the engine shares with its workers and runs the jobs it hands
// out with the engine's own code (the engine_worker module); and decodes
// images (with the engine's decoder) and KTX2 textures at load boundaries
// off the page's thread.
// Messages take effect in the order they came, so shares come before the
// jobs reading them. The module runs its most urgent job a slice at a
// time; between slices the worker reads its messages, and a job the engine
// cancelled stops there. A part of a job's output ready before the rest, as
// a tile's shape, goes back after its slice. The engine's code panicking
// here (or its module trapping) stops the engine: the worker says so with
// the panic's report and runs no more. A worker that breaks otherwise says
// so, and the pool replaces it.

import type { JobWorker } from "../../pkg/engine_worker";
import { type TranscodedImage, transcodeWith, workerTranscoder } from "./ktx2";
import { fatal, panicReport } from "./panic";

type Request =
  | { type: "init"; bindings?: string; module?: WebAssembly.Module }
  | { type: "clear" }
  | { type: "share"; key: string; blob: Blob }
  | { type: "unshare"; key: string }
  | { type: "run"; id: number; kind: string; input: Uint8Array; priority: number }
  | { type: "cancel"; id: number }
  | { type: "ktx2"; task: number; uri: string; bytes: Uint8Array; format: TranscodedImage["format"]; basis: string }
  | { type: "image"; task: number; uri: string; bytes: Uint8Array };

/** What a slice came to (`engine_worker::Report`), as the pool reads it. */
type Report = { type: "done" | "part"; id: number; output: Uint8Array } | { type: "failed"; id: number; error: string };

const scope = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<Request>) => void) | null;
};

let worker: JobWorker | undefined;
let broken = false;
let queue: Promise<void> = Promise.resolve();
let running = false;

scope.onmessage = (event) => {
  queue = queue.then(() => handle(event.data)).catch(error => fatal(error) ? stopped(error) : fail(error));
};

/** Says the module is unusable once. */
function fail(error: unknown) {
  if (broken) return;
  broken = true;
  scope.postMessage({ type: "broken", error: String(error) });
}

/** Says once that the engine's module stopped here: its panic's report,
 * or the trap. */
function stopped(error: unknown) {
  if (broken) return;
  broken = true;
  scope.postMessage({ type: "panicked", report: panicReport(error) ?? null, error: String(error) });
}

// Yields to messages between slices, without timers' clamping.
const channel = new MessageChannel();
const yielded: (() => void)[] = [];
channel.port1.onmessage = () => yielded.shift()?.();
function yieldToMessages(): Promise<void> {
  return new Promise(resolve => {
    yielded.push(resolve);
    channel.port2.postMessage(0);
  });
}

/** Runs the jobs held a slice at a time until none is left, reading
 * messages between slices. */
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

/** Starts the module, from the page's compilation of it where given,
 * and runs a small fixed tile so its hot code reaches
 * its optimized tier before real jobs. */
async function start(bindings?: string, compiled?: WebAssembly.Module) {
  const module = (bindings
    ? await import(/* @vite-ignore */ bindings)
    : await import("../../pkg/engine_worker")) as typeof import("../../pkg/engine_worker");
  await module.default(compiled ? { module_or_path: compiled } : undefined);
  worker = new module.JobWorker();
  worker.warm();
}

async function handle(request: Request) {
  switch (request.type) {
    case "init":
      await start(request.bindings, request.module);
      return;
    case "clear":
      // An earlier engine's jobs are no one's.
      worker?.clear();
      return;
    case "share": {
      const bytes = new Uint8Array(await request.blob.arrayBuffer());
      // A share the module refuses fails the jobs that read it.
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
      worker.hold(request.id, request.kind, request.input, request.priority);
      void work();
      return;
    case "cancel":
      worker?.cancel(request.id);
      return;
    case "ktx2":
      try {
        const image = transcodeWith(await workerTranscoder(request.basis), request.uri, request.bytes, request.format);
        const transfer = image.format === "rgba8" ? [image.pixels.buffer] : [image.data.buffer, image.levels.buffer];
        scope.postMessage({ type: "task", task: request.task, result: image }, transfer as ArrayBuffer[]);
      } catch (error) {
        scope.postMessage({ type: "task_failed", task: request.task, error: String(error) });
      }
      return;
    case "image":
      try {
        if (!worker) throw new Error("worker unavailable");
        const image = worker.decode_image(request.uri, request.bytes) as { width: number; height: number; pixels: Uint8Array };
        scope.postMessage({ type: "task", task: request.task, result: image }, [image.pixels.buffer as ArrayBuffer]);
      } catch (error) {
        scope.postMessage({ type: "task_failed", task: request.task, error: String(error) });
      }
      return;
  }
}
