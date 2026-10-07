// A job worker (decision 110): runs in a Web Worker the page's pool starts,
// holds what the engine shares with its workers and runs the jobs it hands
// out with the engine's own code (the engine_worker module); and decodes
// images and KTX2 textures at load boundaries off the page's thread.
// Messages take effect in the order they came, so shares come before the
// jobs reading them. Jobs run a slice at a time (decision 112): between
// slices the worker reads its messages and runs its most urgent job, and
// a job the engine cancelled stops there. A module that traps is unusable:
// the worker then says it broke, and the pool replaces it.

import type { JobWorker } from "../pkg/engine_worker";
import { decodeImagePixels } from "./decode";
import { type TranscodedImage, transcodeWith, workerTranscoder } from "./ktx2";

type Request =
  | { type: "init"; bindings?: string }
  | { type: "clear" }
  | { type: "share"; key: string; blob: Blob }
  | { type: "unshare"; key: string }
  | { type: "run"; id: number; kind: string; input: Uint8Array; priority: number }
  | { type: "cancel"; id: number }
  | { type: "ktx2"; task: number; uri: string; bytes: Uint8Array; format: TranscodedImage["format"]; basis: string }
  | { type: "image"; task: number; uri: string; blob: Blob };

interface Job {
  id: number;
  kind: string;
  input: Uint8Array;
  priority: number;
  started: boolean;
}

/** Milliseconds a job runs between looks at messages and more urgent jobs,
 * and the work units each step takes (about one per sample). */
const SLICE_MS = 8;
const STEP_UNITS = 4096;

const scope = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<Request>) => void) | null;
};

let worker: JobWorker | undefined;
let broken = false;
let queue: Promise<void> = Promise.resolve();
const jobs: Job[] = [];
let running = false;

scope.onmessage = (event) => {
  queue = queue.then(() => handle(event.data)).catch(error => fail(error));
};

/** Says the module is unusable once. */
function fail(error: unknown) {
  if (broken) return;
  broken = true;
  scope.postMessage({ type: "broken", error: String(error) });
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

/** The most urgent job held: lowest priority, then first handed. */
function urgent(): Job | undefined {
  let best: Job | undefined;
  for (const job of jobs) {
    if (!best || job.priority < best.priority || (job.priority === best.priority && job.id < best.id)) best = job;
  }
  return best;
}

function drop(job: Job) {
  const index = jobs.indexOf(job);
  if (index >= 0) jobs.splice(index, 1);
}

/** Runs the jobs held, the most urgent a slice at a time, until none is
 * left. */
async function work() {
  if (running) return;
  running = true;
  try {
    for (let job = urgent(); job; job = urgent()) {
      if (!worker || broken) {
        for (const left of jobs.splice(0)) scope.postMessage({ type: "failed", id: left.id, error: "worker unavailable" });
        return;
      }
      try {
        if (!job.started) {
          worker.start(job.id, job.kind, job.input);
          job.started = true;
        }
        const start = performance.now();
        let output: Uint8Array | undefined;
        do output = worker.step(job.id, STEP_UNITS);
        while (output === undefined && performance.now() - start < SLICE_MS);
        if (output !== undefined) {
          drop(job);
          scope.postMessage({ type: "done", id: job.id, output }, [output.buffer as ArrayBuffer]);
        }
      } catch (error) {
        drop(job);
        scope.postMessage({ type: "failed", id: job.id, error: String(error) });
        if (error instanceof WebAssembly.RuntimeError) fail(error);
      }
      await yieldToMessages();
    }
  } finally {
    running = false;
  }
}

async function start(bindings?: string) {
  const module = (bindings
    ? await import(/* @vite-ignore */ bindings)
    : await import("../pkg/engine_worker")) as typeof import("../pkg/engine_worker");
  await module.default();
  worker = new module.JobWorker();
}

async function handle(request: Request) {
  switch (request.type) {
    case "init":
      await start(request.bindings);
      return;
    case "clear":
      // An earlier engine's jobs are no one's.
      jobs.length = 0;
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
      jobs.push({ id: request.id, kind: request.kind, input: request.input, priority: request.priority, started: false });
      void work();
      return;
    case "cancel": {
      const job = jobs.find(job => job.id === request.id);
      if (!job) return;
      drop(job);
      if (job.started) worker?.cancel(job.id);
      return;
    }
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
        const image = await decodeImagePixels(request.uri, request.blob);
        scope.postMessage({ type: "task", task: request.task, result: image }, [image.pixels.buffer as ArrayBuffer]);
      } catch (error) {
        scope.postMessage({ type: "task_failed", task: request.task, error: String(error) });
      }
      return;
  }
}
