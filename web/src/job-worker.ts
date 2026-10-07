// A job worker (decision 110): runs in a Web Worker the page's pool starts,
// holds what the engine shares with its workers and runs the jobs it hands
// out with the engine's own code (the engine_worker module), one message at
// a time in the order they came; and decodes images and KTX2 textures at
// load boundaries off the page's thread. A module that traps is unusable:
// the worker then says it broke, and the pool replaces it.

import type { JobWorker } from "../pkg/engine_worker";
import { decodeImagePixels } from "./decode";
import { type TranscodedImage, transcodeWith, workerTranscoder } from "./ktx2";

type Request =
  | { type: "init"; bindings?: string }
  | { type: "clear" }
  | { type: "share"; key: string; blob: Blob }
  | { type: "unshare"; key: string }
  | { type: "run"; id: number; kind: string; input: Uint8Array }
  | { type: "ktx2"; task: number; uri: string; bytes: Uint8Array; format: TranscodedImage["format"]; basis: string }
  | { type: "image"; task: number; uri: string; blob: Blob };

const scope = self as unknown as {
  postMessage(message: unknown, transfer?: Transferable[]): void;
  onmessage: ((event: MessageEvent<Request>) => void) | null;
};

let worker: JobWorker | undefined;
let broken = false;
let queue: Promise<void> = Promise.resolve();

scope.onmessage = (event) => {
  queue = queue.then(() => handle(event.data)).catch(error => fail(error));
};

/** Says the module is unusable once. */
function fail(error: unknown) {
  if (broken) return;
  broken = true;
  scope.postMessage({ type: "broken", error: String(error) });
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
    case "run": {
      if (!worker || broken) {
        scope.postMessage({ type: "failed", id: request.id, error: "worker unavailable" });
        return;
      }
      try {
        const output = worker.run(request.kind, request.input);
        scope.postMessage({ type: "done", id: request.id, output }, [output.buffer as ArrayBuffer]);
      } catch (error) {
        scope.postMessage({ type: "failed", id: request.id, error: String(error) });
        if (error instanceof WebAssembly.RuntimeError) fail(error);
      }
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
