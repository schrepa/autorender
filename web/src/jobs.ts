// Worker jobs (decision 110): a pool of Web Workers beside the page, each
// running the engine's job worker module. The engine hands the page, in
// order, what every worker should hold (terrain stacks and recipes) and jobs
// (a terrain tile, a walk, a bake); the pool hands each share to every
// worker and each job to the worker with the fewest in flight, and hands
// outputs back as workers finish them. Images and KTX2 textures decode on
// it at load boundaries too. Workers hold their own memory: shares travel
// as Blobs, which every worker reads without the page copying them again.
// ?workers=N runs N workers and ?workers=off none, so the engine does the
// work itself; a worker that breaks is replaced once, and the engine takes
// back its jobs. Jobs carry priorities (decision 112): each goes to the
// worker with the fewest in flight, among those the one whose most urgent
// job is least urgent, and workers run their most urgent job a slice at a
// time; jobs the engine cancels stop at their worker's next slice. Parts of
// outputs ready early, as a tile's shape (decision 114), go back as they
// come.

import { engineWorkerModule } from "./bindings";
import type { TranscodedImage } from "./ktx2";

/** The engine bindings a pool serves (Rust `BrowserApp`'s job methods). */
export interface JobRuntime {
  set_job_workers?(workers: number): void;
  take_jobs?(): unknown;
  complete_job?(id: number, output: Uint8Array): void;
  fail_job?(id: number, error: string): void;
  part_job?(id: number, part: Uint8Array): void;
}

/** What `take_jobs` lists, in order. */
type EngineMessage =
  | { cancel: number }
  | { clear: true }
  | { share: string; bytes: Uint8Array }
  | { unshare: string }
  | { id: number; kind: string; input: Uint8Array; priority: number };

/** What workers send back. */
type WorkerMessage =
  | { type: "done"; id: number; output: Uint8Array }
  | { type: "part"; id: number; output: Uint8Array }
  | { type: "failed"; id: number; error: string }
  | { type: "task"; task: number; result: unknown }
  | { type: "task_failed"; task: number; error: string }
  | { type: "broken"; error: string };

/** Decoded image pixels, rows from the top. */
export interface Pixels { width: number; height: number; pixels: Uint8Array }

interface Slot {
  worker: Worker;
  /** Jobs in flight, each with its priority (lower first). */
  jobs: Map<number, number>;
  tasks: Set<number>;
  broken: boolean;
}

interface Task {
  resolve(value: unknown): void;
  reject(error: Error): void;
}

export class JobPool {
  private readonly slots: Slot[] = [];
  private readonly shares = new Map<string, Blob>();
  private readonly tasks = new Map<number, Task>();
  private runtime: JobRuntime | undefined;
  private timer = 0;
  private nextTask = 1;
  private restarts = 0;

  constructor(private readonly count: number) {
    for (let index = 0; index < count; index++) this.slots.push(this.spawn());
  }

  /** Workers able to run jobs. */
  get size(): number {
    return this.slots.filter(slot => !slot.broken).length;
  }

  /** Serves `runtime`'s jobs: it hands them to the pool's workers from now
   * on, and the pool polls it for them between frames too, so jobs made
   * by development commands in paused sessions run. */
  attach(runtime: JobRuntime) {
    if (!runtime.take_jobs || !runtime.set_job_workers) return;
    this.runtime = runtime;
    runtime.set_job_workers(this.size);
    clearInterval(this.timer);
    this.timer = window.setInterval(() => this.exchange(), 50);
  }

  /** Stops serving the attached runtime. */
  detach() {
    clearInterval(this.timer);
    this.timer = 0;
    this.runtime = undefined;
  }

  /** Hands workers what the attached runtime made since the last exchange. */
  exchange() {
    const runtime = this.runtime;
    if (!runtime?.take_jobs) return;
    const messages = runtime.take_jobs() as EngineMessage[];
    for (const message of messages) {
      if ("cancel" in message) {
        const slot = this.slots.find(slot => slot.jobs.has(message.cancel));
        if (!slot) continue;
        slot.jobs.delete(message.cancel);
        slot.worker.postMessage({ type: "cancel", id: message.cancel });
      } else if ("clear" in message) {
        this.shares.clear();
        this.broadcast({ type: "clear" });
      } else if ("share" in message) {
        const blob = new Blob([message.bytes as Uint8Array<ArrayBuffer>]);
        this.shares.set(message.share, blob);
        this.broadcast({ type: "share", key: message.share, blob });
      } else if ("unshare" in message) {
        this.shares.delete(message.unshare);
        this.broadcast({ type: "unshare", key: message.unshare });
      } else {
        const slot = this.least();
        if (!slot) {
          runtime.fail_job?.(message.id, "no workers");
          continue;
        }
        slot.jobs.set(message.id, message.priority);
        slot.worker.postMessage({ type: "run", id: message.id, kind: message.kind, input: message.input, priority: message.priority }, [message.input.buffer as ArrayBuffer]);
      }
    }
  }

  /** KTX2 image `uri` transcoded to `format` on a worker (decision 037). */
  transcodeKtx2(uri: string, bytes: Uint8Array, format: TranscodedImage["format"], basis: string): Promise<TranscodedImage> {
    return this.task({ type: "ktx2", uri, bytes, format, basis }, [bytes.buffer as ArrayBuffer]) as Promise<TranscodedImage>;
  }

  /** Image `uri`'s RGBA8 pixels, decoded on a worker. */
  decodeImage(uri: string, blob: Blob): Promise<Pixels> {
    return this.task({ type: "image", uri, blob }, []) as Promise<Pixels>;
  }

  private task(message: Record<string, unknown>, transfer: Transferable[]): Promise<unknown> {
    const slot = this.least();
    if (!slot) return Promise.reject(new Error("no workers"));
    const task = this.nextTask++;
    slot.tasks.add(task);
    return new Promise((resolve, reject) => {
      this.tasks.set(task, { resolve, reject });
      slot.worker.postMessage({ ...message, task }, transfer);
    });
  }

  private spawn(): Slot {
    const worker = new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module", name: "engine-jobs" });
    const slot: Slot = { worker, jobs: new Map(), tasks: new Set(), broken: false };
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => this.receive(slot, event.data);
    worker.onerror = (event) => { event.preventDefault(); this.lose(slot, event.message || "worker error"); };
    worker.postMessage({ type: "init", bindings: engineWorkerModule() });
    for (const [key, blob] of this.shares) worker.postMessage({ type: "share", key, blob });
    return slot;
  }

  private broadcast(message: Record<string, unknown>) {
    for (const slot of this.slots) if (!slot.broken) slot.worker.postMessage(message);
  }

  /** The working slot with the fewest jobs and tasks in flight; among
   * those, the one whose most urgent job is least urgent, where a new job
   * would wait least (decision 112). */
  private least(): Slot | undefined {
    let best: Slot | undefined;
    let bestLoad = Infinity;
    let bestUrgent = -Infinity;
    for (const slot of this.slots) {
      if (slot.broken) continue;
      const load = slot.jobs.size + slot.tasks.size;
      let urgent = Infinity;
      for (const priority of slot.jobs.values()) urgent = Math.min(urgent, priority);
      if (load < bestLoad || (load === bestLoad && urgent > bestUrgent)) {
        best = slot;
        bestLoad = load;
        bestUrgent = urgent;
      }
    }
    return best;
  }

  private receive(slot: Slot, message: WorkerMessage) {
    switch (message.type) {
      // Jobs cancelled since are no one's.
      case "done":
        if (slot.jobs.delete(message.id)) this.runtime?.complete_job?.(message.id, message.output);
        break;
      case "part":
        if (slot.jobs.has(message.id)) this.runtime?.part_job?.(message.id, message.output);
        break;
      case "failed":
        if (slot.jobs.delete(message.id)) this.runtime?.fail_job?.(message.id, message.error);
        break;
      case "task":
        slot.tasks.delete(message.task);
        this.tasks.get(message.task)?.resolve(message.result);
        this.tasks.delete(message.task);
        break;
      case "task_failed":
        slot.tasks.delete(message.task);
        this.tasks.get(message.task)?.reject(new Error(message.error));
        this.tasks.delete(message.task);
        break;
      case "broken":
        this.lose(slot, message.error);
        break;
    }
  }

  /** Retires a broken worker: its jobs fail back to the engine, its tasks
   * fail, and a new worker holding every share takes its place, while
   * replacements last. */
  private lose(slot: Slot, error: string) {
    if (slot.broken) return;
    slot.broken = true;
    slot.worker.terminate();
    console.warn(`An engine worker stopped: ${error}`);
    for (const id of slot.jobs.keys()) this.runtime?.fail_job?.(id, error);
    slot.jobs.clear();
    for (const task of slot.tasks) {
      this.tasks.get(task)?.reject(new Error(error));
      this.tasks.delete(task);
    }
    slot.tasks.clear();
    if (this.restarts < this.count) {
      this.restarts++;
      this.slots[this.slots.indexOf(slot)] = this.spawn();
    }
    this.runtime?.set_job_workers?.(this.size);
  }
}

let pool: JobPool | null | undefined;

/** The page's worker pool, started on first use; none with ?workers=off or
 * without Web Workers. By default half the logical processors, at most 8,
 * work beside the page. */
export function jobPool(): JobPool | undefined {
  if (pool === undefined) {
    const requested = new URLSearchParams(location.search).get("workers");
    const cores = navigator.hardwareConcurrency || 4;
    const count = requested === "off" ? 0
      : requested && /^\d{1,2}$/.test(requested) ? Math.min(Number(requested), 32)
      : Math.max(1, Math.min(8, Math.floor(cores / 2)));
    pool = count > 0 && typeof Worker !== "undefined" ? new JobPool(count) : null;
  }
  return pool ?? undefined;
}
