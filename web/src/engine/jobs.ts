// Worker jobs: a pool of Web Workers beside the page, each running the
// engine's job worker module. The engine schedules its jobs (which worker
// runs each, what every worker holds, which outputs the job cache answers
// and which a fresh run verifies); the pool carries its orders to the
// workers and the cache (job-cache.ts), and hands back what workers report
// and the cache reads. Workers run their most urgent job a slice at a
// time, in the module. Images and KTX2 textures decode on them at load
// boundaries too. Workers hold their own memory: shares travel as Blobs,
// which every worker reads without the page copying them again.
// ?workers=N runs N workers and ?workers=off none, so the engine does the
// work itself; a worker that breaks is replaced once, and the engine takes
// back its jobs. A worker's panic, as the game module's, stops the engine
// for good: the pool's workers end and it calls the module no more.

import workerWasmUrl from "../../pkg/engine_worker_bg.wasm?url";
import { CORES_PER_DEFAULT_WORKER, MAX_DEFAULT_WORKERS, MAX_WORKERS, PANIC_PREFIX } from "../generated/host";
import type { Runtime } from "../generated/runtime";
import { engineWorkerModule } from "./bindings";
import { JobStore, jobCacheSettings } from "./job-cache";
import type { TranscodedImage } from "./ktx2";
import { fatal, type PanicReport } from "./panic";

/** The game methods a pool serves. */
export type JobRuntime = Pick<Runtime, "set_job_workers" | "set_job_cache" | "take_jobs" | "complete_job" | "fail_job"
  | "part_job" | "job_kept" | "lose_job_worker" | "job_task" | "job_task_done" | "report_job_cache">;

/** What the engine orders, in order (`engine::host::JobOrder`). */
export type JobOrder =
  | { clear: true }
  | { share: string; bytes: Uint8Array }
  | { unshare: string }
  | { run: number; worker: number; kind: string; input: Uint8Array; priority: number }
  | { cancel: number; worker: number }
  | { lookup: number; kind: string; identity: Uint8Array }
  | { keep: string; output: Uint8Array }
  | { mismatch: string };

/** What workers send back. */
type WorkerMessage =
  | { type: "done"; id: number; output: Uint8Array }
  | { type: "part"; id: number; output: Uint8Array }
  | { type: "failed"; id: number; error: string }
  | { type: "task"; task: number; result: unknown }
  | { type: "task_failed"; task: number; error: string }
  | { type: "broken"; error: string }
  | { type: "panicked"; report: PanicReport | null; error: string };

/** Decoded image pixels, rows from the top. */
export interface Pixels { width: number; height: number; pixels: Uint8Array }

interface Slot {
  worker: Worker;
  /** The page's tasks on it, each with whether the engine counts it. */
  tasks: Map<number, boolean>;
  broken: boolean;
  /** Messages held until the worker has its module. */
  held?: [unknown, Transferable[]][];
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
  /** What the attached runtime's host does once a module stopped. */
  private failed: ((error: unknown) => void) | undefined;
  /** The cache is open (or off) and the runtime knows: jobs flow. */
  private ready = false;
  private timer = 0;
  private nextTask = 1;
  private restarts = 0;
  private reported = "";
  /** The job worker module, compiled once for every worker;
   * `undefined` where it did not compile, and each compiles its own. */
  private readonly module: Promise<WebAssembly.Module | undefined> = workerWasm()
    .then(bytes => bytes && WebAssembly.compile(bytes))
    .catch(() => undefined);
  /** The output cache, once open; `undefined` when off. */
  private readonly store: Promise<JobStore | undefined> = this.module.then(openStore);

  constructor(private readonly count: number) {
    for (let index = 0; index < count; index++) this.slots.push(this.spawn());
  }

  /** Workers able to run jobs. */
  get size(): number {
    return this.slots.filter(slot => !slot.broken).length;
  }

  /** Serves `runtime`'s jobs: it hands them to the pool's workers from now
   * on, once the cache opened, and the pool polls it for them between
   * frames too, so jobs made by development commands in paused sessions
   * run. Once the runtime's module or a worker's stops, the pool ends and
   * calls `failed`. */
  attach(runtime: JobRuntime, failed: (error: unknown) => void) {
    this.runtime = runtime;
    this.failed = failed;
    this.ready = false;
    this.call(runtime => runtime.set_job_workers(this.size));
    void this.store.then(store => {
      if (this.runtime !== runtime) return;
      this.call(runtime => runtime.set_job_cache(store !== undefined, store?.verify ?? false));
      this.ready = true;
      this.exchange();
    });
    clearInterval(this.timer);
    this.timer = window.setInterval(() => this.exchange(), 50);
  }

  /** Stops serving the attached runtime. */
  detach() {
    clearInterval(this.timer);
    this.timer = 0;
    this.runtime = undefined;
    this.failed = undefined;
    this.ready = false;
  }

  /** Ends the pool for good: it serves no runtime, its workers end and its
   * tasks fail; the page's next pool starts afresh. */
  terminate() {
    this.detach();
    for (const slot of this.slots) {
      slot.broken = true;
      slot.worker.terminate();
    }
    for (const task of this.tasks.values()) task.reject(new Error("the engine stopped"));
    this.tasks.clear();
    if (pool === this) pool = undefined;
  }

  /** Runs `use` on the attached runtime; a module that stopped ends the pool
   * and tells the host. */
  private call<T>(use: (runtime: JobRuntime) => T): T | undefined {
    const runtime = this.runtime;
    if (!runtime) return undefined;
    try {
      return use(runtime);
    } catch (error) {
      if (!fatal(error)) throw error;
      this.stop(error);
      return undefined;
    }
  }

  /** Ends the pool after `error` stopped a module, and tells the host. */
  private stop(error: unknown) {
    const failed = this.failed;
    this.terminate();
    failed?.(error);
  }

  /** Carries out what the attached runtime ordered since the last exchange. */
  exchange() {
    if (!this.ready) return;
    const orders = this.call(runtime => runtime.take_jobs()) ?? [];
    for (const order of orders) this.carry(order);
    void this.report();
  }

  /** Carries `order` to the workers or the cache. */
  private carry(order: JobOrder) {
    if ("clear" in order) {
      // A new engine's first order: workers drop the earlier engine's jobs.
      this.shares.clear();
      this.broadcast({ type: "clear" });
    } else if ("share" in order) {
      const blob = new Blob([order.bytes as Uint8Array<ArrayBuffer>]);
      this.shares.set(order.share, blob);
      this.broadcast({ type: "share", key: order.share, blob });
    } else if ("unshare" in order) {
      this.shares.delete(order.unshare);
      this.broadcast({ type: "unshare", key: order.unshare });
    } else if ("run" in order) {
      this.post(this.slots[order.worker], { type: "run", id: order.run, kind: order.kind, input: order.input, priority: order.priority },
        [order.input.buffer as ArrayBuffer]);
    } else if ("cancel" in order) {
      this.post(this.slots[order.worker], { type: "cancel", id: order.cancel });
    } else if ("lookup" in order) {
      void this.lookup(order.lookup, order.kind, order.identity);
    } else if ("keep" in order) {
      void this.store.then(store => store?.keep(order.keep, order.output));
    } else {
      void this.store.then(store => store?.mismatch(order.mismatch));
    }
  }

  /** Hands the runtime what the cache keeps of job `id`'s work, if
   * anything, and carries out what that orders. */
  private async lookup(id: number, kind: string, identity: Uint8Array) {
    const runtime = this.runtime;
    const store = await this.store;
    const [key, kept] = store ? await store.lookup(kind, identity) : ["", undefined];
    if (this.runtime !== runtime) return;
    this.call(runtime => runtime.job_kept(id, key, kept));
    this.exchange();
  }

  /** Tells the runtime the cache's counters when they change. */
  private async report() {
    const runtime = this.runtime;
    const store = await this.store;
    if (!store || this.runtime !== runtime) return;
    const [hits, misses, bytes, mismatches] = store.index.counters();
    const text = `${hits} ${misses} ${bytes} ${mismatches}`;
    if (text === this.reported) return;
    this.reported = text;
    this.call(runtime => runtime.report_job_cache(hits, misses, bytes, mismatches));
  }

  /** KTX2 image `uri` transcoded to `format` on a worker,
   * or kept from an earlier transcoding by the same build. */
  async transcodeKtx2(uri: string, bytes: Uint8Array, format: TranscodedImage["format"], basis: string): Promise<TranscodedImage> {
    const store = await this.store;
    const [key, kept] = store ? await store.lookup(`ktx2 ${format} ${basis}`, bytes) : [undefined, undefined];
    const image = kept && readTranscoded(kept);
    if (image && !store?.verify) return image;
    const fresh = await this.task({ type: "ktx2", uri, bytes, format, basis }, [bytes.buffer as ArrayBuffer]) as TranscodedImage;
    if (store && key) {
      const written = writeTranscoded(fresh);
      if (!kept) store.keep(key, written);
      else if (!sameBytes(kept, written)) store.mismatch(key);
    }
    return fresh;
  }

  /** Image `uri`'s RGBA8 pixels, decoded on a worker. */
  decodeImage(uri: string, blob: Blob): Promise<Pixels> {
    return this.task({ type: "image", uri, blob }, []) as Promise<Pixels>;
  }

  /** Runs the page's own task on the worker the engine says it waits least
   * on, or, before an engine, the one with the fewest tasks. */
  private task(message: Record<string, unknown>, transfer: Transferable[]): Promise<unknown> {
    const counted = this.call(runtime => runtime.job_task());
    const slot = counted !== undefined ? this.slots[counted] : this.fewestTasks();
    if (!slot || slot.broken) return Promise.reject(new Error("no workers"));
    const task = this.nextTask++;
    slot.tasks.set(task, counted !== undefined);
    return new Promise((resolve, reject) => {
      this.tasks.set(task, { resolve, reject });
      this.post(slot, { ...message, task }, transfer);
    });
  }

  private fewestTasks(): Slot | undefined {
    let best: Slot | undefined;
    for (const slot of this.slots) if (!slot.broken && (!best || slot.tasks.size < best.tasks.size)) best = slot;
    return best;
  }

  private spawn(): Slot {
    const worker = new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module", name: "engine-jobs" });
    const slot: Slot = { worker, tasks: new Map(), broken: false, held: [] };
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => this.receive(slot, event.data);
    worker.onerror = (event) => { event.preventDefault(); this.lose(slot, event.message || "worker error"); };
    // Workers start from the module compiled once; what is
    // posted meanwhile follows in order.
    void this.module.then(module => {
      worker.postMessage({ type: "init", bindings: engineWorkerModule(), module });
      for (const [message, transfer] of slot.held ?? []) worker.postMessage(message, transfer);
      slot.held = undefined;
    });
    for (const [key, blob] of this.shares) this.post(slot, { type: "share", key, blob });
    return slot;
  }

  /** Posts `message` to `slot`'s worker, or holds it until the worker has
   * its module. */
  private post(slot: Slot, message: unknown, transfer: Transferable[] = []) {
    if (slot.held) slot.held.push([message, transfer]);
    else slot.worker.postMessage(message, transfer);
  }

  private broadcast(message: Record<string, unknown>) {
    for (const slot of this.slots) if (!slot.broken) this.post(slot, message);
  }

  private receive(slot: Slot, message: WorkerMessage) {
    switch (message.type) {
      case "done":
        this.call(runtime => runtime.complete_job(message.id, message.output));
        break;
      case "part":
        this.call(runtime => runtime.part_job(message.id, message.output));
        break;
      case "failed":
        this.call(runtime => runtime.fail_job(message.id, message.error));
        break;
      case "task":
      case "task_failed": {
        const index = this.slots.indexOf(slot);
        if (slot.tasks.get(message.task)) this.call(runtime => runtime.job_task_done(index));
        slot.tasks.delete(message.task);
        const task = this.tasks.get(message.task);
        this.tasks.delete(message.task);
        if (message.type === "task") task?.resolve(message.result);
        else task?.reject(new Error(message.error));
        break;
      }
      case "broken":
        this.lose(slot, message.error);
        break;
      case "panicked":
        // The engine's code panicked on a worker: the engine stops, as when
        // the game's module panics.
        this.stop(message.report ? new Error(`${PANIC_PREFIX}${JSON.stringify(message.report)}`)
          : new WebAssembly.RuntimeError(message.error));
        break;
    }
  }

  /** Retires a broken worker: the engine takes back its jobs, its tasks
   * fail, and a new worker holding every share takes its place, while
   * replacements last. */
  private lose(slot: Slot, error: string) {
    if (slot.broken) return;
    slot.broken = true;
    slot.worker.terminate();
    console.warn(`An engine worker stopped: ${error}`);
    for (const task of slot.tasks.keys()) {
      this.tasks.get(task)?.reject(new Error(error));
      this.tasks.delete(task);
    }
    const index = this.slots.indexOf(slot);
    const replaced = this.restarts < this.count;
    if (replaced) {
      this.restarts++;
      this.slots[index] = this.spawn();
    }
    this.call(runtime => runtime.lose_job_worker(index, error, replaced));
  }
}

let wasm: Promise<ArrayBuffer | undefined> | undefined;

/** The job worker module's bytes, fetched once by a page with workers:
 * compiled once for every worker and hashed for the job
 * cache's version; `undefined` where the fetch failed. */
function workerWasm(): Promise<ArrayBuffer | undefined> {
  wasm ??= fetchWorkerWasm();
  return wasm;
}

async function fetchWorkerWasm(): Promise<ArrayBuffer | undefined> {
  try {
    const pinned = engineWorkerModule();
    const url = pinned ? pinned.replace(/\.js$/, "_bg.wasm") : workerWasmUrl;
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: ${response.status}`);
    return await response.arrayBuffer();
  } catch (error) {
    console.warn(`The job worker module was not fetched: ${String(error)}`);
    return undefined;
  }
}

/** Opens the page's job output cache, its index in the job worker module
 * (`module`, compiled), unless the page turned it off. */
async function openStore(module: WebAssembly.Module | undefined): Promise<JobStore | undefined> {
  const settings = jobCacheSettings();
  const bytes = await workerWasm();
  if (!settings || !module || !bytes) return undefined;
  const pinned = engineWorkerModule();
  const bindings = (pinned ? await import(/* @vite-ignore */ pinned) : await import("../../pkg/engine_worker")) as typeof import("../../pkg/engine_worker");
  await bindings.default({ module_or_path: module });
  return JobStore.open(bytes, bindings, settings.budget, settings.verify);
}

function sameBytes(a: Uint8Array, b: Uint8Array): boolean {
  if (a.length !== b.length) return false;
  for (let index = 0; index < a.length; index++) if (a[index] !== b[index]) return false;
  return true;
}

/** A transcoded image as the cache keeps it: its header's length, its
 * header as JSON and its texels. */
function writeTranscoded(image: TranscodedImage): Uint8Array {
  const header = image.format === "rgba8"
    ? { format: image.format, width: image.width, height: image.height }
    : { format: image.format, width: image.width, height: image.height, levels: Array.from(image.levels) };
  const head = new TextEncoder().encode(JSON.stringify(header));
  const body = image.format === "rgba8" ? image.pixels : image.data;
  const out = new Uint8Array(4 + head.length + body.length);
  new DataView(out.buffer).setUint32(0, head.length, true);
  out.set(head, 4);
  out.set(body, 4 + head.length);
  return out;
}

function readTranscoded(bytes: Uint8Array): TranscodedImage | undefined {
  try {
    const length = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength).getUint32(0, true);
    const header = JSON.parse(new TextDecoder().decode(bytes.subarray(4, 4 + length))) as {
      format: TranscodedImage["format"]; width: number; height: number; levels?: number[];
    };
    const body = bytes.slice(4 + length);
    if (!Number.isSafeInteger(header.width) || !Number.isSafeInteger(header.height)) return undefined;
    if (header.format === "rgba8") return { format: "rgba8", width: header.width, height: header.height, pixels: body };
    if (!["bc7", "etc2", "astc"].includes(header.format) || !Array.isArray(header.levels)) return undefined;
    return { format: header.format, width: header.width, height: header.height, data: body, levels: Uint32Array.from(header.levels) };
  } catch {
    return undefined;
  }
}

let pool: JobPool | null | undefined;

/** The page's worker pool, started on first use; none with ?workers=off or
 * without Web Workers. A ?workers or ?jobcache the native host would refuse
 * fails the page. */
export function jobPool(): JobPool | undefined {
  if (pool === undefined) {
    const count = workerCount(new URLSearchParams(location.search).get("workers"));
    jobCacheSettings();
    pool = count > 0 && typeof Worker !== "undefined" ? new JobPool(count) : null;
  }
  return pool ?? undefined;
}

/** The job workers `requested` (the page's ?workers) runs: none for `off`,
 * 0 to MAX_WORKERS as given, and by default one per CORES_PER_DEFAULT_WORKER
 * logical processors, at most MAX_DEFAULT_WORKERS. */
function workerCount(requested: string | null): number {
  if (requested === null) {
    return Math.max(1, Math.min(MAX_DEFAULT_WORKERS, Math.floor((navigator.hardwareConcurrency || 4) / CORES_PER_DEFAULT_WORKER)));
  }
  if (requested === "off") return 0;
  if (/^\d+$/.test(requested) && Number(requested) <= MAX_WORKERS) return Number(requested);
  throw new Error(`?workers is off or 0 to ${MAX_WORKERS}, not ${requested}.`);
}
