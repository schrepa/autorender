// Worker jobs: a pool of Web Workers beside the page, each
// running the engine's job worker module. The engine hands the page, in
// order, what every worker should hold (terrain stacks and recipes) and jobs
// (a terrain tile, a walk, a bake); the pool hands each share to every
// worker and each job to the worker with the fewest in flight, and hands
// outputs back as workers finish them. Images and KTX2 textures decode on
// it at load boundaries too. Workers hold their own memory: shares travel
// as Blobs, which every worker reads without the page copying them again.
// ?workers=N runs N workers and ?workers=off none, so the engine does the
// work itself; a worker that breaks is replaced once, and the engine takes
// back its jobs. A worker's panic, as the game module's, stops the engine
// for good: the pool's workers end and it calls the module no more. Jobs
// carry priorities: each goes to the
// worker with the fewest in flight, among those the one whose most urgent
// job is least urgent, and workers run their most urgent job a slice at a
// time; jobs the engine cancels stop at their worker's next slice. Parts of
// outputs ready early, as a tile's shape, go back as they
// come. Outputs of jobs the engine marks keepable, and KTX2 textures, come
// from the page's job cache where the same build made them before
// (job-cache.ts).

import workerWasmUrl from "../../pkg/engine_worker_bg.wasm?url";
import { PANIC_PREFIX } from "../generated/host";
import type { Runtime } from "../generated/runtime";
import { engineWorkerModule } from "./bindings";
import { JobCache, jobCacheSettings } from "./job-cache";
import type { TranscodedImage } from "./ktx2";
import { fatal, type PanicReport } from "./panic";

/** The game methods a pool serves. */
export type JobRuntime = Pick<Runtime, "set_job_workers" | "take_jobs" | "complete_job" | "fail_job" | "part_job" | "report_job_cache">;

/** A job `take_jobs` lists: its identity where its output may be kept. */
interface JobRequest { id: number; kind: string; input: Uint8Array; priority: number; identity?: Uint8Array }

/** What `take_jobs` lists, in order. */
export type JobMessage =
  | { cancel: number }
  | { clear: true }
  | { share: string; bytes: Uint8Array }
  | { unshare: string }
  | JobRequest;

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
  /** Jobs in flight, each with its priority (lower first). */
  jobs: Map<number, number>;
  tasks: Set<number>;
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
  private timer = 0;
  private nextTask = 1;
  private restarts = 0;
  /** The output cache, once open; the keys of jobs whose
   * outputs it keeps when done; jobs looking it up and those cancelled
   * meanwhile; outputs kept that fresh runs verify; and the counters last
   * reported. */
  private readonly cache: Promise<JobCache | undefined> = workerWasm().then(openCache);
  /** The job worker module, compiled once for every worker;
   * `undefined` where it did not compile, and each compiles its
   * own. */
  private readonly module: Promise<WebAssembly.Module | undefined> = workerWasm()
    .then(bytes => bytes && WebAssembly.compile(bytes))
    .catch(() => undefined);
  private readonly keys = new Map<number, string>();
  private readonly looking = new Set<number>();
  private readonly dropped = new Set<number>();
  private readonly verifying = new Map<number, Uint8Array>();
  private reported = "";

  constructor(private readonly count: number) {
    for (let index = 0; index < count; index++) this.slots.push(this.spawn());
  }

  /** Workers able to run jobs. */
  get size(): number {
    return this.slots.filter(slot => !slot.broken).length;
  }

  /** Serves `runtime`'s jobs: it hands them to the pool's workers from now
   * on, and the pool polls it for them between frames too, so jobs made
   * by development commands in paused sessions run. Once the runtime's
   * module or a worker's stops, the pool ends and calls `failed`. */
  attach(runtime: JobRuntime, failed: (error: unknown) => void) {
    this.runtime = runtime;
    this.failed = failed;
    this.call(runtime => runtime.set_job_workers(this.size));
    clearInterval(this.timer);
    this.timer = window.setInterval(() => this.exchange(), 50);
  }

  /** Stops serving the attached runtime. */
  detach() {
    clearInterval(this.timer);
    this.timer = 0;
    this.runtime = undefined;
    this.failed = undefined;
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
  private call(use: (runtime: JobRuntime) => void) {
    const runtime = this.runtime;
    if (!runtime) return;
    try {
      use(runtime);
    } catch (error) {
      if (!fatal(error)) throw error;
      this.stop(error);
    }
  }

  /** Ends the pool after `error` stopped a module, and tells the host. */
  private stop(error: unknown) {
    const failed = this.failed;
    this.terminate();
    failed?.(error);
  }

  /** Hands workers what the attached runtime made since the last exchange. */
  exchange() {
    this.call(runtime => this.hand(runtime, runtime.take_jobs()));
    this.report();
  }

  /** Hands workers `messages`, which `runtime` made. */
  private hand(runtime: JobRuntime, messages: JobMessage[]) {
    for (const message of messages) {
      if ("cancel" in message) {
        if (this.looking.has(message.cancel)) {
          this.dropped.add(message.cancel);
          continue;
        }
        this.keys.delete(message.cancel);
        const slot = this.slots.find(slot => slot.jobs.has(message.cancel));
        if (!slot) continue;
        slot.jobs.delete(message.cancel);
        this.post(slot, { type: "cancel", id: message.cancel });
      } else if ("clear" in message) {
        // A new engine's first message: workers drop the earlier engine's
        // jobs, and so do their slots, keys and lookups here.
        for (const slot of this.slots) slot.jobs.clear();
        this.keys.clear();
        this.verifying.clear();
        for (const id of this.looking) this.dropped.add(id);
        this.shares.clear();
        this.broadcast({ type: "clear" });
      } else if ("share" in message) {
        const blob = new Blob([message.bytes as Uint8Array<ArrayBuffer>]);
        this.shares.set(message.share, blob);
        this.broadcast({ type: "share", key: message.share, blob });
      } else if ("unshare" in message) {
        this.shares.delete(message.unshare);
        this.broadcast({ type: "unshare", key: message.unshare });
      } else if (message.identity) {
        void this.cached(runtime, message, message.identity);
      } else {
        this.dispatch(runtime, message);
      }
    }
  }

  /** Hands `message`'s job to the worker with the fewest in flight. */
  private dispatch(runtime: JobRuntime, message: JobRequest) {
    const slot = this.least();
    if (!slot) {
      runtime.fail_job(message.id, "no workers");
      return;
    }
    slot.jobs.set(message.id, message.priority);
    this.post(slot, { type: "run", id: message.id, kind: message.kind, input: message.input, priority: message.priority }, [message.input.buffer as ArrayBuffer]);
  }

  /** Answers a keepable job from the cache, or hands it to
   * a worker and keeps its output once done; verifying, runs it anyway
   * and compares. */
  private async cached(runtime: JobRuntime, message: JobRequest, identity: Uint8Array) {
    this.looking.add(message.id);
    try {
      const cache = await this.cache;
      const key = cache && await cache.key(message.kind, identity);
      const kept = cache && key ? await cache.lookup(key) : undefined;
      if (this.dropped.has(message.id) || this.runtime !== runtime) return;
      if (kept) {
        this.call(runtime => runtime.complete_job(message.id, kept));
        if (!cache?.verify || this.runtime !== runtime) return;
        this.verifying.set(message.id, kept);
      }
      if (key) this.keys.set(message.id, key);
      this.call(runtime => this.dispatch(runtime, message));
    } catch (error) {
      this.call(runtime => runtime.fail_job(message.id, String(error)));
    } finally {
      this.looking.delete(message.id);
      this.dropped.delete(message.id);
    }
  }

  /** Tells the runtime the cache's counters when they change. */
  private report() {
    const runtime = this.runtime;
    if (!runtime) return;
    void this.cache.then(cache => {
      if (!cache) return;
      const { hits, misses, bytes, mismatches } = cache.stats();
      const text = `${hits} ${misses} ${bytes} ${mismatches}`;
      if (text === this.reported || this.runtime !== runtime) return;
      this.reported = text;
      this.call(runtime => runtime.report_job_cache(hits, misses, bytes, mismatches));
    });
  }

  /** KTX2 image `uri` transcoded to `format` on a worker,
   * or kept from an earlier transcoding by the same build. */
  async transcodeKtx2(uri: string, bytes: Uint8Array, format: TranscodedImage["format"], basis: string): Promise<TranscodedImage> {
    const cache = await this.cache;
    const key = cache && await cache.key(`ktx2 ${format} ${basis}`, bytes);
    const kept = cache && key ? await cache.lookup(key) : undefined;
    const image = kept && readTranscoded(kept);
    if (image && !cache?.verify) return image;
    const fresh = await this.task({ type: "ktx2", uri, bytes, format, basis }, [bytes.buffer as ArrayBuffer]) as TranscodedImage;
    if (cache && key) {
      const written = writeTranscoded(fresh);
      if (!kept) cache.store(key, written);
      else if (!sameBytes(kept, written)) cache.mismatch(key);
    }
    return fresh;
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
      this.post(slot, { ...message, task }, transfer);
    });
  }

  private spawn(): Slot {
    const worker = new Worker(new URL("./job-worker.ts", import.meta.url), { type: "module", name: "engine-jobs" });
    const slot: Slot = { worker, jobs: new Map(), tasks: new Set(), broken: false, held: [] };
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

  /** The working slot with the fewest jobs and tasks in flight; among
   * those, the one whose most urgent job is least urgent, where a new job
   * would wait least. */
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
      case "done": {
        if (!slot.jobs.delete(message.id)) break;
        const key = this.keys.get(message.id);
        this.keys.delete(message.id);
        const kept = this.verifying.get(message.id);
        this.verifying.delete(message.id);
        // A kept output the engine already took is compared, not taken.
        if (kept) {
          if (key && !sameBytes(kept, message.output)) void this.cache.then(cache => cache?.mismatch(key));
          break;
        }
        this.call(runtime => runtime.complete_job(message.id, message.output));
        if (key) void this.cache.then(cache => cache?.store(key, message.output));
        break;
      }
      case "part":
        if (slot.jobs.has(message.id)) this.call(runtime => runtime.part_job(message.id, message.output));
        break;
      case "failed":
        this.keys.delete(message.id);
        if (this.verifying.delete(message.id)) {
          slot.jobs.delete(message.id);
          break;
        }
        if (slot.jobs.delete(message.id)) this.call(runtime => runtime.fail_job(message.id, message.error));
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
      case "panicked":
        // The engine's code panicked on a worker: the engine stops, as when
        // the game's module panics.
        this.stop(message.report ? new Error(`${PANIC_PREFIX}${JSON.stringify(message.report)}`)
          : new WebAssembly.RuntimeError(message.error));
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
    for (const id of slot.jobs.keys()) {
      this.keys.delete(id);
      if (!this.verifying.delete(id)) this.call(runtime => runtime.fail_job(id, error));
    }
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
    this.call(runtime => runtime.set_job_workers(this.size));
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

/** Opens the page's job output cache for the job worker
 * module `module` (its bytes), unless the page turned it off. */
async function openCache(module: ArrayBuffer | undefined): Promise<JobCache | undefined> {
  const settings = jobCacheSettings();
  if (!settings || !module) return undefined;
  return JobCache.open(module, settings.budget, settings.verify);
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
