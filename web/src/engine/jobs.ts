import { engineFailure } from "../generated/errors";
import { FRAME_CORES, PANIC_PREFIX } from "../generated/host";
import type { Runtime } from "../generated/runtime";
import { type Coder, type IndexBindings, JobStore, jobCacheSettings } from "./job-cache";
import { fatal, type PanicReport } from "./panic";

export type JobRuntime = Pick<Runtime, "set_job_workers" | "set_job_cache" | "take_jobs" | "complete_job" | "fail_job"
  | "job_kept" | "lose_job_worker" | "job_task" | "job_task_done" | "report_job_cache" | "job_hash">;

export type JobOrder =
  | { clear: true }
  | { share: string; bytes: Uint8Array }
  | { unshare: string }
  | { run: number; worker: number; kind: string; source: string; input: Uint8Array; priority: number }
  | { cancel: number; worker: number }
  | { lookup: number; key: string }
  | { keep: string; output: Uint8Array }
  | { mismatch: string };

type Lookup = Extract<JobOrder, { lookup: number }>;

type WorkerMessage =
  | { type: "done"; id: number; output: Uint8Array }
  | { type: "failed"; id: number; error: string }
  | { type: "wants"; id: number; keys: string[] }
  | { type: "task"; task: number; result: unknown }
  | { type: "task_failed"; task: number; error: string }
  | { type: "broken"; error: string }
  | { type: "panicked"; report: PanicReport | null; error: string };

export interface Pixels { width: number; height: number; pixels: Uint8Array }

export type JobWorkerClass = new (options?: { name?: string }) => Worker;

export interface JobBindings extends IndexBindings {
  JobWorker: {
    module(): WebAssembly.Module;
    unpack(stored: Uint8Array, raw: number): Uint8Array;
    compress(bytes: Uint8Array): Uint8Array;
  };
}

interface JobGame {
  bindings: JobBindings;
  module: WebAssembly.Module;
  worker: JobWorkerClass;
}

let game: JobGame | undefined;

export function useJobs(bindings: JobBindings, worker: JobWorkerClass) {
  game ??= { bindings, module: bindings.JobWorker.module(), worker };
}

interface Task {
  counted: boolean;
  resolve(value: unknown): void;
  reject(error: Error): void;
}

type Slot =
  | { state: "live"; worker: Worker; tasks: Map<number, Task>; shared: Set<string> }
  | { state: "broken" };

type Live = Extract<Slot, { state: "live" }>;

export class JobPool {
  private readonly slots: Slot[] = [];
  private readonly shares = new Map<string, Blob>();
  private swept = false;
  private runtime: JobRuntime | undefined;
  private hash: Uint8Array | undefined;
  private failed: ((error: unknown) => void) | undefined;
  private ready = false;
  private timer = 0;
  private nextTask = 1;
  private restarts = 0;
  private reported = "";
  private readonly store: Promise<JobStore | undefined>;

  constructor(private readonly count: number, private readonly game: JobGame) {
    this.store = openStore(game.bindings, { unpack: (bytes, raw) => this.unpack(bytes, raw), compress: bytes => this.compress(bytes) });
    for (let index = 0; index < count; index++) this.slots.push(this.spawn(index));
  }

  get size(): number {
    return this.slots.filter(slot => slot.state === "live").length;
  }

  attach(runtime: JobRuntime, failed: (error: unknown) => void) {
    this.runtime = runtime;
    this.failed = failed;
    this.ready = false;
    this.hash = this.call(runtime => runtime.job_hash());
    if (this.hash) {
      for (const slot of this.slots) if (slot.state === "live") slot.shared.clear();
      this.broadcast({ type: "list", hash: this.hash });
    }
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

  detach() {
    clearInterval(this.timer);
    this.timer = 0;
    this.runtime = undefined;
    this.failed = undefined;
    this.ready = false;
  }

  terminate() {
    this.detach();
    for (const [index, slot] of this.slots.entries()) {
      if (slot.state !== "live") continue;
      slot.worker.terminate();
      for (const task of slot.tasks.values()) task.reject(new Error("the engine stopped"));
      this.slots[index] = { state: "broken" };
    }
    if (pool === this) pool = undefined;
  }

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

  started() {
    if (this.swept) return;
    this.swept = true;
    void this.store.then(store => store?.sweep());
  }

  private stop(error: unknown) {
    const failed = this.failed;
    this.terminate();
    failed?.(error);
  }

  exchange() {
    if (!this.ready) return;
    const orders = this.call(runtime => runtime.take_jobs()) ?? [];
    const lookups: Lookup[] = [];
    for (const order of orders) {
      if ("lookup" in order) lookups.push(order);
      else this.carry(order);
    }
    if (lookups.length > 0) void this.lookup(lookups);
    void this.report();
  }

  private carry(order: Exclude<JobOrder, Lookup>) {
    if ("clear" in order) {
      this.shares.clear();
      for (const slot of this.slots) if (slot.state === "live") slot.shared.clear();
      this.broadcast({ type: "clear" });
    } else if ("share" in order) {
      this.shares.set(order.share, new Blob([order.bytes as Uint8Array<ArrayBuffer>]));
    } else if ("unshare" in order) {
      this.shares.delete(order.unshare);
      this.forget(order.unshare);
    } else if ("run" in order) {
      this.post(this.slots[order.worker], { type: "run", id: order.run, kind: order.kind, source: order.source, input: order.input,
        priority: order.priority }, [order.input.buffer as ArrayBuffer]);
    } else if ("cancel" in order) {
      this.post(this.slots[order.worker], { type: "cancel", id: order.cancel });
    } else if ("keep" in order) {
      void this.store.then(store => store?.keep(order.keep, order.output));
    } else {
      void this.store.then(store => store?.mismatch(order.mismatch));
    }
  }

  private forget(key: string) {
    for (const slot of this.slots) {
      if (slot.state !== "live" || !slot.shared.delete(key)) continue;
      this.post(slot, { type: "unshare", key });
    }
  }

  private async lookup(lookups: Lookup[]) {
    const runtime = this.runtime;
    const store = await this.store;
    const answers = store?.lookup(lookups.map(order => order.key));
    await Promise.all(lookups.map(async (order, at) => {
      const kept = await answers?.[at].catch(() => undefined);
      if (this.runtime !== runtime) return;
      this.call(runtime => runtime.job_kept(order.lookup, order.key, kept));
      this.exchange();
    }));
  }

  private supply(slot: Live, id: number, keys: string[]) {
    for (const key of keys) {
      const blob = this.shares.get(key);
      if (!blob || slot.shared.has(key)) continue;
      slot.shared.add(key);
      this.post(slot, { type: "share", key, blob });
    }
    this.post(slot, { type: "resume", id });
  }

  private async report() {
    const runtime = this.runtime;
    const store = await this.store;
    if (!store || this.runtime !== runtime) return;
    const [hits, misses, bytes, mismatches] = store.counters();
    const text = `${hits} ${misses} ${bytes} ${mismatches}`;
    if (text === this.reported) return;
    this.reported = text;
    this.call(runtime => runtime.report_job_cache(hits, misses, bytes, mismatches));
  }

  private unpack(stored: Uint8Array, raw: number): Promise<Uint8Array> {
    if (this.size === 0) return Promise.resolve().then(() => this.game.bindings.JobWorker.unpack(stored, raw));
    const frames = zstdFrames(stored, raw);
    if (!frames) return this.task({ type: "unpack", bytes: stored, raw }, [stored.buffer as ArrayBuffer]) as Promise<Uint8Array>;
    const parts = frames.map(([start, end, size]) => {
      const bytes = stored.slice(start, end);
      return this.task({ type: "unpack", bytes, raw: size }, [bytes.buffer]) as Promise<Uint8Array>;
    });
    return Promise.all(parts).then(parts => {
      const output = new Uint8Array(raw);
      let at = 0;
      for (const part of parts) {
        output.set(part, at);
        at += part.length;
      }
      return output;
    });
  }

  private compress(bytes: Uint8Array): Promise<Uint8Array> {
    if (this.size === 0) return Promise.resolve().then(() => this.game.bindings.JobWorker.compress(bytes));
    return this.task({ type: "compress", bytes }, [bytes.buffer as ArrayBuffer]) as Promise<Uint8Array>;
  }

  decodeImage(uri: string, bytes: Uint8Array): Promise<Pixels> {
    return this.task({ type: "image", uri, bytes }, [bytes.buffer as ArrayBuffer]) as Promise<Pixels>;
  }

  private task(message: Record<string, unknown>, transfer: Transferable[]): Promise<unknown> {
    const counted = this.call(runtime => runtime.job_task());
    const slot = counted !== undefined ? this.slots[counted] : this.fewestTasks();
    if (slot?.state !== "live") return Promise.reject(new Error("no workers"));
    const task = this.nextTask++;
    return new Promise((resolve, reject) => {
      slot.tasks.set(task, { counted: counted !== undefined, resolve, reject });
      this.post(slot, { ...message, task }, transfer);
    });
  }

  private fewestTasks(): Live | undefined {
    let best: Live | undefined;
    for (const slot of this.slots) if (slot.state === "live" && (!best || slot.tasks.size < best.tasks.size)) best = slot;
    return best;
  }

  private spawn(index: number): Live {
    const worker = new this.game.worker({ name: "engine-jobs" });
    const slot: Live = { state: "live", worker, tasks: new Map(), shared: new Set() };
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => this.receive(index, worker, event.data);
    worker.onerror = (event) => { event.preventDefault(); this.lose(index, worker, event.message || "worker error"); };
    this.post(slot, { type: "init", module: this.game.module });
    if (this.hash) this.post(slot, { type: "list", hash: this.hash });
    return slot;
  }

  private live(index: number, worker: Worker): Live | undefined {
    const slot = this.slots[index];
    return slot?.state === "live" && slot.worker === worker ? slot : undefined;
  }

  private post(slot: Slot | undefined, message: unknown, transfer: Transferable[] = []) {
    if (slot?.state !== "live") return;
    slot.worker.postMessage(message, transfer);
  }

  private broadcast(message: Record<string, unknown>) {
    for (const slot of this.slots) this.post(slot, message);
  }

  private receive(index: number, worker: Worker, message: WorkerMessage) {
    const slot = this.live(index, worker);
    if (!slot) return;
    switch (message.type) {
      case "done":
        this.call(runtime => runtime.complete_job(message.id, message.output));
        this.exchange();
        break;
      case "failed":
        this.call(runtime => runtime.fail_job(message.id, message.error));
        this.exchange();
        break;
      case "wants":
        this.supply(slot, message.id, message.keys);
        break;
      case "task":
      case "task_failed": {
        const task = slot.tasks.get(message.task);
        slot.tasks.delete(message.task);
        if (task?.counted) this.call(runtime => runtime.job_task_done(index));
        if (message.type === "task") task?.resolve(message.result);
        else task?.reject(new Error(message.error));
        break;
      }
      case "broken":
        this.lose(index, worker, message.error);
        break;
      case "panicked":
        this.stop(message.report ? new Error(`${PANIC_PREFIX}${JSON.stringify(message.report)}`)
          : new WebAssembly.RuntimeError(message.error));
        break;
    }
  }

  private lose(index: number, worker: Worker, error: string) {
    const slot = this.live(index, worker);
    if (!slot) return;
    slot.worker.terminate();
    console.warn(`An engine worker stopped: ${error}`);
    for (const task of slot.tasks.values()) task.reject(new Error(error));
    const replaced = this.restarts < this.count;
    if (replaced) this.restarts++;
    this.slots[index] = replaced ? this.spawn(index) : { state: "broken" };
    this.call(runtime => runtime.lose_job_worker(index, error, replaced));
  }
}

async function openStore(bindings: IndexBindings, coder: Coder): Promise<JobStore | undefined> {
  const settings = jobCacheSettings();
  if (!settings) return undefined;
  return JobStore.open(bindings, settings, coder);
}

let pool: JobPool | null | undefined;

export function jobPool(): JobPool | undefined {
  if (pool === undefined && game) {
    const count = workerCount(new URLSearchParams(location.search).get("workers"));
    jobCacheSettings();
    pool = new JobPool(typeof Worker !== "undefined" ? count : 0, game);
  }
  return pool ?? undefined;
}

function workerCount(requested: string | null): number {
  if (requested === null) {
    return Math.max(0, navigator.hardwareConcurrency - FRAME_CORES);
  }
  if (requested === "off") return 0;
  if (/^\d+$/.test(requested) && Number.isSafeInteger(Number(requested))) return Number(requested);
  throw engineFailure({ code: "invalid_value", path: "?workers", allowed: { kind: "at_least", min: 0 }, got: requested });
}

const ZSTD_MAGIC = 0xfd2fb528;
const DICTIONARY_BYTES = [0, 1, 2, 4];

function zstdFrames(stored: Uint8Array, raw: number): [number, number, number][] | undefined {
  const view = new DataView(stored.buffer, stored.byteOffset, stored.byteLength);
  const frames: [number, number, number][] = [];
  let at = 0;
  let total = 0;
  while (at < stored.length) {
    if (at + 5 > stored.length || view.getUint32(at, true) !== ZSTD_MAGIC) return undefined;
    const start = at;
    const descriptor = stored[at + 4];
    const single = (descriptor & 0x20) !== 0;
    at += 5 + (single ? 0 : 1) + DICTIONARY_BYTES[descriptor & 3];
    const sizeBytes = [single ? 1 : 0, 2, 4, 8][descriptor >> 6];
    if (sizeBytes === 0 || at + sizeBytes > stored.length) return undefined;
    const size = sizeBytes === 1 ? view.getUint8(at)
      : sizeBytes === 2 ? view.getUint16(at, true) + 256
      : sizeBytes === 4 ? view.getUint32(at, true) : Number(view.getBigUint64(at, true));
    at += sizeBytes;
    let last = false;
    while (!last) {
      if (at + 3 > stored.length) return undefined;
      const header = view.getUint16(at, true) | (view.getUint8(at + 2) << 16);
      const type = (header >> 1) & 3;
      if (type === 3) return undefined;
      last = (header & 1) !== 0;
      at += 3 + (type === 1 ? 1 : header >>> 3);
    }
    at += (descriptor & 4) !== 0 ? 4 : 0;
    if (at > stored.length) return undefined;
    frames.push([start, at, size]);
    total += size;
  }
  return frames.length > 1 && total === raw ? frames : undefined;
}
