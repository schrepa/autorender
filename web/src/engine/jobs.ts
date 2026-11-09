import { engineFailure } from "../generated/errors";
import { FRAME_CORES, PANIC_PREFIX } from "../generated/host";
import type { Runtime } from "../generated/runtime";
import { type IndexBindings, JobStore, jobCacheSettings, type Unpacker } from "./job-cache";
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

type WorkerMessage =
  | { type: "done"; id: number; output: Uint8Array }
  | { type: "failed"; id: number; error: string }
  | { type: "task"; task: number; result: unknown }
  | { type: "task_failed"; task: number; error: string }
  | { type: "broken"; error: string }
  | { type: "panicked"; report: PanicReport | null; error: string };

export interface Pixels { width: number; height: number; pixels: Uint8Array }

export type JobWorkerClass = new (options?: { name?: string }) => Worker;

export interface JobBindings extends IndexBindings {
  JobWorker: { module(): WebAssembly.Module };
}

interface JobGame {
  index: IndexBindings;
  module: WebAssembly.Module;
  worker: JobWorkerClass;
}

let game: JobGame | undefined;

export function useJobs(bindings: JobBindings, worker: JobWorkerClass) {
  game ??= { index: bindings, module: bindings.JobWorker.module(), worker };
}

interface Task {
  counted: boolean;
  resolve(value: unknown): void;
  reject(error: Error): void;
}

type Slot =
  | { state: "live"; worker: Worker; tasks: Map<number, Task> }
  | { state: "broken" };

type Live = Extract<Slot, { state: "live" }>;

export class JobPool {
  private readonly slots: Slot[] = [];
  private readonly shares = new Map<string, Blob>();
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
    this.store = openStore(game.index, (bytes, raw) => this.unpack(bytes, raw));
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
    if (this.hash) this.broadcast({ type: "list", hash: this.hash });
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

  private stop(error: unknown) {
    const failed = this.failed;
    this.terminate();
    failed?.(error);
  }

  exchange() {
    if (!this.ready) return;
    const orders = this.call(runtime => runtime.take_jobs()) ?? [];
    for (const order of orders) this.carry(order);
    void this.report();
  }

  private carry(order: JobOrder) {
    if ("clear" in order) {
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
      this.post(this.slots[order.worker], { type: "run", id: order.run, kind: order.kind, source: order.source, input: order.input,
        priority: order.priority }, [order.input.buffer as ArrayBuffer]);
    } else if ("cancel" in order) {
      this.post(this.slots[order.worker], { type: "cancel", id: order.cancel });
    } else if ("lookup" in order) {
      void this.lookup(order.lookup, order.key);
    } else if ("keep" in order) {
      void this.store.then(store => store?.keep(order.keep, order.output));
    } else {
      void this.store.then(store => store?.mismatch(order.mismatch));
    }
  }

  private async lookup(id: number, key: string) {
    const runtime = this.runtime;
    const store = await this.store;
    const kept = store ? await store.lookup(key) : undefined;
    if (this.runtime !== runtime) return;
    this.call(runtime => runtime.job_kept(id, key, kept));
    this.exchange();
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

  private unpack(stored: Uint8Array, raw: number): Promise<Uint8Array> | undefined {
    if (this.size === 0) return undefined;
    return this.task({ type: "unpack", bytes: stored, raw }, [stored.buffer as ArrayBuffer]) as Promise<Uint8Array>;
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
    const slot: Live = { state: "live", worker, tasks: new Map() };
    worker.onmessage = (event: MessageEvent<WorkerMessage>) => this.receive(index, worker, event.data);
    worker.onerror = (event) => { event.preventDefault(); this.lose(index, worker, event.message || "worker error"); };
    this.post(slot, { type: "init", module: this.game.module });
    if (this.hash) this.post(slot, { type: "list", hash: this.hash });
    for (const [key, blob] of this.shares) this.post(slot, { type: "share", key, blob });
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
        break;
      case "failed":
        this.call(runtime => runtime.fail_job(message.id, message.error));
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

async function openStore(bindings: IndexBindings, unpack: Unpacker): Promise<JobStore | undefined> {
  const settings = jobCacheSettings();
  if (!settings) return undefined;
  return JobStore.open(bindings, settings.budget, settings.verify, unpack);
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
