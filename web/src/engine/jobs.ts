import workerWasmUrl from "../../pkg/engine_worker_bg.wasm?url";
import { engineFailure } from "../generated/errors";
import { FRAME_CORES, MAX_WORKERS, PANIC_PREFIX } from "../generated/host";
import type { Runtime } from "../generated/runtime";
import { engineWorkerModule } from "./bindings";
import { JobStore, jobCacheSettings, type Unpacker } from "./job-cache";
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

interface Slot {
  worker: Worker;
  tasks: Map<number, boolean>;
  broken: boolean;
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
  private hash: Uint8Array | undefined;
  private failed: ((error: unknown) => void) | undefined;
  private ready = false;
  private timer = 0;
  private nextTask = 1;
  private restarts = 0;
  private reported = "";
  private readonly module: Promise<WebAssembly.Module | undefined> = workerWasm()
    .then(bytes => bytes && WebAssembly.compile(bytes))
    .catch(() => undefined);
  private readonly store: Promise<JobStore | undefined> = this.module.then(module =>
    openStore(module, (bytes, raw) => this.unpack(bytes, raw)));

  constructor(private readonly count: number) {
    for (let index = 0; index < count; index++) this.slots.push(this.spawn());
  }

  get size(): number {
    return this.slots.filter(slot => !slot.broken).length;
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
    for (const slot of this.slots) {
      slot.broken = true;
      slot.worker.terminate();
    }
    for (const task of this.tasks.values()) task.reject(new Error("the engine stopped"));
    this.tasks.clear();
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
    void this.module.then(module => {
      worker.postMessage({ type: "init", bindings: engineWorkerModule(), module });
      for (const [message, transfer] of slot.held ?? []) worker.postMessage(message, transfer);
      slot.held = undefined;
    });
    if (this.hash) this.post(slot, { type: "list", hash: this.hash });
    for (const [key, blob] of this.shares) this.post(slot, { type: "share", key, blob });
    return slot;
  }

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
        this.stop(message.report ? new Error(`${PANIC_PREFIX}${JSON.stringify(message.report)}`)
          : new WebAssembly.RuntimeError(message.error));
        break;
    }
  }

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

function workerWasm(): Promise<ArrayBuffer | undefined> {
  wasm ??= fetchWorkerWasm();
  return wasm;
}

async function fetchWorkerWasm(): Promise<ArrayBuffer | undefined> {
  try {
    const pinned = engineWorkerModule();
    const url = pinned ? pinned.replace(/\.js$/, "_bg.wasm") : workerWasmUrl;
    const response = await fetch(url);
    if (!response.ok) throw engineFailure({ code: "source_io", path: "$", got: { url, status: response.status } });
    return await response.arrayBuffer();
  } catch (error) {
    console.warn(`The job worker module was not fetched: ${String(error)}`);
    return undefined;
  }
}

async function openStore(module: WebAssembly.Module | undefined, unpack: Unpacker): Promise<JobStore | undefined> {
  const settings = jobCacheSettings();
  if (!settings || !module) return undefined;
  const pinned = engineWorkerModule();
  const bindings = (pinned ? await import(/* @vite-ignore */ pinned) : await import("../../pkg/engine_worker")) as typeof import("../../pkg/engine_worker");
  await bindings.default({ module_or_path: module });
  return JobStore.open(bindings, settings.budget, settings.verify, unpack);
}

let pool: JobPool | null | undefined;

export function jobPool(): JobPool | undefined {
  if (pool === undefined) {
    const count = workerCount(new URLSearchParams(location.search).get("workers"));
    jobCacheSettings();
    pool = new JobPool(typeof Worker !== "undefined" ? count : 0);
  }
  return pool ?? undefined;
}

function workerCount(requested: string | null): number {
  if (requested === null) {
    return Math.max(1, Math.min(MAX_WORKERS, (navigator.hardwareConcurrency || 4) - FRAME_CORES));
  }
  if (requested === "off") return 0;
  if (/^\d+$/.test(requested) && Number(requested) <= MAX_WORKERS) return Number(requested);
  throw engineFailure({ code: "invalid_value", path: "?workers", allowed: { kind: "range", min: 0, max: MAX_WORKERS }, got: requested });
}
