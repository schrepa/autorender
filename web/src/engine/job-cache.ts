// The page's job output cache: outputs of jobs the engine
// marks keepable (terrain tiles, their far objects, walks and bakes) and
// KTX2 textures transcoded on workers, kept in the origin's private file
// system. Which outputs it keeps, under which keys, and which go beyond
// its byte budget is the job worker module's (its JobCacheIndex, as the
// native host's); this file keeps the bytes and the index's JSON beside
// them. ?jobcache=off keeps none, ?jobcache=verify runs every job anyway
// and compares, and ?jobcache=N sets the budget in MiB: 1 GiB by default,
// at most half of what storage the browser offers is free.

import type { JobCacheIndex } from "../../pkg/engine_worker";
import { JOB_CACHE_MIB, MAX_JOB_CACHE_MIB } from "../generated/host";

const DIRECTORY = "engine-job-cache";
const INDEX = "index.json";
/** Milliseconds the index waits for more changes before it is written. */
const SAVE_DELAY = 2000;
const MIB = 1024 * 1024;

/** The job worker module's bindings, which hold the cache's index. */
type IndexBindings = { JobCacheIndex: typeof JobCacheIndex };

/** The cache's settings from the page's URL: `undefined` when off. A
 * value the native host's --jobcache refuses fails. */
export function jobCacheSettings(): { budget: number; verify: boolean } | undefined {
  const requested = new URLSearchParams(location.search).get("jobcache");
  const mebibytes = requested !== null && /^\d+$/.test(requested) ? Number(requested) : undefined;
  if (requested !== null && requested !== "off" && requested !== "verify" && !(mebibytes !== undefined && mebibytes <= MAX_JOB_CACHE_MIB)) {
    throw new Error(`?jobcache is off, verify or 0 to ${MAX_JOB_CACHE_MIB} MiB, not ${requested}.`);
  }
  if (requested === "off" || typeof navigator === "undefined" || !navigator.storage?.getDirectory) return undefined;
  return { budget: (mebibytes ?? JOB_CACHE_MIB) * MIB, verify: requested === "verify" };
}

export class JobStore {
  private timer = 0;
  private readonly writing = new Map<string, Promise<void>>();

  private constructor(private readonly directory: FileSystemDirectoryHandle, readonly index: JobCacheIndex) {}

  /** Opens the cache of the job worker module `module` (its bytes), its
   * index in `bindings`; `undefined` where the file system refuses. */
  static async open(module: ArrayBuffer, bindings: IndexBindings, budget: number, verify: boolean): Promise<JobStore | undefined> {
    try {
      const root = await navigator.storage.getDirectory();
      const directory = await root.getDirectoryHandle(DIRECTORY, { create: true });
      const saved = await directory.getFileHandle(INDEX).then(handle => handle.getFile()).then(file => file.text()).catch(() => undefined);
      const estimate = await navigator.storage.estimate?.().catch(() => undefined);
      const available = estimate?.quota !== undefined && estimate.usage !== undefined ? Math.max(0, estimate.quota - estimate.usage) : undefined;
      const store = new JobStore(directory, new bindings.JobCacheIndex(new Uint8Array(module), budget, available, verify, saved));
      // Files the index does not list, as those written just before a page
      // closed, are no one's.
      const names = directory as unknown as { keys(): AsyncIterable<string> };
      for await (const name of names.keys()) {
        if (name !== INDEX && !store.index.lists(name)) await directory.removeEntry(name).catch(() => undefined);
      }
      store.changed();
      return store;
    } catch (error) {
      console.warn(`The job cache is off: ${String(error)}`);
      return undefined;
    }
  }

  get verify(): boolean {
    return this.index.verify();
  }

  /** The key of the work of `kind` on `identity`, and the output kept as
   * it, if any. */
  async lookup(kind: string, identity: Uint8Array): Promise<[string, Uint8Array | undefined]> {
    const key = this.index.key(kind, identity);
    if (!this.index.holds(key)) {
      this.index.miss();
      return [key, undefined];
    }
    let bytes: Uint8Array | undefined;
    try {
      await this.writing.get(key);
      bytes = new Uint8Array(await (await (await this.directory.getFileHandle(key)).getFile()).arrayBuffer());
    } catch { /* A file that is gone is a miss. */ }
    const hit = this.index.hit(key, bytes?.length);
    this.changed();
    return [key, hit ? bytes : undefined];
  }

  /** Keeps `output` as `key`, where the index takes it. */
  keep(key: string, output: Uint8Array) {
    if (this.writing.has(key) || !this.index.store(key, output.length)) return;
    const done = this.write(key, output).finally(() => this.writing.delete(key));
    this.writing.set(key, done);
    this.changed();
  }

  /** A verified output differed from the one kept as `key`, which goes. */
  mismatch(key: string) {
    console.warn(`A kept job output differs from a fresh run: ${key}`);
    this.index.mismatch(key);
    this.changed();
  }

  private async write(key: string, output: Uint8Array) {
    try {
      const writable = await (await this.directory.getFileHandle(key, { create: true })).createWritable();
      await writable.write(output as Uint8Array<ArrayBuffer>);
      await writable.close();
    } catch (error) {
      this.index.failed();
      console.warn(`The job cache stopped keeping outputs: ${String(error)}`);
    }
  }

  /** The index changed: the files it dropped go, and it is written a
   * little after its last change. */
  private changed() {
    for (const key of this.index.take_removed()) void this.directory.removeEntry(key).catch(() => undefined);
    if (this.timer) return;
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      void this.save();
    }, SAVE_DELAY);
  }

  private async save() {
    const index = this.index.index();
    if (index === undefined) return;
    try {
      const writable = await (await this.directory.getFileHandle(INDEX, { create: true })).createWritable();
      await writable.write(index);
      await writable.close();
    } catch (error) {
      console.warn(`The job cache's index was not written: ${String(error)}`);
    }
  }
}
