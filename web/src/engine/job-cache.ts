// The page's job output cache: outputs of jobs the engine
// marks keepable (terrain tiles, their far objects, walks and bakes) and
// KTX2 textures transcoded on workers, kept in the origin's private file
// system under a hash of the job worker module, the job's kind and its
// identity, so the same build hands them back without running their jobs,
// in this session or a later one. The least recently used beyond a byte
// budget go. ?jobcache=off keeps none, ?jobcache=verify runs every job
// anyway and compares, and ?jobcache=N sets the budget in MiB: 1 GiB by
// default, at most half of what storage the browser offers is free.

import { JOB_CACHE_MIB, MAX_JOB_CACHE_MIB } from "../generated/host";

const DIRECTORY = "engine-job-cache";
const INDEX = "index.json";
/** Milliseconds the index waits for more changes before it is written. */
const SAVE_DELAY = 2000;
const MIB = 1024 * 1024;

interface Entry {
  bytes: number;
  used: number;
}

interface JobCacheStats {
  hits: number;
  misses: number;
  bytes: number;
  mismatches: number;
}

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

function hex(buffer: ArrayBuffer): string {
  return Array.from(new Uint8Array(buffer), byte => byte.toString(16).padStart(2, "0")).join("");
}

export class JobCache {
  private hits = 0;
  private misses = 0;
  private mismatches = 0;
  private total = 0;
  private timer = 0;
  private broken = false;
  private readonly writing = new Map<string, Promise<void>>();

  private constructor(
    private readonly directory: FileSystemDirectoryHandle,
    private readonly version: string,
    private readonly budget: number,
    readonly verify: boolean,
    private readonly entries: Map<string, Entry>,
  ) {
    for (const entry of entries.values()) this.total += entry.bytes;
  }

  /** Opens the cache for the worker module `module` (its bytes), within
   * `budget` bytes; `undefined` where the file system refuses. */
  static async open(module: ArrayBuffer, budget: number, verify: boolean): Promise<JobCache | undefined> {
    try {
      const root = await navigator.storage.getDirectory();
      const directory = await root.getDirectoryHandle(DIRECTORY, { create: true });
      const version = hex(await crypto.subtle.digest("SHA-256", module));
      const entries = await readIndex(directory) ?? new Map<string, Entry>();
      // Files the index does not list, as those written just before a page
      // closed, are no one's.
      const names = directory as unknown as { keys(): AsyncIterable<string> };
      for await (const name of names.keys()) {
        if (name !== INDEX && !entries.has(name)) await directory.removeEntry(name).catch(() => undefined);
      }
      let held = 0;
      for (const entry of entries.values()) held += entry.bytes;
      const estimate = await navigator.storage.estimate?.().catch(() => undefined);
      const free = estimate?.quota !== undefined && estimate.usage !== undefined
        ? Math.max(0, estimate.quota - estimate.usage + held) / 2
        : budget;
      const cache = new JobCache(directory, version, Math.min(budget, free), verify, entries);
      await cache.evict();
      return cache;
    } catch (error) {
      console.warn(`The job cache is off: ${String(error)}`);
      return undefined;
    }
  }

  stats(): JobCacheStats {
    return { hits: this.hits, misses: this.misses, bytes: this.total, mismatches: this.mismatches };
  }

  /** The key of a job of `kind` with `identity` from this build. */
  async key(kind: string, identity: Uint8Array): Promise<string> {
    const head = new TextEncoder().encode(`${this.version}\u0000${kind}\u0000`);
    const bytes = new Uint8Array(head.length + identity.length);
    bytes.set(head);
    bytes.set(identity, head.length);
    return hex(await crypto.subtle.digest("SHA-256", bytes));
  }

  /** The output kept as `key`, if any. */
  async lookup(key: string): Promise<Uint8Array | undefined> {
    const entry = this.entries.get(key);
    if (!entry || this.broken) {
      this.misses++;
      return undefined;
    }
    try {
      await this.writing.get(key);
      const file = await (await this.directory.getFileHandle(key)).getFile();
      const bytes = new Uint8Array(await file.arrayBuffer());
      if (bytes.length !== entry.bytes) throw new Error("truncated");
      entry.used = Date.now();
      this.save();
      this.hits++;
      return bytes;
    } catch {
      await this.remove(key);
      this.misses++;
      return undefined;
    }
  }

  /** Keeps `output` as `key`. */
  store(key: string, output: Uint8Array) {
    if (this.broken || output.length > this.budget || this.writing.has(key)) return;
    const done = this.write(key, output).finally(() => this.writing.delete(key));
    this.writing.set(key, done);
  }

  /** Counts a verified output that differed from the one kept. */
  mismatch(key: string) {
    this.mismatches++;
    console.warn(`A kept job output differs from a fresh run: ${key}`);
    void this.remove(key);
  }

  private async write(key: string, output: Uint8Array) {
    try {
      const writable = await (await this.directory.getFileHandle(key, { create: true })).createWritable();
      await writable.write(output as Uint8Array<ArrayBuffer>);
      await writable.close();
      this.total += output.length - (this.entries.get(key)?.bytes ?? 0);
      this.entries.set(key, { bytes: output.length, used: Date.now() });
      await this.evict();
      this.save();
    } catch (error) {
      this.broken = true;
      console.warn(`The job cache stopped keeping outputs: ${String(error)}`);
    }
  }

  /** Drops the least recently used outputs beyond the budget. */
  private async evict() {
    if (this.total <= this.budget) return;
    const oldest = [...this.entries].sort((a, b) => a[1].used - b[1].used);
    for (const [key] of oldest) {
      if (this.total <= this.budget) break;
      await this.remove(key);
    }
    this.save();
  }

  private async remove(key: string) {
    const entry = this.entries.get(key);
    if (!entry) return;
    this.entries.delete(key);
    this.total -= entry.bytes;
    this.save();
    await this.directory.removeEntry(key).catch(() => undefined);
  }

  /** Writes the index a little after its last change. */
  private save() {
    if (this.timer) return;
    this.timer = window.setTimeout(() => {
      this.timer = 0;
      void this.writeIndex();
    }, SAVE_DELAY);
  }

  private async writeIndex() {
    try {
      const index: Record<string, [number, number]> = {};
      for (const [key, entry] of this.entries) index[key] = [entry.bytes, entry.used];
      const writable = await (await this.directory.getFileHandle(INDEX, { create: true })).createWritable();
      await writable.write(JSON.stringify(index));
      await writable.close();
    } catch (error) {
      console.warn(`The job cache's index was not written: ${String(error)}`);
    }
  }
}

/** The index's entries; `undefined` when it is missing or unreadable. */
async function readIndex(directory: FileSystemDirectoryHandle): Promise<Map<string, Entry> | undefined> {
  try {
    const file = await (await directory.getFileHandle(INDEX)).getFile();
    const index = JSON.parse(await file.text()) as Record<string, [number, number]>;
    const entries = new Map<string, Entry>();
    for (const [key, value] of Object.entries(index)) {
      if (/^[0-9a-f]{64}$/.test(key) && Array.isArray(value) && Number.isSafeInteger(value[0]) && Number.isFinite(value[1])) {
        entries.set(key, { bytes: value[0], used: value[1] });
      }
    }
    return entries;
  } catch {
    return undefined;
  }
}
