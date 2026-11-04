import { engineFailure } from "../generated/errors";
import { JOB_CACHE_MIB, MAX_JOB_CACHE_MIB } from "../generated/host";

declare const __PUBLIC_PACKS__: boolean;

const DIRECTORY = "engine-job-cache";
const INDEX = "index.json";
const SAVE_DELAY = 2000;
const MIB = 1024 * 1024;
const PACKS = `${import.meta.env.BASE_URL}cooked/`;
const PACK_HEADER = 32;

interface JobCacheIndex {
  verify(): boolean;
  counters(): Float64Array;
  lists(name: string): boolean;
  holds(key: string): boolean;
  hit(key: string, bytes?: number): boolean;
  miss(): void;
  store(key: string, bytes: number): boolean;
  mismatch(key: string): void;
  failed(): void;
  take_removed(): string[];
  index(): string | undefined;
}

interface PackDirectory {
  find(key: string): Float64Array | undefined;
  unpack(key: string, stored: Uint8Array): Uint8Array;
}

export interface IndexBindings {
  JobCacheIndex: new (budget: number, available: number | undefined, verify: boolean, index: string | undefined) => JobCacheIndex;
  PackDirectory: { new (header: Uint8Array, directory: Uint8Array): PackDirectory; locate(header: Uint8Array): Float64Array };
}

interface Pack { url: string; directory: PackDirectory }

export type Unpacker = (stored: Uint8Array, raw: number) => Promise<Uint8Array> | undefined;

export function jobCacheSettings(): { budget: number; verify: boolean } | undefined {
  const requested = new URLSearchParams(location.search).get("jobcache");
  const mebibytes = requested !== null && /^\d+$/.test(requested) ? Number(requested) : undefined;
  if (requested !== null && requested !== "off" && requested !== "verify" && !(mebibytes !== undefined && mebibytes <= MAX_JOB_CACHE_MIB)) {
    throw engineFailure({ code: "invalid_value", path: "?jobcache", allowed: { kind: "range", min: 0, max: MAX_JOB_CACHE_MIB }, got: requested });
  }
  if (requested === "off" || typeof navigator === "undefined" || !navigator.storage?.getDirectory) return undefined;
  return { budget: (mebibytes ?? JOB_CACHE_MIB) * MIB, verify: requested === "verify" };
}

export class JobStore {
  private timer = 0;
  private readonly writing = new Map<string, Promise<void>>();
  private packHits = 0;

  private constructor(private readonly directory: FileSystemDirectoryHandle, readonly index: JobCacheIndex,
    private readonly packs: Pack[], private readonly unpack: Unpacker) {}

  static async open(bindings: IndexBindings, budget: number, verify: boolean, unpack: Unpacker): Promise<JobStore | undefined> {
    try {
      const root = await navigator.storage.getDirectory();
      const directory = await root.getDirectoryHandle(DIRECTORY, { create: true });
      const saved = await directory.getFileHandle(INDEX).then(handle => handle.getFile()).then(file => file.text()).catch(() => undefined);
      const estimate = await navigator.storage.estimate?.().catch(() => undefined);
      const available = estimate?.quota !== undefined && estimate.usage !== undefined ? Math.max(0, estimate.quota - estimate.usage) : undefined;
      void navigator.storage.persist?.().catch(() => false);
      const store = new JobStore(directory, new bindings.JobCacheIndex(budget, available, verify, saved), await openPacks(bindings), unpack);
      const names = directory as unknown as { keys(): AsyncIterable<string> };
      for await (const name of names.keys()) {
        if (name !== INDEX && !store.index.lists(name)) await directory.removeEntry(name).catch(() => undefined);
      }
      store.changed();
      return store;
    } catch (error) {
      console.warn(`The record store is off: ${String(error)}`);
      return undefined;
    }
  }

  get verify(): boolean {
    return this.index.verify();
  }

  counters(): [number, number, number, number] {
    const [hits, misses, bytes, mismatches] = this.index.counters();
    return [hits + this.packHits, misses, bytes, mismatches];
  }

  async lookup(key: string): Promise<Uint8Array | undefined> {
    if (this.index.holds(key)) {
      let bytes: Uint8Array | undefined;
      try {
        await this.writing.get(key);
        bytes = new Uint8Array(await (await (await this.directory.getFileHandle(key)).getFile()).arrayBuffer());
      } catch {}
      const hit = this.index.hit(key, bytes?.length);
      this.changed();
      if (hit) return bytes;
    }
    for (const pack of this.packs) {
      const entry = pack.directory.find(key);
      if (!entry) continue;
      try {
        const [offset, stored, raw] = entry;
        const bytes = await fetchRange(pack.url, offset, stored);
        const record = await this.unpack(bytes, raw) ?? pack.directory.unpack(key, bytes);
        this.packHits++;
        this.keep(key, record);
        return record;
      } catch (error) {
        console.warn(`A record of ${pack.url} does not read: ${String(error)}`);
      }
    }
    this.index.miss();
    return undefined;
  }

  keep(key: string, output: Uint8Array) {
    if (this.writing.has(key) || !this.index.store(key, output.length)) return;
    const done = this.write(key, output).finally(() => this.writing.delete(key));
    this.writing.set(key, done);
    this.changed();
  }

  mismatch(key: string) {
    console.warn(`A kept record differs from a fresh run: ${key}`);
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
      console.warn(`The record store stopped keeping records: ${String(error)}`);
    }
  }

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
      console.warn(`The record store's index was not written: ${String(error)}`);
    }
  }
}

async function packNames(): Promise<string[]> {
  if (!__PUBLIC_PACKS__) return [];
  const url = `${PACKS}index.json`;
  const response = await fetch(url);
  if (!response.ok) throw engineFailure({ code: "source_io", path: "$", got: { url, status: response.status } });
  const listed: unknown = await response.json();
  if (!Array.isArray(listed) || !listed.every(name => typeof name === "string")) throw engineFailure({ code: "invalid_value", path: "$", got: url });
  return listed as string[];
}

async function openPacks(bindings: IndexBindings): Promise<Pack[]> {
  const names = await packNames().catch((error: unknown): string[] => {
    console.error(`The packs are left out: ${String(error)}`);
    return [];
  });
  const packs: Pack[] = [];
  for (const name of names) {
    const url = `${PACKS}${name}`;
    try {
      const header = await fetchRange(url, 0, PACK_HEADER);
      const [offset, stored] = bindings.PackDirectory.locate(header);
      const directory = new bindings.PackDirectory(header, await fetchRange(url, offset, stored));
      packs.push({ url, directory });
    } catch (error) {
      console.warn(`The pack ${url} is left out: ${String(error)}`);
    }
  }
  return packs;
}

async function fetchRange(url: string, offset: number, length: number): Promise<Uint8Array> {
  const response = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  if (!response.ok) throw engineFailure({ code: "source_io", path: "$", got: { url, status: response.status } });
  const bytes = new Uint8Array(await response.arrayBuffer());
  const range = response.status === 206 ? bytes : bytes.slice(offset, offset + length);
  if (range.length !== length) {
    throw engineFailure({ code: "source_io", path: "$", allowed: { kind: "range", min: length, max: length }, got: range.length });
  }
  return range;
}
