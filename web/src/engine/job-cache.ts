// The page's records: what the page kept in the origin's private file
// system (records the engine's jobs made and the engine handed over,
// records it read from packs, and KTX2 textures transcoded on workers),
// then the cook's packs, read by byte range from the server
// (`cooked/index.json` lists them), whose records the page keeps as it
// reads them, so a later visit reads them here. Keys are the engine's, so
// what is kept outlives builds that leave the code making it alone. Which records the store keeps, under which keys, and which go
// beyond its byte budget is the job worker module's (its JobCacheIndex,
// as the native host's); this file keeps the bytes and the index's JSON
// beside them. ?jobcache=off keeps none, ?jobcache=verify runs every job
// anyway and compares, and ?jobcache=N sets the budget in MiB: 1 GiB by
// default, at most half of what storage the browser offers is free.

import type { JobCacheIndex, PackDirectory } from "../../pkg/engine_worker";
import { JOB_CACHE_MIB, MAX_JOB_CACHE_MIB } from "../generated/host";

const DIRECTORY = "engine-job-cache";
const INDEX = "index.json";
/** Milliseconds the index waits for more changes before it is written. */
const SAVE_DELAY = 2000;
const MIB = 1024 * 1024;
/** The folder the cook's packs are served from, and their list. */
const PACKS = `${import.meta.env.BASE_URL}cooked/`;
/** Bytes of a pack's header, which names its directory's range. */
const PACK_HEADER = 32;

/** The job worker module's bindings, which hold the store's index and
 * read packs. */
type IndexBindings = { JobCacheIndex: typeof JobCacheIndex; PackDirectory: typeof PackDirectory };

/** A pack the page reads: its URL and directory. */
interface Pack { url: string; directory: PackDirectory }

/** Unpacks a pack's record of `raw` bytes from its `stored` bytes off the
 * page's thread: `undefined` where it cannot, and the page does. */
export type Unpacker = (stored: Uint8Array, raw: number) => Promise<Uint8Array> | undefined;

/** The store's settings from the page's URL: `undefined` when off. A
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
  /** Records the packs handed back. */
  private packHits = 0;

  private constructor(private readonly directory: FileSystemDirectoryHandle, readonly index: JobCacheIndex,
    private readonly packs: Pack[], private readonly unpack: Unpacker) {}

  /** Opens the store, its index in `bindings`, and the packs the server
   * lists; `undefined` where the file system refuses. */
  static async open(bindings: IndexBindings, budget: number, verify: boolean, unpack: Unpacker): Promise<JobStore | undefined> {
    try {
      const root = await navigator.storage.getDirectory();
      const directory = await root.getDirectoryHandle(DIRECTORY, { create: true });
      const saved = await directory.getFileHandle(INDEX).then(handle => handle.getFile()).then(file => file.text()).catch(() => undefined);
      const estimate = await navigator.storage.estimate?.().catch(() => undefined);
      const available = estimate?.quota !== undefined && estimate.usage !== undefined ? Math.max(0, estimate.quota - estimate.usage) : undefined;
      // Records kept are worth keeping through storage pressure.
      void navigator.storage.persist?.().catch(() => false);
      const store = new JobStore(directory, new bindings.JobCacheIndex(budget, available, verify, saved), await openPacks(bindings), unpack);
      // Files the index does not list, as those written just before a page
      // closed, are no one's.
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

  /** Records handed back (from the packs or the store), records lacked,
   * the store's bytes and verified records that differed from a fresh
   * run. */
  counters(): [number, number, number, number] {
    const [hits, misses, bytes, mismatches] = this.index.counters();
    return [hits + this.packHits, misses, bytes, mismatches];
  }

  /** The record kept as `key`, if the store or a pack holds it: one read
   * from a pack is kept too. */
  async lookup(key: string): Promise<Uint8Array | undefined> {
    if (this.index.holds(key)) {
      let bytes: Uint8Array | undefined;
      try {
        await this.writing.get(key);
        bytes = new Uint8Array(await (await (await this.directory.getFileHandle(key)).getFile()).arrayBuffer());
      } catch { /* A file that is gone is a miss. */ }
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

  /** Keeps `output` as `key`, where the index takes it. */
  keep(key: string, output: Uint8Array) {
    if (this.writing.has(key) || !this.index.store(key, output.length)) return;
    const done = this.write(key, output).finally(() => this.writing.delete(key));
    this.writing.set(key, done);
    this.changed();
  }

  /** A verified record differed from the one kept as `key`, which goes. */
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
      console.warn(`The record store's index was not written: ${String(error)}`);
    }
  }
}

/** The packs the server lists, their directories read by range; none
 * where it lists none. A pack that does not read is said so and left
 * out. */
async function openPacks(bindings: IndexBindings): Promise<Pack[]> {
  const listed = await fetch(`${PACKS}index.json`).then(response => response.ok ? response.json() as Promise<unknown> : []).catch(() => []);
  const names = Array.isArray(listed) ? listed.filter((name): name is string => typeof name === "string") : [];
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

/** Bytes `offset` to `offset + length` of `url`, fetched by range, in a
 * buffer of their own (which a worker may take): a server that answers
 * with the whole file has it cut to the range. */
async function fetchRange(url: string, offset: number, length: number): Promise<Uint8Array> {
  const response = await fetch(url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  const range = response.status === 206 ? bytes : bytes.slice(offset, offset + length);
  if (range.length !== length) throw new Error(`${url}: ${range.length} bytes, not ${length}`);
  return range;
}
