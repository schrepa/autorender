import { engineFailure } from "../generated/errors";
import { JOB_CACHE_MIB, PACK_CACHE_MIB } from "../generated/host";

declare const __PUBLIC_PACKS__: boolean;

const OUTPUTS = "engine-job-cache";
const RECORDS = "engine-pack-records";
const INDEX = "kept.json";
const SAVE_DELAY = 2000;
const MIB = 1024 * 1024;
const PACKS = `${import.meta.env.BASE_URL}cooked/`;
const PACK_HEADER = 32;
const RAW = 4;

interface JobCacheIndex {
  verify(): boolean;
  budget(): number;
  counters(): Float64Array;
  lists(name: string): boolean;
  holds(key: string): boolean;
  bytes(key: string): number | undefined;
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
}

export interface IndexBindings {
  JobCacheIndex: new (budget: number, available: number | undefined, verify: boolean, index: string | undefined) => JobCacheIndex;
  PackDirectory: { new (header: Uint8Array, directory: Uint8Array): PackDirectory; locate(header: Uint8Array): Float64Array };
}

export interface Coder {
  unpack(stored: Uint8Array, raw: number): Promise<Uint8Array>;
  compress(bytes: Uint8Array): Promise<Uint8Array>;
}

export interface JobCacheSettings { outputs: number; records: number; verify: boolean }

interface Pack { url: string; directory: PackDirectory }

interface Wanted { key: string; url: string; offset: number; stored: number; raw: number }

export function jobCacheSettings(): JobCacheSettings | undefined {
  const requested = new URLSearchParams(location.search).get("jobcache");
  const mebibytes = requested !== null && /^\d+$/.test(requested) ? Number(requested) : undefined;
  if (requested !== null && requested !== "off" && requested !== "verify" && mebibytes === undefined) {
    throw engineFailure({ code: "invalid_value", path: "?jobcache", allowed: { kind: "at_least", min: 0 }, got: requested });
  }
  if (requested === "off" || typeof navigator === "undefined" || !navigator.storage?.getDirectory) return undefined;
  return { outputs: (mebibytes ?? JOB_CACHE_MIB) * MIB, records: PACK_CACHE_MIB * MIB, verify: requested === "verify" };
}

class Shelf {
  private timer = 0;
  private readonly writing = new Map<string, Promise<void>>();
  private readonly handles = new Map<string, FileSystemFileHandle>();

  private constructor(private readonly directory: FileSystemDirectoryHandle, readonly index: JobCacheIndex) {}

  static async open(root: FileSystemDirectoryHandle, name: string, index: (saved: string | undefined) => JobCacheIndex): Promise<Shelf> {
    const directory = await root.getDirectoryHandle(name, { create: true });
    const saved = await directory.getFileHandle(INDEX).then(handle => handle.getFile()).then(file => file.text()).catch(() => undefined);
    const shelf = new Shelf(directory, index(saved));
    shelf.changed();
    void shelf.list();
    return shelf;
  }

  private async list() {
    const entries = this.directory as unknown as { entries(): AsyncIterable<[string, FileSystemHandle]> };
    try {
      for await (const [name, handle] of entries.entries()) {
        if (handle.kind === "file" && this.index.lists(name) && !this.handles.has(name)) this.handles.set(name, handle as FileSystemFileHandle);
      }
    } catch {}
  }

  size(key: string): number {
    return this.index.bytes(key) ?? 0;
  }

  holds(key: string): boolean {
    return this.writing.has(key) || this.index.holds(key);
  }

  async read(key: string): Promise<{ stored: Uint8Array; raw: number } | undefined> {
    await this.writing.get(key);
    if (!this.index.holds(key)) return undefined;
    let file: Uint8Array | undefined;
    try {
      const handle = this.handles.get(key) ?? await this.directory.getFileHandle(key);
      file = new Uint8Array(await (await handle.getFile()).arrayBuffer());
    } catch {}
    const hit = this.index.hit(key, file?.length);
    this.changed();
    if (!hit || !file || file.length < RAW) return undefined;
    return { stored: file.subarray(RAW), raw: new DataView(file.buffer).getUint32(0, true) };
  }

  drop(key: string) {
    this.index.hit(key);
    this.changed();
  }

  keep(key: string, raw: number, stored: Promise<Uint8Array>) {
    if (this.holds(key)) return;
    const done = stored.then(stored => this.write(key, raw, stored), (error: unknown) => {
      console.warn(`A record was not kept: ${String(error)}`);
    }).finally(() => this.writing.delete(key));
    this.writing.set(key, done);
  }

  async sweep() {
    const names = this.directory as unknown as { keys(): AsyncIterable<string> };
    for await (const name of names.keys()) {
      if (name !== INDEX && !this.writing.has(name) && !this.index.lists(name)) {
        this.handles.delete(name);
        await this.directory.removeEntry(name).catch(() => undefined);
      }
    }
  }

  private async write(key: string, raw: number, stored: Uint8Array) {
    const head = new Uint8Array(RAW);
    new DataView(head.buffer).setUint32(0, raw, true);
    const file = new Blob([head, stored as Uint8Array<ArrayBuffer>]);
    if (!this.index.store(key, file.size)) return;
    this.changed();
    try {
      const handle = await this.directory.getFileHandle(key, { create: true });
      const writable = await handle.createWritable();
      await writable.write(file);
      await writable.close();
      this.handles.set(key, handle);
    } catch (error) {
      this.index.failed();
      console.warn(`The record store stopped keeping records: ${String(error)}`);
    }
  }

  changed() {
    for (const key of this.index.take_removed()) {
      this.handles.delete(key);
      void this.directory.removeEntry(key).catch(() => undefined);
    }
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

export class JobStore {
  private packs: Promise<Pack[]> | undefined;
  private packHits = 0;

  private constructor(private readonly bindings: IndexBindings, private readonly outputs: Shelf, private readonly records: Shelf,
    private readonly coder: Coder) {}

  static async open(bindings: IndexBindings, settings: JobCacheSettings, coder: Coder): Promise<JobStore | undefined> {
    try {
      const root = await navigator.storage.getDirectory();
      const estimate = await navigator.storage.estimate?.().catch(() => undefined);
      const available = estimate?.quota !== undefined && estimate.usage !== undefined ? Math.max(0, estimate.quota - estimate.usage) : undefined;
      void navigator.storage.persist?.().catch(() => false);
      const records = await Shelf.open(root, RECORDS, saved => new bindings.JobCacheIndex(settings.records, available, false, saved));
      const growth = Math.max(0, records.index.budget() - records.index.counters()[2]);
      const left = available === undefined ? undefined : Math.max(0, available - growth);
      const outputs = await Shelf.open(root, OUTPUTS, saved => new bindings.JobCacheIndex(settings.outputs, left, settings.verify, saved));
      const store = new JobStore(bindings, outputs, records, coder);
      if (records.index.counters()[2] === 0) void store.opened();
      return store;
    } catch (error) {
      console.warn(`The record store is off: ${String(error)}`);
      return undefined;
    }
  }

  get verify(): boolean {
    return this.outputs.index.verify();
  }

  counters(): [number, number, number, number] {
    const [hits, misses, bytes, mismatches] = this.outputs.index.counters();
    const [held, , mirrored] = this.records.index.counters();
    return [hits + held + this.packHits, misses, bytes + mirrored, mismatches];
  }

  lookup(keys: string[]): Promise<Uint8Array | undefined>[] {
    const remote = keys.filter(key => !this.records.holds(key) && !this.outputs.holds(key));
    const fetched = this.fetch(remote);
    const held = [...new Set(keys.filter(key => !fetched.has(key)))];
    held.sort((a, b) => this.size(b) - this.size(a));
    const local = new Map(held.map(key => [key, this.local(key)]));
    return keys.map(key => fetched.get(key) ?? local.get(key) ?? this.local(key));
  }

  keep(key: string, output: Uint8Array) {
    if (this.outputs.holds(key)) return;
    this.outputs.keep(key, output.length, this.coder.compress(output));
  }

  mismatch(key: string) {
    console.warn(`A kept record differs from a fresh run: ${key}`);
    this.outputs.index.mismatch(key);
    this.outputs.changed();
  }

  async sweep() {
    await this.records.sweep();
    await this.outputs.sweep();
  }

  private opened(): Promise<Pack[]> {
    return this.packs ??= openPacks(this.bindings);
  }

  private size(key: string): number {
    return Math.max(this.records.size(key), this.outputs.size(key));
  }

  private async local(key: string): Promise<Uint8Array | undefined> {
    for (const shelf of [this.records, this.outputs]) {
      if (!shelf.holds(key)) continue;
      const kept = await shelf.read(key);
      if (!kept) continue;
      try {
        return await this.coder.unpack(kept.stored, kept.raw);
      } catch (error) {
        console.warn(`A kept record does not read: ${String(error)}`);
        shelf.drop(key);
      }
    }
    return this.fetch([key]).get(key);
  }

  private fetch(keys: string[]): Map<string, Promise<Uint8Array | undefined>> {
    if (keys.length === 0) return new Map();
    const planned = this.opened().then(packs => this.plan(keys, packs));
    return new Map(keys.map(key => [key, planned.then(answers => answers.get(key))]));
  }

  private plan(keys: string[], packs: Pack[]): Map<string, Promise<Uint8Array | undefined>> {
    const answers = new Map<string, Promise<Uint8Array | undefined>>();
    const wanted: Wanted[] = [];
    for (const key of keys) {
      const found = packs.map(pack => ({ url: pack.url, entry: pack.directory.find(key) })).find(({ entry }) => entry);
      if (!found?.entry) {
        this.outputs.index.miss();
        answers.set(key, Promise.resolve(undefined));
        continue;
      }
      const [offset, stored, raw] = found.entry;
      wanted.push({ key, url: found.url, offset, stored, raw });
    }
    for (const span of spans(wanted)) {
      const start = span[0].offset;
      const last = span[span.length - 1];
      const read = fetchRange(last.url, start, last.offset + last.stored - start);
      for (const record of span) {
        answers.set(record.key, read.then(bytes => this.answer(record, bytes.subarray(record.offset - start, record.offset - start + record.stored)),
          (error: unknown) => {
            console.warn(`A record of ${record.url} does not read: ${String(error)}`);
            this.outputs.index.miss();
            return undefined;
          }));
      }
    }
    return answers;
  }

  private async answer(record: Wanted, stored: Uint8Array): Promise<Uint8Array | undefined> {
    try {
      const output = await this.coder.unpack(stored.slice(), record.raw);
      this.packHits++;
      this.records.keep(record.key, record.raw, Promise.resolve(stored));
      return output;
    } catch (error) {
      console.warn(`A record of ${record.url} does not read: ${String(error)}`);
      this.outputs.index.miss();
      return undefined;
    }
  }
}

function spans(wanted: Wanted[]): Wanted[][] {
  const sorted = [...wanted].sort((a, b) => a.url < b.url ? -1 : a.url > b.url ? 1 : a.offset - b.offset);
  const out: Wanted[][] = [];
  for (const record of sorted) {
    const span = out[out.length - 1];
    const last = span?.[span.length - 1];
    const gap = last ? record.offset - (last.offset + last.stored) : -1;
    if (last && last.url === record.url && gap >= 0 && gap <= Math.min(last.stored, record.stored)) span.push(record);
    else out.push([record]);
  }
  return out;
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

async function fetchRange(url: string, offset: number, length: number, again = true): Promise<Uint8Array> {
  let response: Response;
  try {
    response = await fetch(url, { cache: "no-store", headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  } catch (error) {
    if (again) return fetchRange(url, offset, length, false);
    throw error;
  }
  if (response.status !== 206) {
    void response.body?.cancel().catch(() => undefined);
    if (again && response.status !== 200) return fetchRange(url, offset, length, false);
    throw engineFailure({ code: "source_io", path: "$", got: { url, status: response.status } });
  }
  let bytes: Uint8Array;
  try {
    bytes = new Uint8Array(await response.arrayBuffer());
  } catch (error) {
    if (again) return fetchRange(url, offset, length, false);
    throw error;
  }
  if (bytes.length !== length) {
    throw engineFailure({ code: "source_io", path: "$", allowed: { kind: "range", min: length, max: length }, got: bytes.length });
  }
  return bytes;
}
