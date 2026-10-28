import { jobPool } from "./jobs.ts";

export async function loadSceneSource(url: string, signal: AbortSignal): Promise<string> {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw new Error(`Scene fetch failed: HTTP ${response.status}`);
  }
  const limit = 64 * 1024 * 1024;
  const declared = Number(response.headers.get("content-length"));
  if (declared > limit) {
    await response.body?.cancel().catch(() => {});
    throw new Error("Scene exceeds the 64 MiB source budget.");
  }
  if (!response.body) throw new Error("Scene response has no body.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let length = 0;
  let source = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit) throw new Error("Scene exceeds the 64 MiB source budget.");
      source += decoder.decode(value, { stream: true });
    }
    return source + decoder.decode();
  } catch (error) {
    await reader.cancel().catch(() => {});
    throw error;
  } finally { reader.releaseLock(); }
}

interface SceneLoad<Files> {
  take_reads(): { uri: string; prefix?: number }[];
  read(uri: string, bytes: Uint8Array, url: string): boolean;
  image(uri: string, width: number, height: number, pixels: Uint8Array): void;
  decode(uri: string, bytes: Uint8Array): void;
  finish(): Files;
  free(): void;
}

interface SceneFileBindings<Files> {
  SceneLoad: new (source: string) => SceneLoad<Files>;
}

export async function loadSceneFiles<Files>(sceneUrl: string, source: string, bindings: SceneFileBindings<Files>, signal: AbortSignal): Promise<Files> {
  const load = new bindings.SceneLoad(source);
  const base = new URL(sceneUrl, location.href);
  const pool = jobPool();
  const file = async ({ uri, prefix }: { uri: string; prefix?: number }) => {
    const url = new URL(uri, base).href;
    const bytes = await fetchBytes(url, prefix, signal);
    if (load.read(uri, bytes, url)) {
      const image = pool && await pool.decodeImage(uri, bytes.slice()).catch(() => undefined);
      if (image) load.image(uri, image.width, image.height, image.pixels);
      else load.decode(uri, bytes);
    }
  };
  try {
    for (let reads = load.take_reads(); reads.length > 0; reads = load.take_reads()) {
      await Promise.all(reads.map(file));
    }
    return load.finish();
  } finally {
    load.free();
  }
}

export async function fetchBytes(url: string, prefix: number | undefined, signal?: AbortSignal): Promise<Uint8Array> {
  const headers: HeadersInit = prefix === undefined ? {} : { Range: `bytes=0-${prefix - 1}` };
  const response = await fetch(url, { headers, cache: "no-store", signal });
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw new Error(`Scene file ${url} fetch failed: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}
