import { jobPool } from "./jobs.ts";
import { basisBase, targetFormat, transcodeKtx2 } from "./ktx2.ts";

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
  take_reads(): { uri: string }[];
  read(uri: string, bytes: Uint8Array): string | undefined;
  image(uri: string, width: number, height: number, pixels: Uint8Array): void;
  decode(uri: string, bytes: Uint8Array): void;
  compressed_image(uri: string, format: string, width: number, height: number, data: Uint8Array, levels: Uint32Array): void;
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
  const file = async ({ uri }: { uri: string }) => {
    const url = new URL(uri, base).href;
    const bytes = await fetchBytes(url, signal);
    const decoder = load.read(uri, bytes);
    if (decoder === "engine") {
      const image = pool && await pool.decodeImage(uri, bytes.slice()).catch(() => undefined);
      if (image) load.image(uri, image.width, image.height, image.pixels);
      else load.decode(uri, bytes);
    } else if (decoder === "transcoder") {
      const image = (pool && await pool.transcodeKtx2(uri, bytes.slice(), await targetFormat(), basisBase()).catch(() => undefined))
        || await transcodeKtx2(uri, bytes);
      if (image.format === "rgba8") load.image(uri, image.width, image.height, image.pixels);
      else load.compressed_image(uri, image.format, image.width, image.height, image.data, image.levels);
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

export async function fetchBytes(url: string, signal?: AbortSignal): Promise<Uint8Array> {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw new Error(`Scene file ${url} fetch failed: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}
