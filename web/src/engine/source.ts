import { decodeImagePixels } from "./decode.ts";
import { jobPool } from "./jobs.ts";
import { basisBase, targetFormat, transcodeKtx2 } from "./ktx2.ts";

/** Read a UTF-8 authoring source with a hard transport budget, including chunked replies. */
export async function loadSceneSource(url: string, signal: AbortSignal): Promise<string> {
  const response = await fetch(url, { cache: "no-store", signal });
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw new Error(`Scene fetch failed: HTTP ${response.status}`);
  }
  // A sanity bound on JSON; bulk pixels and geometry are separate files.
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

/** A scene's files loading (the engine's `SceneLoad`), as the game's
 * bindings make it, ending in the game's `Files`. */
interface SceneLoad<Files> {
  take_reads(): { uri: string; prefix?: number }[];
  /** For an image, who decodes it here (`IMAGE_FORMATS`): `engine`,
   * `browser` or `transcoder`. */
  read(uri: string, bytes: Uint8Array, url: string): string | undefined;
  image(uri: string, width: number, height: number, pixels: Uint8Array): void;
  decode(uri: string, bytes: Uint8Array): void;
  compressed_image(uri: string, format: string, width: number, height: number, data: Uint8Array, levels: Uint32Array): void;
  finish(): Files;
  free(): void;
}

/** The generated bindings a scene's file loader needs. */
interface SceneFileBindings<Files> {
  SceneLoad: new (source: string) => SceneLoad<Files>;
}

/**
 * The files a scene references, as the engine loads them: the page
 * fetches what it asks, relative to the scene's URL (files whole, and
 * streamed files' prefixes by byte range), and decodes each image as
 * `IMAGE_FORMATS` says: with the engine's decoder, the browser's on the
 * page's job workers where it runs them, or its Basis transcoder. Runs at
 * load boundaries only, never per frame; files load several at once.
 */
export async function loadSceneFiles<Files>(sceneUrl: string, source: string, bindings: SceneFileBindings<Files>, signal: AbortSignal): Promise<Files> {
  const load = new bindings.SceneLoad(source);
  const base = new URL(sceneUrl, location.href);
  const pool = jobPool();
  const file = async ({ uri, prefix }: { uri: string; prefix?: number }) => {
    const url = new URL(uri, base).href;
    const bytes = await fetchBytes(url, prefix, signal);
    const decoder = load.read(uri, bytes, url);
    if (decoder === "engine") {
      load.decode(uri, bytes);
    } else if (decoder === "browser") {
      // A worker that fails leaves the image to the page.
      const blob = new Blob([bytes as Uint8Array<ArrayBuffer>]);
      const image = (pool && await pool.decodeImage(uri, blob).catch(() => undefined)) || await decodeImagePixels(uri, blob);
      load.image(uri, image.width, image.height, image.pixels);
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

/** File `url`'s first `prefix` bytes by a range request, or the whole
 * file: a server ignoring ranges sends the whole file, which serves too. */
export async function fetchBytes(url: string, prefix: number | undefined, signal?: AbortSignal): Promise<Uint8Array> {
  const headers: HeadersInit = prefix === undefined ? {} : { Range: `bytes=0-${prefix - 1}` };
  const response = await fetch(url, { headers, cache: "no-store", signal });
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw new Error(`Scene file ${url} fetch failed: HTTP ${response.status}`);
  }
  return new Uint8Array(await response.arrayBuffer());
}
