import { engineFailure } from "../generated/errors";
import type { Fetched } from "./host";

export async function loadSceneSource(url: string, signal: AbortSignal): Promise<string> {
  const response = await fetch(url, { cache: "no-cache", signal });
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw engineFailure({ code: "source_io", path: "$", got: { url, status: response.status } });
  }
  const limit = 64 * 1024 * 1024;
  const declared = Number(response.headers.get("content-length"));
  if (declared > limit) {
    await response.body?.cancel().catch(() => {});
    throw engineFailure({ code: "source_too_large", path: "$", allowed: { kind: "at_most", max: limit }, got: declared });
  }
  if (!response.body) throw engineFailure({ code: "source_io", path: "$", expected: "present", got: { url } });
  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8", { fatal: true });
  let length = 0;
  let source = "";
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      length += value.byteLength;
      if (length > limit) throw engineFailure({ code: "source_too_large", path: "$", allowed: { kind: "at_most", max: limit }, got: length });
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
  read(uri: string, bytes: Uint8Array): void;
  finish(): Files;
  free(): void;
}

export interface SceneFileBindings<Files = unknown> {
  SceneLoad: new (source: string) => SceneLoad<Files>;
}

export async function loadSceneFiles<Files>(sceneUrl: string, source: string, bindings: SceneFileBindings<Files>, signal: AbortSignal, fetched?: (fetched: Fetched) => void): Promise<Files> {
  const load = new bindings.SceneLoad(source);
  const base = new URL(sceneUrl, location.href);
  try {
    const reads = load.take_reads();
    const files = { done: 0, total: reads.length };
    let bytes = 0;
    fetched?.({ files: { ...files }, bytes });
    await Promise.all(reads.map(async ({ uri }) => {
      const read = await fetchBytes(new URL(uri, base).href, signal);
      load.read(uri, read);
      files.done++;
      bytes += read.byteLength;
      fetched?.({ files: { ...files }, bytes });
    }));
    return load.finish();
  } finally {
    load.free();
  }
}

async function fetchBytes(url: string, signal: AbortSignal): Promise<Uint8Array> {
  const response = await fetch(url, { signal });
  if (!response.ok) {
    await response.body?.cancel().catch(() => {});
    throw engineFailure({ code: "missing_file", path: "$", got: { url, status: response.status } });
  }
  return new Uint8Array(await response.arrayBuffer());
}
