import { engineFailure } from "../generated/errors";
import { parseJson, type Runtime } from "../generated/runtime";

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

async function fetchPool(uris: string[], limit: number, work: (uri: string) => Promise<void>): Promise<void> {
  let next = 0;
  const workers = Array.from({ length: Math.min(limit, uris.length) }, async () => {
    while (next < uris.length) {
      const uri = uris[next];
      next += 1;
      await work(uri);
    }
  });
  await Promise.all(workers);
}

export async function feedSceneFiles(game: Runtime, sceneUrl: string, signal: AbortSignal, shown?: Promise<void>): Promise<void> {
  const base = new URL(sceneUrl, location.href);
  let waited = false;
  for (let files = parseJson(game.files()); files.length; files = parseJson(game.files())) {
    const groups = new Map<string, string[]>();
    for (const { uri, group } of files) {
      const uris = groups.get(group);
      if (uris) uris.push(uri);
      else groups.set(group, [uri]);
    }
    let index = 0;
    for (const uris of groups.values()) {
      if (index > 0 && shown && !waited) {
        waited = true;
        await shown;
        signal.throwIfAborted();
      }
      const limit = index === 0 && !waited ? 6 : 4;
      await fetchPool(uris, limit, async uri => {
        const read = await fetchBytes(new URL(uri, base).href, signal);
        signal.throwIfAborted();
        if (typeof read === "string") game.file_failed(uri, read);
        else game.file(uri, read);
      });
      index += 1;
    }
  }
}

async function fetchBytes(url: string, signal: AbortSignal): Promise<Uint8Array | string> {
  try {
    const response = await fetch(url, { signal });
    if (response.ok) return new Uint8Array(await response.arrayBuffer());
    await response.body?.cancel().catch(() => {});
    return JSON.stringify({ code: "missing_file", path: "$", got: { url, status: response.status } });
  } catch (error) {
    signal.throwIfAborted();
    return JSON.stringify({ code: "source_io", path: "$", got: { url, error: String(error) } });
  }
}
