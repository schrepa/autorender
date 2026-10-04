/** Streamed files as hosts load them: virtual meshes (decision 044) and terrain files (decision 051). */

/**
 * The prefix of the file at `url` whose first `headerBytes` bytes say how
 * long it is (`prefixBytes`), then that byte range; `what` names the file
 * in errors. A server ignoring ranges sends the whole file, which serves as
 * it is.
 */
export async function fetchPrefix(url: string, headerBytes: number, prefixBytes: (header: Uint8Array) => number | undefined,
  what: string, signal?: AbortSignal) {
  const range = async (bytes: number) => {
    const response = await fetch(url, { headers: { Range: `bytes=0-${bytes - 1}` }, cache: "no-store", signal });
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      throw new Error(`${what} ${url} fetch failed: HTTP ${response.status}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  };
  const head = await range(headerBytes);
  const length = prefixBytes(head.subarray(0, headerBytes));
  if (length === undefined) throw new Error(`${url} is not a ${what} file of this engine version.`);
  return length <= head.length ? head.slice(0, length) : await range(length);
}

/** The prefix of the virtual mesh file at `url` (decision 044). */
export function fetchVirtualPrefix(url: string, prefixBytes: (header: Uint8Array) => number | undefined, signal?: AbortSignal) {
  return fetchPrefix(url, 128, prefixBytes, "virtual mesh", signal);
}
