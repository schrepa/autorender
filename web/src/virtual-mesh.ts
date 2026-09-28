/** Virtual mesh files (decision 044) as hosts load them. */

/**
 * The prefix of the virtual mesh file at `url` (decision 044): its header
 * says how long (`prefixBytes`), then that byte range follows. A server
 * ignoring ranges sends the whole file, which serves as it is.
 */
export async function fetchVirtualPrefix(url: string, prefixBytes: (header: Uint8Array) => number | undefined, signal?: AbortSignal) {
  const range = async (bytes: number) => {
    const response = await fetch(url, { headers: { Range: `bytes=0-${bytes - 1}` }, cache: "no-store", signal });
    if (!response.ok) {
      await response.body?.cancel().catch(() => {});
      throw new Error(`Virtual mesh ${url} fetch failed: HTTP ${response.status}`);
    }
    return new Uint8Array(await response.arrayBuffer());
  };
  const head = await range(128);
  const length = prefixBytes(head.subarray(0, 128));
  if (length === undefined) throw new Error(`${url} is not a virtual mesh file of this engine version.`);
  return length <= head.length ? head.slice(0, length) : await range(length);
}
