import { decodeImagePixels } from "./decode.ts";
import { jobPool } from "./jobs.ts";
import { basisBase, isKtx2, targetFormat, transcodeKtx2 } from "./ktx2.ts";
import { fetchPrefix, fetchVirtualPrefix } from "./virtual-mesh.ts";

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

/** The generated bindings a scene's file loader needs. */
export interface SceneFileBindings {
  scene_file_list(source: string): string;
  virtual_prefix_bytes(header: Uint8Array): number | undefined;
  terrain_prefix_bytes(header: Uint8Array): number | undefined;
  SceneFileSet: new () => {
    add_image(uri: string, width: number, height: number, pixels: Uint8Array): void;
    add_compressed_image(uri: string, format: string, width: number, height: number, data: Uint8Array, levels: Uint32Array): void;
    add_buffer(uri: string, bytes: Uint8Array): void;
    add_virtual_mesh(uri: string, bytes: Uint8Array, url: string | undefined): void;
    add_terrain(uri: string, bytes: Uint8Array, url: string | undefined): void;
    free(): void;
  };
}


/**
 * Fetch the image and buffer files a scene references, relative to its URL,
 * decode images with the browser's own decoder into RGBA8 pixels, and
 * transcode KTX2 images to a block-compressed format the GPU samples.
 * Runs at load boundaries only, never per frame: on the
 * page's job workers where it runs them, several at once,
 * else on the page.
 */
export async function loadSceneFiles(sceneUrl: string, source: string, bindings: SceneFileBindings, signal: AbortSignal) {
  const list = JSON.parse(bindings.scene_file_list(source)) as { uri: string; kind: "image" | "buffer" | "virtual_mesh" | "terrain" }[];
  const files = new bindings.SceneFileSet();
  const base = new URL(sceneUrl, location.href);
  const pool = jobPool();
  try {
    await Promise.all(list.map(async ({ uri, kind }) => {
      // Virtual meshes load their prefix; the renderer
      // streams their other pages from the same URL as views need them.
      if (kind === "virtual_mesh") {
        const url = new URL(uri, base).href;
        files.add_virtual_mesh(uri, await fetchVirtualPrefix(url, bindings.virtual_prefix_bytes, signal), url);
        return;
      }
      // Terrain files load their header and directory; the
      // engine fetches tiles by byte range as the camera needs them.
      if (kind === "terrain") {
        const url = new URL(uri, base).href;
        files.add_terrain(uri, await fetchPrefix(url, 64, bindings.terrain_prefix_bytes, "terrain", signal), url);
        return;
      }
      const response = await fetch(new URL(uri, base), { cache: "no-store", signal });
      if (!response.ok) {
        await response.body?.cancel().catch(() => {});
        throw new Error(`Scene file ${uri} fetch failed: HTTP ${response.status}`);
      }
      const blob = await response.blob();
      if (kind === "buffer") {
        files.add_buffer(uri, new Uint8Array(await blob.arrayBuffer()));
        return;
      }
      const head = new Uint8Array(await blob.slice(0, 12).arrayBuffer());
      if (isKtx2(head)) {
        const bytes = new Uint8Array(await blob.arrayBuffer());
        // A worker that fails leaves the image to the page.
        const image = (pool && await pool.transcodeKtx2(uri, bytes.slice(), await targetFormat(), basisBase()).catch(() => undefined))
          || await transcodeKtx2(uri, bytes);
        if (image.format === "rgba8") files.add_image(uri, image.width, image.height, image.pixels);
        else files.add_compressed_image(uri, image.format, image.width, image.height, image.data, image.levels);
        return;
      }
      const image = (pool && await pool.decodeImage(uri, blob).catch(() => undefined)) || await decodeImagePixels(uri, blob);
      files.add_image(uri, image.width, image.height, image.pixels);
    }));
    return files;
  } catch (error) {
    files.free();
    throw error;
  }
}
