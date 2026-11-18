import type { Runtime } from "./host";
import { fatal } from "./panic";
import { engineError, engineFailure } from "../generated/errors";
import { parseJson } from "../generated/runtime";

interface Host {
  game: Runtime;
  signal: AbortSignal;
  source?: { load(signal: AbortSignal): Promise<string>; files(source: string, signal: AbortSignal): Promise<unknown> };
  wake(): void;
  fail(error: unknown): void;
}

export interface Development {
  pending(): boolean;
  turned(): void;
  close(reason: string): void;
}

declare global {
  interface Window {
    __engineDev?: { request(line: string): Promise<string> };
    __engineFile?: (file: string) => void;
  }
}

export function connectDevelopment(host: Host): Development {
  const { game, signal } = host;
  if (!game.dev_push || !game.dev_take || !game.dev_pixels || !game.dev_reloaded || !game.dev_reload_failed) {
    throw engineFailure({ code: "runtime_unavailable", path: "$", expected: "present", got: "dev_push" });
  }
  const waiting = new Map<number, { resolve(reply: string): void; reject(error: Error): void }>();
  let closed: string | undefined;
  let reloading = false;
  let behind = 0;
  let writing = Promise.resolve();
  const api = {
    request(line: string): Promise<string> {
      if (closed !== undefined) return Promise.reject(new Error(closed));
      return new Promise((resolve, reject) => {
        try {
          waiting.set(game.dev_push!(line), { resolve, reject });
        } catch (error) {
          if (fatal(error)) host.fail(error);
          reject(new Error(closed ?? String(error)));
          return;
        }
        host.wake();
      });
    },
  };
  function deliver(files: { run: string | null; name: string; base64: string }[], replies: [number, string][]) {
    const file = window.__engineFile;
    for (const written of files) {
      if (file) file(JSON.stringify(written));
      else console.error(JSON.stringify({ event: "file_dropped", name: written.name }));
    }
    for (const [ticket, reply] of replies) {
      waiting.get(ticket)?.resolve(reply);
      waiting.delete(ticket);
    }
  }
  async function reload() {
    reloading = true;
    try {
      if (!host.source) throw engineFailure({ code: "missing_resource", path: "$.scene", expected: "present" });
      const text = await host.source.load(signal);
      const files = await host.source.files(text, signal);
      if (closed === undefined) game.dev_reloaded!(text, files);
    } catch (cause) {
      if (fatal(cause)) { host.fail(cause); return; }
      if (closed === undefined) game.dev_reload_failed!(engineError(cause)?.message ?? JSON.stringify({ code: "browser_command_failed", path: "$", got: String(cause) }));
    } finally {
      reloading = false;
    }
    host.wake();
  }
  function close(reason: string) {
    if (closed !== undefined) return;
    closed = reason;
    for (const { reject } of waiting.values()) reject(new Error(reason));
    waiting.clear();
    if (window.__engineDev === api) delete window.__engineDev;
  }
  window.__engineDev = api;
  signal.addEventListener("abort", () => close("session_closed"), { once: true });
  return {
    pending: () => waiting.size > 0,
    turned() {
      const taken = parseJson(game.dev_take!());
      if (taken.reload && !reloading) void reload();
      if (!taken.pictures.length && !behind) {
        deliver([], taken.replies);
        return;
      }
      const pixels = taken.pictures.length ? game.dev_pixels!() : new Uint8Array();
      let at = 0;
      const encoded = Promise.all(taken.pictures.map(({ run, file, width, height }) => {
        const rgba = pixels.subarray(at, at += width * height * 4);
        return png(width, height, rgba).then(base64 => ({ run, name: file, base64 }));
      }));
      behind++;
      writing = writing
        .then(async () => deliver(await encoded, taken.replies))
        .catch(host.fail)
        .finally(() => { behind--; });
    },
    close,
  };
}

const SIGNATURE = new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]);
const CRC_TABLE = Uint32Array.from({ length: 256 }, (_, n) => {
  let c = n;
  for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
  return c >>> 0;
});

function chunk(type: string, data: Uint8Array): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(12 + data.length);
  const view = new DataView(out.buffer);
  view.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  let crc = ~0;
  for (const byte of out.subarray(4, 8 + data.length)) crc = CRC_TABLE[(crc ^ byte) & 255] ^ (crc >>> 8);
  view.setUint32(8 + data.length, ~crc >>> 0);
  return out;
}

async function png(width: number, height: number, rgba: Uint8Array): Promise<string> {
  const stride = width * 4;
  const rows = new Uint8Array((stride + 1) * height);
  for (let y = 0; y < height; y++) rows.set(rgba.subarray(y * stride, (y + 1) * stride), y * (stride + 1) + 1);
  const deflated = new Uint8Array(await new Response(new Blob([rows]).stream().pipeThrough(new CompressionStream("deflate"))).arrayBuffer());
  const header = new Uint8Array(13);
  const view = new DataView(header.buffer);
  view.setUint32(0, width);
  view.setUint32(4, height);
  header.set([8, 6, 0, 0, 0], 8);
  const file = new Blob([SIGNATURE, chunk("IHDR", header), chunk("IDAT", deflated), chunk("IEND", new Uint8Array())]);
  const url = await new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result as string);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
  return url.slice(url.indexOf(",") + 1);
}
