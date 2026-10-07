// KTX2 images (decision 037): Basis Universal supercompressed textures,
// transcoded at scene load (never per frame) to the block-compressed format
// this device samples, or to RGBA8 without one, and uncompressed RGBA8
// images read as they are, alpha exact (terrain materials keep height in
// it, decision 054). The transcoder is the vendored upstream build in
// public/basis, loaded on first use.

import { adapterFeatures } from "./adapter.ts";

/** A transcoded KTX2 image: every level of a compressed format, or RGBA8. */
export type TranscodedImage =
  | { format: "bc7" | "etc2" | "astc"; width: number; height: number; data: Uint8Array; levels: Uint32Array }
  | { format: "rgba8"; width: number; height: number; pixels: Uint8Array };

// The parts of the transcoder's embind API used here.
interface BasisEnum { value: number }
interface KTX2File {
  isValid(): boolean;
  getWidth(): number;
  getHeight(): number;
  getLevels(): number;
  getLayers(): number;
  getFaces(): number;
  startTranscoding(): number | boolean;
  getImageTranscodedSizeInBytes(level: number, layer: number, face: number, format: number): number;
  transcodeImage(dst: Uint8Array, level: number, layer: number, face: number, format: number, alphaForOpaque: number, channel0: number, channel1: number): number | boolean;
  close(): void;
  delete(): void;
}
export interface BasisModule {
  initializeBasis(): void;
  KTX2File: new (bytes: Uint8Array) => KTX2File;
  transcoder_texture_format: Record<string, BasisEnum>;
}
type BasisFactory = (options: { locateFile(path: string): string }) => Promise<BasisModule>;

const KTX2_MAGIC = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];

/** Whether `bytes` start with the KTX2 identifier. */
export function isKtx2(bytes: Uint8Array): boolean {
  return bytes.length >= KTX2_MAGIC.length && KTX2_MAGIC.every((value, index) => bytes[index] === value);
}

// Uncompressed RGBA8 formats: VK_FORMAT_R8G8B8A8_UNORM and _SRGB. Whether
// a map decodes sRGB is the material's choice, not the file's.
const RGBA8_FORMATS = [37, 43];

/**
 * The pixels of an uncompressed RGBA8 KTX2 image's base level, or
 * `undefined` for any other KTX2 image; `uri` names it in messages.
 */
export function readRgba8Ktx2(uri: string, bytes: Uint8Array): { width: number; height: number; pixels: Uint8Array } | undefined {
  if (bytes.byteLength < 104) throw new Error(`Scene file ${uri} is not a valid KTX2 image.`);
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const word = (at: number) => view.getUint32(at, true);
  if (!RGBA8_FORMATS.includes(word(12)) || word(44) !== 0) return undefined;
  const width = word(20), height = word(24);
  if (word(28) > 1 || word(32) > 1 || word(36) !== 1 || width === 0 || height === 0) {
    throw new Error(`Scene file ${uri} must be a single 2D image.`);
  }
  // The base level's entry leads the level index.
  const offset = word(80), length = word(88);
  if (word(84) !== 0 || word(92) !== 0 || length !== width * height * 4 || offset + length > bytes.byteLength) {
    throw new Error(`Scene file ${uri} has an invalid base level.`);
  }
  return { width, height, pixels: bytes.slice(offset, offset + length) };
}

let transcoder: Promise<BasisModule> | undefined;
let target: Promise<TranscodedImage["format"]> | undefined;

/** Where the vendored transcoder's files are, as an absolute URL. */
export function basisBase(): string {
  return new URL(`${import.meta.env?.BASE_URL ?? "/"}basis/`, location.href).href;
}

/** The transcoder `factory` makes, its files under `base`. */
async function startTranscoder(factory: BasisFactory | undefined, base: string): Promise<BasisModule> {
  if (!factory) throw new Error("The KTX2 transcoder did not initialize.");
  const module = await factory({ locateFile: (path) => `${base}${path}` });
  module.initializeBasis();
  return module;
}

function loadTranscoder(): Promise<BasisModule> {
  transcoder ??= (async () => {
    const base = basisBase();
    // A classic script: it defines the global BASIS factory.
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `${base}basis_transcoder.js`;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("The KTX2 transcoder could not be loaded."));
      document.head.appendChild(script);
    });
    return startTranscoder((globalThis as { BASIS?: BasisFactory }).BASIS, base);
  })();
  transcoder.catch(() => { transcoder = undefined; });
  return transcoder;
}

/** The transcoder in a job worker (decision 110), which has no document to
 * add its classic script to: its source runs as a function returning the
 * factory it defines. */
export function workerTranscoder(base: string): Promise<BasisModule> {
  transcoder ??= (async () => {
    const response = await fetch(`${base}basis_transcoder.js`);
    if (!response.ok) throw new Error("The KTX2 transcoder could not be loaded.");
    const factory = new Function(`${await response.text()}\nreturn BASIS;`)() as BasisFactory;
    return startTranscoder(factory, base);
  })();
  transcoder.catch(() => { transcoder = undefined; });
  return transcoder;
}

// The adapter the engine requests: its first block-compressed format.
export function targetFormat(): Promise<TranscodedImage["format"]> {
  target ??= (async () => {
    const features = await adapterFeatures();
    if (features.has("texture-compression-bc")) return "bc7";
    if (features.has("texture-compression-astc")) return "astc";
    if (features.has("texture-compression-etc2")) return "etc2";
    return "rgba8";
  })();
  return target;
}

const TRANSCODER_FORMATS = { bc7: "cTFBC7_RGBA", astc: "cTFASTC_4x4_RGBA", etc2: "cTFETC2_RGBA", rgba8: "cTFRGBA32" } as const;

/** Transcodes one 2D KTX2 image `uri` (for messages) at a load boundary. */
export async function transcodeKtx2(uri: string, bytes: Uint8Array): Promise<TranscodedImage> {
  const plain = readRgba8Ktx2(uri, bytes);
  if (plain) return { format: "rgba8", ...plain };
  const [module, preferred] = await Promise.all([loadTranscoder(), targetFormat()]);
  return transcodeWith(module, uri, bytes, preferred);
}

/** Transcodes KTX2 image `uri` with `module` to `preferred`, or to RGBA8
 * where its base level is not whole blocks; uncompressed RGBA8 images read
 * as they are. */
export function transcodeWith(module: BasisModule, uri: string, bytes: Uint8Array, preferred: TranscodedImage["format"]): TranscodedImage {
  const plain = readRgba8Ktx2(uri, bytes);
  if (plain) return { format: "rgba8", ...plain };
  const file = new module.KTX2File(bytes);
  try {
    if (!file.isValid()) throw new Error(`Scene file ${uri} is not a valid KTX2 image.`);
    const width = file.getWidth();
    const height = file.getHeight();
    if (file.getLayers() > 1 || file.getFaces() > 1) throw new Error(`Scene file ${uri} must be a single 2D image.`);
    if (!file.startTranscoding()) throw new Error(`Scene file ${uri} could not be transcoded.`);
    // Block formats need whole blocks at the base level.
    const format = width % 4 === 0 && height % 4 === 0 ? preferred : "rgba8";
    const code = module.transcoder_texture_format[TRANSCODER_FORMATS[format]].value;
    // RGBA8 textures take their base level; the renderer makes the rest.
    const count = format === "rgba8" ? 1 : Math.max(1, file.getLevels());
    const levels = new Uint32Array(count);
    const parts: Uint8Array[] = [];
    for (let level = 0; level < count; level++) {
      const size = file.getImageTranscodedSizeInBytes(level, 0, 0, code);
      const part = new Uint8Array(size);
      if (!file.transcodeImage(part, level, 0, 0, code, 0, -1, -1)) throw new Error(`Scene file ${uri} level ${level} could not be transcoded.`);
      parts.push(part);
      levels[level] = size;
    }
    if (format === "rgba8") return { format, width, height, pixels: parts[0] };
    const data = new Uint8Array(parts.reduce((total, part) => total + part.byteLength, 0));
    let offset = 0;
    for (const part of parts) {
      data.set(part, offset);
      offset += part.byteLength;
    }
    return { format, width, height, data, levels };
  } finally {
    file.close();
    file.delete();
  }
}
