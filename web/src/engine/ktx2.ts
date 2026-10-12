// KTX2 images: Basis Universal supercompressed textures,
// transcoded at scene load (never per frame) to the block-compressed format
// this device samples, or to RGBA8 without one. Uncompressed RGBA8 KTX2
// images are the engine's decoder's (IMAGE_FORMATS). The transcoder is the
// vendored upstream build in public/basis, loaded on first use.

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
interface BasisModule {
  initializeBasis(): void;
  KTX2File: new (bytes: Uint8Array) => KTX2File;
  transcoder_texture_format: Record<string, BasisEnum>;
}
type BasisFactory = (options: { locateFile(path: string): string }) => Promise<BasisModule>;

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

/** The transcoder in a job worker, which has no document to
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

/** Transcodes one 2D Basis KTX2 image `uri` (for messages) at a load
 * boundary. */
export async function transcodeKtx2(uri: string, bytes: Uint8Array): Promise<TranscodedImage> {
  const [module, preferred] = await Promise.all([loadTranscoder(), targetFormat()]);
  return transcodeWith(module, uri, bytes, preferred);
}

/** Transcodes Basis KTX2 image `uri` with `module` to `preferred`, or to
 * RGBA8 where its base level is not whole blocks. */
export function transcodeWith(module: BasisModule, uri: string, bytes: Uint8Array, preferred: TranscodedImage["format"]): TranscodedImage {
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
