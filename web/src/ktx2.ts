// KTX2 images (decision 037): Basis Universal supercompressed textures,
// transcoded at scene load (never per frame) to the block-compressed format
// this device samples, or to RGBA8 without one. The transcoder is the
// vendored upstream build in public/basis, loaded on first use.

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
interface GpuAdapter { features: ReadonlySet<string> }
interface Gpu { requestAdapter(options?: { powerPreference?: string }): Promise<GpuAdapter | null> }

const KTX2_MAGIC = [0xab, 0x4b, 0x54, 0x58, 0x20, 0x32, 0x30, 0xbb, 0x0d, 0x0a, 0x1a, 0x0a];

/** Whether `bytes` start with the KTX2 identifier. */
export function isKtx2(bytes: Uint8Array): boolean {
  return bytes.length >= KTX2_MAGIC.length && KTX2_MAGIC.every((value, index) => bytes[index] === value);
}

let transcoder: Promise<BasisModule> | undefined;
let target: Promise<TranscodedImage["format"]> | undefined;

function loadTranscoder(): Promise<BasisModule> {
  transcoder ??= (async () => {
    const basisBase = `${import.meta.env?.BASE_URL ?? "/"}basis/`;
    // A classic script: it defines the global BASIS factory.
    await new Promise<void>((resolve, reject) => {
      const script = document.createElement("script");
      script.src = `${basisBase}basis_transcoder.js`;
      script.onload = () => resolve();
      script.onerror = () => reject(new Error("The KTX2 transcoder could not be loaded."));
      document.head.appendChild(script);
    });
    const factory = (globalThis as { BASIS?: BasisFactory }).BASIS;
    if (!factory) throw new Error("The KTX2 transcoder did not initialize.");
    const module = await factory({ locateFile: (path) => `${basisBase}${path}` });
    module.initializeBasis();
    return module;
  })();
  transcoder.catch(() => { transcoder = undefined; });
  return transcoder;
}

// The adapter the engine requests: its first block-compressed format.
function targetFormat(): Promise<TranscodedImage["format"]> {
  target ??= (async () => {
    const gpu = (navigator as Navigator & { gpu?: Gpu }).gpu;
    const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" });
    const features = adapter?.features ?? new Set<string>();
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
  const [module, preferred] = await Promise.all([loadTranscoder(), targetFormat()]);
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
