/** The features of the GPU adapter the engine requests, asked once. */
interface GpuAdapter { features: ReadonlySet<string> }
interface Gpu { requestAdapter(options?: { powerPreference?: string }): Promise<GpuAdapter | null> }

let features: Promise<ReadonlySet<string>> | undefined;

/** The adapter's features; none without WebGPU or an adapter. */
export function adapterFeatures(): Promise<ReadonlySet<string>> {
  features ??= (async () => {
    const gpu = (navigator as Navigator & { gpu?: Gpu }).gpu;
    const adapter = await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
    return adapter?.features ?? new Set<string>();
  })();
  return features;
}
