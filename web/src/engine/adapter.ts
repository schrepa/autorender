interface AdapterInfo { vendor: string; architecture: string; device: string; description: string }
interface GpuAdapter { features: ReadonlySet<string>; info?: AdapterInfo }
interface Gpu { requestAdapter(options?: { powerPreference?: string }): Promise<GpuAdapter | null> }

let adapter: Promise<GpuAdapter | null> | undefined;
function requested(): Promise<GpuAdapter | null> {
  adapter ??= (async () => {
    const gpu = (navigator as Navigator & { gpu?: Gpu }).gpu;
    return await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null) ?? null;
  })();
  return adapter;
}

export async function adapterFeatures(): Promise<ReadonlySet<string>> {
  return (await requested())?.features ?? new Set<string>();
}

export async function adapterInfo(): Promise<AdapterInfo | null> {
  const info = (await requested())?.info;
  return info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : null;
}
