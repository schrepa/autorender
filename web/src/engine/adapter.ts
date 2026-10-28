interface AdapterInfo { vendor: string; architecture: string; device: string; description: string }
interface GpuAdapter { info?: AdapterInfo }
interface Gpu { requestAdapter(options?: { powerPreference?: string }): Promise<GpuAdapter | null> }

let adapter: Promise<GpuAdapter | null> | undefined;
function requested(): Promise<GpuAdapter | null> {
  adapter ??= (async () => {
    const gpu = (navigator as Navigator & { gpu?: Gpu }).gpu;
    return await gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null) ?? null;
  })();
  return adapter;
}

export async function adapterInfo(): Promise<AdapterInfo | null> {
  const info = (await requested())?.info;
  return info ? { vendor: info.vendor, architecture: info.architecture, device: info.device, description: info.description } : null;
}
