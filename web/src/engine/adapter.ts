interface Adapter { features: ReadonlySet<string> }
interface Gpu { requestAdapter(options?: { powerPreference?: string }): Promise<Adapter | null> }

let adapter: Promise<Adapter | null | undefined> | undefined;

export function warmAdapter(): void {
  adapter ??= (navigator as Navigator & { gpu?: Gpu }).gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
}

export async function adapterFeatures(): Promise<string[]> {
  warmAdapter();
  const found = await adapter;
  if (!found) throw new Error("no WebGPU adapter: its features cannot be read");
  return [...found.features];
}
