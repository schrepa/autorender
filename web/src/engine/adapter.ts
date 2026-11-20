interface Gpu { requestAdapter(options?: { powerPreference?: string }): Promise<object | null> }

let adapter: Promise<object | null | undefined> | undefined;

export function warmAdapter(): void {
  adapter ??= (navigator as Navigator & { gpu?: Gpu }).gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
}

export async function adapterFound(): Promise<boolean> {
  warmAdapter();
  return Boolean(await adapter);
}
