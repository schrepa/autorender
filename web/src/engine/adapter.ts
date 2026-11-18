interface Gpu { requestAdapter(options?: { powerPreference?: string }): Promise<unknown> }

let adapter: Promise<unknown> | undefined;

export function warmAdapter(): void {
  adapter ??= (navigator as Navigator & { gpu?: Gpu }).gpu?.requestAdapter({ powerPreference: "high-performance" }).catch(() => null);
}
