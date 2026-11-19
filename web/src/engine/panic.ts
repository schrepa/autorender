import { PANIC_GLOBAL, PANIC_PREFIX, type PanicReport } from "../generated/host";

export function panicReport(error?: unknown): PanicReport | undefined {
  const thrown = error instanceof Error && error.message.startsWith(PANIC_PREFIX) ? error.message.slice(PANIC_PREFIX.length) : undefined;
  const recorded = thrown ?? (globalThis as Record<string, unknown>)[PANIC_GLOBAL];
  return typeof recorded === "string" ? JSON.parse(recorded) as PanicReport : undefined;
}

export function fatal(error: unknown): boolean {
  return error instanceof WebAssembly.RuntimeError || panicReport(error) !== undefined;
}
