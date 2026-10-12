/** How the page and its workers know the engine's module stopped: the
 * engine's panic hook keeps a panic's report on the global object and
 * throws it out of the module (Rust's `engine_panic`), and a module that
 * traps without one (a stack overflow) throws a `RuntimeError`. Either way
 * the module is unusable, and nothing calls into it again. Imported by
 * workers too: nothing here touches the page. */
import { PANIC_GLOBAL, PANIC_PREFIX, type PanicReport } from "../generated/host";

export type { PanicReport };

/** The report of the engine's panic `error` carries, else the one this
 * page or worker recorded; none while the engine has not panicked. */
export function panicReport(error?: unknown): PanicReport | undefined {
  const thrown = error instanceof Error && error.message.startsWith(PANIC_PREFIX) ? error.message.slice(PANIC_PREFIX.length) : undefined;
  const recorded = thrown ?? (globalThis as Record<string, unknown>)[PANIC_GLOBAL];
  return typeof recorded === "string" ? JSON.parse(recorded) as PanicReport : undefined;
}

/** Whether `error` means the module stopped: the engine panicked, or the
 * module trapped. */
export function fatal(error: unknown): boolean {
  return error instanceof WebAssembly.RuntimeError || panicReport(error) !== undefined;
}

/** What a stopped engine's development replies say: its panic's message,
 * or the trap's. */
export function stoppedMessage(error: unknown): string {
  return panicReport(error)?.message ?? String(error);
}
