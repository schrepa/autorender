import type { ErrorCode } from "../generated/errors";

export interface SmokeFailure {
  smoke: string;
  check: string;
  got: unknown;
  expected: unknown;
  at: string;
}

export interface SmokeRecord {
  smoke: string;
  brick: string;
  ok: boolean;
  ms: number;
  skipped?: boolean;
  failure?: SmokeFailure;
}

export interface SmokeState {
  records: SmokeRecord[];
  done: boolean;
  passed: boolean;
  error?: { code: ErrorCode | string; path: string };
}

export type SmokeRun = (canvas: HTMLCanvasElement, names: string, report: (line: string) => void) => Promise<boolean>;

declare global {
  interface Window {
    __engineSmokes?: SmokeState;
  }
}

const SIZE = [960, 540] as const;

export function smokeSelection(): string | null {
  return new URLSearchParams(location.search).get("smoke");
}

function thrown(error: unknown): { code: ErrorCode | string; path: string } {
  const fields = (error ?? {}) as { code?: unknown; path?: unknown };
  return typeof fields.code === "string"
    ? { code: fields.code, path: typeof fields.path === "string" ? fields.path : "$" }
    : { code: "browser_command_failed", path: "?smoke" };
}

export async function runSmokes(names: string, load: () => Promise<SmokeRun | undefined>): Promise<void> {
  const state: SmokeState = { records: [], done: false, passed: false };
  window.__engineSmokes = state;
  const canvas = document.createElement("canvas");
  [canvas.width, canvas.height] = SIZE;
  const log = document.createElement("pre");
  log.id = "smokes";
  document.body.append(canvas, log);
  try {
    const run = await load();
    if (run) {
      state.passed = await run(canvas, names, line => {
        state.records.push(JSON.parse(line) as SmokeRecord);
        log.append(`${line}\n`);
      });
    } else {
      state.error = { code: "unsupported_command", path: "?smoke" };
    }
  } catch (error) {
    state.error = thrown(error);
  } finally {
    state.done = true;
    document.body.dataset.smokes = state.error ? "error" : state.passed ? "passed" : "failed";
    if (state.error) log.append(`${JSON.stringify(state.error)}\n`);
  }
}
