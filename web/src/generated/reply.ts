import type { ErrorCode } from "./errors";

export interface Reply {
  error?: Error | null;
  frame: number;
  id?: unknown;
  ok: boolean;
  ready: Readiness;
  tick: number;
  value?: unknown;
}

export type Allowed = { kind: "range"; max: number; min: number } | { kind: "at_most"; max: number } | { kind: "at_least"; min: number } | { kind: "variants"; values: string[] };

export type Awaits = "work" | "view" | "finished" | "frames";

export interface Error {
  allowed?: Allowed | null;
  code: ErrorCode;
  expected?: Expected | null;
  got?: unknown;
  nearest?: unknown;
  path: string;
}

export type Expected = "number" | "integer" | "string" | "boolean" | "object" | "array" | "id" | "path" | "file" | "entity" | "unique" | "present" | "absent" | "finite" | "normalized" | "sorted" | "nonzero" | "above" | "power_of_two" | "multiple" | "json" | "utf8" | "uri" | "base64" | "image" | "paused" | "running" | "gpu" | "live" | "non_empty" | "readable" | "writable" | "matching" | "supported" | "acyclic" | "opaque" | "triangles" | "invertible";

export interface Progress {
  done: number;
  total: number;
}

export interface Readiness {
  compiling: number;
  drawing: number;
  failed: [string, Error][];
  finishing: number;
  loading: Progress;
  settling: number;
  unsettled: Unsettled[];
}

export interface Unsettled {
  awaits: Awaits;
  brick: string;
  left: number;
}
