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

export interface Error {
  allowed?: Allowed | null;
  code: ErrorCode;
  expected?: Expected | null;
  got?: unknown;
  nearest?: unknown;
  path: string;
}

export type Expected = "number" | "integer" | "string" | "boolean" | "object" | "array" | "id" | "file" | "entity" | "unique" | "present" | "absent" | "finite" | "sorted" | "nonzero" | "above" | "multiple" | "json" | "utf8" | "uri" | "base64" | "image" | "gpu" | "live" | "non_empty" | "readable" | "writable" | "matching" | "supported" | "acyclic" | "opaque" | "triangles" | "invertible";

export interface GroupReport {
  bytes_in: number;
  files: number;
  files_in: number;
  name: string;
  stage: GroupStage;
}

export type GroupStage = "arriving" | "compiling" | "ready" | { failed: Error };

export interface Progress {
  done: number;
  total: number;
}

export interface Readiness {
  compiling: number;
  failed: [string, Error][];
  groups: GroupReport[];
  loading: Progress;
  settling: number;
  unsettled: Unsettled[];
}

export interface Unsettled {
  brick: string;
  left: number;
}
