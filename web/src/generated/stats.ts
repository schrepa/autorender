export interface FrameStats {
  cut_seconds: number;
  draw_calls: number;
  extraction_ms: number;
  fixed_steps: number;
  frame: number;
  instances: number;
  mesh_upload_bytes: number;
  pipelines: number;
  pipelines_created: number;
  preparation_ms: number;
  readiness: Readiness;
  refused_views: number;
  render_ms: number;
  render_passes: string[];
  resolution: ResolutionStats;
  simulation_ms: number;
  texture_upload_bytes: number;
  tick: number;
  upload_bytes: number;
}

export type Allowed = { kind: "range"; max: number; min: number } | { kind: "at_most"; max: number } | { kind: "at_least"; min: number } | { kind: "variants"; values: string[] };

export interface Error {
  allowed: Allowed | null;
  code: ErrorCode;
  expected: Expected | null;
  got: unknown;
  nearest: unknown;
  path: string;
}

export type ErrorCode = string;

export type Expected = "number" | "integer" | "string" | "boolean" | "object" | "array" | "id" | "file" | "entity" | "unique" | "present" | "absent" | "finite" | "sorted" | "nonzero" | "above" | "multiple" | "json" | "utf8" | "uri" | "base64" | "image" | "gpu" | "live" | "non_empty" | "readable" | "writable" | "matching" | "supported" | "acyclic" | "opaque" | "triangles" | "invertible";

export type GpuTime = number | { unavailable: string };

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

export interface ResolutionStats {
  display_height: number;
  display_width: number;
  dropped_pixels: number;
  gpu_ms: GpuTime;
}

export interface Unsettled {
  brick: string;
  left: number;
}
