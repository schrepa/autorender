export interface FrameStats {
  cut_seconds: number;
  draw_calls: number;
  dropped_seconds: number;
  extraction_ms: number;
  fixed_steps: number;
  frame: number;
  indirect_draws: number;
  instances: number;
  jobs: JobStats;
  mesh_upload_bytes: number;
  pipelines: number;
  pipelines_created: number;
  pipelines_listed: number;
  pipelines_unlisted: number;
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

export type Awaits = "work" | "view" | "finished" | "frames";

export interface Error {
  allowed: Allowed | null;
  code: ErrorCode;
  expected: Expected | null;
  got: unknown;
  nearest: unknown;
  path: string;
}

export type ErrorCode = string;

export type Expected = "number" | "integer" | "string" | "boolean" | "object" | "array" | "id" | "path" | "file" | "entity" | "unique" | "present" | "absent" | "finite" | "normalized" | "sorted" | "nonzero" | "above" | "power_of_two" | "multiple" | "json" | "utf8" | "uri" | "base64" | "image" | "paused" | "running" | "gpu" | "live" | "non_empty" | "readable" | "writable" | "matching" | "supported" | "acyclic" | "opaque" | "triangles" | "invertible";

export type GpuTime = number | { unavailable: string };

export interface JobStats {
  cache_bytes: number;
  cache_hits: number;
  cache_mismatches: number;
  cache_misses: number;
  cancelled: number;
  completed: number;
  failed: number;
  held: number;
  in_place: number;
  running: number;
  shared_bytes: number;
  shares: number;
  submitted: number;
  workers: number;
}

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

export interface ResolutionStats {
  display_height: number;
  display_width: number;
  dropped_pixels: number;
  gpu_ms: GpuTime;
}

export interface Unsettled {
  awaits: Awaits;
  brick: string;
  left: number;
}
