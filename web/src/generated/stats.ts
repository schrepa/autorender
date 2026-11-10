export interface FrameStats {
  bricks: Record<string, unknown>;
  bricks_cost: BrickCosts | null;
  budget: BudgetUse[];
  cut_seconds: number;
  disabled_features: string[];
  draw_calls: number;
  dropped_draws: DroppedDraws[];
  dropped_seconds: number;
  extraction_ms: number;
  fixed_steps: number;
  frame: number;
  instances: number;
  jobs: JobStats;
  kind_passes: KindPass[];
  kind_problems: Error[];
  linked: { faces: Record<string, string>; lists: Record<string, string[]>; nobody: string[] };
  mesh_upload_bytes: number;
  pipelines: number;
  pipelines_created: number;
  pipelines_listed: number;
  pipelines_unlisted: number;
  preparation_ms: number;
  read_back_age: number | null;
  readiness: Readiness;
  refused_views: number;
  render_ms: number;
  render_passes: string[];
  resident_mesh_bytes: number;
  resident_meshes: number;
  resident_texture_bytes: number;
  resident_textures: number;
  resolution: ResolutionStats;
  simulation_ms: number;
  texture_upload_bytes: number;
  tick: number;
  triangles: number | null;
  upload_bytes: number;
}

export type Allowed = { kind: "range"; max: number; min: number } | { kind: "at_most"; max: number } | { kind: "at_least"; min: number } | { kind: "variants"; values: string[] };

export type Awaits = "work" | "view";

export interface BrickCost {
  allocations: number;
  brick: string;
  cpu_ms: number;
  created_after_warmup: number;
  duplicate_upload_bytes: number;
  gpu_bytes: number;
  gpu_ms: number | null;
  heap_live_bytes: number;
  heap_peak_bytes: number;
}

export interface BrickCosts {
  frame: number;
  measuring_ms: number;
  rows: BrickCost[];
}

export interface BudgetUse {
  asked: number;
  by: string;
  cut: number;
  spent: number;
}

export type DropReason = "unknown_pipeline" | "pass_mismatch" | "unfilled_holes" | "geometry_not_resident" | "geometry_not_held" | "geometry_waiting" | "geometry_failed" | "material_not_resident" | "material_not_held" | "material_waiting" | "material_failed" | "geometry_lacks_streams" | "no_render_pass" | "pipeline_over_limit" | "pipeline_compiling" | "pipeline_refused" | "missing_input" | "missing_bind_group" | "row_incomplete";

export interface DroppedDraws {
  brick: string;
  count: number;
  faults: boolean;
  kind: string | null;
  part: string;
  pass: string;
  reason: DropReason;
}

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

export interface KindPass {
  brick: string;
  kind: string;
  pass: string;
  state: KindPassState;
}

export type KindPassState = { draws: number; state: "drew" } | { draws: number; reason: DropReason; state: "dropped" } | { on: string; state: "waiting" } | { on: string; state: "refused" } | { reason: string; state: "skipped" } | { state: "silent" };

export interface Progress {
  done: number;
  total: number;
}

export interface Readiness {
  compiling: number;
  failed: [string, Error][];
  loading: Progress;
  settling: number;
  unsettled: Unsettled[];
}

export interface ResolutionStats {
  display_height: number;
  display_width: number;
  dropped_pixels: number;
  gpu_ms: number | null;
  render_height: number;
  render_width: number;
}

export interface Unsettled {
  awaits: Awaits;
  brick: string;
  left: number;
}
