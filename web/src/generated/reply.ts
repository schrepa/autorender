import type { ErrorCode } from "./errors";

export interface Reply {
  error?: Error | null;
  observation?: Observation | null;
  ok: boolean;
  result?: Record<string, unknown> | null;
  version?: number | null;
}

export type Allowed = { kind: "range"; max: number; min: number } | { kind: "at_most"; max: number } | { kind: "at_least"; min: number } | { kind: "variants"; values: string[] };

export interface Artifact {
  bytes: number;
  mime_type: string;
  path: string;
  sha256: string;
}

export interface AssetDigests {
  materials: DigestEntry;
  meshes: DigestEntry;
  textures: DigestEntry;
}

export interface AssetsResult {
  materials: number;
  meshes: number;
  missing: MissingAsset[];
  missing_total: number;
  readiness: string;
  resident_cpu_bytes: number;
  texture_assets: TextureAsset[];
  texture_cpu_bytes: number;
  textures: number;
}

export interface AuthoredEntity {
  id?: string | null;
  name?: string | null;
  scene?: string | null;
}

export interface CameraResult {
  camera_changed: boolean;
  temporary: boolean;
}

export interface CaptureResult {
  artifact?: KeptArtifact | null;
  digest?: DigestResult | null;
  rendered: boolean;
  temporary?: boolean | null;
}

export interface CullingResult {
  culling_enabled: boolean;
  temporary: boolean;
}

export interface DebugFaultResult {
  injected: boolean;
}

export interface DebugMissingAssetResult {
  entities: number;
  temporary: boolean;
}

export type DescribeResult = { bricks: DescribedBrick[]; types: DescribedType[]; verbs: string[] } | { control_state?: Record<string, unknown> | null; kind: TypeKind; name: string; schema: unknown } | { kind: TypeKind; name: string; opt_in_sections: string[]; schema: unknown } | { kind: TypeKind; name: string; schema: unknown } | { codes: string[]; kind: TypeKind; name: string } | { at?: string | null; brick?: string | null; descriptor: InspectorDescriptor; kind: TypeKind } | { args: unknown; at?: string | null; brick?: string | null; kind: TypeKind; name: string; shows?: string | null; verb: string } | { brick: DescribedBrick; kind: TypeKind } | { at: string; brick: string; kind: TypeKind; name: string; schema: unknown };

export interface DescribedBrick {
  crate: string;
  explains: string[];
  inspectors: DescribedEntry[];
  lists: string[];
  name: string;
  operations: DescribedEntry[];
  settings: DescribedEntry[];
  slot?: string | null;
  stats: DescribedEntry[];
}

export interface DescribedEntry {
  at?: string | null;
  key?: string | null;
  name: string;
  verb?: string | null;
}

export interface DescribedType {
  brick?: string | null;
  kind: TypeKind;
  name: string;
}

export interface DetailShares {
  averaged: number;
  canopy: number;
  detailed: number;
}

export interface DigestEntry {
  count: number;
  hash: string;
}

export interface DigestResult {
  assets?: AssetDigests | null;
  components: Record<string, DigestEntry>;
  entities: number;
  hash: string;
}

export interface Distribution {
  max: number;
  median: number;
  min: number;
  p95: number;
  p99: number;
}

export interface DistributionResult {
  distribution?: Distribution | null;
}

export type DrawnKind = "mesh" | "placed" | "card" | "placed_card" | "terrain" | "cover" | "water" | "falls" | "sky";

export type Effect = { kind: "pause"; paused: boolean } | { kind: "wait"; within_ms: number } | { kind: "frame"; timestamp: number } | { kind: "show" } | { args: Record<string, unknown>; kind: "game"; name: string } | { kind: "source" } | { contents: unknown; kind: "save"; name: string } | { kind: "stream"; records: unknown[] };

export type EntityComponents = Record<string, unknown> | string[];

export interface EntityRow {
  components: EntityComponents;
  id: string;
  name: string;
  tags?: string[] | null;
}

export interface EnvironmentResult {
  after: Record<string, unknown>;
  before: Record<string, unknown>;
  environment_changed: string[];
  temporary: boolean;
}

export interface Error {
  allowed?: Allowed | null;
  code: ErrorCode;
  expected?: Expected | null;
  got?: unknown;
  nearest?: unknown;
  path: string;
}

export type Expected = "number" | "integer" | "string" | "boolean" | "object" | "array" | "id" | "path" | "file" | "entity" | "unique" | "present" | "absent" | "finite" | "normalized" | "sorted" | "nonzero" | "above" | "power_of_two" | "multiple" | "json" | "utf8" | "uri" | "base64" | "image" | "paused" | "running" | "gpu" | "live" | "non_empty" | "readable" | "writable" | "matching" | "supported" | "acyclic" | "opaque" | "triangles" | "invertible";

export interface ExplainResult {
  about: string;
  records: Record<string, unknown>[];
  total: number;
}

export interface FramesResult {
  artifact?: KeptArtifact | null;
  frames: number;
  interval_ms: number;
  temporary: boolean;
}

export interface GameResult {
  game: string;
  result: unknown;
}

export interface HeightSample {
  height: number;
  hole: boolean;
  level: number;
  normal: number[];
  path?: number | null;
  standing_water?: StandingWater | null;
  terrain: TerrainKey;
  water?: HydrologySample | null;
}

export interface HostContext {
  device_pixel_ratio: number;
  height: number;
  paused: boolean;
  source?: string | null;
  timing: string;
  user_agent: string;
  width: number;
}

export interface HydrologySample {
  flood: number;
  flow: number;
  grain: number;
  lake: number;
  power: number;
  river: number;
  wetness: number;
}

export interface InputResult {
  applies_on: string;
  temporary: boolean;
}

export type InspectResult = { resource: string; value: unknown } | { entity: EntityRow } | { entities: EntityRow[]; limit: number; next_offset?: number | null; offset: number; total: number };

export interface InspectorDescriptor {
  default?: unknown;
  name: string;
  schema: unknown;
}

export interface KeptArtifact {
  bytes: number;
  frame?: string | null;
  mime_type: string;
  path: string;
  sha256: string;
  tick?: string | null;
}

export interface MaterialShare {
  material: number;
  share: number;
}

export interface MissingAsset {
  draws: string;
  entity: AuthoredEntity;
  missing: string;
  position?: number[] | null;
}

export interface Observation {
  frame?: string | null;
  presentation: PresentationStateView;
  scene?: SceneObservation | null;
  tick?: string | null;
}

export interface OpResults {
  assets: AssetsResult;
  camera: CameraResult;
  capture: CaptureResult;
  culling: CullingResult;
  debug_fault: DebugFaultResult;
  debug_missing_asset: DebugMissingAssetResult;
  describe: DescribeResult;
  digest: DigestResult;
  distribution: DistributionResult;
  environment: EnvironmentResult;
  explain: ExplainResult;
  frames: FramesResult;
  game: GameResult;
  input: InputResult;
  inspect: InspectResult;
  pause: StatusResult;
  present: PresentResult;
  probe: ProbeResult;
  profile: ProfileResult;
  render: RenderResult;
  resume: StatusResult;
  status: StatusResult;
  step: StepResult;
  terrain: TerrainControlResult;
  terrain_edit: TerrainEdited;
  terrain_height: TerrainHeightResult;
  terrain_redo: TerrainEdited;
  terrain_river: TerrainEdited;
  terrain_route: TerrainRouteResult;
  terrain_sweep: TerrainSweepResult;
  terrain_undo: TerrainEdited;
  watch: WatchResult;
}

export interface Outcome {
  artifact?: Artifact | null;
  failed?: string | null;
  interrupted?: boolean;
  missing?: string[] | null;
  now?: number | null;
  result?: unknown;
}

export interface PresentResult {
  rendered: boolean;
  temporary: boolean;
}

export type PresentationModeView = "current" | "interpolated";

export interface PresentationStateView {
  alpha: number;
  frame?: string | null;
  gpu_pending: boolean;
  history_bytes: number;
  history_capacity: number;
  history_nodes: number;
  mode: PresentationModeView;
  presentation_scratch_bytes: number;
  presented_nodes: number;
  presented_store_bytes: number;
  tick?: string | null;
}

export interface ProbeResult {
  casts_shadows: boolean;
  depth: number;
  detail?: number | null;
  displayed?: number[] | null;
  entity?: AuthoredEntity | null;
  fading: boolean;
  fragments: number;
  kind?: DrawnKind | null;
  level: number;
  normal?: number[] | null;
  object?: number | null;
  pixel: number[];
  shaded?: number[] | null;
  shadowed: boolean;
  sun: SunLookup;
  sun_index: number;
  terrain?: TerrainProbe | null;
  water?: WaterProbe | null;
  world?: number[] | null;
}

export interface ProfileResult {
  artifact?: KeptArtifact | null;
  bricks: ProfiledBrick[];
  cpu_ms: number;
  frames: number;
  gpu_ms?: number | null;
  interval_ms: number;
  temporary: boolean;
  total: number;
}

export interface ProfiledBrick {
  allocations_per_frame?: number | null;
  brick: string;
  cpu_ms: number;
  cpu_ms_max: number;
  created_after_warmup?: number | null;
  duplicate_upload_bytes?: number | null;
  gpu_bytes?: number | null;
  gpu_ms?: number | null;
  gpu_ms_max?: number | null;
  heap_live_bytes?: number | null;
  heap_peak_bytes?: number | null;
}

export interface Reach {
  max: number[];
  min: number[];
}

export interface RenderResult {
  rendered: boolean;
}

export interface RouteFound {
  hairpin_steps: number;
  length: number;
  points: number[][];
  steepest: number;
}

export interface RouteOptions {
  grade_cost?: number;
  level?: number;
  max_grade?: number;
  turn_cost?: number;
  water_cost?: number;
}

export interface SceneObservation {
  id: string;
  revision: string;
}

export interface StandingWater {
  depth: number;
  surface: number;
}

export interface StatusLimits {
  page_size: number;
  request_bytes: number;
  response_bytes: number;
  scan_entities: number;
}

export interface StatusResult {
  authored_entities?: number | null;
  collision?: Record<string, unknown> | null;
  culling_enabled: boolean;
  features?: string[] | null;
  fixed_delta_seconds?: number | null;
  inspectors: number;
  limits: StatusLimits;
  presentation: PresentationStateView;
  world_entities: number;
}

export type Step = { reply: Reply } | { effect: Effect };

export interface StepResult {
  temporary: boolean;
  ticks: number;
}

export type SunLookup = "none" | "level" | "horizon" | "beyond";

export interface SweepExample {
  at: number[];
  height: number;
}

export interface SweepLayers {
  compared: number;
  worst: Worst;
}

export interface SweepRays {
  compared: number;
  missed: number;
  placed_first: number;
  worst: Worst;
}

export interface TerrainControlResult {
  frozen_at?: number[] | null;
  pixel_error?: number | null;
  subdivision_error?: number | null;
  temporary: boolean;
  terrain_changed: boolean;
}

export type TerrainEdited = { applied: boolean; dirty?: Reach | null; id: string; rivers: number; temporary: boolean } | { applied: boolean; dirty: Reach; edits: number; id: string; largest_change: number; temporary: boolean } | { dirty?: Reach | null; edits: number; temporary: boolean; undone?: string | null } | { dirty?: Reach | null; edits: number; redone?: string | null; temporary: boolean };

export interface TerrainHeightResult {
  samples: HeightSample | null[];
  stats: TerrainStats;
}

export type TerrainKey = string | number;

export interface TerrainProbe {
  detail: DetailShares;
  materials: MaterialShare[];
  patch: number[];
  sky: number;
  tile: number[];
}

export interface TerrainRouteResult {
  options: RouteOptions;
  origin: number[];
  route?: RouteFound | null;
  terrain: TerrainKey;
}

export interface TerrainStats {
  baking: number;
  bounced: number;
  bouncing: number;
  bridges: number;
  cached_bytes: number;
  cached_tiles: number;
  colored: number;
  coloring: number;
  dropped_tiles: number;
  fallback_tiles: number;
  falls: number;
  lighting: number;
  lit: number;
  loading: boolean;
  loading_tiles: number;
  missing_tiles: number;
  prefetching_tiles: number;
  resident_tiles: number;
  restored_tiles: number;
  rewatered_tiles: number;
  settled_records: number;
  settling_records: number;
  shaded_tiles: number;
  shading_records: number;
  stale_water_tiles: number;
  surroundings_tiles: number;
  terrains: number;
  tunnels: number;
  waiting_tiles: number;
  white_water: number;
}

export interface TerrainSweepResult {
  collision?: SweepRays | null;
  examples: SweepExample[];
  holes: number;
  layers: SweepLayers;
  level_zero: number;
  points: number;
  resident: number;
}

export interface TextureAsset {
  bytes?: number | null;
  gpu_resident?: boolean | null;
  height?: number | null;
  id: string;
  state: TextureState;
  width?: number | null;
}

export type TextureState = "cpu_ready" | "missing";

export type TypeKind = "component" | "resource" | "authoring" | "verb" | "operation" | "brick" | "stats" | "observation" | "catalog";

export interface WatchRecord {
  component?: string | null;
  entity?: string | null;
  frame?: string | null;
  stat?: string | null;
  tick?: string | null;
  value?: unknown;
}

export interface WatchResult {
  frames: number;
  interval_ms: number;
  records: number;
  temporary: boolean;
}

export interface WaterProbe {
  depth: number;
  eddies: number;
  froude: number;
  reach?: WaterReach | null;
  roughness: number;
  span: number;
  speed: number;
  turbulence: number;
  white: number;
}

export type WaterReach = "pool" | "plane" | "steps" | "cascade" | "falls";

export interface Worst {
  at?: number[] | null;
  max_difference: number;
}
