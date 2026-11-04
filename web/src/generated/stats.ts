export interface FrameStats {
  ambient_occlusion_stats: AmbientOcclusionStats;
  audio_stats: AudioStats;
  bloom_stats: BloomStats;
  bricks_cost: BrickCosts | null;
  cloud_stats: CloudStats;
  collision_stats: CollisionWorldView;
  disabled_features: string[];
  draw_calls: number;
  dropped_seconds: number;
  dynamic_resolution: DynamicResolutionStats;
  exposure_stats: ExposureStats;
  extraction_ms: number;
  fixed_steps: number;
  frame: number;
  ground_cover_stats: GroundCoverStats;
  haze_stats: HazeStats;
  instances: number;
  jobs: JobStats;
  linked: { faces: Record<string, string>; lists: Record<string, string[]>; nobody: string[] };
  loading: boolean;
  local_light_stats: LightStats;
  mesh_upload_bytes: number;
  meshes_stats: MeshStats;
  navigation_stats: NavStats;
  particle_stats: ParticleStats;
  physics_stats: PhysicsStats;
  pipelines: number;
  pipelines_compiling: boolean;
  pipelines_created: number;
  pipelines_listed: number;
  pipelines_unlisted: number;
  read_back_age: number | null;
  reflection_probes_stats: ReflectionProbeStats;
  reflections_stats: ReflectionsStats;
  render_ms: number;
  render_passes: string[];
  resident_mesh_bytes: number;
  resident_meshes: number;
  resident_texture_bytes: number;
  resident_textures: number;
  resolution: ResolutionStats;
  scatter_stats: ScatterStats;
  shadow_stats: ShadowStats;
  simulation_ms: number;
  sky_stats: SkyStats;
  static_tables: StaticTablesView;
  temporal_antialiasing_stats: TemporalAntialiasingStats;
  terrain_shading_stats: TerrainShadingStats;
  terrain_stats: TerrainStats;
  text_stats: TextStats;
  texture_upload_bytes: number;
  tick: number;
  triangles: number | null;
  ui_stats: UiStats;
  upload_bytes: number;
  water_stats: WaterStats;
  world_streaming_stats: CellStats;
}

export interface AmbientOcclusionStats {
  active: boolean;
}

export interface AudioStats {
  decode_failures: number;
  decoding: number;
  device: boolean;
  latest_failure: string | null;
  sounds_loaded: number;
  voices: number;
  voices_cut: number;
}

export interface BloomStats {
  active: boolean;
}

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

export interface CellStats {
  cells: number;
  despawned: number;
  entities: number;
  items: number;
  loaded: number;
  pending: number;
  spawned: number;
}

export interface CloudStats {
  rows: number;
  view_pixels: number;
}

export interface CollisionLimitsView {
  max_candidate_pairs: number;
  max_colliders: number;
  max_query_candidates: number;
  max_query_hits: number;
  max_sensor_pairs: number;
}

export interface CollisionWorldView {
  active_colliders: number;
  bvh_nodes: number;
  cache_bytes: number;
  colliders: number;
  complete: boolean;
  frame: string;
  invalid_colliders: number;
  limits: CollisionLimitsView;
  overflow_colliders: number;
  placed_colliders: number;
  tick: string;
}

export interface CullingStats {
  active: boolean;
  instances: number | null;
}

export interface Cut {
  asked: number;
  cut: number;
  kept: number;
}

export interface DetailStats {
  impostor_cut: Cut;
  impostor_instances: number | null;
  impostors: number;
  lod0_instances: number | null;
  lod1_instances: number | null;
  lod2_instances: number | null;
  meshes_without_levels: number;
}

export interface DynamicResolutionStats {
  active: boolean;
  budget_ms: number;
  scale: number;
}

export interface ExposureStats {
  active: boolean;
  metered: number;
  stops: number;
  view: number;
}

export interface GroundCoverStats {
  blade_cut: Cut;
  reach: number;
  tiers: number;
}

export interface HazeStats {
  active: boolean;
}

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

export interface LightStats {
  cluster_cut: Cut;
  light_cut: Cut;
  shaded: number;
  visible: number | null;
}

export interface MeshStats {
  casters: number | null;
  culled_instances: number;
  culling_enabled: boolean;
  dropped_instances: number;
  gpu_visible: number | null;
  instances: number;
  invalid_transforms: number;
  level_of_detail: DetailStats;
  missing_assets: number;
  moving_casters: number | null;
  occlusion_culling: CullingStats;
  read_back_age: number | null;
  render_candidates: number;
  subpixel_culling: CullingStats;
  triangles: number | null;
  visible_instances: number;
}

export interface NavStats {
  agents: number;
  arrived: number;
  cooking: number;
  grounds: number;
  portals: number;
  searching: number;
  unreachable: number;
  waiting: number;
  walking: number;
  work: number;
}

export interface ParticleStats {
  drawn: number;
  particle_cut: Cut;
}

export interface PhysicsStats {
  awake: number;
  bodies: number;
  contacts: number;
  held: number;
  solids: number;
  water_samples: number;
  wet: number;
}

export interface ReflectionProbeStats {
  faces: number;
  probe_cut: Cut;
  ready: number;
}

export interface ReflectionsStats {
  active: boolean;
}

export interface ResolutionStats {
  display_height: number;
  display_width: number;
  dropped_pixels: number;
  gpu_ms: number | null;
  render_height: number;
  render_width: number;
}

export interface ScatterStats {
  dropped_objects: number;
  impostors: Cut;
  lods: Cut;
  occupancy_averages: number;
  occupancy_redraws: number;
  parts: Cut;
  placed_far_objects: number;
  placed_impostors: number;
  placed_objects: number;
}

export interface ShadowStats {
  active: boolean;
  draws: number;
  level_cut: Cut;
  moving_cut: Cut;
  moving_slot_cut: Cut;
  sky_drawn: number;
  sky_maps: number;
  tile_cut: Cut;
  tiles_drawn: number;
  tiles_marked: number;
  tiles_reclaimed: number;
  tiles_resident: number;
}

export interface SkyStats {
  tables: number;
}

export interface StaticTableRow {
  arrived: number;
  kind: string;
  parts: number;
  showing: boolean;
}

export interface StaticTablesView {
  failed_jobs: number;
  latest_job_failure: string | null;
  loading: number;
  tables: StaticTableRow[];
}

export interface TemporalAntialiasingStats {
  active: boolean;
}

export interface TerrainShadingStats {
  bounce_dropped: number;
  bounce_levels: number;
  bounce_probes: number;
  bounce_relit: number;
  bounced: number;
  bouncing: number;
  colored: number;
  coloring: number;
  dropped_materials: number;
  failed_jobs: number;
  gpu_controls: number;
  horizons: number;
  latest_job_failure: string | null;
  leaves: Cut;
  lighting: number;
  lighting_uploads: number;
  lit: number;
  macro_maps: number;
  pending: number;
  shaded_tiles: number;
  shading_records: number;
  shadow_patches: number;
  shallow_subdivisions: number;
  sky_maps: number;
  surface_slots: Cut;
  tiles: number;
  tiles_dropped: Cut;
  triangles: number;
}

export interface TerrainStats {
  baking: number;
  bridges: number;
  cached_bytes: number;
  cached_tiles: number;
  dropped_tiles: number;
  failed_jobs: number;
  fallback_tiles: number;
  latest_job_failure: string | null;
  loading: boolean;
  loading_tiles: number;
  missing_tiles: number;
  pending: number;
  prefetching_tiles: number;
  resident_tiles: number;
  restored_tiles: number;
  surroundings_tiles: number;
  terrains: number;
  tile_cut: Cut;
  tunnels: number;
  waiting_tiles: number;
}

export interface TextStats {
  atlas_resets: number;
  failed_fonts: number;
  fonts: number;
  glyphs_cut: number;
  glyphs_drawn: number;
  runs: number;
}

export interface UiStats {
  events: number;
  layouts: number;
  missing_bindings: number;
  nodes: number;
  rects_cut: number;
  rects_drawn: number;
  roots: number;
}

export interface WaterStats {
  blocks: number;
  cell_cut: Cut;
  dropped_blocks: number;
  dropped_particles: number;
  dropped_touches: number;
  failed_jobs: number;
  falls: number;
  foam_cut: Cut;
  foam_particles: number;
  foam_points: number;
  foam_reach: number;
  leaf_cut: Cut;
  live_behind: number;
  live_cut: number;
  live_setup: number;
  live_steps: number;
  live_waiting: number;
  looks_left: number;
  particle_texels: number;
  particles_born: number;
  rewatered_tiles: number;
  settled_records: number;
  settles_left: number;
  settling_records: number;
  stale_water_tiles: number;
  touches: number;
  wave_particles: number;
  white_water: number;
}
