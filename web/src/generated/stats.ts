export interface FrameStats {
  ambient_occlusion: Applied;
  bloom: Applied;
  bricks_cost: BrickCosts | null;
  clouds: CloudStats;
  culled_instances: number;
  culling_enabled: boolean;
  disabled_features: string[];
  draw_calls: number;
  dropped_instances: number;
  dropped_seconds: number;
  dynamic_resolution: DynamicResolutionStats;
  extraction_ms: number;
  fixed_steps: number;
  frame: number;
  ground_cover: GroundCoverStats;
  haze: Applied;
  instances: number;
  invalid_transforms: number;
  jobs: JobStats;
  level_of_detail: DetailStats;
  linked: { defaulted: string[]; faces: Record<string, string>; lists: Record<string, string[]>; nobody: string[] };
  loading: boolean;
  local_lights: LightStats;
  mesh_upload_bytes: number;
  missing_assets: number;
  navigation: NavStats;
  occlusion_culling: CullingStats;
  occupancy: OccupancyStats;
  particles: ParticleStats;
  physics: PhysicsStats;
  pipelines: number;
  pipelines_compiling: boolean;
  pipelines_created: number;
  pipelines_listed: number;
  pipelines_unlisted: number;
  read_back_age: number | null;
  reflection_probes: ProbeStats;
  reflections: Applied;
  render_candidates: number;
  render_ms: number;
  render_passes: string[];
  resident_mesh_bytes: number;
  resident_meshes: number;
  resident_texture_bytes: number;
  resident_textures: number;
  resolution: ResolutionStats;
  shadows: ShadowStats;
  simulation_ms: number;
  sky: SkyStats;
  subpixel_culling: CullingStats;
  temporal_antialiasing: Applied;
  terrain: TerrainRenderStats;
  texture_upload_bytes: number;
  tick: number;
  triangles: number | null;
  upload_bytes: number;
  visible_instances: number;
  water: WaterStats;
  world_streaming: CellStats;
}

export interface Applied {
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
  invalid: number;
  light_cut: Cut;
  shaded: number;
  visible: number | null;
}

export interface NavStats {
  agents: number;
  arrived: number;
  grounds: number;
  portals: number;
  searching: number;
  unreachable: number;
  waiting: number;
  walking: number;
  work: number;
}

export interface OccupancyStats {
  averages: number;
  redraws: number;
}

export interface ParticleStats {
  drawn: number;
  invalid: number;
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

export interface ProbeStats {
  faces: number;
  probe_cut: Cut;
  ready: number;
}

export interface ResolutionStats {
  display_height: number;
  display_width: number;
  dropped_pixels: number;
  gpu_ms: number | null;
  render_height: number;
  render_width: number;
}

export interface ShadowStats {
  active: boolean;
  casters: number | null;
  draws: number;
  moving_casters: number | null;
  moving_cut: Cut;
  sky_drawn: number;
  sky_maps: number;
  tile_cut: Cut;
  tiles_drawn: number;
  tiles_marked: number;
  tiles_reclaimed: number;
  tiles_resident: number;
}

export interface SkyStats {
  exposure: ExposureStats;
  tables: number;
}

export interface TerrainBounceStats {
  dropped: number;
  levels: number;
  probes: number;
  relit: number;
}

export interface TerrainRenderStats {
  bounce: TerrainBounceStats;
  dropped_materials: number;
  dropped_placed_objects: number;
  dropped_tiles: number;
  fallback_tiles: number;
  gpu_controls: number;
  horizons: number;
  leaf_cut: Cut;
  lighting_uploads: number;
  loading_tiles: number;
  macro_maps: number;
  placed_far_objects: number;
  placed_impostors: number;
  placed_objects: number;
  shadow_patches: number;
  shallow_subdivisions: number;
  sky_maps: number;
  tile_cut: Cut;
  tiles: number;
  triangles: number;
  white_water: number;
}

export interface WaterStats {
  blocks: number;
  cell_cut: Cut;
  dropped_blocks: number;
  dropped_particles: number;
  dropped_touches: number;
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
  particle_texels: number;
  particles_born: number;
  touches: number;
  wave_particles: number;
}
