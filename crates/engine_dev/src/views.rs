use serde::Serialize;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainView {
    #[cfg_attr(not(target_arch = "wasm32"), schemars(with = "serde_json::Value"))]
    pub terrain: engine_scene::SceneTerrain,
    pub levels: u32,
    pub extent: [f32; 2],
    pub resident_tiles_by_level: Vec<u32>,
    pub level_errors: Vec<f32>,
    pub level_ranges: Vec<f32>,
    pub hydrology: Option<HydrologyView>,
    pub erosion: Option<ErosionView>,
    pub paths: Vec<PathView>,
    pub edits: Vec<TerrainEditView>,
    pub rivers: Vec<TerrainRiverView>,
    pub surroundings: Option<SurroundingsView>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SurroundingsView {
    pub reach: [f32; 2],
    pub tiles: u32,
    pub spacing: f32,
    pub resident_tiles: u32,
    pub drawn: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainEditView {
    pub id: String,
    pub stamp: engine_core::terrain_data::Stamp,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainRiverView {
    pub id: String,
    pub river: engine_core::GivenRiver,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct PathView {
    pub layer: usize,
    pub baked: bool,
    pub length: Option<f32>,
    pub samples: Option<usize>,
    pub steepest: Option<f32>,
    pub deepest_cut: Option<f32>,
    pub highest_fill: Option<f32>,
    pub bridges: Option<usize>,
    pub bridged: Option<f32>,
    pub tunnels: Option<usize>,
    pub tunneled: Option<f32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ErosionView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub rounds: Option<u32>,
    pub deepest: Option<f32>,
    pub highest: Option<f32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HydrologyView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub channel_samples: Option<usize>,
    pub lake_samples: Option<usize>,
    pub max_area: Option<f32>,
    pub falls: Option<usize>,
    pub given: Option<usize>,
    pub uncut: Option<usize>,
    pub flow: Option<RiverFlowView>,
    pub fall_list: Vec<FallView>,
    pub rivers: Vec<RiverView>,
}

pub const FALLS_SHOWN: usize = 16;
pub const RIVERS_SHOWN: usize = 16;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RiverView {
    pub name: Option<String>,
    pub given: Option<[u32; 2]>,
    pub courses: usize,
    pub length: f32,
    pub first: [f32; 2],
    pub last: [f32; 2],
    pub width: [f32; 2],
    pub depth: [f32; 2],
    pub flood: [f32; 2],
    pub given_width: bool,
    pub given_depth: bool,
    pub given_flood: bool,
    pub uncut: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainWaterView {
    pub flow: RiverFlowView,
    pub falls: Vec<FallView>,
    pub lakes: Vec<LakeView>,
    pub sights: Vec<RiverSight>,
    pub blocks: usize,
    pub pools: Vec<PoolView>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct PoolView {
    pub at: [f32; 3],
    pub rise: f32,
    pub inflow: f32,
    pub outflow: f32,
    pub blocked: f32,
    pub crest: Option<f32>,
    pub area: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct LakeView {
    pub level: f32,
    pub area: f32,
    pub middle: [f32; 3],
    pub shore: [f32; 3],
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RiverSight {
    pub kind: &'static str,
    pub point: [f32; 3],
    pub direction: [f32; 2],
    pub half: f32,
    pub speed: f32,
    pub discharge: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RiverFlowView {
    pub rivers: usize,
    pub lakes: usize,
    pub points: usize,
    pub max_discharge: f32,
    pub fastest: f32,
    pub rapid_share: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct FallView {
    pub top: [f32; 3],
    pub bottom: [f32; 3],
    pub width: f32,
    pub speed: f32,
    pub discharge: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WorldCellsView {
    pub cells: u32,
    pub loaded: u32,
    pub pending: u32,
    pub items: u32,
    pub entities: u32,
    pub spawned: u32,
    pub despawned: u32,
    pub velocity: [f32; 3],
    pub grids: Vec<StreamingGridView>,
    pub nearest: Vec<StreamedCellView>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StreamingGridView {
    pub cell: f32,
    pub range: f32,
    pub reach: Option<f32>,
    pub cells: u32,
    pub loaded: u32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StreamedCellView {
    pub grid: u8,
    pub key: [i32; 2],
    pub items: u32,
    pub entities: u32,
    pub loaded: bool,
    pub distance: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct JobsView {
    pub workers: u32,
    pub running: u32,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub last_failure: Option<String>,
    pub cancelled: u64,
    pub shares: u32,
    pub shared_bytes: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes: u64,
    pub cache_mismatches: u64,
    pub held: u32,
    pub in_place: u64,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StaticTablesView {
    pub tables: Vec<StaticTableRow>,
    pub failed_jobs: u64,
    pub latest_job_failure: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StaticTableRow {
    pub kind: String,
    pub parts: u64,
    pub arrived: u64,
    pub showing: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WorldChangesView {
    pub removed: usize,
    pub ids: Vec<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StreamedIdView {
    pub id: String,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct GroundedView {
    pub offset: f32,
    pub level: Option<u32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainsView {
    pub terrains: u32,
    pub resident_tiles: u32,
    pub loading_tiles: u32,
    pub waiting_tiles: u32,
    pub cached_tiles: u32,
    pub cached_bytes: u64,
    pub restored_tiles: u64,
    pub prefetching_tiles: u32,
    pub dropped_tiles: u32,
    pub fallback_tiles: u32,
    pub baking: u32,
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
    pub surroundings_tiles: u32,
    pub water: Option<TerrainWaterView>,
    pub failed_jobs: u64,
    pub latest_job_failure: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SplineMeshView {
    pub points: Vec<[f32; 3]>,
    pub mesh: Option<String>,
    pub material: Option<String>,
    pub length: Option<f32>,
    pub copies: Option<u32>,
    pub chunks: Option<usize>,
    pub vertices: Option<u32>,
    pub error: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainDebugView {
    pub freeze: Option<[f32; 3]>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InputView {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
    pub held: Vec<u32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TransformView {
    pub translation: [f32; 3],
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct GlobalTransformView {
    pub valid: bool,
    pub matrix: Option<[f32; 16]>,
    pub translation: Option<[f32; 3]>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InterpolateTransformView {
    pub enabled: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum PresentationModeView {
    Current,
    Interpolated,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct PresentationStateView {
    pub frame: Option<String>,
    pub tick: Option<String>,
    pub mode: PresentationModeView,
    pub alpha: f32,
    pub gpu_pending: bool,
    pub history_nodes: usize,
    pub history_capacity: usize,
    pub history_bytes: usize,
    pub presentation_scratch_bytes: usize,
    pub presented_nodes: usize,
    pub presented_store_bytes: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HierarchyView {
    pub parent: Option<String>,
    pub parent_scene: Option<String>,
    pub parent_alive: Option<bool>,
    pub parent_in_active_scene: Option<bool>,
    pub authored_parent: Option<String>,
    pub prefab_id: Option<String>,
    pub prefab_instance: Option<String>,
    pub prefab_local_id: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CameraView {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct Mesh3dView {
    pub mesh: Option<MeshView>,
    pub material: Option<engine_scene::SceneMaterial>,
    pub textures: Vec<TextureView>,
    pub drawn_level: Option<u32>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct MeshView {
    pub id: Option<String>,
    pub source: Option<engine_scene::MeshOrigin>,
    pub vertices: usize,
    pub indices: usize,
    pub local_bounds: MeshBoundsView,
    pub colors: bool,
    pub uv1: bool,
    pub rendered: Option<MeshRenderView>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct MeshRenderView {
    pub levels: Vec<LodLevelView>,
    pub impostor: Option<bool>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct LodLevelView {
    pub indices: u32,
    pub error: f32,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TextureView {
    pub id: Option<String>,
    pub uri: Option<String>,
    pub inline: bool,
    pub alive: bool,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub emission_mask: Option<bool>,
    pub gpu_resident: Option<bool>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct MeshBoundsView {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ColliderShapeView {
    Cuboid { half_extents: [f32; 3] },
    Sphere { radius: f32 },
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ColliderView {
    pub shape: ColliderShapeView,
    pub memberships: u32,
    pub filter: u32,
    pub sensor: bool,
    pub world_valid: bool,
    pub world_error: Option<String>,
    pub cache_status: String,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CollisionLimitsView {
    pub max_colliders: usize,
    pub max_candidate_pairs: usize,
    pub max_sensor_pairs: usize,
    pub max_query_candidates: usize,
    pub max_query_hits: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CollisionWorldView {
    pub revision: String,
    pub frame: String,
    pub tick: String,
    pub colliders: usize,
    pub active_colliders: usize,
    pub invalid_colliders: usize,
    pub overflow_colliders: usize,
    pub bvh_nodes: usize,
    pub cache_bytes: usize,
    pub complete: bool,
    pub placed_colliders: usize,
    pub limits: CollisionLimitsView,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct TintView(pub [f32; 4]);
