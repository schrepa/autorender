use serde::Serialize;

/// A terrain as its inspector reports it (decision 051).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainView {
    /// The terrain as a scene version 9 `terrain` field: `SceneTerrain` of
    /// the SceneDocument schema (describe SceneDocument).
    #[cfg_attr(not(target_arch = "wasm32"), schemars(with = "serde_json::Value"))]
    pub terrain: engine_scene::SceneTerrain,
    pub levels: u32,
    /// Meters along x and z.
    pub extent: [f32; 2],
    pub resident_tiles_by_level: Vec<u32>,
    /// Per level, finest first: the most meters its height differs from
    /// the next coarser level's, which its patches morph by (0 until
    /// measured), and meters from the eye it draws to.
    pub level_errors: Vec<f32>,
    pub level_ranges: Vec<f32>,
    /// Its hydrology bake (decision 066), when its layers hold one.
    pub hydrology: Option<HydrologyView>,
    /// Its erosion bake (decision 068), when its layers hold one.
    pub erosion: Option<ErosionView>,
    /// Its path layers (decision 069), in stack order.
    pub paths: Vec<PathView>,
    /// Its edits (decision 080), in order: each one's `id` and its `stamp`,
    /// terrain-local, which a scene saves as a `{"stamp": ...}` layer after
    /// its others.
    pub edits: Vec<TerrainEditView>,
}

/// One terrain edit (decision 080): its id and its stamp (terrain-local).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainEditView {
    pub id: String,
    pub stamp: engine_core::terrain_data::Stamp,
}

/// A path layer (decision 069): its index in the stack, whether its
/// profile is baked, and once baked its length (meters), the points its
/// profile read the ground at, its steepest grade (rise per meter), the
/// most it cuts into and fills over the ground along its line off bridges
/// (meters), its bridges (decision 072) and the meters they span, and its
/// tunnels (decision 073) and the meters they run.
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

/// A terrain's erosion bake (decision 068): whether a bake is in progress
/// (tiles wait for it), and once done the samples it eroded, its rounds
/// and the most it wore away and filled (meters).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ErosionView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub rounds: Option<u32>,
    pub deepest: Option<f32>,
    pub highest: Option<f32>,
}

/// A terrain's hydrology bake (decision 066): whether it is in progress
/// (tiles wait for it), and once done the samples it read, those channels
/// run through, those of filled pits, the largest area draining through
/// one (square meters) and the waterfalls rivers drop over (decision 075).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HydrologyView {
    pub baking: bool,
    pub samples: Option<usize>,
    pub channel_samples: Option<usize>,
    pub lake_samples: Option<usize>,
    pub max_area: Option<f32>,
    pub falls: Option<usize>,
}

/// The log of runtime deformation's stamps (decision 083).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TracksView {
    pub stamps: usize,
    pub pressed: u64,
    pub dropped: u64,
    /// The newest stamp: where tracks are being made.
    pub latest: Option<engine_core::Track>,
}

/// World streaming at the last extraction (decision 086): cells holding
/// items and those loaded, cells within range still waiting, items,
/// entities streamed items hold spawned, the frame's spawns and despawns,
/// the camera velocity prefetch followed, each grid's cells, and the
/// cells nearest a streaming source.
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

/// One streaming grid (decision 086): its cells' side, the meters within
/// which they load (before the settings' distance), the most an item in
/// it reaches, and its cells and those loaded.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StreamingGridView {
    pub cell: f32,
    pub range: f32,
    pub reach: Option<f32>,
    pub cells: u32,
    pub loaded: u32,
}

/// One streamed cell (decision 086): its grid (0 small, 1 medium, 2
/// large), its key along x and z, its items and their entities, whether it
/// is loaded, and its meters from the nearest streaming source.
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

/// A grounded entity (decision 086) in the shape of a scene version 9
/// `grounded` field, with the level of the terrain tile it last stood on
/// (`None` before it sampled, or while no tile answered).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct GroundedView {
    pub offset: f32,
    pub level: Option<u32>,
}

/// Terrain streaming at the last extraction (decision 051).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainsView {
    pub terrains: u32,
    pub resident_tiles: u32,
    pub loading_tiles: u32,
    pub dropped_tiles: u32,
    pub patches: u32,
    pub fallback_patches: u32,
    /// Terrains whose bakes (decisions 066 and 068) are in progress.
    pub baking: u32,
    /// Bridges terrains' roads span (decision 072), tunnels they bore
    /// (decision 073), and waterfalls their rivers drop over (decision 075).
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
}

/// An animation player (decision 035) in the shape of a scene version 8
/// `animation` field, with its clip's duration and its targets.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct AnimationPlayerView {
    /// The clip's authored animation ID; None when no scene instance owns it.
    pub clip: Option<String>,
    /// Whether the clip is still resident.
    pub clip_alive: bool,
    /// Seconds from the clip's start to its last key.
    pub duration: Option<f32>,
    pub time: f32,
    pub speed: f32,
    #[serde(rename = "loop")]
    pub looping: bool,
    /// Each clip target's authored entity ID; None when unbound or not
    /// authored.
    pub targets: Vec<Option<String>>,
}

/// A decal (decision 039) in the shape of a scene version 8 `decal` field
/// (`engine_scene::SceneDecal::from_runtime`), with whether its material is
/// resident.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DecalView {
    /// The material's authored ID; None when no scene instance owns it.
    pub material: Option<String>,
    /// Whether the material is still resident.
    pub material_alive: bool,
    pub blend: engine_scene::SceneDecalBlend,
    pub opacity: f32,
    pub fade_degrees: f32,
}

/// A spline mesh (decision 071) in the shape of a scene version 9 `spline`
/// field (`engine_scene::SceneSpline::from_runtime`), and what it built:
/// the curve's length in meters, the copies of its mesh along it, its
/// chunks and their vertices, or why it built nothing.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SplineMeshView {
    pub points: Vec<[f32; 3]>,
    /// The mesh's and material's authored IDs; None when no scene instance
    /// owns them.
    pub mesh: Option<String>,
    pub material: Option<String>,
    pub length: Option<f32>,
    pub copies: Option<u32>,
    pub chunks: Option<usize>,
    pub vertices: Option<u32>,
    pub error: Option<String>,
}

/// A skin (decision 036): its joints' authored entity IDs.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SkinView {
    /// Each joint's authored entity ID; None when not authored.
    pub joints: Vec<Option<String>>,
}

/// Serializable development view, separate from the gameplay component.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TransformView {
    pub translation: [f32; 3],
    /// Quaternion components in XYZW order.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct GlobalTransformView {
    pub valid: bool,
    /// Column-major matrix; preserves shear from rotated, nonuniform ancestors.
    pub matrix: Option<[f32; 16]>,
    pub translation: Option<[f32; 3]>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InterpolateTransformView {
    /// Marker presence opts this local transform into fixed-tick interpolation.
    pub enabled: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct PresentedTransformView {
    pub valid: bool,
    /// Last extracted world matrix, column-major; inspection rebuilds it only to
    /// materialize a GPU-presented frame.
    pub matrix: Option<[f32; 16]>,
    pub translation: Option<[f32; 3]>,
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
    /// Frame at the last prepared extraction; null before the first extraction.
    /// Decimal strings preserve all 64-bit values in JavaScript.
    pub frame: Option<String>,
    /// Fixed tick at that extraction; does not imply GPU submission/completion.
    pub tick: Option<String>,
    pub mode: PresentationModeView,
    /// Actual fraction used at the last extraction; current mode uses one.
    pub alpha: f32,
    /// The last extraction was blended on the GPU (decision 014) and CPU
    /// presented poses still hold earlier values. Inspecting PresentedTransform
    /// materializes that frame first while its inputs are unchanged.
    pub gpu_pending: bool,
    pub history_nodes: usize,
    pub history_capacity: usize,
    /// Allocated history storage; excludes allocator overhead and other ECS data.
    pub history_bytes: usize,
    /// Reserved presentation scratch payload; excludes maps and allocator overhead.
    pub presentation_scratch_bytes: usize,
    pub presented_nodes: usize,
    /// Reserved canonical presented-pose and entity storage; excludes maps/allocator overhead.
    pub presented_store_bytes: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct HierarchyView {
    /// Current parent authored ID, if the parent has scene metadata.
    pub parent: Option<String>,
    pub parent_scene: Option<String>,
    /// None for roots; false for a removed runtime parent.
    pub parent_alive: Option<bool>,
    /// Whether parent is addressable by ID in this inspection's scene instance.
    pub parent_in_active_scene: Option<bool>,
    /// Parent recorded in source, which may differ after runtime reparenting.
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
    pub mesh_alive: bool,
    pub material_alive: bool,
    pub vertices: Option<usize>,
    pub indices: Option<usize>,
    pub local_bounds: Option<MeshBoundsView>,
    pub color: Option<[f32; 4]>,
    /// Fraction of light behind the surface it covers; below 1 it draws
    /// transparent (decision 026).
    pub opacity: Option<f32>,
    /// The material's response to light (decision 019).
    pub surface: Option<SurfaceView>,
    /// None denotes an untextured material; false denotes a missing dependency.
    pub texture_alive: Option<bool>,
    pub texture_dimensions: Option<[u32; 2]>,
    /// Whether the texture's alpha limits emission to some texels.
    pub texture_emission_mask: Option<bool>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SurfaceView {
    pub roughness: f32,
    pub metalness: f32,
    /// Emitted light as a multiple of the surface color, masked by texture alpha.
    pub emission: f32,
    pub ripples: f32,
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
    /// Admission of the current authoritative world transform, independent of visibility.
    pub world_valid: bool,
    pub world_error: Option<String>,
    /// Active/invalid in the refreshed cache; not_cached includes overflow;
    /// unavailable means the plugin is absent.
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
    /// Cache revision and source clocks are decimal strings, not runtime handles.
    pub revision: String,
    pub frame: String,
    pub tick: String,
    pub colliders: usize,
    pub active_colliders: usize,
    pub invalid_colliders: usize,
    pub overflow_colliders: usize,
    pub bvh_nodes: usize,
    /// Reserved collision cache payload; not total process memory.
    pub cache_bytes: usize,
    pub complete: bool,
    /// Placed objects colliding within terrains' collision radius
    /// (decision 064).
    pub placed_colliders: usize,
    pub limits: CollisionLimitsView,
}
