use serde::Serialize;

/// A heightfield terrain on this entity as a scene version 9 `terrain` field:
/// its origin, tiles, spacing, layers or file, material, collision and
/// surroundings, with its streaming state: resident tiles by level, and its
/// surroundings' once they stream.
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
    /// Its hydrology bake, when its layers hold one.
    pub hydrology: Option<HydrologyView>,
    /// Its erosion bake, when its layers hold one.
    pub erosion: Option<ErosionView>,
    /// Its path layers, in stack order.
    pub paths: Vec<PathView>,
    /// Its edits, in order: each one's `id` and its `stamp`,
    /// terrain-local, which a scene saves as a `{"stamp": ...}` layer after
    /// its others.
    pub edits: Vec<TerrainEditView>,
    /// Its surroundings once they stream, after its bakes:
    /// how far they reach past its edges along x and z, their level-0 tiles
    /// along a side and sample spacing, and their resident tiles and the
    /// patches drawn.
    pub surroundings: Option<SurroundingsView>,
}

/// A terrain's surroundings as they stream.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SurroundingsView {
    pub reach: [f32; 2],
    pub tiles: u32,
    pub spacing: f32,
    pub resident_tiles: u32,
    pub patches: u32,
}

/// One terrain edit: its id and its stamp (terrain-local).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainEditView {
    pub id: String,
    pub stamp: engine_core::terrain_data::Stamp,
}

/// A path layer: its index in the stack, whether its
/// profile is baked, and once baked its length (meters), the points its
/// profile read the ground at, its steepest grade (rise per meter), the
/// most it cuts into and fills over the ground along its line off bridges
/// (meters), its bridges and the meters they span, and its
/// tunnels and the meters they run.
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

/// A terrain's erosion bake: whether a bake is in progress
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

/// A terrain's hydrology bake: whether it is in progress
/// (tiles wait for it), and once done the samples it read, those channels
/// run through, those of filled pits, the largest area draining through
/// one (square meters) and the waterfalls rivers drop over.
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

/// The log of runtime deformation's stamps: stamps held, pressed and dropped by
/// the deformation budget, and the newest. Read tracks with deformation_query;
/// press one with the deform control.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TracksView {
    pub stamps: usize,
    pub pressed: u64,
    pub dropped: u64,
    /// The newest stamp: where tracks are being made.
    pub latest: Option<engine_core::Track>,
}

/// World streaming at the last extraction: cells holding streamed items, loaded
/// and pending (within range, waiting), items, entities streamed items hold,
/// the frame's spawns and despawns, the camera velocity prefetch followed, each
/// grid's cells and the 32 cells nearest a streaming source. Settings: the
/// world_streaming environment section.
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

/// One streaming grid: its cells' side, the meters within
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

/// One streamed cell: its grid (0 small, 1 medium, 2
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

/// A navigation agent as a scene version 9 `nav_agent` field: it walks over the
/// terrain to its destination (world x, z) at its speed, on the ground, around
/// obstacles widened by its radius, no steeper than max_slope, wading no deeper
/// than max_water; with its walk's status, length, points, next point, meters
/// left and samples searched. Send it with nav_go.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct NavAgentView {
    pub agent: engine_core::NavAgent,
    pub status: engine_core::NavStatus,
    pub searched_for: Option<[f32; 2]>,
    pub length: f32,
    pub points: usize,
    pub next: usize,
    pub left: f32,
    pub searched: usize,
}

/// What navigation did at the last fixed tick: agents, those searching,
/// walking, arrived and unable to reach their destination, the samples of
/// search work the tick did, and the first 16 agents with their status,
/// position, destination and meters left. Settings: the navigation environment
/// section.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct NavAgentsView {
    pub agents: u32,
    pub searching: u32,
    pub walking: u32,
    pub arrived: u32,
    pub unreachable: u32,
    pub work: u64,
    pub first: Vec<NavAgentRow>,
}

/// The host's job workers: workers (none keeps bakes, tiles and walks on the
/// engine's thread), jobs in flight, submitted, completed, failed and
/// cancelled, parts handed back early, keys shared and their bytes, and the
/// page's output cache. Workers run their most urgent job a slice at a time.
/// Pages take ?workers=N or off, and ?jobcache=off, verify or a budget in MiB.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct JobsView {
    pub workers: u32,
    pub running: u32,
    pub submitted: u64,
    pub completed: u64,
    pub failed: u64,
    pub cancelled: u64,
    /// Parts of outputs handed back before the rest, as
    /// tiles' shapes.
    pub parts: u64,
    pub shares: u32,
    pub shared_bytes: u64,
    /// The host's output cache: outputs it handed back
    /// instead of running their jobs, jobs it lacked, its bytes and,
    /// verifying, outputs that differed from a fresh run.
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_bytes: u64,
    pub cache_mismatches: u64,
}

/// One agent of [`NavAgentsView`].
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct NavAgentRow {
    pub id: Option<String>,
    pub status: engine_core::NavStatus,
    pub position: [f32; 3],
    pub destination: Option<[f32; 2]>,
    pub left: f32,
}

/// What gameplay changed of the streamed world, which games save: items removed
/// for good (an item is removed when gameplay despawns its root), by ID
/// (`scene/root` for scenes), which their cells no longer spawn. The count and
/// the first 64 IDs. Change it with the world_changes control.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WorldChangesView {
    pub removed: usize,
    pub ids: Vec<String>,
}

/// The streamed item whose root this entity is: its ID. Despawning the entity
/// removes the item for good (WorldChanges); its cell no longer spawns it.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StreamedIdView {
    pub id: String,
}

/// Stands on the terrain as a scene version 9 `grounded` field: the entity's
/// translation's y is kept `offset` meters above the ground under it, from the
/// finest terrain tile resident there, sampling again as finer tiles arrive and
/// when its x, z or offset changes; with the level of the tile it last stood
/// on. Unparented entities only.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct GroundedView {
    pub offset: f32,
    pub level: Option<u32>,
}

/// Terrain streaming at the last extraction: terrains; resident, loading,
/// shape-only, waiting, kept, restored and idle, banded and dropped tiles;
/// patches drawn and those standing in for finer levels; and the surroundings'.
/// Query heights with terrain_height, compare queries, collision and layers
/// with terrain_sweep, and find routes for path layers with terrain_route.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TerrainsView {
    pub terrains: u32,
    pub resident_tiles: u32,
    pub loading_tiles: u32,
    /// Of the resident tiles, those holding their shape alone so far:
    /// drawn with coarser levels' materials while the rest
    /// loads.
    pub shaped_tiles: u32,
    /// Tiles the view wants that wait while the camera moves fast.
    pub waiting_tiles: u32,
    /// Tiles kept in memory after the view left them and
    /// their bytes, tiles returned from there since terrains started, and
    /// tiles generating ahead of the view.
    pub cached_tiles: u32,
    pub cached_bytes: u64,
    pub restored_tiles: u64,
    pub idle_tiles: u32,
    /// Tiles that generated in bands of rows on several workers since
    /// terrains started.
    pub banded_tiles: u64,
    pub dropped_tiles: u32,
    pub patches: u32,
    pub fallback_patches: u32,
    /// Terrains whose bakes are in progress.
    pub baking: u32,
    /// Bridges terrains' roads span, tunnels they bore,
    /// and waterfalls their rivers drop over.
    pub bridges: u32,
    pub tunnels: u32,
    pub falls: u32,
    /// Of the resident tiles and patches, those of terrains' surroundings.
    pub surroundings_tiles: u32,
    pub surroundings_patches: u32,
}

/// An animation player as a scene version 8 `animation` field with its clip's
/// duration and its targets: every fixed tick it advances `time` by the tick
/// times `speed`, wrapping when `loop` is set, and poses its targets'
/// transforms.
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

/// A decal on this entity as a scene version 8 `decal` field: its material
/// projected down the entity's local y onto the opaque surfaces inside its unit
/// box, fading on surfaces turning away beyond `fade_degrees`.
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

/// A mesh bent along a curve on this entity as a scene version 9 `spline`
/// field: copies of the mesh end to end along a centripetal Catmull-Rom curve
/// through `points` (entity space), its x along the curve, y up and z across,
/// drawn by child entities a chunk of at most 256 m each; with the curve's
/// length, copies, chunks and vertices once built, or the error that stopped
/// it.
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

/// The joints deforming this entity's mesh, as a scene version 8 `skin` names
/// them: each vertex moves by its weighted joints' displayed world matrices
/// times their inverse bind matrices.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct SkinView {
    /// Each joint's authored entity ID; None when not authored.
    pub joints: Vec<Option<String>>,
}

/// Local transform relative to Parent, or world transform for a root.
/// Quaternion is normalized XYZW; authored scale is positive and nonsingular.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TransformView {
    pub translation: [f32; 3],
    /// Quaternion components in XYZW order.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

/// Derived world transform, refreshed before inspection. Column-major matrix
/// preserves shear; invalid hierarchies return null matrix and translation and
/// skip rendering.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct GlobalTransformView {
    pub valid: bool,
    /// Column-major matrix; preserves shear from rotated, nonuniform ancestors.
    pub matrix: Option<[f32; 16]>,
    pub translation: Option<[f32; 3]>,
}

/// Opt-in marker for fixed-tick interpolation of this local transform.
/// Descendants inherit the presented parent matrix. Gameplay and authored
/// transforms remain authoritative.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InterpolateTransformView {
    /// Marker presence opts this local transform into fixed-tick interpolation.
    pub enabled: bool,
}

/// Last extracted world matrix, including interpolated ancestors; column-major
/// and preserves shear. Inspection does not recompute it, except to materialize
/// the exact CPU reference of a GPU-presented frame while its inputs are
/// unchanged. Null until extracted or when the presented branch is invalid. See
/// PresentationState for actual mode, alpha and gpu_pending.
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

/// Last prepared extraction's frame/tick and actual mode/fraction; null
/// frame/tick before extraction. Current mode uses alpha one. History counters
/// measure bounded storage. gpu_pending marks a GPU-presented extraction whose
/// CPU poses are not materialized. Inspection prepares presentation only to
/// materialize such a frame; this is not GPU submission/completion or total
/// process memory.
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
    /// The last extraction was blended on the GPU and CPU
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

/// Current parent and source prefab metadata. IDs are authored IDs, never
/// runtime handles. Use inspect with parent to page direct children.
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

/// Right-handed Y-up perspective. Lens distances use world units; 0 < near <
/// far; FOV is in radians.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CameraView {
    pub position: [f32; 3],
    pub target: [f32; 3],
    pub vertical_fov_radians: f32,
    pub near: f32,
    pub far: f32,
}

/// Resolved immutable mesh/material readiness. Missing assets skip rendering.
/// This view contains no persistent runtime handles.
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
    /// transparent.
    pub opacity: Option<f32>,
    /// The material's response to light.
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

/// Local cuboid/sphere, symmetric layer filters and sensor flag. Current
/// authoritative world admission and refreshed collision-cache status are
/// independent of visibility and interpolation.
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

/// Authoritative spatial cache refreshed for inspection without advancing
/// clocks or committing sensor events. Includes source frame/tick, cache
/// revision, completeness, invalid/overflow counts and query budgets. Memory
/// counts are cache payload, not process totals.
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
    /// Placed objects colliding within terrains' collision radius.
    pub placed_colliders: usize,
    pub limits: CollisionLimitsView,
}

/// Per-object color factors as a scene version 8 `tint` field: red, green, blue
/// and opacity multiply this object's material or cube color, each in [0, 1].
/// Changing it uploads only this object's colors.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(transparent)]
pub struct TintView(pub [f32; 4]);
