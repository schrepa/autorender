use serde::Serialize;

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
    /// Compatibility field: zero because presented poses are no longer ECS components.
    pub presented_component_bytes: usize,
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
    pub limits: CollisionLimitsView,
}
