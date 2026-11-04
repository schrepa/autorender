use serde::Serialize;

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InputView {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub look_right_pixels: f32,
    pub look_down_pixels: f32,
    pub held: Vec<u32>,
    pub pointer_pixels: Option<[f32; 2]>,
    pub pointer_delta_pixels: [f32; 2],
    pub pointer_inside: bool,
    pub wheel_up_notches: f32,
    pub wheel_right_notches: f32,
    pub axes: Vec<f32>,
    pub menu: Vec<crate::MenuPress>,
    pub menu_wanted: bool,
    pub cursor: &'static str,
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
pub struct PresentationStateView {
    pub frame: Option<String>,
    pub tick: Option<String>,
    pub alpha: f32,
    pub history_nodes: usize,
    pub history_capacity: usize,
    pub history_bytes: usize,
    pub presentation_scratch_bytes: usize,
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
    pub vertical_fov_radians: Option<f32>,
    pub height: Option<f32>,
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
    pub provided: Option<String>,
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
