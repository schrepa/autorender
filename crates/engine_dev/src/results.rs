use engine_core::Distribution;
use serde::Serialize;
use serde_json::Value;

use crate::views::{CollisionWorldView, PresentationStateView};
use crate::{InspectorDescriptor, InspectorKind};

pub const UNTYPED_RESULTS: &[&str] = &["benchmark", "debug_device_loss", "debug_panic", "reload"];

macro_rules! results {
    ($($op:ident: $result:ty),* $(,)?) => {
        pub mod result {
            $(
                pub fn $op(result: $result) -> serde_json::Value {
                    serde_json::to_value(result).expect("results serialize")
                }
            )*
        }

        pub const TYPED_RESULTS: &[&str] = &[$(stringify!($op)),*];

        #[cfg(not(target_arch = "wasm32"))]
        pub fn result_schemas() -> Vec<(&'static str, schemars::Schema)> {
            vec![$((stringify!($op), schemars::schema_for!($result))),*]
        }
    };
}

results! {
    status: crate::results::StatusResult,
    assets: crate::results::AssetsResult,
    digest: crate::results::DigestResult,
    distribution: crate::results::DistributionResult,
    inspect: crate::results::InspectResult,
    describe: crate::results::DescribeResult,
    raycast: crate::features::collision::RaycastResult,
    overlap: crate::features::collision::OverlapResult,
    shape_cast: crate::features::collision::ShapeCastResult,
    collision_events: crate::features::collision::CollisionEventsResult,
    nav_path: crate::features::navigation::NavPathResult,
    nav_go: crate::features::navigation::NavGoResult,
    water_query: crate::features::ocean::WaterQueryResult,
    deformation_query: crate::features::deformation::DeformationQueryResult,
    deform: crate::features::deformation::DeformResult,
    animation: crate::features::animation::AnimationResult,
    world_changes: crate::features::world_streaming::WorldChangesResult,
    terrain: crate::features::terrain::TerrainControlResult,
    present: crate::results::PresentResult,
    culling: crate::results::CullingResult,
    camera: crate::results::CameraResult,
    render: crate::results::RenderResult,
    debug_fault: crate::results::DebugFaultResult,
    debug_missing_asset: crate::results::DebugMissingAssetResult,
    input: crate::results::InputResult,
    step: crate::results::StepResult,
    environment: crate::results::EnvironmentResult,
    terrain_height: crate::features::terrain::TerrainHeightResult,
    terrain_sweep: crate::features::terrain::TerrainSweepResult,
    terrain_route: crate::features::terrain::TerrainRouteResult,
    terrain_edit: crate::features::terrain::TerrainEdited,
    terrain_undo: crate::features::terrain::TerrainEdited,
    terrain_redo: crate::features::terrain::TerrainEdited,
    terrain_river: crate::features::terrain::TerrainEdited,
    frames: crate::results::FramesResult,
    capture: crate::results::CaptureResult,
    probe: crate::results::ProbeResult,
    game: crate::results::GameResult,
    pause: crate::results::StatusResult,
    resume: crate::results::StatusResult,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StatusResult {
    pub world_entities: u32,
    pub authored_entities: Option<usize>,
    pub inspectors: usize,
    pub culling_enabled: bool,
    pub features: Option<Vec<&'static str>>,
    pub presentation: PresentationStateView,
    pub collision: Option<CollisionWorldView>,
    pub fixed_delta_seconds: Option<f32>,
    pub limits: StatusLimits,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StatusLimits {
    pub request_bytes: usize,
    pub response_bytes: usize,
    pub page_size: usize,
    pub scan_entities: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct AssetsResult {
    pub meshes: usize,
    pub materials: usize,
    pub textures: usize,
    pub texture_cpu_bytes: usize,
    pub texture_assets: Vec<TextureAsset>,
    pub resident_cpu_bytes: usize,
    pub readiness: &'static str,
    pub ownership: &'static str,
    pub missing: Vec<MissingAsset>,
    pub missing_total: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct MissingAsset {
    pub entity: AuthoredEntity,
    pub draws: &'static str,
    pub missing: &'static str,
    pub position: Option<[f32; 3]>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TextureAsset {
    pub id: String,
    pub state: TextureState,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub bytes: Option<usize>,
    pub gpu_resident: Option<bool>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TextureState {
    CpuReady,
    Missing,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DigestResult {
    pub entities: usize,
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(with = "std::collections::BTreeMap<String, DigestEntry>")
    )]
    pub components: Value,
    #[cfg_attr(not(target_arch = "wasm32"), schemars(with = "Option<AssetDigests>"))]
    pub assets: Value,
    pub hash: String,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DigestEntry {
    pub count: usize,
    pub hash: String,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct AssetDigests {
    pub meshes: DigestEntry,
    pub materials: DigestEntry,
    pub textures: DigestEntry,
    pub clips: DigestEntry,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum InspectResult {
    Resource {
        resource: String,
        value: Value,
    },
    Entity {
        entity: EntityRow,
    },
    Page {
        entities: Vec<EntityRow>,
        offset: usize,
        limit: usize,
        total: usize,
        next_offset: Option<usize>,
    },
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct EntityRow {
    pub id: String,
    pub name: String,
    pub tags: Option<Vec<String>>,
    pub components: EntityComponents,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum EntityComponents {
    Values(serde_json::Map<String, Value>),
    Names(Vec<String>),
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(untagged)]
pub enum DescribeResult {
    Listing {
        types: Vec<DescribedType>,
    },
    Protocol {
        name: String,
        kind: TypeKind,
        schema: Value,
        constraints: String,
        control_state: serde_json::Map<String, Value>,
    },
    Authoring {
        name: String,
        kind: TypeKind,
        schema: Value,
        constraints: String,
    },
    Observation {
        name: String,
        kind: TypeKind,
        schema: Value,
        source: &'static str,
    },
    Catalog {
        name: String,
        kind: TypeKind,
        codes: Vec<&'static str>,
    },
    Inspector {
        kind: TypeKind,
        descriptor: InspectorDescriptor,
        feature: Option<&'static str>,
    },
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DescribedType {
    pub name: String,
    pub kind: TypeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub feature: Option<Option<&'static str>>,
}

#[derive(Clone, Copy, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    Component,
    Resource,
    Authoring,
    Protocol,
    Observation,
    Catalog,
}

impl From<InspectorKind> for TypeKind {
    fn from(kind: InspectorKind) -> Self {
        match kind {
            InspectorKind::Component => Self::Component,
            InspectorKind::Resource => Self::Resource,
        }
    }
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct PresentResult {
    pub temporary: bool,
    pub rendered: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CullingResult {
    pub temporary: bool,
    pub culling_enabled: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CameraResult {
    pub temporary: bool,
    pub camera_changed: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct RenderResult {
    pub rendered: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DebugFaultResult {
    pub injected: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DebugMissingAssetResult {
    pub temporary: bool,
    pub entities: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct InputResult {
    pub temporary: bool,
    pub applies_on: &'static str,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StepResult {
    pub ticks: u32,
    pub temporary: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct EnvironmentResult {
    pub temporary: bool,
    pub environment_changed: Vec<&'static str>,
    pub before: serde_json::Map<String, Value>,
    pub after: serde_json::Map<String, Value>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct FramesResult {
    pub frames: u32,
    pub interval_ms: f64,
    pub temporary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<KeptArtifact>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct CaptureResult {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temporary: Option<bool>,
    pub rendered: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<KeptArtifact>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ProbeResult {
    #[serde(flatten)]
    pub probe: engine_core::PixelProbe,
    pub entity: Option<AuthoredEntity>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct AuthoredEntity {
    pub id: Option<String>,
    pub scene: Option<String>,
    pub name: Option<String>,
}

impl AuthoredEntity {
    pub fn of(
        world: &engine_core::bevy_ecs::world::World,
        entity: engine_core::bevy_ecs::entity::Entity,
    ) -> Self {
        let metadata = world.get::<engine_scene::SceneEntity>(entity);
        Self {
            id: metadata.map(|metadata| metadata.authored_id.clone()),
            scene: metadata.map(|metadata| metadata.scene_id.clone()),
            name: metadata.map(|metadata| metadata.name.clone()),
        }
    }
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct KeptArtifact {
    pub path: String,
    pub mime_type: String,
    pub bytes: u64,
    pub sha256: String,
    pub frame: Option<String>,
    pub tick: Option<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct GameResult {
    pub game: String,
    pub result: Value,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DistributionResult {
    pub distribution: Option<Distribution>,
}
