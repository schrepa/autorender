use engine_core::Distribution;
use serde::Serialize;
use serde_json::Value;

use crate::views::PresentationStateView;
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
    describe: crate::results::DescribeResult,
    inspect: crate::results::InspectResult,
    status: crate::results::StatusResult,
    assets: crate::results::AssetsResult,
    digest: crate::results::DigestResult,
    distribution: crate::results::DistributionResult,
    present: crate::results::PresentResult,
    culling: crate::results::CullingResult,
    camera: crate::results::CameraResult,
    render: crate::results::RenderResult,
    debug_fault: crate::results::DebugFaultResult,
    debug_missing_asset: crate::results::DebugMissingAssetResult,
    input: crate::results::InputResult,
    step: crate::results::StepResult,
    environment: crate::results::EnvironmentResult,
    frames: crate::results::FramesResult,
    probe: crate::results::ProbeResult,
    game: crate::results::GameResult,
    pause: crate::results::StatusResult,
    resume: crate::results::StatusResult,
    capture: crate::results::CaptureResult,
    profile: crate::results::ProfileResult,
    explain: crate::results::ExplainResult,
    watch: crate::results::WatchResult,
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
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(with = "Option<serde_json::Map<String, Value>>")
    )]
    pub collision: Option<Value>,
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
        verbs: Vec<&'static str>,
        bricks: Vec<DescribedBrick>,
        types: Vec<DescribedType>,
    },
    Verb {
        name: String,
        kind: TypeKind,
        schema: Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        control_state: Option<serde_json::Map<String, Value>>,
    },
    Authoring {
        name: String,
        kind: TypeKind,
        schema: Value,
        opt_in_sections: Vec<&'static str>,
    },
    Observation {
        name: String,
        kind: TypeKind,
        schema: Value,
    },
    Catalog {
        name: String,
        kind: TypeKind,
        codes: Vec<&'static str>,
    },
    Inspector {
        kind: TypeKind,
        descriptor: InspectorDescriptor,
        brick: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        at: Option<String>,
    },
    Operation {
        name: String,
        kind: TypeKind,
        verb: &'static str,
        brick: Option<&'static str>,
        #[serde(skip_serializing_if = "Option::is_none")]
        at: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        shows: Option<&'static str>,
        args: Value,
    },
    Brick {
        kind: TypeKind,
        brick: DescribedBrick,
    },
    Stats {
        name: String,
        kind: TypeKind,
        brick: &'static str,
        at: String,
        schema: Value,
    },
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DescribedType {
    pub name: String,
    pub kind: TypeKind,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub brick: Option<Option<&'static str>>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DescribedBrick {
    pub name: &'static str,
    #[serde(rename = "crate")]
    pub crate_name: &'static str,
    pub slot: Option<&'static str>,
    pub lists: Vec<&'static str>,
    pub settings: Vec<DescribedEntry>,
    pub stats: Vec<DescribedEntry>,
    pub operations: Vec<DescribedEntry>,
    pub inspectors: Vec<DescribedEntry>,
    pub explains: Vec<String>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DescribedEntry {
    pub name: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub key: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub verb: Option<&'static str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub at: Option<String>,
}

#[derive(Clone, Copy, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TypeKind {
    Component,
    Resource,
    Authoring,
    Verb,
    Operation,
    Brick,
    Stats,
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
    pub environment_changed: Vec<String>,
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
    #[serde(skip_serializing_if = "Option::is_none")]
    pub digest: Option<DigestResult>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ProfileResult {
    pub frames: u32,
    pub interval_ms: f64,
    pub cpu_ms: f64,
    pub gpu_ms: Option<f64>,
    pub bricks: Vec<ProfiledBrick>,
    pub total: usize,
    pub temporary: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub artifact: Option<KeptArtifact>,
}

#[derive(Clone, Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ProfiledBrick {
    pub brick: &'static str,
    pub cpu_ms: f64,
    pub cpu_ms_max: f64,
    pub gpu_ms: Option<f64>,
    pub gpu_ms_max: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heap_live_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub heap_peak_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub allocations_per_frame: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_after_warmup: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duplicate_upload_bytes: Option<u64>,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct ExplainResult {
    pub about: &'static str,
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(with = "Vec<serde_json::Map<String, Value>>")
    )]
    pub records: Vec<engine_core::ExplainRecord>,
    pub total: usize,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WatchResult {
    pub frames: u32,
    pub interval_ms: f64,
    pub records: u32,
    pub temporary: bool,
}

#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct WatchRecord {
    pub frame: Option<String>,
    pub tick: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stat: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub entity: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub component: Option<String>,
    #[cfg_attr(not(target_arch = "wasm32"), schemars(with = "Option<Value>"))]
    pub value: Value,
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
