//! What each development op answers, typed: a successful reply's `result`
//! is its op's result type below, or the op is one [`UNTYPED_RESULTS`]
//! lists. Hosts add to every result what they carried out (the reply's
//! documentation says what). The typed results' schemas are generated
//! beside the reply's, as `OpResults` in the page's reply types.

use engine_core::Distribution;
use serde::Serialize;
use serde_json::Value;

use crate::views::{CollisionWorldView, PresentationStateView};

/// Ops whose results are still built as untyped JSON, sorted. The list only
/// shrinks: a new op's result is typed, in `results!` below.
pub const UNTYPED_RESULTS: &[&str] = &[
    "animation",
    "benchmark",
    "camera",
    "capture",
    "collision_events",
    "culling",
    "debug_panic",
    "deform",
    "deformation_query",
    "describe",
    "environment",
    "frames",
    "game",
    "input",
    "inspect",
    "nav_go",
    "nav_path",
    "overlap",
    "pause",
    "present",
    "raycast",
    "reload",
    "render",
    "resume",
    "shape_cast",
    "step",
    "terrain",
    "terrain_edit",
    "terrain_height",
    "terrain_redo",
    "terrain_route",
    "terrain_sweep",
    "terrain_undo",
    "water_query",
    "world_changes",
];

/// Declares each typed op's result type, `op: Type`: handlers hand a result
/// on through `result::op`, which takes nothing else.
macro_rules! results {
    ($($op:ident: $result:ty),* $(,)?) => {
        /// Each typed op's result as its reply carries it.
        pub mod result {
            $(
                #[doc = concat!("The `", stringify!($op), "` op's result as its reply carries it.")]
                pub fn $op(result: $result) -> serde_json::Value {
                    serde_json::to_value(result).expect("results serialize")
                }
            )*
        }

        /// The ops whose results are typed.
        pub const TYPED_RESULTS: &[&str] = &[$(stringify!($op)),*];

        /// Each typed op with its result's schema.
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
}

/// The engine as a session finds it: the world's size, the inspectors,
/// features and limits it has, and what it presents.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StatusResult {
    /// Entities in the world.
    pub world_entities: u32,
    /// Entities the authored scene expands to, where one runs.
    pub authored_entities: Option<usize>,
    /// Inspectors registered.
    pub inspectors: usize,
    /// Whether frames cull what the camera does not see.
    pub culling_enabled: bool,
    /// The features whose plugins the engine has.
    pub features: Option<Vec<&'static str>>,
    /// What the last frame presented.
    pub presentation: PresentationStateView,
    /// The collision world, where the engine has one.
    pub collision: Option<CollisionWorldView>,
    /// Seconds a fixed tick advances.
    pub fixed_delta_seconds: Option<f32>,
    /// What requests and replies keep within.
    pub limits: StatusLimits,
}

/// What requests and replies keep within.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct StatusLimits {
    /// Bytes of a request at most.
    pub request_bytes: usize,
    /// Bytes of a reply at most.
    pub response_bytes: usize,
    /// Entries a page holds at most.
    pub page_size: usize,
    /// Authored entities a lookup scans at most.
    pub scan_entities: usize,
}

/// What the asset library holds, and each of the scene's textures.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct AssetsResult {
    /// Meshes the library holds.
    pub meshes: usize,
    /// Materials the library holds.
    pub materials: usize,
    /// Textures the library holds.
    pub textures: usize,
    /// Bytes of the textures' pixels.
    pub texture_cpu_bytes: usize,
    /// Each texture the authored scene names.
    pub texture_assets: Vec<TextureAsset>,
    /// Bytes the library's payloads hold.
    pub resident_cpu_bytes: usize,
    /// When assets are ready: `synchronous`, as they are inserted.
    pub readiness: &'static str,
    /// What the byte counts leave out.
    pub ownership: &'static str,
}

/// One texture the authored scene names: its ID, whether the library holds
/// it, its size and whether the GPU holds it.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct TextureAsset {
    /// The texture's authored ID.
    pub id: String,
    /// Whether the library holds its pixels.
    pub state: TextureState,
    /// Its width in pixels, where held.
    pub width: Option<u32>,
    /// Its height in pixels, where held.
    pub height: Option<u32>,
    /// Bytes of its pixels, where held.
    pub bytes: Option<usize>,
    /// Whether the GPU holds it, where a renderer runs.
    pub gpu_resident: Option<bool>,
}

/// Whether the library holds a texture's pixels.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum TextureState {
    CpuReady,
    Missing,
}

/// The world's content as hashes that depend on neither entity ids nor
/// spawn order (the `digest` query's documentation says how).
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DigestResult {
    /// Entities in the world.
    pub entities: usize,
    /// Per component inspector, the entities holding its component and a
    /// hash of their views.
    #[cfg_attr(
        not(target_arch = "wasm32"),
        schemars(with = "std::collections::BTreeMap<String, DigestEntry>")
    )]
    pub components: Value,
    /// Per kind of asset, how many the library holds and a hash of their
    /// contents; null without a library.
    #[cfg_attr(not(target_arch = "wasm32"), schemars(with = "Option<AssetDigests>"))]
    pub assets: Value,
    /// One hash over the components and assets.
    pub hash: String,
}

/// How many of a kind and a hash of them.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DigestEntry {
    /// How many.
    pub count: usize,
    /// The first 16 hexadecimal digits of their content hash.
    pub hash: String,
}

/// Each kind of asset's digest.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct AssetDigests {
    /// Meshes, slot by slot.
    pub meshes: DigestEntry,
    /// Materials, slot by slot.
    pub materials: DigestEntry,
    /// Textures, slot by slot.
    pub textures: DigestEntry,
    /// Animation clips, slot by slot.
    pub clips: DigestEntry,
}

/// The spread of the values a `distribution` request named.
#[derive(Serialize)]
#[cfg_attr(not(target_arch = "wasm32"), derive(schemars::JsonSchema))]
pub struct DistributionResult {
    /// Null without values, or with one not a finite number of at least 0.
    pub distribution: Option<Distribution>,
}
