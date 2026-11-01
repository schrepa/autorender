use engine_core::__private::{schemars, serde};
use engine_core::{Cut, MaterialHandle, MeshHandle, settings, stats};

pub const MAX_OCCUPANCY_LEVELS: u32 = 8;

settings! {
    #[section("foliage")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneFoliage")]
    pub struct FoliageSettings {
        pub transmission: f32 = 1.0 => range(0.0, 4.0),
        pub variation: f32 = 1.0 => range(0.0, 1.0),
        pub hashed_cutoff: bool = false => any,
        pub ground_blend: f32 = 0.4 => range(0.0, 4.0),
        pub crown_depth: f32 = 2.0 => range(0.0, 8.0),
    }
}

settings! {
    #[section("occupancy")]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneOccupancy")]
    pub struct OccupancySettings {
        pub enabled: bool = true => any,
        pub radius: f32 = 512.0 => range(64.0, 4096.0),
        pub resolution: u32 = 1024 => range(128, 4096),
        pub levels: u32 = 4 => range(1, MAX_OCCUPANCY_LEVELS),
        pub strength: f32 = 0.75 => range(0.0, 1.0),
    }
}

settings! {
    #[section("scatter")]
    #[serde(deny_unknown_fields)]
    pub struct ScatterSettings {
        pub looks: Vec<PlacedLook> = Vec::new() => any,
        pub parts: u32 = 2 => any,
        pub lods: u32 = 4 => at_most(4),
    }
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct PlacedLook {
    pub mesh: MeshHandle,
    #[serde(default)]
    pub lods: Vec<PlacedLod>,
    #[serde(default)]
    pub parts: Vec<PlacedPart>,
    #[serde(default)]
    pub shadow: Option<MeshHandle>,
    #[serde(default)]
    pub sway: f32,
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct PlacedLod {
    pub mesh: MeshHandle,
    pub error: f32,
}

#[derive(
    Clone,
    Debug,
    PartialEq,
    serde::Serialize,
    serde::Deserialize,
    schemars::JsonSchema,
)]
#[serde(crate = "engine_core::__private::serde", deny_unknown_fields)]
#[schemars(crate = "engine_core::__private::schemars")]
pub struct PlacedPart {
    pub material: MaterialHandle,
    pub meshes: Vec<MeshHandle>,
}

stats! {
    #[section("scatter_stats")]
    pub struct ScatterStats {
        pub placed_objects: u32,
        pub placed_far_objects: u32,
        pub placed_impostors: u32,
        pub dropped_objects: u32,
        pub impostors: Cut,
        pub parts: Cut,
        pub lods: Cut,
        pub occupancy_redraws: u32,
        pub occupancy_averages: u32,
    }
}

impl ScatterSettings {
    pub(crate) fn look(&self, mesh: MeshHandle) -> Option<&PlacedLook> {
        self.looks.iter().find(|look| look.mesh == mesh)
    }
}

impl OccupancySettings {
    pub(crate) fn runs(&self) -> bool {
        self.enabled && self.strength > 0.0
    }
}
