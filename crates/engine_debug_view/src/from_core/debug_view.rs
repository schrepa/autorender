use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_core::sections::environment_section;
use engine_settings::{Setting, settings};

macro_rules! modes {
    ($($(#[$doc:meta])* $mode:ident = $name:literal,)*) => {
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
        pub enum DebugViewMode {
            $($(#[$doc])* #[serde(rename = $name)] $mode,)*
        }

        impl DebugViewMode {
            pub const NAMES: &'static [&'static str] = &[$($name),*];

            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$mode => $name,)*
                }
            }

            pub fn named(name: &str) -> Option<Self> {
                match name {
                    $($name => Some(Self::$mode),)*
                    _ => None,
                }
            }
        }
    };
}

modes! {
    #[default]
    Off = "off",
    Kinds = "kinds",
    Levels = "levels",
    Cards = "cards",
    Shadows = "shadows",
    Casters = "casters",
    Overdraw = "overdraw",
    Normals = "normals",
    Tiles = "tiles",
    Materials = "materials",
    Sky = "sky",
    Detail = "detail",
    WaterDepth = "water_depth",
    WaterSpeed = "water_speed",
    WaterFroude = "water_froude",
    WaterTurbulence = "water_turbulence",
    WaterWhite = "water_white",
    WaterReach = "water_reach",
    WaterSpan = "water_span",
    WaterEddies = "water_eddies",
    WaterRoughness = "water_roughness",
}

impl Setting for DebugViewMode {}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDebugView")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct DebugView {
        pub view: DebugViewMode = DebugViewMode::Off => any,
        pub shade: bool = true => any,
    }
}

environment_section!(DebugView, "debug_view", opt_in: true);

impl DebugView {
    pub fn off() -> Self {
        Self {
            view: DebugViewMode::Off,
            ..Self::default()
        }
    }

    pub fn showing(&self) -> bool {
        self.view != DebugViewMode::Off
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DrawnKind {
    Mesh,
    Placed,
    Card,
    PlacedCard,
    #[serde(rename = "terrain")]
    Ground,
    #[serde(rename = "cover")]
    Blades,
    #[serde(rename = "water")]
    Lakes,
    Falls,
    Sky,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SunLookup {
    None,
    Level,
    Horizon,
    Beyond,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MaterialShare {
    pub material: u32,
    pub share: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DetailShares {
    pub detailed: f32,
    pub averaged: f32,
    pub canopy: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainProbe {
    pub tile: [i32; 2],
    pub patch: [u32; 2],
    pub materials: Vec<MaterialShare>,
    pub sky: f32,
    pub detail: DetailShares,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WaterReach {
    Pool,
    Plane,
    Steps,
    Cascade,
    Falls,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WaterProbe {
    pub depth: f32,
    pub speed: f32,
    pub froude: f32,
    pub turbulence: f32,
    pub white: f32,
    pub reach: Option<WaterReach>,
    pub span: f32,
    pub eddies: f32,
    pub roughness: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PixelProbe {
    pub pixel: [u32; 2],
    pub kind: Option<DrawnKind>,
    pub level: u32,
    pub fading: bool,
    pub casts_shadows: bool,
    pub object: Option<u32>,
    pub detail: Option<u32>,
    pub normal: Option<[f32; 3]>,
    pub sun: SunLookup,
    pub sun_index: u32,
    pub shadowed: bool,
    pub fragments: u32,
    pub depth: f32,
    pub world: Option<[f32; 3]>,
    pub shaded: Option<[u8; 3]>,
    pub displayed: Option<[u8; 3]>,
    pub terrain: Option<TerrainProbe>,
    pub water: Option<WaterProbe>,
}
