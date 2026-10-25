//! Debug views: frames that show, instead of the shaded image, what drew
//! each pixel and how: its kind of geometry, its level of detail, whether it
//! is an impostor card, which level of sun shadows decided its shadow, how it
//! casts sun shadows, how many fragments covered it, its normal, what
//! terrain's ground is made of: its tiles and patches, surface materials,
//! the sky it sees and how much detail it shows; or what terrains' water is
//! doing, each field of its state: its depth, speed, Froude number,
//! turbulence, white water, the reach it runs in, the foam it carries, how
//! far across its water body is, its eddies' turbulent Froude number and
//! the roughness its shading gives it. Renderers draw them
//! with pipelines of their own, created the first time a view shows, into
//! targets of their own; the shaded frame's pipelines, targets and passes
//! stay as they are, and a world without the settings draws no view.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::{Setting, settings};

/// Declares [`DebugViewMode`] and its names from one list: `Variant =
/// "name"`, the name scenes, controls and host switches give it.
macro_rules! modes {
    ($($(#[$doc:meta])* $mode:ident = $name:literal,)*) => {
        /// What a debug view shows each pixel as.
        #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
        #[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
        pub enum DebugViewMode {
            $($(#[$doc])* #[serde(rename = $name)] $mode,)*
        }

        impl DebugViewMode {
            /// Every mode's name, in order.
            pub const NAMES: &'static [&'static str] = &[$($name),*];

            /// The mode's name in scenes, controls and host switches.
            pub fn name(self) -> &'static str {
                match self {
                    $(Self::$mode => $name,)*
                }
            }

            /// The mode named `name`, if one is.
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
    /// The shaded image: no debug view.
    #[default]
    Off = "off",
    /// What drew it, a color per kind: meshes gray-blue, placed objects
    /// teal, impostor cards magenta, virtual geometry violet, terrain brown,
    /// ground cover green, water blue, the sky pale blue.
    Kinds = "kinds",
    /// Its level of detail, from the finest: red, orange, yellow, green,
    /// cyan, blue, violet and gray, for meshes and placed objects (their own
    /// mesh, then each simplification) and terrain's patches alike;
    /// impostor cards magenta, virtual geometry a color per cluster. Frames
    /// GPU presentation draws (`capture` with `alpha` and `gpu`) show levels
    /// and cards as play does; exact frames draw every mesh whole.
    Levels = "levels",
    /// Impostor cards magenta (cards fading over their meshes pink) over
    /// everything else gray: where cards stand in for meshes.
    Cards = "cards",
    /// Which sun shadow lookup decided it: the clipmap's levels green to
    /// blue, from the finest, a terrain's horizon purple, where no level is
    /// drawn gray; darker where it lies in shadow, black without sun
    /// shadows.
    Shadows = "shadows",
    /// How it casts sun shadows: meshes red, placed objects orange,
    /// virtual geometry violet, impostor cards casting cards of their own
    /// magenta, terrain brown, what casts none gray.
    Casters = "casters",
    /// How many fragments passed the depth test there as the frame drew,
    /// from dark blue (one) through green and yellow to white (ten or more).
    Overdraw = "overdraw",
    /// Its shading normal, as colors: x red, y green, z blue, each from 0
    /// (pointing back) to 1 (along the axis); terrain's its smooth
    /// surface's.
    Normals = "normals",
    /// Terrain's level-0 tiles and patches (sixteen samples a side): tile
    /// edges red, patch edges dark, the ground between gray; everything else
    /// as in `cards`.
    Tiles = "tiles",
    /// Terrain's surface materials, each in its own color, blended in the
    /// shares the ground shows them in.
    Materials = "materials",
    /// The share of the sky terrain's ground sees, unlit, from black (none)
    /// to white (all of it).
    Sky = "sky",
    /// How much detail terrain's ground shows: green where all its
    /// materials' maps show, yellow where their color alone does, red where
    /// their means stand in, blue where it shows as its cover's canopy.
    Detail = "detail",
    /// How deep terrains' water stands, tiles' and simulated alike, from
    /// pale (none) through blue (half a meter) to navy (two meters or
    /// more), on a square-root scale. The water's views draw water wherever
    /// it lies, however little of the pixel it covers, and show the sea,
    /// falls' sheets and everything else gray, the sky dark, all unlit.
    WaterDepth = "water_depth",
    /// How fast the water runs, from dark violet (still) through crimson to
    /// yellow (six meters a second or more).
    WaterSpeed = "water_speed",
    /// Its Froude number, its speed over its shallow waves' (√(g·depth)):
    /// calm water from deep blue (0) to pale cyan, shooting water from
    /// yellow to red (3 or more), water running at its waves' speed (1)
    /// black.
    WaterFroude = "water_froude",
    /// How turbulent it runs, from dark teal (0) through green to lime (1).
    WaterTurbulence = "water_turbulence",
    /// Where white water is born, from dark blue (none) to white (1).
    WaterWhite = "water_white",
    /// The reach of the stream it runs in, by the stream's slope: pools
    /// blue, plane beds cyan, steps and pools green, cascades orange, falls
    /// red; water no stream holds (lakes) violet.
    WaterReach = "water_reach",
    /// The foam it carries, as its flow churns it up, carries it and loses
    /// it (`engine_terrain::shallow_water`), from dark brown (none) through
    /// sand to white (the most water carries).
    WaterFoam = "water_foam",
    /// How far across its water body is, the fetch the wind has over it: a
    /// river's width, a lake's equal-area diameter, on a logarithmic scale
    /// from violet (a quarter meter) through teal (32 meters) to pale
    /// yellow (4096 meters or more).
    WaterSpan = "water_span",
    /// Its eddies' turbulent Froude number: the speed they turn at,
    /// (g·U·S·depth)^(1/3) for its speed U and its surface's slope S, over
    /// √(g·depth); from dark (calm) through orange, where they start
    /// breaking its surface white (`engine_terrain::BREAK_ONSET`), to white,
    /// where they break it white all over (`engine_terrain::BREAK_FULL`).
    WaterEddies = "water_eddies",
    /// The roughness its shading gives its surface, from black (0, a
    /// mirror) through copper to white (1): calm water's, widened by the
    /// slopes of the waves a pixel hides.
    WaterRoughness = "water_roughness",
}

impl Setting for DebugViewMode {}

settings! {
    /// Debug view settings: what frames show each pixel as, for agents and
    /// people finding out why a pixel looks as it does. Off by default; a
    /// view draws with its own pipelines, created the first time it shows,
    /// and leaves screen effects, decals, transparent surfaces and temporal
    /// antialiasing out.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDebugView")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct DebugView {
        /// What each pixel shows.
        pub view: DebugViewMode = DebugViewMode::Off => any,
        /// Whether its colors darken where surfaces turn from the sun, so
        /// shapes read.
        pub shade: bool = true => any,
    }
}

impl DebugView {
    /// What a world without these settings reads as: no debug view.
    pub fn off() -> Self {
        Self {
            view: DebugViewMode::Off,
            ..Self::default()
        }
    }

    /// Whether frames show a debug view.
    pub fn showing(&self) -> bool {
        self.view != DebugViewMode::Off
    }
}

/// What drew a pixel, as the pixel probe reads it from a debug view's
/// targets.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum DrawnKind {
    /// A mesh: an entity's, its own mesh or one of its simplifications.
    Mesh,
    /// An object a terrain placed, as its mesh or a simplification.
    Placed,
    /// An impostor card standing in for an entity's mesh.
    Card,
    /// An impostor card standing in for a placed object.
    PlacedCard,
    /// Virtual geometry's clusters.
    Virtual,
    /// A terrain's ground.
    #[serde(rename = "terrain")]
    Ground,
    /// Ground cover's blades.
    #[serde(rename = "cover")]
    Blades,
    /// A terrain's lakes and rivers.
    #[serde(rename = "water")]
    Lakes,
    Sea,
    /// Water falling from a river's bed.
    Falls,
    /// Nothing nearer: the sky.
    Sky,
}

/// Which lookup decided a pixel's sun shadow, in the order shading looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum SunLookup {
    /// No sun shadows: shadows are off, or the pixel is the sky.
    None,
    /// The clipmap's level the probe's `sun_index` names, from the finest.
    Level,
    /// A terrain's horizon, below which the sun is gone.
    Horizon,
    /// No level drawn where it lies: lit.
    Beyond,
}

/// A surface material's share of a terrain pixel's ground.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MaterialShare {
    /// Its index among the terrain's surface materials.
    pub material: u32,
    /// Its share, 0 to 1, to 1/255.
    pub share: f32,
}

/// How much detail a terrain pixel's ground shows, each a share of its
/// materials, 0 to 1, to 1/255.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct DetailShares {
    /// Where all their maps show.
    pub full: f32,
    /// Where their maps' color shows, with or without the rest.
    pub colored: f32,
    /// Where the cover's canopy stands in for the ground.
    pub canopy: f32,
}

/// What a terrain pixel's ground is made of, as shading computes it where
/// no virtual texture page stands in.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct TerrainProbe {
    /// The level-0 tile it lies in, 256 cells a side, counted along x and z
    /// from the terrain's origin.
    pub tile: [i32; 2],
    /// The patch it lies in within the tile, 16 cells a side, from the
    /// tile's corner.
    pub patch: [u32; 2],
    /// The surface materials shading blends there and their shares,
    /// heaviest first.
    pub materials: Vec<MaterialShare>,
    /// The share of the sky the ground sees, 0 (none) to 1 (all of it), to
    /// 1/255.
    pub sky: f32,
    /// How much detail the ground shows.
    pub detail: DetailShares,
}

/// The kind of reach a stream runs in where a pixel's water lies, by the
/// stream's slope (`engine_terrain::Reach`), as the pixel probe reads it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum WaterReach {
    /// Pools and riffles.
    Pool,
    /// A plane bed.
    Plane,
    /// Steps and pools.
    Steps,
    /// A cascade.
    Cascade,
    /// Falls.
    Falls,
}

impl From<engine_terrain::Reach> for WaterReach {
    fn from(reach: engine_terrain::Reach) -> Self {
        match reach {
            engine_terrain::Reach::Pool => Self::Pool,
            engine_terrain::Reach::Plane => Self::Plane,
            engine_terrain::Reach::Steps => Self::Steps,
            engine_terrain::Reach::Cascade => Self::Cascade,
            engine_terrain::Reach::Falls => Self::Falls,
        }
    }
}

/// What a terrain's water is doing at a pixel, tiles' or simulated, as the
/// water's debug views show it: its numbers to a half float's precision,
/// its shares to 1/255.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct WaterProbe {
    /// Meters of water over the ground.
    pub depth: f32,
    /// Meters a second it runs.
    pub speed: f32,
    /// Its speed over its shallow waves' (√(g·depth)): under 1 it runs
    /// calm, over 1 it shoots.
    pub froude: f32,
    /// How turbulent it runs, 0 to 1.
    pub turbulence: f32,
    /// How much white water is born there, 0 to 1.
    pub white: f32,
    /// The reach of the stream it runs in; `None` where no stream holds it
    /// (lakes).
    pub reach: Option<WaterReach>,
    /// The foam it carries, 0 (none) to 1 (the most water carries).
    pub foam: f32,
    /// Meters across its water body: a river's width, a lake's equal-area
    /// diameter.
    pub span: f32,
    /// Its eddies' turbulent Froude number: the speed they turn at over
    /// √(g·depth), from which they break its surface white.
    pub eddies: f32,
    /// The roughness its shading gives its surface, 0 to 1.
    pub roughness: f32,
}

/// What drew one pixel and how it looks: the pixel probe's answer.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PixelProbe {
    /// The pixel, in physical pixels from the view's top left.
    pub pixel: [u32; 2],
    /// What drew it; `None` where nothing did.
    pub kind: Option<DrawnKind>,
    /// Its level of detail: meshes' and placed objects' 0 for their own
    /// mesh and 1 up for their simplifications, impostor cards' the level
    /// past the simplifications, terrain's leaf level (the pyramid level
    /// whose spacing its subdivision leaf's meshlet vertices stand apart).
    pub level: u32,
    /// Whether it is an impostor card fading in or out over its mesh.
    pub fading: bool,
    /// Whether what drew it casts sun shadows.
    pub casts_shadows: bool,
    /// The object it drew: an entity's renderer slot (meshes, cards and
    /// virtual geometry) or the placed object's index.
    pub object: Option<u32>,
    /// A part of the object: virtual geometry's cluster, an impostor
    /// card's record.
    pub detail: Option<u32>,
    /// Its shading normal, unit length; `None` for the sky.
    pub normal: Option<[f32; 3]>,
    /// Which lookup decided its sun shadow, which level of the clipmap, and
    /// whether it leaves the pixel in shadow.
    pub sun: SunLookup,
    pub sun_index: u32,
    pub shadowed: bool,
    /// Fragments that passed the depth test there as the frame drew it.
    pub fragments: u32,
    /// Its depth in the depth buffer (reversed: 1 nearest, 0 the far
    /// plane), and where that lies in the world; `None` at the far plane.
    pub depth: f32,
    pub world: Option<[f32; 3]>,
    /// Its shaded color as the opaque passes drew it, before screen
    /// effects, as stored (sRGB, 0 to 255); `None` where the frame draws
    /// no screen effects.
    pub shaded: Option<[u8; 3]>,
    /// Its color as displayed, as stored (sRGB, 0 to 255); `None` where
    /// the host's canvas cannot be read back.
    pub displayed: Option<[u8; 3]>,
    /// What terrain's ground is made of there; `None` off terrain.
    pub terrain: Option<TerrainProbe>,
    /// What terrains' water is doing there; `None` off their water. Water
    /// covering under half the pixel reads as what lies under it, unless
    /// the settings show one of the water's views.
    pub water: Option<WaterProbe>,
}
