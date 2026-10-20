//! Debug views: frames that show, instead of the shaded image, what drew
//! each pixel and how: its kind of geometry, its level of detail, whether it
//! is an impostor card, which sun shadow map decided its shadow, how it
//! casts sun shadows, or how many fragments covered it. Renderers draw them
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
    /// Its level of detail, from the finest: green, yellow, orange, red,
    /// violet, blue, cyan and white, for meshes and placed objects (their own
    /// mesh, then each simplification) and terrain's patches alike;
    /// impostor cards magenta, virtual geometry a color per cluster. Frames
    /// GPU presentation draws (`capture` with `alpha` and `gpu`) show levels
    /// and cards as play does; exact frames draw every mesh whole.
    Levels = "levels",
    /// Impostor cards magenta (cards fading over their meshes pink) over
    /// everything else gray: where cards stand in for meshes.
    Cards = "cards",
    /// Which sun shadow lookup decided it: the near cascades green, from
    /// bright (the first) to dark, the kept levels cyan to blue, the kept far
    /// map indigo, the moving map orange, a terrain's horizon purple, past
    /// the shadow distance gray; darker where it lies in shadow, black
    /// without sun shadows.
    Shadows = "shadows",
    /// How it casts sun shadows: meshes red, placed objects orange (yellow
    /// where the kept map holds them), virtual geometry violet, impostor
    /// cards casting cards of their own magenta, terrain brown, what casts
    /// none gray.
    Casters = "casters",
    /// How many fragments passed the depth test there as the frame drew,
    /// from dark blue (one) through green and yellow to white (ten or more).
    Overdraw = "overdraw",
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
    /// The near cascade the probe's `sun_index` names, from the first.
    Cascade,
    /// The kept level `sun_index` names, from the finest.
    Level,
    /// The kept far map, of still casters.
    Kept,
    /// The far map redrawn every frame, of moving casters.
    Moving,
    /// A terrain's horizon, below which the sun is gone.
    Horizon,
    /// Past the shadow distance: lit.
    Beyond,
}

/// What drew one pixel and how it looks: the pixel probe's answer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PixelProbe {
    /// The pixel, in physical pixels from the view's top left.
    pub pixel: [u32; 2],
    /// What drew it; `None` where nothing did.
    pub kind: Option<DrawnKind>,
    /// Its level of detail: meshes' and placed objects' 0 for their own
    /// mesh and 1 up for their simplifications, impostor cards' the level
    /// past the simplifications, terrain's patch level.
    pub level: u32,
    /// Whether it is an impostor card fading in or out over its mesh.
    pub fading: bool,
    /// Whether what drew it casts sun shadows, and for placed objects
    /// whether the kept map holds them.
    pub casts_shadows: bool,
    pub kept_shadows: bool,
    /// The object it drew: an entity's renderer slot (meshes, cards and
    /// virtual geometry) or the placed object's index.
    pub object: Option<u32>,
    /// A part of the object: virtual geometry's cluster, an impostor
    /// card's record.
    pub detail: Option<u32>,
    /// Its shading normal, unit length; `None` for the sky.
    pub normal: Option<[f32; 3]>,
    /// Which lookup decided its sun shadow, which cascade or level among
    /// several, and whether it leaves the pixel in shadow.
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
}
