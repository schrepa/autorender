//! Rendering effects of the scene: sun shadows, ambient
//! occlusion and bloom,
//! screen-space reflections and refraction through glass.
//! Every extracted frame carries sanitized copies with the
//! atmosphere; renderers skip the work of disabled effects.
//!
//! Each effect is its own resource: disable one with its settings, or remove
//! the resource to remove the effect. The environment section list
//! (`environment_sections!`) says which are inserted by default and which
//! are opt-in, inserted to turn them on. Without an `Atmosphere`
//! frames use the default one. Each is also a scene environment section in
//! the same JSON shape and checks itself with `validate`;
//! their docs are the sections' schema descriptions, written for authors.
//!
//! Shadows: every rendered instance casts and receives them. They lie in a
//! clipmap of `levels` around the camera, the finest of `finest_texel`
//! texels and each next one twice as coarse and twice as wide, each a
//! window of square tiles of `tile_texels` texels that follows the camera a
//! tile at a time. A tile has memory, one of the `atlas_slots` of one
//! atlas, only while pixels read it: each frame the depth buffer marks the
//! tiles its pixels read, at the level whose texels match the pixel's size,
//! and `redraw_budget` of them redraw a frame, coarse levels first, those
//! never drawn before those whose still casters changed. On GPU-presented
//! frames instances that stay still are kept in the tiles they were drawn
//! into; moving ones, and on other frames every instance but virtual
//! geometry, are drawn every frame over the tiles they cross,
//! `moving_tiles` at most. Where a level is not drawn yet, a coarser one
//! shows.
//!
//! Ambient occlusion darkens the sky's ambient light in creases, corners and
//! under objects. Bloom makes light glow beyond bright pixels such as
//! emissive surfaces, sun glints and the sun itself: emission and highlights
//! usually exceed its default threshold, which applies to brightness before
//! tone mapping; sunlit walls do not. Screen-space reflections fall
//! back to the sky where their rays leave the screen; where they meet the
//! scene, it takes the place of the sky a surface mirrors, drawn out into
//! streaks along the view where the view grazes the surface, as its
//! slopes spread the mirrored rays. Refraction bends the
//! opaque scene seen through transmissive materials along their index of
//! refraction and thickness and blurs it by their roughness; frames without
//! glass never pay for it.

use bevy_ecs::prelude::{Resource, World};
use serde::{Deserialize, Serialize};

use crate::{Camera, Environment, FrameTime, RenderSnapshot};
use engine_settings::settings;

/// Copies the camera eye, the detail eye, the height of what lies under the
/// eye, presentation time and environment of `world` into `snapshot`,
/// sanitized ([`Environment::current`]): a missing atmosphere reads as the
/// default one, a missing effect resource turns that effect off.
pub(crate) fn extract_lighting(world: &World, snapshot: &mut RenderSnapshot) {
    let eye = world.resource::<Camera>().eye();
    snapshot.camera_position = eye;
    snapshot.detail_eye = crate::TerrainDebug::detail_eye(world);
    snapshot.surface_under_eye = surface_under(world, eye);
    snapshot.elapsed_seconds = world
        .get_resource::<FrameTime>()
        .map_or(0.0, |time| time.elapsed_seconds as f32);
    snapshot.presented_seconds = world
        .get_resource::<crate::PresentedTime>()
        .map_or(0.0, |time| time.seconds);
    snapshot.environment = Environment::current(world).sanitized();
}

/// The height of what lies straight under `eye` in `world`: the surface of
/// the water standing there, over the ground under it; or else that ground.
fn surface_under(world: &World, eye: glam::Vec3) -> Option<f32> {
    let water = crate::WaterSources::sample_in(world, eye);
    if let Some(water) = water.filter(|water| water.depth > 0.0) {
        return Some(water.surface);
    }
    let ground = crate::Ground::sample_in(world, eye.x, eye.z);
    ground.map(|ground| ground.height)
}

/// Most levels of the sun's shadow clipmap ([`Shadows::levels`]).
pub const MAX_SHADOW_LEVELS: u32 = 20;
/// Most tiles moving casters draw over in a frame
/// ([`Shadows::moving_tiles`]): a caster's draw into one names it in eight
/// bits.
pub const MAX_MOVING_TILES: u32 = 256;

settings! {
    /// Sun shadows; omitted fields take the engine defaults.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneShadows")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Shadows {
        /// Whether the sun casts shadows.
        pub enabled: bool = true => any,
        /// Levels of the clipmap around the camera, each with texels twice
        /// the size of the one inside it: sixteen reach from the finest
        /// texel to about a kilometer a texel.
        pub levels: u32 = 16 => range(1, MAX_SHADOW_LEVELS),
        /// World size of the finest level's texels: shadows are this sharp
        /// at most, where pixels are smaller.
        pub finest_texel: f32 = 0.04 => range(0.005, 1.0),
        /// Texels along a side of a tile, the square that takes memory and
        /// redraws.
        pub tile_texels: u32 = 128 => power_of_two(64, 512),
        /// Tiles the atlas holds; tiles pixels read beyond these show a
        /// coarser level.
        pub atlas_slots: u32 = 1024 => range(64, 4096),
        /// Tiles a frame redraws at most, coarse levels first; the rest
        /// redraw over the next frames, a coarser level showing meanwhile.
        pub redraw_budget: u32 = 32 => range(1, 4096),
        /// Tiles moving casters draw over each frame at most, coarse levels
        /// first; they cast nothing on the tiles past these.
        pub moving_tiles: u32 = MAX_MOVING_TILES => range(0, MAX_MOVING_TILES),
    }
}

impl Shadows {
    /// What a world without these settings reads as: no sun shadows.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    /// Ambient occlusion; an intensity of 0 turns it off.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneAmbientOcclusion")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct AmbientOcclusion {
        /// How much creases and corners darken the ambient light; 0 turns it
        /// off.
        pub intensity: f32 = 0.8 => range(0.0, 1.0),
        /// World distance within which geometry occludes, the same at every
        /// distance: where it shrinks on screen to what the finest occupancy
        /// map's texels span where that map hands over, the occlusion fades
        /// out, as the maps and terrains' probes hold coarser shade.
        pub radius: f32 = 2.5 => above(0.0),
    }
}

impl AmbientOcclusion {
    /// What a world without these settings reads as: no ambient occlusion.
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

settings! {
    /// Bloom; an intensity of 0 turns it off.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneBloom")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Bloom {
        /// How strongly light glows beyond bright pixels; 0 turns it off.
        pub intensity: f32 = 0.6 => range(0.0, 4.0),
        /// Exposed brightness above which light glows.
        pub threshold: f32 = 2.0 => at_least(0.0),
    }
}

impl Bloom {
    /// What a world without these settings reads as: no bloom.
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

settings! {
    /// Dithering: every write of color into an 8-bit target (the canvas, the
    /// screen effects' scene and temporal antialiasing's history) adds noise
    /// of about one step of what the target stores, so smooth gradients such
    /// as dusk skies, fog and dark interiors show no bands, and temporal
    /// antialiasing's history settles on the color instead of stopping a few
    /// steps short. The noise is fixed per pixel, and turns every frame under
    /// temporal antialiasing, which averages it. Off by default.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDither")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Dither {
        /// Whether 8-bit color writes dither.
        pub enabled: bool = true => any,
    }
}

impl Dither {
    /// What a world without these settings reads as: no dithering.
    pub fn off() -> Self {
        Self { enabled: false }
    }
}

settings! {
    /// Auto exposure: the eye adapting to the light in view. Each frame a
    /// meter reads the frame drawn (a grid of samples, the middle of the view
    /// counting more than its corners and the sky less than the rest) and
    /// averages their light in stops, leaving out shares of the darkest and
    /// the brightest. Later frames' exposure, eased quickly when the view
    /// grows brighter and slowly when it grows darker, as eyes do, rises over
    /// the atmosphere's authored exposure by two parts. The time of day's:
    /// under the physically based sky, its own adaptation to its light (up to
    /// 4 stops), but at most `night` stops with the sun 14 degrees or more
    /// below the horizon, easing there from 4 degrees below, so dusk dims as
    /// before and night keeps dark. The view's: that average against what the
    /// light falling on level ground gives an open view, beyond a stop either
    /// way, so a cave by day brightens and a view of the bright sky darkens;
    /// at night dark views stay dark. Opt-in.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneAutoExposure")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct AutoExposure {
        /// Whether exposure follows the meter.
        pub enabled: bool = true => any,
        /// How far exposure follows the view: 0 not at all, 1 brings its
        /// average all the way to within a stop of what its light gives an
        /// open view.
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        /// Stops added to the exposure: positive brightens the view,
        /// negative darkens it.
        pub compensation: f32 = 0.0 => range(-8.0, 8.0),
        /// The most stops the time of day raises exposure at night.
        pub night: f32 = 1.0 => range(0.0, 16.0),
        /// The most stops a dark view raises exposure, as a cave by day; none
        /// at night, easing as the time of day's limit does.
        pub brighten: f32 = 3.0 => range(0.0, 16.0),
        /// The most stops a bright view lowers exposure.
        pub darken: f32 = 1.0 => range(0.0, 16.0),
        /// Seconds over which exposure goes most of the way (all but 1/e)
        /// toward its new value after the view grows brighter.
        pub to_brighter: f32 = 0.5 => range(0.0, 60.0),
        /// The same after the view grows darker.
        pub to_darker: f32 = 2.0 => range(0.0, 60.0),
        /// The share of the darkest samples the meter's average leaves out.
        pub skip_dark: f32 = 0.3 => range(0.0, 0.45),
        /// The share of the brightest samples it leaves out (the sun, lamps).
        pub skip_bright: f32 = 0.05 => range(0.0, 0.45),
        /// How many times what the view's corners count the middle counts.
        pub center: f32 = 2.0 => range(1.0, 16.0),
        /// How much the sky counts against what stands under it.
        pub sky: f32 = 0.5 => range(0.0, 1.0),
    }
}

impl AutoExposure {
    /// What a world without these settings reads as: the authored exposure.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    /// Screen-space reflections; an intensity
    /// of 0 turns them off.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneReflections")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Reflections {
        /// How strongly glossy surfaces reflect the screen; 0 turns them off.
        pub intensity: f32 = 1.0 => range(0.0, 1.0),
        /// Surfaces rougher than this across the view, the way their
        /// mirrored lobe stays narrowest, reflect only the sky: their
        /// reflections of the screen fade out over its last fifth.
        pub max_roughness: f32 = 0.75 => range(0.05, 1.0),
        /// World distance reflection rays travel.
        pub distance: f32 = 120.0 => range(0.001, 100_000.0),
    }
}

impl Reflections {
    /// What a world without these settings reads as: no screen-space reflections.
    pub fn off() -> Self {
        Self {
            intensity: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}

settings! {
    /// Refraction through glass; a strength
    /// of 0 turns it off.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneRefraction")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Refraction {
        /// Scales how far rays bend through transmissive materials; 0 turns
        /// it off.
        pub strength: f32 = 1.0 => range(0.0, 4.0),
        /// Blur radius in pixels that fully rough glass reaches.
        pub blur: f32 = 12.0 => range(0.0, 64.0),
    }
}

impl Refraction {
    /// What a world without these settings reads as: no refraction.
    pub fn off() -> Self {
        Self {
            strength: 0.0,
            ..Self::default()
        }
    }

    pub fn enabled(&self) -> bool {
        self.strength > 0.0
    }
}
