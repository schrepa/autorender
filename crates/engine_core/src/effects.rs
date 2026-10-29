use bevy_ecs::prelude::{Resource, World};
use serde::{Deserialize, Serialize};

use crate::sections::environment_section;
use crate::{Camera, Environment, FrameTime, RenderSnapshot};
use engine_settings::settings;

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

fn surface_under(world: &World, eye: glam::Vec3) -> Option<f32> {
    let water = crate::Ground::water_in(world, eye);
    if let Some(water) = water.filter(|water| water.depth > 0.0) {
        return Some(water.surface);
    }
    let ground = crate::Ground::sample_in(world, eye.x, eye.z);
    ground.map(|ground| ground.height)
}

pub const MAX_SHADOW_LEVELS: u32 = 20;
pub const MAX_MOVING_TILES: u32 = 256;

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneShadows")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Shadows {
        pub enabled: bool = true => any,
        pub levels: u32 = 16 => range(1, MAX_SHADOW_LEVELS),
        pub finest_texel: f32 = 0.04 => range(0.005, 1.0),
        pub tile_texels: u32 = 128 => power_of_two(64, 512),
        pub atlas_slots: u32 = 1024 => range(64, 4096),
        pub redraw_budget: u32 = 32 => range(1, 4096),
        pub moving_tiles: u32 = MAX_MOVING_TILES => range(0, MAX_MOVING_TILES),
    }
}

environment_section!(Shadows, "shadows", opt_in: false);

impl Shadows {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneAmbientOcclusion")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct AmbientOcclusion {
        pub intensity: f32 = 0.8 => range(0.0, 1.0),
        pub radius: f32 = 2.5 => above(0.0),
    }
}

environment_section!(AmbientOcclusion, "ambient_occlusion", opt_in: false);

impl AmbientOcclusion {
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
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneBloom")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Bloom {
        pub intensity: f32 = 0.6 => range(0.0, 4.0),
        pub threshold: f32 = 2.0 => at_least(0.0),
    }
}

environment_section!(Bloom, "bloom", opt_in: false);

impl Bloom {
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
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneDither")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Dither {
        pub enabled: bool = true => any,
    }
}

environment_section!(Dither, "dither", opt_in: false);

impl Dither {
    pub fn off() -> Self {
        Self { enabled: false }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneAutoExposure")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct AutoExposure {
        pub enabled: bool = true => any,
        pub strength: f32 = 1.0 => range(0.0, 1.0),
        pub compensation: f32 = 0.0 => range(-8.0, 8.0),
        pub night: f32 = 1.0 => range(0.0, 16.0),
        pub brighten: f32 = 3.0 => range(0.0, 16.0),
        pub darken: f32 = 1.0 => range(0.0, 16.0),
        pub to_brighter: f32 = 0.5 => range(0.0, 60.0),
        pub to_darker: f32 = 2.0 => range(0.0, 60.0),
        pub skip_dark: f32 = 0.3 => range(0.0, 0.45),
        pub skip_bright: f32 = 0.05 => range(0.0, 0.45),
        pub center: f32 = 2.0 => range(1.0, 16.0),
        pub sky: f32 = 0.5 => range(0.0, 1.0),
    }
}

environment_section!(AutoExposure, "auto_exposure", opt_in: true);

impl AutoExposure {
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

settings! {
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneReflections")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Reflections {
        pub intensity: f32 = 1.0 => range(0.0, 1.0),
        pub max_roughness: f32 = 0.75 => range(0.05, 1.0),
        pub distance: f32 = 120.0 => range(0.001, 100_000.0),
    }
}

environment_section!(Reflections, "reflections", opt_in: true);

impl Reflections {
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
