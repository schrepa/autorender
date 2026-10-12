//! Clouds: a layer of cumulus over the world, drifting with
//! the wind and lit by the sun and the sky. Renderers march it from the
//! camera into a dome of the sky above it, a slice of the dome's rows each
//! frame (all of them on exact frames), which the sky, its reflections and
//! the environment maps show; and from above the world toward the sun into
//! a map of the light it lets through, which shades the ground under it.
//! The view marches them again a ray per block of pixels after the opaque
//! scene, so they stand in front of mountains, around a
//! camera inside the layer and below one above it.

use bevy_ecs::prelude::*;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// The cloud layer. Opt-in: a scene without a `clouds` section has none.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneClouds")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Clouds {
        /// Whether the layer draws.
        pub enabled: bool = true => any,
        /// How much of the sky it covers: 0 clear, 1 overcast.
        pub coverage: f32 = 0.45 => range(0.0, 1.0),
        /// The world height of its base, meters.
        pub base: f32 = 1800.0 => range(-10_000.0, 30_000.0),
        /// How thick it is, meters.
        pub thickness: f32 = 1400.0 => range(50.0, 10_000.0),
        /// How dense its clouds are.
        pub density: f32 = 1.0 => range(0.01, 4.0),
        /// The wind it drifts in, meters per second along x and z.
        pub wind: [f32; 2] = [8.0, 3.0] => range(-100.0, 100.0),
        /// The meters across its largest shapes.
        pub scale: f32 = 3000.0 => range(200.0, 50_000.0),
        /// The seed of its shapes.
        pub seed: u32 = 1 => any,
        /// How much of the sun's light its shadows take from the ground.
        pub shadows: f32 = 0.75 => range(0.0, 1.0),
        /// The dome's texels around the horizon; a quarter as many from the
        /// horizon up.
        pub resolution: u32 = 2048 => multiple(64, 256, 4096),
        /// The frames over which ordinary frames march the whole dome.
        pub slices: u32 = 64 => range(1, 64),
        /// Whether the view marches the clouds after the opaque scene: then
        /// they stand in front of whatever lies beyond them, all around a camera
        /// inside the layer and below one above it, and the dome shows them
        /// behind every surface only while the camera is under the layer.
        pub in_view: bool = true => any,
        /// The pixels along a side of the blocks the view marches a ray each.
        pub view_block: u32 = 4 => range(1, 16),
    }
}

impl Clouds {
    /// What a world without these settings reads as: a clear sky.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
