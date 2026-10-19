//! Lit windows switching on room by room through the evening: on
//! materials divided into rooms ([`crate::Material::with_windows`]), each
//! room of each building lights once the sun sinks below its own elevation,
//! spread around dusk, so a city lights up gradually as the sun sets (with
//! a time of day, through the evening) and goes dark the same way at dawn;
//! a share of rooms stays dark, and lit rooms vary in brightness and from
//! cool to sodium-warm light. Only rooms whose texels already emit light,
//! and unlit rooms' windows darken where their texels emit by their alpha.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Window settings.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneWindows")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Windows {
        /// Whether lit windows switch on room by room as the sun sets.
        pub enabled: bool = true => any,
        /// The sun's elevation, degrees, around which rooms switch on.
        pub dusk: f32 = -3.0 => range(-90.0, 90.0),
        /// How far either side of `dusk` rooms' elevations spread, degrees.
        pub spread: f32 = 12.0 => above_to(0.0, 90.0),
        /// The share of rooms that stay dark.
        pub dark: f32 = 0.1 => range(0.0, 1.0),
    }
}

impl Windows {
    /// What a world without these settings reads as: windows lit as their
    /// textures light them, whatever the time.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
