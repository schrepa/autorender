//! Highlights sized by their lights: the sun's highlight spreads over its
//! disk, local lights' over the spheres and tubes their `radius` and
//! `length` give, each lobe widened and renormalized for its light's angular
//! size (Karis's representative point), so smooth surfaces show highlights
//! of their lights' shape instead of dots brighter than a pixel can hold,
//! which shimmer under temporal antialiasing and bloom. Apart from that,
//! local lights' highlights can ease their exposed brightness toward a
//! cap, which keeps lamps' highlights on wet and polished surfaces from
//! blooming into white blobs at night while they stay as bright as the cap
//! and keep their color, and leaves dimmer highlights and the sun's as they
//! are.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use engine_settings::settings;

settings! {
    /// Highlight settings.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneHighlights")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Highlights {
        /// Whether highlights take their lights' sizes: the sun's disk and
        /// local lights' spheres and tubes.
        pub enabled: bool = true => any,
        /// Whether local lights' highlights ease toward `cap`, sized or
        /// not; the sun's stay as they are.
        pub capped: bool = false => any,
        /// The sun's angular radius, degrees, while the physically based sky
        /// is off; with it on, the sky's own `sun_size`.
        pub sun_size: f32 = 0.27 => range(0.0, 10.0),
        /// The exposed brightness local lights' highlights ease toward
        /// while `capped`, in the units of the bloom's threshold: unchanged
        /// up to half of it, approaching it above; 0 caps none.
        pub cap: f32 = 4.0 => at_least(0.0),
        /// The radius of the sphere local lights shine from where theirs is
        /// smaller ([`crate::LocalLight::radius`]), world units: no lamp is a
        /// point.
        pub light_radius: f32 = 0.1 => range(0.0, 100.0),
    }
}

impl Highlights {
    /// What a world without these settings reads as: point lights and
    /// uncapped highlights, as before they existed.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}
