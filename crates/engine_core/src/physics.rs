//! The physics world as scenes set it and frames observe it: the settings
//! rigid bodies move under (gravity, and the water and air about them) and
//! what the last fixed tick did. The physics feature's plugin, in a crate of
//! its own, reads the one and reports the other.

use bevy_ecs::prelude::Resource;
use serde::{Deserialize, Serialize};

use crate::terrain_data::hydraulics::GRAVITY;
use engine_settings::settings;

settings! {
    /// Rigid bodies' world.
    #[derive(Resource, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "ScenePhysics")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct Physics {
        /// Whether bodies move; off, each holds where it stands.
        pub enabled: bool = true => any,
        /// Every body's acceleration, meters per second squared: by default
        /// the pull the terrains' water runs and falls by.
        pub gravity: [f32; 3] = [0.0, -GRAVITY, 0.0] => range(-100.0, 100.0),
        /// Kilograms a cubic meter of water weighs: bodies in water are
        /// buoyed by the weight of what they displace and dragged with its
        /// current, so those lighter than it float.
        pub water_density: f32 = 1000.0 => range(0.0, 20_000.0),
        /// Kilograms a cubic meter of air weighs: what the wind drags
        /// bodies with.
        pub air_density: f32 = 1.225 => range(0.0, 100.0),
        /// Constraint solver iterations a tick: more keep stacks and piles
        /// stiffer, at their cost.
        pub iterations: u32 = 4 => range(1, 16),
    }
}

impl Physics {
    /// What a world without these settings reads as: nothing moves.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Self::default()
        }
    }
}

/// Physics at the last fixed tick, read when the stats are.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct PhysicsStats {
    /// Bodies the solver holds.
    pub bodies: u32,
    /// Of them, awake: moving, or pushed this tick.
    pub awake: u32,
    /// Of them, held still where the world's solids under them are not
    /// resident yet.
    pub held: u32,
    /// Of them, in water.
    pub wet: u32,
    /// The world's solids the solver holds for bodies to meet: terrain
    /// tiles, placed objects and colliders near them.
    pub solids: u32,
    /// Contacts touching between bodies and solids or each other.
    pub contacts: u32,
    /// Samples of the world's water the tick's bodies took.
    pub water_samples: u32,
}
