//! Opt-in authoritative collision queries and fixed-tick sensor transitions.
#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bvh;
mod geometry;
mod placed;
mod solids;
mod terrain;
mod types;
mod world;

pub use placed::BRIDGE_LAYER;
pub use solids::{ColliderSolid, GroundSolid, PlacedSolid, Solid};
pub use types::*;
pub use world::{CollisionWorld, SensorEvents, refresh_collision_world};

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use engine_core::{FixedPostSimulation, FixedPreSimulation};

/// Install once before the first simulation frame. Rendering needs no plugin.
#[derive(Default)]
pub struct CollisionPlugin;

/// Named insertion points in the engine's fixed boundary schedules.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollisionSystems {
    Sync,
    Detect,
}

impl Plugin for CollisionPlugin {
    fn build(&self, app: &mut App) {
        engine_core::install(app, engine_core::Feature::Collision);
        app.init_resource::<CollisionConfig>()
            .init_resource::<CollisionWorld>()
            .init_resource::<SensorEvents>()
            .add_systems(
                FixedPreSimulation,
                world::sync.in_set(CollisionSystems::Sync),
            )
            .configure_sets(
                FixedPostSimulation,
                (CollisionSystems::Sync, CollisionSystems::Detect).chain(),
            )
            .add_systems(
                FixedPostSimulation,
                (
                    world::sync.in_set(CollisionSystems::Sync),
                    world::detect.in_set(CollisionSystems::Detect),
                ),
            );
    }
}

pub mod prelude {
    pub use crate::{
        Collider, ColliderShape, CollisionConfig, CollisionError, CollisionLayers, CollisionPlugin,
        CollisionRay, CollisionReport, CollisionSystems, CollisionWorld, PlacedPart, QueryFilter,
        RayHit, SensorEvent, SensorEventKind, SensorEvents, SensorFilter, SensorPair,
        ShapeCastOptions, ShapeHit, refresh_collision_world, validate_collider_transform,
    };
}
