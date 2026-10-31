#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod bvh;
mod geometry;
mod placed;
mod queries;
pub mod results;
mod scene;
mod smoke;
mod solids;
mod terrain;
mod terrain_sweep;
mod types;
mod views;
mod world;

pub use queries::{
    CollisionEventsQuery, CollisionFilterRequest, CollisionSensorFilter, CollisionShapeRequest,
    OverlapQuery, RaycastQuery, ShapeCastQuery,
};
pub use scene::{COLLIDER, SceneCollider, SceneColliderShape};
pub use solids::{ColliderSolid, GroundSolid, PlacedSolid, Solid};
pub use terrain_sweep::TerrainSweepQuery;
pub use types::*;
pub use views::{
    ColliderShapeView, ColliderView, CollisionLimitsView, CollisionWorldView, collider_view,
};
pub use world::{CollisionWorld, SensorEvents, refresh_collision_world};

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, FixedPostSimulation, FixedPreSimulation, Join};
use engine_scene::SceneComponentsList;

pub struct Collision;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CollisionSystems {
    Sync,
    Detect,
}

impl Brick for Collision {
    fn register(bricks: &mut Bricks) {
        bricks
            .resource::<CollisionConfig>()
            .resource::<CollisionWorld>()
            .resource::<SensorEvents>()
            .join(Join::<SceneComponentsList>::new().cpu(&COLLIDER))
            .systems(
                FixedPreSimulation,
                (world::sync_ground, world::sync)
                    .chain()
                    .in_set(CollisionSystems::Sync),
            )
            .configure_sets(
                FixedPostSimulation,
                (CollisionSystems::Sync, CollisionSystems::Detect).chain(),
            )
            .systems(
                FixedPostSimulation,
                (
                    (world::sync_ground, world::sync)
                        .chain()
                        .in_set(CollisionSystems::Sync),
                    world::detect.in_set(CollisionSystems::Detect),
                ),
            )
            .inspector::<CollisionWorld, CollisionWorldView>(|_, world| {
                CollisionWorldView::of(world)
            })
            .operation::<RaycastQuery>()
            .operation::<OverlapQuery>()
            .operation::<ShapeCastQuery>()
            .operation::<CollisionEventsQuery>()
            .operation::<TerrainSweepQuery>()
            .smoke("collision", false, smoke::run);
    }
}

pub mod prelude {
    pub use crate::{
        Collider, ColliderShape, Collision, CollisionConfig, CollisionLayers, CollisionRay,
        CollisionReport, CollisionSystems, CollisionWorld, PlacedPart, QueryFilter, RayHit,
        SensorEvent, SensorEventKind, SensorEvents, SensorFilter, SensorPair, ShapeCastOptions,
        ShapeHit, refresh_collision_world, validate_collider_transform,
    };
}
