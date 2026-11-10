#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod bounds;
mod bvh;
mod geometry;
mod placed;
mod queries;
mod results;
mod scene;
mod terrain;
mod types;
mod views;
mod world;

pub use types::{CollisionConfig, CollisionLimits, CollisionStats, Count};
pub use world::COLLIDER_PART;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, ComponentKind, ErrorCode, SceneComponents, Schedule, Source};
use engine_places::queries::SceneQueries;
use engine_places_simulation::ground::Ground;
use engine_places_simulation::physics::{
    COLLISION_CANDIDATE_BUDGET, COLLISION_COLLIDER_BUDGET, COLLISION_INCOMPLETE_WORLD,
    COLLISION_INVALID_CAST, COLLISION_INVALID_CONFIG, COLLISION_INVALID_RAY,
    COLLISION_INVALID_SHAPE, COLLISION_INVALID_TRANSFORM, COLLISION_NUMERICAL_FAILURE,
    COLLISION_PAIR_BUDGET, COLLISION_RESULT_BUDGET, Collider, Collision, INVALID_COLLIDER,
    INVALID_COLLIDER_TRANSFORM,
};

const ERRORS: &[ErrorCode] = &[
    COLLISION_INVALID_SHAPE,
    COLLISION_INVALID_TRANSFORM,
    COLLISION_INVALID_RAY,
    COLLISION_INVALID_CAST,
    COLLISION_INVALID_CONFIG,
    COLLISION_INCOMPLETE_WORLD,
    COLLISION_COLLIDER_BUDGET,
    COLLISION_CANDIDATE_BUDGET,
    COLLISION_PAIR_BUDGET,
    COLLISION_RESULT_BUDGET,
    COLLISION_NUMERICAL_FAILURE,
    INVALID_COLLIDER,
    INVALID_COLLIDER_TRANSFORM,
];

pub struct CollisionBrick;

impl Brick for CollisionBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|config: Res<CollisionConfig>| config.enabled)
            .settings::<CollisionConfig>()
            .stats::<CollisionStats>()
            .errors(ERRORS)
            .fill::<Collision>(None)
            .fill::<Ground>(None)
            .fill::<SceneQueries>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<Collider>("collider")
                    .check(scene::checked)
                    .check_placed(scene::checked_world)
                    .schema(scene::schema),
                None,
            )
            .systems(
                Schedule::FixedPreSimulation,
                (world::sync_ground, world::sync),
            )
            .systems(
                Schedule::FixedPostSimulation,
                (world::sync_ground, world::sync, world::detect),
            )
            .operation::<queries::RaycastQuery>()
            .operation::<queries::OverlapQuery>()
            .operation::<queries::ShapeCastQuery>()
            .operation::<views::ColliderQuery>();
    }
}
