#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod bvh;
mod geometry;
mod placed;
mod queries;
mod results;
mod scene;
mod smoke;
mod terrain;
mod terrain_sweep;
mod types;
mod views;
mod world;

pub use types::{CollisionConfig, CollisionLimits, CollisionStats, Count};

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, ErrorCode, Runs, SceneComponents, Schedule, Smoke, Source,
};
use engine_places_simulation::physics::{
    COLLISION_CANDIDATE_BUDGET, COLLISION_INCOMPLETE_WORLD, COLLISION_INVALID_CAST,
    COLLISION_INVALID_CONFIG, COLLISION_INVALID_RAY, COLLISION_INVALID_SHAPE,
    COLLISION_INVALID_TRANSFORM, COLLISION_NUMERICAL_FAILURE, COLLISION_PAIR_BUDGET,
    COLLISION_RESULT_BUDGET, Collider, Collision, INVALID_COLLIDER, INVALID_COLLIDER_TRANSFORM,
};
use engine_places_simulation::sweeps::Sweeps;

const ERRORS: &[ErrorCode] = &[
    COLLISION_INVALID_SHAPE,
    COLLISION_INVALID_TRANSFORM,
    COLLISION_INVALID_RAY,
    COLLISION_INVALID_CAST,
    COLLISION_INVALID_CONFIG,
    COLLISION_INCOMPLETE_WORLD,
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
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "collision",
        runs: Runs::Headless,
        run: smoke::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(|config: Res<CollisionConfig>| config.enabled)
            .settings::<CollisionConfig>()
            .settled_stats::<CollisionStats, _>(world::refresh)
            .errors(ERRORS)
            .fill::<Collision>(None)
            .fill::<Sweeps>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<Collider>("collider")
                    .check(scene::checked)
                    .check_placed(scene::checked_world)
                    .schema(scene::schema)
                    .example(scene::example),
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
            .settled::<queries::RaycastQuery, _>(world::refresh)
            .settled::<queries::OverlapQuery, _>(world::refresh)
            .settled::<queries::ShapeCastQuery, _>(world::refresh)
            .observation::<queries::CollisionEventsQuery>()
            .settled::<terrain_sweep::TerrainSweepQuery, _>(world::refresh)
            .settled::<views::ColliderQuery, _>(world::refresh);
    }
}
