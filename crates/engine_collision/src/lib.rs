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
use engine_core::{Brick, Bricks, ComponentKind, Runs, SceneComponents, Schedule, Smoke, Source};
use engine_places::physics::{Collider, Collision};

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
            .stats::<CollisionStats>()
            .fill::<Collision>(None)
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
            .off(world::forget)
            .settled::<queries::RaycastQuery, _>(world::refresh)
            .settled::<queries::OverlapQuery, _>(world::refresh)
            .settled::<queries::ShapeCastQuery, _>(world::refresh)
            .observation::<queries::CollisionEventsQuery>()
            .settled::<terrain_sweep::TerrainSweepQuery, _>(world::refresh)
            .settled::<views::ColliderQuery, _>(world::refresh);
    }
}
