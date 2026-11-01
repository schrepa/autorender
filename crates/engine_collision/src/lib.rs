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
mod smoke;
mod terrain;
mod terrain_sweep;
mod types;
mod views;
mod world;

pub use types::{CollisionConfig, CollisionStats};

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, Runs, SceneComponents, Schedule, Smoke, Source,
};
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
            .join::<SceneComponents>(ComponentKind::of::<Collider>("collider"), None)
            .systems(
                Schedule::FixedPreSimulation,
                (world::sync_ground, world::sync),
            )
            .systems(Schedule::FixedPostSimulation, (world::sync, world::detect))
            .off(world::forget)
            .observation::<queries::RaycastQuery>()
            .observation::<queries::OverlapQuery>()
            .observation::<queries::ShapeCastQuery>()
            .observation::<queries::CollisionEventsQuery>()
            .observation::<terrain_sweep::TerrainSweepQuery>()
            .observation::<views::ColliderQuery>();
    }
}
