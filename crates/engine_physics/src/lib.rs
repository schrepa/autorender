#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod body;
mod bounds;
mod coupling;
mod obstacles;
mod settings;
mod shapes;
mod smoke;
mod solids;
mod solver;
mod statics;
mod views;
mod world;

pub use settings::{Physics, PhysicsStats};

use bevy_ecs::prelude::*;
use engine_core::{
    Brick, Bricks, ComponentKind, ErrorCode, Placement, Runs, SceneComponents, Schedule, Smoke,
    Source,
};
use engine_places::physics::RigidBody;
use engine_places::water::WaterObstacles;

pub struct PhysicsBrick;

impl Brick for PhysicsBrick {
    const SOURCE: &'static Source = &SOURCE;
    const SMOKES: &'static [Smoke] = &[Smoke {
        name: "physics",
        runs: Runs::Headless,
        run: smoke::run,
    }];

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(simulating)
            .settings::<Physics>()
            .stats::<PhysicsStats>()
            .resource::<world::PhysicsWorld>()
            .resource::<solids::Solids>()
            .fill::<WaterObstacles>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<RigidBody>("rigid_body")
                    .check(body::check_body)
                    .placement(Placement::Roots(ErrorCode::InvalidRigidBody))
                    .schema(body::body_schema)
                    .example(body::body_example),
                None,
            )
            .systems(
                Schedule::FixedPreSimulation,
                (solids::sync_sources, solids::sync_colliders),
            )
            .systems(
                Schedule::FixedSimulation,
                (world::step, obstacles::obstruct),
            )
            .systems(Schedule::FixedPostSimulation, solids::sync_colliders)
            .observation::<views::PhysicsBody>();
    }
}

fn simulating(physics: Res<Physics>) -> bool {
    physics.enabled
}
