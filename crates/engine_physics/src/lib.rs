#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

include!(concat!(env!("OUT_DIR"), "/brick.rs"));

mod body;
mod coupling;
mod obstacles;
mod settings;
mod shapes;
mod solids;
mod solver;
mod statics;
mod views;
mod world;

pub use settings::{Physics, PhysicsStats};
pub use world::BODY_PART;

use bevy_ecs::prelude::*;
use engine_core::{
    Asked, Brick, Bricks, ComponentKind, ErrorCode, Pace, Placement, SceneComponents, Schedule,
    Source,
};
use engine_places_simulation::physics::{BodyAsk, BodyPush, RigidBody};
use engine_places_simulation::water::WaterObstacles;

pub const INVALID_RIGID_BODY: ErrorCode = ErrorCode::brick("invalid_rigid_body");

pub struct PhysicsBrick;

impl Brick for PhysicsBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .when(simulating)
            .settings::<Physics>()
            .stats::<PhysicsStats>()
            .errors(&[INVALID_RIGID_BODY])
            .resource::<world::PhysicsWorld>()
            .fill::<WaterObstacles>(None)
            .fill::<Asked<BodyAsk>>(None)
            .join::<SceneComponents>(
                ComponentKind::of::<RigidBody>("rigid_body")
                    .check(body::check_body)
                    .placement(Placement::Roots(INVALID_RIGID_BODY))
                    .schema(body::body_schema),
                None,
            )
            .join::<SceneComponents>(
                ComponentKind::of::<BodyPush>("body_push")
                    .check(body::check_push)
                    .schema(body::push_schema),
                None,
            )
            .moves::<RigidBody>(Pace::Tick)
            .systems(
                Schedule::FixedSimulation,
                (world::step, obstacles::obstruct),
            )
            .observation::<views::PhysicsBody>()
            .operation::<views::BodyAskOperation>();
    }
}

fn simulating(physics: Res<Physics>) -> bool {
    physics.enabled
}
