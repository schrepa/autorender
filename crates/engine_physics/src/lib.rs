#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod body;
mod coupling;
mod obstacles;
mod scene;
mod settings;
mod shapes;
mod smoke;
mod solver;
mod statics;
mod views;
mod world;

pub use body::{BodyMotion, BodyProblem, BodyStatus, BodyWater, RigidBody};
pub use scene::RIGID_BODY;
pub use settings::{Physics, PhysicsStats};
pub use views::{PhysicsWorldView, RigidBodyView, rigid_body_view};
pub use world::PhysicsWorld;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, Fill, FixedSimulation, FixedSimulationSystems, Join};
use engine_places::{PhysicsSlot, WaterObstacles};
use engine_scene::SceneComponentsList;

pub struct PhysicsBrick;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsSystems {
    Step,
    Obstruct,
}

impl Brick for PhysicsBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .fill(Fill::<PhysicsSlot>::new())
            .settings::<Physics>()
            .stats::<PhysicsStats>()
            .resource::<PhysicsWorld>()
            .resource::<WaterObstacles>()
            .resource::<obstacles::WaterReaches>()
            .configure_sets(
                FixedSimulation,
                (PhysicsSystems::Step, PhysicsSystems::Obstruct)
                    .chain()
                    .in_set(FixedSimulationSystems::Simulate),
            )
            .systems(
                FixedSimulation,
                (
                    world::step.in_set(PhysicsSystems::Step),
                    (obstacles::reach_water, obstacles::obstruct)
                        .chain()
                        .in_set(PhysicsSystems::Obstruct),
                ),
            )
            .inspector::<PhysicsWorld, PhysicsWorldView>(|_, world| PhysicsWorldView::of(world))
            .join(Join::<SceneComponentsList>::new().cpu(&RIGID_BODY))
            .smoke("physics", false, smoke::run);
    }
}
