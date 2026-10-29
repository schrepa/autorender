#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod body;
mod coupling;
mod obstacles;
mod settings;
mod shapes;
mod solver;
mod statics;
mod views;
mod world;

pub use body::{BodyMotion, BodyProblem, BodyStatus, BodyWater, RigidBody};
pub use settings::{Physics, PhysicsStats};
pub use views::{PhysicsWorldView, RigidBodyView, rigid_body_view};
pub use world::PhysicsWorld;

use bevy_ecs::prelude::*;
use engine_core::{Brick, Bricks, FixedSimulation, FixedSimulationSystems};

pub struct PhysicsBrick;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsSystems {
    Step,
    Obstruct,
}

impl Brick for PhysicsBrick {
    fn register(bricks: &mut Bricks) {
        bricks
            .settings::<Physics>()
            .stats::<PhysicsStats>()
            .resource::<PhysicsWorld>()
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
                    obstacles::obstruct.in_set(PhysicsSystems::Obstruct),
                ),
            )
            .inspector::<PhysicsWorld, PhysicsWorldView>(|_, world| PhysicsWorldView::of(world));
    }
}
