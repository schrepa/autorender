#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

mod body;
mod coupling;
mod obstacles;
mod shapes;
mod solver;
mod statics;
mod world;

pub use body::{BodyMotion, BodyProblem, BodyStatus, BodyWater, RigidBody};
pub use world::PhysicsWorld;

use bevy_app::{App, Plugin};
use bevy_ecs::prelude::*;
use engine_core::{FixedSimulation, FixedSimulationSystems};

#[derive(Default)]
pub struct PhysicsPlugin;

#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsSystems {
    Step,
    Obstruct,
}

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        engine_core::install(app, engine_core::Feature::Physics);
        app.init_resource::<PhysicsWorld>()
            .init_resource::<engine_core::WaterObstacles>()
            .configure_sets(
                FixedSimulation,
                (PhysicsSystems::Step, PhysicsSystems::Obstruct)
                    .chain()
                    .in_set(FixedSimulationSystems::Simulate),
            )
            .add_systems(
                FixedSimulation,
                (
                    world::step.in_set(PhysicsSystems::Step),
                    obstacles::obstruct.in_set(PhysicsSystems::Obstruct),
                ),
            );
    }
}
