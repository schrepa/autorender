//! Rigid bodies as a primitive of the world: [`RigidBody`] entities move
//! under gravity, their contacts with the world's solids and with each
//! other, and the water and wind about them. The solids are the collision
//! world's own (terrain heightfields, placed objects, bridge decks and
//! colliders), so bodies meet exactly what queries hit; the water is the
//! world's ([`engine_core::WaterSources`]): bodies lighter than it float, its
//! currents carry them and its falls drop them; the wind is the one placed
//! objects sway in.
//!
//! Each fixed tick, in [`FixedSimulationSystems::Simulate`] after agents
//! move: bodies join, follow and leave the solver as their entities do, a
//! pose written to a body since physics wrote one moving it there; the
//! world's solids near the bodies follow the collision world; water and
//! wind push them; the solver steps; and each dynamic body's pose is written
//! to its transform, which interpolates between ticks. Bodies over terrain
//! whose solids collision does not hold yet stand still until it does.
//! Then the bodies and the colliders near rivers are listed as what stands
//! in the world's water, where rivers meet them: those resting on a bed hold
//! the river back. Worlds without bodies or rivers do no work.
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

/// Adds physics: the [`PhysicsWorld`], the [`Physics`](engine_core::Physics)
/// settings and the fixed tick's step. It needs collision's plugin, whose
/// world holds the solids bodies meet.
#[derive(Default)]
pub struct PhysicsPlugin;

/// Physics' systems, in [`FixedSimulationSystems::Simulate`] in this order.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PhysicsSystems {
    /// Bodies move.
    Step,
    /// The solids standing in the world's water are listed for rivers to
    /// meet ([`engine_core::WaterObstacles`]).
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
