//! Rigid bodies: what an entity's [`RigidBody`] asks of physics, and what
//! physics reports of it ([`BodyStatus`]).

use bevy_ecs::prelude::Component;
use engine_core::WaterKind;
use engine_settings::{Setting, settings};
use glam::Vec3;
use serde::{Deserialize, Serialize};

/// How a body moves.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum BodyMotion {
    /// Under gravity, contacts, water and wind: physics writes its pose.
    #[default]
    Dynamic,
    /// Along the poses written to it, pushing dynamic bodies aside as it
    /// goes: platforms, doors, paddles.
    Kinematic,
}

impl Setting for BodyMotion {}

settings! {
    /// A rigid body: physics moves its entity by gravity, its contacts with
    /// the world's solids (terrain, placed objects, bridge decks and
    /// colliders) and other bodies, and the water and wind about it. Its shape
    /// is its entity's collider, scaled by its transform; its mass is that
    /// shape's volume times its density, so bodies lighter than water float.
    /// Roots only. Each fixed tick physics writes a dynamic body's pose to its
    /// transform, which interpolates between ticks; a pose written to it
    /// since moves it there. Bodies where the ground under them is not
    /// resident hold still until it is.
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneRigidBody")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct RigidBody {
        /// How it moves.
        pub motion: BodyMotion = BodyMotion::Dynamic => any,
        /// Kilograms a cubic meter: water's is 1000, wood's about 600,
        /// stone's about 2600.
        pub density: f32 = 600.0 => range(1.0, 100_000.0),
        /// How its surface grips what it slides on.
        pub friction: f32 = 0.6 => range(0.0, 10.0),
        /// How much of its speed into a contact it bounces back with.
        pub restitution: f32 = 0.05 => range(0.0, 1.0),
        /// Its drag coefficient: how hard water's current and the wind push
        /// on the area it turns to them.
        pub drag: f32 = 1.0 => range(0.0, 10.0),
        /// Its velocity as it joins, world meters a second.
        pub velocity: [f32; 3] = [0.0; 3] => range(-1000.0, 1000.0),
        /// Its spin as it joins, radians a second about the world's axes.
        pub spin: [f32; 3] = [0.0; 3] => range(-1000.0, 1000.0),
        /// Swept between ticks, so it stops at thin solids however fast it
        /// moves.
        pub fast: bool = false => any,
    }
}

/// Why physics leaves a body out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum BodyProblem {
    /// Its entity has no collider: no shape to move.
    NoCollider,
    /// Its entity has a parent: bodies move in the world, as roots.
    NotRoot,
    /// Its entity has no transform: nowhere to stand.
    NoTransform,
    /// Its transform's scale gives its shape no usable size.
    BadScale,
}

/// The water a body is in.
#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BodyWater {
    /// The water deepest about it.
    pub kind: WaterKind,
    /// The share of its volume below the surface, 0 to 1.
    pub submerged: f32,
    /// The water's velocity at its middle, world meters a second.
    pub current: [f32; 3],
}

/// What physics did with a body at the last fixed tick.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyStatus {
    /// Why it is left out, if it is.
    pub problem: Option<BodyProblem>,
    /// Kilograms.
    pub mass: f32,
    /// Cubic meters.
    pub volume: f32,
    /// World meters a second.
    pub velocity: Vec3,
    /// Radians a second about the world's axes.
    pub spin: Vec3,
    /// Whether it rests asleep until something wakes it.
    pub sleeping: bool,
    /// Whether it holds still where the ground under it is not resident.
    pub held: bool,
    /// The water it is in, if any.
    pub water: Option<BodyWater>,
}
