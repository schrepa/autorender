use bevy_ecs::prelude::Component;
use engine_places::WaterKind;
use engine_settings::{Setting, settings};
use glam::Vec3;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum BodyMotion {
    #[default]
    Dynamic,
    Kinematic,
}

impl Setting for BodyMotion {}

settings! {
    #[derive(Component, Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
    #[cfg_attr(
        feature = "schema",
        derive(schemars::JsonSchema),
        schemars(rename = "SceneRigidBody")
    )]
    #[serde(default, deny_unknown_fields)]
    pub struct RigidBody {
        pub motion: BodyMotion = BodyMotion::Dynamic => any,
        pub density: f32 = 600.0 => range(1.0, 100_000.0),
        pub friction: f32 = 0.6 => range(0.0, 10.0),
        pub restitution: f32 = 0.05 => range(0.0, 1.0),
        pub drag: f32 = 1.0 => range(0.0, 10.0),
        pub velocity: [f32; 3] = [0.0; 3] => range(-1000.0, 1000.0),
        pub spin: [f32; 3] = [0.0; 3] => range(-1000.0, 1000.0),
        pub fast: bool = false => any,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
pub enum BodyProblem {
    NoCollider,
    NotRoot,
    NoTransform,
    BadScale,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct BodyWater {
    pub kind: WaterKind,
    pub submerged: f32,
    pub current: [f32; 3],
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BodyStatus {
    pub problem: Option<BodyProblem>,
    pub mass: f32,
    pub volume: f32,
    pub velocity: Vec3,
    pub spin: Vec3,
    pub sleeping: bool,
    pub held: bool,
    pub water: Option<BodyWater>,
}
