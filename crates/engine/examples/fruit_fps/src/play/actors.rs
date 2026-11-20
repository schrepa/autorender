use engine::prelude::{Component, Entity};
use engine_solids::Kind;
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::walk::Landed;
use crate::rules::{AbilitySlot, BuyRequest, Team};

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Actor {
    pub slot: u32,
    pub team: Team,
    pub character: String,
    #[serde(default)]
    pub yaw_radians: f32,
}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Human {}

#[derive(Component, Clone, Copy, Debug, Default, PartialEq)]
pub struct Alive;

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Health {
    pub health: f32,
    pub armour: f32,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct SpeedFactor(pub f32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Gait {
    #[default]
    Run,
    Walk,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Equip {
    Primary,
    Sidearm,
    Melee,
    Device,
}

#[derive(Component, Clone, Debug, Default, PartialEq)]
pub struct Wish {
    pub direction: Vec3,
    pub gait: Gait,
    pub crouch: bool,
    pub jump: bool,
    pub yaw_radians: f32,
    pub pitch_radians: f32,
    pub fire_held: bool,
    pub fire_pressed: bool,
    pub reload: bool,
    pub inspect: bool,
    pub drop: bool,
    pub cycle: i32,
    pub ready: bool,
    pub scope: bool,
    pub equip: Option<Equip>,
    pub interact_held: bool,
    pub cast: Option<AbilitySlot>,
    pub buys: Vec<BuyRequest>,
    pub buy_menu: bool,
    pub scoreboard: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Motion {
    Standing,
    Walking,
    Running,
    Crouching,
    Air,
}

#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub struct Eye {
    pub node: Entity,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub feet: Vec3,
    pub velocity: Vec3,
    pub grounded: bool,
    pub crouched: bool,
    pub crouch: f32,
    pub eye: Vec3,
    pub motion: Motion,
    pub ground: Option<Kind>,
    pub landed: Option<Landed>,
}
