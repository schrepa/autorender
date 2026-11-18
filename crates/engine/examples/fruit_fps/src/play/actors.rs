use engine::prelude::Component;
use glam::Vec3;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::rules::{AbilitySlot, BuyRequest, Team};
use crate::solids::{Landed, Surface};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
enum TeamName {
    First,
    Second,
}

mod team_name {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::{Team, TeamName};

    pub(super) fn serialize<S: Serializer>(team: &Team, serializer: S) -> Result<S::Ok, S::Error> {
        match team {
            Team::First => TeamName::First,
            Team::Second => TeamName::Second,
        }
        .serialize(serializer)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        deserializer: D,
    ) -> Result<Team, D::Error> {
        Ok(match TeamName::deserialize(deserializer)? {
            TeamName::First => Team::First,
            TeamName::Second => Team::Second,
        })
    }
}

#[derive(Component, Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Actor {
    pub slot: u32,
    #[serde(with = "team_name")]
    #[schemars(with = "TeamName")]
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

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct Body {
    pub feet: Vec3,
    pub velocity: Vec3,
    pub grounded: bool,
    pub crouched: bool,
    pub crouch: f32,
    pub eye: Vec3,
    pub eye_before: Vec3,
    pub motion: Motion,
    pub ground: Option<Surface>,
    pub landed: Option<Landed>,
}
