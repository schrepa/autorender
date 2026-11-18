engine::brick!(puppet);

mod carry;
mod choose;
mod device;
mod drops;
mod locomotion;
mod props;

use std::collections::BTreeMap;

use engine::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::puppet::{DeviceLamp, Muzzles, PuppetFile, Tasks};

const PUPPET_PART: &str = "puppet";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ActorStats {
    pub slot: u32,
    pub base: String,
    pub rate: f32,
    pub upper: Option<String>,
    pub hold: f32,
    pub crouch: f32,
    pub aim: f32,
    pub weapon: Option<String>,
    pub dead: bool,
    pub look_yaw: f32,
    pub step_m: f32,
    pub reload: Option<f32>,
    pub dip: f32,
}

stats! {
    #[section("puppet")]
    pub struct PuppetStats {
        pub actors: Vec<ActorStats>,
        pub held: u32,
        pub dropped: u32,
        pub props: u32,
        pub props_skipped: u32,
        pub props_skipped_keys: Vec<String>,
        pub locomotion: Option<LocomotionStats>,
        pub head: Option<HeadStats>,
        pub refusal: Option<String>,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClipWeight {
    pub clip: String,
    pub weight: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LocomotionStats {
    pub slot: u32,
    pub clips: Vec<ClipWeight>,
    pub rate: f32,
    pub stride: f32,
    pub hip_turn: f32,
    pub speed_mps: f32,
    pub planted_mps: f32,
    pub planned_slide_mps: f32,
    pub slide_mps: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HeadStats {
    pub slot: u32,
    pub head_top_m: f32,
    pub height_m: f32,
    pub top_gap_m: f32,
    pub eye_m: f32,
    pub body_eye_m: f32,
    pub eye_gap_m: f32,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct CarriedWeapon {
    slot: u32,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct DroppedWeapon {
    id: u32,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct PropShown {
    effect: u32,
    part: u32,
}

#[derive(Component, Clone, Copy, Debug, PartialEq)]
pub struct DeviceShown {
    form: device::Form,
}

#[derive(Resource, Default)]
struct Memories {
    held: BTreeMap<Entity, choose::Memory>,
    abilities: Option<(u64, Option<std::sync::Arc<crate::abilities::AbilityBook>>)>,
}

pub struct PuppetBrick;

impl Brick for PuppetBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<PuppetFile>("puppet")
                    .schema(engine_core::schema_of::<PuppetFile>)
                    .files(PuppetFile::files)
                    .prepare(PuppetFile::prepared),
            )
            .resource::<Memories>()
            .resource::<Muzzles>()
            .resource::<Tasks>()
            .resource::<DeviceLamp>()
            .resource::<drops::Flights>()
            .moves::<DroppedWeapon>(Pace::Tick)
            .resource::<props::Spins>()
            .moves::<PropShown>(Pace::Tick)
            .stats::<PuppetStats>()
            .systems(Schedule::FixedSimulation, choose::animate)
            .systems(Schedule::FixedSimulation, carry::carry)
            .systems(Schedule::FixedSimulation, drops::lie)
            .systems(Schedule::FixedSimulation, device::place)
            .systems(Schedule::FixedSimulation, props::show);
    }
}
