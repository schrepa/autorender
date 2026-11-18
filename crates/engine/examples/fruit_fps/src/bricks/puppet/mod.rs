engine::brick!(puppet);

mod carry;
mod choose;
mod device;
mod drops;
mod locomotion;
mod props;
mod seen;

use std::collections::BTreeMap;

use engine::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::puppet::{DeviceLamp, Muzzles, PuppetFile, Seen, Tasks};

const PUPPET_PART: &str = "puppet";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ActorStats {
    pub slot: u32,
    pub base: String,
    pub rate: f32,
    pub upper: Option<String>,
    pub upper_seconds: f32,
    pub upper_rate: f32,
    pub upper_active: bool,
    pub hold: f32,
    pub crouch: f32,
    pub aim: f32,
    pub weapon: Option<String>,
    pub dead: bool,
    pub look_yaw: f32,
    pub step_m: f32,
    pub drift_m: f32,
    pub reload: Option<f32>,
    pub dip: f32,
    pub in_hand: Option<String>,
    pub team: u32,
    pub feet: [f32; 3],
    pub eye: [f32; 3],
}

stats! {
    #[section("puppet")]
    pub struct PuppetStats {
        pub actors: Vec<ActorStats>,
        pub held: u32,
        pub dropped: u32,
        pub props: Vec<PropStats>,
        pub props_skipped: u32,
        pub props_skipped_keys: Vec<String>,
        pub in_hand: Vec<InHandStats>,
        pub locomotion: Option<LocomotionStats>,
        pub head: Option<HeadStats>,
        pub aim: Option<AimStats>,
        pub eyes: Vec<EyeFitStats>,
        pub missing_clips: Vec<String>,
        pub missing_count: u32,
        pub formless: Vec<String>,
        pub carried: Vec<CarriedStats>,
        pub refusal: Option<String>,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PropStats {
    pub key: String,
    pub part: u32,
    pub position: [f32; 3],
    pub up: [f32; 3],
    pub scale: [f32; 3],
    pub ally: Option<bool>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct InHandStats {
    pub owner: u32,
    pub first_person: bool,
    pub prop: String,
    pub joint: String,
    pub size_m: f32,
    pub extent_m: Option<f32>,
    pub scale: Option<f32>,
    pub shown: bool,
    pub cast_tick: Option<u64>,
    pub release_tick: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CarriedStats {
    pub slot: u32,
    pub parented: bool,
    pub shown: bool,
    pub holder_shown: bool,
    pub drawn: bool,
    pub grip_gap_m: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ClipWeight {
    pub clip: String,
    pub weight: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct LocomotionStats {
    pub slot: u32,
    pub state: String,
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
pub struct EyeFitStats {
    pub character: String,
    pub idle_eye_m: f32,
    pub body_eye_m: f32,
    pub gap_m: f32,
    pub fit: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct AimStats {
    pub slot: u32,
    pub weapon: String,
    pub barrel_error_deg: f32,
    pub hold: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HeadStats {
    pub slot: u32,
    pub head_top_m: f32,
    pub height_m: f32,
    pub top_gap_m: f32,
    pub crown_m: f32,
    pub crown_gap_m: f32,
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
pub struct PropInHand {
    holder: Entity,
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
            .resource::<Seen>()
            .moves::<PropShown>(Pace::Tick)
            .stats::<PuppetStats>()
            .systems(Schedule::FixedSimulation, choose::animate)
            .systems(Schedule::FixedSimulation, carry::carry)
            .systems(Schedule::FixedSimulation, drops::lie)
            .systems(Schedule::FixedSimulation, device::place)
            .systems(Schedule::FixedSimulation, props::show)
            .systems(Schedule::Presentation, seen::cull)
            .systems(Schedule::Presentation, carry::checked);
    }
}
