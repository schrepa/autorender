engine::brick!(puppet);

mod carry;
mod choose;
mod device;
mod drops;
mod heads;
mod props;
mod seen;

use std::collections::BTreeMap;
use std::sync::Arc;

use engine::prelude::*;
use engine_core::{AssetKind, AssetKinds, Assets, Error};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::play::puppet::{
    self, DeviceLamp, Muzzles, PlantedLook, Puppet, PuppetBook, PuppetFile, PuppetTable,
    PuppetTableFile, Seen, Tasks, WeaponLook,
};

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
    pub event_s: Option<f32>,
    pub hold: f32,
    pub crouch: f32,
    pub aim: f32,
    pub look_weight: f32,
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
        pub lobby: bool,
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

fn book_of(
    file: &PuppetFile,
    table: &Arc<PuppetTable>,
    assets: &Assets,
) -> Result<PuppetBook, Error> {
    let mut weapons = BTreeMap::new();
    for (key, files) in &file.weapons {
        let markers = *assets.get(files.markers)?;
        let files = files.clone();
        weapons.insert(key.clone(), WeaponLook { files, markers });
    }
    let planted = match &file.planted {
        Some(files) => Some(PlantedLook {
            files: files.clone(),
            markers: *assets.get(files.markers)?,
        }),
        None => None,
    };
    Ok(PuppetBook {
        table: Arc::clone(table),
        weapons,
        planted,
        hands: file.hands.clone(),
    })
}

fn read(table: Option<&PuppetTableFile>, file: Option<&PuppetFile>, assets: &Assets) -> Puppet {
    let table = match table.map(|table| assets.get(table.file)) {
        None => return Puppet::default(),
        Some(Err(error)) => {
            return Puppet {
                refusal: Some(error.to_string()),
                ..Puppet::default()
            };
        }
        Some(Ok(table)) => Arc::clone(table),
    };
    match file.map(|file| book_of(file, &table, assets)).transpose() {
        Ok(book) => Puppet {
            table: Some(table),
            book: book.map(Arc::new),
            refusal: None,
        },
        Err(error) => Puppet {
            table: Some(table),
            book: None,
            refusal: Some(error.to_string()),
        },
    }
}

fn adopt(
    (tables, files): (Query<Ref<PuppetTableFile>>, Query<Ref<PuppetFile>>),
    assets: Res<Assets>,
    mut puppet: ResMut<Puppet>,
) {
    let (table, file) = (tables.iter().next(), files.iter().next());
    let added = table.as_ref().is_some_and(|table| table.is_added())
        || file.as_ref().is_some_and(|file| file.is_added());
    let gone = (table.is_none() && (puppet.table.is_some() || puppet.refusal.is_some()))
        || (file.is_none() && puppet.book.is_some());
    if added || gone {
        *puppet = read(table.as_deref(), file.as_deref(), &assets);
    }
}

pub struct PuppetBrick;

impl Brick for PuppetBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .join::<SceneComponents>(
                ComponentKind::of::<PuppetFile>("puppet")
                    .schema(engine_core::schema_of::<PuppetFile>),
            )
            .join::<SceneComponents>(
                ComponentKind::of::<PuppetTableFile>("puppet_table")
                    .schema(engine_core::schema_of::<PuppetTableFile>),
            )
            .join::<AssetKinds>(AssetKind::of("puppet_table", puppet::table))
            .join::<AssetKinds>(AssetKind::of("markers", puppet::markers))
            .resource::<Puppet>()
            .resource::<Memories>()
            .resource::<Muzzles>()
            .resource::<Tasks>()
            .resource::<DeviceLamp>()
            .resource::<drops::Flights>()
            .moves::<DroppedWeapon>(Pace::Tick)
            .resource::<props::Spins>()
            .resource::<Seen>()
            .moves::<PropShown>(Pace::Tick)
            .systems(Schedule::BeforeTicks, adopt)
            .systems(Schedule::FixedSimulation, choose::animate)
            .systems(Schedule::FixedSimulation, carry::carry)
            .systems(Schedule::FixedSimulation, drops::lie)
            .systems(Schedule::FixedSimulation, device::place)
            .systems(Schedule::FixedSimulation, props::show)
            .systems(Schedule::Presentation, seen::cull)
            .dev(|dev| {
                dev.stats::<PuppetStats>()
                    .systems(Schedule::Presentation, carry::checked);
            });
    }
}
