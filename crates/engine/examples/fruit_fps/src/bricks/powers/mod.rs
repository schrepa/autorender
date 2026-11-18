engine::brick!(powers);

mod apply;
mod cast;
mod hold;
mod orbs;
mod publish;

use engine::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::abilities::{AbilityBook, AbilityId, OrbNumbers};
use crate::play::events::powers::PowerEvents;
use crate::play::numbers::Numbers;
use crate::play::powers::Powers;
use crate::solids::Kind;
use crate::weapons::{Absorb, DamageLedger, WeaponNumbers};

const RECENT_CUES: usize = 12;
const RECENT_ORBS: usize = 12;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum Relay {
    #[default]
    Own,
    Slot(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum TestSlot {
    C,
    Q,
    E,
    X,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct CastAt {
    pub id: u32,
    pub slot: u32,
    pub ability: TestSlot,
    pub eye: [f32; 3],
    pub yaw_deg: f32,
    pub pitch_deg: f32,
}

settings! {
    #[section("powers_test")]
    #[serde(deny_unknown_fields)]
    pub struct PowerSetup {
        pub free: bool = false => any,
        pub cast_as: Relay = Relay::Own => any,
        pub cast_at: Option<CastAt> = None => any,
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct Count {
    pub kind: String,
    pub count: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CastStat {
    pub tick: u64,
    pub slot: u32,
    pub ability: String,
    pub reason: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct SlotStat {
    pub slot: String,
    pub name: String,
    pub block: String,
    pub charges: u32,
    pub max: u32,
    pub cooldown_ticks: u32,
    pub ready: bool,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct CueStat {
    pub tick: u64,
    pub cue: String,
    pub slot: String,
    pub character: usize,
    pub at: [f32; 3],
    pub size: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct PropStat {
    pub effect: u32,
    pub prop: String,
    pub place: String,
    pub at: [f32; 3],
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct HeldStat {
    pub slot: u32,
    pub ability: String,
    pub name: String,
    pub since: u64,
    pub preview: bool,
    pub valid: bool,
    pub at: [f32; 3],
    pub shape: String,
    pub path_points: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct ActorPowers {
    pub slot: u32,
    pub blindness: f32,
    pub sight_limit_m: Option<f32>,
    pub speed: f32,
    pub fire_rate: f32,
    pub reload: f32,
    pub damage_taken: f32,
    pub spread: f32,
    pub revealed_to: [bool; 2],
    pub held: Option<String>,
    pub weapon_down_until: Option<u64>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrbChannelStat {
    pub slot: u32,
    pub progress: f32,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrbStat {
    pub name: String,
    pub at: [f32; 3],
    pub state: String,
    pub by: Option<u32>,
    pub channels: Vec<OrbChannelStat>,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize, JsonSchema)]
pub struct OrbEventStat {
    pub tick: u64,
    pub orb: u32,
    pub slot: Option<u32>,
    pub change: String,
}

stats! {
    #[section("powers")]
    pub struct PowerStats {
        pub refusal: Option<String>,
        pub tick: u64,
        pub free: bool,
        pub live: Vec<Count>,
        pub accepted: Vec<CastStat>,
        pub refused: Vec<CastStat>,
        pub events: Vec<Count>,
        pub casts_total: u64,
        pub refused_total: u64,
        pub last_cast: Option<CastStat>,
        pub last_refusal: Option<CastStat>,
        pub recent_cues: Vec<CueStat>,
        pub hidden_from_human: Vec<u32>,
        pub held: Vec<HeldStat>,
        pub props: Vec<PropStat>,
        pub holds: Vec<CastStat>,
        pub human_slots: Vec<SlotStat>,
        pub actors: Vec<ActorPowers>,
        pub orbs: Vec<OrbStat>,
        pub orb_problem: Option<String>,
        pub recent_orbs: Vec<OrbEventStat>,
    }
}

#[derive(Deserialize)]
struct AssistNumbers {
    assist_seconds: f32,
}

#[derive(Deserialize)]
struct PartNumbers {
    economy: Absorb,
    combat: AssistNumbers,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Parts {
    armor_absorb: f32,
    assist_ticks: u64,
    max_health: f32,
    radius_m: f32,
    height_m: f32,
    crouch_height_m: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct ShieldHeld {
    slot: u32,
    until: u64,
    armour_before: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct SolidKind {
    ability: AbilityId,
    kind: Kind,
}

#[derive(Resource, Default)]
struct Book {
    revision: Option<u64>,
    abilities: Option<AbilityBook>,
    parts: Option<Parts>,
    refusal: Option<String>,
    round_seen: Option<u64>,
    matches_seen: u32,
    shields: Vec<ShieldHeld>,
    ledger: DamageLedger,
    casts_total: u64,
    refused_total: u64,
    last_cast: Option<CastStat>,
    last_refusal: Option<CastStat>,
    recent_cues: Vec<CueStat>,
    solid_kinds: Vec<SolidKind>,
    ticks_per_second: u32,
    guns: Option<WeaponNumbers>,
    orbs: Option<OrbNumbers>,
    orb_problem: Option<String>,
    recent_orbs: Vec<OrbEventStat>,
    cast_at_done: Option<u32>,
}

pub struct PowersBrick;

impl Brick for PowersBrick {
    const SOURCE: &'static Source = &SOURCE;

    fn register(bricks: &mut Bricks<'_>) {
        bricks
            .settings::<PowerSetup>()
            .stats::<PowerStats>()
            .resource::<Powers>()
            .resource::<PowerEvents>()
            .resource::<Book>()
            .systems(Schedule::FixedSimulation, setup)
            .systems(Schedule::FixedSimulation, apply::act);
    }
}

fn ticks_per_second(time: &FixedTime) -> Option<u32> {
    (time.delta_seconds > 0.0).then(|| (1.0 / time.delta_seconds).round() as u32)
}

fn setup(
    time: Res<FixedTime>,
    numbers: Res<Numbers>,
    mut book: ResMut<Book>,
    mut powers: ResMut<Powers>,
) {
    let Some(game) = numbers.get() else {
        return;
    };
    let Some(tps) = ticks_per_second(&time) else {
        return;
    };
    if book.revision == Some(numbers.revision()) {
        return;
    }
    book.revision = Some(numbers.revision());
    let parsed = AbilityBook::parse(&game.text, tps)
        .map_err(|error| {
            let slot = error.slot.map_or("", |slot| slot.key());
            format!("abilities {} {}: {}", error.character, slot, error.message)
        })
        .and_then(|abilities| {
            serde_json::from_str::<PartNumbers>(&game.text)
                .map(|parts| (abilities, parts))
                .map_err(|error| error.to_string())
        });
    match parsed {
        Ok((abilities, parts)) => {
            let player = &game.player;
            book.parts = Some(Parts {
                armor_absorb: parts.economy.armor_absorb,
                assist_ticks: (parts.combat.assist_seconds * tps as f32).round() as u64,
                max_health: player.health,
                radius_m: player.moving.radius_m,
                height_m: player.moving.height_m,
                crouch_height_m: player.moving.crouch_height_m,
            });
            book.solid_kinds = abilities
                .characters
                .iter()
                .flat_map(|character| character.abilities.iter())
                .filter_map(|spec| {
                    let name = spec.solid_kind.as_deref()?;
                    Some(SolidKind {
                        ability: spec.id,
                        kind: Kind::named(name),
                    })
                })
                .collect();
            powers.surfaces = abilities
                .throw_surfaces
                .iter()
                .map(|surface| (Kind::named(&surface.kind), surface.bounce))
                .collect();
            book.ticks_per_second = tps;
            book.guns = Some(game.weapons.clone());
            powers.orbs = Default::default();
            match OrbNumbers::parse(&game.text, tps) {
                Ok(orbs) => {
                    book.orbs = Some(orbs);
                    book.orb_problem = None;
                }
                Err(problem) => {
                    book.orbs = None;
                    book.orb_problem = Some(problem);
                }
            }
            book.abilities = Some(abilities);
            book.refusal = None;
        }
        Err(reason) => {
            book.abilities = None;
            book.parts = None;
            book.refusal = Some(reason);
        }
    }
}
