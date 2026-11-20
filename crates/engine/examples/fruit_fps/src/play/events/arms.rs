use engine::prelude::Resource;
use engine_solids::Kind;
use glam::Vec3;

use crate::rules::{AbilitySlot, Catalog, Cause, Death};
use crate::weapons::{BodyPart, GunKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Weapon {
    Gun(GunKind),
    Melee,
}

impl Weapon {
    pub fn key(self) -> &'static str {
        match self {
            Weapon::Gun(GunKind::Sidearm) => "sidearm",
            Weapon::Gun(GunKind::Smg) => "smg",
            Weapon::Gun(GunKind::Rifle) => "rifle",
            Weapon::Gun(GunKind::Sniper) => "sniper",
            Weapon::Melee => "melee",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Target {
    Nothing,
    World,
    Actor { slot: u32, part: BodyPart },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ShotEvent {
    pub shooter: u32,
    pub weapon: Weapon,
    pub origin: Vec3,
    pub direction: Vec3,
    pub end: Vec3,
    pub normal: Vec3,
    pub pierced: Option<Vec3>,
    pub pierced_kind: Option<Kind>,
    pub kind: Option<Kind>,
    pub heavy: bool,
    pub target: Target,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HitEvent {
    pub shooter: u32,
    pub victim: u32,
    pub weapon: Weapon,
    pub part: BodyPart,
    pub distance: f32,
    pub damage: f32,
    pub health_lost: f32,
    pub armor_lost: f32,
    pub health: f32,
    pub armor: f32,
    pub armor_broke: bool,
    pub killed: bool,
    pub headshot: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AbilityKill {
    pub slot: AbilitySlot,
    pub name: String,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeathEvent {
    pub victim: u32,
    pub killer: Option<u32>,
    pub assistants: Vec<u32>,
    pub headshot: bool,
    pub weapon: Option<Weapon>,
    pub ability: Option<AbilityKill>,
}

impl DeathEvent {
    pub fn rules_death(&self, catalog: &Catalog) -> Option<Death> {
        let cause = match (&self.ability, self.weapon) {
            (Some(kill), _) => Cause::Ability(kill.slot),
            (None, Some(weapon)) => Cause::Weapon(catalog.weapon_id(weapon.key())?),
            (None, None) => Cause::World,
        };
        Some(Death {
            victim: self.victim as usize,
            killer: self.killer.map(|killer| killer as usize),
            assistants: self.assistants.iter().map(|slot| *slot as usize).collect(),
            headshot: self.headshot,
            cause,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Handling {
    Equip,
    ReloadStarted,
    ReloadDone,
    DryFire,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HandlingEvent {
    pub slot: u32,
    pub weapon: Weapon,
    pub kind: Handling,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SwingEvent {
    pub slot: u32,
    pub heavy: bool,
    pub step: u8,
    pub strike_tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Inspect {
    Started { seconds: f32 },
    Cancelled,
    Ended,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InspectEvent {
    pub slot: u32,
    pub weapon: GunKind,
    pub change: Inspect,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PickupEvent {
    pub slot: u32,
    pub weapon: GunKind,
    pub left: Option<GunKind>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DropEvent {
    pub slot: u32,
    pub weapon: GunKind,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
pub struct ArmsEvents {
    pub tick: u64,
    pub shots: Vec<ShotEvent>,
    pub hits: Vec<HitEvent>,
    pub deaths: Vec<DeathEvent>,
    pub handling: Vec<HandlingEvent>,
    pub swings: Vec<SwingEvent>,
    pub inspects: Vec<InspectEvent>,
    pub pickups: Vec<PickupEvent>,
    pub drops: Vec<DropEvent>,
}
