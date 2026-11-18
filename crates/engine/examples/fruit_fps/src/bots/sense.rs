use glam::Vec3;

use super::places::{Places, ZoneId};
use super::team::Order;

pub trait World {
    fn sees(&self, from: Vec3, to: Vec3) -> bool;
    fn path(&self, from: Vec3, to: Vec3, corners: &mut Vec<Vec3>) -> bool;
    fn cover_from(&self, threat: Vec3, near: Vec3, max_distance: f32) -> Option<Vec3>;
    fn places(&self) -> &Places;
    fn throw(&self, slot: usize, origin: Vec3, yaw: f32, pitch: f32) -> Option<Vec3>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GunClass {
    Sidearm,
    Smg,
    Rifle,
    Sniper,
    Melee,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WeaponChoice {
    Primary,
    Sidearm,
    Melee,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GunSense {
    pub class: GunClass,
    pub auto: bool,
    pub magazine: u32,
    pub magazine_size: u32,
    pub reserve: u32,
    pub reloading: bool,
    pub ready: bool,
    pub spread_deg: f32,
    pub scope_level: usize,
    pub recoil_pitch: f32,
    pub recoil_yaw: f32,
    pub range_m: f32,
    pub has_primary: bool,
    pub holding: WeaponChoice,
    pub has_scope: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Block {
    Smoke,
    Flash,
    Dash,
    Wall,
    Trap,
    Heal,
    Reveal,
    DamageZone,
    Buff,
}

impl Block {
    pub fn from_data_name(name: &str) -> Option<Block> {
        match name {
            "smoke" | "zone_nearsight" => Some(Block::Smoke),
            "flash" | "flash_cone" => Some(Block::Flash),
            "dash" => Some(Block::Dash),
            "wall_solid" | "wall_vision" => Some(Block::Wall),
            "trap" | "trap_reveal" | "trap_tripwire" => Some(Block::Trap),
            "zone_heal" | "team_heal" => Some(Block::Heal),
            "reveal_pulse" | "zone_slow_reveal" => Some(Block::Reveal),
            "zone_damage" | "grenade" | "airstrike" | "zone_slow" => Some(Block::DamageZone),
            "self_buff" => Some(Block::Buff),
            _ => None,
        }
    }

    pub fn bit(self) -> u16 {
        1 << self as u16
    }
}

pub fn tools_of(abilities: &[AbilitySense]) -> u16 {
    abilities
        .iter()
        .filter(|ability| ability.ready)
        .filter_map(|ability| ability.block)
        .fold(0, |tools, block| tools | block.bit())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbilitySense {
    pub block: Option<Block>,
    pub ready: bool,
    pub charges: u32,
    pub range_m: f32,
    pub radius_m: f32,
    pub thrown: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seen {
    pub id: u32,
    pub feet: Vec3,
    pub head: Vec3,
    pub chest: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    pub shows: f32,
    pub carries_device: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SoundKind {
    Step,
    Shot,
    Plant,
    Defuse,
    Ability,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Heard {
    pub kind: SoundKind,
    pub at: Vec3,
    pub loudness: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Mate {
    pub id: u32,
    pub position: Vec3,
    pub health: f32,
    pub alive: bool,
    pub human: bool,
    pub yaw: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Smoke {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hazard {
    pub center: Vec3,
    pub radius: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CallKind {
    Spotted,
    Heard,
    Device,
    Hurt,
    Down,
    Flash,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrbSense {
    pub at: Vec3,
    pub waiting: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ObjectKind {
    Trap,
    Wire,
    Sensor,
    Wall,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObjectSense {
    pub id: u32,
    pub kind: ObjectKind,
    pub at: Vec3,
    pub enemy: bool,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Call {
    pub from: u32,
    pub kind: CallKind,
    pub at: Vec3,
    pub zone: Option<ZoneId>,
    pub count: u8,
    pub tick: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GamePhase {
    Buy,
    Live,
    Planted,
    Ended,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Side {
    Attack,
    Defend,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeviceSense {
    Unknown,
    Carried {
        carrier: u32,
    },
    Dropped {
        at: Vec3,
    },
    Planted {
        site: usize,
        at: Vec3,
        fuse_ticks_left: u32,
        defuser: Option<u32>,
    },
    Over,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MatchSense {
    pub phase: GamePhase,
    pub side: Side,
    pub ticks_left: u32,
    pub round: u32,
    pub device: DeviceSense,
}

#[derive(Clone, Copy, Debug)]
pub struct Sense<'a> {
    pub me: usize,
    pub alive: bool,
    pub position: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub crouched: bool,
    pub grounded: bool,
    pub health: f32,
    pub armor: f32,
    pub gun: GunSense,
    pub abilities: [AbilitySense; 4],
    pub credits: u32,
    pub carrying_device: bool,
    pub revealed: bool,
    pub orbs: &'a [OrbSense],
    pub objects: &'a [ObjectSense],
    pub blind: f32,
    pub enemies: &'a [Seen],
    pub sounds: &'a [Heard],
    pub mates: &'a [Mate],
    pub calls: &'a [Call],
    pub smokes: &'a [Smoke],
    pub hazards: &'a [Hazard],
    pub game: MatchSense,
    pub order: Order,
    pub buy: BuySense,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct BuySense {
    pub device_asked: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Purchase {
    Nothing,
    Own,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gait {
    Still,
    Walk,
    Run,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AbilityUse {
    pub slot: usize,
    pub target: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Intent {
    pub wish: Vec3,
    pub gait: Gait,
    pub crouch: bool,
    pub jump: bool,
    pub yaw: f32,
    pub pitch: f32,
    pub fire: bool,
    pub reload: bool,
    pub scope: bool,
    pub switch: Option<WeaponChoice>,
    pub plant: bool,
    pub interact: bool,
    pub defuse: bool,
    pub ability: Option<AbilityUse>,
    pub drop_device: bool,
    pub buy: Purchase,
    pub call: Option<Call>,
    pub goal: Vec3,
}

impl Intent {
    pub fn rest(sense: &Sense<'_>) -> Intent {
        Intent {
            wish: Vec3::ZERO,
            gait: Gait::Still,
            crouch: false,
            jump: false,
            yaw: sense.yaw,
            pitch: sense.pitch,
            fire: false,
            reload: false,
            scope: false,
            switch: None,
            plant: false,
            interact: false,
            defuse: false,
            ability: None,
            drop_device: false,
            buy: Purchase::Nothing,
            call: None,
            goal: sense.position,
        }
    }
}
