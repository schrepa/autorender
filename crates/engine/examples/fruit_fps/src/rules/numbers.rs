use std::collections::BTreeMap;
use std::fmt;

use serde::Deserialize;

use super::{ArmorTier, Team, WeaponId};

#[derive(Clone, Debug, Deserialize)]
pub struct RulesNumbers {
    #[serde(rename = "match")]
    pub round: MatchNumbers,
    pub economy: EconomyNumbers,
    pub player: PlayerNumbers,
    pub weapons: BTreeMap<String, WeaponNumbers>,
    pub characters: BTreeMap<String, CharacterNumbers>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct MatchNumbers {
    pub team_size: usize,
    pub rounds_to_win: u32,
    pub swap_sides_after_round: u32,
    pub buy_seconds: f64,
    pub buy_seconds_first_round_of_half: f64,
    pub round_seconds: f64,
    pub plant_seconds: f64,
    pub defuse_seconds: f64,
    pub defuse_half_checkpoint: bool,
    pub device_fuse_seconds: f64,
    pub round_end_seconds: f64,
    pub device_name: String,
    pub sites: Vec<String>,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct ArmorNumbers {
    pub cost: u32,
    pub points: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct EconomyNumbers {
    pub start: u32,
    pub cap: u32,
    pub kill: u32,
    pub plant_team_bonus: u32,
    pub round_win: u32,
    pub round_loss_by_streak: Vec<u32>,
    pub light_armor: ArmorNumbers,
    pub heavy_armor: ArmorNumbers,
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct PlayerNumbers {
    pub health: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct WeaponNumbers {
    pub name: String,
    pub cost: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BoughtAbilityNumbers {
    pub name: String,
    pub cost: u32,
    pub charges: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct SignatureNumbers {
    pub name: String,
    pub charges: u32,
    pub cooldown_seconds: Option<f64>,
    pub refresh_on_kills: Option<u32>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct UltimateNumbers {
    pub name: String,
    pub points: u32,
    pub seconds: Option<f64>,
    pub dash_refresh_on_kill: Option<bool>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CharacterNumbers {
    pub name: String,
    pub role: String,
    pub c: BoughtAbilityNumbers,
    pub q: BoughtAbilityNumbers,
    pub e: SignatureNumbers,
    pub x: UltimateNumbers,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeaponSlot {
    Primary,
    Sidearm,
    Melee,
}

#[derive(Clone, Debug)]
pub struct WeaponEntry {
    pub key: String,
    pub name: String,
    pub cost: Option<u32>,
    pub slot: WeaponSlot,
}

#[derive(Clone, Copy, Debug)]
pub struct BoughtEntry {
    pub cost: u32,
    pub charges: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct SignatureEntry {
    pub charges: u32,
    pub cooldown_ticks: Option<u32>,
    pub refresh_on_kills: Option<u32>,
}

#[derive(Clone, Copy, Debug)]
pub struct UltimateEntry {
    pub cost: u32,
    pub active_ticks: Option<u32>,
    pub refresh_signature_on_kill: bool,
}

#[derive(Clone, Debug)]
pub struct CharacterEntry {
    pub key: String,
    pub name: String,
    pub role: String,
    pub ability_names: [String; 4],
    pub bought: [BoughtEntry; 2],
    pub signature: SignatureEntry,
    pub ultimate: UltimateEntry,
}

#[derive(Clone, Copy, Debug)]
pub struct Timings {
    pub buy: u32,
    pub buy_first_round_of_half: u32,
    pub round: u32,
    pub plant: u32,
    pub defuse: u32,
    pub defuse_checkpoint: Option<u32>,
    pub fuse: u32,
    pub round_end: u32,
}

#[derive(Clone, Debug)]
pub struct Catalog {
    pub ticks_per_second: u32,
    pub team_size: usize,
    pub rounds_to_win: u32,
    pub swap_sides_after_round: u32,
    pub timings: Timings,
    pub device_name: String,
    pub sites: Vec<String>,
    pub economy: EconomyNumbers,
    pub health: u32,
    pub weapons: Vec<WeaponEntry>,
    pub sidearm: WeaponId,
    pub characters: Vec<CharacterEntry>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RulesError {
    ZeroTicksPerSecond,
    ZeroTeamSize,
    ZeroRoundsToWin,
    NoSites,
    NoLossRewards,
    Sidearms(usize),
    RosterSize {
        found: usize,
        expected: usize,
    },
    TeamSize {
        team: Team,
        found: usize,
        expected: usize,
    },
    UnknownCharacter(String),
}

impl fmt::Display for RulesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RulesError::ZeroTicksPerSecond => write!(f, "ticks per second is 0"),
            RulesError::ZeroTeamSize => write!(f, "match.team_size is 0"),
            RulesError::ZeroRoundsToWin => write!(f, "match.rounds_to_win is 0"),
            RulesError::NoSites => write!(f, "match.sites is empty"),
            RulesError::NoLossRewards => write!(f, "economy.round_loss_by_streak is empty"),
            RulesError::Sidearms(found) => {
                write!(
                    f,
                    "weapons: {found} weapons cost 0, exactly one free sidearm is needed"
                )
            }
            RulesError::RosterSize { found, expected } => {
                write!(f, "roster has {found} slots, 2 x team_size = {expected}")
            }
            RulesError::TeamSize {
                team,
                found,
                expected,
            } => write!(
                f,
                "team {team:?} has {found} players, team_size is {expected}"
            ),
            RulesError::UnknownCharacter(key) => {
                write!(f, "character `{key}` is not in characters")
            }
        }
    }
}

impl std::error::Error for RulesError {}

pub fn ticks(seconds: f64, ticks_per_second: u32) -> u32 {
    (seconds * f64::from(ticks_per_second)).round() as u32
}

fn weapon_entry(key: &str, numbers: &WeaponNumbers) -> WeaponEntry {
    let slot = match numbers.cost {
        Some(0) => WeaponSlot::Sidearm,
        Some(_) => WeaponSlot::Primary,
        None => WeaponSlot::Melee,
    };
    WeaponEntry {
        key: key.to_owned(),
        name: numbers.name.clone(),
        cost: numbers.cost,
        slot,
    }
}

fn character_entry(key: &str, numbers: &CharacterNumbers, tps: u32) -> CharacterEntry {
    CharacterEntry {
        key: key.to_owned(),
        name: numbers.name.clone(),
        role: numbers.role.clone(),
        ability_names: [
            numbers.c.name.clone(),
            numbers.q.name.clone(),
            numbers.e.name.clone(),
            numbers.x.name.clone(),
        ],
        bought: [
            BoughtEntry {
                cost: numbers.c.cost,
                charges: numbers.c.charges,
            },
            BoughtEntry {
                cost: numbers.q.cost,
                charges: numbers.q.charges,
            },
        ],
        signature: SignatureEntry {
            charges: numbers.e.charges,
            cooldown_ticks: numbers.e.cooldown_seconds.map(|s| ticks(s, tps)),
            refresh_on_kills: numbers.e.refresh_on_kills,
        },
        ultimate: UltimateEntry {
            cost: numbers.x.points,
            active_ticks: numbers.x.seconds.map(|s| ticks(s, tps)),
            refresh_signature_on_kill: numbers.x.dash_refresh_on_kill.unwrap_or(false),
        },
    }
}

impl Catalog {
    pub fn new(numbers: &RulesNumbers, ticks_per_second: u32) -> Result<Catalog, RulesError> {
        let round = &numbers.round;
        if ticks_per_second == 0 {
            return Err(RulesError::ZeroTicksPerSecond);
        }
        if round.team_size == 0 {
            return Err(RulesError::ZeroTeamSize);
        }
        if round.rounds_to_win == 0 {
            return Err(RulesError::ZeroRoundsToWin);
        }
        if round.sites.is_empty() {
            return Err(RulesError::NoSites);
        }
        if numbers.economy.round_loss_by_streak.is_empty() {
            return Err(RulesError::NoLossRewards);
        }
        let weapons: Vec<WeaponEntry> = numbers
            .weapons
            .iter()
            .map(|(key, w)| weapon_entry(key, w))
            .collect();
        let sidearms: Vec<usize> = weapons
            .iter()
            .enumerate()
            .filter(|(_, w)| w.slot == WeaponSlot::Sidearm)
            .map(|(i, _)| i)
            .collect();
        let [sidearm] = sidearms[..] else {
            return Err(RulesError::Sidearms(sidearms.len()));
        };
        let tps = ticks_per_second;
        let defuse = ticks(round.defuse_seconds, tps);
        let timings = Timings {
            buy: ticks(round.buy_seconds, tps),
            buy_first_round_of_half: ticks(round.buy_seconds_first_round_of_half, tps),
            round: ticks(round.round_seconds, tps),
            plant: ticks(round.plant_seconds, tps),
            defuse,
            defuse_checkpoint: round.defuse_half_checkpoint.then_some(defuse / 2),
            fuse: ticks(round.device_fuse_seconds, tps),
            round_end: ticks(round.round_end_seconds, tps),
        };
        Ok(Catalog {
            ticks_per_second,
            team_size: round.team_size,
            rounds_to_win: round.rounds_to_win,
            swap_sides_after_round: round.swap_sides_after_round,
            timings,
            device_name: round.device_name.clone(),
            sites: round.sites.clone(),
            economy: numbers.economy.clone(),
            health: numbers.player.health,
            weapons,
            sidearm: WeaponId(sidearm),
            characters: numbers
                .characters
                .iter()
                .map(|(key, c)| character_entry(key, c, tps))
                .collect(),
        })
    }

    pub fn weapon(&self, id: WeaponId) -> Option<&WeaponEntry> {
        self.weapons.get(id.0)
    }

    pub fn weapon_id(&self, key: &str) -> Option<WeaponId> {
        self.weapons.iter().position(|w| w.key == key).map(WeaponId)
    }

    pub fn character_id(&self, key: &str) -> Option<usize> {
        self.characters.iter().position(|c| c.key == key)
    }

    pub fn armor(&self, tier: ArmorTier) -> &ArmorNumbers {
        match tier {
            ArmorTier::Light => &self.economy.light_armor,
            ArmorTier::Heavy => &self.economy.heavy_armor,
        }
    }

    pub fn loss_reward(&self, streak: u32) -> u32 {
        let table = &self.economy.round_loss_by_streak;
        let index = (streak as usize).min(table.len().saturating_sub(1));
        table.get(index).copied().unwrap_or(0)
    }

    pub fn primaries(&self) -> Vec<WeaponId> {
        let mut ids: Vec<(u32, usize)> = self
            .weapons
            .iter()
            .enumerate()
            .filter(|(_, w)| w.slot == WeaponSlot::Primary)
            .map(|(i, w)| (w.cost.unwrap_or(0), i))
            .collect();
        ids.sort_unstable();
        ids.into_iter().map(|(_, i)| WeaponId(i)).collect()
    }

    pub fn weapon_cost(&self, id: WeaponId) -> u32 {
        self.weapon(id).and_then(|w| w.cost).unwrap_or(0)
    }
}
