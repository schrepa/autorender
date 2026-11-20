use std::collections::{BTreeMap, BTreeSet};

use engine_solids::Kind;
use serde::Deserialize;

use crate::play::numbers::GameNumbers;
use crate::play::world::STONE;

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct DeviceTiming {
    pub plant_repeat_seconds: f32,
    pub defuse_repeat_seconds: f32,
    pub beep_slow_seconds: f32,
    pub beep_fast_seconds: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct CueNumbers {
    pub quiet_step_gain: f32,
    pub drop_probe_m: f32,
    pub barrier_keep_ticks: u32,
    pub orb_hold_ticks: u32,
    pub break_large_m: f32,
    pub kill_steps: u32,
    pub countdown_seconds: u32,
    pub device: DeviceTiming,
    #[serde(skip)]
    pub default_kind: String,
    #[serde(skip)]
    pub named: BTreeMap<Kind, String>,
    #[serde(skip)]
    pub kill_confirms: Vec<String>,
}

#[derive(Deserialize)]
struct Heard {
    footstep_heard_m: f32,
}

#[derive(Deserialize)]
struct Read {
    sound_cues: CueNumbers,
    player: Heard,
}

pub(super) struct Cues {
    pub numbers: CueNumbers,
    pub reach: BTreeMap<String, f32>,
    pub expected: BTreeSet<String>,
}

const KIND_EVENTS: [&str; 5] = ["step", "impact", "land.soft", "land.hard", "drop"];

pub(super) fn read(game: &GameNumbers) -> Result<Cues, String> {
    let Read {
        sound_cues: mut numbers,
        player,
    } = engine::bricks::parse_value_at(&game.whole, "$").map_err(|error| error.to_string())?;
    if numbers.kill_steps == 0 {
        return Err("sound_cues.kill_steps must be above 0".to_owned());
    }
    numbers.kill_confirms = (1..=numbers.kill_steps)
        .map(|step| format!("kill.confirm.{step}"))
        .collect();
    let kinds = &game.player.surface_kinds;
    numbers.named = (kinds.iter())
        .map(|name| (Kind::named(name), name.clone()))
        .collect();
    numbers.default_kind = (numbers.named.get(&STONE).cloned())
        .ok_or_else(|| "player.surface_kinds has no stone".to_owned())?;
    Ok(Cues {
        numbers,
        reach: BTreeMap::from([("footstep_heard".to_owned(), player.footstep_heard_m)]),
        expected: (kinds.iter())
            .flat_map(|kind| KIND_EVENTS.map(|event| format!("{event}.{kind}")))
            .collect(),
    })
}

impl CueNumbers {
    pub fn kind(&self, kind: Kind) -> &str {
        self.named.get(&kind).unwrap_or(&self.default_kind)
    }
}
