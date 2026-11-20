use std::sync::Arc;

use engine::prelude::Resource;
use engine_bindings::Binding;
use serde::Deserialize;
use serde_json::Value;

use crate::play::walk::WalkNumbers;
use crate::rules::RulesNumbers;
use crate::weapons::{HitShapeNumbers, WeaponNumbers};

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PlayerNumbers {
    #[serde(flatten)]
    pub moving: WalkNumbers,
    pub health: f32,
    pub eye_height_m: f32,
    pub crouch_eye_height_m: f32,
    pub run_speed_mps: f32,
    pub walk_factor: f32,
    pub crouch_factor: f32,
    pub crouch_seconds: f32,
    pub stride_run_m: f32,
    pub stride_walk_m: f32,
    pub stride_crouch_m: f32,
    pub hard_landing_mps: f32,
    pub walk_is_silent: bool,
    pub push_apart_mps: f32,
    pub slow_jump_share: f32,
    pub barrier_column_m: f32,
    pub surface_kinds: Vec<String>,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct NavNumbers {
    pub weld_m: f32,
    pub locate_height_m: f32,
    pub locate_distance_m: f32,
    pub cover_gap_m: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct HazardFactors {
    pub damage: f32,
    pub strike: f32,
    pub slow: f32,
    pub near_sight: f32,
    pub reveal: f32,
    pub smoke: f32,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct PathsNumbers {
    pub heat_seconds: f32,
    pub heat_radius_m: f32,
    pub heat_factor: f32,
    pub hazard_factors: HazardFactors,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectateNumbers {
    pub corpse_seconds: f32,
    pub pull_back_m: f32,
    pub rise_m: f32,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeyMap {
    pub forward: Binding,
    pub back: Binding,
    pub left: Binding,
    pub right: Binding,
    pub walk: Binding,
    pub crouch: Binding,
    pub jump: Binding,
    pub fire: Binding,
    pub scope: Binding,
    pub reload: Binding,
    pub inspect: Binding,
    pub drop: Binding,
    pub next_item: Binding,
    pub previous_item: Binding,
    pub ready: Binding,
    pub primary: Binding,
    pub sidearm: Binding,
    pub melee: Binding,
    pub device: Binding,
    pub interact: Binding,
    pub ability_c: Binding,
    pub ability_q: Binding,
    pub ability_e: Binding,
    pub ability_x: Binding,
    pub buy_menu: Binding,
    pub scoreboard: Binding,
    pub capture: Binding,
    pub release: Binding,
}

#[derive(Clone, Debug)]
pub struct GameNumbers {
    pub player: PlayerNumbers,
    pub keys: KeyMap,
    pub spectate: SpectateNumbers,
    pub weapons: WeaponNumbers,
    pub hit_shapes: HitShapeNumbers,
    pub rules: RulesNumbers,
    pub nav: NavNumbers,
    pub paths: Option<PathsNumbers>,
    pub whole: Value,
}

#[derive(Resource, Clone, Debug, Default)]
pub struct Numbers {
    parsed: Option<Arc<GameNumbers>>,
    revision: u64,
}

impl Numbers {
    pub fn get(&self) -> Option<&GameNumbers> {
        self.parsed.as_deref()
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn adopt(&mut self, parsed: Option<Arc<GameNumbers>>) {
        self.parsed = parsed;
        self.revision += 1;
    }
}
