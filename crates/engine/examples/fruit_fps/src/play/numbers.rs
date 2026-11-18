use std::sync::Arc;

use engine::prelude::Resource;
use engine::{Devices, MouseButton};
use serde::Deserialize;

use crate::rules::RulesNumbers;
use crate::solids::Player;
use crate::weapons::{HitShapeNumbers, WeaponNumbers};

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct PlayerNumbers {
    #[serde(flatten)]
    pub moving: Player,
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
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SpectateNumbers {
    pub corpse_seconds: f32,
    pub pull_back_m: f32,
    pub rise_m: f32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Input {
    Key(String),
    Mouse(MouseButton),
}

const MOUSE: [(&str, MouseButton); 5] = [
    ("MouseLeft", MouseButton::Left),
    ("MouseMiddle", MouseButton::Middle),
    ("MouseRight", MouseButton::Right),
    ("MouseBack", MouseButton::Back),
    ("MouseForward", MouseButton::Forward),
];

impl<'de> Deserialize<'de> for Input {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Ok(MOUSE
            .iter()
            .find(|(known, _)| *known == name)
            .map_or(Input::Key(name), |&(_, button)| Input::Mouse(button)))
    }
}

pub type Binding = Vec<Input>;

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

impl KeyMap {
    pub fn held(devices: &Devices, binding: &[Input]) -> bool {
        binding.iter().any(|input| match input {
            Input::Key(code) => devices.keyboard.held.contains(code),
            Input::Mouse(button) => devices.mouse.buttons.held.contains(button),
        })
    }

    pub fn pressed(devices: &Devices, binding: &[Input]) -> bool {
        binding.iter().any(|input| match input {
            Input::Key(code) => devices.keyboard.pressed.contains(code),
            Input::Mouse(button) => devices.mouse.buttons.pressed.contains(button),
        })
    }
}

#[derive(Clone, Debug)]
pub struct GameNumbers {
    pub player: PlayerNumbers,
    pub keys: KeyMap,
    pub spectate: SpectateNumbers,
    pub weapons: WeaponNumbers,
    pub hit_shapes: HitShapeNumbers,
    pub rules: RulesNumbers,
    pub text: Arc<str>,
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
