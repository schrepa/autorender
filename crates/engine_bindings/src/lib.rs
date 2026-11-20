#![deny(
    clippy::wildcard_enum_match_arm,
    clippy::match_wildcard_for_single_variants
)]

use engine_core::math::max;
use engine_core::{Devices, KeyCode, MouseButton};
use serde::{Deserialize, Deserializer, de::Error};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    Up,
    Down,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Input {
    Key(KeyCode),
    Mouse(MouseButton),
    Wheel(Spin),
}

pub type Binding = Vec<Input>;

const NAMED: [Input; 7] = [
    Input::Mouse(MouseButton::Left),
    Input::Mouse(MouseButton::Middle),
    Input::Mouse(MouseButton::Right),
    Input::Mouse(MouseButton::Back),
    Input::Mouse(MouseButton::Forward),
    Input::Wheel(Spin::Up),
    Input::Wheel(Spin::Down),
];

impl Input {
    pub fn named(name: &str) -> Option<Input> {
        (NAMED.into_iter())
            .find(|input| input.code() == name)
            .or_else(|| KeyCode::named(name).map(Input::Key))
    }

    pub fn code(&self) -> &'static str {
        match self {
            Input::Key(key) => key.name(),
            Input::Mouse(MouseButton::Left) => "MouseLeft",
            Input::Mouse(MouseButton::Middle) => "MouseMiddle",
            Input::Mouse(MouseButton::Right) => "MouseRight",
            Input::Mouse(MouseButton::Back) => "MouseBack",
            Input::Mouse(MouseButton::Forward) => "MouseForward",
            Input::Wheel(Spin::Up) => "WheelUp",
            Input::Wheel(Spin::Down) => "WheelDown",
        }
    }
}

impl<'de> Deserialize<'de> for Input {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let name = String::deserialize(deserializer)?;
        Input::named(&name).ok_or_else(|| D::Error::custom(format!("unknown input {name}")))
    }
}

fn spun(devices: &Devices, spin: Spin) -> f32 {
    let up = devices.mouse.wheel_notches.y;
    match spin {
        Spin::Up => max(up, 0.0),
        Spin::Down => max(-up, 0.0),
    }
}

pub fn held(devices: &Devices, binding: &[Input]) -> bool {
    binding.iter().any(|&input| match input {
        Input::Key(key) => devices.keyboard.held(key),
        Input::Mouse(button) => devices.mouse.buttons.held(button),
        Input::Wheel(_) => false,
    })
}

pub fn pressed(devices: &Devices, binding: &[Input]) -> bool {
    binding.iter().any(|&input| match input {
        Input::Key(key) => devices.keyboard.pressed(key),
        Input::Mouse(button) => devices.mouse.buttons.pressed(button),
        Input::Wheel(spin) => spun(devices, spin) > 0.0,
    })
}

pub fn turns(devices: &Devices, binding: &[Input]) -> f32 {
    (binding.iter())
        .map(|input| match input {
            Input::Wheel(spin) => spun(devices, *spin),
            Input::Key(_) | Input::Mouse(_) => f32::from(u8::from(pressed(
                devices,
                std::slice::from_ref(input),
            ))),
        })
        .sum()
}
