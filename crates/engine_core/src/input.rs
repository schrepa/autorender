use std::collections::BTreeSet;

use bevy_ecs::prelude::{Resource, World};
use glam::Vec2;
use serde::{Deserialize, Serialize};

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum MouseButton {
    Left,
    Middle,
    Right,
    Back,
    Forward,
}

impl MouseButton {
    pub const ALL: [Self; 5] = [
        Self::Left,
        Self::Middle,
        Self::Right,
        Self::Back,
        Self::Forward,
    ];

    pub fn of(index: u32) -> Option<Self> {
        Self::ALL.get(index as usize).copied()
    }
}

#[derive(
    Clone,
    Copy,
    Debug,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    schemars::JsonSchema,
)]
#[serde(rename_all = "snake_case")]
pub enum PadButton {
    South,
    East,
    West,
    North,
    LeftShoulder,
    RightShoulder,
    Select,
    Start,
    LeftStick,
    RightStick,
    Up,
    Down,
    Left,
    Right,
}

impl PadButton {
    pub const ALL: [(Self, u32); 14] = [
        (Self::South, 0),
        (Self::East, 1),
        (Self::West, 2),
        (Self::North, 3),
        (Self::LeftShoulder, 4),
        (Self::RightShoulder, 5),
        (Self::Select, 8),
        (Self::Start, 9),
        (Self::LeftStick, 10),
        (Self::RightStick, 11),
        (Self::Up, 12),
        (Self::Down, 13),
        (Self::Left, 14),
        (Self::Right, 15),
    ];

    pub fn held(buttons: u32) -> impl Iterator<Item = Self> {
        Self::ALL
            .into_iter()
            .filter(move |&(_, bit)| buttons & (1 << bit) != 0)
            .map(|(button, _)| button)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Buttons<B: Ord> {
    pub held: BTreeSet<B>,
    pub pressed: BTreeSet<B>,
    pub released: BTreeSet<B>,
}

impl<B: Ord> Default for Buttons<B> {
    fn default() -> Self {
        Self {
            held: BTreeSet::new(),
            pressed: BTreeSet::new(),
            released: BTreeSet::new(),
        }
    }
}

impl<B: Ord + Clone> Buttons<B> {
    pub fn set(&mut self, button: B, down: bool) {
        if down {
            if self.held.insert(button.clone()) {
                self.pressed.insert(button);
            }
        } else if self.held.remove(&button) {
            self.released.insert(button);
        }
    }

    pub fn drop_edges(&mut self) {
        self.pressed.clear();
        self.released.clear();
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Mouse {
    pub buttons: Buttons<MouseButton>,
    #[cfg_attr(feature = "schema", schemars(with = "Option<[f32; 2]>"))]
    pub position_pixels: Option<Vec2>,
    pub inside: bool,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub motion_pixels: Vec2,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub wheel_notches: Vec2,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Pad {
    pub index: u32,
    pub buttons: Buttons<PadButton>,
    #[cfg_attr(feature = "schema", schemars(with = "[[f32; 2]; 2]"))]
    pub sticks: [Vec2; 2],
    pub triggers: [f32; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum TouchPhase {
    Began = 0,
    Moved = 1,
    Ended = 2,
    Cancelled = 3,
}

impl TouchPhase {
    pub const ALL: [(Self, &'static str); 4] = [
        (Self::Began, "began"),
        (Self::Moved, "moved"),
        (Self::Ended, "ended"),
        (Self::Cancelled, "cancelled"),
    ];

    pub fn of(index: u32) -> Option<Self> {
        Self::ALL.get(index as usize).map(|&(phase, _)| phase)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct Touch {
    pub id: u32,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub position_pixels: Vec2,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub motion_pixels: Vec2,
    pub phase: TouchPhase,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct Devices {
    pub keyboard: Buttons<String>,
    pub mouse: Mouse,
    pub pads: Vec<Pad>,
    pub touches: Vec<Touch>,
}

impl Devices {
    #[must_use]
    pub fn held(&self) -> Self {
        let mut held = self.clone();
        held.keyboard.drop_edges();
        held.mouse.buttons.drop_edges();
        held.mouse.motion_pixels = Vec2::ZERO;
        held.mouse.wheel_notches = Vec2::ZERO;
        for pad in &mut held.pads {
            pad.buttons.drop_edges();
        }
        held.touches
            .retain(|touch| matches!(touch.phase, TouchPhase::Began | TouchPhase::Moved));
        for touch in &mut held.touches {
            touch.phase = TouchPhase::Moved;
            touch.motion_pixels = Vec2::ZERO;
        }
        held
    }

    fn edges(&self) -> Option<String> {
        let mut edges: Vec<String> = Vec::new();
        edges.extend(self.keyboard.pressed.iter().map(|key| format!("+{key}")));
        edges.extend(self.keyboard.released.iter().map(|key| format!("-{key}")));
        edges.extend(
            self.mouse
                .buttons
                .pressed
                .iter()
                .map(|button| format!("+{button:?}")),
        );
        edges.extend(
            self.mouse
                .buttons
                .released
                .iter()
                .map(|button| format!("-{button:?}")),
        );
        if self.mouse.wheel_notches != Vec2::ZERO {
            edges.push("wheel".to_owned());
        }
        if self.mouse.motion_pixels != Vec2::ZERO {
            edges.push("motion".to_owned());
        }
        for pad in &self.pads {
            edges.extend(
                pad.buttons
                    .pressed
                    .iter()
                    .map(|button| format!("+pad{} {button:?}", pad.index)),
            );
            edges.extend(
                pad.buttons
                    .released
                    .iter()
                    .map(|button| format!("-pad{} {button:?}", pad.index)),
            );
        }
        edges.extend(
            self.touches
                .iter()
                .filter(|touch| {
                    touch.phase != TouchPhase::Moved || touch.motion_pixels != Vec2::ZERO
                })
                .map(|touch| format!("touch{} {:?}", touch.id, touch.phase)),
        );
        (!edges.is_empty()).then(|| edges.join(" "))
    }
}

#[doc(hidden)]
pub fn accept_input(world: &mut World, devices: Devices) {
    if engine_profile::tracing()
        && let Some(edges) = devices.edges()
    {
        engine_profile::open_noting(
            "input",
            &edges,
            &[("systems", engine_profile::Fact::Unavailable("unattributed"))],
        );
    }
    world.insert_resource(devices);
}

#[doc(hidden)]
pub fn hold_input(world: &mut World) {
    if let Some(held) = world.get_resource::<Devices>().map(Devices::held) {
        world.insert_resource(held);
    }
}
