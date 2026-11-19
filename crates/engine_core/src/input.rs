use std::collections::BTreeSet;

use bevy_ecs::prelude::Resource;
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
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
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

    fn carry(&mut self, older: &Self) {
        self.pressed.extend(older.pressed.iter().cloned());
        self.released.extend(older.released.iter().cloned());
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Mouse {
    pub buttons: Buttons<MouseButton>,
    pub position_pixels: Option<Vec2>,
    pub inside: bool,
    pub motion_pixels: Vec2,
    pub wheel_notches: Vec2,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Pad {
    pub index: u32,
    pub buttons: Buttons<PadButton>,
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
pub struct Touch {
    pub id: u32,
    pub position_pixels: Vec2,
    pub motion_pixels: Vec2,
    pub phase: TouchPhase,
}

#[derive(Resource, Clone, Debug, Default, PartialEq, Serialize)]
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

    fn carry(&mut self, older: &Self) {
        self.keyboard.carry(&older.keyboard);
        self.mouse.buttons.carry(&older.mouse.buttons);
        self.mouse.motion_pixels += older.mouse.motion_pixels;
        self.mouse.wheel_notches += older.mouse.wheel_notches;
        for pad in &mut self.pads {
            if let Some(old) = older.pads.iter().find(|old| old.index == pad.index) {
                pad.buttons.carry(&old.buttons);
            }
        }
        for touch in &mut self.touches {
            if let Some(old) = older.touches.iter().find(|old| old.id == touch.id) {
                touch.motion_pixels += old.motion_pixels;
                if old.phase == TouchPhase::Began && touch.phase == TouchPhase::Moved {
                    touch.phase = TouchPhase::Began;
                }
            }
        }
        let ended: Vec<Touch> = older
            .touches
            .iter()
            .filter(|old| matches!(old.phase, TouchPhase::Ended | TouchPhase::Cancelled))
            .filter(|old| self.touches.iter().all(|touch| touch.id != old.id))
            .copied()
            .collect();
        self.touches.extend(ended);
    }
}

impl crate::Engine {
    pub fn accept_input(&mut self, devices: Devices) {
        let mut ticked = devices.clone();
        ticked.carry(&self.ticked);
        self.ticked = ticked;
        self.app.world_mut().insert_resource(devices);
    }

    pub fn drop_marks(&mut self) {
        self.drop_tick_marks();
        let world = self.app.world_mut();
        let held = world.resource::<Devices>().held();
        world.insert_resource(held);
    }

    pub(crate) fn drop_tick_marks(&mut self) {
        self.ticked = self.ticked.held();
    }

    pub(crate) fn run_ticks(&mut self, ticks: u32) {
        if ticks == 0 {
            return;
        }
        let held = self.ticked.held();
        let marked = std::mem::replace(&mut self.ticked, held);
        let frame = self.swap_input(marked);
        for tick in 0..ticks {
            if tick == 1 {
                let held = std::mem::take(&mut self.ticked);
                self.swap_input(held);
            }
            self.run_fixed_tick();
        }
        let held = self.swap_input(frame);
        if ticks > 1 {
            self.ticked = held;
        }
    }

    fn swap_input(&mut self, devices: Devices) -> Devices {
        let world = self.app.world_mut();
        let before = world.remove_resource::<Devices>().unwrap_or_default();
        world.insert_resource(devices);
        before
    }
}
