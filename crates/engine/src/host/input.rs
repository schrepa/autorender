use std::collections::BTreeSet;

use engine_core::{
    AxisId, AxisInput, Error, ErrorCode, Expected, MenuInput, MoveInput, PointerInput,
};
use glam::Vec2;

use super::policy::LOOK_LIMIT;

pub const KEYS: &[&str] = &[
    "Space",
    "Enter",
    "Tab",
    "Escape",
    "Backspace",
    "ShiftLeft",
    "ShiftRight",
    "ControlLeft",
    "ControlRight",
    "AltLeft",
    "AltRight",
    "KeyA",
    "KeyB",
    "KeyC",
    "KeyD",
    "KeyE",
    "KeyF",
    "KeyG",
    "KeyH",
    "KeyI",
    "KeyJ",
    "KeyK",
    "KeyL",
    "KeyM",
    "KeyN",
    "KeyO",
    "KeyP",
    "KeyQ",
    "KeyR",
    "KeyS",
    "KeyT",
    "KeyU",
    "KeyV",
    "KeyW",
    "KeyX",
    "KeyY",
    "KeyZ",
    "Digit0",
    "Digit1",
    "Digit2",
    "Digit3",
    "Digit4",
    "Digit5",
    "Digit6",
    "Digit7",
    "Digit8",
    "Digit9",
    "ArrowUp",
    "ArrowDown",
    "ArrowLeft",
    "ArrowRight",
    "F1",
    "F2",
    "F3",
    "F4",
    "F5",
    "F6",
    "F7",
    "F8",
    "F9",
    "F10",
    "F11",
    "F12",
    "Minus",
    "Equal",
    "BracketLeft",
    "BracketRight",
    "Semicolon",
    "Quote",
    "Backquote",
    "Backslash",
    "Comma",
    "Period",
    "Slash",
    "CapsLock",
    "Delete",
    "Insert",
    "Home",
    "End",
    "PageUp",
    "PageDown",
    "MouseLeft",
    "MouseRight",
    "MouseMiddle",
    "PadSouth",
    "PadEast",
    "PadWest",
    "PadNorth",
    "PadLeftShoulder",
    "PadRightShoulder",
    "PadSelect",
    "PadStart",
    "PadLeftStick",
    "PadRightStick",
    "PadUp",
    "PadDown",
    "PadLeft",
    "PadRight",
];

const ANALOG: [&str; 6] = [
    "PadLeftX",
    "PadLeftY",
    "PadRightX",
    "PadRightY",
    "PadLeftTrigger",
    "PadRightTrigger",
];

const PAD_BUTTONS: [(&str, u32); 14] = [
    ("PadSouth", 0),
    ("PadEast", 1),
    ("PadWest", 2),
    ("PadNorth", 3),
    ("PadLeftShoulder", 4),
    ("PadRightShoulder", 5),
    ("PadSelect", 8),
    ("PadStart", 9),
    ("PadLeftStick", 10),
    ("PadRightStick", 11),
    ("PadUp", 12),
    ("PadDown", 13),
    ("PadLeft", 14),
    ("PadRight", 15),
];

const STICK_DEAD_ZONE: f32 = 0.15;
const PAD_LOOK_RATE: f32 = 600.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuKey {
    Up,
    Down,
    Left,
    Right,
    Accept,
    Back,
}

impl MenuKey {
    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

pub const MENU_KEYS: &[(&str, MenuKey)] = &[
    ("ArrowUp", MenuKey::Up),
    ("PadUp", MenuKey::Up),
    ("ArrowDown", MenuKey::Down),
    ("Tab", MenuKey::Down),
    ("PadDown", MenuKey::Down),
    ("ArrowLeft", MenuKey::Left),
    ("PadLeft", MenuKey::Left),
    ("ArrowRight", MenuKey::Right),
    ("PadRight", MenuKey::Right),
    ("Enter", MenuKey::Accept),
    ("PadSouth", MenuKey::Accept),
    ("Escape", MenuKey::Back),
    ("PadEast", MenuKey::Back),
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Source {
    Key(usize),
    Analog(usize),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoveDirection {
    Forward,
    Back,
    Right,
    Left,
    Up,
    Down,
}

pub const MOVE_KEYS: &[(&str, MoveDirection)] = &[
    ("KeyW", MoveDirection::Forward),
    ("ArrowUp", MoveDirection::Forward),
    ("KeyS", MoveDirection::Back),
    ("ArrowDown", MoveDirection::Back),
    ("KeyD", MoveDirection::Right),
    ("ArrowRight", MoveDirection::Right),
    ("KeyA", MoveDirection::Left),
    ("ArrowLeft", MoveDirection::Left),
    ("KeyE", MoveDirection::Up),
    ("KeyQ", MoveDirection::Down),
];

#[derive(Clone, Debug, Default)]
pub struct FrameInput {
    pub state: MoveInput,
    pub held: BTreeSet<u32>,
    pub pressed: BTreeSet<u32>,
    pub released: BTreeSet<u32>,
    pub pointer: PointerInput,
    pub axes: AxisInput,
    pub menu: MenuInput,
}

#[derive(Clone, Debug)]
pub struct InputGatherer {
    holds: [u8; KEYS.len()],
    actions: Vec<Vec<u32>>,
    menu: [u8; KEYS.len()],
    primary: usize,
    movement: Vec<(usize, MoveDirection)>,
    axes: Vec<(Source, usize, f32)>,
    yaw: f32,
    pitch: f32,
    held: BTreeSet<u32>,
    pressed: BTreeSet<u32>,
    released: BTreeSet<u32>,
    menu_pressed: u8,
    pointer: PointerInput,
    pad_buttons: u32,
    pad_held: u32,
    analog: [f32; 6],
    timestamp: Option<f64>,
}

impl InputGatherer {
    pub fn new(
        actions: &[(&str, u32)],
        axes: &[(&str, u32, f32)],
        move_keys: &[(&str, MoveDirection)],
        menu_keys: &[(&str, MenuKey)],
    ) -> Result<Self, Error> {
        let mut bound = vec![Vec::new(); KEYS.len()];
        for (at, &(name, action)) in actions.iter().enumerate() {
            let index = index(name).ok_or_else(|| {
                Error::new(ErrorCode::InvalidInput, format!("$.actions[{at}]"))
                    .got(name)
                    .variants(KEYS.iter().copied())
            })?;
            bound[index].push(action);
        }
        Ok(Self {
            holds: [0; KEYS.len()],
            actions: bound,
            menu: menu_table(menu_keys)?,
            primary: index("MouseLeft").expect("mouse_left_known"),
            movement: movement_keys(move_keys)?,
            axes: axis_sources(axes)?,
            yaw: 0.0,
            pitch: 0.0,
            held: BTreeSet::new(),
            pressed: BTreeSet::new(),
            released: BTreeSet::new(),
            menu_pressed: 0,
            pointer: PointerInput::default(),
            pad_buttons: 0,
            pad_held: 0,
            analog: [0.0; 6],
            timestamp: None,
        })
    }

    pub fn key(&mut self, code: &str, down: bool, menu_wanted: bool) -> bool {
        let Some(index) = index(code) else {
            return false;
        };
        let menu = self.menu[index];
        let source = Source::Key(index);
        let primary = index == self.primary;
        let used = primary
            || !self.actions[index].is_empty()
            || self.holds[index] > 0
            || self.movement.iter().any(|&(key, _)| key == index)
            || self.axes.iter().any(|&(key, _, _)| key == source)
            || (menu_wanted && menu != 0);
        if down && self.holds[index] == 0 {
            self.menu_pressed |= menu;
        }
        if !used {
            return false;
        }
        let actions = &self.actions[index];
        let holds = &mut self.holds[index];
        if down {
            if *holds == 0 {
                self.held.extend(actions);
                self.pressed.extend(actions);
                self.pointer.held |= primary;
                self.pointer.pressed |= primary;
            }
            *holds = holds.saturating_add(1);
        } else if *holds > 0 {
            *holds -= 1;
            if *holds == 0 {
                for action in actions {
                    self.held.remove(action);
                }
                self.released.extend(actions);
                self.pointer.held &= !primary;
                self.pointer.released |= primary;
            }
        }
        true
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        let turn = |total: f32, delta: f32| {
            engine_core::math::clamp(total + finite(delta), -LOOK_LIMIT, LOOK_LIMIT)
        };
        self.yaw = turn(self.yaw, dx);
        self.pitch = turn(self.pitch, dy);
    }

    pub fn motion(&mut self, dx: f32, dy: f32) {
        self.pointer.delta_pixels += Vec2::new(finite(dx), finite(dy));
        self.look(dx, dy);
    }

    pub fn pointer(&mut self, x: f32, y: f32) {
        let position = Vec2::new(finite(x), finite(y));
        if self.pointer.inside
            && let Some(last) = self.pointer.position_pixels
        {
            self.pointer.delta_pixels += position - last;
        }
        self.pointer.position_pixels = Some(position);
        self.pointer.inside = true;
    }

    pub fn pointer_left(&mut self) {
        self.pointer.inside = false;
    }

    pub fn wheel(&mut self, right_notches: f32, up_notches: f32) {
        self.pointer.wheel_right_notches += finite(right_notches);
        self.pointer.wheel_up_notches += finite(up_notches);
    }

    pub fn pad(&mut self, buttons: u32, [lx, ly, rx, ry, lt, rt]: [f32; 6]) {
        self.pad_buttons |= buttons;
        let [lx, ly] = stick(lx, ly);
        let [rx, ry] = stick(rx, ry);
        let [lt, rt] = [lt, rt].map(|value| engine_core::math::clamp(finite(value), 0.0, 1.0));
        for (merged, value) in self.analog.iter_mut().zip([lx, ly, rx, ry, lt, rt]) {
            if value.abs() > merged.abs() {
                *merged = value;
            }
        }
    }

    pub fn clear(&mut self) {
        self.holds = [0; KEYS.len()];
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.held.clear();
        self.pressed.clear();
        self.released.clear();
        self.menu_pressed = 0;
        self.pointer = PointerInput::default();
        self.pad_buttons = 0;
        self.analog = [0.0; 6];
        self.timestamp = None;
    }

    pub fn take(&mut self, timestamp: f64) -> FrameInput {
        self.take_pad_buttons();
        let seconds = self
            .timestamp
            .map_or(0.0, |last| ((timestamp - last).max(0.0) * 0.001) as f32);
        self.timestamp = Some(timestamp);
        let analog = std::mem::take(&mut self.analog);
        let [
            left_x,
            left_y,
            right_x,
            right_y,
            left_trigger,
            right_trigger,
        ] = analog;
        let turn = PAD_LOOK_RATE * seconds;
        let unit = |value: f32| engine_core::math::clamp(value, -1.0, 1.0);
        let axis = |positive: MoveDirection, negative: MoveDirection| {
            let held = |wanted: MoveDirection| {
                self.movement
                    .iter()
                    .any(|&(index, direction)| direction == wanted && self.holds[index] > 0)
            };
            f32::from(u8::from(held(positive))) - f32::from(u8::from(held(negative)))
        };
        let input = FrameInput {
            state: MoveInput {
                move_forward: unit(axis(MoveDirection::Forward, MoveDirection::Back) + left_y),
                move_right: unit(axis(MoveDirection::Right, MoveDirection::Left) + left_x),
                move_up: unit(
                    axis(MoveDirection::Up, MoveDirection::Down) + right_trigger - left_trigger,
                ),
                look_right_pixels: self.yaw + right_x * turn,
                look_down_pixels: self.pitch - right_y * turn,
            },
            held: self.held.clone(),
            pressed: std::mem::take(&mut self.pressed),
            released: std::mem::take(&mut self.released),
            pointer: self.pointer,
            axes: AxisInput::of(self.axis_values(&analog)),
            menu: menu_input(self.menu_pressed),
        };
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.menu_pressed = 0;
        self.pointer.delta_pixels = Vec2::ZERO;
        self.pointer.wheel_right_notches = 0.0;
        self.pointer.wheel_up_notches = 0.0;
        self.pointer.pressed = false;
        self.pointer.released = false;
        input
    }

    fn take_pad_buttons(&mut self) {
        let changed = self.pad_buttons ^ self.pad_held;
        for (code, bit) in PAD_BUTTONS {
            let mask = 1_u32 << bit;
            if changed & mask != 0 {
                self.key(code, self.pad_buttons & mask != 0, true);
            }
        }
        self.pad_held = self.pad_buttons;
        self.pad_buttons = 0;
    }

    fn axis_values(&self, analog: &[f32; 6]) -> [f32; AxisId::COUNT as usize] {
        let mut values = [0.0; AxisId::COUNT as usize];
        for &(source, axis, scale) in &self.axes {
            let value = match source {
                Source::Key(index) => f32::from(u8::from(self.holds[index] > 0)),
                Source::Analog(index) => analog[index],
            };
            values[axis] += value * scale;
        }
        values.map(|value| engine_core::math::clamp(value, -1.0, 1.0))
    }
}

fn index(code: &str) -> Option<usize> {
    KEYS.iter().position(|&key| key == code)
}

fn movement_keys(keys: &[(&str, MoveDirection)]) -> Result<Vec<(usize, MoveDirection)>, Error> {
    let mut movement = Vec::with_capacity(keys.len());
    for (at, &(name, direction)) in keys.iter().enumerate() {
        let key = index(name).ok_or_else(|| {
            Error::new(ErrorCode::InvalidInput, format!("$.move_keys[{at}]"))
                .got(name)
                .variants(KEYS.iter().copied())
        })?;
        movement.push((key, direction));
    }
    Ok(movement)
}

fn menu_table(keys: &[(&str, MenuKey)]) -> Result<[u8; KEYS.len()], Error> {
    let mut menu = [0; KEYS.len()];
    for (at, &(name, press)) in keys.iter().enumerate() {
        let key = index(name).ok_or_else(|| {
            Error::new(ErrorCode::InvalidInput, format!("$.menu_keys[{at}]"))
                .got(name)
                .variants(KEYS.iter().copied())
        })?;
        menu[key] |= press.bit();
    }
    Ok(menu)
}

fn axis_sources(axes: &[(&str, u32, f32)]) -> Result<Vec<(Source, usize, f32)>, Error> {
    let mut sources = Vec::with_capacity(axes.len());
    for (at, &(name, axis, scale)) in axes.iter().enumerate() {
        let path = format!("$.axes[{at}]");
        if axis >= AxisId::COUNT {
            return Err(Error::new(ErrorCode::InvalidInput, path)
                .at_most(u64::from(AxisId::COUNT - 1))
                .got(axis));
        }
        if !scale.is_finite() {
            return Err(Error::new(ErrorCode::InvalidInput, path)
                .expected(Expected::Finite)
                .got(scale));
        }
        let analog = ANALOG.iter().position(|&source| source == name);
        let source = index(name)
            .map(Source::Key)
            .or_else(|| analog.map(Source::Analog))
            .ok_or_else(|| {
                Error::new(ErrorCode::InvalidInput, path)
                    .got(name)
                    .variants(KEYS.iter().chain(&ANALOG).copied())
            })?;
        sources.push((source, axis as usize, scale));
    }
    Ok(sources)
}

fn stick(x: f32, y: f32) -> [f32; 2] {
    let (x, y) = (finite(x), finite(y));
    let length = (x * x + y * y).sqrt();
    if length <= STICK_DEAD_ZONE {
        return [0.0, 0.0];
    }
    let reach = engine_core::math::min(length, 1.0) - STICK_DEAD_ZONE;
    let scale = reach / (1.0 - STICK_DEAD_ZONE) / length;
    [x * scale, y * scale]
}

fn finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn menu_input(pressed: u8) -> MenuInput {
    let has = |key: MenuKey| pressed & key.bit() != 0;
    MenuInput {
        up: has(MenuKey::Up),
        down: has(MenuKey::Down),
        left: has(MenuKey::Left),
        right: has(MenuKey::Right),
        accept: has(MenuKey::Accept),
        back: has(MenuKey::Back),
        wanted: false,
    }
}
