//! Input every host gathers the same way between frames: keys by their UI
//! Events code names (`KeyW`, `Space`, `ArrowUp`...), each held while any
//! source holds it (a keyboard key, a finger on a touch button); the
//! movement keys' axes; the game's action keys' bits, held and pressed or
//! released since the last frame; and the look deltas of dragging, within
//! [`LOOK_LIMIT`]. Hosts turn their events into [`InputGatherer::key`] and
//! [`InputGatherer::look`]; each frame takes what gathered.

use engine_core::InputState;

use super::policy::{LOOK_LIMIT, MAX_ACTIONS};

/// The keys hosts report, by their UI Events code names: the movement keys
/// and those games may bind actions to.
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
];

/// The axes the movement keys move along.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Axis {
    Forward,
    Right,
    Up,
}

/// The movement keys: each one's axis, and whether it moves the positive
/// way along it.
const MOVEMENT: &[(&str, Axis, bool)] = &[
    ("KeyW", Axis::Forward, true),
    ("ArrowUp", Axis::Forward, true),
    ("KeyS", Axis::Forward, false),
    ("ArrowDown", Axis::Forward, false),
    ("KeyD", Axis::Right, true),
    ("ArrowRight", Axis::Right, true),
    ("KeyA", Axis::Right, false),
    ("ArrowLeft", Axis::Right, false),
    ("KeyE", Axis::Up, true),
    ("KeyQ", Axis::Up, false),
];

/// What a frame takes of the input gathered since the last one.
#[derive(Clone, Copy, Debug, Default)]
pub struct FrameInput {
    /// The movement axes and look deltas.
    pub state: InputState,
    /// The action bits held now.
    pub held: u32,
    /// The action bits pressed since the last frame.
    pub pressed: u32,
    /// The action bits released since the last frame.
    pub released: u32,
}

/// Input gathered between frames.
#[derive(Clone, Debug)]
pub struct InputGatherer {
    // The sources holding each key of `KEYS`, by its index.
    holds: [u8; KEYS.len()],
    // Each key's action bit, by its index; 0 where no action binds it.
    bits: [u32; KEYS.len()],
    // `MOVEMENT` with each key's index.
    movement: [(usize, Axis, bool); MOVEMENT.len()],
    yaw: f32,
    pitch: f32,
    held: u32,
    pressed: u32,
    released: u32,
}

impl InputGatherer {
    /// Input whose action keys are `actions`: UI Events code names by
    /// action index. An index from [`MAX_ACTIONS`] on, or a key [`KEYS`]
    /// lacks, fails.
    pub fn new(actions: &[(&str, u32)]) -> Result<Self, String> {
        let mut bits = [0; KEYS.len()];
        for &(name, action) in actions {
            if action >= MAX_ACTIONS {
                return Err(format!(
                    "action {action} of {name} must be 0 to {}",
                    MAX_ACTIONS - 1
                ));
            }
            let index = index(name).ok_or_else(|| format!("{name} names no key"))?;
            bits[index] |= 1 << action;
        }
        let movement = std::array::from_fn(|entry| {
            let (name, axis, way) = MOVEMENT[entry];
            (index(name).expect("movement keys are keys"), axis, way)
        });
        Ok(Self {
            holds: [0; KEYS.len()],
            bits,
            movement,
            yaw: 0.0,
            pitch: 0.0,
            held: 0,
            pressed: 0,
            released: 0,
        })
    }

    /// A source pressed (`down`) or released key `code`; whether the game
    /// uses the key, a movement or action key, which hosts then keep from
    /// anything else that would take it. Each source presses a key once:
    /// hosts leave out the repeats a held key sends.
    pub fn key(&mut self, code: &str, down: bool) -> bool {
        let used = |index: usize| {
            self.bits[index] != 0 || self.movement.iter().any(|&(key, _, _)| key == index)
        };
        let Some(index) = index(code).filter(|&index| used(index)) else {
            return false;
        };
        let bit = self.bits[index];
        let holds = &mut self.holds[index];
        if down {
            if *holds == 0 {
                self.held |= bit;
                self.pressed |= bit;
            }
            *holds = holds.saturating_add(1);
        } else if *holds > 0 {
            *holds -= 1;
            if *holds == 0 {
                self.held &= !bit;
                self.released |= bit;
            }
        }
        true
    }

    /// The view turned by a drag of `dx`, `dy` pixels: the frame's deltas
    /// add up within [`LOOK_LIMIT`] either way.
    pub fn look(&mut self, dx: f32, dy: f32) {
        let turn = |total: f32, delta: f32| {
            let delta = if delta.is_finite() { delta } else { 0.0 };
            engine_core::math::clamp(total + delta, -LOOK_LIMIT, LOOK_LIMIT)
        };
        self.yaw = turn(self.yaw, dx);
        self.pitch = turn(self.pitch, dy);
    }

    /// Nothing is held, and what the next frame would take goes: focus
    /// left, or the page took input away.
    pub fn clear(&mut self) {
        self.holds = [0; KEYS.len()];
        self.held = 0;
        self.take();
    }

    /// The frame's input, which it takes: the movement axes, the look
    /// deltas and the action bits.
    pub fn take(&mut self) -> FrameInput {
        let axis = |axis: Axis| {
            // Any key held along a way counts once.
            let way = |positive: bool| {
                self.movement.iter().any(|&(index, along, way)| {
                    along == axis && way == positive && self.holds[index] > 0
                })
            };
            f32::from(u8::from(way(true))) - f32::from(u8::from(way(false)))
        };
        let input = FrameInput {
            state: InputState {
                move_forward: axis(Axis::Forward),
                move_right: axis(Axis::Right),
                move_up: axis(Axis::Up),
                look_yaw: self.yaw,
                look_pitch: self.pitch,
            },
            held: self.held,
            pressed: self.pressed,
            released: self.released,
        };
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.pressed = 0;
        self.released = 0;
        input
    }
}

/// Key `code`'s index in [`KEYS`].
fn index(code: &str) -> Option<usize> {
    KEYS.iter().position(|&key| key == code)
}
