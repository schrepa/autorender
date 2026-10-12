//! Keyboard and mouse input as the page reads it: movement keys by their
//! physical position, the game's action keys by the page's key names, and
//! look deltas from dragging with the left button.

use std::collections::HashMap;

use engine::host::policy::{LOOK_LIMIT, MAX_ACTIONS};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

/// Input gathered between frames: held keys, action edges and look deltas.
pub(crate) struct Input {
    keys: std::collections::HashSet<KeyCode>,
    // Each bound key's action bit.
    bindings: HashMap<KeyCode, u32>,
    dragging: bool,
    cursor: Option<(f64, f64)>,
    pub yaw: f32,
    pub pitch: f32,
    pub held: u32,
    pub pressed: u32,
    pub released: u32,
}

impl Input {
    /// Input with the game's `actions`: the page's key names (`Space`,
    /// `KeyF`...) by action index.
    pub(crate) fn new(actions: &[(&str, u32)]) -> engine::Result<Self> {
        let mut bindings = HashMap::new();
        for &(name, action) in actions {
            if action >= MAX_ACTIONS {
                return Err(engine::Error::new(
                    engine::ErrorCode::InvalidInput,
                    "$.actions",
                    format!("action {action} of {name} must be 0 to {}", MAX_ACTIONS - 1),
                ));
            }
            let code = key_code(name).ok_or_else(|| {
                engine::Error::new(
                    engine::ErrorCode::InvalidInput,
                    "$.actions",
                    format!("{name} names no key"),
                )
            })?;
            bindings.insert(code, 1 << action);
        }
        Ok(Self {
            keys: Default::default(),
            bindings,
            dragging: false,
            cursor: None,
            yaw: 0.0,
            pitch: 0.0,
            held: 0,
            pressed: 0,
            released: 0,
        })
    }

    pub(crate) fn key(&mut self, code: KeyCode, state: ElementState) {
        let bit = self.bindings.get(&code).copied().unwrap_or(0);
        match state {
            ElementState::Pressed => {
                if self.keys.insert(code) {
                    self.held |= bit;
                    self.pressed |= bit;
                }
            }
            ElementState::Released => {
                if self.keys.remove(&code) {
                    self.held &= !bit;
                    self.released |= bit;
                }
            }
        }
    }

    pub(crate) fn button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.dragging = state == ElementState::Pressed;
        }
    }

    /// The cursor moved to `position`, in logical pixels.
    pub(crate) fn cursor(&mut self, position: (f64, f64)) {
        if let Some((x, y)) = self.cursor
            && self.dragging
        {
            let turn = |value: f32| engine::math::clamp(value, -LOOK_LIMIT, LOOK_LIMIT);
            self.yaw = turn(self.yaw + (position.0 - x) as f32);
            self.pitch = turn(self.pitch + (position.1 - y) as f32);
        }
        self.cursor = Some(position);
    }

    fn axis(&self, positive: [KeyCode; 2], negative: [KeyCode; 2]) -> f32 {
        let any = |codes: [KeyCode; 2]| codes.iter().any(|code| self.keys.contains(code));
        f32::from(u8::from(any(positive))) - f32::from(u8::from(any(negative)))
    }

    pub(crate) fn forward(&self) -> f32 {
        self.axis(
            [KeyCode::KeyW, KeyCode::ArrowUp],
            [KeyCode::KeyS, KeyCode::ArrowDown],
        )
    }

    pub(crate) fn right(&self) -> f32 {
        self.axis(
            [KeyCode::KeyD, KeyCode::ArrowRight],
            [KeyCode::KeyA, KeyCode::ArrowLeft],
        )
    }

    pub(crate) fn up(&self) -> f32 {
        self.axis(
            [KeyCode::KeyE, KeyCode::KeyE],
            [KeyCode::KeyQ, KeyCode::KeyQ],
        )
    }

    /// The frame took the look deltas and action edges.
    pub(crate) fn consumed(&mut self) {
        self.yaw = 0.0;
        self.pitch = 0.0;
        self.pressed = 0;
        self.released = 0;
    }

    /// Focus left: nothing is held.
    pub(crate) fn clear(&mut self) {
        self.keys.clear();
        self.held = 0;
        self.dragging = false;
        self.cursor = None;
        self.consumed();
    }
}

/// The key the page's key name `name` names: the UI Events code names,
/// which winit's key codes take.
fn key_code(name: &str) -> Option<KeyCode> {
    KEY_CODES
        .iter()
        .find(|code| format!("{code:?}") == name)
        .copied()
}

/// The keys games bind actions to.
const KEY_CODES: &[KeyCode] = &[
    KeyCode::Space,
    KeyCode::Enter,
    KeyCode::Tab,
    KeyCode::Escape,
    KeyCode::Backspace,
    KeyCode::ShiftLeft,
    KeyCode::ShiftRight,
    KeyCode::ControlLeft,
    KeyCode::ControlRight,
    KeyCode::AltLeft,
    KeyCode::AltRight,
    KeyCode::KeyA,
    KeyCode::KeyB,
    KeyCode::KeyC,
    KeyCode::KeyD,
    KeyCode::KeyE,
    KeyCode::KeyF,
    KeyCode::KeyG,
    KeyCode::KeyH,
    KeyCode::KeyI,
    KeyCode::KeyJ,
    KeyCode::KeyK,
    KeyCode::KeyL,
    KeyCode::KeyM,
    KeyCode::KeyN,
    KeyCode::KeyO,
    KeyCode::KeyP,
    KeyCode::KeyQ,
    KeyCode::KeyR,
    KeyCode::KeyS,
    KeyCode::KeyT,
    KeyCode::KeyU,
    KeyCode::KeyV,
    KeyCode::KeyW,
    KeyCode::KeyX,
    KeyCode::KeyY,
    KeyCode::KeyZ,
    KeyCode::Digit0,
    KeyCode::Digit1,
    KeyCode::Digit2,
    KeyCode::Digit3,
    KeyCode::Digit4,
    KeyCode::Digit5,
    KeyCode::Digit6,
    KeyCode::Digit7,
    KeyCode::Digit8,
    KeyCode::Digit9,
    KeyCode::ArrowUp,
    KeyCode::ArrowDown,
    KeyCode::ArrowLeft,
    KeyCode::ArrowRight,
];
