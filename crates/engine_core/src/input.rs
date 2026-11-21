use std::marker::PhantomData;

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

impl From<MouseButton> for u8 {
    fn from(button: MouseButton) -> u8 {
        button as u8
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

impl From<PadButton> for u8 {
    fn from(button: PadButton) -> u8 {
        button as u8
    }
}

macro_rules! key_codes {
    ($($key:ident)*) => {
        #[derive(
            Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize, schemars::JsonSchema,
        )]
        #[repr(u8)]
        pub enum KeyCode {
            $($key,)*
        }

        impl KeyCode {
            pub const ALL: &'static [KeyCode] = &[$(KeyCode::$key,)*];

            pub fn name(self) -> &'static str {
                match self {
                    $(KeyCode::$key => stringify!($key),)*
                }
            }

            pub fn named(name: &str) -> Option<KeyCode> {
                Self::ALL.iter().copied().find(|key| key.name() == name)
            }
        }
    };
}

key_codes! {
    Backquote Backslash BracketLeft BracketRight Comma Digit0 Digit1 Digit2 Digit3 Digit4 Digit5
    Digit6 Digit7 Digit8 Digit9 Equal IntlBackslash IntlRo IntlYen KeyA KeyB KeyC KeyD KeyE KeyF
    KeyG KeyH KeyI KeyJ KeyK KeyL KeyM KeyN KeyO KeyP KeyQ KeyR KeyS KeyT KeyU KeyV KeyW KeyX KeyY
    KeyZ Minus Period Quote Semicolon Slash AltLeft AltRight Backspace CapsLock ContextMenu
    ControlLeft ControlRight Enter MetaLeft MetaRight ShiftLeft ShiftRight Space Tab Convert
    KanaMode Lang1 Lang2 Lang3 Lang4 Lang5 NonConvert Delete End Help Home Insert PageDown PageUp
    ArrowDown ArrowLeft ArrowRight ArrowUp NumLock Numpad0 Numpad1 Numpad2 Numpad3 Numpad4 Numpad5
    Numpad6 Numpad7 Numpad8 Numpad9 NumpadAdd NumpadBackspace NumpadClear NumpadClearEntry
    NumpadComma NumpadDecimal NumpadDivide NumpadEnter NumpadEqual NumpadHash NumpadMemoryAdd
    NumpadMemoryClear NumpadMemoryRecall NumpadMemoryStore NumpadMemorySubtract NumpadMultiply
    NumpadParenLeft NumpadParenRight NumpadStar NumpadSubtract Escape F1 F2 F3 F4 F5 F6 F7 F8 F9
    F10 F11 F12 F13 F14 F15 F16 F17 F18 F19 F20 F21 F22 F23 F24 Fn FnLock PrintScreen ScrollLock
    Pause BrowserBack BrowserFavorites BrowserForward BrowserHome BrowserRefresh BrowserSearch
    BrowserStop Eject LaunchApp1 LaunchApp2 LaunchMail MediaPlayPause MediaSelect MediaStop
    MediaTrackNext MediaTrackPrevious Power Sleep AudioVolumeDown AudioVolumeMute AudioVolumeUp
    WakeUp Hyper Super Turbo Abort Resume Suspend Again Copy Cut Find Open Paste Props Select Undo
    Hiragana Katakana
}

impl From<KeyCode> for u8 {
    fn from(key: KeyCode) -> u8 {
        key as u8
    }
}

const KEY_WORDS: usize = KeyCode::ALL.len().div_ceil(64);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buttons<B, const WORDS: usize = 1> {
    held: [u64; WORDS],
    pressed: [u64; WORDS],
    released: [u64; WORDS],
    kind: PhantomData<B>,
}

impl<B, const WORDS: usize> Default for Buttons<B, WORDS> {
    fn default() -> Self {
        Self {
            held: [0; WORDS],
            pressed: [0; WORDS],
            released: [0; WORDS],
            kind: PhantomData,
        }
    }
}

fn spot(button: impl Into<u8>) -> (usize, u64) {
    let index = usize::from(button.into());
    (index / 64, 1 << (index % 64))
}

impl<B: Copy + Into<u8>, const WORDS: usize> Buttons<B, WORDS> {
    pub fn set(&mut self, button: B, down: bool) {
        let (word, bit) = spot(button);
        let was = self.held[word] & bit != 0;
        if down && !was {
            self.held[word] |= bit;
            self.pressed[word] |= bit;
        } else if !down && was {
            self.held[word] &= !bit;
            self.released[word] |= bit;
        }
    }

    pub fn held(&self, button: B) -> bool {
        let (word, bit) = spot(button);
        self.held[word] & bit != 0
    }

    pub fn pressed(&self, button: B) -> bool {
        let (word, bit) = spot(button);
        self.pressed[word] & bit != 0
    }

    pub fn released(&self, button: B) -> bool {
        let (word, bit) = spot(button);
        self.released[word] & bit != 0
    }

    pub fn release_all(&mut self) {
        for (released, held) in self.released.iter_mut().zip(&mut self.held) {
            *released |= std::mem::take(held);
        }
    }

    pub fn any_held(&self) -> bool {
        self.held.iter().any(|&word| word != 0)
    }

    pub fn any_pressed(&self) -> bool {
        self.pressed.iter().any(|&word| word != 0)
    }
}

impl<B, const WORDS: usize> Buttons<B, WORDS> {
    fn drop_edges(&mut self) {
        self.pressed = [0; WORDS];
        self.released = [0; WORDS];
    }

    fn carry(&mut self, older: &Self) {
        for word in 0..WORDS {
            self.pressed[word] |= older.pressed[word];
            self.released[word] |= older.released[word];
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Mouse {
    pub buttons: Buttons<MouseButton>,
    pub position_pixels: Option<Vec2>,
    pub inside: bool,
    pub motion_pixels: Vec2,
    pub wheel_notches: Vec2,
}

#[derive(Clone, Debug, Default, PartialEq)]
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

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Touch {
    pub id: u32,
    pub position_pixels: Vec2,
    pub motion_pixels: Vec2,
    pub phase: TouchPhase,
}

#[derive(Resource, Clone, Debug, Default, PartialEq)]
#[component(immutable)]
pub struct Devices {
    pub keyboard: Buttons<KeyCode, KEY_WORDS>,
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum PointerKind {
    #[default]
    Fine,
    Coarse,
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq)]
pub struct Pointer {
    pub position_pixels: Option<Vec2>,
    pub delta_pixels: Vec2,
    pub down: bool,
    pub pressed: bool,
    pub released: bool,
    pub cancelled: bool,
    pub hovering: bool,
    pub kind: PointerKind,
    primary_touch: Option<u32>,
}

impl Pointer {
    pub const fn is_coarse(self) -> bool {
        matches!(self.kind, PointerKind::Coarse)
    }

    pub fn update(&mut self, devices: &Devices) {
        let prev = self.position_pixels;
        if let Some(id) = self.primary_touch {
            if let Some(touch) = devices.touches.iter().find(|touch| touch.id == id) {
                self.apply_touch(*touch, prev);
                return;
            }
            self.primary_touch = None;
            self.position_pixels = None;
            self.delta_pixels = Vec2::ZERO;
            self.down = false;
            self.pressed = false;
            self.released = false;
            self.cancelled = true;
            self.hovering = false;
            self.kind = PointerKind::Coarse;
            return;
        }
        if let Some(touch) = devices
            .touches
            .iter()
            .find(|touch| touch.phase == TouchPhase::Began)
        {
            self.primary_touch = Some(touch.id);
            self.apply_touch(*touch, None);
            return;
        }
        if !devices.touches.is_empty() {
            self.position_pixels = None;
            self.delta_pixels = Vec2::ZERO;
            self.down = false;
            self.pressed = false;
            self.released = false;
            self.cancelled = false;
            self.hovering = false;
            self.kind = PointerKind::Coarse;
            return;
        }
        let left = &devices.mouse.buttons;
        let down = left.held(MouseButton::Left);
        let pressed = left.pressed(MouseButton::Left);
        let released = left.released(MouseButton::Left);
        let mouse_present = devices.mouse.inside || down || pressed || released;
        if mouse_present {
            let pos = devices.mouse.position_pixels;
            self.delta_pixels = match (pos, prev) {
                (Some(now), Some(before)) if self.kind == PointerKind::Fine => now - before,
                _ => Vec2::ZERO,
            };
            self.position_pixels = pos;
            self.down = down;
            self.pressed = pressed;
            self.released = released;
            self.cancelled = false;
            self.hovering = devices.mouse.inside;
            self.kind = PointerKind::Fine;
        } else {
            self.position_pixels = None;
            self.delta_pixels = Vec2::ZERO;
            self.down = false;
            self.pressed = false;
            self.released = false;
            self.cancelled = false;
            self.hovering = false;
        }
    }

    fn apply_touch(&mut self, touch: Touch, prev: Option<Vec2>) {
        self.kind = PointerKind::Coarse;
        self.hovering = false;
        match touch.phase {
            TouchPhase::Began => {
                self.position_pixels = Some(touch.position_pixels);
                self.delta_pixels = Vec2::ZERO;
                self.down = true;
                self.pressed = true;
                self.released = false;
                self.cancelled = false;
            }
            TouchPhase::Moved => {
                self.delta_pixels = prev.map_or(Vec2::ZERO, |before| touch.position_pixels - before);
                self.position_pixels = Some(touch.position_pixels);
                self.down = true;
                self.pressed = false;
                self.released = false;
                self.cancelled = false;
            }
            TouchPhase::Ended => {
                self.delta_pixels = prev.map_or(Vec2::ZERO, |before| touch.position_pixels - before);
                self.position_pixels = Some(touch.position_pixels);
                self.down = false;
                self.pressed = false;
                self.released = true;
                self.cancelled = false;
                self.primary_touch = None;
            }
            TouchPhase::Cancelled => {
                self.position_pixels = None;
                self.delta_pixels = Vec2::ZERO;
                self.down = false;
                self.pressed = false;
                self.released = false;
                self.cancelled = true;
                self.primary_touch = None;
            }
        }
    }

    pub fn drop_edges(&mut self) {
        self.delta_pixels = Vec2::ZERO;
        self.pressed = false;
        self.released = false;
        self.cancelled = false;
        if self.kind == PointerKind::Coarse && !self.down {
            self.position_pixels = None;
        }
    }
}

impl crate::Engine {
    pub fn accept_input(&mut self, devices: Devices) {
        let mut ticked = devices.clone();
        ticked.carry(&self.ticked);
        self.ticked = ticked;
        let world = self.app.world_mut();
        let mut profile = world
            .get_resource::<crate::host::InputProfile>()
            .copied()
            .unwrap_or_default();
        let prev_profile = profile;
        profile.observe(&devices);
        if profile != prev_profile {
            world.insert_resource(profile);
        }
        let mut pointer = world
            .get_resource::<Pointer>()
            .copied()
            .unwrap_or_default();
        pointer.update(&devices);
        world.insert_resource(pointer);
        world.insert_resource(devices);
    }

    pub fn drop_marks(&mut self) {
        self.drop_tick_marks();
        let world = self.app.world_mut();
        let held = world.resource::<Devices>().held();
        world.insert_resource(held);
        if let Some(mut pointer) = world.get_resource_mut::<Pointer>() {
            pointer.drop_edges();
        }
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
