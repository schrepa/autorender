use engine_core::{FrameFacts, InputModality, InputProfile, ViewportClass};

use super::{Seg, Slot, Val};

const ENV_BASE: Slot = Slot::MAX - 31;

const VIEWPORT_COMPACT: Slot = ENV_BASE;
const VIEWPORT_MEDIUM: Slot = ENV_BASE + 1;
const VIEWPORT_WIDE: Slot = ENV_BASE + 2;
const VIEWPORT_PORTRAIT: Slot = ENV_BASE + 3;
const VIEWPORT_LANDSCAPE: Slot = ENV_BASE + 4;
const VIEWPORT_CLASS: Slot = ENV_BASE + 5;
const INPUT_TOUCH: Slot = ENV_BASE + 6;
const INPUT_KB_MOUSE: Slot = ENV_BASE + 7;
const INPUT_GAMEPAD: Slot = ENV_BASE + 8;
const INPUT_COARSE: Slot = ENV_BASE + 9;
const INPUT_CAN_HOVER: Slot = ENV_BASE + 10;
const INPUT_CAN_LOCK_POINTER: Slot = ENV_BASE + 11;
const INPUT_MODE: Slot = ENV_BASE + 12;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct EnvFacts {
    pub viewport_class: ViewportClass,
    pub portrait: bool,
    pub modality: InputModality,
    pub coarse_pointer: bool,
    pub can_hover: bool,
    pub can_lock_pointer: bool,
}

impl EnvFacts {
    pub(super) fn of(frame: FrameFacts, profile: Option<InputProfile>) -> Self {
        let profile = profile.unwrap_or_default();
        Self {
            viewport_class: frame.viewport_class(),
            portrait: frame.portrait(),
            modality: profile.active,
            coarse_pointer: profile.capabilities.coarse_pointer,
            can_hover: profile.capabilities.can_hover,
            can_lock_pointer: profile.capabilities.can_lock_pointer,
        }
    }

    pub(super) fn read(self, slot: Slot) -> Option<Val<'static>> {
        match slot {
            VIEWPORT_COMPACT => Some(Val::Bool(self.viewport_class == ViewportClass::Compact)),
            VIEWPORT_MEDIUM => Some(Val::Bool(self.viewport_class == ViewportClass::Medium)),
            VIEWPORT_WIDE => Some(Val::Bool(self.viewport_class == ViewportClass::Wide)),
            VIEWPORT_PORTRAIT => Some(Val::Bool(self.portrait)),
            VIEWPORT_LANDSCAPE => Some(Val::Bool(!self.portrait)),
            VIEWPORT_CLASS => Some(Val::Text(self.viewport_class.as_str())),
            INPUT_TOUCH => Some(Val::Bool(self.modality == InputModality::Touch)),
            INPUT_KB_MOUSE => Some(Val::Bool(self.modality == InputModality::KeyboardMouse)),
            INPUT_GAMEPAD => Some(Val::Bool(self.modality == InputModality::Gamepad)),
            INPUT_COARSE => Some(Val::Bool(self.coarse_pointer)),
            INPUT_CAN_HOVER => Some(Val::Bool(self.can_hover)),
            INPUT_CAN_LOCK_POINTER => Some(Val::Bool(self.can_lock_pointer)),
            INPUT_MODE => Some(Val::Text(self.modality.as_str())),
            _ => None,
        }
    }
}

pub(super) fn slot(path: &[Seg]) -> Option<Slot> {
    let [Seg::Key(root), Seg::Key(field)] = path else {
        return None;
    };
    match (root.as_str(), field.as_str()) {
        ("viewport", "compact") => Some(VIEWPORT_COMPACT),
        ("viewport", "medium") => Some(VIEWPORT_MEDIUM),
        ("viewport", "wide") => Some(VIEWPORT_WIDE),
        ("viewport", "portrait") => Some(VIEWPORT_PORTRAIT),
        ("viewport", "landscape") => Some(VIEWPORT_LANDSCAPE),
        ("viewport", "class") => Some(VIEWPORT_CLASS),
        ("input", "touch") => Some(INPUT_TOUCH),
        ("input", "kb_mouse") => Some(INPUT_KB_MOUSE),
        ("input", "gamepad") => Some(INPUT_GAMEPAD),
        ("input", "coarse") => Some(INPUT_COARSE),
        ("input", "can_hover") => Some(INPUT_CAN_HOVER),
        ("input", "can_lock_pointer") => Some(INPUT_CAN_LOCK_POINTER),
        ("input", "mode") => Some(INPUT_MODE),
        _ => None,
    }
}
