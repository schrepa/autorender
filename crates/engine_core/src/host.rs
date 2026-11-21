use bevy_ecs::prelude::Resource;
use serde::Serialize;

use crate::clock::Interval;
use crate::input::Devices;

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum CursorMode {
    Drag = 0,
    #[default]
    Free = 1,
    Captured = 2,
}

impl CursorMode {
    pub const ALL: [(Self, &'static str); 3] = [
        (Self::Drag, "drag"),
        (Self::Free, "free"),
        (Self::Captured, "captured"),
    ];
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum InputModality {
    #[default]
    KeyboardMouse,
    Touch,
    Gamepad,
}

impl InputModality {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::KeyboardMouse => "kb_mouse",
            Self::Touch => "touch",
            Self::Gamepad => "gamepad",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct InputCapabilities {
    pub coarse_pointer: bool,
    pub can_hover: bool,
    pub can_lock_pointer: bool,
}

impl Default for InputCapabilities {
    fn default() -> Self {
        Self {
            coarse_pointer: false,
            can_hover: true,
            can_lock_pointer: true,
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct InputProfile {
    pub capabilities: InputCapabilities,
    pub active: InputModality,
    pub seen_input: bool,
}

impl InputProfile {
    pub fn apply_capabilities(&mut self, capabilities: InputCapabilities) {
        self.capabilities = capabilities;
        if !self.seen_input {
            self.active = if capabilities.coarse_pointer && !capabilities.can_hover {
                InputModality::Touch
            } else {
                InputModality::KeyboardMouse
            };
        }
    }

    pub fn observe(&mut self, devices: &Devices) {
        if !devices.touches.is_empty() {
            self.active = InputModality::Touch;
            self.seen_input = true;
            return;
        }
        let mouse_used = devices.mouse.buttons.any_pressed()
            || devices.mouse.buttons.any_held()
            || devices.mouse.motion_pixels.length_squared() > 0.0
            || devices.mouse.wheel_notches.length_squared() > 0.0;
        let key_used = devices.keyboard.any_pressed() || devices.keyboard.any_held();
        if key_used || mouse_used {
            self.active = InputModality::KeyboardMouse;
            self.seen_input = true;
            return;
        }
        let pad_used = devices.pads.iter().any(|pad| {
            pad.buttons.any_pressed()
                || pad.buttons.any_held()
                || pad.sticks[0].length_squared() > 0.0625
                || pad.sticks[1].length_squared() > 0.0625
                || pad.triggers[0] > 0.25
                || pad.triggers[1] > 0.25
        });
        if pad_used {
            self.active = InputModality::Gamepad;
            self.seen_input = true;
        }
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Host {
    pub focused: bool,
    pub hidden: bool,
    pub showing: bool,
}

impl Host {
    pub(crate) const fn shown(self) -> bool {
        self.showing && !self.hidden
    }
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub struct Present {
    pub vsync: bool,
    pub latency: u32,
}

#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Show {
    WhenReady,
    Now,
}

impl crate::Engine {
    pub fn report_host(&mut self, host: Host) {
        if !host.shown() {
            self.interval = Interval::Fresh;
        }
        let world = self.app.world_mut();
        if world.get_resource::<Host>() != Some(&host) {
            world.insert_resource(host);
        }
    }

    pub fn report_input_capabilities(&mut self, capabilities: InputCapabilities) {
        let world = self.app.world_mut();
        let mut profile = world
            .get_resource::<InputProfile>()
            .copied()
            .unwrap_or_default();
        let before = profile;
        profile.apply_capabilities(capabilities);
        if profile != before {
            world.insert_resource(profile);
        }
    }

    pub(crate) fn shown(&self) -> bool {
        self.app
            .world()
            .get_resource::<Host>()
            .is_some_and(|host| host.shown())
    }
}
