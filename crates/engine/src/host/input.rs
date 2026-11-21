use std::collections::BTreeSet;

use engine_core::{Devices, KeyCode, MouseButton, Pad, PadButton, Touch, TouchPhase};
use glam::Vec2;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum InputSource {
    #[default]
    Player,
    Agent,
}

#[derive(Clone, Debug, Default)]
pub struct DeviceGatherer {
    devices: Devices,
    polled: BTreeSet<u32>,
    ending: Vec<Touch>,
}

impl DeviceGatherer {
    pub fn key(&mut self, key: KeyCode, down: bool) {
        self.devices.keyboard.set(key, down);
    }

    pub fn keys_held(&self) -> impl Iterator<Item = KeyCode> + '_ {
        (KeyCode::ALL.iter().copied()).filter(|&key| self.devices.keyboard.held(key))
    }

    pub fn mouse_button(&mut self, button: MouseButton, down: bool) {
        self.devices.mouse.buttons.set(button, down);
    }

    pub fn mouse_held(&self) -> impl Iterator<Item = MouseButton> + '_ {
        (MouseButton::ALL.into_iter()).filter(|&button| self.devices.mouse.buttons.held(button))
    }

    pub fn motion(&mut self, dx: f32, dy: f32) {
        self.devices.mouse.motion_pixels += Vec2::new(finite(dx), finite(dy));
    }

    pub fn pointer(&mut self, x: f32, y: f32) {
        self.devices.mouse.position_pixels = Some(Vec2::new(finite(x), finite(y)));
        self.devices.mouse.inside = true;
    }

    pub fn pointer_left(&mut self) {
        self.devices.mouse.inside = false;
    }

    pub fn wheel(&mut self, right_notches: f32, up_notches: f32) {
        self.devices.mouse.wheel_notches += Vec2::new(finite(right_notches), finite(up_notches));
    }

    pub fn pad(&mut self, index: u32, buttons: u32, [lx, ly, rx, ry, lt, rt]: [f32; 6]) {
        self.polled.insert(index);
        let at = match self.devices.pads.iter().position(|pad| pad.index == index) {
            Some(at) => at,
            None => {
                self.devices.pads.push(Pad {
                    index,
                    ..Pad::default()
                });
                self.devices.pads.len() - 1
            }
        };
        let pad = &mut self.devices.pads[at];
        for (button, bit) in PadButton::ALL {
            pad.buttons.set(button, buttons & (1 << bit) != 0);
        }
        pad.sticks = [
            Vec2::new(finite(lx), finite(ly)),
            Vec2::new(finite(rx), finite(ry)),
        ];
        pad.triggers = [finite(lt), finite(rt)];
    }

    pub fn touch(&mut self, id: u32, x: f32, y: f32, phase: TouchPhase) {
        let position_pixels = Vec2::new(finite(x), finite(y));
        if phase == TouchPhase::Began {
            self.ending.retain(|touch| touch.id != id);
        }
        let Some(touch) = self.devices.touches.iter_mut().find(|touch| touch.id == id) else {
            if phase == TouchPhase::Began {
                self.devices.touches.push(Touch {
                    id,
                    position_pixels,
                    motion_pixels: Vec2::ZERO,
                    phase,
                });
            }
            return;
        };
        touch.position_pixels = position_pixels;
        match (touch.phase, phase) {
            (TouchPhase::Began, TouchPhase::Ended | TouchPhase::Cancelled) => {
                self.ending.retain(|touch| touch.id != id);
                self.ending.push(Touch {
                    id,
                    position_pixels,
                    motion_pixels: Vec2::ZERO,
                    phase,
                });
            }
            (TouchPhase::Began, TouchPhase::Began) => {
                touch.motion_pixels = Vec2::ZERO;
            }
            (TouchPhase::Began, TouchPhase::Moved) => {}
            (TouchPhase::Moved | TouchPhase::Ended | TouchPhase::Cancelled, TouchPhase::Began) => {
                touch.phase = TouchPhase::Began;
                touch.motion_pixels = Vec2::ZERO;
            }
            (
                TouchPhase::Moved | TouchPhase::Ended | TouchPhase::Cancelled,
                TouchPhase::Moved | TouchPhase::Ended | TouchPhase::Cancelled,
            ) => {
                touch.phase = phase;
            }
        }
    }

    pub fn touch_motion(&mut self, id: u32, dx: f32, dy: f32) {
        if let Some(touch) = self.devices.touches.iter_mut().find(|touch| touch.id == id) {
            touch.motion_pixels += Vec2::new(finite(dx), finite(dy));
        }
    }

    pub fn clear(&mut self) {
        self.devices.keyboard.release_all();
        self.devices.mouse.buttons.release_all();
        self.devices.mouse.position_pixels = None;
        self.devices.mouse.inside = false;
        self.polled.clear();
        self.release_unpolled();
        for touch in &mut self.devices.touches {
            touch.phase = TouchPhase::Cancelled;
        }
        self.ending.clear();
        self.drop_edges();
    }

    pub fn drop_edges(&mut self) {
        self.devices = self.devices.held();
        self.devices
            .pads
            .retain(|pad| self.polled.contains(&pad.index));
        self.polled.clear();
        self.end_touches();
    }

    pub fn take(&mut self) -> Devices {
        self.release_unpolled();
        let taken = self.devices.clone();
        self.drop_edges();
        taken
    }

    fn release_unpolled(&mut self) {
        for pad in &mut self.devices.pads {
            if self.polled.contains(&pad.index) {
                continue;
            }
            pad.buttons.release_all();
            pad.sticks = [Vec2::ZERO; 2];
            pad.triggers = [0.0; 2];
        }
    }

    fn end_touches(&mut self) {
        for ended in std::mem::take(&mut self.ending) {
            if let Some(touch) = self
                .devices
                .touches
                .iter_mut()
                .find(|touch| touch.id == ended.id)
            {
                *touch = ended;
            }
        }
    }
}

fn finite(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}
