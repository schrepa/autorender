use std::collections::BTreeSet;

use engine_core::{Devices, MouseButton, Pad, PadButton, Touch, TouchPhase};
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
    pub fn key(&mut self, code: &str, down: bool) {
        self.devices.keyboard.set(code.to_owned(), down);
    }

    pub fn keys_held(&self) -> impl Iterator<Item = &str> + '_ {
        self.devices.keyboard.held.iter().map(String::as_str)
    }

    pub fn mouse_button(&mut self, button: MouseButton, down: bool) {
        self.devices.mouse.buttons.set(button, down);
    }

    pub fn mouse_held(&self) -> impl Iterator<Item = MouseButton> + '_ {
        self.devices.mouse.buttons.held.iter().copied()
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
        let ends = matches!(phase, TouchPhase::Ended | TouchPhase::Cancelled);
        let Some(touch) = self.devices.touches.iter_mut().find(|touch| touch.id == id) else {
            if phase == TouchPhase::Began {
                self.devices.touches.push(Touch {
                    id,
                    position_pixels,
                    phase,
                });
            }
            return;
        };
        touch.position_pixels = position_pixels;
        match (touch.phase, ends) {
            (TouchPhase::Began, true) => self.ending.push(Touch {
                id,
                position_pixels,
                phase,
            }),
            (TouchPhase::Began, false) => {}
            (TouchPhase::Moved | TouchPhase::Ended | TouchPhase::Cancelled, _) => {
                touch.phase = phase;
            }
        }
    }

    pub fn clear(&mut self) {
        let keys: Vec<String> = self.devices.keyboard.held.iter().cloned().collect();
        for code in keys {
            self.devices.keyboard.set(code, false);
        }
        for button in MouseButton::ALL {
            self.devices.mouse.buttons.set(button, false);
        }
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
        self.devices.pads.retain(|pad| self.polled.contains(&pad.index));
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
            for (button, _) in PadButton::ALL {
                pad.buttons.set(button, false);
            }
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
