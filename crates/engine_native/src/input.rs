use engine::host::policy::WHEEL_PIXELS;
use engine::{CursorMode, Game, GameHost, MouseButton, TouchPhase};
use gilrs::{Axis, Button, Gamepad, Gilrs};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseScrollDelta, Touch};
use winit::keyboard::KeyCode;
use winit::window::{CursorGrabMode, Window};

const PAD_BUTTONS: [(Button, u32); 14] = [
    (Button::South, 0),
    (Button::East, 1),
    (Button::West, 2),
    (Button::North, 3),
    (Button::LeftTrigger, 4),
    (Button::RightTrigger, 5),
    (Button::Select, 8),
    (Button::Start, 9),
    (Button::LeftThumb, 10),
    (Button::RightThumb, 11),
    (Button::DPadUp, 12),
    (Button::DPadDown, 13),
    (Button::DPadLeft, 14),
    (Button::DPadRight, 15),
];

#[derive(Default)]
pub(crate) struct Input {
    cursor: Option<(f64, f64)>,
    touches: Vec<(u32, (f64, f64))>,
    grabbed: bool,
    pads: Option<Gilrs>,
}

impl Input {
    pub(crate) fn key<G: Game>(host: &mut GameHost<G>, code: KeyCode, state: ElementState) {
        host.key(&format!("{code:?}"), state == ElementState::Pressed);
    }

    pub(crate) fn button<G: Game>(
        host: &mut GameHost<G>,
        button: winit::event::MouseButton,
        state: ElementState,
    ) {
        let button = match button {
            winit::event::MouseButton::Left => MouseButton::Left,
            winit::event::MouseButton::Middle => MouseButton::Middle,
            winit::event::MouseButton::Right => MouseButton::Right,
            winit::event::MouseButton::Back => MouseButton::Back,
            winit::event::MouseButton::Forward => MouseButton::Forward,
            winit::event::MouseButton::Other(_) => return,
        };
        host.mouse_button(button, state == ElementState::Pressed);
    }

    pub(crate) fn cursor<G: Game>(
        &mut self,
        host: &mut GameHost<G>,
        position: (f64, f64),
        scale: f64,
    ) {
        if host.cursor_mode() == CursorMode::Captured {
            self.cursor = None;
            return;
        }
        if let Some((x, y)) = self.cursor {
            host.motion(
                ((position.0 - x) / scale) as f32,
                ((position.1 - y) / scale) as f32,
            );
        }
        self.cursor = Some(position);
        host.pointer(position.0 as f32, position.1 as f32);
    }

    pub(crate) fn left<G: Game>(&mut self, host: &mut GameHost<G>) {
        self.cursor = None;
        host.pointer_left();
    }

    pub(crate) fn wheel<G: Game>(host: &mut GameHost<G>, delta: MouseScrollDelta, scale: f64) {
        let (x, y) = match delta {
            MouseScrollDelta::LineDelta(x, y) => (x, y),
            MouseScrollDelta::PixelDelta(pixels) => {
                let pixels = pixels.to_logical::<f32>(scale);
                (pixels.x / WHEEL_PIXELS, pixels.y / WHEEL_PIXELS)
            }
        };
        host.wheel(x, y);
    }

    pub(crate) fn motion<G: Game>(host: &mut GameHost<G>, (dx, dy): (f64, f64)) {
        if host.cursor_mode() == CursorMode::Captured {
            host.motion(dx as f32, dy as f32);
        }
    }

    pub(crate) fn touch<G: Game>(&mut self, host: &mut GameHost<G>, touch: Touch, scale: f64) {
        let phase = match touch.phase {
            winit::event::TouchPhase::Started => TouchPhase::Began,
            winit::event::TouchPhase::Moved => TouchPhase::Moved,
            winit::event::TouchPhase::Ended => TouchPhase::Ended,
            winit::event::TouchPhase::Cancelled => TouchPhase::Cancelled,
        };
        let id = touch.id as u32;
        let (x, y) = (touch.location.x, touch.location.y);
        host.touch(id, x as f32, y as f32, phase);
        let known = self.touches.iter().position(|&(held, _)| held == id);
        match (phase, known) {
            (TouchPhase::Began, None) => self.touches.push((id, (x, y))),
            (TouchPhase::Began, Some(at)) => self.touches[at].1 = (x, y),
            (TouchPhase::Moved, Some(at)) => {
                let (before_x, before_y) = self.touches[at].1;
                host.touch_motion(
                    id,
                    ((x - before_x) / scale) as f32,
                    ((y - before_y) / scale) as f32,
                );
                self.touches[at].1 = (x, y);
            }
            (TouchPhase::Ended | TouchPhase::Cancelled, Some(at)) => {
                self.touches.swap_remove(at);
            }
            (TouchPhase::Moved | TouchPhase::Ended | TouchPhase::Cancelled, None) => {}
        }
    }

    pub(crate) fn connect_pads(&mut self) {
        self.pads = Gilrs::new().ok();
    }

    pub(crate) fn poll_pads<G: Game>(&mut self, host: &mut GameHost<G>, window: &Window) {
        let Some(pads) = &mut self.pads else {
            return;
        };
        while pads.next_event().is_some() {}
        if !window.has_focus() {
            return;
        }
        for (id, pad) in pads.gamepads() {
            let buttons = PAD_BUTTONS
                .iter()
                .filter(|&&(button, _)| pad.is_pressed(button))
                .fold(0_u32, |bits, &(_, bit)| bits | (1_u32 << bit));
            host.pad(usize::from(id) as u32, buttons, pad_axes(&pad));
        }
    }

    pub(crate) fn apply_cursor<G: Game>(&mut self, host: &GameHost<G>, window: &Window) {
        let grabbed = host.cursor_mode() == CursorMode::Captured && window.has_focus();
        if grabbed == self.grabbed {
            return;
        }
        self.grabbed = grabbed;
        if grabbed {
            let size = window.inner_size();
            let middle = PhysicalPosition::new(size.width / 2, size.height / 2);
            let _ = window.set_cursor_position(middle);
            if window.set_cursor_grab(CursorGrabMode::Locked).is_err() {
                let _ = window.set_cursor_grab(CursorGrabMode::Confined);
            }
        } else {
            let _ = window.set_cursor_grab(CursorGrabMode::None);
        }
        window.set_cursor_visible(!grabbed);
    }

    pub(crate) fn regrab(&mut self) {
        self.grabbed = false;
    }

    pub(crate) fn clear(&mut self) {
        self.cursor = None;
        self.touches.clear();
    }
}

fn pad_axes(pad: &Gamepad<'_>) -> [f32; 6] {
    let trigger = |button: Button| pad.button_data(button).map_or(0.0, |data| data.value());
    [
        pad.value(Axis::LeftStickX),
        pad.value(Axis::LeftStickY),
        pad.value(Axis::RightStickX),
        pad.value(Axis::RightStickY),
        trigger(Button::LeftTrigger2),
        trigger(Button::RightTrigger2),
    ]
}
