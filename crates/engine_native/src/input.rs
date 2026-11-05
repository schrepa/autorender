use engine::host::policy::WHEEL_PIXELS;
use engine::{CursorMode, Game, GameHost};
use gilrs::{Axis, Button, Gamepad, Gilrs};
use winit::dpi::PhysicalPosition;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
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
    dragging: bool,
    cursor: Option<(f64, f64)>,
    grabbed: bool,
    pads: Option<Gilrs>,
}

impl Input {
    pub(crate) fn key<G: Game>(host: &mut GameHost<G>, code: KeyCode, state: ElementState) {
        host.key(&format!("{code:?}"), state == ElementState::Pressed);
    }

    pub(crate) fn button<G: Game>(
        &mut self,
        host: &mut GameHost<G>,
        button: MouseButton,
        state: ElementState,
    ) {
        let down = state == ElementState::Pressed;
        let code = match button {
            MouseButton::Left => {
                self.dragging = down && host.cursor_mode() == CursorMode::Drag;
                "MouseLeft"
            }
            MouseButton::Right => "MouseRight",
            MouseButton::Middle => "MouseMiddle",
            MouseButton::Back | MouseButton::Forward | MouseButton::Other(_) => return,
        };
        host.key(code, down);
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
        if let Some((x, y)) = self.cursor
            && self.dragging
        {
            host.look(
                ((position.0 - x) / scale) as f32,
                ((position.1 - y) / scale) as f32,
            );
        }
        self.cursor = Some(position);
        host.pointer(position.0 as f32, position.1 as f32);
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
        for (_, pad) in pads.gamepads() {
            let buttons = PAD_BUTTONS
                .iter()
                .filter(|&&(button, _)| pad.is_pressed(button))
                .fold(0_u32, |bits, &(_, bit)| bits | (1_u32 << bit));
            host.pad(buttons, pad_axes(&pad));
        }
    }

    pub(crate) fn apply_cursor<G: Game>(&mut self, host: &GameHost<G>, window: &Window) {
        let mode = host.cursor_mode();
        if mode != CursorMode::Drag {
            self.dragging = false;
        }
        let grabbed = mode == CursorMode::Captured && window.has_focus();
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
        self.dragging = false;
        self.cursor = None;
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
