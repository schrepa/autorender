use engine::{Game, GameHost};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

#[derive(Default)]
pub(crate) struct Input {
    dragging: bool,
    cursor: Option<(f64, f64)>,
}

impl Input {
    pub(crate) fn key<G: Game>(host: &mut GameHost<G>, code: KeyCode, state: ElementState) {
        host.key(&format!("{code:?}"), state == ElementState::Pressed);
    }

    pub(crate) fn button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.dragging = state == ElementState::Pressed;
        }
    }

    pub(crate) fn cursor<G: Game>(&mut self, host: &mut GameHost<G>, position: (f64, f64)) {
        if let Some((x, y)) = self.cursor
            && self.dragging
        {
            host.look((position.0 - x) as f32, (position.1 - y) as f32);
        }
        self.cursor = Some(position);
    }

    pub(crate) fn clear(&mut self) {
        self.dragging = false;
        self.cursor = None;
    }
}
