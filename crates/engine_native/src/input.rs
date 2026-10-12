//! A window's keyboard and mouse events as the input every host gathers
//! (`engine::host::InputGatherer`): keys by their UI Events code names,
//! which winit's key codes take, and the deltas of dragging with the left
//! button.

use engine::{Game, GameHost};
use winit::event::{ElementState, MouseButton};
use winit::keyboard::KeyCode;

/// A window's drag: whether the left button is down, and where the cursor
/// last was.
#[derive(Default)]
pub(crate) struct Input {
    dragging: bool,
    cursor: Option<(f64, f64)>,
}

impl Input {
    /// Key `code` went down or up, a held key's repeats left out: to the
    /// input `host` gathers.
    pub(crate) fn key<G: Game>(host: &mut GameHost<G>, code: KeyCode, state: ElementState) {
        // winit names its key codes as UI Events does.
        host.key(&format!("{code:?}"), state == ElementState::Pressed);
    }

    pub(crate) fn button(&mut self, button: MouseButton, state: ElementState) {
        if button == MouseButton::Left {
            self.dragging = state == ElementState::Pressed;
        }
    }

    /// The cursor moved to `position`, in logical pixels: a drag turns
    /// `host`'s view.
    pub(crate) fn cursor<G: Game>(&mut self, host: &mut GameHost<G>, position: (f64, f64)) {
        if let Some((x, y)) = self.cursor
            && self.dragging
        {
            host.look((position.0 - x) as f32, (position.1 - y) as f32);
        }
        self.cursor = Some(position);
    }

    /// Focus left: no drag goes on.
    pub(crate) fn clear(&mut self) {
        self.dragging = false;
        self.cursor = None;
    }
}
