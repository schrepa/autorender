use std::collections::BTreeSet;

use glam::{Mat4, UVec2, Vec3};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameFacts {
    pub capture: bool,
    pub debug: bool,
    pub display_size: UVec2,
    pub render_size: UVec2,
    pub pixel_ratio: f32,
}

impl Default for FrameFacts {
    fn default() -> Self {
        Self {
            capture: false,
            debug: false,
            display_size: UVec2::new(1920, 1080),
            render_size: UVec2::new(1920, 1080),
            pixel_ratio: 1.0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ViewState {
    facts: FrameFacts,
    view_projection: Mat4,
    eye: Vec3,
    frozen: Option<Vec3>,
    scene: u64,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            facts: FrameFacts::default(),
            view_projection: Mat4::IDENTITY,
            eye: Vec3::ZERO,
            frozen: None,
            scene: 0,
        }
    }
}

impl ViewState {
    pub fn identified(scene: u64) -> Self {
        Self {
            scene,
            ..Self::default()
        }
    }

    pub fn facts(&self) -> FrameFacts {
        self.facts
    }

    pub fn render_size(&self) -> UVec2 {
        self.facts.render_size
    }

    pub fn display_size(&self) -> UVec2 {
        self.facts.display_size
    }

    pub fn scene(&self) -> u64 {
        self.scene
    }

    pub fn view_projection(&self) -> Mat4 {
        self.view_projection
    }

    pub fn eye(&self) -> Vec3 {
        self.eye
    }

    pub fn detail_eye(&self) -> Vec3 {
        self.frozen.unwrap_or(self.eye)
    }

    pub fn frozen(&self) -> Option<Vec3> {
        self.frozen
    }

    pub fn set(&mut self, facts: FrameFacts, view_projection: Mat4, eye: Vec3) {
        self.facts = facts;
        self.view_projection = view_projection;
        self.eye = eye;
    }

    pub fn freeze(&mut self, at: Option<Vec3>) {
        self.frozen = at;
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct Edges {
    pressed: BTreeSet<u32>,
    released: BTreeSet<u32>,
}

impl Edges {
    fn is_empty(&self) -> bool {
        self.pressed.is_empty() && self.released.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InputState {
    held: BTreeSet<u32>,
    consumer: u64,
    next_tick: Edges,
    next_frame: Edges,
    shown: Edges,
}

impl Default for InputState {
    fn default() -> Self {
        Self {
            held: BTreeSet::new(),
            consumer: 1,
            next_tick: Edges::default(),
            next_frame: Edges::default(),
            shown: Edges::default(),
        }
    }
}

impl InputState {
    pub fn ingest(
        &mut self,
        held: &BTreeSet<u32>,
        pressed: &BTreeSet<u32>,
        released: &BTreeSet<u32>,
    ) {
        let rose = held.difference(&self.held);
        let fell = self.held.difference(held);
        for edges in [&mut self.next_tick, &mut self.next_frame] {
            edges.pressed.extend(pressed.iter().chain(rose.clone()));
            edges.released.extend(released.iter().chain(fell.clone()));
        }
        self.held.clone_from(held);
    }

    pub fn pressed(&self, action: u32) -> bool {
        self.held.contains(&action)
    }

    pub fn held(&self) -> impl Iterator<Item = u32> {
        self.held.iter().copied()
    }

    pub fn just_pressed(&self, action: u32) -> bool {
        self.shown.pressed.contains(&action)
    }

    pub fn just_released(&self, action: u32) -> bool {
        self.shown.released.contains(&action)
    }

    pub fn pending_tick(&self) -> Option<u64> {
        (!self.next_tick.is_empty()).then_some(self.consumer)
    }

    pub fn drop_tick_edges(&mut self) {
        self.next_tick = Edges::default();
    }

    pub fn begin_tick(&mut self, tick: u64) {
        let pending = std::mem::take(&mut self.next_tick);
        self.shown = if tick == self.consumer {
            pending
        } else {
            Edges::default()
        };
        self.consumer = tick.saturating_add(1);
    }

    pub fn begin_frame(&mut self) {
        self.shown = std::mem::take(&mut self.next_frame);
    }

    pub fn hide_edges(&mut self) {
        self.shown = Edges::default();
    }
}
