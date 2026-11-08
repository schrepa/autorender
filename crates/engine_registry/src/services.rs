use std::collections::BTreeSet;

use glam::UVec2;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FrameFacts {
    pub display_size: UVec2,
    pub render_size: UVec2,
    pub pixel_ratio: f32,
}

impl Default for FrameFacts {
    fn default() -> Self {
        Self {
            display_size: UVec2::ZERO,
            render_size: UVec2::ZERO,
            pixel_ratio: 1.0,
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize, schemars::JsonSchema)]
struct Edges {
    pressed: BTreeSet<u32>,
    released: BTreeSet<u32>,
}

impl Edges {
    fn is_empty(&self) -> bool {
        self.pressed.is_empty() && self.released.is_empty()
    }
}

#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, schemars::JsonSchema)]
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
