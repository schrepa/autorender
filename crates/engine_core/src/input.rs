use std::collections::BTreeSet;

use bevy_ecs::prelude::Resource;
use glam::Vec2;
use serde::Serialize;

use crate::{Interest, List, Requests, Ticket};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(u32);

impl ActionId {
    pub const fn new(index: u32) -> Self {
        Self(index)
    }
}

#[derive(Resource, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(transparent)]
#[component(immutable)]
pub struct ActionInput {
    state: engine_registry::InputState,
}

impl ActionInput {
    pub fn pressed(&self, action: ActionId) -> bool {
        self.state.pressed(action.0)
    }

    pub fn held(&self) -> impl Iterator<Item = u32> {
        self.state.held()
    }

    pub fn just_pressed(&self, action: ActionId) -> bool {
        self.state.just_pressed(action.0)
    }

    pub fn just_released(&self, action: ActionId) -> bool {
        self.state.just_released(action.0)
    }

    #[doc(hidden)]
    pub fn ingest(
        &mut self,
        held: &BTreeSet<u32>,
        pressed: &BTreeSet<u32>,
        released: &BTreeSet<u32>,
    ) {
        self.state.ingest(held, pressed, released);
    }

    pub fn pending_tick(&self) -> Option<u64> {
        self.state.pending_tick()
    }

    pub(crate) fn drop_tick_edges(&mut self) {
        self.state.drop_tick_edges();
    }

    pub(crate) fn begin_tick(&mut self, tick: u64) {
        self.state.begin_tick(tick);
    }

    pub(crate) fn begin_frame(&mut self) {
        self.state.begin_frame();
    }

    pub(crate) fn hide_edges(&mut self) {
        self.state.hide_edges();
    }
}

#[derive(Resource, Debug, Clone, Copy, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct MoveInput {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub look_right_pixels: f32,
    pub look_down_pixels: f32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct AxisId(u8);

impl AxisId {
    pub const COUNT: u32 = 16;

    pub const fn new(index: u8) -> Option<Self> {
        if (index as u32) < Self::COUNT {
            Some(Self(index))
        } else {
            None
        }
    }
}

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct AxisInput {
    values: [f32; AxisId::COUNT as usize],
}

impl AxisInput {
    pub fn value(&self, axis: AxisId) -> f32 {
        self.values[usize::from(axis.0)]
    }

    pub fn values(&self) -> &[f32; AxisId::COUNT as usize] {
        &self.values
    }

    #[doc(hidden)]
    pub fn of(values: [f32; AxisId::COUNT as usize]) -> Self {
        Self { values }
    }
}

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[component(immutable)]
pub struct PointerInput {
    #[cfg_attr(feature = "schema", schemars(with = "Option<[f32; 2]>"))]
    pub position_pixels: Option<Vec2>,
    #[cfg_attr(feature = "schema", schemars(with = "[f32; 2]"))]
    pub delta_pixels: Vec2,
    pub wheel_up_notches: f32,
    pub wheel_right_notches: f32,
    pub inside: bool,
    pub held: bool,
    pub pressed: bool,
    pub released: bool,
}

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct MenuInput {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub accept: bool,
    pub back: bool,
    pub wanted: bool,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
#[serde(rename_all = "snake_case")]
#[repr(u32)]
pub enum CursorMode {
    #[default]
    Drag = 0,
    Free = 1,
    Captured = 2,
}

impl CursorMode {
    pub const ALL: [(Self, &'static str); 3] = [
        (Self::Drag, "drag"),
        (Self::Free, "free"),
        (Self::Captured, "captured"),
    ];

    pub const fn name(self) -> &'static str {
        Self::ALL[self as usize].1
    }
}

pub type CursorAsk = (&'static str, CursorMode);

#[derive(Resource, Debug, Default, Serialize)]
#[cfg_attr(feature = "schema", derive(schemars::JsonSchema))]
pub struct CursorRequests {
    #[serde(skip)]
    requests: Requests<CursorAsk, ()>,
    applied: CursorMode,
}

impl List for CursorRequests {
    const NAME: &'static str = "cursor";
    type Member = CursorAsk;
}

impl CursorRequests {
    pub fn ask(&mut self, by: &'static str, mode: CursorMode) -> Interest<CursorAsk> {
        self.requests.ask((by, mode))
    }

    pub fn wanted(&self) -> CursorMode {
        self.requests
            .asked()
            .max_by_key(|&(_, ticket)| ticket)
            .map_or_else(CursorMode::default, |(&(_, mode), _)| mode)
    }

    pub fn asked(&self) -> Vec<CursorAsk> {
        let mut asked: Vec<(Ticket, CursorAsk)> = self
            .requests
            .asked()
            .map(|(&ask, ticket)| (ticket, ask))
            .collect();
        asked.sort_unstable_by_key(|&(ticket, _)| ticket);
        asked.into_iter().map(|(_, ask)| ask).collect()
    }

    pub fn applied(&self) -> CursorMode {
        self.applied
    }

    #[doc(hidden)]
    pub fn apply(&mut self) {
        self.requests.settle(0.0);
        self.applied = self.wanted();
    }
}
