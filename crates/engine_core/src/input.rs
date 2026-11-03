use bevy_ecs::prelude::Resource;
use glam::Vec2;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(u8);

impl ActionId {
    pub const COUNT: u32 = u32::BITS;

    pub const fn new(index: u8) -> Option<Self> {
        if (index as u32) < Self::COUNT {
            Some(Self(index))
        } else {
            None
        }
    }
}

#[derive(Resource, Default)]
#[component(immutable)]
pub struct ActionInput {
    state: engine_registry::InputState,
}

impl ActionInput {
    pub fn pressed(&self, action: ActionId) -> bool {
        self.state.pressed(action.0)
    }

    pub fn just_pressed(&self, action: ActionId) -> bool {
        self.state.just_pressed(action.0)
    }

    pub fn just_released(&self, action: ActionId) -> bool {
        self.state.just_released(action.0)
    }

    #[doc(hidden)]
    pub fn ingest(&mut self, held: u32, pressed: u32, released: u32) {
        self.state.ingest(held, pressed, released);
    }

    #[doc(hidden)]
    pub fn clear(&mut self) {
        self.state.clear();
    }

    pub(crate) fn begin_tick(&mut self) {
        self.state.begin_tick();
    }

    pub(crate) fn end_tick(&mut self) {
        self.state.end_tick();
    }
}

#[derive(Resource, Debug, Clone, Copy, Default)]
#[component(immutable)]
pub struct MoveInput {
    pub move_forward: f32,
    pub move_right: f32,
    pub move_up: f32,
    pub look_yaw: f32,
    pub look_pitch: f32,
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

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
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

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq)]
#[component(immutable)]
pub struct PointerInput {
    pub position: Option<Vec2>,
    pub delta: Vec2,
    pub wheel: Vec2,
    pub inside: bool,
    pub held: bool,
    pub pressed: bool,
    pub released: bool,
}

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MenuInput {
    pub up: bool,
    pub down: bool,
    pub left: bool,
    pub right: bool,
    pub accept: bool,
    pub back: bool,
    pub wanted: bool,
}

#[derive(Resource, Debug, Clone, Copy, Default, PartialEq, Eq)]
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
