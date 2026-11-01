use bevy_ecs::prelude::Resource;

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
