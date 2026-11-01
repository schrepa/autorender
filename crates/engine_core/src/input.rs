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

    const fn mask(self) -> u32 {
        1_u32 << self.0
    }
}

#[derive(Debug, Default)]
struct ActionBits {
    held: u32,
    pending_pressed: u32,
    pending_released: u32,
    tick_pressed: u32,
    tick_released: u32,
}

#[derive(Resource, Default)]
#[component(immutable)]
pub struct ActionInput {
    state: ActionBits,
}

impl ActionInput {
    pub fn pressed(&self, action: ActionId) -> bool {
        self.state.held & action.mask() != 0
    }

    pub fn just_pressed(&self, action: ActionId) -> bool {
        self.state.tick_pressed & action.mask() != 0
    }

    pub fn just_released(&self, action: ActionId) -> bool {
        self.state.tick_released & action.mask() != 0
    }

    #[doc(hidden)]
    pub fn ingest(&mut self, held: u32, pressed: u32, released: u32) {
        let state = &mut self.state;
        state.pending_pressed |= pressed | (held & !state.held);
        state.pending_released |= released | (state.held & !held);
        state.held = held;
    }

    #[doc(hidden)]
    pub fn clear(&mut self) {
        self.state = ActionBits::default();
    }

    pub(crate) fn begin_tick(&mut self) {
        self.state.tick_pressed = std::mem::take(&mut self.state.pending_pressed);
        self.state.tick_released = std::mem::take(&mut self.state.pending_released);
    }

    pub(crate) fn end_tick(&mut self) {
        self.state.tick_pressed = 0;
        self.state.tick_released = 0;
    }
}
