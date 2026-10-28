use bevy_ecs::prelude::Resource;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(u8);

impl ActionId {
    pub const JUMP: Self = Self(0);
    pub const COUNT: u32 = u32::BITS;

    pub const fn new(index: u8) -> Option<Self> {
        if (index as u32) < Self::COUNT {
            Some(Self(index))
        } else {
            None
        }
    }

    pub const fn mask(self) -> u32 {
        1_u32 << self.0
    }
}

#[derive(Resource, Debug, Default)]
pub struct ActionInput {
    held: u32,
    pending_pressed: u32,
    pending_released: u32,
    tick_pressed: u32,
    tick_released: u32,
}

impl ActionInput {
    pub fn ingest(&mut self, held: u32, pressed: u32, released: u32) {
        self.pending_pressed |= pressed | (held & !self.held);
        self.pending_released |= released | (self.held & !held);
        self.held = held;
    }

    pub fn pressed(&self, action: ActionId) -> bool {
        self.held & action.mask() != 0
    }
    pub fn just_pressed(&self, action: ActionId) -> bool {
        self.tick_pressed & action.mask() != 0
    }
    pub fn just_released(&self, action: ActionId) -> bool {
        self.tick_released & action.mask() != 0
    }

    pub fn clear(&mut self) {
        *self = Self::default();
    }

    pub(crate) fn begin_tick(&mut self) {
        self.tick_pressed = std::mem::take(&mut self.pending_pressed);
        self.tick_released = std::mem::take(&mut self.pending_released);
    }

    pub(crate) fn end_tick(&mut self) {
        self.tick_pressed = 0;
        self.tick_released = 0;
    }
}
