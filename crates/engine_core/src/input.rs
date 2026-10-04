//! Bounded action input. Host transitions survive frames with no fixed tick.
use bevy_ecs::prelude::Resource;

/// An index in the engine's fixed 32-action bitset. Bind physical keys in the host.
/// Projects can allocate their own IDs; these four names are conventional defaults.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ActionId(u8);

impl ActionId {
    pub const JUMP: Self = Self(0);
    pub const PRIMARY: Self = Self(1);
    pub const INTERACT: Self = Self(2);
    pub const SECONDARY: Self = Self(3);

    pub const fn new(index: u8) -> Option<Self> {
        if index < 32 { Some(Self(index)) } else { None }
    }

    pub const fn mask(self) -> u32 {
        1_u32 << self.0
    }
}

/// Held actions and edges for the currently executing fixed tick.
///
/// `ingest` merges a coarse browser batch before `Engine::advance`. Press/release
/// edges are consumed by the first fixed tick only. They remain pending if no
/// tick runs; a press and release between frames are both retained. Repeated
/// presses before a tick coalesce into one edge. Storage is fixed and allocation
/// free. Read `just_pressed`/`just_released` in `FixedSimulation`, not presentation.
#[derive(Resource, Debug, Default)]
pub struct ActionInput {
    held: u32,
    pending_pressed: u32,
    pending_released: u32,
    tick_pressed: u32,
    tick_released: u32,
}

impl ActionInput {
    /// Merge explicit host transitions and transitions inferred from held state.
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

    /// Release held state and discard pending edges on pause/focus loss.
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
