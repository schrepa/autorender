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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quick_tap_is_preserved_and_consumed_once() {
        let mut input = ActionInput::default();
        input.ingest(0, ActionId::JUMP.mask(), ActionId::JUMP.mask());
        input.ingest(0, 0, 0); // Another render frame with no fixed simulation.
        input.begin_tick();
        assert!(input.just_pressed(ActionId::JUMP));
        assert!(input.just_released(ActionId::JUMP));
        assert!(!input.pressed(ActionId::JUMP));
        input.end_tick();
        input.begin_tick();
        assert!(!input.just_pressed(ActionId::JUMP));
        assert!(!input.just_released(ActionId::JUMP));
    }

    #[test]
    fn held_state_survives_ticks_and_focus_reset_discards_edges() {
        let mut input = ActionInput::default();
        input.ingest(ActionId::PRIMARY.mask(), 0, 0);
        input.begin_tick();
        assert!(input.just_pressed(ActionId::PRIMARY));
        input.end_tick();
        input.begin_tick();
        assert!(input.pressed(ActionId::PRIMARY));
        assert!(!input.just_pressed(ActionId::PRIMARY));
        input.clear();
        input.begin_tick();
        assert!(!input.pressed(ActionId::PRIMARY));
        assert!(!input.just_released(ActionId::PRIMARY));
        assert!(ActionId::new(31).is_some());
        assert!(ActionId::new(32).is_none());
    }

    #[test]
    fn schedule_consumes_edges_on_first_tick_and_preserves_between_render_frames() {
        use crate::prelude::*;
        #[derive(Resource, Default)]
        struct Presses(u32);
        let mut engine = Engine::default();
        engine.app.init_resource::<Presses>().add_systems(
            FixedSimulation,
            |input: Res<ActionInput>, mut count: ResMut<Presses>| {
                count.0 += u32::from(input.just_pressed(ActionId::JUMP));
            },
        );
        engine
            .app
            .world_mut()
            .resource_mut::<ActionInput>()
            .ingest(0, 1, 1);
        assert_eq!(engine.advance(0.001).fixed_steps, 0);
        assert_eq!(engine.app.world().resource::<Presses>().0, 0);
        assert_eq!(engine.advance(0.1).fixed_steps, 5);
        assert_eq!(engine.app.world().resource::<Presses>().0, 1);
        engine.advance(0.1);
        assert_eq!(engine.app.world().resource::<Presses>().0, 1);
        engine
            .app
            .world_mut()
            .resource_mut::<ActionInput>()
            .ingest(1, 1, 0);
        engine.reset_clock();
        engine.advance(0.1);
        assert_eq!(engine.app.world().resource::<Presses>().0, 1);
    }
}
