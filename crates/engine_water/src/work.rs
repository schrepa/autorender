use std::cell::Cell;

use bevy_ecs::prelude::*;
use engine_core::WorkClock;
use engine_places::ground::GroundWork;
use engine_places::water::WaterWork;

use crate::state::WaterState;

pub(crate) struct Work {
    clock: Option<fn() -> f64>,
    end: f64,
    budget: u64,
    uploaded: Cell<u64>,
}

impl Work {
    pub(crate) fn of(clock: &WorkClock, ground: &GroundWork, uploaded: u64) -> Self {
        Self {
            clock: clock.reading,
            end: ground.end,
            budget: ground.upload,
            uploaded: Cell::new(uploaded),
        }
    }

    pub(crate) fn expired(&self) -> bool {
        self.clock.is_some_and(|clock| clock() >= self.end)
    }

    pub(crate) fn uploads(&self) -> bool {
        self.uploaded.get() < self.budget
    }

    pub(crate) fn upload(&self, bytes: u64) {
        self.uploaded.set(self.uploaded.get().saturating_add(bytes));
    }

    pub(crate) fn uploaded(&self) -> u64 {
        self.uploaded.get()
    }
}

pub(crate) fn publish_work(state: Res<WaterState>, mut work: ResMut<WaterWork>) {
    work.set_if_neq(WaterWork {
        uploaded: state.uploaded,
    });
}
