use std::cell::Cell;

use engine_core::WorkClock;
use engine_places::ground::GroundWork;

pub(crate) struct Work {
    pub(crate) slice: u64,
    clock: Option<fn() -> f64>,
    end: f64,
    upload: Cell<u64>,
}

impl Work {
    pub(crate) fn of(clock: &WorkClock, ground: &GroundWork) -> Self {
        Self::until(
            clock,
            ground,
            clock.now().map_or(0.0, |now| now + ground.millis),
        )
    }

    pub(crate) fn until(clock: &WorkClock, ground: &GroundWork, end: f64) -> Self {
        Self {
            slice: ground.slice.max(1),
            clock: clock.reading,
            end,
            upload: Cell::new(ground.upload),
        }
    }

    pub(crate) fn end(&self) -> f64 {
        self.end
    }

    pub(crate) fn refill(&self, budget: &mut u64) -> bool {
        if *budget > 0 {
            return false;
        }
        match self.clock {
            Some(clock) if clock() < self.end => {
                *budget = self.slice;
                true
            }
            _ => false,
        }
    }

    pub(crate) fn expired(&self) -> bool {
        self.clock.is_some_and(|clock| clock() >= self.end)
    }

    pub(crate) fn uploads(&self) -> bool {
        self.upload.get() > 0
    }

    pub(crate) fn upload(&self, bytes: u64) {
        self.upload.set(self.upload.get().saturating_sub(bytes));
    }
}
