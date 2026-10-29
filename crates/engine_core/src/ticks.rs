use bevy_ecs::change_detection::{MAX_CHANGE_AGE, Tick};
use bevy_ecs::system::{SystemParam, SystemState};

pub(crate) const STAMP_BLOCK: usize = 16;

#[derive(Clone, Copy)]
pub(crate) struct ChangeWindow {
    this_run: u32,
    since: u32,
    #[cfg(debug_assertions)]
    last_run: Tick,
}

impl ChangeWindow {
    pub(crate) fn new(last_run: Tick, this_run: Tick) -> Self {
        Self {
            this_run: this_run.get(),
            since: this_run
                .get()
                .wrapping_sub(last_run.get())
                .min(MAX_CHANGE_AGE),
            #[cfg(debug_assertions)]
            last_run,
        }
    }

    #[inline]
    pub(crate) fn contains(self, tick: Tick) -> bool {
        let changed = self.this_run.wrapping_sub(tick.get()) < self.since;
        #[cfg(debug_assertions)]
        assert_eq!(
            changed,
            tick.is_newer_than(self.last_run, Tick::new(self.this_run)),
            "change_window_matches_bevy {tick:?}"
        );
        changed
    }

    #[cfg_attr(not(feature = "attribution"), inline)]
    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn any_in_block(self, ticks: &[Tick]) -> bool {
        ticks
            .iter()
            .fold(false, |any, &tick| any | self.contains(tick))
    }

    #[inline]
    pub(crate) fn any(self, ticks: &[Tick]) -> bool {
        let (blocks, rest) = ticks.as_chunks::<STAMP_BLOCK>();
        blocks.iter().any(|block| self.any_in_block(block)) || self.any_in_block(rest)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScanClock {
    last: Option<Tick>,
}

impl ScanClock {
    pub(crate) fn aged(self, now: Tick) -> bool {
        self.last
            .is_none_or(|last| now.get().wrapping_sub(last.get()) >= MAX_CHANGE_AGE)
    }

    pub(crate) fn changed(self, stamp: Tick, now: Tick) -> bool {
        match self.last {
            Some(last) if !self.aged(now) => stamp.is_newer_than(last, now),
            _ => true,
        }
    }

    pub(crate) fn scanned(&mut self, now: Tick) {
        self.last = Some(now);
    }

    pub(crate) fn forget(&mut self) {
        self.last = None;
    }
}

pub(crate) fn consume_through<P: SystemParam + 'static>(state: &mut SystemState<P>, tick: Tick) {
    state.meta_mut().set_last_run(tick);
}
