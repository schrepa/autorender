//! The one place that does arithmetic on Bevy's change ticks: windows of
//! what changed since a scan last ran, and the clocks of caches that follow
//! changes themselves. Everything else asks these; in debug builds each
//! answer is checked against Bevy's own [`Tick::is_newer_than`].

use bevy_ecs::change_detection::{MAX_CHANGE_AGE, Tick};
use bevy_ecs::system::{SystemParam, SystemState};

/// Stamps classified per block before any payload or source record is read.
pub(crate) const STAMP_BLOCK: usize = 16;

/// Bevy's `Tick::is_newer_than` for one fixed scan window. With wrapping u32
/// subtraction and `since = min(this_run - last_run, MAX_CHANGE_AGE)`, a stamp
/// `t` changed exactly when `min(this_run - t, MAX_CHANGE_AGE) < since`. Because
/// `since <= MAX_CHANGE_AGE`, that is `this_run - t < since`.
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
            "the change window disagrees with Bevy on tick {tick:?}"
        );
        changed
    }

    /// Whether any stamp of a block of at most [`STAMP_BLOCK`] changed.
    /// Branch-free over the whole block so it can compile to vector compares.
    #[cfg_attr(not(feature = "attribution"), inline)]
    #[cfg_attr(feature = "attribution", inline(never))]
    pub(crate) fn any_in_block(self, ticks: &[Tick]) -> bool {
        ticks
            .iter()
            .fold(false, |any, &tick| any | self.contains(tick))
    }

    /// Whether any stamp changed. Fixed-size blocks compile to unrolled vector
    /// compares, and the test stops at the first changed block.
    #[inline]
    pub(crate) fn any(self, ticks: &[Tick]) -> bool {
        let (blocks, rest) = ticks.as_chunks::<STAMP_BLOCK>();
        blocks.iter().any(|block| self.any_in_block(block)) || self.any_in_block(rest)
    }
}

/// When a cache that follows changes itself last scanned. A scan
/// [`Self::aged`] by [`MAX_CHANGE_AGE`] ticks or more can no longer tell
/// what changed (Bevy clamps older stamps), so the next must scan
/// everything; one never made counts as aged.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ScanClock {
    last: Option<Tick>,
}

impl ScanClock {
    /// Whether no scan was made, or the last lies [`MAX_CHANGE_AGE`] ticks
    /// or more before `now`: what changed since is unknown.
    pub(crate) fn aged(self, now: Tick) -> bool {
        self.last
            .is_none_or(|last| now.get().wrapping_sub(last.get()) >= MAX_CHANGE_AGE)
    }

    /// Whether a value stamped `stamp` changed since the last scan, as of
    /// `now`: always, once the clock is aged.
    pub(crate) fn changed(self, stamp: Tick, now: Tick) -> bool {
        match self.last {
            Some(last) if !self.aged(now) => stamp.is_newer_than(last, now),
            _ => true,
        }
    }

    /// Records a scan as of `now`.
    pub(crate) fn scanned(&mut self, now: Tick) {
        self.last = Some(now);
    }

    /// Forgets the last scan: the next scans everything.
    pub(crate) fn forget(&mut self) {
        self.last = None;
    }
}

/// Makes `state`'s queries see changes after `tick` alone from now on: a
/// scan that consumed the window up to it owns it, whatever ran between.
pub(crate) fn consume_through<P: SystemParam + 'static>(state: &mut SystemState<P>, tick: Tick) {
    state.meta_mut().set_last_run(tick);
}
