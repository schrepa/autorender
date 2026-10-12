//! What every host runs at each animation frame, the page's or a window's
//! redraw: preparation, which draws the starting state without advancing
//! time until frames stop creating pipelines and terrains have loaded the
//! view (or [`PREPARE_LIMIT_MS`] passed), loading meanwhile while the GPU
//! compiles; then ordinary frames of play, paced by the recent interval
//! between them, so terrain loads in the time frames leave free. Hosts keep
//! their transports: animation frames and tasks, or redraws.

use super::now_ms;
use super::policy::{
    BETWEEN_MARGIN_MS, FRAME_INTERVAL_MS, PACE_INTERVAL_LIMIT_MS, PACE_SMOOTHING, PREPARE_LIMIT_MS,
    QUIET_FRAMES,
};
use crate::{Game, GameHost};

/// What a host's animation frame did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FramePhase {
    /// Nothing drew: the GPU compiles pipelines, which a frame would only
    /// queue behind. The frame loaded terrain instead, and hosts whose
    /// frames stop meanwhile (browsers', while their GPU process compiles)
    /// load in tasks ([`GameHost::work_while_compiling`]).
    Compiling = 0,
    /// A preparing frame drew the starting state; play has yet to start.
    Preparing = 1,
    /// Preparation ended with this frame: play starts with the next.
    Started = 2,
    /// An ordinary frame of play.
    Playing = 3,
}

impl FramePhase {
    /// Every phase with its name, as the page's constants name them.
    pub const ALL: [(Self, &'static str); 4] = [
        (Self::Compiling, "compiling"),
        (Self::Preparing, "preparing"),
        (Self::Started, "started"),
        (Self::Playing, "playing"),
    ];
}

/// Preparation under way: since when (the host's clock) and how many
/// frames in a row created no pipeline.
#[derive(Clone, Copy, Debug)]
struct Preparing {
    since: Option<f64>,
    quiet: u32,
}

/// When the next frame is due: the recent interval between frames, without
/// the time loading between them took.
#[derive(Clone, Copy, Debug)]
struct Pace {
    last: Option<f64>,
    interval: f64,
    spent: f64,
}

/// A host's frames: preparing, then playing, paced.
#[derive(Clone, Debug)]
pub struct FrameLoop {
    preparing: Option<Preparing>,
    pace: Pace,
    // Whether terrain loads between frames.
    between: bool,
}

impl Default for FrameLoop {
    fn default() -> Self {
        Self {
            preparing: Some(Preparing {
                since: None,
                quiet: 0,
            }),
            pace: Pace {
                last: None,
                interval: FRAME_INTERVAL_MS,
                spent: 0.0,
            },
            between: true,
        }
    }
}

impl FrameLoop {
    /// Whether frames prepare the starting state before play (the
    /// default), or play at once, as development sessions start.
    pub fn set_preparation(&mut self, enabled: bool) {
        self.preparing = enabled.then_some(Preparing {
            since: None,
            quiet: 0,
        });
    }

    /// Whether play has yet to start.
    pub fn preparing(&self) -> bool {
        self.preparing.is_some()
    }

    /// Whether terrain loads between frames (the default), or within them
    /// alone, for comparisons.
    pub fn set_between_frames(&mut self, enabled: bool) {
        self.between = enabled;
    }

    /// The next interval starts afresh: after a pause, a suspension or
    /// preparation.
    pub fn skip(&mut self) {
        self.pace.last = None;
        self.pace.spent = 0.0;
    }

    /// The animation frame at `timestamp` (the host's clock,
    /// milliseconds) of `host` on a `width` × `height` view.
    pub fn frame<G: Game>(
        &mut self,
        host: &mut GameHost<G>,
        timestamp: f64,
        (width, height): (u32, u32),
    ) -> crate::Result<FramePhase> {
        let Some(preparing) = &mut self.preparing else {
            host.frame(timestamp, width, height)?;
            self.paced(timestamp);
            return Ok(FramePhase::Playing);
        };
        let since = *preparing.since.get_or_insert_with(now_ms);
        let late = now_ms() - since > PREPARE_LIMIT_MS as f64;
        if host.frame_stats().pipelines_compiling && !late {
            host.work(width, height);
            return Ok(FramePhase::Compiling);
        }
        host.suspend();
        host.frame(timestamp, width, height)?;
        let after = host.frame_stats();
        preparing.quiet = if after.render.pipelines_created == 0 {
            preparing.quiet + 1
        } else {
            0
        };
        if (preparing.quiet < QUIET_FRAMES || after.loading) && !late {
            return Ok(FramePhase::Preparing);
        }
        self.preparing = None;
        host.suspend();
        self.skip();
        Ok(FramePhase::Started)
    }

    /// Records a frame of play that started at `timestamp`: the interval
    /// since the last, less the loading between them, moves the pace.
    fn paced(&mut self, timestamp: f64) {
        let pace = &mut self.pace;
        if let Some(last) = pace.last {
            let interval = timestamp - last - pace.spent;
            if interval > 0.0 && interval < PACE_INTERVAL_LIMIT_MS {
                pace.interval += (interval - pace.interval) * PACE_SMOOTHING;
            }
        }
        pace.last = Some(timestamp);
        pace.spent = 0.0;
    }

    /// Loads terrain for `host` until shortly before the frame after the
    /// last of play is due, while the last found tiles to load; whether
    /// tiles are still to load.
    pub fn between<G: Game>(&mut self, host: &mut GameHost<G>) -> bool {
        let Some(last) = self
            .pace
            .last
            .filter(|_| self.between && self.preparing.is_none())
        else {
            return false;
        };
        let start = now_ms();
        let spare = last + self.pace.interval - start - BETWEEN_MARGIN_MS;
        if spare <= 1.0 {
            return false;
        }
        let remaining = host.work_between_frames(spare);
        self.pace.spent = now_ms() - start;
        remaining
    }
}
