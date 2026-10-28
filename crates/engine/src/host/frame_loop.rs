use super::now_ms;
use super::policy::{
    BETWEEN_MARGIN_MS, FRAME_INTERVAL_MS, PACE_INTERVAL_LIMIT_MS, PACE_SMOOTHING, PREPARE_LIMIT_MS,
    QUIET_FRAMES,
};
use crate::{Game, GameHost};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum FramePhase {
    Compiling = 0,
    Preparing = 1,
    Started = 2,
    Playing = 3,
    Idle = 4,
}

impl FramePhase {
    pub const ALL: [(Self, &'static str); 5] = [
        (Self::Compiling, "compiling"),
        (Self::Preparing, "preparing"),
        (Self::Started, "started"),
        (Self::Playing, "playing"),
        (Self::Idle, "idle"),
    ];
}

#[derive(Clone, Copy, Debug)]
struct Preparing {
    since: Option<f64>,
    quiet: u32,
}

#[derive(Clone, Copy, Debug)]
struct Pace {
    last: Option<f64>,
    interval: f64,
    spent: f64,
}

#[derive(Clone, Debug)]
pub struct FrameLoop {
    preparing: Option<Preparing>,
    pace: Pace,
    between: bool,
    idle: bool,
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
            idle: false,
        }
    }
}

impl FrameLoop {
    pub fn set_preparation(&mut self, enabled: bool) {
        self.preparing = enabled.then_some(Preparing {
            since: None,
            quiet: 0,
        });
    }

    pub fn preparing(&self) -> bool {
        self.preparing.is_some()
    }

    pub fn set_between_frames(&mut self, enabled: bool) {
        self.between = enabled;
    }

    pub fn skip(&mut self) {
        self.pace.last = None;
        self.pace.spent = 0.0;
    }

    pub fn frame<G: Game>(
        &mut self,
        host: &mut GameHost<G>,
        timestamp: f64,
        (width, height): (u32, u32),
    ) -> crate::Result<FramePhase> {
        if width == 0 || height == 0 {
            if !self.idle {
                host.suspend();
                self.idle = true;
            }
            return Ok(FramePhase::Idle);
        }
        self.idle = false;
        let Some(preparing) = &mut self.preparing else {
            host.frame(timestamp, width, height)?;
            self.paced(timestamp);
            return Ok(FramePhase::Playing);
        };
        let since = *preparing.since.get_or_insert_with(now_ms);
        let late = now_ms() - since > PREPARE_LIMIT_MS as f64;
        if host.prepare_pipelines() && !late {
            host.work(width, height);
            return Ok(FramePhase::Preparing);
        }
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
