use engine_core::PauseReason;

use super::now_ms;
use super::policy::{BETWEEN_MARGIN_MS, PREPARE_LIMIT_MS};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum HostPause {
    Focus = 0,
    Hidden = 1,
    Suspended = 2,
}

impl HostPause {
    pub const ALL: [(Self, &'static str); 3] = [
        (Self::Focus, "focus"),
        (Self::Hidden, "hidden"),
        (Self::Suspended, "suspended"),
    ];

    pub const fn reason(self) -> PauseReason {
        match self {
            Self::Focus => PauseReason::Focus,
            Self::Hidden => PauseReason::Hidden,
            Self::Suspended => PauseReason::Suspended,
        }
    }

    pub fn of(code: u32) -> Option<Self> {
        Self::ALL
            .iter()
            .find(|&&(pause, _)| pause as u32 == code)
            .map(|&(pause, _)| pause)
    }
}

#[derive(Clone, Copy, Debug)]
enum Stage {
    Unprepared,
    Preparing { since: f64 },
    Playing,
}

#[derive(Clone, Copy, Debug)]
enum View {
    Sized,
    Empty,
}

#[derive(Clone, Debug)]
pub struct FrameLoop {
    stage: Stage,
    view: View,
    started: f64,
    between: bool,
}

impl Default for FrameLoop {
    fn default() -> Self {
        Self {
            stage: Stage::Unprepared,
            view: View::Sized,
            started: 0.0,
            between: true,
        }
    }
}

impl FrameLoop {
    pub fn set_preparation(&mut self, enabled: bool) {
        self.stage = if enabled {
            Stage::Unprepared
        } else {
            Stage::Playing
        };
    }

    pub fn preparing(&self) -> bool {
        match self.stage {
            Stage::Unprepared | Stage::Preparing { .. } => true,
            Stage::Playing => false,
        }
    }

    pub fn set_between_frames(&mut self, enabled: bool) {
        self.between = enabled;
    }

    pub fn frame<G: Game>(
        &mut self,
        host: &mut GameHost<G>,
        timestamp: f64,
        (width, height): (u32, u32),
    ) -> crate::Result<FramePhase> {
        if width == 0 || height == 0 {
            host.pause(HostPause::Hidden);
            self.view = View::Empty;
            return Ok(FramePhase::Idle);
        }
        if let View::Empty = self.view {
            host.resume(HostPause::Hidden);
        }
        self.view = View::Sized;
        self.started = timestamp;
        let (stage, phase) = self.advance(host, timestamp, (width, height))?;
        self.stage = stage;
        Ok(phase)
    }

    fn advance<G: Game>(
        &self,
        host: &mut GameHost<G>,
        timestamp: f64,
        (width, height): (u32, u32),
    ) -> crate::Result<(Stage, FramePhase)> {
        let since = match self.stage {
            Stage::Playing => {
                host.frame(timestamp, width, height)?;
                host.paced();
                self.spare(host);
                return Ok((Stage::Playing, FramePhase::Playing));
            }
            Stage::Unprepared => now_ms(),
            Stage::Preparing { since } => since,
        };
        let preparing = Stage::Preparing { since };
        let late = now_ms() - since > PREPARE_LIMIT_MS as f64;
        if host.prepare_pipelines() && !late {
            host.work(width, height);
            return Ok((preparing, FramePhase::Preparing));
        }
        if host.frame_stats().readiness.compiling > 0 && !late {
            host.work(width, height);
            return Ok((preparing, FramePhase::Compiling));
        }
        host.suspend();
        host.frame(timestamp, width, height)?;
        self.spare(host);
        let after = host.frame_stats();
        if (after.render.pipelines_created > 0 || !after.readiness.settled()) && !late {
            return Ok((preparing, FramePhase::Preparing));
        }
        host.suspend();
        Ok((Stage::Playing, FramePhase::Started))
    }

    fn spare<G: Game>(&self, host: &mut GameHost<G>) {
        let spare = if self.between {
            self.started + host.refresh_interval() - now_ms() - BETWEEN_MARGIN_MS
        } else {
            0.0
        };
        host.set_spare_time(spare);
    }
}
