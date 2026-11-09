use engine_core::PauseReason;

use super::now_ms;
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
}

impl Default for FrameLoop {
    fn default() -> Self {
        Self {
            stage: Stage::Unprepared,
            view: View::Sized,
            started: 0.0,
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
            Stage::Unprepared => {
                host.suspend();
                host.hold_preparation(true);
                now_ms()
            }
            Stage::Preparing { since } => since,
        };
        let preparing = Stage::Preparing { since };
        host.set_preparation_ms(now_ms() - since);
        host.frame(timestamp, width, height)?;
        self.spare(host);
        let after = host.frame_stats();
        if after.readiness.compiling > 0 {
            return Ok((preparing, FramePhase::Compiling));
        }
        let listing = host.prepare_pipelines();
        if listing || after.render.pipelines_created > 0 || !after.readiness.settled() {
            return Ok((preparing, FramePhase::Preparing));
        }
        host.set_preparation_ms(now_ms() - since);
        host.hold_preparation(false);
        host.suspend();
        Ok((Stage::Playing, FramePhase::Started))
    }

    fn spare<G: Game>(&self, host: &mut GameHost<G>) {
        let margin = crate::pacing::of(host.engine().app.world()).margin_ms;
        let spare = self.started + host.refresh_interval() - now_ms() - margin;
        host.set_spare_time(spare);
    }
}
