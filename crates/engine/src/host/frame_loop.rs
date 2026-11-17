use engine_core::{JobStats, PauseReason, Readiness};
use engine_render::Picture;

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
    Collecting = 5,
}

impl FramePhase {
    pub const ALL: [(Self, &'static str); 6] = [
        (Self::Compiling, "compiling"),
        (Self::Preparing, "preparing"),
        (Self::Started, "started"),
        (Self::Playing, "playing"),
        (Self::Idle, "idle"),
        (Self::Collecting, "collecting"),
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

#[derive(Clone, Debug)]
enum Stage {
    Unprepared,
    Preparing {
        since: f64,
        picture: Picture,
        before: Option<Sample>,
    },
    Playing,
}

#[derive(Clone, Debug, PartialEq)]
struct Sample {
    readiness: Readiness,
    landed: u64,
}

impl Sample {
    fn of(readiness: &Readiness, jobs: &JobStats) -> Self {
        Self {
            readiness: readiness.clone(),
            landed: jobs.completed + jobs.failed,
        }
    }
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
        let (since, picture, before) = match &self.stage {
            Stage::Playing => {
                host.frame(timestamp, width, height)?;
                host.paced();
                self.spare(host);
                return Ok((Stage::Playing, FramePhase::Playing));
            }
            Stage::Unprepared => {
                host.suspend();
                host.hold_preparation(true);
                (now_ms(), Picture::Hidden, None)
            }
            Stage::Preparing {
                since,
                picture,
                before,
            } => (*since, *picture, before.clone()),
        };
        host.set_preparation_ms(now_ms() - since);
        if picture == Picture::Hidden {
            whole_tick(host);
        }
        let drawn = host.framed(timestamp, (width, height), picture)?;
        if picture == Picture::Shown {
            self.spare(host);
        }
        let (readiness, jobs) = host.progress();
        let render = &host.frame_times().render;
        let created = render.pipelines_created;
        let uploaded = render.mesh_upload_bytes + render.texture_upload_bytes > 0;
        let listing = host.prepare_pipelines();
        let busy = listing || created > 0 || in_flight(host, &readiness);
        let waiting = if readiness.compiling > 0 {
            FramePhase::Compiling
        } else {
            FramePhase::Preparing
        };
        let preparing = |picture, before| Stage::Preparing {
            since,
            picture,
            before,
        };
        if picture == Picture::Shown {
            if busy {
                return Ok((preparing(Picture::Hidden, None), waiting));
            }
            if !readiness.worked() {
                let sample = Sample::of(&readiness, &jobs);
                return Ok((preparing(Picture::Hidden, Some(sample)), waiting));
            }
            if !readiness.settled() {
                return Ok((preparing(Picture::Shown, None), waiting));
            }
            host.set_preparation_ms(now_ms() - since);
            host.hold_preparation(false);
            host.suspend();
            return Ok((Stage::Playing, FramePhase::Started));
        }
        if !drawn {
            return Ok((preparing(Picture::Hidden, before), waiting));
        }
        if !busy && readiness.worked() {
            return Ok((preparing(Picture::Shown, None), waiting));
        }
        let sample = Sample::of(&readiness, &jobs);
        let progressed = uploaded || before.as_ref() != Some(&sample);
        let phase = if progressed {
            FramePhase::Collecting
        } else {
            waiting
        };
        Ok((preparing(Picture::Hidden, Some(sample)), phase))
    }

    fn spare<G: Game>(&self, host: &mut GameHost<G>) {
        let margin = crate::pacing::of(host.engine().app.world()).margin_ms;
        let spare = self.started + host.refresh_interval() - now_ms() - margin;
        host.set_spare_time(spare);
    }
}

fn whole_tick<G: Game>(host: &mut GameHost<G>) {
    let margin = crate::pacing::of(host.engine().app.world()).margin_ms;
    let tick = host.refresh_interval() - margin;
    host.set_spare_time(tick);
}

fn in_flight<G: Game>(host: &GameHost<G>, readiness: &Readiness) -> bool {
    let world = host.engine().app.world();
    let records = world
        .get_resource::<engine_core::Records>()
        .is_some_and(|records| records.census().asked > 0);
    let files = world
        .get_resource::<engine_core::Files>()
        .is_some_and(|files| files.census().asked > 0);
    readiness.loading.total > 0
        || readiness.compiling > 0
        || host.jobs_held() > 0
        || records
        || files
}
