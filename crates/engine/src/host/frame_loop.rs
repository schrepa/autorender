use engine_core::{GroupStage, Readiness, Show};
use engine_render::Picture;

use super::now_ms;
use crate::app_host::Advance;
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
    Paused = 6,
}

impl FramePhase {
    pub const ALL: [(Self, &'static str); 7] = [
        (Self::Compiling, "compiling"),
        (Self::Preparing, "preparing"),
        (Self::Started, "started"),
        (Self::Playing, "playing"),
        (Self::Idle, "idle"),
        (Self::Collecting, "collecting"),
        (Self::Paused, "paused"),
    ];
}

#[derive(Clone, Debug, Default)]
enum Stage {
    #[default]
    Unprepared,
    Preparing {
        since: f64,
        picture: Picture,
        before: Option<Readiness>,
    },
    Playing,
}

#[derive(Clone, Debug, Default)]
pub struct FrameLoop {
    stage: Stage,
}

impl FrameLoop {
    pub fn frame<G: Game>(
        &mut self,
        host: &mut GameHost<G>,
        timestamp: f64,
        (width, height): (u32, u32),
    ) -> crate::Result<FramePhase> {
        #[cfg(feature = "dev")]
        host.dev_apply();
        let empty = width == 0 || height == 0;
        host.sized(empty);
        if empty {
            return Ok(FramePhase::Idle);
        }
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
                let phase = host.play(timestamp, (width, height))?;
                return Ok((Stage::Playing, phase));
            }
            Stage::Unprepared if *host.engine().app.world().resource::<Show>() == Show::Now => {
                host.show(true);
                host.suspend();
                host.play(timestamp, (width, height))?;
                return Ok((Stage::Playing, FramePhase::Started));
            }
            Stage::Unprepared => {
                host.suspend();
                (now_ms(), Picture::Hidden, None)
            }
            Stage::Preparing {
                since,
                picture,
                before,
            } => (*since, *picture, before.clone()),
        };
        host.set_preparation_ms(now_ms() - since);
        let drawn = host.framed(Advance::Clock(timestamp), (width, height), picture)?;
        let readiness = host.progress();
        let render = &host.frame_times().render;
        let created = render.pipelines_created;
        let uploaded = render.mesh_upload_bytes + render.texture_upload_bytes > 0;
        let busy = created > 0 || in_flight(host, &readiness);
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
            if !readiness.settled() {
                return Ok((preparing(Picture::Hidden, Some(readiness)), waiting));
            }
            host.set_preparation_ms(now_ms() - since);
            host.show(true);
            host.suspend();
            return Ok((Stage::Playing, FramePhase::Started));
        }
        if !drawn {
            return Ok((preparing(Picture::Hidden, before), waiting));
        }
        if !busy && readiness.settled() {
            return Ok((preparing(Picture::Shown, None), waiting));
        }
        let progressed = uploaded || before.as_ref() != Some(&readiness);
        let phase = if progressed {
            FramePhase::Collecting
        } else {
            waiting
        };
        Ok((preparing(Picture::Hidden, Some(readiness)), phase))
    }
}

pub(crate) fn in_flight<G: Game>(host: &GameHost<G>, readiness: &Readiness) -> bool {
    let world = host.engine().app.world();
    let files = world
        .get_resource::<engine_core::Files>()
        .is_some_and(|files| files.census().asked > 0);
    let first = readiness
        .groups
        .first()
        .is_some_and(|group| group.stage != GroupStage::Ready);
    readiness.loading.total > 0 || readiness.compiling > 0 || files || first
}
