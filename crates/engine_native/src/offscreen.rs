use engine::host::FramePhase;
use engine::{BrickList, Game, GameHost, RenderTarget, Start};

pub(crate) struct Offscreen<G: Game> {
    host: Box<GameHost<G>>,
    size: (u32, u32),
    speaker: crate::sound::Speaker,
    io: crate::io::Io,
}

impl<G: Game> Offscreen<G> {
    pub(crate) fn launch(
        size: (u32, u32),
        start: Start,
        bricks: BrickList,
    ) -> engine::Result<Self> {
        #[cfg(feature = "profiling")]
        engine_profiler::sampler::listen_for_threads();
        let target = RenderTarget::Offscreen { size };
        let mut host = Box::new(pollster::block_on(Box::pin(GameHost::launch(
            target, start, bricks,
        )))?);
        host.offer_close();
        Ok(Self {
            host,
            size,
            speaker: crate::sound::Speaker::silent(),
            io: crate::io::Io::default(),
        })
    }

    pub(crate) fn host(&self) -> &GameHost<G> {
        &self.host
    }

    pub(crate) fn host_mut(&mut self) -> &mut GameHost<G> {
        &mut self.host
    }

    fn turn(&mut self, timestamp: f64) -> engine::Result<FramePhase> {
        let (width, height) = self.size;
        let phase = self.host.animation_frame(timestamp, width, height)?;
        self.speaker.exchange(&mut self.host);
        self.io.exchange(&mut self.host);
        self.host.wait_for_gpu()?;
        Ok(phase)
    }

    pub(crate) fn play(
        &mut self,
        frames: Option<u32>,
        mut between: impl FnMut(&mut GameHost<G>, FramePhase) -> engine::Result<bool>,
    ) -> engine::Result<u32> {
        let mut played = 0_u32;
        loop {
            let timestamp = match frames {
                Some(_) => f64::from(played) * self.host.refresh_interval(),
                None => engine::host::now_ms(),
            };
            let phase = self.turn(timestamp)?;
            if phase == FramePhase::Playing {
                played += 1;
            }
            if !between(&mut self.host, phase)?
                || frames.is_some_and(|frames| played >= frames)
                || self.host.close_asked()
            {
                return Ok(played);
            }
        }
    }
}
