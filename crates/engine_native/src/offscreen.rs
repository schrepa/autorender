use std::time::Instant;

use engine::host::FramePhase;
use engine::{BrickList, Engine, Game, GameHost, GpuPreference, RenderTarget};

pub struct Offscreen<G: Game> {
    host: Box<GameHost<G>>,
    size: (u32, u32),
    speaker: crate::sound::Speaker,
}

impl<G: Game> Offscreen<G> {
    pub fn launch<E: Into<engine::Error>>(
        size: (u32, u32),
        gpu: GpuPreference,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
    ) -> engine::Result<Self> {
        #[cfg(feature = "profiling")]
        engine_profiler::sampler::listen_for_threads();
        let target = RenderTarget::Offscreen { size, gpu };
        let host = Box::new(pollster::block_on(Box::pin(GameHost::launch(
            target, bricks, build,
        )))?);
        Ok(Self {
            host,
            size,
            speaker: crate::sound::Speaker::silent(),
        })
    }

    pub fn host(&self) -> &GameHost<G> {
        &self.host
    }

    pub fn host_mut(&mut self) -> &mut GameHost<G> {
        &mut self.host
    }

    fn turn(&mut self, timestamp: f64) -> engine::Result<FramePhase> {
        let (width, height) = self.size;
        let phase = self.host.animation_frame(timestamp, width, height)?;
        self.speaker.exchange(&mut self.host);
        self.host.wait_for_gpu()?;
        Ok(phase)
    }

    pub fn play(
        &mut self,
        frames: Option<u32>,
        mut between: impl FnMut(&mut GameHost<G>, FramePhase) -> engine::Result<bool>,
    ) -> engine::Result<u32> {
        let started = Instant::now();
        let (mut played, mut playing) = (0_u32, false);
        loop {
            let timestamp = match frames {
                Some(_) if playing => f64::from(played) * self.host.refresh_interval(),
                Some(_) | None => started.elapsed().as_secs_f64() * 1000.0,
            };
            let phase = self.turn(timestamp)?;
            match phase {
                FramePhase::Started => playing = true,
                FramePhase::Playing => played += 1,
                FramePhase::Compiling
                | FramePhase::Preparing
                | FramePhase::Collecting
                | FramePhase::Idle
                | FramePhase::Paused => {}
            }
            if !between(&mut self.host, phase)? || frames.is_some_and(|frames| played >= frames) {
                return Ok(played);
            }
        }
    }
}
