//! A game drawn into an offscreen texture, without a window: for tools,
//! agents and measurements on machines whose GPU has no display, or where
//! none is wanted.

use std::path::Path;

use engine::{Engine, Game, GameHost, GpuPreference, InputState, RenderTarget};

use crate::files::Image;
use crate::jobs::{JobOptions, JobPool};
use crate::{Preparation, host_error};

/// A game drawn offscreen.
pub struct Offscreen<G: Game> {
    host: GameHost<G>,
    // Physical pixels.
    size: (u32, u32),
    jobs: Option<JobPool>,
}

impl<G: Game> Offscreen<G> {
    /// The game `build` makes, drawn into a `size` texture on the `gpu`
    /// preferred, its jobs run as `jobs` say.
    pub fn launch<E: Into<engine::Error>>(
        size: (u32, u32),
        gpu: GpuPreference,
        jobs: JobOptions,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> engine::Result<Self> {
        let target = RenderTarget::Offscreen { size, gpu };
        let mut host = pollster::block_on(GameHost::launch(target, build))?;
        let mut jobs = jobs.pool();
        if let Some(pool) = &mut jobs {
            pool.attach(&mut host);
        }
        Ok(Self { host, size, jobs })
    }

    /// Its size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Hands workers the jobs the engine made, and the engine what workers
    /// finished: frames do so, and hosts between them.
    pub fn exchange_jobs(&mut self) {
        if let Some(jobs) = &mut self.jobs {
            jobs.exchange(&mut self.host);
        }
    }

    pub fn host(&self) -> &GameHost<G> {
        &self.host
    }

    pub fn host_mut(&mut self) -> &mut GameHost<G> {
        &mut self.host
    }

    /// Draws the starting state at `timestamp` milliseconds until frames
    /// stop creating pipelines and terrains have loaded the view, as a
    /// window does before play starts.
    pub fn prepare(&mut self, timestamp: f64) -> engine::Result<()> {
        let mut preparation = Preparation::new();
        while !preparation.frame(&mut self.host, timestamp, self.size)? {
            self.exchange_jobs();
        }
        Ok(())
    }

    /// One frame at `timestamp` milliseconds, without input, once the GPU
    /// drew it: offscreen frames present nowhere that would pace them.
    pub fn frame(&mut self, timestamp: f64) -> engine::Result<()> {
        let (width, height) = self.size;
        self.host
            .frame(timestamp, width, height, InputState::default(), 0, 0, 0)?;
        self.exchange_jobs();
        self.host.wait_for_gpu()
    }

    /// The last frame's pixels, once the GPU finished it.
    pub fn image(&self) -> engine::Result<Image> {
        let (width, height, pixels) = self
            .host
            .read_offscreen()?
            .expect("offscreen hosts draw offscreen");
        Ok(Image {
            width,
            height,
            pixels,
        })
    }

    /// The last frame as a PNG file at `path`.
    pub fn save_png(&self, path: &Path) -> engine::Result<()> {
        let image = self.image()?;
        let failed = |error: &dyn std::fmt::Display| {
            host_error(&format!("Could not write {}", path.display()), error)
        };
        let file = std::fs::File::create(path).map_err(|error| failed(&error))?;
        let mut encoder =
            png::Encoder::new(std::io::BufWriter::new(file), image.width, image.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(|error| failed(&error))?;
        writer
            .write_image_data(&image.pixels)
            .map_err(|error| failed(&error))?;
        writer.finish().map_err(|error| failed(&error))
    }
}
