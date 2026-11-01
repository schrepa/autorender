use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine::host::FramePhase;
use engine::{
    BrickList, Engine, Error, ErrorCode, Expected, Game, GameHost, GpuPreference, RenderTarget,
};
use serde_json::Value;

use crate::files::Image;
use crate::jobs::{JobOptions, JobPool};
use crate::{DYNAMIC_RESOLUTION, Pix, RenderDoc, file_error};

pub struct Offscreen<G: Game> {
    host: Box<GameHost<G>>,
    size: (u32, u32),
    jobs: JobPool,
}

#[derive(Clone, Copy, Debug)]
pub enum Frames {
    Count(u32),
    For(Duration),
}

#[derive(Clone, Debug, Default)]
pub struct Record {
    pub gpu_capture: Option<PathBuf>,
    pub profile: Option<ProfileRecord>,
    pub pix_counters: Option<PixRecord>,
}

#[derive(Clone, Debug)]
pub struct ProfileRecord {
    pub folder: PathBuf,
    pub stem: String,
    pub max_events: Option<usize>,
    pub cpu_samples: bool,
    pub context: Value,
}

#[derive(Clone, Debug)]
pub struct PixRecord {
    pub folder: PathBuf,
    pub counters: Vec<String>,
}

#[derive(Debug)]
pub struct Recorded {
    pub frames: u32,
    pub cpu_ms: Vec<f64>,
    pub elapsed_ms: f64,
    pub captured: Option<&'static str>,
    pub capture: Option<PathBuf>,
    pub profile: Option<Value>,
    pub pix_counters: Option<Value>,
    pub failed: Option<engine::Error>,
}

impl<G: Game> Offscreen<G> {
    pub fn launch<E: Into<engine::Error>>(
        size: (u32, u32),
        gpu: GpuPreference,
        jobs: &JobOptions,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
    ) -> engine::Result<Self> {
        Self::launch_with(size, gpu, jobs.pool(false), bricks, build)
    }

    pub fn launch_with<E: Into<engine::Error>>(
        size: (u32, u32),
        gpu: GpuPreference,
        mut jobs: JobPool,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
    ) -> engine::Result<Self> {
        let target = RenderTarget::Offscreen { size, gpu };
        let mut host = Box::new(pollster::block_on(Box::pin(GameHost::launch(
            target, bricks, build,
        )))?);
        host.set_setting(DYNAMIC_RESOLUTION, false.into())?;
        jobs.attach(&mut host);
        Ok(Self { host, size, jobs })
    }

    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    pub fn record(
        &mut self,
        frames: Frames,
        mut at: impl FnMut(u32) -> f64,
        record: &Record,
        pix: Option<&Pix>,
    ) -> engine::Result<Recorded> {
        if !cfg!(feature = "profiling")
            && (record.profile.is_some() || record.pix_counters.is_some())
        {
            return Err(
                Error::new(ErrorCode::UnsupportedCommand, "$").expected(Expected::Supported)
            );
        }
        let gpu_capture = record.gpu_capture.clone().or_else(|| {
            record
                .pix_counters
                .as_ref()
                .map(|dive| dive.folder.join("frame.wpix"))
        });
        if let Some(dive) = &record.pix_counters {
            std::fs::create_dir_all(&dive.folder)
                .map_err(|error| file_error("$.folder", &dive.folder, &error))?;
        }
        #[cfg(feature = "profiling")]
        let capture = match &record.profile {
            Some(profile) => Some(self.start_profile(
                profile.max_events.unwrap_or(engine::profiling::MAX_EVENTS),
                profile.cpu_samples,
            )?),
            None => None,
        };
        let started = Instant::now();
        let (mut drawn, mut cpu_ms, mut captured, mut failed) = (0, Vec::new(), None, None);
        loop {
            let last = match frames {
                Frames::Count(count) => drawn + 1 >= count,
                Frames::For(time) => started.elapsed() >= time,
            };
            let timestamp = at(drawn);
            let frame = match (&gpu_capture, last) {
                (Some(path), true) => self
                    .capture_frame(timestamp, pix, path)
                    .map(|by| captured = Some(by)),
                _ => self.frame(timestamp),
            };
            if let Err(error) = frame {
                failed = Some(error);
                break;
            }
            let stats = self.host.frame_stats();
            cpu_ms.push(stats.simulation_ms + stats.extraction_ms + stats.render_ms);
            drawn += 1;
            if last {
                break;
            }
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(feature = "profiling")]
        let profile = match (capture, &record.profile) {
            (Some(capture), Some(profile)) => {
                let mut context = profile.context.clone();
                context["frames"] = drawn.into();
                context["frame_failed"] = serde_json::json!(failed);
                let files =
                    self.finish_profile(capture, &profile.folder, &profile.stem, context)?;
                Some(serde_json::json!({
                    "summary": files.summary.display().to_string(),
                    "firefox": files.firefox.display().to_string(),
                    "report": files.report.display().to_string(),
                    "complete": files.complete && failed.is_none(),
                }))
            }
            _ => None,
        };
        #[cfg(not(feature = "profiling"))]
        let profile = None;
        #[cfg(feature = "profiling")]
        let pix_counters = match (&record.pix_counters, &gpu_capture) {
            (Some(dive), Some(path)) if failed.is_none() => {
                if captured != Some("pix") {
                    return Err(Error::new(ErrorCode::PixUnavailable, "$")
                        .expected(Expected::Supported)
                        .got(serde_json::json!({
                            "backend": self.host.adapter().backend,
                            "captured": captured,
                        })));
                }
                Some(engine_profiler::vendor::pix_counters(
                    path,
                    &dive.folder,
                    &dive.counters,
                )?)
            }
            _ => None,
        };
        #[cfg(not(feature = "profiling"))]
        let pix_counters = None;
        Ok(Recorded {
            frames: drawn,
            cpu_ms,
            elapsed_ms,
            captured,
            capture: gpu_capture,
            profile,
            pix_counters,
            failed,
        })
    }

    #[cfg(feature = "profiling")]
    fn start_profile(
        &mut self,
        max_events: usize,
        cpu_samples: bool,
    ) -> engine::Result<engine_profiler::DeepCapture> {
        let threads = if cpu_samples {
            engine_profiler::sampler::SampledThread::current("Main thread")
                .into_iter()
                .chain(self.jobs.sampled_threads())
                .collect()
        } else {
            Vec::new()
        };
        engine_profiler::DeepCapture::start(&mut self.host, threads, max_events)
    }

    #[cfg(feature = "profiling")]
    fn finish_profile(
        &mut self,
        capture: engine_profiler::DeepCapture,
        folder: &Path,
        stem: &str,
        context: Value,
    ) -> engine::Result<engine_profiler::ProfileFiles> {
        capture.finish(&mut self.host, folder, stem, context)
    }

    pub fn exchange_jobs(&mut self) {
        self.jobs.exchange(&mut self.host);
    }

    pub fn jobs_mut(&mut self) -> &mut JobPool {
        &mut self.jobs
    }

    pub fn host(&self) -> &GameHost<G> {
        &self.host
    }

    pub fn host_mut(&mut self) -> &mut GameHost<G> {
        &mut self.host
    }

    pub fn prepare(&mut self, timestamp: f64) -> engine::Result<()> {
        let (width, height) = self.size;
        self.host.set_preparation(true);
        while self.host.animation_frame(timestamp, width, height)? != FramePhase::Started {
            self.exchange_jobs();
        }
        self.exchange_jobs();
        Ok(())
    }

    pub fn frame(&mut self, timestamp: f64) -> engine::Result<()> {
        self.frame_in_flight(timestamp)?;
        self.host.wait_for_gpu()
    }

    pub fn frame_in_flight(&mut self, timestamp: f64) -> engine::Result<()> {
        let (width, height) = self.size;
        self.host.frame(timestamp, width, height)?;
        self.exchange_jobs();
        Ok(())
    }

    pub fn capture_frame(
        &mut self,
        timestamp: f64,
        pix: Option<&Pix>,
        path: &Path,
    ) -> engine::Result<&'static str> {
        let pix = pix.filter(|_| self.host.adapter().backend == "Dx12");
        self.host.wait_for_gpu()?;
        if let Some(pix) = pix {
            pix.begin(path)?;
            let frame = self.frame(timestamp);
            pix.end()?;
            return frame.map(|()| "pix");
        }
        if let Some(renderdoc) = RenderDoc::attached() {
            renderdoc.begin(path)?;
            let frame = self.frame(timestamp);
            renderdoc.end()?;
            return frame.map(|()| "renderdoc");
        }
        self.host.start_debugger_capture();
        let frame = self.frame(timestamp);
        self.host.stop_debugger_capture()?;
        frame.map(|()| "debugger")
    }

    pub fn image(&self) -> engine::Result<Image> {
        let (width, height, pixels) = self.host.read_offscreen()?.expect("offscreen_target");
        Ok(Image {
            width,
            height,
            pixels,
        })
    }

    pub fn save_png(&self, path: &Path) -> engine::Result<()> {
        let image = self.image()?;
        let failed = |_| {
            Error::new(ErrorCode::SourceIo, "$")
                .expected(Expected::Writable)
                .got(path.display().to_string())
        };
        let file = std::fs::File::create(path).map_err(|error| file_error("$", path, &error))?;
        let mut encoder =
            png::Encoder::new(std::io::BufWriter::new(file), image.width, image.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder.write_header().map_err(failed)?;
        writer.write_image_data(&image.pixels).map_err(failed)?;
        writer.finish().map_err(failed)
    }
}
