//! A game drawn into an offscreen texture, without a window: for tools,
//! agents and measurements on machines whose GPU has no display, or where
//! none is wanted. [`Offscreen::record`] is the one way tools draw frames
//! and keep what they ask for: a frame-exact GPU capture of the last
//! frame, a deep profile of them all, and the last frame's per-draw
//! hardware counters replayed through PIX; the command line
//! ([`crate::HostOptions`]) and development sessions' `profile` both draw
//! through it.

use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use engine::{Engine, Game, GameHost, GpuPreference, InputState, RenderTarget};
use serde_json::Value;

use crate::files::Image;
use crate::jobs::{JobOptions, JobPool};
use crate::{Pix, Preparation, RenderDoc, host_error};

/// A game drawn offscreen.
pub struct Offscreen<G: Game> {
    // Tens of kilobytes: boxed, as moves would copy it through the stack.
    host: Box<GameHost<G>>,
    // Physical pixels.
    size: (u32, u32),
    jobs: Option<JobPool>,
}

/// How many frames [`Offscreen::record`] draws: a count, or as many as
/// fit in a time, and then one more, the last.
#[derive(Clone, Copy, Debug)]
pub enum Frames {
    Count(u32),
    For(Duration),
}

/// What [`Offscreen::record`] keeps of the frames it draws.
#[derive(Clone, Debug, Default)]
pub struct Record {
    /// The last frame's GPU work, frame-exact, at this path (see
    /// [`Offscreen::capture_frame`]).
    pub gpu_capture: Option<PathBuf>,
    /// A deep profile of every frame (builds with the `profiling` feature).
    pub profile: Option<ProfileRecord>,
    /// The last frame captured through PIX, on D3D12, and replayed for each
    /// draw's hardware counters (builds with the `profiling` feature).
    pub pix_counters: Option<PixRecord>,
}

/// A deep profile's files: into `folder`, named after `stem`, of at most
/// `max_events` CPU events (the engine's most, where `None`), with CPU
/// samples of the game's threads where `cpu_samples`, and `context` (what
/// ran) in its summary.
#[derive(Clone, Debug)]
pub struct ProfileRecord {
    pub folder: PathBuf,
    pub stem: String,
    pub max_events: Option<usize>,
    pub cpu_samples: bool,
    pub context: Value,
}

/// PIX's per-draw hardware counters of the last frame: its capture
/// (`frame.wpix`), event list and summary into `folder`, of the counters
/// `counters` names (PIX's names), or of the summary's own where empty.
#[derive(Clone, Debug)]
pub struct PixRecord {
    pub folder: PathBuf,
    pub counters: Vec<String>,
}

/// What [`Offscreen::record`] drew and kept.
#[derive(Debug)]
pub struct Recorded {
    pub frames: u32,
    /// Each frame's CPU time: its simulation, extraction and rendering.
    pub cpu_ms: Vec<f64>,
    pub elapsed_ms: f64,
    /// The tool that captured the last frame: `pix`, `renderdoc` or
    /// `debugger`.
    pub captured: Option<&'static str>,
    /// Where the last frame was captured to.
    pub capture: Option<PathBuf>,
    /// The deep profile's files, and whether it was complete.
    pub profile: Option<Value>,
    /// The summary of PIX's replayed counters.
    pub pix_counters: Option<Value>,
    /// The failure that stopped the frames early.
    pub failed: Option<engine::Error>,
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
        // Normal warm-up overlaps renderer creation, within the same host
        // launch boundary used by callers and measurement.
        let mut jobs = jobs.pool();
        // The startup future holds the renderer's whole start (some 80 KB for
        // the Stress Lab): on the heap, not the calling thread's stack.
        let mut host = Box::new(pollster::block_on(Box::pin(GameHost::launch(
            target, build,
        )))?);
        if let Some(pool) = &mut jobs {
            pool.attach(&mut host);
        }
        Ok(Self { host, size, jobs })
    }

    /// Its size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        self.size
    }

    /// Draws `frames` frames, the `n`th at the timestamp `at(n)` gives, and
    /// keeps what `record` asks for. PIX captures need `pix`, loaded before
    /// the game's device existed. Frames stop at the first that fails, which
    /// the result tells; what was kept of the others is kept.
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
            return Err(engine::Error::new(
                engine::ErrorCode::UnsupportedCommand,
                "$",
                "profiles and PIX's counters need a build with the profiling feature",
            ));
        }
        let gpu_capture = record.gpu_capture.clone().or_else(|| {
            record
                .pix_counters
                .as_ref()
                .map(|dive| dive.folder.join("frame.wpix"))
        });
        if let Some(dive) = &record.pix_counters {
            std::fs::create_dir_all(&dive.folder).map_err(|error| {
                host_error(&format!("Cannot create {}", dive.folder.display()), error)
            })?;
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
                context["frame_failed"] = failed.as_ref().map(ToString::to_string).into();
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
                    return Err(engine::Error::from(format!(
                        "PIX's counters replay D3D12 frames PIX captured, where it is installed; \
                         this one drew on {} ({})",
                        self.host.adapter().backend,
                        captured.unwrap_or("no capture"),
                    )));
                }
                Some(
                    engine_profiler::vendor::pix_counters(path, &dive.folder, &dive.counters)
                        .map_err(engine::Error::from)?,
                )
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

    /// Starts a deep profile of the frames drawn until
    /// [`Self::finish_profile`]: the engine's capture of at most
    /// `max_events` CPU events and every frame's GPU spans, and with
    /// `cpu_samples` samples of the calling thread, which draws the game,
    /// and the job workers.
    #[cfg(feature = "profiling")]
    fn start_profile(
        &mut self,
        max_events: usize,
        cpu_samples: bool,
    ) -> engine::Result<engine_profiler::DeepCapture> {
        let threads = if cpu_samples {
            engine_profiler::sampler::SampledThread::current("Main thread")
                .into_iter()
                .chain(self.jobs.iter().flat_map(JobPool::sampled_threads))
                .collect()
        } else {
            Vec::new()
        };
        engine_profiler::DeepCapture::start(&mut self.host, threads, max_events)
            .map_err(|error| host_error("Profiling could not start", error))
    }

    /// Stops the deep profile `capture` and writes its files into `folder`
    /// as `stem`, with `context` (what ran) in its summary.
    #[cfg(feature = "profiling")]
    fn finish_profile(
        &mut self,
        capture: engine_profiler::DeepCapture,
        folder: &Path,
        stem: &str,
        context: Value,
    ) -> engine::Result<engine_profiler::ProfileFiles> {
        capture
            .finish(&mut self.host, folder, stem, context)
            .map_err(|error| host_error("Profiling failed", error))
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
        self.host.work(self.size.0, self.size.1);
        self.exchange_jobs();
        while !preparation.frame(&mut self.host, timestamp, self.size)? {
            self.exchange_jobs();
        }
        Ok(())
    }

    /// One frame at `timestamp` milliseconds, without input, once the GPU
    /// drew it: offscreen frames present nowhere that would pace them.
    pub fn frame(&mut self, timestamp: f64) -> engine::Result<()> {
        self.frame_in_flight(timestamp)?;
        self.host.wait_for_gpu()
    }

    /// One frame at `timestamp` milliseconds, without input, submitted to
    /// the GPU without waiting for it to draw: hosts that keep frames in
    /// flight wait for their marks ([`GameHost::gpu_mark`]).
    pub fn frame_in_flight(&mut self, timestamp: f64) -> engine::Result<()> {
        let (width, height) = self.size;
        self.host
            .frame(timestamp, width, height, InputState::default(), 0, 0, 0)?;
        self.exchange_jobs();
        Ok(())
    }

    /// One frame at `timestamp`, captured frame-exact into `path`: by PIX
    /// where `pix` is loaded and the game draws on D3D12; else by RenderDoc
    /// where the program runs under it (into a file named after `path`,
    /// `_capture.rdc`); else by the graphics debugger wgpu reaches (Xcode with
    /// `MTL_CAPTURE_ENABLED=1`), which writes its own files. Which captured
    /// it: `pix`, `renderdoc` or `debugger` (where one is attached).
    pub fn capture_frame(
        &mut self,
        timestamp: f64,
        pix: Option<&Pix>,
        path: &Path,
    ) -> engine::Result<&'static str> {
        let pix = pix.filter(|_| self.host.adapter().backend == "Dx12");
        // Work submitted before belongs to earlier frames.
        self.host.wait_for_gpu()?;
        if let Some(pix) = pix {
            pix.begin(path).map_err(|error| host_error("PIX", error))?;
            let frame = self.frame(timestamp);
            pix.end().map_err(|error| host_error("PIX", error))?;
            return frame.map(|()| "pix");
        }
        if let Some(renderdoc) = RenderDoc::attached() {
            renderdoc
                .begin(path)
                .map_err(|error| host_error("RenderDoc", error))?;
            let frame = self.frame(timestamp);
            renderdoc
                .end()
                .map_err(|error| host_error("RenderDoc", error))?;
            return frame.map(|()| "renderdoc");
        }
        self.host.start_debugger_capture();
        let frame = self.frame(timestamp);
        self.host.stop_debugger_capture()?;
        frame.map(|()| "debugger")
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
