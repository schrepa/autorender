//! A game in a native window: frames as the page draws them, one each time
//! the display can show one, with keyboard and mouse input, and terrain
//! loading between frames.

use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::host::{FramePhase, HostSettings};
use engine::{Engine, Game, GameHost, GpuPreference, RenderTarget};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;

use crate::host_error;
use crate::input::Input;
use crate::jobs::{JobOptions, JobPool};

/// How a game's window opens.
#[derive(Clone, Debug)]
pub struct Window {
    pub title: String,
    /// Its inner size in physical pixels; 1280 × 720 logical pixels unset.
    pub size: Option<(u32, u32)>,
    pub gpu: GpuPreference,
    pub jobs: JobOptions,
    /// Frames wait for the display's refresh (the default), or show as they
    /// finish, for measurements.
    pub vsync: bool,
    /// The engine switches given, as every host takes them.
    pub settings: HostSettings,
    /// Frames to measure, after which the window closes and [`run`]
    /// reports them; none keeps it open until closed.
    pub measure: Option<Measure>,
}

/// Frames a measured window draws: `frames` of play after `settle`.
#[derive(Clone, Copy, Debug)]
pub struct Measure {
    pub settle: Duration,
    pub frames: u32,
}

/// What a measured window drew.
#[derive(Clone, Debug)]
pub struct WindowReport {
    pub frames: u32,
    /// From the first measured frame's start to the last's end.
    pub elapsed_ms: f64,
    /// Each measured frame's simulation, extraction and render time.
    pub cpu_ms: Vec<f64>,
    /// The last frame's statistics, as JSON.
    pub stats: String,
    pub adapter: engine::AdapterInfo,
    /// The size frames drew at, in physical pixels.
    pub size: (u32, u32),
    /// The window's size, in physical pixels.
    pub window: (u32, u32),
    /// Whether frames drew at the size the resolution policy gives the
    /// window ([`engine::host::view_size`]), which D3D12 and Metal stretch
    /// over it; Vulkan's surfaces take the window's own size.
    pub capped: bool,
}

impl Window {
    /// A window titled `title`, whose `Space` is action 0 as on the page.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            size: None,
            gpu: GpuPreference::default(),
            jobs: JobOptions::default(),
            vsync: true,
            settings: HostSettings::default(),
            measure: None,
        }
    }
}

type Build<'a, G> = Box<dyn FnOnce() -> engine::Result<(Engine, G)> + 'a>;

/// Runs the game `build` makes in `window` until it closes: what it
/// measured, where `window` measures.
pub fn run<'a, G: Game, E: Into<engine::Error>>(
    window: Window,
    build: impl FnOnce() -> Result<(Engine, G), E> + 'a,
) -> engine::Result<Option<WindowReport>> {
    let event_loop =
        EventLoop::new().map_err(|error| host_error("Could not open a window", error))?;
    let mut app = App {
        options: window,
        build: Some(Box::new(move || build().map_err(Into::into))),
        input: Input::default(),
        running: None,
        outcome: Ok(None),
    };
    event_loop
        .run_app(&mut app)
        .map_err(|error| host_error("The window failed", error))?;
    app.outcome
}

struct App<'a, G: Game> {
    options: Window,
    build: Option<Build<'a, G>>,
    input: Input,
    running: Option<Running<G>>,
    outcome: engine::Result<Option<WindowReport>>,
}

/// The game running in its window.
struct Running<G: Game> {
    window: Arc<winit::window::Window>,
    // Tens of kilobytes: boxed, as moves would copy it through the stack.
    host: Box<GameHost<G>>,
    // The window's, in physical pixels.
    size: (u32, u32),
    // Whether frames draw at the resolution policy's size: D3D12 and Metal
    // stretch a smaller surface over the window, Vulkan's takes its size.
    capped: bool,
    title: Title,
    jobs: Option<JobPool>,
    // When play started, and each measured frame's start and CPU time.
    playing: Option<Instant>,
    measured: Vec<(Instant, f64)>,
}

impl<'a, G: Game> App<'a, G> {
    fn start(
        &mut self,
        event_loop: &ActiveEventLoop,
        build: Build<'a, G>,
    ) -> engine::Result<Running<G>> {
        let attributes =
            winit::window::Window::default_attributes().with_title(&self.options.title);
        let attributes = match self.options.size {
            Some((width, height)) => attributes.with_inner_size(PhysicalSize::new(width, height)),
            None => attributes.with_inner_size(LogicalSize::new(1280.0, 720.0)),
        };
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .map_err(|error| host_error("Could not open a window", error))?,
        );
        let size = window.inner_size();
        let size = (size.width, size.height);
        let target = RenderTarget::Window {
            window: window.clone(),
            size,
            gpu: self.options.gpu,
            vsync: self.options.vsync,
        };
        // The startup future holds the renderer's whole start (some 80 KB for
        // the Stress Lab): on the heap, not the calling thread's stack.
        let mut host = Box::new(pollster::block_on(Box::pin(GameHost::launch(
            target, build,
        )))?);
        self.options.settings.apply(&mut host);
        let backend = host.adapter().backend.clone();
        let capped = ["Dx12", "Metal"].contains(&backend.as_str());
        if !capped {
            eprintln!(
                "The resolution policy is not applied on {backend}: its surfaces take the window's size."
            );
        }
        let mut jobs = self.options.jobs.pool();
        if let Some(pool) = &mut jobs {
            pool.attach(&mut host);
        }
        Ok(Running {
            window,
            host,
            size,
            capped,
            title: Title::new(&self.options.title),
            jobs,
            playing: None,
            measured: Vec::new(),
        })
    }
}

impl<G: Game> ApplicationHandler for App<'_, G> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(build) = self.build.take() else {
            return;
        };
        match self.start(event_loop, build) {
            Ok(running) => {
                running.window.request_redraw();
                self.running = Some(running);
            }
            Err(error) => {
                self.outcome = Err(error);
                event_loop.exit();
            }
        }
    }

    #[expect(clippy::wildcard_enum_match_arm, reason = "winit's other events")]
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        let Some(running) = &mut self.running else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => running.size = (size.width, size.height),
            WindowEvent::Focused(false) => {
                self.input.clear();
                running.host.suspend();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key
                    && !event.repeat
                {
                    Input::key(&mut running.host, code, event.state);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => self.input.button(button, state),
            WindowEvent::CursorMoved { position, .. } => {
                let position = position.to_logical::<f64>(running.window.scale_factor());
                self.input
                    .cursor(&mut running.host, (position.x, position.y));
            }
            WindowEvent::RedrawRequested => {
                let start = Instant::now();
                if let Err(error) = running.redraw() {
                    self.outcome = Err(error);
                    event_loop.exit();
                    return;
                }
                if let Some(measure) = self.options.measure
                    && let Some(report) = running.measured(measure, start)
                {
                    self.outcome = Ok(Some(report));
                    event_loop.exit();
                    return;
                }
                running.window.request_redraw();
            }
            _ => {}
        }
    }
}

impl<G: Game> Running<G> {
    /// The size frames draw at, in physical pixels: the resolution
    /// policy's for the window's size and scale factor where the surface
    /// stretches over the window, else the window's.
    fn drawn(&self) -> (u32, u32) {
        let (width, height) = self.size;
        if !self.capped {
            return self.size;
        }
        let scale = self.window.scale_factor();
        engine::host::view_size(
            f64::from(width) / scale,
            f64::from(height) / scale,
            scale,
            1.0,
        )
        .unwrap_or((0, 0))
    }

    fn redraw(&mut self) -> engine::Result<()> {
        let timestamp = engine::host::now_ms();
        let (width, height) = self.drawn();
        let phase = self.host.animation_frame(timestamp, width, height)?;
        self.exchange_jobs();
        if phase == FramePhase::Playing {
            self.title.frame(&self.window, timestamp, &self.host);
            // Terrain loads until shortly before the next frame is due.
            if self.host.load_between_frames() {
                self.exchange_jobs();
            }
        }
        Ok(())
    }

    fn exchange_jobs(&mut self) {
        if let Some(jobs) = &mut self.jobs {
            jobs.exchange(&mut self.host);
        }
    }

    /// Counts the frame that started at `start` toward `measure` once play
    /// settled: the report, once every frame was measured.
    fn measured(&mut self, measure: Measure, start: Instant) -> Option<WindowReport> {
        if self.host.preparing() {
            return None;
        }
        let playing = *self.playing.get_or_insert(start);
        if start.duration_since(playing) < measure.settle {
            return None;
        }
        let stats = self.host.frame_stats();
        self.measured.push((
            start,
            stats.simulation_ms + stats.extraction_ms + stats.render_ms,
        ));
        if self.measured.len() < measure.frames as usize {
            return None;
        }
        Some(WindowReport {
            frames: measure.frames,
            elapsed_ms: self.measured[0].0.elapsed().as_secs_f64() * 1000.0,
            cpu_ms: self.measured.iter().map(|&(_, cpu)| cpu).collect(),
            stats: self.host.stats(),
            adapter: self.host.adapter().clone(),
            size: self.drawn(),
            window: self.size,
            capped: self.capped,
        })
    }
}

/// The window's title with its recent frame rate and CPU time per frame.
struct Title {
    name: String,
    since: Option<f64>,
    frames: u32,
    cpu_ms: f64,
}

impl Title {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            since: None,
            frames: 0,
            cpu_ms: 0.0,
        }
    }

    fn frame<G: Game>(
        &mut self,
        window: &winit::window::Window,
        timestamp: f64,
        host: &GameHost<G>,
    ) {
        let stats = host.frame_stats();
        self.frames += 1;
        self.cpu_ms += stats.simulation_ms + stats.extraction_ms + stats.render_ms;
        let since = *self.since.get_or_insert(timestamp);
        let elapsed = timestamp - since;
        if elapsed >= 500.0 {
            let frames = f64::from(self.frames);
            window.set_title(&format!(
                "{} · {:.0} fps · CPU {:.2} ms",
                self.name,
                frames * 1000.0 / elapsed,
                self.cpu_ms / frames
            ));
            *self = Self::new(&self.name);
            self.since = Some(timestamp);
        }
    }
}
