use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::host::{FramePhase, HostSettings};
use engine::{BrickList, Engine, ErrorCode, Game, GameHost, GpuPreference, RenderTarget};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;

use crate::host_error;
use crate::input::Input;
use crate::jobs::{JobOptions, JobPool};

#[derive(Clone, Debug)]
pub struct Window {
    pub title: String,
    pub size: Option<(u32, u32)>,
    pub gpu: GpuPreference,
    pub jobs: JobOptions,
    pub vsync: bool,
    pub settings: HostSettings,
    pub measure: Option<Measure>,
}

#[derive(Clone, Copy, Debug)]
pub struct Measure {
    pub settle: Duration,
    pub frames: u32,
}

#[derive(Clone, Debug)]
pub struct WindowReport {
    pub frames: u32,
    pub elapsed_ms: f64,
    pub cpu_ms: Vec<f64>,
    pub stats: String,
    pub adapter: engine::AdapterInfo,
    pub size: (u32, u32),
    pub window: (u32, u32),
}

impl Window {
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

type Build<'a, G> = (
    Box<dyn FnOnce(BrickList) -> BrickList + 'a>,
    Box<dyn FnOnce(&mut Engine) -> engine::Result<G> + 'a>,
);

pub fn run<'a, G: Game, E: Into<engine::Error>>(
    window: Window,
    bricks: impl FnOnce(BrickList) -> BrickList + 'a,
    build: impl FnOnce(&mut Engine) -> Result<G, E> + 'a,
) -> engine::Result<Option<WindowReport>> {
    let event_loop = EventLoop::new()
        .map_err(|error| host_error(ErrorCode::RenderFailed, "$.event_loop", error))?;
    let mut app = App {
        options: window,
        build: Some((
            Box::new(bricks),
            Box::new(move |engine: &mut Engine| build(engine).map_err(Into::<engine::Error>::into)),
        )),
        input: Input::default(),
        running: None,
        outcome: Ok(None),
    };
    event_loop
        .run_app(&mut app)
        .map_err(|error| host_error(ErrorCode::RenderFailed, "$.event_loop", error))?;
    app.outcome
}

struct App<'a, G: Game> {
    options: Window,
    build: Option<Build<'a, G>>,
    input: Input,
    running: Option<Running<G>>,
    outcome: engine::Result<Option<WindowReport>>,
}

struct Running<G: Game> {
    window: Arc<winit::window::Window>,
    host: Box<GameHost<G>>,
    size: (u32, u32),
    title: Title,
    jobs: JobPool,
    playing: Option<Instant>,
    measured: Vec<(Instant, f64)>,
}

impl<'a, G: Game> App<'a, G> {
    fn start(
        &mut self,
        event_loop: &ActiveEventLoop,
        (bricks, build): Build<'a, G>,
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
                .map_err(|error| host_error(ErrorCode::RenderFailed, "$.window", error))?,
        );
        let size = window.inner_size();
        let size = (size.width, size.height);
        let target = RenderTarget::Window {
            window: window.clone(),
            size,
            gpu: self.options.gpu,
            vsync: self.options.vsync,
        };
        let mut host = Box::new(pollster::block_on(Box::pin(GameHost::launch(
            target, bricks, build,
        )))?);
        host.set_pixel_ratio(window.scale_factor());
        host.set_refresh_interval(refresh_interval(&window));
        host.set_setting(
            crate::DYNAMIC_RESOLUTION,
            self.options.measure.is_none().into(),
        )?;
        self.options.settings.apply(&mut host)?;
        let mut jobs = self.options.jobs.pool(false);
        jobs.attach(&mut host);
        Ok(Running {
            window,
            host,
            size,
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
            WindowEvent::Resized(size) => {
                let idle = running.size.0 == 0 || running.size.1 == 0;
                running.size = (size.width, size.height);
                if idle {
                    running.window.request_redraw();
                }
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                running.host.set_pixel_ratio(scale_factor);
                running
                    .host
                    .set_refresh_interval(refresh_interval(&running.window));
            }
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
                let phase = match running.redraw() {
                    Ok(phase) => phase,
                    Err(error) => {
                        self.outcome = Err(error);
                        event_loop.exit();
                        return;
                    }
                };
                if phase == FramePhase::Idle {
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
    fn redraw(&mut self) -> engine::Result<FramePhase> {
        let timestamp = engine::host::now_ms();
        let (width, height) = self.size;
        let phase = self.host.animation_frame(timestamp, width, height)?;
        self.exchange_jobs();
        if phase == FramePhase::Playing {
            self.title.frame(&self.window, timestamp, &self.host);
            if self.host.load_between_frames() {
                self.exchange_jobs();
            }
        }
        Ok(phase)
    }

    fn exchange_jobs(&mut self) {
        self.jobs.exchange(&mut self.host);
    }

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
        let resolution = stats.resolution;
        Some(WindowReport {
            frames: measure.frames,
            elapsed_ms: self.measured[0].0.elapsed().as_secs_f64() * 1000.0,
            cpu_ms: self.measured.iter().map(|&(_, cpu)| cpu).collect(),
            stats: self.host.stats(),
            adapter: self.host.adapter().clone(),
            size: (resolution.render_width, resolution.render_height),
            window: self.size,
        })
    }
}

fn refresh_interval(window: &winit::window::Window) -> Option<f64> {
    let millihertz = window.current_monitor()?.refresh_rate_millihertz()?;
    (millihertz > 0).then(|| 1_000_000.0 / f64::from(millihertz))
}

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
