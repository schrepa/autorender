use std::sync::Arc;
use std::time::{Duration, Instant};

use engine::host::{FramePhase, HostPause};
use engine::stats::GpuTime;
use engine::{BrickList, Engine, ErrorCode, Game, GameHost, RenderTarget};
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, StartCause, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;

use crate::host_error;
use crate::input::Input;
use crate::jobs::JobPool;
use crate::options::{HostOptions, StartScene};

#[derive(Clone, Debug)]
pub struct Window {
    pub title: String,
    pub options: HostOptions,
    pub scene: StartScene,
}

type Build<'a, G> = (
    Box<dyn FnOnce(BrickList) -> BrickList + 'a>,
    Box<dyn FnOnce(&mut Engine) -> engine::Result<G> + 'a>,
);

pub fn run<'a, G: Game, E: Into<engine::Error>>(
    window: Window,
    bricks: impl FnOnce(BrickList) -> BrickList + 'a,
    build: impl FnOnce(&mut Engine) -> Result<G, E> + 'a,
) -> engine::Result<()> {
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
        outcome: Ok(()),
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
    outcome: engine::Result<()>,
}

struct Running<G: Game> {
    window: Arc<winit::window::Window>,
    host: Box<GameHost<G>>,
    size: (u32, u32),
    title: Title,
    jobs: JobPool,
    #[cfg(feature = "dev")]
    session: crate::dev::Session,
}

impl<'a, G: Game> App<'a, G> {
    fn start(
        &mut self,
        event_loop: &ActiveEventLoop,
        (bricks, build): Build<'a, G>,
    ) -> engine::Result<Running<G>> {
        let options = &self.options.options;
        let mut attributes =
            winit::window::Window::default_attributes().with_title(&self.options.title);
        if let Some((width, height)) = options.size {
            attributes = attributes.with_inner_size(PhysicalSize::new(width, height));
        }
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
            gpu: options.gpu,
            vsync: options.vsync,
        };
        let mut host = Box::new(pollster::block_on(Box::pin(GameHost::launch(
            target, bricks, build,
        )))?);
        host.set_pixel_ratio(window.scale_factor());
        host.set_refresh_interval(refresh_interval(&window));
        crate::open_scene(&mut host, &self.options.scene)?;
        options.settings.apply(&mut host)?;
        let mut jobs = options.jobs.pool(false);
        jobs.attach(&mut host);
        #[cfg(feature = "dev")]
        let session =
            options.session(&mut host, size, &self.options.scene, ("window", jobs.packed()))?;
        self.input.connect_pads();
        Ok(Running {
            window,
            host,
            size,
            title: Title::new(&self.options.title),
            jobs,
            #[cfg(feature = "dev")]
            session,
        })
    }
}

impl<G: Game> ApplicationHandler for App<'_, G> {
    fn suspended(&mut self, _: &ActiveEventLoop) {
        if let Some(running) = &mut self.running {
            running.host.pause(HostPause::Suspended);
        }
    }

    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let Some(build) = self.build.take() else {
            if let Some(running) = &mut self.running {
                running.host.resume(HostPause::Suspended);
                running.window.request_redraw();
            }
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
            WindowEvent::Moved(_) => self.input.regrab(),
            WindowEvent::Resized(size) => {
                self.input.regrab();
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
                running.host.pause(HostPause::Focus);
            }
            WindowEvent::Focused(true) => running.host.resume(HostPause::Focus),
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key
                    && !event.repeat
                {
                    Input::key(&mut running.host, code, event.state);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                Input::button(&mut running.host, button, state);
            }
            WindowEvent::MouseWheel { delta, .. } => {
                Input::wheel(&mut running.host, delta, running.window.scale_factor());
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.input.cursor(
                    &mut running.host,
                    (position.x, position.y),
                    running.window.scale_factor(),
                );
            }
            WindowEvent::CursorLeft { .. } => self.input.left(&mut running.host),
            WindowEvent::Touch(touch) => {
                self.input
                    .touch(&mut running.host, touch, running.window.scale_factor());
            }
            WindowEvent::RedrawRequested => {
                let start = Instant::now();
                self.input.poll_pads(&mut running.host, &running.window);
                #[cfg(feature = "dev")]
                if !running.session.receive(&mut running.host, Duration::ZERO) {
                    event_loop.exit();
                }
                let phase = match running.redraw() {
                    Ok(phase) => phase,
                    Err(error) => {
                        self.outcome = Err(error);
                        event_loop.exit();
                        return;
                    }
                };
                self.input.apply_cursor(&running.host, &running.window);
                if phase == FramePhase::Idle {
                    return;
                }
                match phase {
                    FramePhase::Compiling | FramePhase::Preparing | FramePhase::Paused => {
                        let interval = running.host.refresh_interval() / 1000.0;
                        let wait = Duration::try_from_secs_f64(interval).unwrap_or_default();
                        event_loop.set_control_flow(ControlFlow::WaitUntil(start + wait));
                    }
                    FramePhase::Collecting
                    | FramePhase::Started
                    | FramePhase::Playing
                    | FramePhase::Idle => running.window.request_redraw(),
                }
            }
            _ => {}
        }
    }

    fn new_events(&mut self, event_loop: &ActiveEventLoop, cause: StartCause) {
        if let (StartCause::ResumeTimeReached { .. }, Some(running)) = (cause, &self.running) {
            event_loop.set_control_flow(ControlFlow::Wait);
            running.window.request_redraw();
        }
    }

    fn device_event(&mut self, _: &ActiveEventLoop, _: DeviceId, event: DeviceEvent) {
        if let DeviceEvent::MouseMotion { delta } = event
            && let Some(running) = &mut self.running
        {
            Input::motion(&mut running.host, delta);
        }
    }
}

impl<G: Game> Running<G> {
    fn redraw(&mut self) -> engine::Result<FramePhase> {
        #[cfg(feature = "dev")]
        crate::dev::follow_source(&mut self.host);
        let timestamp = engine::host::now_ms();
        let (width, height) = self.size;
        self.exchange_jobs();
        let phase = self.host.animation_frame(timestamp, width, height)?;
        self.exchange_jobs();
        #[cfg(feature = "dev")]
        self.session.after(&mut self.host)?;
        match phase {
            FramePhase::Playing => self.title.frame(&self.window, &self.host),
            FramePhase::Compiling | FramePhase::Preparing | FramePhase::Collecting => {
                self.title.loading(&self.window);
            }
            FramePhase::Started | FramePhase::Idle | FramePhase::Paused => {}
        }
        Ok(phase)
    }

    fn exchange_jobs(&mut self) {
        self.jobs.exchange(&mut self.host);
    }
}

fn refresh_interval(window: &winit::window::Window) -> Option<f64> {
    let millihertz = window.current_monitor()?.refresh_rate_millihertz()?;
    (millihertz > 0).then(|| 1_000_000.0 / f64::from(millihertz))
}

struct Title {
    name: String,
    loading: bool,
}

impl Title {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            loading: false,
        }
    }

    fn loading(&mut self, window: &winit::window::Window) {
        if !self.loading {
            window.set_title(&format!("{} · loading, the world is paused", self.name));
            self.loading = true;
        }
    }

    fn frame<G: Game>(&mut self, window: &winit::window::Window, host: &GameHost<G>) {
        self.loading = false;
        let stats = host.frame_times();
        let gpu = match stats.resolution.gpu_ms {
            GpuTime::Ms(ms) => format!("{ms:.2} ms"),
            GpuTime::Unavailable { unavailable } => unavailable.to_owned(),
        };
        window.set_title(&format!(
            "{} · frame {} · simulation {:.2} ms · extraction {:.2} ms · render {:.2} ms · GPU {gpu}",
            self.name, stats.frame, stats.simulation_ms, stats.extraction_ms, stats.render_ms,
        ));
    }
}
