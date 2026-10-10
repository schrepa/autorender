//! A game in a native window: frames as the page draws them, one each time
//! the display can show one, with keyboard and mouse input, and terrain
//! loading between frames.

use std::sync::Arc;
use std::time::Instant;

use engine::{Engine, Game, GameHost, GpuPreference, InputState, RenderTarget};
use winit::application::ApplicationHandler;
use winit::dpi::{LogicalSize, PhysicalSize};
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::keyboard::PhysicalKey;
use winit::window::WindowId;

use crate::input::Input;
use crate::jobs::{JobOptions, JobPool};
use crate::{Preparation, host_error};

/// How a game's window opens.
#[derive(Clone, Debug)]
pub struct Window {
    pub title: String,
    /// Its inner size in physical pixels; 1280 × 720 logical pixels unset.
    pub size: Option<(u32, u32)>,
    /// The game's action keys: the page's key names (`Space`, `KeyF`...)
    /// by action index.
    pub actions: Vec<(String, u32)>,
    pub gpu: GpuPreference,
    pub jobs: JobOptions,
}

impl Window {
    /// A window titled `title`, whose `Space` is action 0 as on the page.
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            size: None,
            actions: vec![("Space".into(), 0)],
            gpu: GpuPreference::default(),
            jobs: JobOptions::default(),
        }
    }
}

type Build<'a, G> = Box<dyn FnOnce() -> engine::Result<(Engine, G)> + 'a>;

/// Runs the game `build` makes in `window` until it closes.
pub fn run<'a, G: Game, E: Into<engine::Error>>(
    window: Window,
    build: impl FnOnce() -> Result<(Engine, G), E> + 'a,
) -> engine::Result<()> {
    let actions: Vec<(&str, u32)> = window
        .actions
        .iter()
        .map(|(key, action)| (key.as_str(), *action))
        .collect();
    let input = Input::new(&actions)?;
    let event_loop =
        EventLoop::new().map_err(|error| host_error("Could not open a window", error))?;
    let mut app = App {
        options: window,
        build: Some(Box::new(move || build().map_err(Into::into))),
        input,
        running: None,
        outcome: Ok(()),
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
    outcome: engine::Result<()>,
}

/// The game running in its window.
struct Running<G: Game> {
    window: Arc<winit::window::Window>,
    host: GameHost<G>,
    // Physical pixels.
    size: (u32, u32),
    started: Instant,
    // Until play starts.
    preparation: Option<Preparation>,
    pace: Pace,
    title: Title,
    jobs: Option<JobPool>,
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
        };
        let mut host = pollster::block_on(GameHost::launch(target, build))?;
        let mut jobs = self.options.jobs.pool();
        if let Some(pool) = &mut jobs {
            pool.attach(&mut host);
        }
        Ok(Running {
            window,
            host,
            size,
            started: Instant::now(),
            preparation: Some(Preparation::new()),
            pace: Pace::default(),
            title: Title::new(&self.options.title),
            jobs,
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
                running.pace.skip();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    self.input.key(code, event.state);
                }
            }
            WindowEvent::MouseInput { state, button, .. } => self.input.button(button, state),
            WindowEvent::CursorMoved { position, .. } => {
                let position = position.to_logical::<f64>(running.window.scale_factor());
                self.input.cursor((position.x, position.y));
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = running.redraw(&mut self.input) {
                    self.outcome = Err(error);
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
    /// Milliseconds since the game started.
    fn now(&self) -> f64 {
        self.started.elapsed().as_secs_f64() * 1000.0
    }

    fn redraw(&mut self, input: &mut Input) -> engine::Result<()> {
        let timestamp = self.now();
        let (width, height) = self.size;
        if let Some(preparation) = &mut self.preparation {
            if preparation.frame(&mut self.host, timestamp, self.size)? {
                self.preparation = None;
                self.pace.skip();
            }
            self.exchange_jobs();
            return Ok(());
        }
        let look = InputState {
            move_forward: input.forward(),
            move_right: input.right(),
            move_up: input.up(),
            look_yaw: input.yaw,
            look_pitch: input.pitch,
        };
        self.host.frame(
            timestamp,
            width,
            height,
            look,
            input.held,
            input.pressed,
            input.released,
        )?;
        input.consumed();
        self.exchange_jobs();
        self.title.frame(&self.window, timestamp, &self.host);
        // Terrain loads until shortly before the next frame is due.
        let spare = self.pace.spare(timestamp, self.now());
        if spare > 1.0 {
            let start = self.now();
            self.host.work_between_frames(spare);
            self.exchange_jobs();
            self.pace.spent(self.now() - start);
        }
        Ok(())
    }

    fn exchange_jobs(&mut self) {
        if let Some(jobs) = &mut self.jobs {
            jobs.exchange(&mut self.host);
        }
    }
}

/// When the next frame is due, as the page reckons it: the recent interval
/// between frames, without the time spent loading between them.
struct Pace {
    last: Option<f64>,
    interval: f64,
    spent: f64,
}

impl Default for Pace {
    fn default() -> Self {
        Self {
            last: None,
            interval: 1000.0 / 60.0,
            spent: 0.0,
        }
    }
}

// What loading between frames leaves before the next one is due.
const MARGIN_MS: f64 = 4.0;

impl Pace {
    /// Milliseconds to load in before the frame after the one that
    /// started at `timestamp` is due, at `now`.
    fn spare(&mut self, timestamp: f64, now: f64) -> f64 {
        if let Some(last) = self.last {
            let interval = timestamp - last - self.spent;
            if interval > 0.0 && interval < 100.0 {
                self.interval += (interval - self.interval) * 0.1;
            }
        }
        self.last = Some(timestamp);
        self.spent = 0.0;
        timestamp + self.interval - now - MARGIN_MS
    }

    fn spent(&mut self, milliseconds: f64) {
        self.spent = milliseconds;
    }

    /// The next interval starts afresh: after a pause or preparation.
    fn skip(&mut self) {
        self.last = None;
        self.spent = 0.0;
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
