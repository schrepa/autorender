//! The frame loop every host runs, a page or a native window: one engine
//! and the renderer drawing it, advanced a frame at a time with the host's
//! input, terrain tiles read by byte range and jobs handed to the host's
//! workers.
use engine_core::{
    Engine, ErrorCode, FrameReport, FrameStats, InputState, JobId, JobRequest, JobShare,
    PresentationMode,
};
use engine_render::{RenderError, RenderTarget, Renderer};

use crate::{Error, Result};

#[cfg(feature = "profiling")]
#[path = "app_profiling.rs"]
mod profiling;

/// Milliseconds on the host's clock, which terrain times its work by and
/// profiling captures time events by: the page's (0 without one), or
/// natively since the first reading.
pub fn now_ms() -> f64 {
    #[cfg(target_arch = "wasm32")]
    {
        thread_local! {
            static CLOCK: Option<web_sys::Performance> =
                web_sys::window().and_then(|window| window.performance());
        }
        CLOCK.with(|clock| clock.as_ref().map_or(0.0, web_sys::Performance::now))
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START
            .get_or_init(std::time::Instant::now)
            .elapsed()
            .as_secs_f64()
            * 1000.0
    }
}

/// Terrain tiles in flight: read by byte range from their files and handed
/// back to the engine's terrains the next frame.
#[derive(Default)]
struct TerrainFetches {
    inbox: std::rc::Rc<std::cell::RefCell<Vec<TerrainArrival>>>,
    requests: Vec<engine_core::TileRequest>,
}

/// A read tile: its terrain, key and bytes, `None` when the read failed.
type TerrainArrival = (
    engine_core::bevy_ecs::entity::Entity,
    engine_core::TileKey,
    Option<Vec<u8>>,
);

impl TerrainFetches {
    /// Hands arrived tiles to `engine`'s terrains and starts reading the
    /// tiles they request.
    fn exchange(&mut self, engine: &mut Engine) {
        let Some(mut terrains) = engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Terrains>()
        else {
            return;
        };
        for (terrain, key, bytes) in self.inbox.borrow_mut().drain(..) {
            terrains.deliver(terrain, key, bytes);
        }
        terrains.take_requests(&mut self.requests);
        for request in self.requests.drain(..) {
            let range = request.offset..request.offset + u64::from(request.length);
            let inbox = std::rc::Rc::clone(&self.inbox);
            engine_render::read_range(&request.url, range, move |bytes| {
                inbox
                    .borrow_mut()
                    .push((request.terrain, request.key, bytes.ok()));
            });
        }
    }
}

/// What a host hands its workers, in the order
/// [`GameHost::take_jobs`](crate::GameHost::take_jobs) lists
/// them.
#[derive(Debug)]
pub enum JobMessage {
    /// A job handed out that its owner cancelled, which its worker stops.
    Cancel(JobId),
    /// Bytes every worker holds by key, or drops.
    Share(JobShare),
    /// A job for one worker.
    Run(JobRequest),
}

/// What terrain times its work by on a host.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainClock {
    /// The host's clock: each frame works on terrain for the settings' work
    /// time, and hosts load more in the time frames leave free.
    #[default]
    Wall,
    /// Frames alone: each works one slice of the settings'
    /// `samples_per_frame` and none loads between frames, so what streams
    /// depends on the frames drawn, never on how long they took.
    Frames,
}

impl TerrainClock {
    /// The clock terrain reads: none for frames alone.
    fn reading(self) -> Option<fn() -> f64> {
        match self {
            Self::Wall => Some(now_ms),
            Self::Frames => None,
        }
    }
}

/// Wires `engine` to the host `renderer` draws for: terrain residency never
/// exceeds the device's texture array layers and its work runs by `clock`;
/// jobs run on the host's `workers`.
fn wire(engine: &mut Engine, renderer: &Renderer, workers: u32, clock: TerrainClock) {
    let world = engine.app.world_mut();
    if let Some(mut terrains) = world.get_resource_mut::<engine_core::Terrains>() {
        terrains.set_device_tiles(Some(renderer.terrain_tile_capacity()));
        terrains.set_clock(clock.reading());
    }
    if let Some(mut jobs) = world.get_resource_mut::<engine_core::Jobs>() {
        jobs.set_workers(workers);
    }
}

/// Owns the runtime, extracted data, and GPU state for one canvas or
/// window.
pub(crate) struct AppHost {
    // Replaced only through `replace_engine`, which wires each engine to
    // this host.
    pub(crate) engine: Engine,
    // Tens of kilobytes: boxed, as moves of hosts would copy it through the
    // stack (native hosts' main threads have a megabyte).
    pub(crate) renderer: Box<Renderer>,
    last_timestamp_ms: Option<f64>,
    stats: FrameStats,
    size: (u32, u32),
    // Ordinary interpolated frames blend, compose and cull on the GPU.
    gpu_presentation: bool,
    // GPU-presented frames start the next fixed tick ahead of its due frame.
    tick_pipelining: bool,
    // The last GPU frame was not submitted, so the renderer missed its gather.
    gpu_inputs_stale: bool,
    // Terrain tiles read by byte range.
    terrain_fetches: TerrainFetches,
    terrain_clock: TerrainClock,
    // The host's culling and antialiasing choices over each engine's own
    // settings: unset, an engine keeps its own.
    overrides: Overrides,
    #[cfg(feature = "profiling")]
    profile_frame: u64,
}

#[derive(Clone, Copy, Default)]
struct Overrides {
    temporal_antialiasing: Option<bool>,
    occlusion_culling: Option<bool>,
    subpixel_culling: Option<bool>,
}

/// What a submitted frame shows. Only development controls submit exact
/// frames and previews.
#[derive(Clone, Copy)]
#[cfg_attr(
    not(feature = "dev"),
    expect(
        dead_code,
        reason = "only development controls submit exact frames and previews"
    )
)]
enum FrameKind {
    /// An ordinary frame at fraction `alpha` of the fixed tick: GPU-presented
    /// unless the game opted out, and blended by temporal antialiasing.
    Display { alpha: f32 },
    /// The exact current state: steps, renders and captures.
    Exact,
    /// Fraction `alpha` of the latest completed tick, presented on the GPU
    /// when `gpu`, else exactly.
    Preview { alpha: f32, gpu: bool },
}

/// The engine `build` makes, in a frame of its own: games' builds hold
/// large temporaries, which inlined into the launch's poll would hold the
/// stack while the renderer starts.
#[inline(never)]
fn built(build: impl FnOnce() -> Result<Engine>) -> Result<Engine> {
    build()
}

impl AppHost {
    /// Starts the renderer presenting to `target` and, while the GPU
    /// compiles its startup pipelines, the engine `build` makes.
    pub(crate) async fn launch(
        target: RenderTarget,
        build: impl FnOnce() -> Result<Engine>,
    ) -> Result<Self> {
        let starting = Renderer::start(target).await.map_err(render_failure)?;
        let engine = built(build)?;
        let renderer = Box::new(starting.finish().await.map_err(render_failure)?);
        Ok(Self::with(engine, renderer))
    }

    fn with(mut engine: Engine, renderer: Box<Renderer>) -> Self {
        wire(&mut engine, &renderer, 0, TerrainClock::Wall);
        Self {
            engine,
            renderer,
            last_timestamp_ms: None,
            stats: FrameStats::default(),
            size: (0, 0),
            gpu_presentation: true,
            tick_pipelining: true,
            gpu_inputs_stale: false,
            terrain_fetches: TerrainFetches::default(),
            terrain_clock: TerrainClock::Wall,
            overrides: Overrides::default(),
            #[cfg(feature = "profiling")]
            profile_frame: 0,
        }
    }

    /// The running engine.
    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// The running engine, to change its world.
    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    /// Runs `engine` from the next frame, wired to this host as the first
    /// was; returns the replaced one.
    pub fn replace_engine(&mut self, mut engine: Engine) -> Engine {
        let workers = self
            .engine
            .app
            .world()
            .get_resource::<engine_core::Jobs>()
            .map_or(0, engine_core::Jobs::workers);
        wire(&mut engine, &self.renderer, workers, self.terrain_clock);
        // Tiles in flight belong to the replaced engine's terrains.
        self.terrain_fetches = TerrainFetches::default();
        self.gpu_inputs_stale = false;
        let replaced = std::mem::replace(&mut self.engine, engine);
        self.apply_overrides();
        self.suspend();
        replaced
    }

    /// Sets the host's culling and antialiasing choices over the world's
    /// own settings, which a new engine or scene brought.
    pub fn apply_overrides(&mut self) {
        let Overrides {
            temporal_antialiasing,
            occlusion_culling,
            subpixel_culling,
        } = self.overrides;
        let world = self.engine.app.world_mut();
        if let Some(enabled) = temporal_antialiasing {
            world.insert_resource(engine_core::TemporalAntialiasing { enabled });
        }
        if let Some(enabled) = occlusion_culling {
            world.insert_resource(engine_core::OcclusionCulling { enabled });
        }
        if let Some(enabled) = subpixel_culling {
            world.insert_resource(engine_core::SubpixelCulling { enabled });
        }
    }

    /// What terrain times its work by, in this and later engines: the
    /// host's clock (the default), or frames alone.
    pub fn set_terrain_clock(&mut self, clock: TerrainClock) {
        self.terrain_clock = clock;
        if let Some(mut terrains) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Terrains>()
        {
            terrains.set_clock(clock.reading());
        }
    }

    /// Selects GPU blending, composition and culling for ordinary interpolated
    /// frames (the default) or the exact CPU path. Development
    /// steps, renders and captures always use the exact CPU path.
    pub fn set_gpu_presentation(&mut self, enabled: bool) {
        self.gpu_presentation = enabled;
        if !enabled {
            self.engine.complete_pending_tick();
        }
    }

    pub fn gpu_presentation(&self) -> bool {
        self.gpu_presentation
    }

    /// Splits each fixed tick across the frames around its due time when GPU
    /// presentation is enabled (the default): after a frame is submitted, the
    /// next tick's gameplay stage starts if the tick will be due by the next
    /// frame, and its propagation stage runs when due. See
    /// [`Engine::run_ahead`]. Disabling completes a started tick.
    pub fn set_tick_pipelining(&mut self, enabled: bool) {
        self.tick_pipelining = enabled;
        if !enabled {
            self.engine.complete_pending_tick();
        }
    }

    pub fn tick_pipelining(&self) -> bool {
        self.tick_pipelining
    }

    /// Temporally antialiases ordinary GPU-presented frames, in this and
    /// later engines: sets the world's [`engine_core::TemporalAntialiasing`]
    /// settings (a scene environment's `temporal_antialiasing` section),
    /// which the next frame follows. Off by default.
    pub fn set_temporal_antialiasing(&mut self, enabled: bool) {
        self.overrides.temporal_antialiasing = Some(enabled);
        self.apply_overrides();
    }

    pub fn temporal_antialiasing(&self) -> bool {
        self.engine
            .app
            .world()
            .get_resource::<engine_core::TemporalAntialiasing>()
            .is_some_and(|settings| settings.enabled)
    }

    /// Occlusion culling of GPU-presented frames, in this and later engines:
    /// sets the world's [`engine_core::OcclusionCulling`] settings (a scene
    /// environment's `occlusion_culling` section). On by default.
    pub fn set_occlusion_culling(&mut self, enabled: bool) {
        self.overrides.occlusion_culling = Some(enabled);
        self.apply_overrides();
    }

    pub fn occlusion_culling(&self) -> bool {
        self.engine
            .app
            .world()
            .get_resource::<engine_core::OcclusionCulling>()
            .is_some_and(|settings| settings.enabled)
    }

    /// Sub-pixel culling of GPU-presented frames, in this and later engines:
    /// sets the world's [`engine_core::SubpixelCulling`] settings (a scene
    /// environment's `subpixel_culling` section). On by default.
    pub fn set_subpixel_culling(&mut self, enabled: bool) {
        self.overrides.subpixel_culling = Some(enabled);
        self.apply_overrides();
    }

    pub fn subpixel_culling(&self) -> bool {
        self.engine
            .app
            .world()
            .get_resource::<engine_core::SubpixelCulling>()
            .is_some_and(|settings| settings.enabled)
    }

    /// Streams terrains for a frame's work time on a `width` × `height`
    /// view without simulating or rendering: hosts call it
    /// in place of frames while the GPU compiles pipelines, which frames
    /// would only queue behind, so loading goes on meanwhile.
    pub fn work(&mut self, width: u32, height: u32) {
        let viewport = engine_core::Viewport {
            width: width.max(1),
            height: height.max(1),
        };
        let world = self.engine.app.world_mut();
        if world.get_resource::<engine_core::Viewport>() != Some(&viewport) {
            world.insert_resource(viewport);
        }
        self.terrain_fetches.exchange(&mut self.engine);
        self.engine.stream_terrain();
    }

    /// Loads terrain for `milliseconds` between frames: hosts
    /// call it after a frame with the time left before the next one is due,
    /// while the last frame found tiles to load. Returns whether tiles are
    /// still to load.
    pub fn work_between_frames(&mut self, milliseconds: f64) -> bool {
        if !milliseconds.is_finite() || milliseconds <= 0.0 {
            return false;
        }
        self.terrain_fetches.exchange(&mut self.engine);
        self.engine.work_terrain(milliseconds)
    }

    /// Workers the host runs jobs on: with any, terrains bake
    /// and generate tiles and agents search walks there; with none, the
    /// default, that work stays in the engine, which takes back every job
    /// handed out.
    pub fn set_job_workers(&mut self, workers: u32) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.set_workers(workers);
        }
    }

    /// What to hand workers since the last call, in order: cancels of jobs
    /// handed out that their owners cancelled, which their workers stop;
    /// shares (clearing every share an earlier engine made, holding bytes
    /// as a key, dropping one), to every worker; then jobs, each to one,
    /// lower priorities first.
    pub fn take_jobs(&mut self) -> Vec<JobMessage> {
        let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        else {
            return Vec::new();
        };
        let mut cancels = Vec::new();
        let mut shares = Vec::new();
        let mut requests = Vec::new();
        jobs.take_cancels(&mut cancels);
        jobs.take_shares(&mut shares);
        jobs.take_requests(&mut requests);
        let cancels = cancels.into_iter().map(JobMessage::Cancel);
        let shares = shares.into_iter().map(JobMessage::Share);
        let requests = requests.into_iter().map(JobMessage::Run);
        cancels.chain(shares).chain(requests).collect()
    }

    /// Job `id`'s output from its worker.
    pub fn complete_job(&mut self, id: JobId, output: Vec<u8>) {
        self.finish_job(id, Ok(output));
    }

    /// Job `id` failed on its worker, for `error`: its owner
    /// takes the work back.
    pub fn fail_job(&mut self, id: JobId, error: String) {
        self.finish_job(id, Err(error));
    }

    /// The host's output cache's counters: outputs handed
    /// back from it, jobs it lacked, its bytes and, verifying, outputs that
    /// differed from a fresh run.
    pub fn report_job_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.report_cache(hits, misses, bytes, mismatches);
        }
    }

    /// A part of job `id`'s output its worker handed back before the rest,
    /// such as a tile's shape.
    pub fn part_job(&mut self, id: JobId, part: Vec<u8>) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.part(id, part);
        }
    }

    fn finish_job(&mut self, id: JobId, output: std::result::Result<Vec<u8>, String>) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.complete(id, output);
        }
    }

    /// Advance once per animation frame; input is a bounded, owned value.
    pub fn frame(
        &mut self,
        timestamp_ms: f64,
        width: u32,
        height: u32,
        input: InputState,
    ) -> Result<()> {
        if !timestamp_ms.is_finite() {
            return Err(Error::new(
                ErrorCode::InvalidInput,
                "$.timestamp",
                "Frame timestamp must be finite.",
            ));
        }
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let _profile = engine_profile::span("cpu", "host.frame");
        let delta = self
            .last_timestamp_ms
            .map_or(0.0, |last| (timestamp_ms - last).max(0.0) * 0.001);
        self.last_timestamp_ms = Some(timestamp_ms);
        self.set_input(input);
        let start = now_ms();
        let report = self.engine.advance(delta);
        let simulation_ms = now_ms() - start;
        self.render_frame(
            width,
            height,
            simulation_ms,
            report,
            FrameKind::Display {
                alpha: report.interpolation,
            },
        )?;
        // After submission, so the GPU draws this frame while the next tick's
        // first stage runs; the frame interval predicts the next one.
        if self.gpu_presentation && self.tick_pipelining {
            let start = now_ms();
            if self.engine.run_ahead(delta) {
                // A tick just completed ahead: stage its current poses now, so
                // the frame that gathers them only records their GPU copy.
                if self.engine.tick_completed_ahead()
                    && let Some(current) = self.engine.gpu_current()
                {
                    self.renderer.prestage_gpu_current(current);
                }
                self.stats.simulation_ms += now_ms() - start;
            }
        }
        Ok(())
    }

    /// Runs exact fixed ticks and submits the resulting state while paused.
    ///
    /// Uses [`Engine::step_fixed`]'s count and presentation contract. Rejected
    /// tick counts leave the engine, input, timestamp, and renderer untouched.
    /// Merge action input before this call, after validating the count. This
    /// method preserves those edges and clears the real-time timestamp baseline
    /// so the next ordinary frame excludes time spent in development control.
    #[cfg(feature = "dev")]
    pub fn dev_step(
        &mut self,
        ticks: u32,
        width: u32,
        height: u32,
        input: InputState,
    ) -> Result<()> {
        let invalid = |error: engine_core::DevStepError| {
            Error::new(ErrorCode::InvalidStep, "$.ticks", error.to_string())
        };
        engine_core::DevStepError::validate(ticks).map_err(invalid)?;
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let _profile = engine_profile::span("cpu", "host.step");
        self.engine.complete_pending_tick();
        self.last_timestamp_ms = None;
        self.set_input(input);
        let start = now_ms();
        let report = self.engine.step_fixed(ticks).map_err(invalid)?;
        let simulation_ms = now_ms() - start;
        self.render_frame(width, height, simulation_ms, report, FrameKind::Exact)
    }

    /// Extracts and submits current state without running either schedule.
    ///
    /// Intended for captures and scene replacement while paused. Frame/tick
    /// counters, time, and input are unchanged; submission diagnostics report
    /// zero simulation time and fixed steps, retaining cumulative dropped time.
    #[cfg(feature = "dev")]
    pub fn dev_render(&mut self, width: u32, height: u32) -> Result<()> {
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        self.render_frame(width, height, 0.0, FrameReport::default(), FrameKind::Exact)
    }

    /// Render a selected fraction of the last fixed tick without running schedules.
    /// Invalid fractions are rejected before touching the engine or renderer.
    /// Ordinary development rendering and captures use current simulation state.
    /// `gpu` presents the fraction through GPU presentation instead of the
    /// exact CPU path, e.g. to compare both.
    #[cfg(feature = "dev")]
    pub fn dev_present(&mut self, alpha: f32, width: u32, height: u32, gpu: bool) -> Result<()> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(Error::new(
                ErrorCode::InvalidPresentationAlpha,
                "$.alpha",
                "Presentation alpha must be finite and in [0, 1].",
            ));
        }
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        // Previews show the latest completed tick.
        self.engine.complete_pending_tick();
        self.engine.reveal_ticks_ahead();
        self.render_frame(
            width,
            height,
            0.0,
            FrameReport::default(),
            FrameKind::Preview { alpha, gpu },
        )
    }

    fn set_input(&mut self, input: InputState) {
        let _profile = engine_profile::span("cpu", "host.input");
        fn bounded(value: f32, limit: f32) -> f32 {
            if value.is_finite() {
                engine_core::math::clamp(value, -limit, limit)
            } else {
                0.0
            }
        }
        *self.engine.app.world_mut().resource_mut::<InputState>() = InputState {
            move_forward: bounded(input.move_forward, 1.0),
            move_right: bounded(input.move_right, 1.0),
            move_up: bounded(input.move_up, 1.0),
            look_yaw: bounded(input.look_yaw, crate::host::policy::LOOK_LIMIT),
            look_pitch: bounded(input.look_pitch, crate::host::policy::LOOK_LIMIT),
        };
    }

    fn render_frame(
        &mut self,
        width: u32,
        height: u32,
        simulation_ms: f64,
        report: FrameReport,
        kind: FrameKind,
    ) -> Result<()> {
        if self.size != (width, height) {
            self.renderer.resize(width, height);
            self.size = (width, height);
            // Terrain level of detail measures pixels.
            self.engine
                .app
                .world_mut()
                .insert_resource(engine_core::Viewport {
                    width: width.max(1),
                    height: height.max(1),
                });
        }
        // Tiles that arrived stream in with this frame's extraction.
        self.terrain_fetches.exchange(&mut self.engine);
        let (frame_number, tick) = {
            let world = self.engine.app.world();
            (
                world.resource::<engine_core::FrameTime>().frame,
                world.resource::<engine_core::FixedTime>().tick,
            )
        };
        let before_extraction = now_ms();
        let culling_enabled = self.engine.frustum_culling();
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let gpu = match kind {
            FrameKind::Display { .. } => self.gpu_presentation,
            FrameKind::Exact => false,
            FrameKind::Preview { gpu, .. } => gpu,
        };
        if gpu && self.gpu_inputs_stale {
            // The renderer missed a gather. Current poses are the hierarchy's
            // own locals, so an earlier interval can no longer be uploaded:
            // complete and present the latest tick instead.
            self.engine.complete_pending_tick();
            self.engine.reveal_ticks_ahead();
        }
        let frame = match kind {
            FrameKind::Display { alpha } | FrameKind::Preview { alpha, .. } if gpu => {
                self.engine.extract_gpu_frame(aspect, alpha)
            }
            FrameKind::Display { alpha } | FrameKind::Preview { alpha, .. } => self
                .engine
                .extract_frame(aspect, PresentationMode::Interpolated { alpha }),
            FrameKind::Exact => self.engine.extract_frame(aspect, PresentationMode::Current),
        }?;
        let after_extraction = now_ms();
        let rendered = if let FrameKind::Display { .. } = kind {
            self.renderer
                .render_display_frame(frame)
                .map_err(render_failure)?
        } else {
            self.renderer.render_frame(frame).map_err(render_failure)?
        };
        if frame.gpu().is_some() {
            self.gpu_inputs_stale = !rendered.submitted;
        }
        let snapshot = frame.snapshot();
        // GPU-culled frames list their valid candidates as visible; culling and
        // budget counts come from a GPU read-back a few frames old.
        let listed = snapshot.visible_instances.min(u32::MAX as usize) as u32;
        let (visible, culled, dropped) = if frame.gpu().is_some() {
            let visible = rendered.gpu_visible.unwrap_or(listed);
            (
                visible,
                listed.saturating_sub(visible),
                visible.saturating_sub(rendered.instances),
            )
        } else {
            (
                listed,
                snapshot.culled_instances.min(u32::MAX as usize) as u32,
                snapshot.dropped_instances.min(u32::MAX as usize) as u32,
            )
        };
        self.stats = FrameStats {
            frame: frame_number,
            tick,
            simulation_ms,
            extraction_ms: after_extraction - before_extraction,
            render_ms: now_ms() - after_extraction,
            invalid_transforms: snapshot.invalid_transforms.min(u32::MAX as usize) as u32,
            render_candidates: snapshot.render_candidates.min(u32::MAX as usize) as u32,
            visible_instances: visible,
            culled_instances: culled,
            dropped_instances: dropped,
            missing_assets: snapshot.missing_assets.min(u32::MAX as usize) as u32,
            culling_enabled,
            fixed_steps: report.fixed_steps,
            dropped_seconds: self.stats.dropped_seconds + report.dropped_seconds,
            pipelines_compiling: false,
            loading: false,
            jobs: engine_core::JobStats::default(),
            navigation: engine_core::NavStats::default(),
            render: rendered,
        };
        Ok(())
    }

    pub fn stats(&self) -> FrameStats {
        FrameStats {
            pipelines_compiling: self.renderer.pipelines_compiling(),
            loading: self
                .engine
                .app
                .world()
                .get_resource::<engine_core::Terrains>()
                .is_some_and(engine_core::Terrains::loading),
            navigation: self
                .engine
                .app
                .world()
                .get_resource::<engine_core::NavAgents>()
                .map(engine_core::NavAgents::stats)
                .unwrap_or_default(),
            jobs: self
                .engine
                .app
                .world()
                .get_resource::<engine_core::Jobs>()
                .map(engine_core::Jobs::stats)
                .unwrap_or_default(),
            ..self.stats
        }
    }

    /// [`Self::stats`] as the JSON object hosts and agents read.
    pub fn stats_json(&self) -> String {
        serde_json::to_string(&self.stats()).expect("frame stats serialize")
    }

    /// Drop suspension time and held input before the next frame.
    pub fn suspend(&mut self) {
        self.last_timestamp_ms = None;
        self.engine.reset_clock();
        *self.engine.app.world_mut().resource_mut::<InputState>() = InputState::default();
    }
}

/// The renderer's failure as hosts receive it.
pub(crate) fn render_failure(error: RenderError) -> Error {
    Error::new(ErrorCode::RenderFailed, "$", error.0)
}
