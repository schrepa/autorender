use engine_core::{
    Engine, ErrorCode, FrameReport, FrameStats, InputState, JobId, JobRequest, JobShare,
    RecordFetch, RecordKey,
};
use engine_render::{RenderError, RenderTarget, Renderer};

use crate::{Error, Result};

#[cfg(feature = "profiling")]
#[path = "app_profiling.rs"]
mod profiling;

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

#[derive(Debug)]
pub enum JobMessage {
    Cancel(JobId),
    Share(JobShare),
    Run(JobRequest),
    Fetch(RecordFetch),
    Keep(RecordKey, Vec<u8>, engine_core::RecordParity),
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TerrainClock {
    #[default]
    Wall,
    Frames,
}

impl TerrainClock {
    fn reading(self) -> Option<fn() -> f64> {
        match self {
            Self::Wall => Some(now_ms),
            Self::Frames => None,
        }
    }
}

fn wire(
    engine: &mut Engine,
    renderer: &Renderer,
    (workers, verifying): (u32, bool),
    clock: TerrainClock,
) {
    let world = engine.app.world_mut();
    if let Some(mut terrains) = world.get_resource_mut::<engine_terrain::Terrains>() {
        terrains.set_device_tiles(Some(renderer.terrain_tile_capacity()));
        terrains.set_clock(clock.reading());
        terrains.set_water_settling(engine_render::sources::WATER, true);
    }
    if let Some(mut records) = world.get_resource_mut::<engine_core::Records>() {
        records.set_sampling(Some(renderer.texture_sampling()));
    }
    if let Some(mut jobs) = world.get_resource_mut::<engine_core::Jobs>() {
        jobs.set_workers(workers);
        jobs.set_verifying(verifying);
    }
}

pub(crate) struct AppHost {
    pub(crate) engine: Engine,
    pub(crate) renderer: Box<Renderer>,
    last_timestamp_ms: Option<f64>,
    stats: FrameStats,
    size: (u32, u32),
    tick_pipelining: bool,
    gpu_inputs_stale: bool,
    terrain_clock: TerrainClock,
    overrides: Overrides,
    record_fetches: Vec<(JobId, RecordKey)>,
    pipeline_list: crate::host::PipelineList,
    pixel_ratio: f64,
    refresh_ms: Option<f64>,
    dynamic: crate::host::DynamicScale,
    gpu_ms: Option<f32>,
    #[cfg(feature = "profiling")]
    profile_frame: u64,
}

#[derive(Clone, Copy, Default)]
struct Overrides {
    temporal_antialiasing: Option<bool>,
    occlusion_culling: Option<bool>,
    subpixel_culling: Option<bool>,
    debug_view: Option<engine_core::DebugViewMode>,
    resolution_scale: Option<f32>,
    dynamic_resolution: Option<bool>,
}

#[derive(Clone, Copy)]
#[cfg_attr(
    not(feature = "dev"),
    expect(
        dead_code,
        reason = "only development controls submit exact frames and previews"
    )
)]
enum FrameKind {
    Display { alpha: f32, interval: Option<f64> },
    Exact,
    Preview { alpha: f32 },
}

#[inline(never)]
fn built(build: impl FnOnce() -> Result<Engine>) -> Result<Engine> {
    build()
}

impl AppHost {
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
        wire(&mut engine, &renderer, (0, false), TerrainClock::Wall);
        Self {
            engine,
            renderer,
            last_timestamp_ms: None,
            stats: FrameStats::default(),
            size: (0, 0),
            tick_pipelining: true,
            gpu_inputs_stale: false,
            terrain_clock: TerrainClock::Wall,
            overrides: Overrides::default(),
            record_fetches: Vec::new(),
            pipeline_list: crate::host::PipelineList::default(),
            pixel_ratio: 1.0,
            refresh_ms: None,
            dynamic: crate::host::DynamicScale::default(),
            gpu_ms: None,
            #[cfg(feature = "profiling")]
            profile_frame: 0,
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    pub fn replace_engine(&mut self, mut engine: Engine) -> Engine {
        let jobs = self
            .engine
            .app
            .world()
            .get_resource::<engine_core::Jobs>()
            .map_or((0, false), |jobs| (jobs.workers(), jobs.verifying()));
        wire(&mut engine, &self.renderer, jobs, self.terrain_clock);
        self.pipeline_list = crate::host::PipelineList::default();
        self.gpu_inputs_stale = false;
        for (_, record) in std::mem::take(&mut self.record_fetches) {
            self.renderer.give_record(record, None);
        }
        let replaced = std::mem::replace(&mut self.engine, engine);
        self.apply_overrides();
        self.suspend();
        replaced
    }

    pub(crate) fn prepare_pipelines(
        &mut self,
        content: impl FnOnce(&Engine) -> Option<Vec<u8>>,
    ) -> bool {
        if self.pipeline_list.unasked() {
            let content = content(&self.engine);
            self.pipeline_list.ask(&mut self.engine, content);
        }
        self.pipeline_list
            .take(&mut self.engine, &mut self.renderer);
        self.pipeline_list.waiting()
    }

    pub(crate) fn forget_pipeline_list(&mut self) {
        self.pipeline_list = crate::host::PipelineList::default();
    }

    pub fn apply_overrides(&mut self) {
        let Overrides {
            temporal_antialiasing,
            occlusion_culling,
            subpixel_culling,
            debug_view,
            resolution_scale,
            dynamic_resolution,
        } = self.overrides;
        let world = self.engine.app.world_mut();
        if let Some(enabled) = temporal_antialiasing {
            let settings = world
                .get_resource::<engine_core::TemporalAntialiasing>()
                .copied()
                .unwrap_or_default();
            world.insert_resource(engine_core::TemporalAntialiasing {
                enabled,
                ..settings
            });
        }
        if let Some(enabled) = occlusion_culling {
            world.insert_resource(engine_core::OcclusionCulling { enabled });
        }
        if let Some(enabled) = subpixel_culling {
            world.insert_resource(engine_core::SubpixelCulling { enabled });
        }
        if let Some(view) = debug_view {
            let settings = world
                .get_resource::<engine_core::DebugView>()
                .copied()
                .unwrap_or_default();
            world.insert_resource(engine_core::DebugView { view, ..settings });
        }
        if let Some(scale) = resolution_scale {
            world.insert_resource(engine_core::Resolution { scale });
        }
        if let Some(enabled) = dynamic_resolution {
            let settings = world
                .get_resource::<engine_core::DynamicResolution>()
                .copied()
                .unwrap_or_default();
            world.insert_resource(engine_core::DynamicResolution {
                enabled,
                ..settings
            });
        }
    }

    pub fn set_terrain_clock(&mut self, clock: TerrainClock) {
        self.terrain_clock = clock;
        if let Some(mut terrains) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_terrain::Terrains>()
        {
            terrains.set_clock(clock.reading());
        }
    }

    pub fn set_tick_pipelining(&mut self, enabled: bool) {
        self.tick_pipelining = enabled;
        if !enabled {
            self.engine.complete_pending_tick();
        }
    }

    pub fn tick_pipelining(&self) -> bool {
        self.tick_pipelining
    }

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

    pub fn set_debug_view(&mut self, view: engine_core::DebugViewMode) {
        self.overrides.debug_view = Some(view);
        self.apply_overrides();
    }

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

    pub fn set_pixel_ratio(&mut self, ratio: f64) {
        self.pixel_ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio
        } else {
            1.0
        };
    }

    pub fn set_refresh_interval(&mut self, milliseconds: Option<f64>) {
        self.refresh_ms = milliseconds.filter(|ms| ms.is_finite() && *ms > 0.0);
    }

    pub fn set_resolution_scale(&mut self, scale: f64) {
        self.overrides.resolution_scale = Some(scale as f32);
        self.apply_overrides();
    }

    pub fn set_dynamic_resolution(&mut self, enabled: bool) {
        self.overrides.dynamic_resolution = Some(enabled);
        self.apply_overrides();
    }

    pub fn dynamic_resolution(&self) -> bool {
        self.engine
            .app
            .world()
            .get_resource::<engine_core::DynamicResolution>()
            .is_some_and(|settings| settings.enabled)
    }

    fn resolution_settings(&self) -> (engine_core::Resolution, engine_core::DynamicResolution) {
        let world = self.engine.app.world();
        (
            world
                .get_resource::<engine_core::Resolution>()
                .copied()
                .unwrap_or_else(engine_core::Resolution::off)
                .sanitized(),
            world
                .get_resource::<engine_core::DynamicResolution>()
                .copied()
                .unwrap_or_else(engine_core::DynamicResolution::off)
                .sanitized(),
        )
    }

    fn resolution_budget(&self, settings: &engine_core::DynamicResolution) -> f64 {
        settings.budget_ms.map_or_else(
            || {
                self.refresh_ms
                    .unwrap_or(crate::host::policy::FRAME_INTERVAL_MS)
            },
            f64::from,
        )
    }

    fn size_frame(
        &mut self,
        kind: FrameKind,
        (width, height): (u32, u32),
    ) -> engine_core::DynamicResolutionStats {
        let timing = self.renderer.take_frame_time();
        if let Some((ms, _)) = timing {
            self.gpu_ms = Some(ms as f32);
        }
        let (resolution, dynamic) = self.resolution_settings();
        let FrameKind::Display { interval, .. } = kind else {
            self.renderer.set_render_size(width, height);
            return engine_core::DynamicResolutionStats {
                active: false,
                scale: 1.0,
                budget_ms: 0.0,
            };
        };
        self.adapt_resolution(&dynamic, timing, interval);
        let scale = self.dynamic.scale(&dynamic);
        let (render_width, render_height) = crate::host::render_size(
            (width, height),
            self.pixel_ratio,
            f64::from(resolution.scale),
            scale,
        )
        .unwrap_or((width, height));
        self.renderer.set_render_size(render_width, render_height);
        engine_core::DynamicResolutionStats {
            active: dynamic.enabled,
            scale: scale as f32,
            budget_ms: if dynamic.enabled {
                self.resolution_budget(&dynamic) as f32
            } else {
                0.0
            },
        }
    }

    fn resolution_stats(&self, (width, height): (u32, u32)) -> engine_core::ResolutionStats {
        let (display_width, display_height) = self.renderer.view_size().unwrap_or((0, 0));
        let (render_width, render_height) = self.renderer.render_size().unwrap_or((0, 0));
        engine_core::ResolutionStats {
            display_width,
            display_height,
            render_width,
            render_height,
            dropped_pixels: (u64::from(width) * u64::from(height))
                .saturating_sub(u64::from(display_width) * u64::from(display_height)),
            gpu_ms: self.gpu_ms,
        }
    }

    fn adapt_resolution(
        &mut self,
        settings: &engine_core::DynamicResolution,
        timing: Option<(f64, (u32, u32))>,
        interval: Option<f64>,
    ) {
        if !settings.enabled {
            self.dynamic.reset();
            return;
        }
        let budget = self.resolution_budget(settings);
        if self.renderer.times_frames() {
            if let Some((ms, size)) = timing
                && Some(size) == self.renderer.render_size()
            {
                self.dynamic.frame(settings, ms, budget, true);
            }
        } else if let Some(interval) =
            interval.filter(|&ms| ms > 0.0 && ms < crate::host::policy::PACE_INTERVAL_LIMIT_MS)
        {
            self.dynamic.frame(settings, interval, budget, false);
        }
    }

    pub fn work(&mut self, width: u32, height: u32) {
        let viewport = engine_core::Viewport {
            width: width.max(1),
            height: height.max(1),
        };
        let world = self.engine.app.world_mut();
        if world.get_resource::<engine_core::Viewport>() != Some(&viewport) {
            world.insert_resource(viewport);
        }
        engine_terrain::stream_now(world);
    }

    pub fn work_between_frames(&mut self, milliseconds: f64) -> bool {
        if !milliseconds.is_finite() || milliseconds <= 0.0 {
            return false;
        }
        engine_terrain::work_between_frames(self.engine.app.world_mut(), milliseconds)
    }

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

    pub fn set_job_verifying(&mut self, verifying: bool) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.set_verifying(verifying);
        }
    }

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
        let mut fetches = Vec::new();
        let mut keeps = Vec::new();
        jobs.take_cancels(&mut cancels);
        jobs.take_shares(&mut shares);
        jobs.take_requests(&mut requests);
        jobs.take_fetches(&mut fetches);
        jobs.take_keeps(&mut keeps);
        let cancels = cancels.into_iter().map(JobMessage::Cancel);
        let shares = shares.into_iter().map(JobMessage::Share);
        let requests = requests.into_iter().map(JobMessage::Run);
        let fetches = fetches.into_iter().map(JobMessage::Fetch);
        let keeps = keeps
            .into_iter()
            .map(|(record, bytes, parity)| JobMessage::Keep(record, bytes, parity));
        cancels
            .chain(shares)
            .chain(requests)
            .chain(fetches)
            .chain(keeps)
            .collect()
    }

    pub fn run_job_here(
        &mut self,
        kind: &str,
        input: &[u8],
    ) -> std::result::Result<Vec<u8>, String> {
        match self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            Some(mut jobs) => jobs.run_here(kind, input),
            None => Err("the engine runs no jobs".to_owned()),
        }
    }

    pub fn complete_job(&mut self, id: JobId, output: Vec<u8>) {
        self.finish_job(id, Ok(output));
    }

    pub fn fail_job(&mut self, id: JobId, error: String) {
        self.finish_job(id, Err(error));
    }

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

    pub fn report_jobs_held(&mut self, held: u32) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::Jobs>()
        {
            jobs.report_held(held);
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
        let interval = self
            .last_timestamp_ms
            .map(|last| (timestamp_ms - last).max(0.0));
        let delta = interval.map_or(0.0, |interval| interval * 0.001);
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
                interval,
            },
        )?;
        if self.tick_pipelining {
            let start = now_ms();
            if self.engine.run_ahead(delta) {
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

    #[cfg(feature = "dev")]
    pub fn dev_render(&mut self, width: u32, height: u32) -> Result<()> {
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        self.render_frame(width, height, 0.0, FrameReport::default(), FrameKind::Exact)
    }

    #[cfg(feature = "dev")]
    pub fn dev_present(&mut self, alpha: f32, width: u32, height: u32) -> Result<()> {
        if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
            return Err(Error::new(
                ErrorCode::InvalidPresentationAlpha,
                "$.alpha",
                "Presentation alpha must be finite and in [0, 1].",
            ));
        }
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        self.engine.complete_pending_tick();
        self.engine.reveal_ticks_ahead();
        self.render_frame(
            width,
            height,
            0.0,
            FrameReport::default(),
            FrameKind::Preview { alpha },
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
            self.engine
                .app
                .world_mut()
                .insert_resource(engine_core::Viewport {
                    width: width.max(1),
                    height: height.max(1),
                });
        }
        let dynamic_resolution = self.size_frame(kind, (width, height));
        let (frame_number, tick) = {
            let world = self.engine.app.world();
            (
                world.resource::<engine_core::FrameTime>().frame,
                world.resource::<engine_core::FixedTime>().tick,
            )
        };
        self.renderer.set_cooking(
            self.engine
                .app
                .world()
                .contains_resource::<engine_core::Cooking>(),
        );
        let before_extraction = now_ms();
        let culling_enabled = self.engine.frustum_culling();
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        let gpu = match kind {
            FrameKind::Display { .. } | FrameKind::Preview { .. } => true,
            FrameKind::Exact => false,
        };
        if gpu && self.gpu_inputs_stale {
            self.engine.complete_pending_tick();
            self.engine.reveal_ticks_ahead();
        }
        let frame = match kind {
            FrameKind::Display { alpha, .. } | FrameKind::Preview { alpha } => {
                self.engine.extract_gpu_frame(aspect, alpha)?
            }
            FrameKind::Exact => self.engine.extract_frame(aspect),
        };
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
                0,
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
            pipelines_listed: 0,
            pipelines_unlisted: 0,
            loading: false,
            resolution: self.resolution_stats((width, height)),
            dynamic_resolution,
            jobs: engine_core::JobStats::default(),
            navigation: engine_core::NavStats::default(),
            physics: engine_core::PhysicsStats::default(),
            disabled_features: self.renderer.disabled_features(),
            render: rendered,
        };
        self.exchange_records();
        self.pipeline_list.after_frame(
            &mut self.engine,
            &self.renderer,
            self.stats.render.pipelines_created,
        );
        Ok(())
    }

    fn exchange_records(&mut self) {
        let world = self.engine.app.world_mut();
        let wants = self.renderer.take_record_wants();
        let made = self.renderer.take_made_records();
        let answers = match world.get_resource_mut::<engine_core::Jobs>() {
            Some(mut jobs) => {
                let mut answers = Vec::new();
                let mut asked = Vec::with_capacity(self.record_fetches.len() + wants.len());
                for (id, record) in std::mem::take(&mut self.record_fetches) {
                    match jobs.take(id) {
                        Some(output) => answers.push((record, output.ok())),
                        None => asked.push((id, record)),
                    }
                }
                let priority = engine_core::job_priority(engine_core::VIEW_JOBS, 0);
                for record in wants {
                    asked.push((jobs.fetch(record, priority), record));
                }
                for (record, bytes, parity) in made {
                    jobs.keep_with(record, bytes, parity);
                }
                self.record_fetches = asked;
                answers
            }
            None => wants.into_iter().map(|record| (record, None)).collect(),
        };
        for (record, bytes) in answers {
            self.renderer.give_record(record, bytes);
        }
        if let Some(mut cooking) = world.get_resource_mut::<engine_core::Cooking>() {
            for (part, left) in self.renderer.records_left() {
                cooking.left(part, left);
            }
        }
        self.exchange_settled_water();
    }

    fn exchange_settled_water(&mut self) {
        let world = self.engine.app.world_mut();
        let cooking = world.get_resource::<engine_core::Cooking>().is_some();
        self.renderer.set_water_cooking(cooking);
        if let Some(mut terrains) = world.get_resource_mut::<engine_terrain::Terrains>() {
            let starts = terrains.take_water_settles();
            if !starts.is_empty() {
                self.renderer.settle_water(starts);
            }
        }
        let settled = self.renderer.take_settled_water();
        if settled.is_empty() {
            return;
        }
        if let Some(mut jobs) = world.get_resource_mut::<engine_core::Jobs>() {
            for (record, made) in &settled {
                if let Ok(bytes) = made {
                    jobs.keep_with(*record, bytes.clone(), engine_terrain::SETTLED_PARITY);
                }
            }
        }
        if let Some(mut terrains) = world.get_resource_mut::<engine_terrain::Terrains>() {
            for (record, made) in settled {
                terrains.water_settled(record, made);
            }
        }
    }

    pub fn stats(&self) -> FrameStats {
        let (pipelines_listed, pipelines_unlisted) = self.renderer.pipelines_listed();
        FrameStats {
            pipelines_compiling: self.renderer.pipelines_compiling(),
            pipelines_listed,
            pipelines_unlisted,
            loading: self
                .engine
                .app
                .world()
                .get_resource::<engine_terrain::Terrains>()
                .is_some_and(engine_terrain::Terrains::loading)
                || self
                    .engine
                    .app
                    .world()
                    .get_resource::<engine_core::Records>()
                    .is_some_and(engine_core::Records::loading)
                || self
                    .engine
                    .app
                    .world()
                    .get_resource::<engine_core::StaticTables>()
                    .is_some_and(engine_core::StaticTables::loading)
                || self.pipeline_list.waiting(),
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
            physics: self
                .engine
                .app
                .world()
                .get_resource::<engine_physics::PhysicsWorld>()
                .map(engine_physics::PhysicsWorld::stats)
                .unwrap_or_default(),
            ..self.stats
        }
    }

    pub fn stats_json(&self) -> String {
        serde_json::to_string(&self.stats()).expect("frame stats serialize")
    }

    pub fn suspend(&mut self) {
        self.last_timestamp_ms = None;
        self.dynamic.reset();
        self.engine.reset_clock();
        *self.engine.app.world_mut().resource_mut::<InputState>() = InputState::default();
    }
}

pub(crate) fn render_failure(error: RenderError) -> Error {
    Error::new(ErrorCode::RenderFailed, "$", error.0)
}
