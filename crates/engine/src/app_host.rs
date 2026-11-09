use engine_core::{
    BrickList, Engine, ErrorCode, FrameFacts, Gathered, JobExchange, JobId, JobMessage, JobRef,
    PauseReason, RecordExchange, RenderStats, WorkClock,
};
use engine_registry::Registry;
use engine_render::{RenderTarget, Renderer};

use crate::frame_stats::{DynamicResolutionStats, FrameStats, ResolutionStats};
use crate::host::policy::{FRAME_INTERVAL_MS, PACE_INTERVAL_LIMIT_MS, PACE_SMOOTHING};
use crate::host::InputSource;
use crate::resolution::{DynamicResolution, Resolution};
use crate::{Error, Result};

#[path = "app_attribution.rs"]
mod attribution;
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum JobClock {
    #[default]
    Wall,
    Frames,
}

impl JobClock {
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
    clock: JobClock,
) {
    let world = engine.app.world_mut();
    world.insert_resource(WorkClock {
        reading: clock.reading(),
    });
    if let Some(mut records) = world.get_resource_mut::<RecordExchange>() {
        records.set_sampling(Some(renderer.texture_sampling()));
    }
    if let Some(mut jobs) = world.get_resource_mut::<JobExchange>() {
        jobs.set_workers(workers);
        jobs.set_verifying(verifying);
    }
    crate::game::limit(engine, renderer.device_limits());
}

#[derive(Clone, Copy, Debug)]
enum Refresh {
    Measured(f64),
    Declared(f64),
}

impl Refresh {
    fn ms(self) -> f64 {
        match self {
            Self::Measured(ms) | Self::Declared(ms) => ms,
        }
    }

    fn measured(self, interval: f64) -> Self {
        match self {
            Self::Measured(ms) if interval > 0.0 && interval < PACE_INTERVAL_LIMIT_MS => {
                Self::Measured(ms + (interval - ms) * PACE_SMOOTHING)
            }
            Self::Measured(_) | Self::Declared(_) => self,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Focus {
    Held,
    Lost,
}

pub(crate) struct AppHost {
    pub(crate) engine: Engine,
    pub(crate) renderer: Box<Renderer>,
    focus: Focus,
    stats: FrameStats,
    size: (u32, u32),
    job_clock: JobClock,
    overlay: crate::host::SettingsLog,
    pipeline_list: crate::host::PipelineList,
    pixel_ratio: f64,
    refresh: Refresh,
    dynamic: crate::host::DynamicScale,
    gpu_ms: Option<f32>,
    #[cfg(feature = "profiling")]
    profile_frame: u64,
    #[cfg(any(feature = "dev", feature = "profiling"))]
    attribution: attribution::Attribution,
}

#[inline(never)]
fn built(build: impl FnOnce() -> Result<Engine>) -> Result<Engine> {
    build()
}

impl AppHost {
    pub(crate) async fn launch(
        target: RenderTarget,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<()>,
    ) -> Result<Self> {
        let starting = Renderer::start(target).await?;
        let engine = built(|| {
            let mut engine = crate::engine(bricks)?;
            build(&mut engine)?;
            Ok(engine)
        })?;
        let renderer = Box::new(starting.finish().await?);
        let mut host = Self::with(engine, renderer);
        host.claim_passes()?;
        Ok(host)
    }

    fn with(mut engine: Engine, renderer: Box<Renderer>) -> Self {
        wire(&mut engine, &renderer, (0, false), JobClock::Wall);
        Self {
            engine,
            renderer,
            focus: Focus::Held,
            stats: FrameStats::default(),
            size: (0, 0),
            job_clock: JobClock::Wall,
            overlay: crate::host::SettingsLog::default(),
            pipeline_list: crate::host::PipelineList::default(),
            pixel_ratio: 1.0,
            refresh: Refresh::Measured(FRAME_INTERVAL_MS),
            dynamic: crate::host::DynamicScale::default(),
            gpu_ms: None,
            #[cfg(feature = "profiling")]
            profile_frame: 0,
            #[cfg(any(feature = "dev", feature = "profiling"))]
            attribution: attribution::Attribution::default(),
        }
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        &mut self.engine
    }

    pub(crate) fn set_focus(&mut self, focus: Focus, source: InputSource) {
        self.focus = focus;
        self.follow_focus(source);
    }

    pub(crate) fn follow_focus(&mut self, source: InputSource) {
        match (self.focus, source) {
            (Focus::Lost, InputSource::Player) => self.engine.pause(PauseReason::Focus),
            (Focus::Lost, InputSource::Agent)
            | (Focus::Held, InputSource::Player | InputSource::Agent) => {
                self.engine.resume(PauseReason::Focus);
            }
        }
    }

    pub fn replace_engine(&mut self, mut engine: Engine) -> Engine {
        let jobs = self
            .engine
            .app
            .world()
            .get_resource::<JobExchange>()
            .map_or((0, false), |jobs| (jobs.workers(), jobs.verifying()));
        wire(&mut engine, &self.renderer, jobs, self.job_clock);
        let replaced = std::mem::replace(&mut self.engine, engine);
        self.reloaded();
        if let Err(error) = self.claim_passes() {
            panic!("{error}");
        }
        if let Err(error) = self.replay_settings() {
            panic!("{error}");
        }
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

    pub(crate) fn cook(&mut self) {
        self.pipeline_list.cook();
        engine_core::cook(self.engine.app.world_mut());
    }

    pub(crate) fn reloaded(&mut self) {
        self.pipeline_list.restart();
        self.restart_attribution();
    }

    pub fn set_setting(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        self.set_settings(&[(key.to_owned(), value)])
    }

    pub fn set_settings(&mut self, entries: &[(String, serde_json::Value)]) -> Result<()> {
        self.overlay.set(self.engine.app.world_mut(), entries)?;
        Ok(())
    }

    pub(crate) fn replay_settings(&mut self) -> Result<()> {
        self.overlay.replay(self.engine.app.world_mut())?;
        Ok(())
    }

    pub fn set_job_clock(&mut self, clock: JobClock) {
        self.job_clock = clock;
        self.engine.app.world_mut().insert_resource(WorkClock {
            reading: clock.reading(),
        });
    }

    pub fn set_pixel_ratio(&mut self, ratio: f64) {
        self.pixel_ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio
        } else {
            1.0
        };
    }

    pub fn set_refresh_interval(&mut self, milliseconds: Option<f64>) {
        self.refresh = match milliseconds.filter(|ms| ms.is_finite() && *ms > 0.0) {
            Some(ms) => Refresh::Declared(ms),
            None => Refresh::Measured(self.refresh.ms()),
        };
    }

    pub fn refresh_interval(&self) -> f64 {
        self.refresh.ms()
    }

    pub(crate) fn paced(&mut self) {
        let interval = self.engine.frame_time().real_seconds * 1000.0;
        self.refresh = self.refresh.measured(interval);
    }

    fn resolution_settings(&self) -> (Resolution, DynamicResolution) {
        let world = self.engine.app.world();
        (
            world
                .get_resource::<Resolution>()
                .copied()
                .unwrap_or_default(),
            world
                .get_resource::<DynamicResolution>()
                .copied()
                .unwrap_or_default(),
        )
    }

    fn resolution_budget(&self, settings: &DynamicResolution) -> f64 {
        settings
            .budget_ms
            .map_or_else(|| self.refresh.ms(), f64::from)
    }

    fn size_frame(
        &mut self,
        interval: Option<f64>,
        (width, height): (u32, u32),
    ) -> (DynamicResolutionStats, (u32, u32)) {
        let timing = self.renderer.take_frame_time();
        if let Some((ms, _)) = timing {
            self.gpu_ms = Some(ms as f32);
        }
        let (resolution, dynamic) = self.resolution_settings();
        self.adapt_resolution(&dynamic, timing, interval);
        let scale = self.dynamic.scale(&dynamic);
        let (render_width, render_height) =
            crate::host::render_size((width, height), self.pixel_ratio, &resolution, scale)
                .unwrap_or((width, height));
        let stats = DynamicResolutionStats {
            active: dynamic.enabled,
            scale: scale as f32,
            budget_ms: if dynamic.enabled {
                self.resolution_budget(&dynamic) as f32
            } else {
                0.0
            },
        };
        (stats, (render_width, render_height))
    }

    fn resolution_stats(&self, (width, height): (u32, u32)) -> ResolutionStats {
        let (display_width, display_height) = self.renderer.view_size().unwrap_or((0, 0));
        let (render_width, render_height) = self.renderer.render_size().unwrap_or((0, 0));
        ResolutionStats {
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
        settings: &DynamicResolution,
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
            interval.filter(|&ms| ms > 0.0 && ms < PACE_INTERVAL_LIMIT_MS)
        {
            self.dynamic.frame(settings, interval, budget, false);
        }
    }

    pub fn set_spare_time(&mut self, milliseconds: f64) {
        self.engine.set_spare_time(milliseconds);
    }

    pub fn set_job_workers(&mut self, workers: u32) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<JobExchange>()
        {
            jobs.set_workers(workers);
        }
    }

    pub fn set_job_verifying(&mut self, verifying: bool) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<JobExchange>()
        {
            jobs.set_verifying(verifying);
        }
    }

    pub fn take_jobs(&mut self) -> Vec<JobMessage> {
        self.engine
            .app
            .world_mut()
            .get_resource_mut::<JobExchange>()
            .map(|mut jobs| jobs.take())
            .unwrap_or_default()
    }

    pub fn run_job_here(
        &mut self,
        kind: &JobRef,
        input: &[u8],
    ) -> std::result::Result<Vec<u8>, String> {
        match self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<JobExchange>()
        {
            Some(mut jobs) => jobs.run_here(kind, input),
            None => Err(Error::new(ErrorCode::MissingResource, "$")
                .got("JobExchange")
                .to_string()),
        }
    }

    pub fn complete_job(&mut self, id: JobId, output: std::sync::Arc<[u8]>) {
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
            .get_resource_mut::<JobExchange>()
        {
            jobs.report_cache(hits, misses, bytes, mismatches);
        }
    }

    pub fn report_jobs_held(&mut self, held: u32) {
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<JobExchange>()
        {
            jobs.report_held(held);
        }
    }

    fn finish_job(&mut self, id: JobId, output: std::result::Result<std::sync::Arc<[u8]>, String>) {
        self.engine.land_job(id, output);
    }

    pub(crate) fn deliver(&mut self) {
        self.renderer.deliver(self.engine.app.world_mut());
    }

    pub fn frame(&mut self, timestamp_ms: f64, width: u32, height: u32) -> Result<()> {
        if !timestamp_ms.is_finite() {
            return Err(Error::new(ErrorCode::InvalidInput, "$.timestamp")
                .expected(crate::Expected::Finite)
                .got(timestamp_ms));
        }
        self.begin_attribution();
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let profile = engine_profile::span("cpu", "host.frame");
        {
            let _core = attribution::core_scope();
            self.deliver();
            let start = now_ms();
            self.engine.frame(timestamp_ms * 0.001)?;
            let simulation_ms = now_ms() - start;
            let time = self.engine.frame_time();
            self.render_frame(
                width,
                height,
                simulation_ms,
                time.ticks,
                Some(time.real_seconds * 1000.0),
            )?;
        }
        drop(profile);
        self.end_attribution();
        Ok(())
    }

    #[cfg(feature = "dev")]
    pub fn dev_step(&mut self, ticks: u32, width: u32, height: u32) -> Result<()> {
        engine_core::check_dev_ticks(ticks)?;
        self.follow_focus(InputSource::Agent);
        self.begin_attribution();
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let profile = engine_profile::span("cpu", "host.step");
        {
            let _core = attribution::core_scope();
            self.deliver();
            let start = now_ms();
            self.engine.step_fixed(1)?;
            let mut fixed_steps = self.engine.frame_time().ticks;
            for _ in 1..ticks {
                self.engine.start_frame();
                self.keep_held();
                self.engine.step_fixed(1)?;
                fixed_steps += self.engine.frame_time().ticks;
            }
            let simulation_ms = now_ms() - start;
            self.render_frame(width, height, simulation_ms, fixed_steps, None)?;
        }
        drop(profile);
        self.end_attribution();
        Ok(())
    }

    #[cfg(feature = "dev")]
    fn keep_held(&mut self) {
        engine_core::hold_input(self.engine.app.world_mut());
    }

    #[cfg(feature = "dev")]
    pub fn dev_render(&mut self, width: u32, height: u32) -> Result<()> {
        self.begin_attribution();
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        {
            let _core = attribution::core_scope();
            self.deliver();
            self.engine.frame_without_ticks();
            self.render_frame(width, height, 0.0, 0, None)?;
        }
        self.end_attribution();
        Ok(())
    }

    #[cfg(feature = "dev")]
    pub fn dev_present(&mut self, alpha: f32, width: u32, height: u32) -> Result<()> {
        self.begin_attribution();
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        {
            let _core = attribution::core_scope();
            self.deliver();
            self.engine
                .present(engine_core::Presented::Between { alpha })?;
            self.render_frame(width, height, 0.0, 0, None)?;
        }
        self.end_attribution();
        Ok(())
    }

    pub(crate) fn accept(&mut self, input: Gathered) {
        let _profile = engine_profile::span("cpu", "host.input");
        engine_core::accept_input(self.engine.app.world_mut(), input);
    }

    fn display_size(&self, (width, height): (u32, u32)) -> glam::UVec2 {
        let (width, height) = self.renderer.view_size().unwrap_or((width, height));
        glam::UVec2::new(width, height)
    }

    fn render_frame(
        &mut self,
        width: u32,
        height: u32,
        simulation_ms: f64,
        fixed_steps: u32,
        interval: Option<f64>,
    ) -> Result<()> {
        if self.size != (width, height) {
            self.renderer.resize(width, height);
            self.size = (width, height);
        }
        let (dynamic_resolution, (render_width, render_height)) =
            self.size_frame(interval, (width, height));
        let time = self.engine.frame_time();
        let tick = self
            .engine
            .app
            .world()
            .resource::<engine_core::FixedTime>()
            .tick;
        let before_acquire = now_ms();
        let acquired = self.renderer.acquire_frame()?;
        let acquire_ms = now_ms() - before_acquire;
        let (extraction_ms, draw_ms, rendered) = if acquired {
            self.draw_frame((width, height), (render_width, render_height))?
        } else {
            (0.0, 0.0, RenderStats::default())
        };
        self.stats = FrameStats {
            frame: time.frame,
            tick,
            simulation_ms,
            extraction_ms,
            render_ms: acquire_ms + draw_ms,
            fixed_steps,
            dropped_seconds: time.dropped_seconds,
            cut_seconds: time.cut_seconds,
            pipelines_listed: 0,
            pipelines_unlisted: 0,
            readiness: self.engine.readiness().clone(),
            budget: self.engine.budget().uses().to_vec(),
            resolution: self.resolution_stats((width, height)),
            dynamic_resolution,
            jobs: engine_core::JobStats::default(),
            refused_views: 0,
            disabled_features: self.renderer.disabled_features(self.engine.app.world()),
            render: rendered,
            bricks_cost: None,
            bricks: std::collections::BTreeMap::new(),
        };
        self.pipeline_list
            .after_frame(&mut self.engine, &self.renderer);
        Ok(())
    }

    fn draw_frame(
        &mut self,
        (width, height): (u32, u32),
        (render_width, render_height): (u32, u32),
    ) -> Result<(f64, f64, RenderStats)> {
        let before_extraction = now_ms();
        self.engine.start_frame();
        let whole = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<engine_core::WholeFrames>()
            .is_some_and(|mut whole| whole.asked());
        let world = self.engine.app.world();
        let post = self.renderer.post(world);
        let display_size = self.display_size((width, height));
        let render_size = if !whole && post {
            glam::UVec2::new(render_width, render_height)
                .clamp(glam::UVec2::ONE, display_size.max(glam::UVec2::ONE))
        } else {
            display_size
        };
        let facts = FrameFacts {
            display_size,
            render_size,
            pixel_ratio: self.pixel_ratio as f32,
        };
        self.engine.set_frame(facts);
        self.collect_pipelines();
        let frame = self.engine.extract_frame()?;
        let after_extraction = now_ms();
        let rendered = self.renderer.render_frame(frame)?;
        self.collect_pipelines();
        self.report_shader_errors();
        Ok((
            after_extraction - before_extraction,
            now_ms() - after_extraction,
            rendered,
        ))
    }

    pub(crate) fn collect_pipelines(&mut self) {
        let compiling = self.renderer.pipelines_compiling();
        self.engine.set_compiling(compiling);
    }

    fn report_shader_errors(&mut self) {
        let errors = self.renderer.shader_errors();
        if errors.is_empty() {
            return;
        }
        let mut readiness = self
            .engine
            .app
            .world_mut()
            .resource_mut::<engine_core::Readiness>();
        for error in errors {
            if !readiness.failed.iter().any(|(_, known)| known == error) {
                readiness.fail(&error.path, error.clone());
            }
        }
    }

    pub fn job_list(&self) -> std::sync::Arc<engine_registry::JobList> {
        self.engine.job_list()
    }

    pub fn frame_times(&self) -> &FrameStats {
        &self.stats
    }

    pub fn stats(&self) -> FrameStats {
        let (pipelines_listed, pipelines_unlisted) = self.renderer.pipelines_listed();
        let world = self.engine.app.world();
        FrameStats {
            pipelines_listed,
            pipelines_unlisted,
            readiness: self.engine.readiness().clone(),
            jobs: world
                .get_resource::<JobExchange>()
                .map(JobExchange::stats)
                .unwrap_or_default(),
            refused_views: world
                .get_resource::<engine_core::Views>()
                .map_or(0, |views| views.refused().len() as u32),
            bricks: world
                .get_resource::<Registry>()
                .map(|registry| registry.frame_stats(world))
                .unwrap_or_default(),
            ..self.stats.clone()
        }
    }

    pub fn stats_json(&self) -> String {
        serde_json::to_string(&self.stats()).expect("stats_serialize")
    }

    pub fn suspend(&mut self) {
        self.dynamic.reset();
        self.engine.reset_clock();
        engine_core::suspend_input(self.engine.app.world_mut());
    }
}
