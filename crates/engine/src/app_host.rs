use std::collections::BTreeMap;

use engine_core::{
    BrickList, Engine, ErrorCode, FrameFacts, FrameReport, FrameStats, JobExchange, JobId, JobRef,
    MenuInput, MoveInput, RecordExchange, RenderStats, WorkClock,
};
use engine_registry::Registry;
use engine_render::{RenderTarget, Renderer};

use crate::host::{FrameInput, JobMessage};
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
    crate::game::limit(engine, &renderer.limits(), renderer.features());
}

pub(crate) struct AppHost {
    pub(crate) engine: Engine,
    pub(crate) renderer: Box<Renderer>,
    last_timestamp_ms: Option<f64>,
    stats: FrameStats,
    size: (u32, u32),
    terrain_clock: TerrainClock,
    overlay: BTreeMap<String, serde_json::Value>,
    pipeline_list: crate::host::PipelineList,
    pixel_ratio: f64,
    refresh_ms: Option<f64>,
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
        wire(&mut engine, &renderer, (0, false), TerrainClock::Wall);
        Self {
            engine,
            renderer,
            last_timestamp_ms: None,
            stats: FrameStats::default(),
            size: (0, 0),
            terrain_clock: TerrainClock::Wall,
            overlay: BTreeMap::new(),
            pipeline_list: crate::host::PipelineList::default(),
            pixel_ratio: 1.0,
            refresh_ms: None,
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

    pub fn replace_engine(&mut self, mut engine: Engine) -> Engine {
        let jobs = self
            .engine
            .app
            .world()
            .get_resource::<JobExchange>()
            .map_or((0, false), |jobs| (jobs.workers(), jobs.verifying()));
        wire(&mut engine, &self.renderer, jobs, self.terrain_clock);
        self.pipeline_list = crate::host::PipelineList::default();
        let replaced = std::mem::replace(&mut self.engine, engine);
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

    pub(crate) fn forget_pipeline_list(&mut self) {
        self.pipeline_list = crate::host::PipelineList::default();
    }

    pub fn set_setting(&mut self, key: &str, value: serde_json::Value) -> Result<()> {
        crate::host::apply_setting(self.engine.app.world_mut(), key, &value)?;
        self.overlay.insert(key.to_owned(), value);
        Ok(())
    }

    pub(crate) fn replay_settings(&mut self) -> Result<()> {
        let world = self.engine.app.world_mut();
        for (key, value) in &self.overlay {
            crate::host::apply_setting(world, key, value)?;
        }
        Ok(())
    }

    pub fn set_terrain_clock(&mut self, clock: TerrainClock) {
        self.terrain_clock = clock;
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
        self.refresh_ms = milliseconds.filter(|ms| ms.is_finite() && *ms > 0.0);
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
        interval: Option<f64>,
        (width, height): (u32, u32),
    ) -> (engine_core::DynamicResolutionStats, (u32, u32)) {
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
        let stats = engine_core::DynamicResolutionStats {
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
        self.engine.stream_now();
    }

    pub fn work_between_frames(&mut self, milliseconds: f64) -> bool {
        if !milliseconds.is_finite() || milliseconds <= 0.0 {
            return false;
        }
        let micros = (milliseconds * 1000.0).min(f64::from(u32::MAX)) as u32;
        self.engine.between_frames(micros)
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
        let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<JobExchange>()
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
            .map(|(record, kind, bytes, parity)| JobMessage::Keep(record, kind, bytes, parity));
        cancels
            .chain(shares)
            .chain(requests)
            .chain(fetches)
            .chain(keeps)
            .collect()
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
        if let Some(mut jobs) = self
            .engine
            .app
            .world_mut()
            .get_resource_mut::<JobExchange>()
        {
            jobs.complete(id, output);
        }
    }

    pub(crate) fn deliver(&mut self) {
        self.renderer.deliver(self.engine.app.world_mut());
    }

    pub fn frame(
        &mut self,
        timestamp_ms: f64,
        width: u32,
        height: u32,
        input: MoveInput,
    ) -> Result<()> {
        if !timestamp_ms.is_finite() {
            return Err(Error::new(ErrorCode::InvalidInput, "$.timestamp")
                .expected(crate::Expected::Finite)
                .got(timestamp_ms));
        }
        self.begin_attribution();
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let profile = engine_profile::span("cpu", "host.frame");
        let interval = self
            .last_timestamp_ms
            .map(|last| (timestamp_ms - last).max(0.0));
        let delta = interval.map_or(0.0, |interval| interval * 0.001);
        self.last_timestamp_ms = Some(timestamp_ms);
        self.set_input(input);
        {
            let _core = attribution::core_scope();
            self.deliver();
            let start = now_ms();
            let report = self.engine.advance(delta);
            let simulation_ms = now_ms() - start;
            self.render_frame(
                width,
                height,
                simulation_ms,
                report,
                report.interpolation,
                interval,
            )?;
        }
        drop(profile);
        self.end_attribution();
        Ok(())
    }

    #[cfg(feature = "dev")]
    pub fn dev_step(
        &mut self,
        ticks: u32,
        width: u32,
        height: u32,
        input: MoveInput,
    ) -> Result<()> {
        engine_core::check_dev_ticks(ticks)?;
        self.begin_attribution();
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        let profile = engine_profile::span("cpu", "host.step");
        {
            let _core = attribution::core_scope();
            self.last_timestamp_ms = None;
            self.set_input(input);
            self.deliver();
            let start = now_ms();
            let mut report = self.engine.step_fixed(1)?;
            for _ in 1..ticks {
                self.engine.start_frame();
                self.keep_held();
                report.fixed_steps += self.engine.step_fixed(1)?.fixed_steps;
            }
            let simulation_ms = now_ms() - start;
            self.render_frame(width, height, simulation_ms, report, 1.0, None)?;
        }
        drop(profile);
        self.end_attribution();
        Ok(())
    }

    #[cfg(feature = "dev")]
    fn keep_held(&mut self) {
        let world = self.engine.app.world();
        let moving = *world.resource::<MoveInput>();
        let pointer = *world.resource::<engine_core::PointerInput>();
        let axes = *world.resource::<engine_core::AxisInput>();
        self.set_input(MoveInput {
            look_right_pixels: 0.0,
            look_down_pixels: 0.0,
            ..moving
        });
        self.set_gathered(&FrameInput {
            pointer: engine_core::PointerInput {
                delta_pixels: glam::Vec2::ZERO,
                wheel_up_notches: 0.0,
                wheel_right_notches: 0.0,
                pressed: false,
                released: false,
                ..pointer
            },
            axes,
            ..FrameInput::default()
        });
    }

    #[cfg(feature = "dev")]
    pub fn dev_render(&mut self, width: u32, height: u32) -> Result<()> {
        self.begin_attribution();
        #[cfg(feature = "profiling")]
        self.begin_profile_frame();
        {
            let _core = attribution::core_scope();
            self.deliver();
            self.render_frame(width, height, 0.0, FrameReport::default(), 1.0, None)?;
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
            self.render_frame(width, height, 0.0, FrameReport::default(), alpha, None)?;
        }
        self.end_attribution();
        Ok(())
    }

    fn set_input(&mut self, input: MoveInput) {
        let _profile = engine_profile::span("cpu", "host.input");
        fn bounded(value: f32, limit: f32) -> f32 {
            if value.is_finite() {
                engine_core::math::clamp(value, -limit, limit)
            } else {
                0.0
            }
        }
        self.engine.app.world_mut().insert_resource(MoveInput {
            move_forward: bounded(input.move_forward, 1.0),
            move_right: bounded(input.move_right, 1.0),
            move_up: bounded(input.move_up, 1.0),
            look_right_pixels: bounded(input.look_right_pixels, crate::host::policy::LOOK_LIMIT),
            look_down_pixels: bounded(input.look_down_pixels, crate::host::policy::LOOK_LIMIT),
        });
    }

    pub(crate) fn set_gathered(&mut self, input: &FrameInput) {
        let world = self.engine.app.world_mut();
        let wanted = world.resource::<MenuInput>().wanted;
        world.insert_resource(input.pointer);
        world.insert_resource(input.axes);
        world.insert_resource(MenuInput {
            wanted,
            ..input.menu
        });
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
        report: FrameReport,
        alpha: f32,
        interval: Option<f64>,
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
        let (dynamic_resolution, (render_width, render_height)) =
            self.size_frame(interval, (width, height));
        let (frame_number, tick) = {
            let world = self.engine.app.world();
            (
                world.resource::<engine_core::FrameTime>().frame,
                world.resource::<engine_core::FixedTime>().tick,
            )
        };
        let before_acquire = now_ms();
        let acquired = self.renderer.acquire_frame()?;
        let acquire_ms = now_ms() - before_acquire;
        let (extraction_ms, draw_ms, rendered) = if acquired {
            self.draw_frame((width, height), (render_width, render_height), alpha)?
        } else {
            (0.0, 0.0, RenderStats::default())
        };
        self.stats = FrameStats {
            frame: frame_number,
            tick,
            simulation_ms,
            extraction_ms,
            render_ms: acquire_ms + draw_ms,
            fixed_steps: report.fixed_steps,
            dropped_seconds: self.stats.dropped_seconds + report.dropped_seconds,
            cut_seconds: self.stats.cut_seconds + report.cut_seconds,
            pipelines_compiling: false,
            pipelines_listed: 0,
            pipelines_unlisted: 0,
            loading: false,
            resolution: self.resolution_stats((width, height)),
            dynamic_resolution,
            jobs: engine_core::JobStats::default(),
            camera_refusals: engine_core::CameraRefusals::default(),
            disabled_features: self.renderer.disabled_features(),
            render: rendered,
            bricks_cost: None,
            bricks: std::collections::BTreeMap::new(),
        };
        self.pipeline_list.after_frame(
            &mut self.engine,
            &self.renderer,
            self.stats.render.pipelines_created,
        );
        Ok(())
    }

    fn draw_frame(
        &mut self,
        (width, height): (u32, u32),
        (render_width, render_height): (u32, u32),
        alpha: f32,
    ) -> Result<(f64, f64, RenderStats)> {
        let before_extraction = now_ms();
        let aspect = width.max(1) as f32 / height.max(1) as f32;
        self.engine.start_frame();
        let world = self.engine.app.world();
        let shown = world
            .get_resource::<engine_render::DebugShown>()
            .copied()
            .unwrap_or_default();
        let debug = shown.next_debug;
        let post = self.renderer.post(world);
        let display_size = self.display_size((width, height));
        let render_size = if !debug && !shown.next_whole && post {
            glam::UVec2::new(render_width, render_height)
                .clamp(glam::UVec2::ONE, display_size.max(glam::UVec2::ONE))
        } else {
            display_size
        };
        let facts = FrameFacts {
            temporal: !debug && post && self.renderer.motion(world),
            capture: shown.next_whole,
            debug,
            display_size,
            render_size,
            pixel_ratio: self.pixel_ratio as f32,
        };
        self.engine.set_frame(facts);
        let frame = self.engine.extract_frame(aspect, alpha)?;
        let after_extraction = now_ms();
        let rendered = self.renderer.render_frame(frame)?;
        Ok((
            after_extraction - before_extraction,
            now_ms() - after_extraction,
            rendered,
        ))
    }

    fn engine_loading(&self) -> bool {
        let world = self.engine.app.world();
        world
            .get_resource::<Registry>()
            .is_some_and(|registry| registry.pending(world) > 0)
            || world
                .get_resource::<RecordExchange>()
                .is_some_and(RecordExchange::loading)
    }

    pub(crate) fn loading(&self) -> bool {
        self.engine_loading() || self.pipeline_list.waiting()
    }

    pub(crate) fn cook_features(&self) -> Vec<(&'static str, u64)> {
        let world = self.engine.app.world();
        let mut features: Vec<(&'static str, u64)> = world
            .get_resource::<Registry>()
            .into_iter()
            .flat_map(|registry| {
                registry
                    .stats()
                    .map(|entry| (entry.key, u64::from(entry.pending(world))))
            })
            .collect();
        features.push(("impostors", engine_core::assets_pending(world)));
        features.push(("pipelines", u64::from(self.pipeline_list.making())));
        features
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
            pipelines_compiling: self.renderer.pipelines_compiling(),
            pipelines_listed,
            pipelines_unlisted,
            loading: self.loading(),
            jobs: world
                .get_resource::<JobExchange>()
                .map(JobExchange::stats)
                .unwrap_or_default(),
            camera_refusals: world
                .get_resource::<engine_core::CameraRefusals>()
                .cloned()
                .unwrap_or_default(),
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
        self.last_timestamp_ms = None;
        self.dynamic.reset();
        self.engine.reset_clock();
        self.engine
            .app
            .world_mut()
            .insert_resource(MoveInput::default());
        self.set_gathered(&FrameInput::default());
    }
}
