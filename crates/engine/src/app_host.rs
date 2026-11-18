use engine_core::{
    BrickList, Devices, Engine, ErrorCode, FrameFacts, JobExchange, JobId, JobMessage, JobRef,
    PauseReason, RecordExchange, RenderStats,
};
use engine_registry::Registry;
use engine_render::{Picture, RenderTarget, Renderer};

use crate::frame_stats::{FrameStats, GpuTime, ResolutionStats};
use crate::host::InputSource;
use crate::host::policy::FRAME_INTERVAL_MS;
use crate::{Error, Result};

#[path = "app_attribution.rs"]
mod attribution;

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

fn wire(engine: &mut Engine, renderer: &Renderer, (workers, verifying): (u32, bool)) {
    let world = engine.app.world_mut();
    world.init_resource::<FrameStats>();
    if let Some(mut records) = world.get_resource_mut::<RecordExchange>() {
        records.set_sampling(Some(renderer.texture_sampling()));
    }
    if let Some(mut jobs) = world.get_resource_mut::<JobExchange>() {
        jobs.set_workers(workers);
        jobs.set_verifying(verifying);
    }
    crate::game::limit(engine, renderer.device_limits());
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Focus {
    Held,
    Lost,
}

#[derive(Clone, Copy, Debug)]
pub(crate) enum Advance {
    Clock(f64),
    #[cfg(feature = "dev")]
    Stepped { ticks: u32, alpha: Option<f32> },
}

pub(crate) struct AppHost {
    pub(crate) engine: Engine,
    pub(crate) renderer: Box<Renderer>,
    focus: Focus,
    preparation_ms: f64,
    size: (u32, u32),
    overlay: crate::host::SettingsLog,
    pipeline_list: crate::host::PipelineList,
    pixel_ratio: f64,
    refresh: f64,
    gpu_ms: GpuTime,
}

fn log_notice(error: &engine_core::Error) {
    let record = serde_json::json!({"event": "notice", "error": error});
    #[cfg(target_arch = "wasm32")]
    web_sys::console::info_1(&wasm_bindgen::JsValue::from_str(&record.to_string()));
    #[cfg(not(target_arch = "wasm32"))]
    eprintln!("{record}");
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
        let install = engine_profile::open("launch", "install");
        let mut engine = engine_profile::caused(install, || {
            built(|| {
                let mut engine = crate::engine(bricks)?;
                build(&mut engine)?;
                Ok(engine)
            })
        })?;
        let mut start = std::pin::pin!(Renderer::start(target));
        let started =
            std::future::poll_fn(|cx| std::task::Poll::Ready(std::future::Future::poll(start.as_mut(), cx))).await;
        let ahead = engine_profile::caused(install, || Renderer::ahead(engine.app.world_mut()));
        let starting = match started {
            std::task::Poll::Ready(starting) => starting,
            std::task::Poll::Pending => start.await,
        }?;
        let renderer = Box::new(starting.finish());
        let mut host = Self::with(engine, renderer);
        engine_profile::caused(install, || host.claim_passes(ahead))?;
        for error in host.engine.notices() {
            log_notice(error);
        }
        Ok(host)
    }

    fn with(mut engine: Engine, renderer: Box<Renderer>) -> Self {
        #[cfg(any(feature = "dev", feature = "profiling"))]
        engine_profile::install_clock(now_ms);
        wire(&mut engine, &renderer, (0, false));
        let gpu_ms = if renderer
            .features()
            .contains(engine_render::wgpu::Features::TIMESTAMP_QUERY)
        {
            GpuTime::default()
        } else {
            GpuTime::Unavailable {
                unavailable: "timestamp_query_unavailable",
            }
        };
        Self {
            engine,
            renderer,
            focus: Focus::Held,
            preparation_ms: 0.0,
            size: (0, 0),
            overlay: crate::host::SettingsLog::default(),
            pipeline_list: crate::host::PipelineList::default(),
            pixel_ratio: 1.0,
            refresh: FRAME_INTERVAL_MS,
            gpu_ms,
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

    pub(crate) fn take_pipelines(&mut self) {
        self.pipeline_list
            .take(&mut self.engine, &mut self.renderer);
    }

    pub(crate) fn cook(&mut self) {
        engine_core::cook(self.engine.app.world_mut());
    }

    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    pub(crate) fn reload_shaders(&mut self) {
        self.renderer.reload_shaders(self.engine.app.world_mut());
    }

    pub(crate) fn reloaded(&mut self) {
        self.pipeline_list.restart();
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

    pub fn set_pixel_ratio(&mut self, ratio: f64) {
        self.pixel_ratio = if ratio.is_finite() && ratio > 0.0 {
            ratio
        } else {
            1.0
        };
    }

    pub fn set_refresh_interval(&mut self, milliseconds: Option<f64>) {
        if let Some(ms) = milliseconds.filter(|ms| ms.is_finite() && *ms > 0.0) {
            self.refresh = ms;
        }
    }

    pub fn refresh_interval(&self) -> f64 {
        self.refresh
    }

    pub(crate) fn set_preparation_ms(&mut self, milliseconds: f64) {
        self.preparation_ms = milliseconds;
    }

    fn report_timing(&mut self) {
        if let Some((ms, _)) = self.renderer.take_frame_time() {
            self.gpu_ms = GpuTime::Ms(ms as f32);
        }
    }

    fn resolution_stats(&self, (width, height): (u32, u32)) -> ResolutionStats {
        let (display_width, display_height) = self.renderer.view_size().unwrap_or((0, 0));
        ResolutionStats {
            display_width,
            display_height,
            dropped_pixels: (u64::from(width) * u64::from(height))
                .saturating_sub(u64::from(display_width) * u64::from(display_height)),
            gpu_ms: self.gpu_ms,
        }
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

    pub(crate) fn frame(
        &mut self,
        advance: Advance,
        size: (u32, u32),
        picture: Picture,
    ) -> Result<bool> {
        engine_profile::set_frame(self.engine.frame_time().frame + 1);
        let _profile = engine_profile::span("cpu", "host.frame");
        self.deliver();
        let start = now_ms();
        let fixed_steps = self.advance(advance)?;
        engine_profile::set_frame(self.engine.frame_time().frame);
        let simulation_ms = now_ms() - start;
        let drawn = self.render_frame(size, (simulation_ms, fixed_steps), picture)?;
        self.engine.exchange();
        Ok(drawn)
    }

    fn advance(&mut self, advance: Advance) -> Result<u32> {
        match advance {
            Advance::Clock(timestamp_ms) => {
                self.engine.frame(timestamp_ms * 0.001)?;
                Ok(self.engine.frame_time().ticks)
            }
            #[cfg(feature = "dev")]
            Advance::Stepped { ticks, alpha } => {
                if ticks == 0 {
                    self.engine.frame_without_ticks();
                }
                let mut fixed_steps = 0;
                for tick in 0..ticks {
                    if tick > 0 {
                        self.engine.start_frame();
                        engine_core::hold_input(self.engine.app.world_mut());
                    }
                    self.engine.step_fixed(1)?;
                    fixed_steps += self.engine.frame_time().ticks;
                }
                if let Some(alpha) = alpha {
                    self.engine
                        .present(engine_core::Presented::Between { alpha })?;
                }
                Ok(fixed_steps)
            }
        }
    }

    pub(crate) fn accept(&mut self, input: Devices) {
        let _profile = engine_profile::span("cpu", "host.input");
        engine_core::accept_input(self.engine.app.world_mut(), input);
    }

    fn display_size(&self, (width, height): (u32, u32)) -> glam::UVec2 {
        let (width, height) = self.renderer.view_size().unwrap_or((width, height));
        glam::UVec2::new(width, height)
    }

    fn render_frame(
        &mut self,
        (width, height): (u32, u32),
        (simulation_ms, fixed_steps): (f64, u32),
        picture: Picture,
    ) -> Result<bool> {
        if self.size != (width, height) {
            self.renderer.resize(width, height);
            self.size = (width, height);
        }
        self.report_timing();
        let time = self.engine.frame_time();
        let tick = self
            .engine
            .app
            .world()
            .resource::<engine_core::FixedTime>()
            .tick;
        let before_acquire = now_ms();
        let acquired = self.renderer.acquire_frame(picture)?;
        let acquire_ms = now_ms() - before_acquire;
        let (extraction_ms, draw_ms, rendered) = if acquired {
            self.draw_frame((width, height), picture)?
        } else {
            (0.0, 0.0, RenderStats::default())
        };
        let (pipelines_listed, pipelines_unlisted) = self.renderer.pipelines_listed();
        let world = self.engine.app.world();
        let summary = FrameStats {
            frame: time.frame,
            tick,
            simulation_ms,
            extraction_ms,
            render_ms: acquire_ms + draw_ms,
            fixed_steps,
            preparation_ms: self.preparation_ms,
            dropped_seconds: time.dropped_seconds,
            cut_seconds: time.cut_seconds,
            pipelines_listed,
            pipelines_unlisted,
            readiness: self.engine.readiness().clone(),
            resolution: self.resolution_stats((width, height)),
            jobs: world
                .get_resource::<JobExchange>()
                .map(JobExchange::stats)
                .unwrap_or_default(),
            refused_views: world
                .get_resource::<engine_core::Views>()
                .map_or(0, |views| views.refused().len() as u32),
            render: rendered,
        };
        *self.engine.app.world_mut().resource_mut::<FrameStats>() = summary;
        if acquired && picture == Picture::Shown && self.engine.readiness().settled() {
            self.renderer.settle_pipelines();
        }
        self.pipeline_list
            .after_frame(&mut self.engine, &self.renderer);
        Ok(acquired)
    }

    fn draw_frame(
        &mut self,
        (width, height): (u32, u32),
        picture: Picture,
    ) -> Result<(f64, f64, RenderStats)> {
        let before_extraction = now_ms();
        self.engine.start_frame();
        let display_size = self.display_size((width, height));
        let facts = FrameFacts {
            display_size,
            pixel_ratio: self.pixel_ratio as f32,
        };
        self.engine.set_frame(facts);
        self.collect_pipelines();
        let frame = self.engine.extract_frame()?;
        let after_extraction = now_ms();
        let rendered = self.renderer.render_frame(frame)?;
        let reports = self.renderer.take_draw_reports().unwrap_or_default();
        self.engine.report_drawing(reports);
        self.collect_pipelines();
        self.report_shader_errors();
        self.report_notices();
        if picture == Picture::Hidden {
            self.cut_shown_views();
        }
        Ok((
            after_extraction - before_extraction,
            now_ms() - after_extraction,
            rendered,
        ))
    }

    fn cut_shown_views(&mut self) {
        let world = self.engine.app.world_mut();
        let shown: Vec<bevy_ecs::entity::Entity> = world
            .get_resource::<engine_core::Views>()
            .map(|views| {
                views
                    .listed()
                    .iter()
                    .filter(|facts| facts.target.surface())
                    .map(|facts| facts.view)
                    .collect()
            })
            .unwrap_or_default();
        for view in shown {
            engine_core::send_ask(world, engine_core::ViewCut { view });
        }
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

    fn report_notices(&mut self) {
        let notices = self.renderer.take_notices();
        if notices.is_empty() {
            return;
        }
        let world = self.engine.app.world_mut();
        let Some(mut registry) = world.get_resource_mut::<Registry>() else {
            return;
        };
        for notice in notices {
            if !registry.notices().contains(&notice) {
                log_notice(&notice);
                registry.report_notice(notice);
            }
        }
    }

    pub fn job_list(&self) -> std::sync::Arc<engine_registry::JobList> {
        self.engine.job_list()
    }

    pub fn frame_times(&self) -> &FrameStats {
        self.engine.app.world().resource::<FrameStats>()
    }

    pub fn progress(&self) -> (engine_core::Readiness, engine_core::JobStats) {
        let readiness = self.engine.readiness();
        let stood_in = self.renderer.pipelines_stood_in();
        let readiness = engine_core::Readiness {
            compiling: readiness.compiling.saturating_add(stood_in),
            ..readiness.clone()
        };
        let jobs = self
            .engine
            .app
            .world()
            .get_resource::<JobExchange>()
            .map(JobExchange::stats)
            .unwrap_or_default();
        (readiness, jobs)
    }

    pub fn suspend(&mut self) {
        self.engine.reset_clock();
        engine_core::hold_input(self.engine.app.world_mut());
    }
}
