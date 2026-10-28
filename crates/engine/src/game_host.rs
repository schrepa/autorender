use crate::app_host::AppHost;
use crate::game::Game;
use crate::host::{FrameLoop, FramePhase, InputGatherer, JobOrder, JobScheduler};
use engine_core::{ActionInput, Engine, ErrorCode, JobId};
use engine_render::RenderTarget;
use engine_scene::SceneFiles;

pub struct GameHost<G: Game> {
    app: AppHost,
    game: G,
    input: InputGatherer,
    frames: FrameLoop,
    jobs: JobScheduler,
    #[cfg(feature = "dev")]
    development: crate::dev_host::DevHost,
    #[cfg(feature = "dev")]
    pub(crate) dev_pending: Option<crate::dev_host::bridge::Pending>,
}

impl<G: Game> GameHost<G> {
    pub async fn launch<E: Into<crate::Error>>(
        target: RenderTarget,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> crate::Result<Self> {
        engine_panic::install();
        let input = InputGatherer::new(G::ACTIONS)
            .map_err(|message| crate::Error::new(ErrorCode::InvalidInput, "$.actions", message))?;
        let mut game = None;
        let app = AppHost::launch(target, || {
            let (engine, built) = build().map_err(Into::into)?;
            game = Some(built);
            Ok(engine)
        })
        .await?;
        let game = game.expect("launching built the game");
        #[cfg(feature = "dev")]
        let development = {
            let mut development = crate::dev_host::DevHost::default();
            G::inspectors(&mut crate::game::Inspectors::new(&mut development.tools));
            development
        };
        Ok(Self {
            app,
            game,
            input,
            frames: FrameLoop::default(),
            jobs: JobScheduler::default(),
            #[cfg(feature = "dev")]
            development,
            #[cfg(feature = "dev")]
            dev_pending: None,
        })
    }

    pub fn engine(&self) -> &Engine {
        self.app.engine()
    }

    pub fn cook(&mut self) {
        self.app
            .engine_mut()
            .app
            .world_mut()
            .insert_resource(engine_core::Cooking::default());
    }

    pub fn cooking(&self) -> Option<&engine_core::Cooking> {
        self.app
            .engine()
            .app
            .world()
            .get_resource::<engine_core::Cooking>()
    }

    pub fn jobs_held(&self) -> u32 {
        self.jobs.held()
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        self.app.engine_mut()
    }

    pub fn game(&self) -> &G {
        &self.game
    }

    pub fn game_mut(&mut self) -> &mut G {
        &mut self.game
    }

    pub fn parts_mut(&mut self) -> (&mut Engine, &mut G) {
        (self.app.engine_mut(), &mut self.game)
    }

    pub fn replace_engine(&mut self, engine: Engine) {
        self.app.replace_engine(engine);
        self.suspend();
    }

    pub fn frame(&mut self, timestamp: f64, width: u32, height: u32) -> crate::Result<()> {
        self.prepare_pipelines();
        let input = self.input.take();
        let engine = self.app.engine_mut();
        self.game.before_frame(engine);
        engine.app.world_mut().resource_mut::<ActionInput>().ingest(
            input.held,
            input.pressed,
            input.released,
        );
        self.app.frame(timestamp, width, height, input.state)
    }

    pub fn animation_frame(
        &mut self,
        timestamp: f64,
        width: u32,
        height: u32,
    ) -> crate::Result<FramePhase> {
        let mut frames = std::mem::take(&mut self.frames);
        let phase = frames.frame(self, timestamp, (width, height));
        self.frames = frames;
        phase
    }

    pub fn prepare_pipelines(&mut self) -> bool {
        let game = &self.game;
        self.app.prepare_pipelines(|engine| game.content(engine))
    }

    pub fn work_while_compiling(&mut self, width: u32, height: u32) -> bool {
        if !self.frames.preparing() || !self.frame_stats().pipelines_compiling {
            return false;
        }
        self.work(width, height);
        true
    }

    pub fn load_between_frames(&mut self) -> bool {
        let mut frames = std::mem::take(&mut self.frames);
        let remaining = frames.between(self);
        self.frames = frames;
        remaining
    }

    pub fn set_preparation(&mut self, enabled: bool) {
        self.frames.set_preparation(enabled);
    }

    pub fn preparing(&self) -> bool {
        self.frames.preparing()
    }

    pub fn set_between_frames(&mut self, enabled: bool) {
        self.frames.set_between_frames(enabled);
    }

    pub fn key(&mut self, code: &str, down: bool) -> bool {
        self.input.key(code, down)
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.input.look(dx, dy);
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    pub fn work(&mut self, width: u32, height: u32) {
        self.app.work(width, height);
    }

    pub fn work_between_frames(&mut self, milliseconds: f64) -> bool {
        self.app.work_between_frames(milliseconds)
    }

    pub fn set_job_workers(&mut self, workers: u32) {
        self.jobs.set_workers(workers);
        self.app.set_job_workers(self.jobs.working());
    }

    pub fn set_job_cache(&mut self, caching: bool, verify: bool) {
        self.jobs.set_cache(caching, verify);
        self.app.set_job_verifying(caching && verify);
    }

    pub fn take_jobs(&mut self) -> Vec<JobOrder> {
        let messages = self.app.take_jobs();
        let mut finished = Vec::new();
        let mut orders = self.jobs.take(messages, &mut finished);
        let here = self.jobs.take_here();
        if !here.is_empty() {
            for request in here {
                let output = self.app.run_job_here(request.kind, &request.input);
                self.jobs.ran_here(request.id, output, &mut finished);
            }
            orders.extend(self.jobs.take(Vec::new(), &mut finished));
        }
        self.finish_jobs(finished);
        orders
    }

    pub fn complete_job(&mut self, id: JobId, output: Vec<u8>) {
        let mut finished = Vec::new();
        self.jobs.done(id, output, &mut finished);
        self.finish_jobs(finished);
    }

    pub fn fail_job(&mut self, id: JobId, error: String) {
        let mut finished = Vec::new();
        self.jobs.failed(id, error, &mut finished);
        self.finish_jobs(finished);
    }

    pub fn job_kept(&mut self, id: JobId, key: String, output: Option<Vec<u8>>) {
        let mut finished = Vec::new();
        self.jobs.kept(id, key, output, &mut finished);
        self.finish_jobs(finished);
    }

    pub fn lose_job_worker(&mut self, worker: u32, error: &str, replaced: bool) {
        let mut finished = Vec::new();
        self.jobs.lost(worker, error, replaced, &mut finished);
        self.finish_jobs(finished);
        self.app.set_job_workers(self.jobs.working());
    }

    pub fn job_task(&mut self) -> Option<u32> {
        self.jobs.task()
    }

    pub fn job_task_done(&mut self, worker: u32) {
        self.jobs.task_done(worker);
    }

    fn finish_jobs(&mut self, finished: Vec<crate::host::Finished>) {
        for (id, result) in finished {
            match result {
                Ok(output) => self.app.complete_job(id, output),
                Err(error) => self.app.fail_job(id, error),
            }
        }
        self.app.report_jobs_held(self.jobs.held());
    }

    pub fn report_job_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.app.report_job_cache(hits, misses, bytes, mismatches);
    }

    pub fn suspend(&mut self) {
        self.app.suspend();
        self.input.clear();
        self.frames.skip();
        #[cfg(feature = "dev")]
        self.development.clear_input();
    }

    pub fn reset_camera(&mut self) {
        self.game.reset_camera(self.app.engine_mut());
        self.suspend();
    }

    pub fn reload_scene(&mut self, source: &str, files: &SceneFiles) -> crate::Result<()> {
        self.game
            .reload_scene(self.app.engine_mut(), source, files)
            .map_err(crate::Error::from)?;
        self.app.apply_overrides();
        self.app.forget_pipeline_list();
        self.suspend();
        Ok(())
    }

    pub fn set_gpu_presentation(&mut self, enabled: bool) {
        self.app.set_gpu_presentation(enabled);
    }

    pub fn set_tick_pipelining(&mut self, enabled: bool) {
        self.app.set_tick_pipelining(enabled);
    }

    pub fn set_temporal_antialiasing(&mut self, enabled: bool) {
        self.app.set_temporal_antialiasing(enabled);
    }

    pub fn set_occlusion_culling(&mut self, enabled: bool) {
        self.app.set_occlusion_culling(enabled);
    }

    pub fn set_subpixel_culling(&mut self, enabled: bool) {
        self.app.set_subpixel_culling(enabled);
    }

    pub fn set_pixel_ratio(&mut self, ratio: f64) {
        self.app.set_pixel_ratio(ratio);
    }

    pub fn set_refresh_interval(&mut self, milliseconds: Option<f64>) {
        self.app.set_refresh_interval(milliseconds);
    }

    pub fn set_resolution_scale(&mut self, scale: f64) {
        self.app.set_resolution_scale(scale);
    }

    pub fn set_dynamic_resolution(&mut self, enabled: bool) {
        self.app.set_dynamic_resolution(enabled);
    }

    pub fn set_debug_view(&mut self, view: engine_core::DebugViewMode) {
        self.app.set_debug_view(view);
    }

    pub fn set_terrain_clock(&mut self, clock: crate::host::TerrainClock) {
        self.app.set_terrain_clock(clock);
    }

    pub fn adapter(&self) -> &crate::AdapterInfo {
        self.app.renderer.adapter()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_debugger_capture(&mut self) -> bool {
        self.app.renderer.start_debugger_capture()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn stop_debugger_capture(&mut self) -> crate::Result<()> {
        self.app
            .renderer
            .stop_debugger_capture()
            .map_err(crate::app_host::render_failure)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_gpu(&self) -> crate::Result<()> {
        self.app
            .renderer
            .wait_for_gpu()
            .map_err(crate::app_host::render_failure)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_mark(&self) -> crate::GpuMark {
        self.app.renderer.gpu_mark()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_mark(&self, mark: crate::GpuMark) -> crate::Result<()> {
        self.app
            .renderer
            .wait_for_mark(mark)
            .map_err(crate::app_host::render_failure)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_offscreen(&self) -> crate::Result<Option<(u32, u32, Vec<u8>)>> {
        self.app
            .renderer
            .read_offscreen()
            .map_err(crate::app_host::render_failure)
    }

    pub fn frame_stats(&self) -> engine_core::FrameStats {
        self.app.stats()
    }

    pub fn stats(&self) -> String {
        self.app.stats_json()
    }

    pub fn gpu_presentation(&self) -> bool {
        self.app.gpu_presentation()
    }

    pub fn tick_pipelining(&self) -> bool {
        self.app.tick_pipelining()
    }

    pub fn temporal_antialiasing(&self) -> bool {
        self.app.temporal_antialiasing()
    }

    pub fn occlusion_culling(&self) -> bool {
        self.app.occlusion_culling()
    }

    pub fn subpixel_culling(&self) -> bool {
        self.app.subpixel_culling()
    }

    pub fn dynamic_resolution(&self) -> bool {
        self.app.dynamic_resolution()
    }
}

#[cfg(feature = "dev")]
impl<G: Game> GameHost<G> {
    pub(crate) fn lose_device(&mut self) {
        self.app.renderer.lose_device();
    }

    pub(crate) fn arm_probe(&mut self, pixel: (u32, u32)) {
        self.app.renderer.arm_probe(pixel);
    }

    pub(crate) fn probe_result(
        &mut self,
    ) -> Option<Result<engine_dev::results::ProbeResult, String>> {
        use engine_core::DrawnKind;
        let probe = match self.app.renderer.probe_result()? {
            Ok(probe) => probe,
            Err(error) => return Some(Err(error.to_string())),
        };
        let slot = probe
            .object
            .filter(|_| matches!(probe.kind, Some(DrawnKind::Mesh | DrawnKind::Card)));
        let engine = self.app.engine();
        let entity = slot
            .and_then(|slot| engine.render_entity(slot))
            .map(|entity| engine_dev::results::AuthoredEntity::of(engine.app.world(), entity));
        Some(Ok(engine_dev::results::ProbeResult { probe, entity }))
    }

    pub fn dev_command(&mut self, request: &str, width: u32, height: u32, paused: bool) -> String {
        if self.development.submits_frame(request, paused) {
            self.game.before_frame(self.app.engine_mut());
        }
        self.development.command(
            &mut self.app,
            self.game.scene(),
            request,
            width,
            height,
            paused,
        )
    }

    pub fn dev_reload(
        &mut self,
        source: &str,
        expected_revision: &str,
        revision: &str,
        files: Option<&SceneFiles>,
    ) -> String {
        use engine_core::ErrorCode;
        use engine_scene::{SceneDocument, SceneError};
        let none = SceneFiles::default();
        let files = files.unwrap_or(&none);
        let result = (|| {
            let Some(scene) = self.game.scene() else {
                return Err(SceneError {
                    code: ErrorCode::UnsupportedCommand,
                    path: "$.op".into(),
                    message: "this game runs no authored scene to reload".into(),
                });
            };
            let (Ok(expected), Ok(target)) =
                (expected_revision.parse::<u64>(), revision.parse::<u64>())
            else {
                return Err(SceneError {
                    code: ErrorCode::InvalidRevision,
                    path: "$.revision".into(),
                    message: "revisions must be unsigned decimal strings".into(),
                });
            };
            if scene.revision() != expected {
                return Err(SceneError {
                    code: ErrorCode::RevisionConflict,
                    path: "$.expected_revision".into(),
                    message: format!("active revision is {}", scene.revision()),
                });
            }
            let document = SceneDocument::from_json(source)?;
            if document.id != scene.id() || document.revision != target {
                return Err(SceneError {
                    code: ErrorCode::RevisionConflict,
                    path: "$.revision".into(),
                    message: "saved scene ID and revision must match the requested scene".into(),
                });
            }
            self.game.reload_scene(self.app.engine_mut(), source, files)
        })();
        match result {
            Ok(()) => {
                self.app.apply_overrides();
                self.app.forget_pipeline_list();
                self.suspend();
                self.development.result(
                    &mut self.app,
                    self.game.scene(),
                    serde_json::json!({"reloaded": true, "saved_by_reload": false}),
                )
            }
            Err(error) => self.development.error(
                &mut self.app,
                self.game.scene(),
                error.code,
                &error.path,
                &error.message,
            ),
        }
    }
}

#[cfg(feature = "profiling")]
impl<G: Game> GameHost<G> {
    pub fn profiling_start(&mut self, max_events: usize) -> Result<(), &'static str> {
        self.app.profiling_start(max_events)
    }

    pub fn profiling_stop(&mut self) -> Result<serde_json::Value, &'static str> {
        self.app.profiling_stop()
    }

    pub fn profiling_poll(&mut self) {
        self.app.profiling_poll();
    }

    pub fn profiling_report(&mut self) -> serde_json::Value {
        self.app.profiling_report()
    }

    pub fn profiling_status(&self) -> serde_json::Value {
        self.app.profiling_status()
    }
}
