use crate::app_host::AppHost;
use crate::game::Game;
use crate::host::{
    FrameInput, FrameLoop, FramePhase, InputGatherer, JobId, JobOrder, JobScheduler, LevelLoad,
    LevelStep,
};
use engine_core::{
    ActionInput, BrickList, CookMode, CursorMode, Engine, ErrorCode, LevelProgress, LevelRequest,
    MenuInput, MoveInput, WorkRemaining,
};
use engine_render::RenderTarget;
use engine_scene::SceneFiles;

pub struct GameHost<G: Game> {
    app: AppHost,
    game: G,
    input: InputGatherer,
    frames: FrameLoop,
    jobs: JobScheduler,
    level: LevelLoad,
    #[cfg(feature = "dev")]
    development: crate::dev_host::DevHost,
    #[cfg(feature = "dev")]
    pub(crate) dev_pending: Option<crate::dev_host::bridge::Pending>,
}

impl<G: Game> GameHost<G> {
    pub async fn launch<E: Into<crate::Error>>(
        target: RenderTarget,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
    ) -> crate::Result<Self> {
        engine_panic::install();
        let input = InputGatherer::new(G::ACTIONS, G::AXES, G::MOVE_KEYS)?;
        let mut game = None;
        let app = AppHost::launch(target, bricks, |engine| {
            game = Some(build(engine).map_err(Into::into)?);
            Ok(())
        })
        .await?;
        let game = game.expect("game_built");
        #[cfg(feature = "dev")]
        let development = {
            let mut development = crate::dev_host::DevHost::default();
            development.gatherer = Some(input.clone());
            G::inspectors(&mut crate::game::Inspectors::new(&mut development.tools));
            development
        };
        Ok(Self {
            app,
            game,
            input,
            frames: FrameLoop::default(),
            jobs: JobScheduler::default(),
            level: LevelLoad::default(),
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
            .insert_resource(CookMode { cooking: true });
    }

    pub fn cook_features(&self) -> Vec<(&'static str, u64)> {
        self.app.cook_features()
    }

    pub fn cook_left(&self) -> u64 {
        self.cook_features().iter().map(|(_, left)| left).sum()
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

    pub fn take_level_request(&mut self) -> Option<String> {
        self.app
            .engine_mut()
            .app
            .world_mut()
            .resource_mut::<LevelRequest>()
            .path
            .take()
    }

    pub fn report_level(&mut self, done: u32, total: u32) {
        self.show_level(LevelProgress::Reading { done, total });
    }

    pub fn fail_level(&mut self, message: &str) {
        self.level_failed(crate::Error::new(ErrorCode::SourceIo, "$.path").got(message));
    }

    fn level_failed(&mut self, error: crate::Error) {
        self.level = LevelLoad::Idle;
        self.show_level(LevelProgress::Failed { error });
    }

    fn show_level(&mut self, progress: LevelProgress) {
        self.app
            .engine_mut()
            .app
            .world_mut()
            .insert_resource(progress);
    }

    fn level_built(&mut self) {
        let world = self.app.engine().app.world();
        if *world.resource::<LevelProgress>() != LevelProgress::Idle {
            self.level = LevelLoad::Building { quiet: 0 };
            self.show_level(LevelProgress::Building);
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn open_level(&mut self) {
        let Some(path) = self.take_level_request() else {
            return;
        };
        let kinds = engine_scene::SceneKinds::of(self.app.engine().app.world());
        match LevelLoad::open(&path, &kinds) {
            Ok(level) => {
                self.level = level;
                self.report_level(0, 0);
            }
            Err(error) => self.level_failed(error),
        }
    }

    fn step_level(&mut self) {
        #[cfg(not(target_arch = "wasm32"))]
        self.open_level();
        if matches!(self.level, LevelLoad::Idle) {
            return;
        }
        let world = self.app.engine().app.world();
        let busy = self.app.loading() || world.resource::<WorkRemaining>().remaining();
        match self.level.step(busy) {
            Ok(LevelStep::Progress(progress)) => self.show_level(progress),
            #[cfg(not(target_arch = "wasm32"))]
            Ok(LevelStep::Ready { source, files }) => {
                if let Err(error) = self.reload_scene(&source, &files) {
                    self.level_failed(error);
                }
            }
            Err(error) => self.level_failed(error),
        }
    }

    pub fn frame(&mut self, timestamp: f64, width: u32, height: u32) -> crate::Result<()> {
        self.step_level();
        self.prepare_pipelines();
        let input = self.frame_input(timestamp);
        self.app.set_gathered(input);
        let engine = self.app.engine_mut();
        self.game.before_frame(engine);
        let _ = engine
            .app
            .world_mut()
            .modify_resource(|actions: &mut ActionInput| {
                actions.ingest(input.held, input.pressed, input.released);
            });
        let state = if engine.app.world().resource::<MenuInput>().wanted {
            MoveInput::default()
        } else {
            input.state
        };
        self.app.frame(timestamp, width, height, state)
    }

    fn frame_input(&mut self, timestamp: f64) -> FrameInput {
        #[cfg(feature = "dev")]
        if self
            .dev_pending
            .as_ref()
            .is_some_and(crate::dev_host::bridge::Pending::takes_agent_input)
        {
            return self.development.take_input();
        }
        self.input.take(timestamp)
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
        let world = self.app.engine().app.world();
        let wanted = world.resource::<MenuInput>().wanted;
        self.input.key(code, down, wanted)
    }

    pub fn look(&mut self, dx: f32, dy: f32) {
        self.input.look(dx, dy);
    }

    pub fn motion(&mut self, dx: f32, dy: f32) {
        self.input.motion(dx, dy);
    }

    pub fn pointer(&mut self, x: f32, y: f32) {
        self.input.pointer(x, y);
    }

    pub fn pointer_left(&mut self) {
        self.input.pointer_left();
    }

    pub fn wheel(&mut self, right_notches: f32, up_notches: f32) {
        self.input.wheel(right_notches, up_notches);
    }

    pub fn pad(&mut self, buttons: u32, axes: [f32; 6]) {
        self.input.pad(buttons, axes);
    }

    pub fn cursor_mode(&self) -> CursorMode {
        *self.app.engine().app.world().resource::<CursorMode>()
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
                let output = self.app.run_job_here(&request.kind, &request.input);
                self.jobs.ran_here(request.id, output, &mut finished);
            }
            orders.extend(self.jobs.take(Vec::new(), &mut finished));
        }
        self.finish_jobs(finished);
        orders
    }

    pub fn complete_job(&mut self, id: JobId, output: std::sync::Arc<[u8]>) {
        let mut finished = Vec::new();
        self.jobs.done(id, output, &mut finished);
        self.finish_jobs(finished);
    }

    pub fn fail_job(&mut self, id: JobId, error: String) {
        let mut finished = Vec::new();
        self.jobs.failed(id, error, &mut finished);
        self.finish_jobs(finished);
    }

    pub fn job_kept(&mut self, id: JobId, key: String, output: Option<std::sync::Arc<[u8]>>) {
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

    pub fn job_list(&self) -> std::sync::Arc<engine_registry::JobList> {
        self.app.job_list()
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
        self.suspend_live();
        #[cfg(feature = "dev")]
        self.development.clear_input();
    }

    pub(crate) fn suspend_live(&mut self) {
        self.app.suspend();
        self.input.clear();
        self.frames.skip();
    }

    pub fn reset_camera(&mut self) {
        self.game.reset_camera(self.app.engine_mut());
        self.suspend();
    }

    pub fn reload_scene(&mut self, source: &str, files: &SceneFiles) -> crate::Result<()> {
        self.game
            .reload_scene(self.app.engine_mut(), source, files)?;
        self.app.replay_settings()?;
        self.app.forget_pipeline_list();
        self.suspend();
        self.level_built();
        Ok(())
    }

    pub fn set_setting(&mut self, key: &str, value: serde_json::Value) -> crate::Result<()> {
        self.app.set_setting(key, value)
    }

    pub fn set_pixel_ratio(&mut self, ratio: f64) {
        self.app.set_pixel_ratio(ratio);
    }

    pub fn set_refresh_interval(&mut self, milliseconds: Option<f64>) {
        self.app.set_refresh_interval(milliseconds);
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
        self.app.renderer.stop_debugger_capture()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_allocations(&self) -> serde_json::Value {
        self.app.renderer.gpu_allocations()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_gpu(&self) -> crate::Result<()> {
        self.app.renderer.wait_for_gpu()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_mark(&self) -> crate::GpuMark {
        self.app.renderer.gpu_mark()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_mark(&self, mark: crate::GpuMark) -> crate::Result<()> {
        self.app.renderer.wait_for_mark(mark)
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_offscreen(&self) -> crate::Result<Option<(u32, u32, Vec<u8>)>> {
        self.app.renderer.read_offscreen()
    }

    pub fn frame_stats(&self) -> engine_core::FrameStats {
        self.app.stats()
    }

    pub fn frame_times(&self) -> &engine_core::FrameStats {
        self.app.frame_times()
    }

    pub fn eye(&self) -> Option<[f32; 3]> {
        let world = self.app.engine().app.world();
        let view = world.get_resource::<engine_core::View>()?;
        Some(view.detail_eye().to_array())
    }

    pub fn stats(&self) -> String {
        self.app.stats_json()
    }
}

#[cfg(feature = "dev")]
impl<G: Game> GameHost<G> {
    pub(crate) fn lose_device(&mut self) {
        self.app.renderer.lose_device();
    }

    pub(crate) fn job_workers(&self) -> u32 {
        self.jobs.working()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) fn deliver_readbacks(&mut self) {
        self.app.deliver();
    }

    pub(crate) fn dev_tools(&self) -> &engine_dev::DevTools {
        &self.development.tools
    }

    pub(crate) fn dev_scene(&self) -> Option<&engine_scene::SceneInstance> {
        self.game.scene()
    }

    pub(crate) fn dev_handle(
        &mut self,
        request: &str,
        size: (u32, u32),
        paused: bool,
    ) -> crate::dev_host::Handled {
        if self
            .development
            .submits_frame(self.app.engine().app.world(), request, paused)
        {
            self.game.before_frame(self.app.engine_mut());
        }
        self.development
            .command(&mut self.app, self.game.scene(), request, size, paused)
    }

    pub(crate) fn dev_run(
        &mut self,
        call: crate::dev_host::Call<'_>,
        size: (u32, u32),
        paused: bool,
    ) -> serde_json::Value {
        if let crate::dev_host::Call::Control(name, args) = &call
            && self.development.submits(
                self.app.engine().app.world(),
                (*name, args.clone()),
                paused,
            )
        {
            self.game.before_frame(self.app.engine_mut());
        }
        self.development
            .call(&mut self.app, self.game.scene(), call, size, paused)
    }

    pub(crate) fn dev_reload(
        &mut self,
        source: &str,
        expected_revision: &str,
        revision: &str,
        files: Option<&SceneFiles>,
    ) -> serde_json::Value {
        let none = SceneFiles::default();
        let files = files.unwrap_or(&none);
        let result = reload_checked(
            &mut self.game,
            self.app.engine_mut(),
            source,
            (expected_revision, revision),
            files,
        )
        .and_then(|()| self.app.replay_settings());
        match result {
            Ok(()) => {
                self.app.forget_pipeline_list();
                self.suspend();
                let skipped = self.game.scene().map(|scene| scene.skipped().to_vec());
                let mut reloaded = serde_json::json!({"reloaded": true, "saved_by_reload": false});
                if let Some(skipped) = skipped.filter(|skipped| !skipped.is_empty()) {
                    reloaded["skipped"] = serde_json::json!(skipped);
                }
                self.development
                    .result(&mut self.app, self.game.scene(), reloaded)
            }
            Err(error) => self
                .development
                .refusal(&mut self.app, self.game.scene(), &error),
        }
    }
}

#[cfg(feature = "profiling")]
impl<G: Game> GameHost<G> {
    pub fn profiling_start(&mut self, max_events: usize) -> crate::Result<()> {
        self.app.profiling_start(max_events)
    }

    pub fn profiling_stop(&mut self) -> crate::Result<serde_json::Value> {
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

    pub fn profile_bricks(&self) -> Vec<&'static str> {
        self.app.profile_bricks()
    }
}

#[cfg(feature = "dev")]
pub(crate) fn reload_checked<G: Game>(
    game: &mut G,
    engine: &mut Engine,
    source: &str,
    (expected_revision, revision): (&str, &str),
    files: &SceneFiles,
) -> crate::Result<()> {
    use engine_core::{Error, ErrorCode};
    use engine_scene::{SceneDocument, SceneKinds};
    use engine_settings::Expected;
    let Some(scene) = game.scene() else {
        return Err(Error::new(ErrorCode::UnsupportedCommand, "$.name").got("reload"));
    };
    let (Ok(expected), Ok(target)) = (expected_revision.parse::<u64>(), revision.parse::<u64>())
    else {
        return Err(Error::new(ErrorCode::InvalidRevision, "$.args.revision")
            .expected(Expected::Integer)
            .got(revision));
    };
    if scene.revision() != expected {
        return Err(
            Error::new(ErrorCode::RevisionConflict, "$.args.expected_revision")
                .got(expected_revision)
                .nearest(scene.revision().to_string()),
        );
    }
    let document = SceneDocument::from_json(source)?;
    document.validate(&SceneKinds::of(engine.app.world()))?;
    if document.id != scene.id() || document.revision != target {
        return Err(Error::new(ErrorCode::RevisionConflict, "$.args.revision")
            .expected(Expected::Matching)
            .got(serde_json::json!({
                "id": document.id,
                "revision": document.revision.to_string(),
            })));
    }
    game.reload_scene(engine, source, files)
}
