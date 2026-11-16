use crate::app_host::{AppHost, Focus};
use crate::game::{Arrival, Game, LoadedScene};
use crate::host::{
    DeviceGatherer, FrameLoop, FramePhase, HostPause, InputSource, JobId, JobOrder, JobScheduler,
    LevelLoad, LevelStep,
};
use engine_core::{
    BrickList, CursorMode, CursorRequests, Engine, ErrorCode, Gathered, Input, Level, MouseButton,
    Pause, PauseReason, Progress, TouchPhase,
};
use engine_render::{Picture, RenderTarget};
use engine_scene::SceneFiles;

pub struct GameHost<G: Game> {
    app: AppHost,
    game: std::marker::PhantomData<fn() -> G>,
    scene: LoadedScene,
    input: DeviceGatherer,
    frames: FrameLoop,
    jobs: JobScheduler,
    level: LevelLoad,
    #[cfg(feature = "dev")]
    development: crate::dev_host::DevHost,
    #[cfg(feature = "dev")]
    pub(crate) dev_pending: Option<crate::dev_host::bridge::Pending>,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    source: Option<std::path::PathBuf>,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    watched: Vec<(std::path::PathBuf, Option<std::time::SystemTime>)>,
    #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
    checked: Option<std::time::Instant>,
}

impl<G: Game> GameHost<G> {
    pub async fn launch<E: Into<crate::Error>>(
        target: RenderTarget,
        bricks: impl FnOnce(BrickList) -> BrickList,
        build: impl FnOnce(&mut Engine) -> Result<G, E>,
    ) -> crate::Result<Self> {
        engine_panic::install();
        let app = AppHost::launch(target, bricks, |engine| {
            let game = build(engine).map_err(Into::into)?;
            engine.app.world_mut().insert_resource(game);
            Ok(())
        })
        .await?;
        Ok(Self {
            app,
            game: std::marker::PhantomData,
            scene: LoadedScene::default(),
            input: DeviceGatherer::default(),
            frames: FrameLoop::default(),
            jobs: JobScheduler::default(),
            level: LevelLoad::default(),
            #[cfg(feature = "dev")]
            development: crate::dev_host::DevHost::default(),
            #[cfg(feature = "dev")]
            dev_pending: None,
            #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
            source: None,
            #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
            watched: Vec::new(),
            #[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
            checked: None,
        })
    }

    pub fn engine(&self) -> &Engine {
        self.app.engine()
    }

    pub fn cook(&mut self) {
        self.app.cook();
    }

    pub fn jobs_held(&self) -> u32 {
        self.jobs.held()
    }

    pub fn engine_mut(&mut self) -> &mut Engine {
        self.app.engine_mut()
    }

    pub fn game(&self) -> &G {
        self.app.engine().app.world().resource::<G>()
    }

    pub fn scene(&self) -> &LoadedScene {
        &self.scene
    }

    pub fn replace_engine(&mut self, mut engine: Engine) {
        if let Some(game) = self.app.engine_mut().app.world_mut().remove_resource::<G>() {
            engine.app.world_mut().insert_resource(game);
        }
        self.scene = LoadedScene::default();
        self.app.replace_engine(engine);
        self.suspend();
    }

    fn level_mut(&mut self) -> bevy_ecs::world::Mut<'_, Level> {
        self.app
            .engine_mut()
            .app
            .world_mut()
            .resource_mut::<Level>()
    }

    pub fn take_level_request(&mut self) -> Option<String> {
        self.level_mut().take_asked()
    }

    pub fn report_level(&mut self, done: u32, total: u32) {
        self.level_mut()
            .progress(Progress::of(u64::from(done), u64::from(total)));
    }

    pub fn fail_level(&mut self, message: &str) {
        self.level_failed(crate::Error::new(ErrorCode::SourceIo, "$.path").got(message));
    }

    fn level_failed(&mut self, error: crate::Error) {
        self.level = LevelLoad::Idle;
        self.level_mut().fail(error);
    }

    fn level_built(&mut self) {
        self.level_mut().built();
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
        match self.level.step() {
            Ok(LevelStep::Progress(progress)) => self.level_mut().progress(progress),
            #[cfg(not(target_arch = "wasm32"))]
            Ok(LevelStep::Ready {
                folder,
                source,
                files,
            }) => match self.reload_scene(&source, &files) {
                Ok(()) => self.set_file_root(&folder.to_string_lossy()),
                Err(error) => self.level_failed(error),
            },
            Err(error) => self.level_failed(error),
        }
    }

    pub fn frame(&mut self, timestamp: f64, width: u32, height: u32) -> crate::Result<()> {
        self.framed(timestamp, (width, height), Picture::Shown)
            .map(drop)
    }

    pub(crate) fn framed(
        &mut self,
        timestamp: f64,
        size: (u32, u32),
        picture: Picture,
    ) -> crate::Result<bool> {
        self.step_level();
        self.prepare_pipelines();
        let source = self.input_source();
        self.app.follow_focus(source);
        let input = self.frame_input(source);
        self.app.accept(input);
        let framed = self.app.frame(timestamp, size, picture);
        self.apply_cursor();
        framed
    }

    pub fn input_source(&self) -> InputSource {
        #[cfg(feature = "dev")]
        if let Some(pending) = &self.dev_pending {
            return pending.input_source();
        }
        InputSource::Player
    }

    fn frame_input(&mut self, source: InputSource) -> Gathered {
        match source {
            InputSource::Agent => {
                self.input.drop_edges();
                self.agent_input()
            }
            InputSource::Player => Gathered {
                devices: self.input.take(),
                driven: Vec::new(),
            },
        }
    }

    #[cfg(feature = "dev")]
    fn agent_input(&mut self) -> Gathered {
        self.development.take_input()
    }

    #[cfg(not(feature = "dev"))]
    fn agent_input(&mut self) -> Gathered {
        Gathered::default()
    }

    pub fn pause(&mut self, pause: HostPause) {
        let reason = pause.reason();
        if self
            .app
            .engine()
            .app
            .world()
            .resource::<Pause>()
            .holds(reason)
        {
            return;
        }
        let source = self.input_source();
        match pause {
            HostPause::Focus => self.app.set_focus(Focus::Lost, source),
            HostPause::Hidden | HostPause::Suspended => self.app.engine_mut().pause(reason),
        }
        self.input.clear();
    }

    pub fn resume(&mut self, pause: HostPause) {
        let source = self.input_source();
        match pause {
            HostPause::Focus => self.app.set_focus(Focus::Held, source),
            HostPause::Hidden | HostPause::Suspended => {
                self.app.engine_mut().resume(pause.reason());
            }
        }
    }

    pub fn refresh_interval(&self) -> f64 {
        self.app.refresh_interval()
    }

    pub(crate) fn paced(&mut self) {
        self.app.paced();
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
        let scene = &self.scene;
        self.app
            .prepare_pipelines(|engine| G::content(engine, scene))
    }

    pub fn set_preparation(&mut self, enabled: bool) {
        self.frames.set_preparation(enabled);
        if !enabled {
            self.hold_preparation(false);
        }
    }

    pub(crate) fn hold_preparation(&mut self, held: bool) {
        let engine = self.app.engine_mut();
        if held {
            engine.pause(PauseReason::Preparing);
        } else {
            engine.resume(PauseReason::Preparing);
        }
    }

    pub(crate) fn set_preparation_ms(&mut self, milliseconds: f64) {
        self.app.set_preparation_ms(milliseconds);
    }

    pub fn preparing(&self) -> bool {
        self.frames.preparing()
    }

    pub fn key(&mut self, code: &str, down: bool) -> bool {
        self.input.key(code, down);
        self.app
            .engine()
            .app
            .world()
            .resource::<Input>()
            .uses_key(code)
    }

    pub fn mouse_button(&mut self, button: MouseButton, down: bool) -> bool {
        self.input.mouse_button(button, down);
        self.app
            .engine()
            .app
            .world()
            .resource::<Input>()
            .uses_mouse(button)
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

    pub fn pad(&mut self, index: u32, buttons: u32, axes: [f32; 6]) {
        self.input.pad(index, buttons, axes);
    }

    pub fn touch(&mut self, id: u32, x: f32, y: f32, phase: TouchPhase) {
        self.input.touch(id, x, y, phase);
    }

    pub fn touch_motion(&mut self, id: u32, dx: f32, dy: f32) {
        self.input.touch_motion(id, dx, dy);
    }

    pub fn cursor_mode(&self) -> CursorMode {
        self.app
            .engine()
            .app
            .world()
            .resource::<CursorRequests>()
            .applied()
    }

    pub(crate) fn apply_cursor(&mut self) {
        self.app
            .engine_mut()
            .app
            .world_mut()
            .resource_mut::<CursorRequests>()
            .apply();
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
    }

    pub fn set_spare_time(&mut self, milliseconds: f64) {
        self.app.set_spare_time(milliseconds);
    }

    pub fn set_job_workers(&mut self, workers: u32) -> bool {
        let set = self.jobs.set_workers(workers);
        self.app.set_job_workers(self.jobs.working());
        set
    }

    pub fn set_job_cache(&mut self, caching: bool, verify: bool) {
        self.jobs.set_cache(caching, verify);
        self.app.set_job_verifying(caching && verify);
    }

    pub fn set_job_sample(&mut self, sample: std::collections::BTreeSet<String>) {
        self.jobs.set_sample(sample);
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
        self.development.drop_edges();
    }

    pub(crate) fn suspend_live(&mut self) {
        self.app.suspend();
        self.input.drop_edges();
    }

    pub fn reset_camera(&mut self) {
        G::reset_camera(self.app.engine_mut(), &self.scene);
        self.suspend();
    }

    pub fn set_file_root(&mut self, root: &str) {
        let world = self.app.engine_mut().app.world_mut();
        if let Some(mut files) = world.get_resource_mut::<engine_core::Files>() {
            files.set_root(root);
        }
    }

    pub fn open_scene(&mut self, source: &str, files: &SceneFiles) -> crate::Result<()> {
        let document = engine_scene::SceneDocument::from_json(source)?;
        self.arrive(&document, files, Arrival::Opening)
    }

    pub fn reload_scene(&mut self, source: &str, files: &SceneFiles) -> crate::Result<()> {
        let document = engine_scene::SceneDocument::from_json(source)?;
        self.arrive(&document, files, Arrival::Reload)
    }

    fn arrive(
        &mut self,
        document: &engine_scene::SceneDocument,
        files: &SceneFiles,
        arrival: Arrival,
    ) -> crate::Result<()> {
        let instance = G::load_scene(self.app.engine_mut(), document, files, arrival)?;
        self.scene
            .replace(self.app.engine_mut().app.world_mut(), instance);
        self.app.replay_settings()?;
        self.app.reloaded();
        self.suspend();
        self.level_built();
        Ok(())
    }

    pub fn change_settings<S: engine_core::Section>(
        &mut self,
        change: impl FnOnce(&mut S),
    ) -> crate::Result<()> {
        let entry = crate::host::changed_section(self.app.engine().app.world(), change)?;
        self.app.set_settings(&[entry])
    }

    pub fn set_setting(&mut self, key: &str, value: serde_json::Value) -> crate::Result<()> {
        self.app.set_setting(key, value)
    }

    pub fn set_settings(&mut self, entries: &[(String, serde_json::Value)]) -> crate::Result<()> {
        self.app.set_settings(entries)
    }

    pub fn set_pixel_ratio(&mut self, ratio: f64) {
        self.app.set_pixel_ratio(ratio);
    }

    pub fn set_refresh_interval(&mut self, milliseconds: Option<f64>) {
        self.app.set_refresh_interval(milliseconds);
    }

    pub fn set_display_paced(&mut self, paced: bool) {
        self.app.set_display_paced(paced);
    }

    pub fn set_job_clock(&mut self, clock: crate::host::JobClock) {
        self.app.set_job_clock(clock);
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
    pub fn read_offscreen(&self) -> crate::Result<Option<(u32, u32, Vec<u8>)>> {
        self.app.renderer.read_offscreen()
    }

    pub fn frame_stats(&self) -> crate::frame_stats::FrameStats {
        self.app.stats()
    }

    pub fn frame_times(&self) -> &crate::frame_stats::FrameStats {
        self.app.frame_times()
    }

    pub fn stats(&self) -> String {
        self.app.stats_json()
    }
}

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
impl<G: Game> GameHost<G> {
    pub fn watch_source(&mut self, path: &std::path::Path) {
        self.source = Some(path.to_owned());
        self.watch();
    }

    pub fn source(&self) -> Option<&std::path::Path> {
        self.source.as_deref()
    }

    pub fn follow_source(&mut self) -> Option<crate::Result<()>> {
        let now = std::time::Instant::now();
        if self
            .checked
            .is_some_and(|checked| now.duration_since(checked) < CHECK_EVERY)
        {
            return None;
        }
        self.checked = Some(now);
        self.app.reload_shaders();
        if self
            .watched
            .iter()
            .all(|(file, stamp)| modified(file) == *stamp)
        {
            return None;
        }
        self.watch();
        let path = self.source.clone()?;
        let kinds = engine_scene::SceneKinds::of(self.app.engine().app.world());
        let mut load = match LevelLoad::open(&path.to_string_lossy(), &kinds) {
            Ok(load) => load,
            Err(error) => return Some(Err(error)),
        };
        loop {
            match load.step() {
                Ok(LevelStep::Progress(_)) => {}
                Ok(LevelStep::Ready { source, files, .. }) => {
                    return Some(self.reload_scene(&source, &files));
                }
                Err(error) => return Some(Err(error)),
            }
        }
    }

    fn watch(&mut self) {
        let Some(path) = self.source.clone() else {
            self.watched.clear();
            return;
        };
        let kinds = engine_scene::SceneKinds::of(self.app.engine().app.world());
        let document = std::fs::read_to_string(&path)
            .ok()
            .and_then(|source| engine_scene::SceneDocument::from_json(&source).ok());
        let folder = path
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .to_owned();
        let stamp = modified(&path);
        let mut watched = vec![(path, stamp)];
        for file in document.iter().flat_map(|document| document.files(&kinds)) {
            let file = folder.join(file.uri);
            let stamp = modified(&file);
            watched.push((file, stamp));
        }
        self.watched = watched;
    }
}

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
const CHECK_EVERY: std::time::Duration = std::time::Duration::from_millis(250);

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
fn modified(path: &std::path::Path) -> Option<std::time::SystemTime> {
    std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
}

#[cfg(feature = "dev")]
impl<G: Game> GameHost<G> {
    pub(crate) fn collect_pipelines(&mut self) {
        self.app.collect_pipelines();
    }

    pub(crate) fn dev_handle(
        &mut self,
        request: &str,
        size: (u32, u32),
        paused: bool,
    ) -> crate::dev_host::Handled {
        self.development
            .command(&mut self.app, self.scene.instance(), request, size, paused)
    }

    pub(crate) fn dev_run(
        &mut self,
        call: crate::dev_host::Call<'_>,
        size: (u32, u32),
        paused: bool,
    ) -> serde_json::Value {
        self.development
            .call(&mut self.app, self.scene.instance(), call, size, paused)
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
            self.scene.instance(),
            self.app.engine(),
            source,
            (expected_revision, revision),
        )
        .and_then(|document| self.arrive(&document, files, Arrival::Reload));
        match result {
            Ok(()) => {
                let skipped = self.scene.instance().map(|scene| scene.skipped().to_vec());
                let mut reloaded = serde_json::json!({"reloaded": true});
                if let Some(skipped) = skipped.filter(|skipped| !skipped.is_empty()) {
                    reloaded["skipped"] = serde_json::json!(skipped);
                }
                self.development
                    .result(&mut self.app, self.scene.instance(), reloaded)
            }
            Err(error) => self
                .development
                .refusal(&mut self.app, self.scene.instance(), &error),
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

    pub fn profile_bricks(&self) -> Vec<&'static str> {
        self.app.profile_bricks()
    }
}

#[cfg(feature = "dev")]
fn reload_checked(
    scene: Option<&engine_scene::SceneInstance>,
    engine: &Engine,
    source: &str,
    (expected_revision, revision): (&str, &str),
) -> crate::Result<engine_scene::SceneDocument> {
    use engine_core::{Error, ErrorCode};
    use engine_scene::{SceneDocument, SceneKinds};
    use engine_settings::Expected;
    let Some(scene) = scene else {
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
    Ok(document)
}
