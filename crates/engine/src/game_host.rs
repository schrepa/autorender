use crate::app_host::{Advance, AppHost, Focus};
use crate::game::{Arrival, Game, LoadedScene};
use crate::host::{
    DeviceGatherer, FrameLoop, FramePhase, HostPause, InputSource, JobId, JobOrder, JobScheduler,
};
use engine_core::{
    BrickList, CursorMode, Devices, Engine, MouseButton, Pause, PauseReason, TouchPhase,
};
use engine_render::{Picture, RenderTarget};
use engine_scene::SceneFiles;

pub struct GameHost<G: Game> {
    pub(crate) app: AppHost,
    game: std::marker::PhantomData<fn() -> G>,
    pub(crate) scene: LoadedScene,
    input: DeviceGatherer,
    frames: FrameLoop,
    jobs: JobScheduler,
    #[cfg(feature = "dev")]
    pub(crate) development: crate::dev_host::DevHost,
    #[cfg(feature = "dev")]
    loading: Option<engine_profile::Cause>,
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
        #[cfg_attr(not(feature = "dev"), expect(unused_mut, reason = "dev installs names"))]
        let mut app = AppHost::launch(target, bricks, |engine| {
            let game = build(engine).map_err(Into::into)?;
            engine.app.world_mut().insert_resource(game);
            Ok(())
        })
        .await?;
        #[cfg(feature = "dev")]
        crate::dev_host::install(app.engine.app.world_mut());
        Ok(Self {
            app,
            game: std::marker::PhantomData,
            scene: LoadedScene::default(),
            input: DeviceGatherer::default(),
            frames: FrameLoop::default(),
            jobs: JobScheduler::default(),
            #[cfg(feature = "dev")]
            development: crate::dev_host::DevHost::default(),
            #[cfg(feature = "dev")]
            loading: None,
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

    pub(crate) fn framed(
        &mut self,
        advance: Advance,
        size: (u32, u32),
        picture: Picture,
    ) -> crate::Result<bool> {
        #[cfg(feature = "dev")]
        if let Some(loading) = self.loading {
            let drawn = engine_profile::caused(loading, || self.frame_once(advance, size, picture));
            let (readiness, _) = self.progress();
            if readiness.settled() && !crate::host::in_flight(self, &readiness) {
                self.loading = None;
            }
            return drawn;
        }
        self.frame_once(advance, size, picture)
    }

    fn frame_once(
        &mut self,
        advance: Advance,
        size: (u32, u32),
        picture: Picture,
    ) -> crate::Result<bool> {
        self.prepare_pipelines();
        let source = self.input_source();
        self.app.follow_focus(source);
        let input = self.frame_input(source);
        self.app.accept(input);
        let drawn = self.app.frame(advance, size, picture)?;
        #[cfg(feature = "dev")]
        self.dev_frame_end();
        Ok(drawn)
    }

    pub(crate) fn play(&mut self, timestamp: f64, size: (u32, u32)) -> crate::Result<FramePhase> {
        #[cfg(feature = "dev")]
        return self.dev_play(timestamp, size);
        #[cfg(not(feature = "dev"))]
        {
            self.framed(Advance::Clock(timestamp), size, Picture::Shown)?;
            Ok(FramePhase::Playing)
        }
    }

    pub fn input_source(&self) -> InputSource {
        #[cfg(feature = "dev")]
        if self.development.stepping() {
            return InputSource::Agent;
        }
        InputSource::Player
    }

    fn frame_input(&mut self, source: InputSource) -> Devices {
        match source {
            InputSource::Agent => {
                self.input.drop_edges();
                self.agent_input()
            }
            InputSource::Player => self.input.take(),
        }
    }

    #[cfg(feature = "dev")]
    fn agent_input(&mut self) -> Devices {
        self.development.take_input()
    }

    #[cfg(not(feature = "dev"))]
    fn agent_input(&mut self) -> Devices {
        Devices::default()
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

    pub fn key(&mut self, code: &str, down: bool) {
        self.input.key(code, down);
    }

    pub fn mouse_button(&mut self, button: MouseButton, down: bool) {
        self.input.mouse_button(button, down);
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
        *self.app.engine().app.world().resource::<CursorMode>()
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
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

    pub fn job_kept(
        &mut self,
        id: JobId,
        key: String,
        kept: Option<(std::sync::Arc<[u8]>, engine_core::JobSource)>,
    ) {
        let found = kept.as_ref().map(|(_, source)| *source);
        engine_core::job_looked(self.app.engine_mut().app.world_mut(), id, found);
        let mut finished = Vec::new();
        self.jobs.kept(id, key, kept.map(|(output, _)| output), &mut finished);
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
        self.app.take_pipelines();
    }

    pub fn report_job_cache(&mut self, counts: [u64; 5]) {
        self.app.report_job_cache(counts);
    }

    pub fn suspend(&mut self) {
        self.app.suspend();
        self.input.drop_edges();
        #[cfg(feature = "dev")]
        self.development.drop_edges();
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
        self.open_document(&document, files)
    }

    pub fn open_document(
        &mut self,
        document: &engine_scene::SceneDocument,
        files: &SceneFiles,
    ) -> crate::Result<()> {
        self.arrive(document, files, Arrival::Opening)
    }

    pub fn reload_scene(&mut self, source: &str, files: &SceneFiles) -> crate::Result<()> {
        let document = engine_scene::SceneDocument::from_json(source)?;
        self.arrive(&document, files, Arrival::Reload)
    }

    pub(crate) fn arrive(
        &mut self,
        document: &engine_scene::SceneDocument,
        files: &SceneFiles,
        arrival: Arrival,
    ) -> crate::Result<()> {
        let cause = match arrival {
            Arrival::Opening => engine_profile::open("launch", "scene"),
            Arrival::Reload => engine_profile::open("reload", &document.id),
        };
        #[cfg(feature = "dev")]
        if cause != engine_profile::Cause::default() {
            self.loading = Some(cause);
        }
        engine_profile::caused(cause, || {
            let instance = G::load_scene(self.app.engine_mut(), document, files, arrival)?;
            self.scene
                .replace(self.app.engine_mut().app.world_mut(), instance);
            self.app.replay_settings()?;
            self.app.reloaded();
            self.suspend();
            Ok(())
        })
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

    pub fn adapter(&self) -> &crate::AdapterInfo {
        self.app.renderer.adapter()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_gpu(&self) -> crate::Result<()> {
        self.app.renderer.wait_for_gpu()
    }

    pub fn progress(&self) -> (engine_core::Readiness, engine_core::JobStats) {
        self.app.progress()
    }

    pub fn frame_times(&self) -> &crate::frame_stats::FrameStats {
        self.app.frame_times()
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
        Some(
            read_scene(&path, &kinds)
                .and_then(|(source, files)| self.reload_scene(&source, &files)),
        )
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

#[cfg(all(feature = "dev", not(target_arch = "wasm32")))]
fn read_scene(
    path: &std::path::Path,
    kinds: &engine_scene::SceneKinds,
) -> crate::Result<(String, SceneFiles)> {
    use engine_core::{Error, ErrorCode};
    let source = std::fs::read_to_string(path).map_err(|error| {
        Error::new(ErrorCode::SourceIo, "$.path").got(format!("{}: {error}", path.display()))
    })?;
    let mut load = crate::host::SceneLoad::new(&source, kinds)?;
    let folder = path.parent().unwrap_or(std::path::Path::new("."));
    for read in load.take_reads().into_iter().rev() {
        let file = folder.join(&read.uri);
        let bytes = std::fs::read(&file).map_err(|error| {
            Error::new(ErrorCode::SourceIo, "$.path").got(format!("{}: {error}", file.display()))
        })?;
        if let Some(image) = load.read(&read.uri, bytes)? {
            load.decode(&read.uri, &image)?;
        }
    }
    Ok((source, load.finish()?))
}

#[cfg(feature = "dev")]
pub(crate) fn reload_checked(
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
