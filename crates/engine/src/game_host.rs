//! The engine's side of a game a host runs, on a page or in a native
//! window: its app, the game and the development tools.
use crate::app_host::{AppHost, JobMessage};
use crate::game::Game;
use engine_core::{ActionInput, Engine, InputState, JobId};
use engine_render::RenderTarget;
use engine_scene::SceneFiles;

/// A hosted game's engine, game and development tools.
pub struct GameHost<G: Game> {
    app: AppHost,
    game: G,
    #[cfg(feature = "dev")]
    development: crate::dev_host::DevHost,
}

impl<G: Game> GameHost<G> {
    /// Starts the renderer presenting to `target` and, while the GPU
    /// compiles its pipelines, the engine and game `build` makes.
    pub async fn launch<E: Into<crate::Error>>(
        target: RenderTarget,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> crate::Result<Self> {
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
            #[cfg(feature = "dev")]
            development,
        })
    }

    pub fn engine(&self) -> &Engine {
        self.app.engine()
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

    /// The engine and the game, to change both.
    pub fn parts_mut(&mut self) -> (&mut Engine, &mut G) {
        (self.app.engine_mut(), &mut self.game)
    }

    /// Runs `engine` from the next frame, wired to the host as the first
    /// was and under the host's culling and antialiasing choices.
    pub fn replace_engine(&mut self, engine: Engine) {
        self.app.replace_engine(engine);
        self.suspend();
    }

    /// Runs one frame at `timestamp` milliseconds on a `width` × `height`
    /// view with `input` and the action bits `held`, `pressed` and
    /// `released` since the last frame.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        timestamp: f64,
        width: u32,
        height: u32,
        input: InputState,
        held: u32,
        pressed: u32,
        released: u32,
    ) -> crate::Result<()> {
        let engine = self.app.engine_mut();
        self.game.before_frame(engine);
        engine
            .app
            .world_mut()
            .resource_mut::<ActionInput>()
            .ingest(held, pressed, released);
        self.app.frame(timestamp, width, height, input)
    }

    pub fn work(&mut self, width: u32, height: u32) {
        self.app.work(width, height);
    }

    pub fn work_between_frames(&mut self, milliseconds: f64) -> bool {
        self.app.work_between_frames(milliseconds)
    }

    pub fn set_job_workers(&mut self, workers: u32) {
        self.app.set_job_workers(workers);
    }

    pub fn take_jobs(&mut self) -> Vec<JobMessage> {
        self.app.take_jobs()
    }

    pub fn complete_job(&mut self, id: JobId, output: Vec<u8>) {
        self.app.complete_job(id, output);
    }

    pub fn fail_job(&mut self, id: JobId, error: String) {
        self.app.fail_job(id, error);
    }

    pub fn part_job(&mut self, id: JobId, part: Vec<u8>) {
        self.app.part_job(id, part);
    }

    pub fn report_job_cache(&mut self, hits: u64, misses: u64, bytes: u64, mismatches: u64) {
        self.app.report_job_cache(hits, misses, bytes, mismatches);
    }

    pub fn suspend(&mut self) {
        self.app.suspend();
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

    /// The GPU the game draws on.
    pub fn adapter(&self) -> &crate::AdapterInfo {
        self.app.renderer.adapter()
    }

    /// Starts a capture in the graphics debugger the program runs under
    /// (RenderDoc, Xcode); whether one started.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn start_debugger_capture(&mut self) -> bool {
        self.app.renderer.start_debugger_capture()
    }

    /// Stops the debugger's capture once the GPU finished its work.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn stop_debugger_capture(&mut self) -> crate::Result<()> {
        self.app
            .renderer
            .stop_debugger_capture()
            .map_err(crate::app_host::render_failure)
    }

    /// Waits until the GPU finished every frame submitted.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_gpu(&self) -> crate::Result<()> {
        self.app
            .renderer
            .wait_for_gpu()
            .map_err(crate::app_host::render_failure)
    }

    /// A mark after the GPU work submitted so far, for hosts that keep
    /// frames in flight: [`Self::wait_for_mark`] waits for it.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn gpu_mark(&self) -> crate::GpuMark {
        self.app.renderer.gpu_mark()
    }

    /// Waits until the GPU finished the work submitted before `mark`.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn wait_for_mark(&self, mark: crate::GpuMark) -> crate::Result<()> {
        self.app
            .renderer
            .wait_for_mark(mark)
            .map_err(crate::app_host::render_failure)
    }

    /// The last frame an offscreen target drew, as `(width, height, RGBA8
    /// rows from the top)`, once the GPU finished it; `None` for a window.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn read_offscreen(&self) -> crate::Result<Option<(u32, u32, Vec<u8>)>> {
        self.app
            .renderer
            .read_offscreen()
            .map_err(crate::app_host::render_failure)
    }

    /// The last frame's statistics.
    pub fn frame_stats(&self) -> engine_core::FrameStats {
        self.app.stats()
    }

    /// The last frame's statistics, as the JSON object `FrameStats` types.
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
}

#[cfg(feature = "dev")]
impl<G: Game> GameHost<G> {
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
    /// Starts a capture of at most `max_events` CPU events.
    pub fn profiling_start(&mut self, max_events: usize) -> Result<(), &'static str> {
        self.app.profiling_start(max_events)
    }

    /// Stops the capture: its report.
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
