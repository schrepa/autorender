//! The engine's side of an exported game: the page's app, the game and the
//! development tools, behind the methods [`crate::export_game!`] exports.
use super::{BrowserApp, SceneFileSet};
use crate::game::Game;
use engine_core::{ActionInput, Engine, InputState};
use wasm_bindgen::prelude::*;
use web_sys::HtmlCanvasElement;

/// An exported game's engine, game and development tools.
pub struct GameHost<G: Game> {
    app: BrowserApp,
    game: G,
    #[cfg(feature = "dev")]
    development: crate::dev_browser::DevBrowser,
}

impl<G: Game> GameHost<G> {
    /// Starts the renderer of `canvas` and, while the browser compiles its
    /// pipelines, the engine and game `build` makes.
    pub async fn launch<E: Into<crate::Error>>(
        canvas: HtmlCanvasElement,
        build: impl FnOnce() -> Result<(Engine, G), E>,
    ) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();
        let mut game = None;
        let app = BrowserApp::launch(canvas, || {
            let (engine, built) = build().map_err(Into::into)?;
            game = Some(built);
            Ok(engine)
        })
        .await?;
        let game = game.expect("launching built the game");
        #[cfg(feature = "dev")]
        let development = {
            let mut development = crate::dev_browser::DevBrowser::default();
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

    /// Runs `engine` from the next frame, wired to the page as the first
    /// was and under the page's culling and antialiasing choices.
    pub fn replace_engine(&mut self, engine: Engine) {
        self.app.replace_engine(engine);
        self.suspend();
    }

    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        &mut self,
        timestamp: f64,
        width: u32,
        height: u32,
        forward: f32,
        right: f32,
        up: f32,
        yaw: f32,
        pitch: f32,
        held: u32,
        pressed: u32,
        released: u32,
    ) -> Result<(), JsValue> {
        let engine = self.app.engine_mut();
        self.game.before_frame(engine);
        engine
            .app
            .world_mut()
            .resource_mut::<ActionInput>()
            .ingest(held, pressed, released);
        self.app.frame(
            timestamp,
            width,
            height,
            InputState {
                move_forward: forward,
                move_right: right,
                move_up: up,
                look_yaw: yaw,
                look_pitch: pitch,
            },
        )
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

    pub fn take_jobs(&mut self) -> js_sys::Array {
        self.app.take_jobs()
    }

    pub fn complete_job(&mut self, id: f64, output: Vec<u8>) {
        self.app.complete_job(id, output);
    }

    pub fn fail_job(&mut self, id: f64, error: String) {
        self.app.fail_job(id, error);
    }

    pub fn part_job(&mut self, id: f64, part: Vec<u8>) {
        self.app.part_job(id, part);
    }

    pub fn report_job_cache(&mut self, hits: f64, misses: f64, bytes: f64, mismatches: f64) {
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

    pub fn reload_scene(&mut self, source: &str, files: SceneFileSet) -> Result<(), JsValue> {
        self.game
            .reload_scene(self.app.engine_mut(), source, files.files())
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
        files: Option<SceneFileSet>,
    ) -> String {
        use engine_core::ErrorCode;
        use engine_scene::{SceneDocument, SceneError};
        let files = files.unwrap_or_default();
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
            self.game
                .reload_scene(self.app.engine_mut(), source, files.files())
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
    pub fn profiling_start(&mut self, max_events: u32) -> Result<(), JsValue> {
        self.app
            .profiling_start(max_events as usize)
            .map_err(JsValue::from_str)
    }

    pub fn profiling_stop(&mut self) -> Result<String, JsValue> {
        self.app
            .profiling_stop()
            .map(|report| report.to_string())
            .map_err(JsValue::from_str)
    }

    pub fn profiling_poll(&mut self) {
        self.app.profiling_poll();
    }

    pub fn profiling_report(&mut self) -> String {
        self.app.profiling_report().to_string()
    }

    pub fn profiling_status(&self) -> String {
        self.app.profiling_status().to_string()
    }
}
