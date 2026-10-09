use super::*;
use engine::browser::{BrowserApp, SceneFileSet};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct PlaygroundGame {
    app: BrowserApp,
    game: Playground,
    // Temporal antialiasing the page chose over the scene's own settings,
    // kept across reloads.
    temporal_antialiasing: Option<bool>,
    #[cfg(feature = "dev")]
    development: engine::dev_browser::DevBrowser,
}

#[wasm_bindgen]
impl PlaygroundGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        source: &str,
        files: SceneFileSet,
    ) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();
        // The browser compiles the renderer's pipelines while the scene
        // loads.
        let starting = BrowserApp::start(canvas).await?;
        let mut engine = Engine::default();
        let game = Playground::new(&mut engine, source, files.files()).map_err(scene_error)?;
        Ok(Self {
            app: starting.finish(engine).await?,
            game,
            temporal_antialiasing: None,
            #[cfg(feature = "dev")]
            development: development_tools(),
        })
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
        self.app
            .engine
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

    /// Splits fixed ticks across frames around their due time (default with GPU
    /// presentation); see `BrowserApp::set_tick_pipelining`.
    pub fn set_tick_pipelining(&mut self, enabled: bool) {
        self.app.set_tick_pipelining(enabled);
    }

    /// Temporal antialiasing of ordinary GPU-presented frames over the
    /// scene's own settings, kept across reloads; see
    /// `BrowserApp::set_temporal_antialiasing`.
    pub fn set_temporal_antialiasing(&mut self, enabled: bool) {
        self.temporal_antialiasing = Some(enabled);
        self.app.set_temporal_antialiasing(enabled);
    }

    /// Loading work in place of a frame while the GPU compiles pipelines;
    /// see `BrowserApp::work`.
    pub fn work(&mut self, width: u32, height: u32) {
        self.app.work(width, height);
    }

    /// Terrain loading between frames; see
    /// `BrowserApp::work_between_frames`.
    pub fn work_between_frames(&mut self, milliseconds: f64) -> bool {
        self.app.work_between_frames(milliseconds)
    }

    /// Workers the page runs jobs on; see
    /// `BrowserApp::set_job_workers`.
    pub fn set_job_workers(&mut self, workers: u32) {
        self.app.set_job_workers(workers);
    }

    /// Shares and jobs to hand workers; see
    /// `BrowserApp::take_jobs`.
    pub fn take_jobs(&mut self) -> JsValue {
        self.app.take_jobs().into()
    }

    /// A worker's output of job `id`.
    pub fn complete_job(&mut self, id: f64, output: Vec<u8>) {
        self.app.complete_job(id, output);
    }

    /// Job `id` failed on its worker.
    pub fn fail_job(&mut self, id: f64, error: String) {
        self.app.fail_job(id, error);
    }

    /// A part of job `id`'s output from its worker.
    pub fn part_job(&mut self, id: f64, part: Vec<u8>) {
        self.app.part_job(id, part);
    }

    /// The page's job output cache counters.
    pub fn report_job_cache(&mut self, hits: f64, misses: f64, bytes: f64, mismatches: f64) {
        self.app.report_job_cache(hits, misses, bytes, mismatches);
    }

    /// The last frame's statistics as a JSON object.
    pub fn stats(&self) -> String {
        self.app.stats_json()
    }
    pub fn suspend(&mut self) {
        self.app.suspend();
        #[cfg(feature = "dev")]
        self.development.clear_input();
    }
    pub fn reset_camera(&mut self) {
        self.game.reset_player(&mut self.app.engine);
        #[cfg(feature = "dev")]
        self.development.clear_input();
    }
    pub fn reload_scene(&mut self, source: &str, files: SceneFileSet) -> Result<(), JsValue> {
        self.game
            .reload(&mut self.app.engine, source, files.files())
            .map_err(scene_error)?;
        if let Some(enabled) = self.temporal_antialiasing {
            self.app.set_temporal_antialiasing(enabled);
        }
        self.app.suspend();
        #[cfg(feature = "dev")]
        self.development.clear_input();
        Ok(())
    }

    /// Small on-demand observation; JSON is never produced in a gameplay loop.
    pub fn progress(&self) -> String {
        let world = self.app.engine.app.world();
        let progress = world.resource::<GameProgress>();
        let player = self
            .game
            .player()
            .and_then(|entity| world.get::<Transform>(entity))
            .map(|transform| transform.translation.to_array());
        let frame = world.resource::<FrameTime>();
        let assets = world.resource::<AssetLibrary>();
        serde_json::json!({
            "scene_id": self.game.scene_id(), "revision": self.game.revision(), "frame":frame.frame,
            "collected":progress.collected, "total":progress.total, "jumps":progress.jumps,
            "collision_errors":progress.collision_errors,
            "player":player, "cpu_meshes":assets.mesh_count(), "cpu_materials":assets.material_count(),
            "cpu_asset_bytes":assets.resident_bytes(),
            "cpu_textures":assets.texture_count(), "cpu_texture_bytes":assets.texture_bytes(),
        }).to_string()
    }
}

#[cfg(feature = "dev")]
#[wasm_bindgen]
impl PlaygroundGame {
    pub fn dev_command(&mut self, request: &str, width: u32, height: u32, paused: bool) -> String {
        self.development.command(
            &mut self.app,
            self.game.scene_instance(),
            request,
            width,
            height,
            paused,
        )
    }

    /// Reload saved authoring data only when both current and target revisions match.
    pub fn dev_reload(
        &mut self,
        source: &str,
        expected_revision: &str,
        revision: &str,
        files: Option<SceneFileSet>,
    ) -> String {
        let files = files.unwrap_or_default();
        let expected = expected_revision.parse::<u64>();
        let target = revision.parse::<u64>();
        let result = (|| {
            let (Ok(expected), Ok(target)) = (expected, target) else {
                return Err(engine::scene::SceneError {
                    code: "invalid_revision",
                    path: "$.revision".into(),
                    message: "revisions must be unsigned decimal strings".into(),
                });
            };
            if self.game.revision() != expected {
                return Err(engine::scene::SceneError {
                    code: "revision_conflict",
                    path: "$.expected_revision".into(),
                    message: format!("active revision is {}", self.game.revision()),
                });
            }
            let document = SceneDocument::from_json(source)?;
            if document.id != self.game.scene_id() || document.revision != target {
                return Err(engine::scene::SceneError {
                    code: "revision_conflict",
                    path: "$.revision".into(),
                    message: "saved scene ID and revision must match the requested scene".into(),
                });
            }
            self.game
                .reload(&mut self.app.engine, source, files.files())
        })();
        match result {
            Ok(()) => {
                self.app.suspend();
                self.development.clear_input();
                self.development.result(
                    &mut self.app,
                    self.game.scene_instance(),
                    serde_json::json!({"reloaded":true,"saved_by_reload":false}),
                )
            }
            Err(error) => self.development.error(
                &mut self.app,
                self.game.scene_instance(),
                error.code,
                &error.path,
                &error.message,
            ),
        }
    }
}

#[cfg(feature = "dev")]
fn development_tools() -> engine::dev_browser::DevBrowser {
    use engine::dev::InspectorDescriptor;
    let mut development = engine::dev_browser::DevBrowser::default();
    development.tools.register_resource::<GameProgress>(
        InspectorDescriptor::new("GameProgress", "Crystal Garden's score and jump count", serde_json::json!({
            "type":"object", "properties":{
                "collected":{"type":"integer","minimum":0},
                "total":{"type":"integer","minimum":0},
                "jumps":{"type":"integer","minimum":0},
                "collision_errors":{"type":"integer","minimum":0}
            }, "required":["collected","total","jumps","collision_errors"], "additionalProperties":false
        })),
        |progress| serde_json::json!({"collected":progress.collected,"total":progress.total,"jumps":progress.jumps,"collision_errors":progress.collision_errors}),
    ).expect("game inspector registration is valid");
    development
}

fn scene_error(error: engine::scene::SceneError) -> JsValue {
    JsValue::from_str(&error.to_string())
}
