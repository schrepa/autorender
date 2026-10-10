use engine::browser::{BrowserApp, SceneFileSet};
use engine::prelude::*;
use wasm_bindgen::prelude::*;

use std::collections::HashMap;

use crate::{
    AuthoredScene, FrameMetrics, StressConfig, StressFile, StressScene, create_workload_with, look,
    set_view, stress_observation, stress_reflections, stress_shafts,
};

/// Files workloads read, by name: the dense scene's virtual meshes,
/// whose prefixes the page fetches, and the highlands'
/// terrain material maps, which it decodes.
#[wasm_bindgen]
#[derive(Default)]
pub struct StressFiles {
    files: HashMap<String, StressFile>,
}

#[wasm_bindgen]
impl StressFiles {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds virtual mesh `name`'s prefix and the URL renderers stream its
    /// other pages from.
    pub fn add_streamed(&mut self, name: &str, bytes: Vec<u8>, url: String) {
        self.files.insert(
            name.to_owned(),
            StressFile {
                bytes,
                url: Some(url),
                size: None,
            },
        );
    }

    /// Adds image `name`: `width` by `height` RGBA8 pixels, rows from the
    /// top.
    pub fn add_image(&mut self, name: &str, width: u32, height: u32, pixels: Vec<u8>) {
        self.files.insert(
            name.to_owned(),
            StressFile {
                bytes: pixels,
                url: None,
                size: Some([width, height]),
            },
        );
    }

    /// The files the dense scene reads, as a JSON array of names.
    pub fn dense_names() -> String {
        serde_json::to_string(&crate::DENSE_FILES).unwrap_or_else(|_| "[]".into())
    }

    /// The images the highlands read, as a JSON array of names.
    pub fn highlands_names() -> String {
        serde_json::to_string(&crate::HIGHLANDS_FILES).unwrap_or_else(|_| "[]".into())
    }

    /// The scanned rock maps the highlands read when present, as a JSON
    /// array of names under `stress/rocks/`; the page adds each found as
    /// `rocks/NAME`.
    pub fn rock_names() -> String {
        serde_json::to_string(&crate::ROCK_FILES).unwrap_or_else(|_| "[]".into())
    }
}

#[wasm_bindgen]
pub struct StressGame {
    app: BrowserApp,
    generation: u32,
    // Files the page supplied, kept for later workloads.
    files: HashMap<String, StressFile>,
    // The lighting preset and effect selection, kept across workload changes.
    look: String,
    effects: String,
    // Settings of the world's resources, kept across workload changes, which
    // replace the world: levels of detail and impostors
    // and the visibility buffer.
    level_of_detail: bool,
    impostors: bool,
    visibility_buffer: bool,
    // Culling and antialiasing the page chose over each workload's own
    // settings; unset, a workload keeps its own.
    occlusion_culling: Option<bool>,
    subpixel_culling: Option<bool>,
    temporal_antialiasing: Option<bool>,
    // An authored scene in place of a workload, when the page loads one.
    authored: Option<AuthoredScene>,
    #[cfg(feature = "dev")]
    development: engine::dev_browser::DevBrowser,
}

#[wasm_bindgen]
impl StressGame {
    pub async fn create(
        canvas: web_sys::HtmlCanvasElement,
        config_json: &str,
        files: Option<StressFiles>,
    ) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();
        let files = files.map(|files| files.files).unwrap_or_default();
        let config = StressConfig::from_json(config_json).map_err(js_error)?;
        // The browser compiles the renderer's pipelines while the workload
        // builds.
        let starting = BrowserApp::start(canvas).await?;
        let engine = create_workload_with(config, &files).map_err(js_error)?;
        Ok(Self {
            app: starting.finish(engine).await?,
            generation: 1,
            files,
            look: "scene".into(),
            effects: "all".into(),
            level_of_detail: true,
            impostors: true,
            visibility_buffer: false,
            occlusion_culling: None,
            subpixel_culling: None,
            temporal_antialiasing: None,
            authored: None,
            #[cfg(feature = "dev")]
            development: development_tools(),
        })
    }

    /// The authored scene `source` in place of a workload, with the image
    /// and buffer `files` it references.
    pub async fn create_scene(
        canvas: web_sys::HtmlCanvasElement,
        source: &str,
        files: SceneFileSet,
    ) -> Result<Self, JsValue> {
        console_error_panic_hook::set_once();
        let starting = BrowserApp::start(canvas).await?;
        let (engine, authored) = AuthoredScene::new(source, files.files()).map_err(scene_error)?;
        Ok(Self {
            app: starting.finish(engine).await?,
            generation: 1,
            files: HashMap::new(),
            look: "scene".into(),
            effects: "all".into(),
            level_of_detail: true,
            impostors: true,
            visibility_buffer: false,
            occlusion_culling: None,
            subpixel_culling: None,
            temporal_antialiasing: None,
            authored: Some(authored),
            #[cfg(feature = "dev")]
            development: development_tools(),
        })
    }

    /// Replaces the authored scene with `source`; a scene that fails keeps
    /// the current one.
    pub fn reload_scene(&mut self, source: &str, files: SceneFileSet) -> Result<(), JsValue> {
        let Some(authored) = self.authored.as_mut() else {
            return Err(js_error("no authored scene is loaded".into()));
        };
        authored
            .reload(&mut self.app.engine, source, files.files())
            .map_err(scene_error)?;
        self.apply_world_settings();
        self.suspend();
        Ok(())
    }

    /// The authored scene's ID, revision, frame, player position and asset
    /// counts, as a JSON object (`null` for workloads).
    pub fn scene_progress(&self) -> String {
        self.authored
            .as_ref()
            .map_or(serde_json::Value::Null, |authored| {
                authored.progress(&self.app.engine)
            })
            .to_string()
    }

    /// Build first, then replace the complete Engine at a browser frame boundary.
    /// Asset library IDs separate old renderer snapshots from the new world.
    pub fn configure(
        &mut self,
        config_json: &str,
        files: Option<StressFiles>,
    ) -> Result<(), JsValue> {
        if let Some(files) = files {
            self.files.extend(files.files);
        }
        let engine = create_workload_with(
            StressConfig::from_json(config_json).map_err(js_error)?,
            &self.files,
        )
        .map_err(js_error)?;
        self.app.engine = engine;
        // A workload replaces an authored scene's world with its own.
        self.authored = None;
        self.generation = self.generation.saturating_add(1);
        self.set_look(&self.look.clone())?;
        self.set_effects(&self.effects.clone())?;
        self.apply_world_settings();
        self.suspend();
        Ok(())
    }

    /// Rendering effects: `all`, `none`, or comma-separated `no-<effect>`
    /// exclusions.
    pub fn set_effects(&mut self, name: &str) -> Result<(), JsValue> {
        // `all`, `none`, or comma-separated `no-<effect>` exclusions.
        const EFFECTS: [&str; 9] = [
            "shadows",
            "occlusion",
            "bloom",
            "lights",
            "shafts",
            "reflections",
            "particles",
            "refraction",
            "probes",
        ];
        let mut on = [true; EFFECTS.len()];
        match name {
            "all" => {}
            "none" => on = [false; EFFECTS.len()],
            _ => {
                for token in name.split(',') {
                    let excluded = token
                        .strip_prefix("no-")
                        .and_then(|effect| EFFECTS.iter().position(|name| *name == effect));
                    match excluded {
                        Some(index) => on[index] = false,
                        None => {
                            return Err(js_error(format!(
                                "effects must be all, none or comma-separated no-<effect> for {}",
                                EFFECTS.join(", ")
                            )));
                        }
                    }
                }
            }
        }
        let [
            shadows,
            occlusion,
            bloom,
            lights,
            shafts,
            reflections,
            particles,
            refraction,
            probes,
        ] = on;
        let scene = self.app.engine.app.world().resource::<StressConfig>().scene;
        let (_, lamps) = look(scene, &self.look).map_err(js_error)?;
        let world = self.app.engine.app.world_mut();
        world.resource_mut::<Shadows>().enabled = shadows;
        world.resource_mut::<AmbientOcclusion>().intensity = if occlusion {
            AmbientOcclusion::default().intensity
        } else {
            0.0
        };
        world.resource_mut::<Bloom>().intensity = if bloom {
            Bloom::default().intensity
        } else {
            0.0
        };
        // Opt-in effects are removed entirely when off.
        if shafts {
            world.insert_resource(stress_shafts());
        } else {
            world.remove_resource::<SunShafts>();
        }
        if reflections {
            world.insert_resource(stress_reflections(scene));
        } else {
            world.remove_resource::<Reflections>();
        }
        {
            let mut local = world.resource_mut::<LocalLights>();
            local.enabled = lights;
            local.intensity = lamps;
        }
        world.resource_mut::<Particles>().enabled = particles;
        // Only the metro has glass that refracts.
        if refraction && scene == StressScene::Metro {
            world.insert_resource(Refraction::default());
        } else {
            world.remove_resource::<Refraction>();
        }
        // Reflection probes capture nothing where none stand.
        if probes {
            world.insert_resource(ReflectionProbes::default());
        } else {
            world.remove_resource::<ReflectionProbes>();
        }
        self.effects = name.into();
        Ok(())
    }

    /// Lighting preset: `scene` (the scene's own), `golden`, `dusk`,
    /// `night` or `day`.
    pub fn set_look(&mut self, name: &str) -> Result<(), JsValue> {
        let scene = self.app.engine.app.world().resource::<StressConfig>().scene;
        let (atmosphere, lamps) = look(scene, name).map_err(js_error)?;
        self.app.engine.app.insert_resource(atmosphere);
        crate::scene::apply_sky(self.app.engine.app.world_mut(), scene, name);
        if let Some(mut local) = self
            .app
            .engine
            .app
            .world_mut()
            .get_resource_mut::<LocalLights>()
        {
            local.intensity = lamps;
        }
        self.look = name.into();
        Ok(())
    }

    /// One frame at `timestamp`: the Stress Lab reads no action buttons, so
    /// it takes the host's input axes only.
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
    ) -> Result<(), JsValue> {
        self.app
            .engine
            .app
            .world_mut()
            .resource_mut::<FrameMetrics>()
            .begin_frame();
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
    pub fn stress_stats(&self) -> String {
        stress_observation(&self.app.engine, self.generation).to_string()
    }
    /// [`crate::stress_sample`]: a recording's per-frame workload numbers.
    pub fn frame_sample(&self) -> Vec<f64> {
        crate::stress_sample(&self.app.engine).to_vec()
    }
    pub fn set_view(&mut self, view: &str) -> Result<(), JsValue> {
        set_view(&mut self.app.engine, view).map_err(js_error)
    }
    pub fn set_culling(&mut self, enabled: bool) {
        self.app.engine.set_frustum_culling(enabled);
        if let Some(mut config) = self
            .app
            .engine
            .app
            .world_mut()
            .get_resource_mut::<StressConfig>()
        {
            config.culling = enabled;
        }
    }
    /// GPU (default) or exact CPU presentation of ordinary frames.
    pub fn set_gpu_presentation(&mut self, enabled: bool) {
        self.app.set_gpu_presentation(enabled);
    }
    /// Splits fixed ticks across frames around their due time (default).
    pub fn set_tick_pipelining(&mut self, enabled: bool) {
        self.app.set_tick_pipelining(enabled);
    }
    /// Temporal antialiasing of GPU-presented frames in this and later
    /// workloads, over their own settings (the highlands' is on).
    pub fn set_temporal_antialiasing(&mut self, enabled: bool) {
        self.temporal_antialiasing = Some(enabled);
        self.apply_world_settings();
    }
    pub fn temporal_antialiasing(&self) -> bool {
        self.app.temporal_antialiasing()
    }
    /// Occlusion culling of GPU-presented frames in this and later
    /// workloads, over their own settings (on).
    pub fn set_occlusion_culling(&mut self, enabled: bool) {
        self.occlusion_culling = Some(enabled);
        self.apply_world_settings();
    }
    pub fn occlusion_culling(&self) -> bool {
        self.app.occlusion_culling()
    }
    /// Sub-pixel culling of GPU-presented frames in this and later
    /// workloads, over their own settings (on).
    pub fn set_subpixel_culling(&mut self, enabled: bool) {
        self.subpixel_culling = Some(enabled);
        self.apply_world_settings();
    }
    pub fn subpixel_culling(&self) -> bool {
        self.app.subpixel_culling()
    }
    /// Whether instances draw at levels of detail (on by
    /// default) or always as their mesh, in this and later workloads.
    pub fn set_level_of_detail(&mut self, enabled: bool) {
        self.level_of_detail = enabled;
        self.apply_world_settings();
    }
    /// Whether draw groups draw impostors where they fit (on
    /// by default) or only their meshes' levels, in this and later
    /// workloads.
    pub fn set_impostors(&mut self, enabled: bool) {
        self.impostors = enabled;
        self.apply_world_settings();
    }
    /// Whether virtual geometry shades through the visibility buffer
    /// or as it draws (the default), in this and later
    /// workloads.
    pub fn set_visibility_buffer(&mut self, enabled: bool) {
        self.visibility_buffer = enabled;
        self.apply_world_settings();
    }
    /// The workload's overview, or the authored scene's player back where
    /// it started.
    pub fn reset_camera(&mut self) {
        match &self.authored {
            Some(authored) => authored.reset_player(&mut self.app.engine),
            None => set_view(&mut self.app.engine, "overview").expect("known view"),
        }
        #[cfg(feature = "dev")]
        self.development.clear_input();
    }
    pub fn suspend(&mut self) {
        self.app.suspend();
        #[cfg(feature = "dev")]
        self.development.clear_input();
    }
}

#[cfg(feature = "dev")]
#[wasm_bindgen]
impl StressGame {
    pub fn dev_command(&mut self, request: &str, width: u32, height: u32, paused: bool) -> String {
        // Read-only polls retain the timings belonging to the last frame.
        let submitted = serde_json::from_str::<serde_json::Value>(request)
            .ok()
            .and_then(|value| {
                value
                    .get("op")
                    .and_then(|op| op.as_str())
                    .map(|op| matches!(op, "step" | "render" | "present" | "camera" | "culling"))
            })
            == Some(true);
        let previous = *self.app.engine.app.world().resource::<FrameMetrics>();
        if submitted {
            self.app
                .engine
                .app
                .world_mut()
                .resource_mut::<FrameMetrics>()
                .begin_frame();
        }
        let reply = self.development.command(
            &mut self.app,
            self.authored.as_ref().and_then(AuthoredScene::instance),
            request,
            width,
            height,
            paused,
        );
        if submitted {
            if serde_json::from_str::<serde_json::Value>(&reply)
                .ok()
                .is_some_and(|value| value["ok"] == true)
            {
                let enabled = self.app.engine.frustum_culling();
                if let Some(mut config) = self
                    .app
                    .engine
                    .app
                    .world_mut()
                    .get_resource_mut::<StressConfig>()
                {
                    config.culling = enabled;
                }
            } else {
                *self
                    .app
                    .engine
                    .app
                    .world_mut()
                    .resource_mut::<FrameMetrics>() = previous;
            }
        }
        reply
    }

    /// Reloads the authored scene's saved source when the active revision
    /// is `expected_revision` and the saved one `revision`.
    pub fn dev_reload(
        &mut self,
        source: &str,
        expected_revision: &str,
        revision: &str,
        files: Option<SceneFileSet>,
    ) -> String {
        let files = files.unwrap_or_default();
        let result = (|| {
            let Some(authored) = self.authored.as_mut() else {
                return Err(engine::scene::SceneError {
                    code: "unsupported_command",
                    path: "$.op".into(),
                    message: "workloads have no authored scene to reload".into(),
                });
            };
            let (Ok(expected), Ok(target)) =
                (expected_revision.parse::<u64>(), revision.parse::<u64>())
            else {
                return Err(engine::scene::SceneError {
                    code: "invalid_revision",
                    path: "$.revision".into(),
                    message: "revisions must be unsigned decimal strings".into(),
                });
            };
            if authored.revision() != expected {
                return Err(engine::scene::SceneError {
                    code: "revision_conflict",
                    path: "$.expected_revision".into(),
                    message: format!("active revision is {}", authored.revision()),
                });
            }
            let document = SceneDocument::from_json(source)?;
            if document.id != authored.scene_id() || document.revision != target {
                return Err(engine::scene::SceneError {
                    code: "revision_conflict",
                    path: "$.revision".into(),
                    message: "saved scene ID and revision must match the requested scene".into(),
                });
            }
            authored.reload(&mut self.app.engine, source, files.files())
        })();
        let instance = self.authored.as_ref().and_then(AuthoredScene::instance);
        match result {
            Ok(()) => {
                self.app.suspend();
                self.development.clear_input();
                self.development.result(
                    &mut self.app,
                    instance,
                    serde_json::json!({"reloaded":true,"saved_by_reload":false}),
                )
            }
            Err(error) => self.development.error(
                &mut self.app,
                instance,
                error.code,
                &error.path,
                &error.message,
            ),
        }
    }
}

fn js_error(message: String) -> JsValue {
    JsValue::from_str(&message)
}

fn scene_error(error: engine::scene::SceneError) -> JsValue {
    JsValue::from_str(&error.to_string())
}

#[cfg(feature = "profiling")]
#[wasm_bindgen]
impl StressGame {
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
    pub fn profiling_status(&self) -> String {
        self.app.profiling_status().to_string()
    }
    pub fn profiling_report(&mut self) -> String {
        self.app.profiling_report().to_string()
    }
}

#[cfg(feature = "dev")]
fn development_tools() -> engine::dev_browser::DevBrowser {
    use engine::dev::InspectorDescriptor;
    let mut development = engine::dev_browser::DevBrowser::default();
    // Scenes and fields come from the types; the schema adds their ranges.
    let fields = serde_json::to_value(StressConfig::default())
        .expect("configuration contains finite integer values");
    let required: Vec<&String> = fields
        .as_object()
        .expect("a configuration serializes as an object")
        .keys()
        .collect();
    development.tools.register_resource::<StressConfig>(
        InspectorDescriptor::new("StressConfig", "Applied deterministic stress workload; changes rebuild its runtime", serde_json::json!({
            "type":"object", "additionalProperties":false,
            "properties":{
                "seed":{"type":"integer","minimum":0,"maximum":4294967295u64},
                "renderables":{"type":"integer","minimum":1,"maximum":4000000},
                "moving_percent":{"type":"integer","minimum":0,"maximum":100},
                "hierarchy":{"type":"boolean"}, "interpolation":{"type":"boolean"},
                "colliders":{"type":"integer","minimum":0,"maximum":10000},
                "sensor_percent":{"type":"integer","minimum":0,"maximum":100},
                "dense_colliders":{"type":"boolean"},
                "queries_per_tick":{"type":"integer","minimum":0,"maximum":512},
                "mesh_variety":{"enum":[1,8,32,64]}, "textured":{"type":"boolean"}, "culling":{"type":"boolean"},
                "scene":{"enum":StressScene::ALL}, "compressed_textures":{"type":"boolean"}
            },
            "required":required
        })),
        |config| serde_json::to_value(config).expect("configuration contains finite integer values"),
    ).expect("stress inspector registration is valid");
    development.tools.register_resource::<FrameMetrics>(
        InspectorDescriptor::new("StressMetrics", "Last frame CPU simulation subsets and bounded spatial query counters", serde_json::json!({
            "type":"object", "additionalProperties":false,
            "properties":{
                "sync_pre_ms":{"type":"number","minimum":0},
                "sync_post_ms":{"type":"number","minimum":0},
                "sensor_ms":{"type":"number","minimum":0},
                "queries_ms":{"type":"number","minimum":0},
                "animation_ms":{"type":"number","minimum":0},
                "query_calls":{"type":"integer","minimum":0},
                "query_hits":{"type":"integer","minimum":0},
                "query_errors":{"type":"integer","minimum":0},
                "total_calls":{"type":"string","pattern":"^[0-9]+$"},
                "total_hits":{"type":"string","pattern":"^[0-9]+$"},
                "total_errors":{"type":"string","pattern":"^[0-9]+$"},
                "last_error":{"type":["string","null"]}
            },
            "required":["sync_pre_ms","sync_post_ms","sensor_ms","queries_ms","animation_ms",
                "query_calls","query_hits","query_errors","total_calls","total_hits","total_errors","last_error"]
        })),
        |metrics| serde_json::json!({
            "sync_pre_ms":metrics.sync_pre_ms,"sync_post_ms":metrics.sync_post_ms,
            "sensor_ms":metrics.sensor_ms,"queries_ms":metrics.queries_ms,"animation_ms":metrics.animation_ms,
            "query_calls":metrics.query_calls,"query_hits":metrics.query_hits,"query_errors":metrics.query_errors,
            "total_calls":metrics.total_calls.to_string(),"total_hits":metrics.total_hits.to_string(),
            "total_errors":metrics.total_errors.to_string(),"last_error":metrics.last_error,
        }),
    ).expect("stress metrics registration is valid");
    development
        .tools
        .register_resource::<crate::PlayerWalk>(
            InspectorDescriptor::new(
                "PlayerWalk",
                "The authored scene's player's failed movement queries",
                serde_json::json!({
                    "type":"object", "additionalProperties":false,
                    "properties":{"collision_errors":{"type":"integer","minimum":0}},
                    "required":["collision_errors"]
                }),
            ),
            |walk| serde_json::json!({"collision_errors":walk.collision_errors}),
        )
        .expect("player walk registration is valid");
    development
}

impl StressGame {
    /// Writes the kept settings into the current world's resources: each
    /// workload builds a new world with their defaults.
    fn apply_world_settings(&mut self) {
        let world = self.app.engine.app.world_mut();
        if let Some(mut settings) = world.get_resource_mut::<LevelOfDetail>() {
            settings.enabled = self.level_of_detail;
            settings.impostors = self.impostors;
        }
        if let Some(mut settings) = world.get_resource_mut::<VirtualGeometry>() {
            settings.visibility_buffer = self.visibility_buffer;
        }
        if let Some(enabled) = self.occlusion_culling {
            self.app.set_occlusion_culling(enabled);
        }
        if let Some(enabled) = self.subpixel_culling {
            self.app.set_subpixel_culling(enabled);
        }
        if let Some(enabled) = self.temporal_antialiasing {
            self.app.set_temporal_antialiasing(enabled);
        }
    }
}
