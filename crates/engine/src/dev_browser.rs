//! Optional browser adapter for the shared development interface.
//!
//! Calls are synchronous between browser frames. Input is a temporary batch,
//! replaced by each input command and consumed only by explicit stepping.
use crate::{ActionInput, DevStepError, InputState, browser::BrowserApp};
use engine_dev::{BrowserControl, DevError, DevTools};
use engine_scene::SceneInstance;
use serde_json::{Value, json};
use wasm_bindgen::JsValue;

const MAX_COMMAND_BYTES: usize = 16 * 1024;

/// Per-session temporary control state; excluded from ordinary builds.
#[derive(Default)]
pub struct DevBrowser {
    pub tools: DevTools,
    input: InputBatch,
}

#[derive(Default)]
struct InputBatch {
    forward: f32,
    right: f32,
    up: f32,
    yaw: f32,
    pitch: f32,
    held: u32,
    pressed: u32,
    released: u32,
}

impl DevBrowser {
    pub fn clear_input(&mut self) {
        self.input = InputBatch::default();
    }

    /// Execute an on-demand request, then attach the observed frame and revision.
    /// The host enforces paused state for input, stepping, and render commands.
    pub fn command(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        source: &str,
        width: u32,
        height: u32,
    ) -> String {
        if source.len() > MAX_COMMAND_BYTES {
            return self.error(
                app,
                scene,
                "request_too_large",
                "$",
                "command exceeds 16 KiB",
            );
        }
        let request: Value = match serde_json::from_str(source) {
            Ok(value) => value,
            Err(error) => return self.error(app, scene, "invalid_json", "$", &error.to_string()),
        };
        // Commands observe and control completed ticks only.
        app.engine.complete_pending_tick();
        let op = request.get("op").and_then(Value::as_str);
        if !op.is_some_and(|op| BrowserControl::OPS.contains(&op)) {
            let assets = op == Some("assets");
            return self.tools_request(app, scene, source, assets);
        }
        match self.control(app, request, (width, height)) {
            Ok(result) => self.result(app, scene, result),
            Err(error) => self.error(app, scene, error.code, &error.path, &error.message),
        }
    }

    /// A request the development tools answer; `assets` replies list each
    /// texture's GPU residency too.
    fn tools_request(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        source: &str,
        assets: bool,
    ) -> String {
        let response = self.tools.execute(&mut app.engine, scene, source);
        if !assets {
            return response;
        }
        let mut envelope: Value =
            serde_json::from_str(&response).expect("development response is JSON");
        if envelope["ok"] == true
            && let (Some(scene), Some(rows)) =
                (scene, envelope["result"]["texture_assets"].as_array_mut())
        {
            // These bounded rows and IDs were generated from the same instance.
            for (row, (_, handle)) in rows.iter_mut().zip(scene.textures()) {
                row["gpu_resident"] = json!(app.texture_resident(handle));
            }
        }
        let response = envelope.to_string();
        if response.len() > engine_dev::MAX_RESPONSE_BYTES {
            return self.error(
                app,
                scene,
                "response_too_large",
                "$",
                "asset response exceeds 64 KiB",
            );
        }
        response
    }

    /// Runs the browser control `request` on a `size` canvas: its result.
    fn control(
        &mut self,
        app: &mut BrowserApp,
        request: Value,
        (width, height): (u32, u32),
    ) -> Result<Value, DevError> {
        let control: BrowserControl = serde_json::from_value(request)
            .map_err(|error| DevError::new("invalid_request", "$", error.to_string()))?;
        control.validate()?;
        match control {
            BrowserControl::Present { alpha, gpu, .. } => {
                let alpha = engine_dev::validated_presentation_alpha(alpha)?;
                rendered(
                    app.dev_present(alpha, width, height, gpu),
                    "presentation prepared",
                )?;
                Ok(json!({"temporary":true,"rendered":true}))
            }
            BrowserControl::Culling { enabled, .. } => {
                app.engine.set_frustum_culling(enabled);
                rendered(app.dev_render(width, height), "culling changed")?;
                Ok(json!({"temporary":true,"culling_enabled":enabled}))
            }
            BrowserControl::Camera {
                position,
                target,
                vertical_fov_radians,
                near,
                far,
                ..
            } => {
                let camera = engine_dev::validated_camera(
                    position,
                    target,
                    vertical_fov_radians,
                    near,
                    far,
                )?;
                let world = app.engine.app.world_mut();
                if let Some(mut controller) =
                    world.get_resource_mut::<crate::camera::FlyCameraController>()
                {
                    controller.align_to(&camera);
                }
                world.insert_resource(camera);
                rendered(app.dev_render(width, height), "camera changed")?;
                Ok(json!({"temporary":true,"camera_changed":true}))
            }
            BrowserControl::Environment {
                atmosphere,
                shadows,
                ambient_occlusion,
                bloom,
                local_lights,
                sun_shafts,
                reflections,
                particles,
                refraction,
                decals,
                reflection_probes,
                virtual_geometry,
                level_of_detail,
                terrain,
                ..
            } => {
                // The sections of a scene environment, validated alike; the
                // named ones replace their resources, the others stay.
                let environment = engine_scene::SceneEnvironment {
                    atmosphere,
                    shadows,
                    ambient_occlusion,
                    bloom,
                    local_lights,
                    sun_shafts,
                    reflections,
                    particles,
                    refraction,
                    decals,
                    reflection_probes,
                    virtual_geometry,
                    level_of_detail,
                    terrain,
                };
                environment.validate("$")?;
                environment.apply_sections(app.engine.app.world_mut());
                rendered(app.dev_render(width, height), "environment changed")?;
                Ok(json!({"temporary":true,"environment_changed":environment.sections()}))
            }
            BrowserControl::Terrain {
                pixel_error,
                freeze,
                view,
                ..
            } => {
                let world = app.engine.app.world_mut();
                if let Some(pixel_error) = pixel_error {
                    if !(pixel_error.is_finite() && (1.0..=64.0).contains(&pixel_error)) {
                        return Err(DevError::new(
                            "invalid_request",
                            "$.pixel_error",
                            "pixel_error must be in [1, 64]",
                        ));
                    }
                    let mut settings = world
                        .get_resource::<engine_core::TerrainSettings>()
                        .copied()
                        .unwrap_or_default();
                    settings.pixel_error = pixel_error;
                    world.insert_resource(settings);
                }
                let eye = world.resource::<engine_core::Camera>().eye();
                let mut debug = world
                    .get_resource::<engine_core::TerrainDebug>()
                    .copied()
                    .unwrap_or_default();
                if let Some(freeze) = freeze {
                    debug.freeze = freeze.then_some(debug.freeze.unwrap_or(eye));
                }
                if let Some(view) = view {
                    debug.view = match view {
                        engine_dev::TerrainViewRequest::Shaded => engine_core::TerrainView::Shaded,
                        engine_dev::TerrainViewRequest::Levels => engine_core::TerrainView::Levels,
                        engine_dev::TerrainViewRequest::Normals => {
                            engine_core::TerrainView::Normals
                        }
                        engine_dev::TerrainViewRequest::Tiles => engine_core::TerrainView::Tiles,
                        engine_dev::TerrainViewRequest::Materials => {
                            engine_core::TerrainView::Materials
                        }
                        engine_dev::TerrainViewRequest::Sky => engine_core::TerrainView::Sky,
                    };
                }
                world.insert_resource(debug);
                rendered(app.dev_render(width, height), "terrain changed")?;
                let world = app.engine.app.world();
                Ok(json!({
                    "temporary": true,
                    "terrain_changed": true,
                    "frozen_at": debug.freeze.map(|eye| eye.to_array()),
                    "view": view.map(|_| debug.view),
                    "pixel_error": world.get_resource::<engine_core::TerrainSettings>().map(|settings| settings.pixel_error),
                }))
            }
            BrowserControl::Input {
                forward,
                right,
                up,
                yaw,
                pitch,
                held,
                pressed,
                released,
                ..
            } => {
                if [forward, right, up]
                    .iter()
                    .any(|v| !v.is_finite() || v.abs() > 1.0)
                    || [yaw, pitch]
                        .iter()
                        .any(|v| !v.is_finite() || v.abs() > 1000.0)
                {
                    return Err(DevError::new(
                        "invalid_input",
                        "$",
                        "axes must be in [-1, 1]; look deltas in [-1000, 1000]",
                    ));
                }
                self.input = InputBatch {
                    forward,
                    right,
                    up,
                    yaw,
                    pitch,
                    held,
                    pressed,
                    released,
                };
                Ok(json!({"temporary":true,"applies_on":"next_step"}))
            }
            BrowserControl::Step { ticks, .. } => {
                DevStepError::validate(ticks)
                    .map_err(|error| DevError::new("invalid_step", "$.ticks", error.to_string()))?;
                app.engine
                    .app
                    .world_mut()
                    .resource_mut::<ActionInput>()
                    .ingest(self.input.held, self.input.pressed, self.input.released);
                let input = InputState {
                    move_forward: self.input.forward,
                    move_right: self.input.right,
                    move_up: self.input.up,
                    look_yaw: self.input.yaw,
                    look_pitch: self.input.pitch,
                };
                // These edges belong to this request even if subsequent GPU submission fails.
                self.input.pressed = 0;
                self.input.released = 0;
                self.input.yaw = 0.0;
                self.input.pitch = 0.0;
                app.dev_step(ticks, width, height, input).map_err(|error| {
                    DevError::new(
                        "render_failed",
                        "$",
                        format!("simulation may have advanced: {error:?}"),
                    )
                })?;
                Ok(json!({"ticks":ticks,"temporary":true}))
            }
            BrowserControl::Render { .. } => {
                app.dev_render(width, height)
                    .map_err(|error| DevError::new("render_failed", "$", format!("{error:?}")))?;
                Ok(json!({"rendered":true}))
            }
            // The browser host runs these: it receives them validated, with
            // their defaults.
            control @ (BrowserControl::Benchmark { .. }
            | BrowserControl::Pause { .. }
            | BrowserControl::Resume { .. }
            | BrowserControl::Capture { .. }
            | BrowserControl::Reload { .. }) => Ok(json!({ "control": control })),
        }
    }

    /// Produce a result from a game-specific development operation.
    pub fn result(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        result: Value,
    ) -> String {
        let mut envelope = self.observation(app, scene);
        envelope["ok"] = json!(true);
        envelope["result"] = result;
        envelope.to_string()
    }

    /// Preserve a structured game/authoring error at the browser boundary.
    pub fn error(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        code: &str,
        path: &str,
        message: &str,
    ) -> String {
        let mut envelope = self.observation(app, scene);
        envelope["ok"] = json!(false);
        envelope
            .as_object_mut()
            .expect("object envelope")
            .remove("result");
        envelope["error"] = json!({"code":code,"path":path,"message":message});
        envelope.to_string()
    }

    fn observation(&mut self, app: &mut BrowserApp, scene: Option<&SceneInstance>) -> Value {
        serde_json::from_str(&self.tools.execute(
            &mut app.engine,
            scene,
            r#"{"version":1,"op":"status"}"#,
        ))
        .expect("development tools serialize an object envelope")
    }
}

/// What rendering after `what` came to: a development error when it failed.
fn rendered(outcome: Result<(), JsValue>, what: &str) -> Result<(), DevError> {
    outcome.map_err(|error| {
        DevError::new(
            "render_failed",
            "$",
            format!("{what} before render failed: {error:?}"),
        )
    })
}
