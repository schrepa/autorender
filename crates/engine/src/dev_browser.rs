//! Optional browser adapter for the shared development interface.
//!
//! Calls are synchronous between browser frames. Input is a temporary batch,
//! replaced by each input command and consumed only by explicit stepping.
use crate::{ActionInput, DevStepError, InputState, browser::BrowserApp};
use engine_dev::{BrowserControl, DevTools};
use engine_scene::SceneInstance;
use serde_json::{Value, json};

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
        if request.get("version").and_then(Value::as_u64) != Some(1) {
            return self.error(
                app,
                scene,
                "protocol_version",
                "$.version",
                "expected version 1",
            );
        }
        if !matches!(
            request.get("op").and_then(Value::as_str),
            Some("input" | "step" | "render" | "culling" | "camera" | "present" | "environment")
        ) {
            let response = self.tools.execute(&mut app.engine, scene, source);
            if request.get("op").and_then(Value::as_str) != Some("assets") {
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
            return response;
        }
        let control: BrowserControl = match serde_json::from_value(request) {
            Ok(control) => control,
            Err(error) => {
                return self.error(app, scene, "invalid_request", "$", &error.to_string());
            }
        };
        match control {
            BrowserControl::Present { alpha, gpu, .. } => {
                let alpha = match engine_dev::validated_presentation_alpha(alpha) {
                    Ok(alpha) => alpha,
                    Err(error) => {
                        return self.error(app, scene, error.code, &error.path, &error.message);
                    }
                };
                match app.dev_present(alpha, width, height, gpu) {
                    Ok(()) => self.result(app, scene, json!({"temporary":true,"rendered":true})),
                    Err(error) => self.error(
                        app,
                        scene,
                        "render_failed",
                        "$",
                        &format!("presentation prepared before render failed: {error:?}"),
                    ),
                }
            }
            BrowserControl::Culling { enabled, .. } => {
                app.engine.set_frustum_culling(enabled);
                match app.dev_render(width, height) {
                    Ok(()) => self.result(
                        app,
                        scene,
                        json!({"temporary":true,"culling_enabled":enabled}),
                    ),
                    Err(error) => self.error(
                        app,
                        scene,
                        "render_failed",
                        "$",
                        &format!("culling changed before render failed: {error:?}"),
                    ),
                }
            }
            BrowserControl::Camera {
                position,
                target,
                vertical_fov_radians,
                near,
                far,
                ..
            } => {
                let camera = match engine_dev::validated_camera(
                    position,
                    target,
                    vertical_fov_radians,
                    near,
                    far,
                ) {
                    Ok(camera) => camera,
                    Err(error) => {
                        return self.error(app, scene, error.code, &error.path, &error.message);
                    }
                };
                let world = app.engine.app.world_mut();
                if let Some(mut controller) =
                    world.get_resource_mut::<crate::camera::FlyCameraController>()
                {
                    controller.align_to(&camera);
                }
                world.insert_resource(camera);
                match app.dev_render(width, height) {
                    Ok(()) => {
                        self.result(app, scene, json!({"temporary":true,"camera_changed":true}))
                    }
                    Err(error) => self.error(
                        app,
                        scene,
                        "render_failed",
                        "$",
                        &format!("camera changed before render failed: {error:?}"),
                    ),
                }
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
                };
                let settings = match environment.resolve("$") {
                    Ok(settings) => settings,
                    Err(error) => {
                        return self.error(app, scene, error.code, &error.path, &error.message);
                    }
                };
                settings.apply_sections(app.engine.app.world_mut());
                let changed: Vec<&str> = [
                    ("atmosphere", settings.atmosphere.is_some()),
                    ("shadows", settings.shadows.is_some()),
                    ("ambient_occlusion", settings.ambient_occlusion.is_some()),
                    ("bloom", settings.bloom.is_some()),
                    ("local_lights", settings.local_lights.is_some()),
                    ("sun_shafts", settings.sun_shafts.is_some()),
                    ("reflections", settings.reflections.is_some()),
                    ("particles", settings.particles.is_some()),
                    ("refraction", settings.refraction.is_some()),
                ]
                .into_iter()
                .filter_map(|(name, set)| set.then_some(name))
                .collect();
                match app.dev_render(width, height) {
                    Ok(()) => self.result(
                        app,
                        scene,
                        json!({"temporary":true,"environment_changed":changed}),
                    ),
                    Err(error) => self.error(
                        app,
                        scene,
                        "render_failed",
                        "$",
                        &format!("environment changed before render failed: {error:?}"),
                    ),
                }
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
                let batch = InputBatch {
                    forward,
                    right,
                    up,
                    yaw,
                    pitch,
                    held,
                    pressed,
                    released,
                };
                if [batch.forward, batch.right, batch.up]
                    .iter()
                    .any(|v| !v.is_finite() || v.abs() > 1.0)
                    || [batch.yaw, batch.pitch]
                        .iter()
                        .any(|v| !v.is_finite() || v.abs() > 1000.0)
                {
                    return self.error(
                        app,
                        scene,
                        "invalid_input",
                        "$",
                        "axes must be in [-1, 1]; look deltas in [-1000, 1000]",
                    );
                }
                self.input = batch;
                self.result(
                    app,
                    scene,
                    json!({"temporary":true,"applies_on":"next_step"}),
                )
            }
            BrowserControl::Step { ticks, .. } => {
                if let Err(error) = DevStepError::validate(ticks) {
                    return self.error(app, scene, "invalid_step", "$.ticks", &error.to_string());
                }
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
                match app.dev_step(ticks, width, height, input) {
                    Ok(()) => self.result(app, scene, json!({"ticks":ticks,"temporary":true})),
                    Err(error) => self.error(
                        app,
                        scene,
                        "render_failed",
                        "$",
                        &format!("simulation may have advanced: {error:?}"),
                    ),
                }
            }
            BrowserControl::Render { .. } => match app.dev_render(width, height) {
                Ok(()) => self.result(app, scene, json!({"rendered":true})),
                Err(error) => self.error(app, scene, "render_failed", "$", &format!("{error:?}")),
            },
            _ => self.error(
                app,
                scene,
                "unsupported_command",
                "$.op",
                "control is owned by the browser host",
            ),
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
