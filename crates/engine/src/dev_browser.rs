//! Optional browser adapter for the shared development interface.
//!
//! Calls are synchronous between browser frames. Input is a temporary batch,
//! replaced by each input command and consumed only by explicit stepping.
use crate::browser::BrowserApp;
use engine_core::ErrorCode;
use engine_core::{ActionInput, DevStepError, InputState};
use engine_dev::{BrowserControl, Context, DevError, DevTools, RenderedMesh, RendererFacts};
use engine_scene::SceneInstance;
use serde_json::{Value, json};
use wasm_bindgen::JsValue;

/// What the page's renderer made of the world's assets, for inspectors.
struct Facts<'a>(&'a engine_render::Renderer);

impl RendererFacts for Facts<'_> {
    fn texture_resident(&self, texture: engine_core::TextureHandle) -> bool {
        self.0.is_texture_resident(texture)
    }

    fn mesh(&self, mesh: engine_core::MeshHandle) -> Option<RenderedMesh> {
        let (skinned, levels) = self.0.mesh_levels(mesh)?;
        Some(RenderedMesh { skinned, levels })
    }

    fn impostor(
        &self,
        mesh: engine_core::MeshHandle,
        material: engine_core::MaterialHandle,
    ) -> Option<bool> {
        self.0.impostor_baked(mesh, material)
    }

    fn drawn_level(&self, slot: u32) -> Option<u32> {
        self.0.drawn_level(slot)
    }
}

/// Per-session temporary control state; excluded from ordinary builds.
#[derive(Default)]
pub(crate) struct DevBrowser {
    pub(crate) tools: DevTools,
    input: InputBatch,
    // Terrain edits undo took away, latest last.
    terrain_redo: Vec<(
        engine_core::bevy_ecs::prelude::Entity,
        engine_core::TerrainEdit,
    )>,
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

    /// Whether `source` is a valid control that submits a frame on a host
    /// `paused` or not.
    pub fn submits_frame(&self, source: &str, paused: bool) -> bool {
        if source.len() > engine_dev::MAX_REQUEST_BYTES {
            return false;
        }
        let Ok(control) = serde_json::from_str::<BrowserControl>(source) else {
            return false;
        };
        if control.validate().is_err() || (control.needs_pause() && !paused) {
            return false;
        }
        !matches!(
            control,
            BrowserControl::Input { .. }
                | BrowserControl::Benchmark { .. }
                | BrowserControl::Pause { .. }
                | BrowserControl::Resume { .. }
                | BrowserControl::Capture { .. }
                | BrowserControl::Game { .. }
                | BrowserControl::Reload { .. }
        )
    }

    /// Execute an on-demand request, then attach the observed frame and
    /// revision. `paused` says whether the host is paused: every control but
    /// pause and resume requires it.
    pub fn command(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        source: &str,
        width: u32,
        height: u32,
        paused: bool,
    ) -> String {
        if source.len() > engine_dev::MAX_REQUEST_BYTES {
            return self.error(
                app,
                scene,
                ErrorCode::RequestTooLarge,
                "$",
                &format!("command exceeds {} bytes", engine_dev::MAX_REQUEST_BYTES),
            );
        }
        let request: Value = match serde_json::from_str(source) {
            Ok(value) => value,
            Err(error) => {
                return self.error(app, scene, ErrorCode::InvalidJson, "$", &error.to_string());
            }
        };
        // Commands observe and control completed ticks only.
        app.engine.complete_pending_tick();
        let op = request.get("op").and_then(Value::as_str);
        if !op.is_some_and(|op| engine_dev::CONTROL_OPS.contains(&op)) {
            return self.tools_request(app, scene, source, op == Some("status"));
        }
        let op = op.unwrap_or_default().to_owned();
        match self.control(app, scene, source, &op, (width, height), paused) {
            Ok(result) => self.result(app, scene, result),
            Err(error) => self.error(app, scene, error.code, &error.path, &error.message),
        }
    }

    /// A request the development tools answer, with the renderer's facts;
    /// `status` replies name the build too.
    fn tools_request(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        source: &str,
        status: bool,
    ) -> String {
        let response = execute(&self.tools, app, scene, source);
        if !status {
            return response;
        }
        let mut envelope: Value =
            serde_json::from_str(&response).expect("development response is JSON");
        if envelope["ok"] == true {
            envelope["result"]["build"] = build_identity();
        }
        envelope.to_string()
    }

    /// Runs the browser control `op` in `source` on a `size` canvas: its
    /// result.
    fn control(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        source: &str,
        op: &str,
        (width, height): (u32, u32),
        paused: bool,
    ) -> Result<Value, DevError> {
        if op == "environment" {
            // Its sections parse as a scene environment's, whose errors
            // keep their whole path (a tagged request's lose it).
            let mut request: Value =
                serde_json::from_str(source).expect("the command parsed as JSON");
            if let Some(fields) = request.as_object_mut() {
                for key in ["op", "version", "merge"] {
                    fields.remove(key);
                }
            }
            engine_scene::parse_json::<engine_scene::SceneEnvironment>(&request.to_string())?;
        }
        let control: BrowserControl = engine_scene::parse_json(source)?;
        control.validate()?;
        if control.needs_pause() && !paused {
            return Err(DevError::new(
                ErrorCode::PauseRequired,
                "$",
                format!("pause the host before {op}"),
            ));
        }
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
            BrowserControl::Environment(control) => {
                // The sections of a scene environment, validated alike; the
                // named ones replace their resources, the others stay.
                let current = |app: &BrowserApp| {
                    serde_json::to_value(engine_scene::SceneEnvironment::current(
                        app.engine.app.world(),
                    ))
                    .expect("environments serialize")
                };
                let before = current(app);
                let environment = if control.merge {
                    // Each named section over its current values.
                    let mut request: Value =
                        serde_json::from_str(source).expect("the control parsed as JSON");
                    let mut merged = serde_json::Map::new();
                    for (key, value) in request.as_object_mut().into_iter().flatten() {
                        if ["op", "version", "merge"].contains(&key.as_str()) {
                            continue;
                        }
                        let mut section = before[key.as_str()].clone();
                        overlay(&mut section, value.take());
                        merged.insert(key.clone(), section);
                    }
                    engine_scene::parse_json::<engine_scene::SceneEnvironment>(
                        &Value::Object(merged).to_string(),
                    )?
                } else {
                    control.sections()
                };
                environment.validate("$")?;
                environment.apply_sections(app.engine.app.world_mut());
                rendered(app.dev_render(width, height), "environment changed")?;
                let after = current(app);
                let named = environment.sections();
                let pick = |all: &Value| {
                    named
                        .iter()
                        .map(|&key| (key.to_owned(), all[key].clone()))
                        .collect::<serde_json::Map<_, _>>()
                };
                Ok(json!({
                    "temporary": true,
                    "environment_changed": named,
                    "before": pick(&before),
                    "after": pick(&after),
                }))
            }
            BrowserControl::TerrainEdit {
                terrain,
                id,
                center,
                radius,
                falloff,
                change,
                ..
            } => self.terrain_editing(
                app,
                scene,
                terrain.as_deref(),
                engine_dev::TerrainEditing::Edit {
                    id,
                    center,
                    radius,
                    falloff,
                    change,
                },
                (width, height),
            ),
            BrowserControl::Deform {
                from,
                to,
                radius,
                depth,
                flatness,
                clear,
                ..
            } => {
                let result = engine_dev::press_track(
                    &mut app.engine,
                    engine_dev::Press {
                        from,
                        to,
                        radius,
                        depth,
                        flatness,
                        clear,
                    },
                )?;
                rendered(app.dev_render(width, height), "track pressed")?;
                Ok(result)
            }
            BrowserControl::NavGo {
                entity,
                destination,
                ..
            } => {
                let result = engine_dev::nav_go(&mut app.engine, scene, &entity, destination)?;
                rendered(app.dev_render(width, height), "agent sent")?;
                Ok(result)
            }
            BrowserControl::Animation {
                entity,
                clip,
                time,
                speed,
                looping,
                ..
            } => {
                let result = engine_dev::animate(
                    &mut app.engine,
                    scene,
                    &entity,
                    engine_dev::Animate {
                        clip,
                        time,
                        speed,
                        looping,
                    },
                )?;
                rendered(app.dev_render(width, height), "animation changed")?;
                Ok(result)
            }
            BrowserControl::WorldChanges {
                remove,
                restore,
                clear,
                ..
            } => {
                let result = engine_dev::change_world(&mut app.engine, &remove, &restore, clear)?;
                rendered(app.dev_render(width, height), "world changed")?;
                Ok(result)
            }
            BrowserControl::TerrainUndo { terrain, .. } => self.terrain_editing(
                app,
                scene,
                terrain.as_deref(),
                engine_dev::TerrainEditing::Undo,
                (width, height),
            ),
            BrowserControl::TerrainRedo { terrain, .. } => self.terrain_editing(
                app,
                scene,
                terrain.as_deref(),
                engine_dev::TerrainEditing::Redo,
                (width, height),
            ),
            BrowserControl::Terrain {
                pixel_error,
                freeze,
                view,
                ..
            } => {
                let world = app.engine.app.world_mut();
                if let Some(pixel_error) = pixel_error {
                    // The settings' own rule.
                    let rule = engine_core::TerrainSettings::rule_of("pixel_error")
                        .expect("terrain settings declare pixel_error");
                    if let Err(invalid) =
                        engine_settings::Setting::check(&pixel_error, "pixel_error", rule)
                    {
                        return Err(DevError::new(
                            ErrorCode::OutOfRange,
                            "$.pixel_error",
                            invalid.to_string(),
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
                        engine_dev::TerrainViewRequest::Detail => engine_core::TerrainView::Detail,
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
                use engine_dev::bounds;
                use engine_settings::Setting;
                for (value, field, rule) in [
                    (forward, "forward", bounds::AXIS),
                    (right, "right", bounds::AXIS),
                    (up, "up", bounds::AXIS),
                    (yaw, "yaw", bounds::TURN),
                    (pitch, "pitch", bounds::TURN),
                ] {
                    if let Err(invalid) = value.check(field, rule) {
                        return Err(DevError::new(
                            ErrorCode::InvalidInput,
                            format!("$.{field}"),
                            invalid.rule,
                        ));
                    }
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
                DevStepError::validate(ticks).map_err(|error| {
                    DevError::new(ErrorCode::InvalidStep, "$.ticks", error.to_string())
                })?;
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
                        ErrorCode::RenderFailed,
                        "$",
                        format!("simulation may have advanced: {error:?}"),
                    )
                })?;
                Ok(json!({"ticks":ticks,"temporary":true}))
            }
            BrowserControl::Render { .. } => {
                app.dev_render(width, height).map_err(|error| {
                    DevError::new(ErrorCode::RenderFailed, "$", format!("{error:?}"))
                })?;
                Ok(json!({"rendered":true}))
            }
            // The browser host runs these: it receives them validated, with
            // their defaults.
            control @ (BrowserControl::Benchmark { .. }
            | BrowserControl::Pause { .. }
            | BrowserControl::Resume { .. }
            | BrowserControl::Capture { .. }
            | BrowserControl::Game { .. }
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
        code: ErrorCode,
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
        let status = json!({"version": engine_dev::PROTOCOL_VERSION, "op": "status"});
        serde_json::from_str(&execute(&self.tools, app, scene, &status.to_string()))
            .expect("development tools serialize an object envelope")
    }
}

/// `tools`' reply to `source` about `app`'s engine, with its renderer's facts.
fn execute(
    tools: &DevTools,
    app: &mut BrowserApp,
    scene: Option<&SceneInstance>,
    source: &str,
) -> String {
    let facts = Facts(&app.renderer);
    let context = Context {
        scene,
        renderer: Some(&facts),
        render_slot: None,
    };
    tools.execute(&mut app.engine, context, source)
}

/// What rendering after `what` came to: a development error when it failed.
impl DevBrowser {
    /// Edits, undoes or redoes the terrain `name` names, then
    /// renders.
    fn terrain_editing(
        &mut self,
        app: &mut BrowserApp,
        scene: Option<&SceneInstance>,
        name: Option<&str>,
        editing: engine_dev::TerrainEditing,
        (width, height): (u32, u32),
    ) -> Result<Value, DevError> {
        let result = engine_dev::edit_terrain(
            &mut app.engine,
            scene,
            name,
            &mut self.terrain_redo,
            editing,
        )?;
        rendered(app.dev_render(width, height), "terrain edited")?;
        Ok(result)
    }
}

/// What this build is: the engine's version, the commit xtask built it from
/// (`-dirty` with uncommitted changes) and its features.
fn build_identity() -> Value {
    let features: Vec<&str> = [
        cfg!(feature = "dev").then_some("dev"),
        cfg!(feature = "profiling").then_some("profiling"),
        cfg!(feature = "attribution").then_some("attribution"),
    ]
    .into_iter()
    .flatten()
    .collect();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "commit": option_env!("ENGINE_COMMIT"),
        "features": features,
    })
}

/// `patch`'s fields over `base`'s, objects merged field by field.
fn overlay(base: &mut Value, patch: Value) {
    match (base, patch) {
        (Value::Object(base), Value::Object(patch)) => {
            for (key, value) in patch {
                overlay(base.entry(key).or_insert(Value::Null), value);
            }
        }
        (base, patch) => *base = patch,
    }
}

fn rendered(outcome: Result<(), JsValue>, what: &str) -> Result<(), DevError> {
    outcome.map_err(|error| {
        DevError::new(
            ErrorCode::RenderFailed,
            "$",
            format!("{what} before render failed: {error:?}"),
        )
    })
}
