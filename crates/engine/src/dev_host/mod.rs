//! A host's side of the shared development interface, a page's or a
//! native window's.
//!
//! Calls are synchronous between frames. Input is a temporary batch,
//! replaced by each input command and consumed only by explicit stepping.
pub(crate) mod bridge;
mod controls;

use crate::app_host::AppHost;
use engine_core::ErrorCode;
use engine_dev::{BrowserControl, Context, DevError, DevTools, RenderedMesh, RendererFacts};
use engine_scene::SceneInstance;
use serde_json::{Value, json};

/// What the host's renderer made of the world's assets, for inspectors.
struct Facts<'a>(&'a engine_render::Renderer);

impl RendererFacts for Facts<'_> {
    fn alpha_coverage(
        &self,
        material: &engine_core::Material,
    ) -> Option<engine_core::AlphaCoverageState> {
        self.0.material_alpha_coverage(material)
    }

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
pub(crate) struct DevHost {
    pub(crate) tools: DevTools,
    input: InputBatch,
}

/// The input the next explicit step applies.
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

impl DevHost {
    pub fn clear_input(&mut self) {
        self.input = InputBatch::default();
    }

    /// Whether `source` is a valid control that submits a frame on a host
    /// `paused` or not.
    pub fn submits_frame(&self, source: &str, paused: bool) -> bool {
        if source.len() > engine_dev::MAX_REQUEST_BYTES {
            return false;
        }
        let Ok(control) = engine_dev::parse_request::<BrowserControl>(source) else {
            return false;
        };
        if control.validate().is_err() || (control.needs_pause() && !paused) {
            return false;
        }
        !matches!(
            control,
            BrowserControl::Input { .. }
                | BrowserControl::Benchmark { .. }
                | BrowserControl::Frames { .. }
                | BrowserControl::Pause { .. }
                | BrowserControl::Resume { .. }
                | BrowserControl::Capture { .. }
                | BrowserControl::Game { .. }
                | BrowserControl::Reload { .. }
                | BrowserControl::DebugPanic { .. }
        )
    }

    /// Execute an on-demand request, then attach the observed frame and
    /// revision. `paused` says whether the host is paused: every control but
    /// pause and resume requires it.
    pub fn command(
        &mut self,
        app: &mut AppHost,
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
    /// `status` replies name the build and the GPU too, as every host does.
    fn tools_request(
        &mut self,
        app: &mut AppHost,
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
            envelope["result"]["adapter"] =
                json!(crate::host::AdapterStatus::of(app.renderer.adapter()));
        }
        envelope.to_string()
    }

    /// Runs the control `op` in `source` on a `size` view: its
    /// result.
    fn control(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        source: &str,
        op: &str,
        (width, height): (u32, u32),
        paused: bool,
    ) -> Result<Value, DevError> {
        let control = BrowserControl::parse(source, op, paused)?;
        controls::run(self, app, scene, (source, control.route()), (width, height))
    }

    /// Produce a result from a game-specific development operation.
    pub fn result(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        result: Value,
    ) -> String {
        let mut envelope = self.observation(app, scene);
        envelope["ok"] = json!(true);
        envelope["result"] = result;
        envelope.to_string()
    }

    /// Preserve a structured game/authoring error at the host boundary.
    pub fn error(
        &mut self,
        app: &mut AppHost,
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

    fn observation(&mut self, app: &mut AppHost, scene: Option<&SceneInstance>) -> Value {
        let status = json!({"version": engine_dev::PROTOCOL_VERSION, "op": "status"});
        serde_json::from_str(&execute(&self.tools, app, scene, &status.to_string()))
            .expect("development tools serialize an object envelope")
    }
}

/// `tools`' reply to `source` about `app`'s engine, with its renderer's facts.
fn execute(
    tools: &DevTools,
    app: &mut AppHost,
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
