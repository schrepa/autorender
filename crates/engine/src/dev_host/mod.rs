pub(crate) mod bridge;
mod controls;

use crate::app_host::AppHost;
use engine_core::ErrorCode;
use engine_dev::{BrowserControl, Context, DevError, DevTools, RenderedMesh, RendererFacts};
use engine_scene::SceneInstance;
use serde_json::{Value, json};

struct Facts<'a>(&'a engine_render::Renderer);

impl RendererFacts for Facts<'_> {
    fn texture_resident(&self, texture: engine_core::TextureHandle) -> bool {
        self.0.is_texture_resident(texture)
    }

    fn mesh(&self, mesh: engine_core::MeshHandle) -> Option<RenderedMesh> {
        let levels = self.0.mesh_levels(mesh)?;
        Some(RenderedMesh { levels })
    }

    fn impostor(
        &self,
        mesh: engine_core::MeshHandle,
        material: engine_core::MaterialHandle,
    ) -> Option<bool> {
        self.0.impostor_resident(mesh, material)
    }

    fn drawn_level(&self, slot: u32) -> Option<u32> {
        self.0.drawn_level(slot)
    }
}

#[derive(Default)]
pub(crate) struct DevHost {
    pub(crate) tools: DevTools,
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

impl DevHost {
    pub fn clear_input(&mut self) {
        self.input = InputBatch::default();
    }

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
                | BrowserControl::Probe { .. }
                | BrowserControl::DebugDeviceLoss { .. }
        )
    }

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
