pub(crate) mod bridge;
mod controls;

use crate::app_host::AppHost;
use engine_core::{Error, ErrorCode, Expected};
use engine_dev::{
    CaptureRequest, Context, Control, DevTools, Envelope, ExplainQuestion, HostControl,
    ProfileRequest, RenderedMesh, RendererFacts, Verb, WatchRequest,
};
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

    pub fn submits_frame(
        &self,
        world: &bevy_ecs::prelude::World,
        source: &str,
        paused: bool,
    ) -> bool {
        let Ok(envelope) = Envelope::parse(source) else {
            return false;
        };
        if envelope.verb != Verb::Control {
            return false;
        }
        match self.tools.parse_control(world, &envelope, paused) {
            Ok(Control::Operation(_)) => true,
            Ok(Control::Host(
                HostControl::Loop(_) | HostControl::Input { .. } | HostControl::DebugPanic,
            ))
            | Err(_) => false,
            Ok(Control::Host(
                HostControl::Present { .. }
                | HostControl::Culling { .. }
                | HostControl::Environment(_)
                | HostControl::Camera { .. }
                | HostControl::Step { .. }
                | HostControl::Render
                | HostControl::DebugFault { .. },
            )) => true,
        }
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
        let envelope = match Envelope::parse(source) {
            Ok(envelope) => envelope,
            Err(error) => return self.refusal(app, scene, &error),
        };
        app.engine.complete_pending_tick();
        let answered = match envelope.verb {
            Verb::Describe | Verb::Inspect | Verb::Query => {
                let status = envelope.verb == Verb::Query
                    && envelope.name().is_ok_and(|name| name == "status");
                return self.tools_request(app, scene, source, status);
            }
            Verb::Control => self.control(app, scene, &envelope, (width, height), paused),
            Verb::Capture | Verb::Profile | Verb::Explain | Verb::Watch => {
                looped(&envelope, paused)
            }
            Verb::Check => {
                Err(Error::new(ErrorCode::UnsupportedCommand, "$.verb").got(Verb::Check.name()))
            }
        };
        match answered {
            Ok(result) => self.result(app, scene, result),
            Err(error) => self.refusal(app, scene, &error),
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
        let mut envelope: Value = serde_json::from_str(&response).expect("reply_json");
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
        envelope: &Envelope,
        (width, height): (u32, u32),
        paused: bool,
    ) -> Result<Value, Error> {
        let control = self
            .tools
            .parse_control(app.engine.app.world(), envelope, paused)?;
        let args = envelope.args()?;
        controls::run(self, app, scene, (args, control), (width, height))
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

    pub fn refusal(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        error: &Error,
    ) -> String {
        let mut envelope = self.observation(app, scene);
        envelope["ok"] = json!(false);
        envelope
            .as_object_mut()
            .expect("envelope_object")
            .remove("result");
        envelope["error"] = json!(error);
        envelope.to_string()
    }

    fn observation(&mut self, app: &mut AppHost, scene: Option<&SceneInstance>) -> Value {
        let status = engine_dev::named(Verb::Query, "status", json!({}));
        serde_json::from_str(&execute(&self.tools, app, scene, &status)).expect("envelope_object")
    }
}

fn looped(envelope: &Envelope, paused: bool) -> Result<Value, Error> {
    let needs_pause = match envelope.verb {
        Verb::Capture => {
            let request: CaptureRequest = envelope.body()?;
            request.validate()?;
            true
        }
        Verb::Profile => {
            let request: ProfileRequest = envelope.body()?;
            request.validate()?;
            if request.deep.is_some() {
                return Err(Error::new(ErrorCode::ProfilingUnavailable, "$.deep")
                    .expected(Expected::Absent));
            }
            true
        }
        Verb::Explain => {
            let question: ExplainQuestion = envelope.body()?;
            question.validate()?;
            question.needs_pause()
        }
        Verb::Watch => {
            let request: WatchRequest = envelope.body()?;
            request.validate()?;
            true
        }
        verb @ (Verb::Describe | Verb::Inspect | Verb::Query | Verb::Control | Verb::Check) => {
            return Err(Error::new(ErrorCode::UnsupportedCommand, "$.verb").got(verb.name()));
        }
    };
    if needs_pause && !paused {
        return Err(Error::new(ErrorCode::PauseRequired, "$")
            .expected(Expected::Paused)
            .got(envelope.verb.name()));
    }
    let mut control = envelope.fields.clone();
    control.insert("op".to_owned(), json!(envelope.verb.name()));
    Ok(json!({ "control": control }))
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
