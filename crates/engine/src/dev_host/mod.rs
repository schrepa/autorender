pub(crate) mod bridge;
mod controls;

use crate::app_host::AppHost;
use engine_core::{Error, ErrorCode};
use engine_dev::{
    BrowserControl, CaptureRequest, Context, Control, DevTools, Envelope, ProfileRequest,
    RendererFacts, ToolCall, Verb,
};
use engine_scene::SceneInstance;
use engine_settings::Expected;
use serde_json::{Value, json};

struct Facts<'a>(&'a engine_render::Renderer);

impl RendererFacts for Facts<'_> {
    fn texture_resident(&self, texture: engine_core::TextureHandle) -> bool {
        self.0.is_texture_resident(texture)
    }
}

pub(crate) struct DevHost {
    pub(crate) tools: DevTools,
    agent: crate::host::DeviceGatherer,
    input: InputBatch,
}

impl Default for DevHost {
    fn default() -> Self {
        Self {
            tools: DevTools::new(crate::stats::frame_stats_schema()),
            agent: crate::host::DeviceGatherer::default(),
            input: InputBatch::default(),
        }
    }
}

#[derive(Default)]
struct InputBatch {
    held: Vec<engine_core::Driven>,
    once: Vec<engine_core::Driven>,
}

pub(crate) enum Call<'a> {
    Tool(ToolCall<'a>),
    Control(&'a str, Value),
}

pub(crate) enum Handled {
    Reply(Value),
    Loop(Looped),
}

pub(crate) enum Looped {
    Control(BrowserControl),
    Capture(CaptureRequest),
    Profile(ProfileRequest),
}

enum Done {
    Result(Value),
    Loop(Looped),
}

impl DevHost {
    pub fn drop_edges(&mut self) {
        self.input.drop_edges();
        self.agent.drop_edges();
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
        let (Ok(name), Ok(args)) = (envelope.name(), envelope.args()) else {
            return false;
        };
        self.submits(world, (name, args), paused)
    }

    pub fn submits(
        &self,
        world: &bevy_ecs::prelude::World,
        named: (&str, Value),
        paused: bool,
    ) -> bool {
        match self.tools.control(world, named, paused) {
            Ok(Control::Operation(_)) => true,
            Ok(Control::Host(
                BrowserControl::Benchmark { .. }
                | BrowserControl::Frames { .. }
                | BrowserControl::Pause { .. }
                | BrowserControl::Resume { .. }
                | BrowserControl::Game { .. }
                | BrowserControl::Reload { .. }
                | BrowserControl::Environment(_)
                | BrowserControl::Parent { .. }
                | BrowserControl::Input { .. },
            ))
            | Err(_) => false,
            Ok(Control::Host(
                BrowserControl::Present { .. }
                | BrowserControl::Camera { .. }
                | BrowserControl::Step { .. }
                | BrowserControl::Render {},
            )) => true,
        }
    }

    pub fn command(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        source: &str,
        size: (u32, u32),
        paused: bool,
    ) -> Handled {
        let envelope = match Envelope::parse(source) {
            Ok(envelope) => envelope,
            Err(error) => return Handled::Reply(self.refusal(app, scene, &error)),
        };
        let answered = match envelope.verb {
            Verb::Describe | Verb::Inspect | Verb::Query => {
                return Handled::Reply(match ToolCall::of(&envelope) {
                    Ok(call) => self.tool(app, scene, call),
                    Err(error) => self.refusal(app, scene, &error),
                });
            }
            Verb::Control => self
                .tools
                .parse_control(app.engine.app.world(), &envelope, paused)
                .and_then(|control| controls::run(self, app, scene, control, size)),
            Verb::Capture | Verb::Profile => looped(&envelope, paused).map(Done::Loop),
        };
        self.answered(app, scene, answered)
    }

    pub fn call(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        call: Call<'_>,
        size: (u32, u32),
        paused: bool,
    ) -> Value {
        match call {
            Call::Tool(call) => self.tool(app, scene, call),
            Call::Control(name, args) => {
                let answered = self
                    .tools
                    .control(app.engine.app.world(), (name, args), paused)
                    .and_then(|control| controls::run(self, app, scene, control, size));
                match self.answered(app, scene, answered) {
                    Handled::Reply(reply) => reply,
                    Handled::Loop(_) => self.refusal(
                        app,
                        scene,
                        &Error::new(ErrorCode::UnsupportedCommand, "$.name").got(name),
                    ),
                }
            }
        }
    }

    fn answered(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        answered: Result<Done, Error>,
    ) -> Handled {
        match answered {
            Ok(Done::Result(result)) => Handled::Reply(self.result(app, scene, result)),
            Ok(Done::Loop(looped)) => Handled::Loop(looped),
            Err(error) => Handled::Reply(self.refusal(app, scene, &error)),
        }
    }

    fn tool(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        call: ToolCall<'_>,
    ) -> Value {
        let status = matches!(call, ToolCall::Query("status", _));
        let mut reply = self.reply(app, scene, call);
        if status && reply["ok"] == true {
            reply["result"]["build"] = build_identity();
            reply["result"]["adapter"] =
                json!(crate::host::AdapterStatus::of(app.renderer.adapter()));
        }
        reply
    }

    fn reply(&self, app: &mut AppHost, scene: Option<&SceneInstance>, call: ToolCall<'_>) -> Value {
        let facts = Facts(&app.renderer);
        let context = Context {
            scene,
            renderer: Some(&facts),
        };
        self.tools.call(&mut app.engine, context, call)
    }

    pub fn result(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        result: Value,
    ) -> Value {
        let mut envelope = self.reply(app, scene, ToolCall::status());
        envelope["ok"] = json!(true);
        envelope["result"] = result;
        envelope
    }

    pub fn refusal(
        &mut self,
        app: &mut AppHost,
        scene: Option<&SceneInstance>,
        error: &Error,
    ) -> Value {
        let mut envelope = self.reply(app, scene, ToolCall::status());
        envelope["ok"] = json!(false);
        envelope
            .as_object_mut()
            .expect("envelope_object")
            .remove("result");
        envelope["error"] = json!(error);
        envelope
    }
}

fn looped(envelope: &Envelope, paused: bool) -> Result<Looped, Error> {
    let looped = match envelope.verb {
        Verb::Capture => {
            let request: CaptureRequest = envelope.body()?;
            request.validate()?;
            Looped::Capture(request)
        }
        Verb::Profile => {
            let request: ProfileRequest = envelope.body()?;
            request.validate()?;
            if request.deep.is_some() {
                return Err(Error::new(ErrorCode::ProfilingUnavailable, "$.deep")
                    .expected(Expected::Absent));
            }
            Looped::Profile(request)
        }
        verb @ (Verb::Describe | Verb::Inspect | Verb::Query | Verb::Control) => {
            return Err(Error::new(ErrorCode::UnsupportedCommand, "$.verb").got(verb.name()));
        }
    };
    if !paused {
        return Err(Error::new(ErrorCode::PauseRequired, "$")
            .expected(Expected::Paused)
            .got(envelope.verb.name()));
    }
    Ok(looped)
}

fn build_identity() -> Value {
    let features: Vec<&str> = [
        cfg!(feature = "dev").then_some("dev"),
        cfg!(feature = "profiling").then_some("profiling"),
    ]
    .into_iter()
    .flatten()
    .collect();
    json!({
        "version": env!("CARGO_PKG_VERSION"),
        "features": features,
    })
}
