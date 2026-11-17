use engine_core::{Error, ErrorCode};
use engine_dev::results::{CaptureResult, FramesResult, GameResult, KeptArtifact, result};
use engine_dev::{
    BrowserControl, CaptureRequest, Effect, HostContext, Outcome, ProfileRequest, Step, ToolCall,
};
use engine_scene::SceneFiles;
use engine_settings::Expected;
use serde_json::{Value, json};

use super::{Call, Handled, Looped};
use crate::host::InputSource;
use crate::{Game, GameHost};

#[derive(Debug)]
pub(crate) enum Pending {
    Paused,
    Frames {
        count: u32,
        interval: f64,
        ran: u32,
        show: bool,
    },
    Shown {
        reply: Value,
    },
    Game {
        name: String,
    },
    Reload {
        expected_revision: String,
        revision: String,
    },
    Profile(Box<ProfileRun>),
}

impl Pending {
    pub(crate) fn input_source(&self) -> InputSource {
        match self {
            Self::Frames { .. } => InputSource::Agent,
            Self::Paused
            | Self::Shown { .. }
            | Self::Game { .. }
            | Self::Reload { .. }
            | Self::Profile(_) => InputSource::Player,
        }
    }
}

pub(crate) struct ProfileRun {
    request: ProfileRequest,
    ran: u32,
    frames: Vec<Vec<engine_profile::Row>>,
}

impl std::fmt::Debug for ProfileRun {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ProfileRun")
            .field("request", &self.request)
            .field("ran", &self.ran)
            .finish_non_exhaustive()
    }
}

fn status() -> Call<'static> {
    Call::Tool(ToolCall::status())
}

fn query(name: &str) -> Call<'_> {
    Call::Tool(ToolCall::Query(name, json!({})))
}

fn render() -> Call<'static> {
    Call::Control("render", json!({}))
}

fn shown(alpha: Option<f64>) -> Call<'static> {
    match alpha {
        Some(alpha) => Call::Control("present", json!({ "alpha": alpha })),
        None => render(),
    }
}

fn failed_frame(code: ErrorCode, frame: u32, frames: u32, failed: &str) -> Error {
    Error::new(code, "$").got(json!({ "frame": frame, "frames": frames, "failed": failed }))
}

impl<G: Game> GameHost<G> {
    pub fn dev_request(&mut self, line: &str, context: &HostContext) -> Step {
        self.dev_pending = None;
        self.collect_pipelines();
        let mut command = match serde_json::from_str::<Value>(line) {
            Ok(command @ Value::Object(_)) if command["verb"].is_string() => command,
            _ => {
                return self.dev_refused(
                    Error::new(ErrorCode::InvalidRequest, "$.command").expected(Expected::Object),
                );
            }
        };
        command["version"] = json!(engine_dev::PROTOCOL_VERSION);
        let size = (context.width, context.height);
        match self.dev_handle(&command.to_string(), size, context.paused) {
            Handled::Reply(reply) => self.dev_done(reply, context),
            Handled::Loop(Looped::Control(control)) => self.dev_control(control),
            Handled::Loop(Looped::Capture(request)) => self.dev_capture(&request, context),
            Handled::Loop(Looped::Profile(request)) => {
                self.suspend();
                drop(engine_profile::take_rows());
                let run = ProfileRun {
                    request,
                    ran: 0,
                    frames: Vec::new(),
                };
                self.dev_effect(
                    Pending::Profile(Box::new(run)),
                    Effect::Frame { timestamp: 0.0 },
                )
            }
        }
    }

    pub fn dev_resume(&mut self, outcome: Outcome, context: &HostContext) -> Step {
        let Some(pending) = self.dev_pending.take() else {
            return self.dev_refused(
                Error::new(ErrorCode::InvalidRequest, "$").expected(Expected::Running),
            );
        };
        match pending {
            Pending::Paused => {
                let status = self.dev_invoke(status(), context);
                self.dev_done(status, context)
            }
            Pending::Frames {
                count,
                interval,
                ran,
                show,
            } => {
                if let Some(failed) = outcome.failed {
                    return self.dev_refused(failed_frame(
                        ErrorCode::RenderFailed,
                        ran + 1,
                        count,
                        &failed,
                    ));
                }
                let ran = ran + 1;
                if ran < count {
                    let pending = Pending::Frames {
                        count,
                        interval,
                        ran,
                        show,
                    };
                    let timestamp = f64::from(ran) * interval;
                    return self.dev_effect(pending, Effect::Frame { timestamp });
                }
                let mut reply = self.dev_invoke(status(), context);
                reply["result"] = result::frames(FramesResult {
                    frames: count,
                    interval_ms: interval,
                    temporary: true,
                    artifact: None,
                });
                if show {
                    return self.dev_effect(Pending::Shown { reply }, Effect::Show);
                }
                self.dev_done(reply, context)
            }
            Pending::Shown { mut reply } => {
                if let Some(failed) = outcome.failed {
                    return self.dev_refused(Error::new(ErrorCode::CaptureFailed, "$").got(failed));
                }
                if let Some(artifact) = outcome.artifact {
                    let observation = &reply["observation"];
                    let kept = KeptArtifact {
                        path: artifact.path,
                        mime_type: artifact.mime_type,
                        bytes: artifact.bytes,
                        sha256: artifact.sha256,
                        frame: observation["frame"].as_str().map(str::to_owned),
                        tick: observation["tick"].as_str().map(str::to_owned),
                    };
                    reply["result"]["artifact"] =
                        serde_json::to_value(kept).expect("result_serializes");
                }
                self.dev_done(reply, context)
            }
            Pending::Game { name } => {
                if let Some(names) = outcome.missing {
                    return self.dev_refused(
                        Error::new(ErrorCode::UnsupportedCommand, "$.args.name")
                            .got(&name)
                            .variants(names),
                    );
                }
                if let Some(failed) = outcome.failed {
                    return self
                        .dev_refused(Error::new(ErrorCode::GameRule, "$.args.args").got(failed));
                }
                let mut rendered = self.dev_invoke(render(), context);
                if rendered["ok"] == true {
                    rendered["result"] = result::game(GameResult {
                        game: name,
                        result: outcome.result.unwrap_or(Value::Null),
                    });
                }
                self.dev_done(rendered, context)
            }
            Pending::Reload { .. } => {
                if outcome.missing.is_some() {
                    return self.dev_refused(
                        Error::new(ErrorCode::UnsupportedCommand, "$.name").got("reload"),
                    );
                }
                let refused = Error::new(ErrorCode::SourceIo, "$");
                self.dev_refused(match outcome.failed {
                    Some(failed) => {
                        serde_json::from_str(&failed).unwrap_or_else(|_| refused.got(failed))
                    }
                    None => refused.expected(Expected::Present),
                })
            }
            Pending::Profile(run) => self.dev_profile(run, &outcome, context),
        }
    }

    pub fn dev_reloaded(
        &mut self,
        source: &str,
        files: Option<&SceneFiles>,
        context: &HostContext,
    ) -> Step {
        match self.dev_pending.take() {
            Some(Pending::Reload {
                expected_revision,
                revision,
            }) => {
                let mut reply = self.dev_reload(source, &expected_revision, &revision, files);
                if reply["ok"] == true {
                    let rendered = self.dev_invoke(render(), context);
                    if rendered["ok"] != true {
                        return self.dev_done(rendered, context);
                    }
                    reply["observation"] = rendered["observation"].clone();
                }
                self.dev_done(reply, context)
            }
            Some(_) | None => self.dev_refused(
                Error::new(ErrorCode::InvalidRequest, "$").expected(Expected::Running),
            ),
        }
    }

    pub fn dev_refusal(&mut self, error: &Error) -> Value {
        self.dev_pending = None;
        let status = self.dev_run(status(), (1, 1), true);
        json!({"ok": false, "observation": status["observation"], "error": error})
    }

    fn dev_invoke(&mut self, call: Call<'_>, context: &HostContext) -> Value {
        self.dev_run(call, (context.width, context.height), context.paused)
    }

    fn dev_control(&mut self, control: BrowserControl) -> Step {
        match control {
            BrowserControl::Pause {} => {
                self.suspend_live();
                self.dev_effect(Pending::Paused, Effect::Pause { paused: true })
            }
            BrowserControl::Resume {} => {
                self.suspend_live();
                self.dev_effect(Pending::Paused, Effect::Pause { paused: false })
            }
            BrowserControl::Frames {
                count,
                interval_ms,
                capture,
            } => {
                self.suspend_live();
                let pending = Pending::Frames {
                    count,
                    interval: interval_ms,
                    ran: 0,
                    show: capture,
                };
                self.dev_effect(pending, Effect::Frame { timestamp: 0.0 })
            }
            BrowserControl::Game { name, args } => {
                let pending = Pending::Game { name: name.clone() };
                self.dev_effect(pending, Effect::Game { name, args })
            }
            BrowserControl::Reload {
                expected_revision,
                revision,
            } => {
                let pending = Pending::Reload {
                    expected_revision,
                    revision,
                };
                self.dev_effect(pending, Effect::Source)
            }
            other @ (BrowserControl::Present { .. }
            | BrowserControl::Environment(_)
            | BrowserControl::Camera { .. }
            | BrowserControl::Step { .. }
            | BrowserControl::Input { .. }
            | BrowserControl::Render {}
            | BrowserControl::Parent { .. }) => {
                let op = serde_json::to_value(&other)
                    .ok()
                    .and_then(|control| control.get("op").cloned())
                    .unwrap_or_default();
                self.dev_refused(Error::new(ErrorCode::UnsupportedCommand, "$.name").got(op))
            }
        }
    }

    fn dev_capture(&mut self, request: &CaptureRequest, context: &HostContext) -> Step {
        let mut reply = self.dev_invoke(shown(request.alpha), context);
        if reply["ok"] != true {
            return self.dev_done(reply, context);
        }
        let digest = if request.digest {
            let digested = self.dev_invoke(query("digest"), context);
            if digested["ok"] != true {
                return self.dev_done(digested, context);
            }
            Some(digested["result"].clone())
        } else {
            None
        };
        reply["result"] = result::capture(CaptureResult {
            temporary: request.alpha.map(|_| true),
            rendered: true,
            artifact: None,
            digest: None,
        });
        if let Some(digest) = digest {
            reply["result"]["digest"] = digest;
        }
        self.dev_effect(Pending::Shown { reply }, Effect::Show)
    }

    fn dev_profile(
        &mut self,
        mut run: Box<ProfileRun>,
        outcome: &Outcome,
        context: &HostContext,
    ) -> Step {
        if let Some(failed) = &outcome.failed {
            return self.dev_refused(failed_frame(
                ErrorCode::RenderFailed,
                run.ran + 1,
                run.request.frames,
                failed,
            ));
        }
        run.frames.push(engine_profile::take_rows());
        run.ran += 1;
        if run.ran < run.request.frames {
            let timestamp = f64::from(run.ran) * run.request.interval_ms;
            return self.dev_effect(Pending::Profile(run), Effect::Frame { timestamp });
        }
        let (profiled, contents) = engine_dev::profiled(&run.frames, &run.request);
        let mut reply = self.dev_invoke(status(), context);
        if reply["ok"] != true {
            return self.dev_done(reply, context);
        }
        reply["result"] = result::profile(profiled);
        self.dev_effect(
            Pending::Shown { reply },
            Effect::Save {
                name: "profile".to_owned(),
                contents,
            },
        )
    }

    fn stats_value(&self) -> Value {
        serde_json::from_str(&self.stats()).expect("stats_json")
    }

    fn dev_effect(&mut self, pending: Pending, effect: Effect) -> Step {
        self.dev_pending = Some(pending);
        Step::Effect(effect)
    }

    fn dev_done(&mut self, mut reply: Value, context: &HostContext) -> Step {
        self.dev_pending = None;
        if reply["ok"] == true {
            let rendering = self.stats_value();
            let observation = reply["observation"].clone();
            let mut result = reply["result"]
                .as_object_mut()
                .map(std::mem::take)
                .unwrap_or_default();
            let [observed, paused, source, rendered] = engine_core::RESERVED_REPLY_KEYS;
            result.insert(observed.into(), observation);
            result.insert(paused.into(), json!(context.paused));
            result.insert(source.into(), json!(context.source));
            result.insert(rendered.into(), rendering);
            reply["result"] = Value::Object(result);
        }
        Step::Reply(reply)
    }

    fn dev_refused(&mut self, error: Error) -> Step {
        Step::Reply(self.dev_refusal(&error))
    }
}
