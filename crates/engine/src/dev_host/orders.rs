use std::any::Any;

use bevy_ecs::world::World;
use engine_core::{Error, ErrorCode, Expected};
use engine_dev::record::MarkPicture;
use engine_registry::{Answer, NameEntry, register, unavailable};
use engine_render::{DeviceHealth, Picture};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};

use super::Shot;
use crate::app_host::Advance;
use crate::host::FramePhase;
use crate::{Game, GameHost};

pub(super) struct Running {
    pub(super) ticket: u32,
    pub(super) id: Option<Value>,
    pub(super) cause: engine_profile::Cause,
    pub(super) order: Order,
}

pub(super) enum Order {
    Step(Step),
    Shoot(Shoot),
    Reload(Reload),
}

impl Order {
    pub(super) fn of(order: Box<dyn Any + Send>) -> Option<Self> {
        let order = match order.downcast::<Step>() {
            Ok(step) => return Some(Self::Step(*step)),
            Err(order) => order,
        };
        let order = match order.downcast::<Shoot>() {
            Ok(shoot) => return Some(Self::Shoot(*shoot)),
            Err(order) => order,
        };
        let order = match order.downcast::<Reload>() {
            Ok(reload) => return Some(Self::Reload(*reload)),
            Err(order) => order,
        };
        let mark = order.downcast::<MarkPicture>().ok()?;
        Some(Self::Shoot(Shoot {
            size: (mark.size[0], mark.size[1]),
            mark: Some(*mark),
            frame: None,
        }))
    }
}

#[derive(Clone, Copy, Deserialize, schemars::JsonSchema)]
#[serde(rename_all = "snake_case")]
enum Until {
    Settled,
    Stable,
    Multiple,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct StepArgs {
    ticks: Option<u32>,
    frames: Option<u32>,
    until: Option<Until>,
    apart: Option<u32>,
    tolerance: Option<u8>,
    of: Option<u64>,
    wall_ms: Option<f64>,
    ticks_per_frame: Option<u32>,
    alpha: Option<f32>,
}

#[derive(Clone, Copy)]
struct Rate {
    ticks: u32,
    alpha: Option<f32>,
}

enum Form {
    Ticks(u32),
    Frames(u32),
    Settled(u32),
    Stable {
        limit: u32,
        apart: u32,
        tolerance: u8,
        asked: Option<u32>,
        last: Option<(u32, Vec<u8>)>,
    },
    Multiple {
        of: u64,
        target: Option<u64>,
    },
    Wall {
        ms: f64,
        since: Option<f64>,
    },
}

pub(super) struct Step {
    form: Form,
    rate: Rate,
    drawn: u32,
    ticks: u64,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
struct CaptureArgs {
    width: u32,
    height: u32,
}

pub(super) struct Shoot {
    size: (u32, u32),
    mark: Option<MarkPicture>,
    frame: Option<u64>,
}

#[derive(Deserialize, schemars::JsonSchema)]
#[serde(deny_unknown_fields)]
pub(super) struct Reload {
    pub(super) expected_revision: String,
    pub(super) revision: String,
    #[serde(skip)]
    asked: bool,
}

fn shape<T: schemars::JsonSchema>() -> Value {
    serde_json::to_value(schemars::schema_for!(T)).expect("schema_serializes")
}

fn step_shape() -> Value {
    let mut shape = shape::<StepArgs>();
    shape["examples"] = json!([
        {"ticks": 600},
        {"frames": 60, "ticks_per_frame": 1},
        {"until": "settled", "frames": 600},
        {"until": "stable", "frames": 600, "apart": 10, "tolerance": 2},
        {"until": "multiple", "of": 2, "ticks_per_frame": 0},
        {"wall_ms": 1000.0},
    ]);
    shape
}

fn parsed<T: serde::de::DeserializeOwned>(args: &Value) -> Result<T, Error> {
    engine_settings::parse_value_at(args.clone(), "$.args")
}

pub(super) fn install(world: &mut World) {
    register(
        world,
        NameEntry::call("step", |_, args| {
            Ok(Answer::Loop(Box::new(Step::of(parsed(args)?)?)))
        })
        .args(step_shape(), json!({"until": "settled", "frames": 600})),
    );
    register(
        world,
        NameEntry::call("capture", |_, args| {
            let CaptureArgs { width, height } = parsed(args)?;
            for (path, value) in [("$.args.width", width), ("$.args.height", height)] {
                if value == 0 {
                    return Err(Error::new(ErrorCode::OutOfRange, path).at_least(1).got(value));
                }
            }
            Ok(Answer::Loop(Box::new(Shoot {
                size: (width, height),
                mark: None,
                frame: None,
            })))
        })
        .args(shape::<CaptureArgs>(), json!({"width": 1280, "height": 720})),
    );
    register(
        world,
        NameEntry::call("reload", |_, args| {
            Ok(Answer::Loop(Box::new(parsed::<Reload>(args)?)))
        })
        .args(
            shape::<Reload>(),
            json!({"expected_revision": "1", "revision": "2"}),
        ),
    );
}

impl Step {
    fn of(args: StepArgs) -> Result<Self, Error> {
        let at_least_one = |path: &str, value: u64| {
            if value == 0 {
                Err(Error::new(ErrorCode::OutOfRange, path).at_least(1).got(value))
            } else {
                Ok(())
            }
        };
        let (named, allowed): (&str, &[&str]) = match (args.until, args.ticks, args.frames) {
            (Some(Until::Settled), ..) => ("settled", &["until", "frames", "ticks_per_frame", "alpha"]),
            (Some(Until::Stable), ..) => (
                "stable",
                &["until", "frames", "apart", "tolerance", "ticks_per_frame", "alpha"],
            ),
            (Some(Until::Multiple), ..) => ("multiple", &["until", "of", "ticks_per_frame", "alpha"]),
            (None, Some(_), _) => ("ticks", &["ticks"]),
            (None, None, Some(_)) => ("frames", &["frames", "ticks_per_frame", "alpha"]),
            (None, None, None) if args.wall_ms.is_some() => ("wall_ms", &["wall_ms"]),
            (None, None, None) => {
                return Err(Error::new(ErrorCode::MissingField, "$.args")
                    .expected(Expected::Present)
                    .variants(["ticks", "frames", "until", "wall_ms"]));
            }
        };
        let given = [
            ("ticks", args.ticks.is_some()),
            ("frames", args.frames.is_some()),
            ("until", args.until.is_some()),
            ("apart", args.apart.is_some()),
            ("tolerance", args.tolerance.is_some()),
            ("of", args.of.is_some()),
            ("wall_ms", args.wall_ms.is_some()),
            ("ticks_per_frame", args.ticks_per_frame.is_some()),
            ("alpha", args.alpha.is_some()),
        ];
        let with = |field: &str, code: ErrorCode, expected: Expected| {
            Error::new(code, format!("$.args.{field}"))
                .expected(expected)
                .variants(allowed.iter().copied())
                .got_whole(json!({"form": named}))
        };
        if let Some((field, _)) = given
            .iter()
            .find(|(field, present)| *present && !allowed.contains(field))
        {
            return Err(with(field, ErrorCode::UnknownField, Expected::Absent));
        }
        let required = |value: Option<u32>, field: &str| {
            let value = value.ok_or_else(|| with(field, ErrorCode::MissingField, Expected::Present))?;
            at_least_one(&format!("$.args.{field}"), u64::from(value))?;
            Ok::<u32, Error>(value)
        };
        if let Some(alpha) = args.alpha
            && !(0.0..=1.0).contains(&alpha)
        {
            return Err(Error::new(ErrorCode::OutOfRange, "$.args.alpha")
                .range(0.0, 1.0)
                .got(alpha));
        }
        let form = match (args.until, args.ticks) {
            (Some(Until::Settled), _) => Form::Settled(required(args.frames, "frames")?),
            (Some(Until::Stable), _) => Form::Stable {
                limit: required(args.frames, "frames")?,
                apart: required(args.apart, "apart")?,
                tolerance: args.tolerance.ok_or_else(|| {
                    with("tolerance", ErrorCode::MissingField, Expected::Present)
                })?,
                asked: None,
                last: None,
            },
            (Some(Until::Multiple), _) => {
                let of = args
                    .of
                    .ok_or_else(|| with("of", ErrorCode::MissingField, Expected::Present))?;
                at_least_one("$.args.of", of)?;
                Form::Multiple { of, target: None }
            }
            (None, Some(_)) => Form::Ticks(required(args.ticks, "ticks")?),
            (None, None) => match args.wall_ms {
                Some(ms) if !(ms.is_finite() && ms > 0.0) => {
                    return Err(Error::new(ErrorCode::OutOfRange, "$.args.wall_ms")
                        .expected(Expected::Above)
                        .got(ms));
                }
                Some(ms) => Form::Wall { ms, since: None },
                None => Form::Frames(required(args.frames, "frames")?),
            },
        };
        Ok(Self {
            form,
            rate: Rate {
                ticks: args.ticks_per_frame.unwrap_or(1),
                alpha: args.alpha,
            },
            drawn: 0,
            ticks: 0,
        })
    }

    fn reply(&self, reached: Option<bool>) -> Value {
        let mut value = json!({"frames": self.drawn, "ticks": self.ticks});
        if let Some(reached) = reached {
            value["reached"] = json!(reached);
        }
        value
    }
}

fn stepped(rate: Rate) -> Advance {
    Advance::Stepped {
        ticks: rate.ticks,
        alpha: rate.alpha,
    }
}

fn hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn refused(error: &Error) -> Value {
    let error = serde_json::to_value(error).unwrap_or_default();
    let reason = error["expected"]
        .as_str()
        .or_else(|| error["code"].as_str())
        .unwrap_or("refused");
    unavailable(format!("picture_not_{reason}"))
}

fn close(before: &[u8], after: &[u8], tolerance: u8) -> bool {
    before.len() == after.len()
        && before
            .iter()
            .zip(after)
            .all(|(before, after)| before.abs_diff(*after) <= tolerance)
}

type Outcome = Option<Result<Value, Error>>;

impl<G: Game> GameHost<G> {
    pub(crate) fn dev_play(
        &mut self,
        timestamp: f64,
        size: (u32, u32),
    ) -> crate::Result<FramePhase> {
        let held = self.held();
        if held != self.development.held {
            self.development.held = held;
            self.suspend();
        }
        let Some(mut running) = self.development.running.take() else {
            if held {
                return Ok(FramePhase::Paused);
            }
            self.framed(Advance::Clock(timestamp), size, Picture::Shown)?;
            return Ok(FramePhase::Playing);
        };
        let played = engine_profile::caused(running.cause, || match &mut running.order {
            Order::Step(step) => self.step(step, timestamp, size),
            Order::Shoot(shoot) => self.shoot(shoot, timestamp, size, held),
            Order::Reload(reload) => self.wait_reload(reload, timestamp, size, held),
        });
        match played {
            Ok((phase, None)) => {
                self.development.running = Some(running);
                Ok(phase)
            }
            Ok((phase, Some(outcome))) => {
                self.answer(&running, outcome);
                Ok(phase)
            }
            Err(error) => {
                self.development.stepping = false;
                self.answer(&running, Err(error.clone()));
                Err(error)
            }
        }
    }

    fn step(
        &mut self,
        step: &mut Step,
        timestamp: f64,
        size: (u32, u32),
    ) -> crate::Result<(FramePhase, Outcome)> {
        self.development.stepping = true;
        let advance = match &mut step.form {
            Form::Ticks(ticks) => Advance::Stepped {
                ticks: *ticks,
                alpha: None,
            },
            Form::Wall { since, .. } => {
                if since.is_none() {
                    self.app.suspend();
                    *since = Some(timestamp);
                }
                Advance::Clock(timestamp)
            }
            Form::Stable { apart, asked, last, .. } => {
                let due = last
                    .as_ref()
                    .is_none_or(|(at, _)| step.drawn - at >= *apart);
                if asked.is_none() && due {
                    self.app.renderer.read_picture();
                    *asked = Some(step.drawn);
                }
                stepped(step.rate)
            }
            Form::Frames(_) | Form::Settled(_) | Form::Multiple { .. } => stepped(step.rate),
        };
        let drawn = self.framed(advance, size, Picture::Shown)?;
        if let Form::Stable { asked, .. } = &mut step.form
            && !drawn
            && *asked == Some(step.drawn)
        {
            *asked = None;
        }
        step.drawn += 1;
        let summary = self.frame_times();
        step.ticks += u64::from(summary.fixed_steps);
        let frame = summary.frame;
        let mut picture = None;
        let reached = match &mut step.form {
            Form::Ticks(_) => Some(None),
            Form::Frames(frames) => (step.drawn >= *frames).then_some(None),
            Form::Settled(limit) => {
                let (readiness, _) = self.progress();
                let settled =
                    readiness.settled() && !crate::host::in_flight(self, &readiness);
                (settled || step.drawn >= *limit).then_some(Some(settled))
            }
            Form::Stable {
                limit,
                tolerance,
                asked,
                last,
                ..
            } => {
                let mut stable = false;
                if let Some(at) = *asked
                    && let Some(read) = self.app.renderer.take_picture()
                {
                    *asked = None;
                    match read {
                        Ok((_, _, pixels)) => {
                            stable = last
                                .as_ref()
                                .is_some_and(|(_, before)| close(before, &pixels, *tolerance));
                            *last = Some((at, pixels));
                        }
                        Err(error) => picture = Some(refused(&error)),
                    }
                }
                (stable || picture.is_some() || step.drawn >= *limit).then_some(Some(stable))
            }
            Form::Multiple { of, target } => {
                let target = *target.get_or_insert((frame.saturating_sub(1) / *of + 1) * *of);
                (frame >= target).then_some(Some(true))
            }
            Form::Wall { ms, since } => {
                (since.is_some_and(|since| timestamp - since >= *ms)).then_some(None)
            }
        };
        let Some(reached) = reached else {
            return Ok((FramePhase::Playing, None));
        };
        self.development.stepping = false;
        self.hold();
        let mut value = step.reply(reached);
        if let Some(picture) = picture {
            value["picture"] = picture;
        }
        Ok((FramePhase::Playing, Some(Ok(value))))
    }

    fn shoot(
        &mut self,
        shoot: &mut Shoot,
        timestamp: f64,
        size: (u32, u32),
        held: bool,
    ) -> crate::Result<(FramePhase, Outcome)> {
        let phase = match shoot.frame {
            None => {
                self.app.renderer.read_picture();
                let advance = if held {
                    Advance::Stepped {
                        ticks: 0,
                        alpha: None,
                    }
                } else {
                    Advance::Clock(timestamp)
                };
                if !self.framed(advance, shoot.size, Picture::Shown)? {
                    let picture = unavailable("frame_not_acquired");
                    return Ok((FramePhase::Playing, Some(Ok(self.pictured(shoot, picture)))));
                }
                shoot.frame = Some(self.frame_times().frame);
                FramePhase::Playing
            }
            Some(_) => self.idle(timestamp, size, held)?,
        };
        let Some(read) = self.app.renderer.take_picture() else {
            let reason = match self.app.renderer.device_health() {
                DeviceHealth::Live | DeviceHealth::Recovered => return Ok((phase, None)),
                DeviceHealth::Failed => "device_failed",
                DeviceHealth::Lost | DeviceHealth::Recovering => "device_lost",
            };
            return Ok((phase, Some(Ok(self.pictured(shoot, unavailable(reason))))));
        };
        let picture = match read {
            Ok((width, height, rgba)) => {
                let frame = shoot.frame.unwrap_or_default();
                let file = shoot
                    .mark
                    .as_ref()
                    .map_or_else(|| format!("{frame}.png"), |mark| mark.file.clone());
                let picture = json!({
                    "path": file,
                    "width": width,
                    "height": height,
                    "sha256": hex(&rgba),
                });
                self.development.out.pictures.push(Shot {
                    run: shoot.mark.as_ref().map(|mark| mark.run.clone()),
                    file,
                    width,
                    height,
                    rgba,
                });
                picture
            }
            Err(error) => refused(&error),
        };
        Ok((phase, Some(Ok(self.pictured(shoot, picture)))))
    }

    fn pictured(&mut self, shoot: &Shoot, picture: Value) -> Value {
        match &shoot.mark {
            Some(mark) => {
                engine_dev::record::pictured(self.app.engine_mut().app.world_mut(), mark, picture)
            }
            None => picture,
        }
    }

    fn wait_reload(
        &mut self,
        reload: &mut Reload,
        timestamp: f64,
        size: (u32, u32),
        held: bool,
    ) -> crate::Result<(FramePhase, Outcome)> {
        if !reload.asked {
            reload.asked = true;
            self.development.out.reload = true;
        }
        Ok((self.idle(timestamp, size, held)?, None))
    }

    fn idle(&mut self, timestamp: f64, size: (u32, u32), held: bool) -> crate::Result<FramePhase> {
        if held {
            return Ok(FramePhase::Paused);
        }
        self.framed(Advance::Clock(timestamp), size, Picture::Shown)?;
        Ok(FramePhase::Playing)
    }
}
