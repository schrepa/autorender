use std::collections::BTreeMap;

use serde_json::Value;

use super::super::{DevHost, InputBatch};
use crate::app_host::AppHost;
use engine_core::{Driven, Error, ErrorCode, Gathered, Input, MouseButton, Player, VerbShape};
use engine_dev::bounds;
use engine_dev::results::{InputResult, StepResult, result};
use engine_dev::{PointerControl, TouchControl};
use engine_settings::{Expected, InvalidSetting, Setting};
use glam::Vec2;

pub(super) struct Asked {
    pub(super) verbs: BTreeMap<String, Value>,
    pub(super) player: u8,
    pub(super) keys: Vec<String>,
    pub(super) pointer: PointerControl,
    pub(super) touches: Vec<TouchControl>,
}

pub(super) fn batch(host: &mut DevHost, app: &AppHost, asked: Asked) -> Result<Value, Error> {
    let input = app.engine.app.world().resource::<Input>();
    let player = player_of(input, asked.player)?;
    let driven = verbs_of(input, player, &asked.verbs)?;
    checked(&asked)?;
    let agent = &mut host.agent;
    let held: Vec<String> = agent.keys_held().map(str::to_owned).collect();
    for key in held.iter().filter(|key| !asked.keys.contains(key)) {
        agent.key(key, false);
    }
    for key in &asked.keys {
        agent.key(key, true);
    }
    let pressed: Vec<MouseButton> = agent.mouse_held().collect();
    for &button in pressed.iter().filter(|button| !asked.pointer.buttons.contains(button)) {
        agent.mouse_button(button, false);
    }
    for &button in &asked.pointer.buttons {
        agent.mouse_button(button, true);
    }
    match asked.pointer.position_pixels {
        Some([x, y]) => agent.pointer(x, y),
        None => agent.pointer_left(),
    }
    let [dx, dy] = asked.pointer.motion_pixels;
    agent.motion(dx, dy);
    let [right, up] = asked.pointer.wheel_notches;
    agent.wheel(right, up);
    for touch in &asked.touches {
        let [x, y] = touch.position_pixels;
        agent.touch(touch.id, x, y, touch.phase);
    }
    let mut batch = InputBatch::default();
    for (shape, drive) in driven {
        match shape {
            VerbShape::Delta | VerbShape::Delta2 => batch.once.push(drive),
            VerbShape::Button | VerbShape::Value | VerbShape::Value2 => batch.held.push(drive),
        }
    }
    host.input = batch;
    Ok(result::input(InputResult {
        temporary: true,
        applies_on: "next_step",
    }))
}

fn player_of(input: &Input, asked: u8) -> Result<Player, Error> {
    let players = input.players();
    if (1..=players).contains(&asked) {
        return Ok(Player(asked));
    }
    Err(Error::new(ErrorCode::InvalidInput, "$.player")
        .range(1.0, f64::from(players))
        .got(asked))
}

fn verbs_of(
    input: &Input,
    player: Player,
    verbs: &BTreeMap<String, Value>,
) -> Result<Vec<(VerbShape, Driven)>, Error> {
    let mut driven = Vec::with_capacity(verbs.len());
    for (name, value) in verbs {
        let path = format!("$.verbs.{name}");
        let Some(shape) = input.find(name).and_then(|verb| input.shape(verb)) else {
            return Err(Error::new(ErrorCode::UnknownVariant, path)
                .got(name.as_str())
                .variants(input.names()));
        };
        let Some(level) = level_of(shape, value) else {
            return Err(Error::new(ErrorCode::InvalidInput, path)
                .expected(expected_of(shape))
                .got(value));
        };
        for part in level.to_array() {
            part.check("value", bounds::TURN).map_err(|invalid| Error {
                path: path.clone(),
                ..invalid.error(ErrorCode::InvalidInput, "$")
            })?;
        }
        driven.push((
            shape,
            Driven {
                player,
                verb: name.clone(),
                value: level,
            },
        ));
    }
    Ok(driven)
}

fn level_of(shape: VerbShape, value: &Value) -> Option<Vec2> {
    let number = |value: &Value| value.as_f64().map(|number| number as f32);
    match (shape, value) {
        (VerbShape::Button, Value::Bool(down)) => Some(Vec2::new(f32::from(u8::from(*down)), 0.0)),
        (VerbShape::Button | VerbShape::Value | VerbShape::Delta, value) => {
            Some(Vec2::new(number(value)?, 0.0))
        }
        (VerbShape::Value2 | VerbShape::Delta2, Value::Array(parts)) => match parts.as_slice() {
            [x, y] => Some(Vec2::new(number(x)?, number(y)?)),
            _ => None,
        },
        (VerbShape::Value2 | VerbShape::Delta2, _) => None,
    }
}

fn expected_of(shape: VerbShape) -> Expected {
    match shape {
        VerbShape::Button => Expected::Boolean,
        VerbShape::Value | VerbShape::Delta => Expected::Number,
        VerbShape::Value2 | VerbShape::Delta2 => Expected::Array,
    }
}

fn checked(asked: &Asked) -> Result<(), Error> {
    let refused = |invalid: InvalidSetting| invalid.error(ErrorCode::InvalidInput, "$");
    for (at, key) in asked.keys.iter().enumerate() {
        if !engine_core::is_id(key) {
            return Err(Error::new(ErrorCode::InvalidInput, format!("$.keys[{at}]"))
                .expected(Expected::Id)
                .got(key.as_str()));
        }
    }
    let pointer = &asked.pointer;
    pointer
        .position_pixels
        .check("position_pixels", bounds::COORDINATE)
        .and_then(|()| pointer.motion_pixels.check("motion_pixels", bounds::TURN))
        .and_then(|()| pointer.wheel_notches.check("wheel_notches", bounds::TURN))
        .map_err(|invalid| refused(invalid.within("pointer")))?;
    for (at, touch) in asked.touches.iter().enumerate() {
        touch
            .position_pixels
            .check("position_pixels", bounds::COORDINATE)
            .map_err(|invalid| refused(invalid.within("touches").indexed("touches", at)))?;
    }
    Ok(())
}

pub(super) fn step(
    host: &mut DevHost,
    app: &mut AppHost,
    ticks: u32,
    (width, height): (u32, u32),
) -> Result<Value, Error> {
    engine_core::check_dev_ticks(ticks)?;
    let input = host.take_input();
    app.accept(input);
    app.dev_step(ticks, width, height)
        .map_err(|error| Error::new(ErrorCode::RenderFailed, "$").got(error))?;
    Ok(result::step(StepResult {
        ticks,
        temporary: true,
    }))
}

impl InputBatch {
    pub(crate) fn drop_edges(&mut self) {
        self.once.clear();
    }
}

impl DevHost {
    pub(crate) fn take_input(&mut self) -> Gathered {
        let mut driven = self.input.held.clone();
        driven.append(&mut self.input.once);
        Gathered {
            devices: self.agent.take(),
            driven,
        }
    }
}
