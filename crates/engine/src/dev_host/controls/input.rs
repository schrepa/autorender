use serde_json::Value;

use super::super::{DevHost, InputBatch};
use crate::app_host::AppHost;
use engine_core::{ActionInput, Error, ErrorCode, InputState};
use engine_dev::bounds;
use engine_dev::results::{InputResult, StepResult, result};
use engine_settings::Setting;

pub(super) fn batch(host: &mut DevHost, input: InputBatch) -> Result<Value, Error> {
    for (value, field, rule) in [
        (input.forward, "forward", bounds::AXIS),
        (input.right, "right", bounds::AXIS),
        (input.up, "up", bounds::AXIS),
        (input.yaw, "yaw", bounds::TURN),
        (input.pitch, "pitch", bounds::TURN),
    ] {
        if let Err(invalid) = value.check(field, rule) {
            return Err(invalid.error(ErrorCode::InvalidInput, "$"));
        }
    }
    host.input = input;
    Ok(result::input(InputResult {
        temporary: true,
        applies_on: "next_step",
    }))
}

pub(super) fn step(
    host: &mut DevHost,
    app: &mut AppHost,
    ticks: u32,
    (width, height): (u32, u32),
) -> Result<Value, Error> {
    engine_core::check_dev_ticks(ticks)?;
    let held = (host.input.held, host.input.pressed, host.input.released);
    let _ = app
        .engine
        .app
        .world_mut()
        .modify_resource(|actions: &mut ActionInput| actions.ingest(held.0, held.1, held.2));
    let input = InputState {
        move_forward: host.input.forward,
        move_right: host.input.right,
        move_up: host.input.up,
        look_yaw: host.input.yaw,
        look_pitch: host.input.pitch,
    };
    host.input.pressed = 0;
    host.input.released = 0;
    host.input.yaw = 0.0;
    host.input.pitch = 0.0;
    app.dev_step(ticks, width, height, input)
        .map_err(|error| Error::new(ErrorCode::RenderFailed, "$").got(error))?;
    Ok(result::step(StepResult {
        ticks,
        temporary: true,
    }))
}
