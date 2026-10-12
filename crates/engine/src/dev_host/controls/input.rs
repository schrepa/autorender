//! Input and stepping: a temporary batch of input, applied by the next
//! explicit step of fixed ticks.

use serde_json::{Value, json};

use super::super::{DevHost, InputBatch};
use crate::app_host::AppHost;
use engine_core::{ActionInput, DevStepError, ErrorCode, InputState};
use engine_dev::{DevError, bounds};
use engine_settings::Setting;

/// Replaces the batch the next step applies.
pub(super) fn batch(host: &mut DevHost, input: InputBatch) -> Result<Value, DevError> {
    for (value, field, rule) in [
        (input.forward, "forward", bounds::AXIS),
        (input.right, "right", bounds::AXIS),
        (input.up, "up", bounds::AXIS),
        (input.yaw, "yaw", bounds::TURN),
        (input.pitch, "pitch", bounds::TURN),
    ] {
        if let Err(invalid) = value.check(field, rule) {
            return Err(DevError::new(
                ErrorCode::InvalidInput,
                format!("$.{field}"),
                invalid.rule,
            ));
        }
    }
    host.input = input;
    Ok(json!({"temporary":true,"applies_on":"next_step"}))
}

/// Steps `ticks` fixed ticks with the batch, then presents.
pub(super) fn step(
    host: &mut DevHost,
    app: &mut AppHost,
    ticks: u32,
    (width, height): (u32, u32),
) -> Result<Value, DevError> {
    DevStepError::validate(ticks)
        .map_err(|error| DevError::new(ErrorCode::InvalidStep, "$.ticks", error.to_string()))?;
    app.engine
        .app
        .world_mut()
        .resource_mut::<ActionInput>()
        .ingest(host.input.held, host.input.pressed, host.input.released);
    let input = InputState {
        move_forward: host.input.forward,
        move_right: host.input.right,
        move_up: host.input.up,
        look_yaw: host.input.yaw,
        look_pitch: host.input.pitch,
    };
    // These edges belong to this request even if subsequent GPU submission fails.
    host.input.pressed = 0;
    host.input.released = 0;
    host.input.yaw = 0.0;
    host.input.pitch = 0.0;
    app.dev_step(ticks, width, height, input).map_err(|error| {
        DevError::new(
            ErrorCode::RenderFailed,
            "$",
            format!("simulation may have advanced: {error}"),
        )
    })?;
    Ok(json!({"ticks":ticks,"temporary":true}))
}
