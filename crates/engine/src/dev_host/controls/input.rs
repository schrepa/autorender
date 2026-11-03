use serde_json::Value;

use super::super::{DevHost, InputBatch};
use crate::app_host::AppHost;
use crate::host::FrameInput;
use engine_core::{
    ActionInput, AxisId, AxisInput, Error, ErrorCode, MenuInput, MoveInput, PointerInput,
};
use engine_dev::MenuPress;
use engine_dev::bounds;
use engine_dev::results::{InputResult, StepResult, result};
use engine_settings::Setting;
use glam::Vec2;

pub(super) fn batch(host: &mut DevHost, input: InputBatch) -> Result<Value, Error> {
    let [pointer_x, pointer_y] = input.pointer.unwrap_or_default();
    let fields = [
        (input.forward, "forward", bounds::AXIS),
        (input.right, "right", bounds::AXIS),
        (input.up, "up", bounds::AXIS),
        (input.yaw, "yaw", bounds::TURN),
        (input.pitch, "pitch", bounds::TURN),
        (pointer_x, "pointer", bounds::COORDINATE),
        (pointer_y, "pointer", bounds::COORDINATE),
        (input.pointer_delta[0], "pointer_delta", bounds::TURN),
        (input.pointer_delta[1], "pointer_delta", bounds::TURN),
        (input.wheel[0], "wheel", bounds::TURN),
        (input.wheel[1], "wheel", bounds::TURN),
    ];
    let axes = input.axes.iter().map(|&axis| (axis, "axes", bounds::AXIS));
    for (value, field, rule) in fields.into_iter().chain(axes) {
        if let Err(invalid) = value.check(field, rule) {
            return Err(invalid.error(ErrorCode::InvalidInput, "$"));
        }
    }
    if input.axes.len() > bounds::MAX_AXES {
        return Err(Error::new(ErrorCode::InvalidInput, "$.axes")
            .at_most(bounds::MAX_AXES as u64)
            .got(input.axes.len()));
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
    let input = MoveInput {
        move_forward: host.input.forward,
        move_right: host.input.right,
        move_up: host.input.up,
        look_yaw: host.input.yaw,
        look_pitch: host.input.pitch,
    };
    let mut axes = [0.0; AxisId::COUNT as usize];
    for (axis, &value) in axes.iter_mut().zip(&host.input.axes) {
        *axis = value;
    }
    let menu = |press: MenuPress| host.input.menu.contains(&press);
    app.set_gathered(FrameInput {
        pointer: PointerInput {
            position: host.input.pointer.map(Vec2::from),
            delta: Vec2::from(host.input.pointer_delta),
            wheel: Vec2::from(host.input.wheel),
            inside: host.input.pointer.is_some(),
            held: host.input.pointer_held,
            pressed: host.input.pointer_pressed,
            released: host.input.pointer_released,
        },
        axes: AxisInput::of(axes),
        menu: MenuInput {
            up: menu(MenuPress::Up),
            down: menu(MenuPress::Down),
            left: menu(MenuPress::Left),
            right: menu(MenuPress::Right),
            accept: menu(MenuPress::Accept),
            back: menu(MenuPress::Back),
            wanted: false,
        },
        ..FrameInput::default()
    });
    host.input.pressed = 0;
    host.input.released = 0;
    host.input.yaw = 0.0;
    host.input.pitch = 0.0;
    host.input.pointer_delta = [0.0; 2];
    host.input.pointer_pressed = false;
    host.input.pointer_released = false;
    host.input.wheel = [0.0; 2];
    host.input.menu.clear();
    app.dev_step(ticks, width, height, input)
        .map_err(|error| Error::new(ErrorCode::RenderFailed, "$").got(error))?;
    Ok(result::step(StepResult {
        ticks,
        temporary: true,
    }))
}
