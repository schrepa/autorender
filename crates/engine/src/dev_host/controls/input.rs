use serde_json::Value;

use super::super::{DevHost, InputBatch};
use crate::app_host::AppHost;
use crate::host::{FrameInput, KEYS};
use engine_core::{ActionInput, AxisInput, Error, ErrorCode, MenuInput, MoveInput, PointerInput};
use engine_dev::MenuPress;
use engine_dev::bounds;
use engine_dev::results::{InputResult, StepResult, result};
use engine_settings::Setting;
use glam::Vec2;

pub(super) fn batch(host: &mut DevHost, input: InputBatch) -> Result<Value, Error> {
    let [pointer_x, pointer_y] = input.pointer_pixels.unwrap_or_default();
    let [delta_x, delta_y] = input.pointer_delta_pixels;
    let fields = [
        (input.forward, "forward", bounds::AXIS),
        (input.right, "right", bounds::AXIS),
        (input.up, "up", bounds::AXIS),
        (input.look_right_pixels, "look_right_pixels", bounds::TURN),
        (input.look_down_pixels, "look_down_pixels", bounds::TURN),
        (pointer_x, "pointer_pixels", bounds::COORDINATE),
        (pointer_y, "pointer_pixels", bounds::COORDINATE),
        (delta_x, "pointer_delta_pixels", bounds::TURN),
        (delta_y, "pointer_delta_pixels", bounds::TURN),
        (input.wheel_up_notches, "wheel_up_notches", bounds::TURN),
        (
            input.wheel_right_notches,
            "wheel_right_notches",
            bounds::TURN,
        ),
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
    for (at, key) in input.keys.iter().enumerate() {
        if !KEYS.contains(&key.as_str()) {
            return Err(Error::new(ErrorCode::InvalidInput, format!("$.keys[{at}]"))
                .got(key.as_str())
                .variants(KEYS.iter().copied()));
        }
    }
    if let Some(gatherer) = &mut host.gatherer {
        for key in host
            .input
            .keys
            .iter()
            .filter(|key| !input.keys.contains(key))
        {
            gatherer.key(key, false, true);
        }
        for key in input
            .keys
            .iter()
            .filter(|key| !host.input.keys.contains(key))
        {
            gatherer.key(key, true, true);
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
    let input = host.take_input();
    let _ = app
        .engine
        .app
        .world_mut()
        .modify_resource(|actions: &mut ActionInput| {
            actions.ingest(&input.held, &input.pressed, &input.released);
        });
    app.set_gathered(&input);
    app.dev_step(ticks, width, height, input.state)
        .map_err(|error| Error::new(ErrorCode::RenderFailed, "$").got(error))?;
    Ok(result::step(StepResult {
        ticks,
        temporary: true,
    }))
}

impl DevHost {
    pub(crate) fn take_input(&mut self) -> FrameInput {
        let gathered = self
            .gatherer
            .as_mut()
            .map(|gatherer| gatherer.take(0.0))
            .unwrap_or_default();
        let batch = &mut self.input;
        let mut axes = *gathered.axes.values();
        for (axis, &value) in axes.iter_mut().zip(&batch.axes) {
            *axis += value;
        }
        let menu = |press: MenuPress| batch.menu.contains(&press);
        let input = FrameInput {
            state: MoveInput {
                move_forward: batch.forward + gathered.state.move_forward,
                move_right: batch.right + gathered.state.move_right,
                move_up: batch.up + gathered.state.move_up,
                look_right_pixels: batch.look_right_pixels,
                look_down_pixels: batch.look_down_pixels,
            },
            held: gathered.held,
            pressed: gathered.pressed,
            released: gathered.released,
            pointer: PointerInput {
                position_pixels: batch.pointer_pixels.map(Vec2::from),
                delta_pixels: Vec2::from(batch.pointer_delta_pixels),
                wheel_up_notches: batch.wheel_up_notches,
                wheel_right_notches: batch.wheel_right_notches,
                inside: batch.pointer_pixels.is_some(),
                held: gathered.pointer.held,
                pressed: gathered.pointer.pressed,
                released: gathered.pointer.released,
            },
            axes: AxisInput::of(axes),
            menu: MenuInput {
                up: gathered.menu.up || menu(MenuPress::Up),
                down: gathered.menu.down || menu(MenuPress::Down),
                left: gathered.menu.left || menu(MenuPress::Left),
                right: gathered.menu.right || menu(MenuPress::Right),
                accept: gathered.menu.accept || menu(MenuPress::Accept),
                back: gathered.menu.back || menu(MenuPress::Back),
                wanted: false,
            },
        };
        batch.look_right_pixels = 0.0;
        batch.look_down_pixels = 0.0;
        batch.pointer_delta_pixels = [0.0; 2];
        batch.wheel_up_notches = 0.0;
        batch.wheel_right_notches = 0.0;
        batch.menu.clear();
        input
    }
}
