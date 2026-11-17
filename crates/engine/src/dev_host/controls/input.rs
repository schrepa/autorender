use serde_json::Value;

use super::super::DevHost;
use crate::app_host::AppHost;
use engine_core::{Devices, Error, ErrorCode, MouseButton};
use engine_dev::bounds;
use engine_dev::results::{InputResult, StepResult, result};
use engine_dev::{PointerControl, TouchControl};
use engine_settings::{Expected, InvalidSetting, Setting};

pub(super) struct Asked {
    pub(super) keys: Vec<String>,
    pub(super) pointer: PointerControl,
    pub(super) touches: Vec<TouchControl>,
}

pub(super) fn batch(host: &mut DevHost, asked: Asked) -> Result<Value, Error> {
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
    for &button in pressed
        .iter()
        .filter(|button| !asked.pointer.buttons.contains(button))
    {
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
        let [dx, dy] = touch.motion_pixels;
        agent.touch_motion(touch.id, dx, dy);
    }
    Ok(result::input(InputResult {
        temporary: true,
        applies_on: "next_step",
    }))
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
            .and_then(|()| touch.motion_pixels.check("motion_pixels", bounds::TURN))
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

impl DevHost {
    pub(crate) fn take_input(&mut self) -> Devices {
        self.agent.take()
    }
}
