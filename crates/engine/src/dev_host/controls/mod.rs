mod environment;
mod input;
mod view;

use engine_core::{Error, ErrorCode};
use engine_dev::{Control, HostControl};
use engine_scene::SceneInstance;
use serde_json::{Value, json};

use super::{DevHost, InputBatch};
use crate::app_host::AppHost;

pub(super) fn run(
    host: &mut DevHost,
    app: &mut AppHost,
    scene: Option<&SceneInstance>,
    (args, control): (Value, Control),
    size: (u32, u32),
) -> Result<Value, Error> {
    match control {
        Control::Operation(call) => {
            let result = call.apply(&mut app.engine, scene)?;
            then_render(app, size, result)
        }
        Control::Host(control) => host_control(host, app, (args, control), size).map_err(in_args),
    }
}

fn in_args(error: Error) -> Error {
    if error.path.starts_with("$.") || error.path.starts_with("$[") {
        error.within("$.args")
    } else {
        error
    }
}

fn host_control(
    host: &mut DevHost,
    app: &mut AppHost,
    (args, control): (Value, HostControl),
    size: (u32, u32),
) -> Result<Value, Error> {
    match control {
        HostControl::Present { alpha } => view::present(app, alpha, size),
        HostControl::Culling { enabled } => view::culling(app, enabled, size),
        HostControl::Camera {
            position,
            target,
            vertical_fov_radians,
            near,
            far,
        } => view::camera(
            app,
            (position, target),
            vertical_fov_radians,
            (near, far),
            size,
        ),
        HostControl::Environment(control) => environment::set(app, args, control, size),
        HostControl::Input {
            forward,
            right,
            up,
            yaw,
            pitch,
            held,
            pressed,
            released,
        } => input::batch(
            host,
            InputBatch {
                forward,
                right,
                up,
                yaw,
                pitch,
                held,
                pressed,
                released,
            },
        ),
        HostControl::Step { ticks } => input::step(host, app, ticks, size),
        HostControl::Render => view::render(app, size),
        HostControl::DebugPanic => {
            panic!("debug_panic")
        }
        HostControl::DebugFault { feature } => view::fault(app, &feature, size),
        HostControl::Loop(control) => Ok(json!({ "control": control })),
    }
}

fn then_render(
    app: &mut AppHost,
    (width, height): (u32, u32),
    result: Value,
) -> Result<Value, Error> {
    rendered(app.dev_render(width, height))?;
    Ok(result)
}

fn rendered(outcome: crate::Result<()>) -> Result<(), Error> {
    outcome.map_err(|error| Error::new(ErrorCode::RenderFailed, "$").got(error))
}
