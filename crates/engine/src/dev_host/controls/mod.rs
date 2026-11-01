mod input;
mod view;

use engine_core::{Error, ErrorCode};
use engine_dev::{BrowserControl, Control};
use engine_scene::SceneInstance;
use serde_json::Value;

use super::{DevHost, Done, InputBatch, Looped};
use crate::app_host::AppHost;

pub(super) fn run(
    host: &mut DevHost,
    app: &mut AppHost,
    scene: Option<&SceneInstance>,
    control: Control,
    size: (u32, u32),
) -> Result<Done, Error> {
    match control {
        Control::Operation(call) => {
            let result = call.apply(&mut app.engine, scene)?;
            then_render(app, size, result).map(Done::Result)
        }
        Control::Host(control) => host_control(host, app, control, size).map_err(in_args),
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
    control: BrowserControl,
    size: (u32, u32),
) -> Result<Done, Error> {
    let done = match control {
        BrowserControl::Present { alpha } => view::present(app, alpha, size),
        BrowserControl::Culling { enabled } => view::culling(app, enabled, size),
        BrowserControl::Camera {
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
        BrowserControl::Input {
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
        BrowserControl::Step { ticks } => input::step(host, app, ticks, size),
        BrowserControl::Render {} => view::render(app, size),
        BrowserControl::DebugPanic {} => {
            panic!("debug_panic")
        }
        BrowserControl::DebugFault { feature } => view::fault(app, &feature, size),
        control @ (BrowserControl::Benchmark { .. }
        | BrowserControl::Frames { .. }
        | BrowserControl::Pause { .. }
        | BrowserControl::Resume { .. }
        | BrowserControl::Probe { .. }
        | BrowserControl::Jobs { .. }
        | BrowserControl::DebugDeviceLoss { .. }
        | BrowserControl::Game { .. }
        | BrowserControl::Reload { .. }
        | BrowserControl::Environment(_)
        | BrowserControl::DebugMissingAsset { .. }
        | BrowserControl::Parent { .. }) => return Ok(Done::Loop(Looped::Control(control))),
    };
    done.map(Done::Result)
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
