mod environment;
mod input;
mod view;

use engine_core::ErrorCode;
use engine_dev::{Control, DevError, HostControl};
use engine_scene::SceneInstance;
use serde_json::{Value, json};

use super::{DevHost, InputBatch};
use crate::app_host::AppHost;

pub(super) fn run(
    host: &mut DevHost,
    app: &mut AppHost,
    scene: Option<&SceneInstance>,
    (source, control): (&str, Control),
    size: (u32, u32),
) -> Result<Value, DevError> {
    match control {
        Control::Feature(control) => {
            let result = control.apply(&mut app.engine, scene)?;
            then_render(app, size, control.changed(), result)
        }
        Control::Host(HostControl::Present { alpha }) => view::present(app, alpha, size),
        Control::Host(HostControl::Culling { enabled }) => view::culling(app, enabled, size),
        Control::Host(HostControl::Camera {
            position,
            target,
            vertical_fov_radians,
            near,
            far,
        }) => view::camera(
            app,
            (position, target),
            vertical_fov_radians,
            (near, far),
            size,
        ),
        Control::Host(HostControl::Environment(control)) => {
            environment::set(app, source, control, size)
        }
        Control::Host(HostControl::Input {
            forward,
            right,
            up,
            yaw,
            pitch,
            held,
            pressed,
            released,
        }) => input::batch(
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
        Control::Host(HostControl::Step { ticks }) => input::step(host, app, ticks, size),
        Control::Host(HostControl::Render) => view::render(app, size),
        Control::Host(HostControl::DebugPanic) => {
            panic!("debug_panic: a development session asked the engine to panic")
        }
        Control::Host(HostControl::DebugFault { feature }) => view::fault(app, &feature, size),
        Control::Host(HostControl::Loop(control)) => Ok(json!({ "control": control })),
    }
}

fn then_render(
    app: &mut AppHost,
    (width, height): (u32, u32),
    what: &str,
    result: Value,
) -> Result<Value, DevError> {
    rendered(app.dev_render(width, height), what)?;
    Ok(result)
}

fn rendered(outcome: crate::Result<()>, what: &str) -> Result<(), DevError> {
    outcome.map_err(|error| {
        DevError::new(
            ErrorCode::RenderFailed,
            "$",
            format!("{what} before render failed: {error}"),
        )
    })
}
