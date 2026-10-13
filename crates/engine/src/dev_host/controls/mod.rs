//! The controls the development host runs: one exhaustive dispatch, a
//! feature's control applied by its development file and the host's own in
//! the modules here, each rendering after it changes something.

mod environment;
mod input;
mod view;

use engine_core::ErrorCode;
use engine_dev::{Control, DevError, HostControl};
use engine_scene::SceneInstance;
use serde_json::{Value, json};

use super::{DevHost, InputBatch};
use crate::app_host::AppHost;

/// Runs `control`, sent as `source`, on a `size` view: its result.
pub(super) fn run(
    host: &mut DevHost,
    app: &mut AppHost,
    scene: Option<&SceneInstance>,
    (source, control): (&str, Control),
    size: (u32, u32),
) -> Result<Value, DevError> {
    match control {
        // Every feature's control: its development file applies it.
        Control::Feature(control) => {
            let result = control.apply(&mut app.engine, scene)?;
            then_render(app, size, control.changed(), result)
        }
        Control::Host(HostControl::Present { alpha, gpu }) => view::present(app, alpha, gpu, size),
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
        // The host's loop runs these: it receives them validated, with
        // their defaults.
        Control::Host(HostControl::Loop(control)) => Ok(json!({ "control": control })),
    }
}

/// Renders the `size` view after `what` changed: `result`, or the render's
/// error.
fn then_render(
    app: &mut AppHost,
    (width, height): (u32, u32),
    what: &str,
    result: Value,
) -> Result<Value, DevError> {
    rendered(app.dev_render(width, height), what)?;
    Ok(result)
}

/// What rendering after `what` came to: a development error when it failed.
fn rendered(outcome: crate::Result<()>, what: &str) -> Result<(), DevError> {
    outcome.map_err(|error| {
        DevError::new(
            ErrorCode::RenderFailed,
            "$",
            format!("{what} before render failed: {error}"),
        )
    })
}
