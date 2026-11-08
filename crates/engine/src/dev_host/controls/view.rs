use serde_json::Value;

use super::{rendered, then_render};
use crate::app_host::AppHost;
use engine_core::{Error, ErrorCode, Expected};
use engine_dev::results::{CameraResult, PresentResult, RenderResult, result};

pub(super) fn present(
    app: &mut AppHost,
    alpha: f64,
    (width, height): (u32, u32),
) -> Result<Value, Error> {
    let alpha = engine_dev::validated_presentation_alpha(alpha)?;
    rendered(app.dev_present(alpha, width, height))?;
    Ok(result::present(PresentResult {
        temporary: true,
        rendered: true,
    }))
}

pub(super) fn camera(
    app: &mut AppHost,
    (position, target): ([f32; 3], [f32; 3]),
    vertical_fov_radians: f32,
    (near, far): (f32, f32),
    size: (u32, u32),
) -> Result<Value, Error> {
    let (at, lens) =
        engine_dev::validated_camera(position, target, vertical_fov_radians, near, far)?;
    let world = app.engine.app.world_mut();
    let camera = engine_core::main_camera(world)
        .ok_or_else(|| Error::new(ErrorCode::InvalidCamera, "$").expected(Expected::Present))?;
    engine_core::place_camera(world, camera, at, lens);
    then_render(
        app,
        size,
        result::camera(CameraResult {
            temporary: true,
            camera_changed: true,
        }),
    )
}

pub(super) fn render(app: &mut AppHost, (width, height): (u32, u32)) -> Result<Value, Error> {
    rendered(app.dev_render(width, height))?;
    Ok(result::render(RenderResult { rendered: true }))
}
