use serde_json::Value;

use super::{rendered, then_render};
use crate::app_host::AppHost;
use engine_core::{Error, ErrorCode};
use engine_dev::results::{
    CameraResult, CullingResult, DebugFaultResult, PresentResult, RenderResult, result,
};

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

pub(super) fn culling(app: &mut AppHost, enabled: bool, size: (u32, u32)) -> Result<Value, Error> {
    app.engine.set_frustum_culling(enabled);
    then_render(
        app,
        size,
        result::culling(CullingResult {
            temporary: true,
            culling_enabled: enabled,
        }),
    )
}

pub(super) fn camera(
    app: &mut AppHost,
    (position, target): ([f32; 3], [f32; 3]),
    vertical_fov_radians: f32,
    (near, far): (f32, f32),
    size: (u32, u32),
) -> Result<Value, Error> {
    let camera = engine_dev::validated_camera(position, target, vertical_fov_radians, near, far)?;
    app.engine.app.world_mut().insert_resource(camera);
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

pub(super) fn fault(app: &mut AppHost, feature: &str, size: (u32, u32)) -> Result<Value, Error> {
    app.renderer.inject_fault(feature).map_err(|error| Error {
        code: ErrorCode::InvalidValue,
        path: "$.feature".to_owned(),
        detail: error.detail,
    })?;
    then_render(
        app,
        size,
        result::debug_fault(DebugFaultResult { injected: true }),
    )
}
