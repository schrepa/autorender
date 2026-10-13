//! Controls of the view: presentation previews, culling, the camera,
//! plain renders and faults injected into the renderer.

use serde_json::Value;

use super::{rendered, then_render};
use crate::app_host::AppHost;
use engine_core::ErrorCode;
use engine_dev::DevError;
use engine_dev::results::{
    CameraResult, CullingResult, DebugFaultResult, PresentResult, RenderResult, result,
};

/// Presents fixed-tick poses at `alpha`, through the GPU path with `gpu`.
pub(super) fn present(
    app: &mut AppHost,
    alpha: f64,
    gpu: bool,
    (width, height): (u32, u32),
) -> Result<Value, DevError> {
    let alpha = engine_dev::validated_presentation_alpha(alpha)?;
    rendered(
        app.dev_present(alpha, width, height, gpu),
        "presentation prepared",
    )?;
    Ok(result::present(PresentResult {
        temporary: true,
        rendered: true,
    }))
}

/// Turns visibility culling on or off, then renders.
pub(super) fn culling(
    app: &mut AppHost,
    enabled: bool,
    size: (u32, u32),
) -> Result<Value, DevError> {
    app.engine.set_frustum_culling(enabled);
    then_render(
        app,
        size,
        "culling changed",
        result::culling(CullingResult {
            temporary: true,
            culling_enabled: enabled,
        }),
    )
}

/// Sets the camera, then renders.
pub(super) fn camera(
    app: &mut AppHost,
    (position, target): ([f32; 3], [f32; 3]),
    vertical_fov_radians: f32,
    (near, far): (f32, f32),
    size: (u32, u32),
) -> Result<Value, DevError> {
    let camera = engine_dev::validated_camera(position, target, vertical_fov_radians, near, far)?;
    let world = app.engine.app.world_mut();
    if let Some(mut controller) = world.get_resource_mut::<crate::camera::FlyCameraController>() {
        controller.align_to(&camera);
    }
    world.insert_resource(camera);
    then_render(
        app,
        size,
        "camera changed",
        result::camera(CameraResult {
            temporary: true,
            camera_changed: true,
        }),
    )
}

/// Renders current state without advancing gameplay.
pub(super) fn render(app: &mut AppHost, (width, height): (u32, u32)) -> Result<Value, DevError> {
    app.dev_render(width, height)
        .map_err(|error| DevError::new(ErrorCode::RenderFailed, "$", error.to_string()))?;
    Ok(result::render(RenderResult { rendered: true }))
}

/// Injects a fault into the renderer's lazily created feature `feature`,
/// then renders.
pub(super) fn fault(app: &mut AppHost, feature: &str, size: (u32, u32)) -> Result<Value, DevError> {
    app.renderer
        .inject_fault(feature)
        .map_err(|error| DevError::new(ErrorCode::InvalidValue, "$.feature", error.to_string()))?;
    then_render(
        app,
        size,
        "the fault",
        result::debug_fault(DebugFaultResult { injected: true }),
    )
}
