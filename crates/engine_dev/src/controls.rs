use crate::DevError;
use engine_core::{Camera, glam::Vec3};

/// Validates a temporary render fraction without mutating clocks or gameplay.
/// Fractions include both fixed-tick endpoints; no clamping hides bad requests.
pub fn validated_presentation_alpha(alpha: f64) -> Result<f32, DevError> {
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(DevError::new(
            "invalid_presentation_alpha",
            "$.alpha",
            "presentation alpha must be finite and in [0, 1]",
        ));
    }
    Ok(alpha as f32)
}

/// Validates a temporary camera without invoking fallback behavior or mutating
/// gameplay. Shared by browser controls and native contract checks.
pub fn validated_camera(
    position: [f32; 3],
    target: [f32; 3],
    vertical_fov_radians: f32,
    near: f32,
    far: f32,
) -> Result<Camera, DevError> {
    let position = Vec3::from_array(position);
    let target = Vec3::from_array(target);
    if !position.is_finite()
        || !target.is_finite()
        || position.abs().max_element() > 1_000_000.0
        || target.abs().max_element() > 1_000_000.0
        || (target - position).try_normalize().is_none()
        || !vertical_fov_radians.is_finite()
        || !(0.01..3.13).contains(&vertical_fov_radians)
        || !near.is_finite()
        || !far.is_finite()
        || near <= 0.0
        || far <= near
        || far > 10_000_000.0
    {
        return Err(DevError::new(
            "invalid_camera",
            "$",
            "camera requires finite position/target within +/-1000000, distinct points, FOV in [0.01,3.13), and 0 < near < far <= 10000000",
        ));
    }
    let camera = Camera {
        position,
        target,
        vertical_fov_radians,
        near,
        far,
    };
    if !camera.view_projection(1.0).is_finite() {
        return Err(DevError::new(
            "invalid_camera",
            "$",
            "camera must produce a finite projection",
        ));
    }
    Ok(camera)
}
