use engine_core::math::{clamp, max, min, sin_cos};
use glam::Quat;

pub fn approach(value: f32, goal: f32, step: f32) -> f32 {
    if value < goal {
        min(value + step, goal)
    } else {
        max(value - step, goal)
    }
}

pub fn eased(share: f32) -> f32 {
    let share = clamp(share, 0.0, 1.0);
    share * share * (3.0 - 2.0 * share)
}

pub fn about_x(angle: f32) -> Quat {
    let (sin, cos) = sin_cos(angle * 0.5);
    Quat::from_xyzw(sin, 0.0, 0.0, cos)
}

pub fn about_y(angle: f32) -> Quat {
    let (sin, cos) = sin_cos(angle * 0.5);
    Quat::from_xyzw(0.0, sin, 0.0, cos)
}
