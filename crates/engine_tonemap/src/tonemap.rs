use bevy_ecs::prelude::*;
use engine_places::tonemap::Display;
use glam::Vec3;

use crate::light::{Grading, GradingValues};

pub(crate) fn publish_display(grading: Res<Grading>, mut display: ResMut<Display>) {
    display.set_if_neq(display_of(&grading.values()));
}

fn contrast_weight(contrast: f32) -> f32 {
    engine_math::clamp(2.0 * (contrast - 1.0), -2.0, 1.0)
}

fn unit_luminance(tint: Vec3) -> [f32; 4] {
    let luminance = tint.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    if luminance > 1.0e-6 {
        (tint / luminance).extend(0.0).to_array()
    } else {
        [1.0, 1.0, 1.0, 0.0]
    }
}

fn display_of(grading: &GradingValues) -> Display {
    Display {
        grade: [
            contrast_weight(grading.contrast),
            grading.vibrance,
            grading.saturation,
            grading.exposure,
        ],
        shadow_tint: unit_luminance(grading.shadow_tint),
        highlight_tint: unit_luminance(grading.highlight_tint),
        exposure: [grading.exposure, 0.0, 0.0, 0.0],
    }
}
