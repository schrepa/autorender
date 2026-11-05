use bevy_ecs::prelude::*;
use engine_core::settings;
use engine_places::tonemap::Display;
use glam::Vec3;

settings! {
    #[section("grading")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneGrading")]
    pub struct Grading {
        pub exposure: f32 = 1.0 => at_least(0.0),
        pub saturation: f32 = 1.05 => at_least(0.0),
        pub contrast: f32 = 1.1 => at_least(0.0),
        pub vibrance: f32 = 0.15 => at_least(0.0),
        pub shadow_tint: [f32; 3] = [0.95, 0.98, 1.06] => at_least(0.0),
        pub highlight_tint: [f32; 3] = [1.04, 1.0, 0.96] => at_least(0.0),
    }
}

pub(crate) fn publish_display(grading: Res<Grading>, mut display: ResMut<Display>) {
    display.set_if_neq(display_of(&grading));
}

fn contrast_weight(contrast: f32) -> f32 {
    engine_math::clamp(2.0 * (contrast - 1.0), -2.0, 1.0)
}

fn unit_luminance(tint: [f32; 3]) -> [f32; 4] {
    let tint = Vec3::from_array(tint);
    let luminance = tint.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    if luminance > 1.0e-6 {
        (tint / luminance).extend(0.0).to_array()
    } else {
        [1.0, 1.0, 1.0, 0.0]
    }
}

fn display_of(grading: &Grading) -> Display {
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
