use bevy_ecs::prelude::*;
use engine_core::{InvalidSetting, Rule, settings};
use engine_places_rendering::tonemap::Display;
use glam::Vec3;

const GREY: f32 = 0.18;

settings! {
    #[section("grading")]
    #[check(rising)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneGrading")]
    pub struct Grading {
        pub exposure: f32 = 1.0 => at_least(0.0),
        pub saturation: f32 = 1.05 => at_least(0.0),
        pub contrast: f32 = 1.753 => above(0.0),
        pub shoulder: f32 = 0.983 => above_to(0.0, 1.0),
        pub mid: f32 = 0.264 => above_to(0.0, 1.0),
        pub white: f32 = 9.0 => above(GREY),
        pub vibrance: f32 = 0.15 => at_least(0.0),
        pub shadow_tint: [f32; 3] = [0.95, 0.98, 1.06] => at_least(0.0),
        pub highlight_tint: [f32; 3] = [1.04, 1.0, 0.96] => at_least(0.0),
        pub split: [f32; 2] = [0.2, 0.8] => range(0.0, 1.0),
        pub per_channel: f32 = 0.5 => range(0.0, 1.0),
        pub fade_to_white: f32 = 0.0 => range(0.0, 8.0),
        pub white_balance: f32 = 0.0 => range(0.0, 1.0),
    }
}

pub(crate) fn publish_display(grading: Res<Grading>, mut display: ResMut<Display>) {
    display.set_if_neq(display_of(&grading));
}

fn solved(grading: &Grading) -> [f32; 2] {
    let bend = grading.contrast * grading.shoulder;
    let white = [
        engine_math::pow(grading.white, grading.contrast),
        engine_math::pow(grading.white, bend),
    ];
    let grey = [
        engine_math::pow(GREY, grading.contrast),
        engine_math::pow(GREY, bend),
    ];
    let span = (white[1] - grey[1]) * grading.mid;
    [
        (white[0] * grading.mid - grey[0]) / span,
        (white[1] * grey[0] - white[0] * grey[1] * grading.mid) / span,
    ]
}

fn rising(grading: &Grading) -> Result<(), InvalidSetting> {
    let [upper, lower] = solved(grading);
    if upper.is_finite() && lower.is_finite() && lower > 0.0 {
        return Ok(());
    }
    let most = engine_math::pow(
        grading.white / GREY,
        grading.contrast * (grading.shoulder - 1.0),
    );
    Err(InvalidSetting::broken("mid", Rule::range(0.0, f64::from(most))).got(grading.mid))
}

fn unit_luminance(tint: [f32; 3], edge: f32) -> [f32; 4] {
    let tint = Vec3::from_array(tint);
    let luminance = tint.dot(Vec3::new(0.2126, 0.7152, 0.0722));
    if luminance > 1.0e-6 {
        (tint / luminance).extend(edge).to_array()
    } else {
        [1.0, 1.0, 1.0, edge]
    }
}

fn display_of(grading: &Grading) -> Display {
    let [upper, lower] = solved(grading);
    Display {
        grade: [
            grading.vibrance,
            grading.saturation,
            grading.fade_to_white,
            grading.per_channel,
        ],
        shadow_tint: unit_luminance(grading.shadow_tint, grading.split[0]),
        highlight_tint: unit_luminance(grading.highlight_tint, grading.split[1]),
        curve: [
            grading.contrast,
            grading.contrast * grading.shoulder,
            upper,
            lower,
        ],
        exposure: [grading.exposure, grading.white_balance, grading.white],
        pad: 0.0,
    }
}
