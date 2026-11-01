use engine_core::settings;
use engine_places::lighting::LightPreset;
use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct GradingValues {
    pub exposure: f32,
    pub saturation: f32,
    pub contrast: f32,
    pub vibrance: f32,
    pub shadow_tint: Vec3,
    pub highlight_tint: Vec3,
}

fn grading_for(preset: LightPreset) -> GradingValues {
    match preset {
        LightPreset::Day => GradingValues {
            exposure: 0.85,
            saturation: 1.2,
            contrast: 1.12,
            vibrance: 0.35,
            shadow_tint: Vec3::new(0.85, 0.95, 1.2),
            highlight_tint: Vec3::new(1.08, 1.02, 0.92),
        },
        LightPreset::GoldenHour => GradingValues {
            exposure: 0.75,
            saturation: 1.2,
            contrast: 1.15,
            vibrance: 0.4,
            shadow_tint: Vec3::new(0.75, 0.9, 1.35),
            highlight_tint: Vec3::new(1.15, 1.0, 0.85),
        },
        LightPreset::Dusk => GradingValues {
            exposure: 0.95,
            saturation: 1.15,
            contrast: 1.18,
            vibrance: 0.35,
            shadow_tint: Vec3::new(0.75, 0.8, 1.45),
            highlight_tint: Vec3::new(1.25, 0.95, 0.75),
        },
        LightPreset::Night => GradingValues {
            exposure: 1.35,
            saturation: 1.2,
            contrast: 1.2,
            vibrance: 0.5,
            shadow_tint: Vec3::new(0.8, 0.9, 1.35),
            highlight_tint: Vec3::new(1.15, 1.0, 0.85),
        },
    }
}

fn vec3_or(authored: Option<[f32; 3]>, preset: Vec3) -> Vec3 {
    authored.map_or(preset, Vec3::from_array)
}

settings! {
    #[section("grading")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneGrading")]
    pub struct Grading {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        pub exposure: Option<f32> = None => at_least(0.0),
        pub saturation: Option<f32> = None => at_least(0.0),
        pub contrast: Option<f32> = None => at_least(0.0),
        pub vibrance: Option<f32> = None => at_least(0.0),
        pub shadow_tint: Option<[f32; 3]> = None => at_least(0.0),
        pub highlight_tint: Option<[f32; 3]> = None => at_least(0.0),
    }
}

impl Grading {
    pub(crate) fn values(&self) -> GradingValues {
        let preset = grading_for(self.preset);
        GradingValues {
            exposure: self.exposure.unwrap_or(preset.exposure),
            saturation: self.saturation.unwrap_or(preset.saturation),
            contrast: self.contrast.unwrap_or(preset.contrast),
            vibrance: self.vibrance.unwrap_or(preset.vibrance),
            shadow_tint: vec3_or(self.shadow_tint, preset.shadow_tint),
            highlight_tint: vec3_or(self.highlight_tint, preset.highlight_tint),
        }
    }
}
