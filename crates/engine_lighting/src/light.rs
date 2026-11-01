use engine_core::settings;
use engine_places::lighting::LightPreset;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct SurfaceLightValues {
    pub rim: f32,
    pub specular: f32,
    pub emission: f32,
}

fn surface_light_for(preset: LightPreset) -> SurfaceLightValues {
    let (rim, emission) = match preset {
        LightPreset::Day => (0.15, 0.1),
        LightPreset::GoldenHour => (0.45, 0.5),
        LightPreset::Dusk => (0.6, 1.0),
        LightPreset::Night => (0.8, 1.0),
    };
    SurfaceLightValues {
        rim,
        specular: 1.0,
        emission,
    }
}

settings! {
    #[section("surface_light")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSurfaceLight")]
    pub struct SurfaceLight {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        pub rim: Option<f32> = None => at_least(0.0),
        pub specular: Option<f32> = None => at_least(0.0),
        pub emission: Option<f32> = None => at_least(0.0),
    }
}

impl SurfaceLight {
    pub(crate) fn values(&self) -> SurfaceLightValues {
        let preset = surface_light_for(self.preset);
        SurfaceLightValues {
            rim: self.rim.unwrap_or(preset.rim),
            specular: self.specular.unwrap_or(preset.specular),
            emission: self.emission.unwrap_or(preset.emission),
        }
    }
}
