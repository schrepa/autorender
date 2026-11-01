use engine_core::settings;
use engine_places::lighting::LightPreset;
use glam::Vec3;

settings! {
    #[section("sky")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSky")]
    pub struct Sky {
        pub enabled: bool = false => any,
        pub planet_radius: f32 = 6360.0 => range(100.0, 100_000.0),
        pub atmosphere_height: f32 = 100.0 => range(1.0, 1000.0),
        pub rayleigh: [f32; 3] = [5.802e-3, 13.558e-3, 33.1e-3] => range(0.0, 1.0),
        pub rayleigh_height: f32 = 8.0 => range(0.1, 100.0),
        pub mie: f32 = 3.996e-3 => range(0.0, 1.0),
        pub mie_absorption: f32 = 4.4e-3 => range(0.0, 1.0),
        pub mie_height: f32 = 1.2 => range(0.1, 100.0),
        pub mie_g: f32 = 0.8 => range(0.0, 0.99),
        pub ozone: [f32; 3] = [0.65e-3, 1.881e-3, 0.085e-3] => range(0.0, 1.0),
        pub ground_albedo: [f32; 3] = [0.3, 0.3, 0.3] => range(0.0, 1.0),
        pub sun_intensity: f32 = 3.0 => range(0.0, 1000.0),
        pub sun_size: f32 = 0.27 => range(0.01, 10.0),
        pub aerial: f32 = 1.0 => range(0.0, 100.0),
        pub altitude: f32 = 0.0 => range(0.0, 100_000.0),
        pub adaptation: f32 = 0.5 => range(0.0, 1.0),
    }
}

settings! {
    #[section("haze")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneHaze")]
    pub struct Haze {
        pub enabled: bool = true => any,
        pub visibility: f32 = 0.0 => at_least(0.0),
        pub strength: f32 = 0.7 => range(0.0, 4.0),
        pub forward: f32 = 0.6 => range(0.0, 0.95),
        pub reach: f32 = 300.0 => range(0.0, 100_000.0),
        pub spread: f32 = 0.6 => range(0.0, 1.0),
        pub size: f32 = 60.0 => range(0.1, 100_000.0),
    }
}

settings! {
    #[section("night_sky")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneNightSky")]
    pub struct NightSky {
        pub stars: f32 = 1.0 => range(0.0, 8.0),
        pub moon: f32 = 1.0 => range(0.0, 8.0),
        pub cirrus: f32 = 0.35 => range(0.0, 1.0),
        pub star_density: f32 = 1.0 => range(0.0, 2.0),
        pub moon_size: f32 = 1.0 => range(0.25, 4.0),
        pub cirrus_height: f32 = 9000.0 => range(500.0, 50_000.0),
        pub cirrus_drift: [f32; 2] = [6.0, 1.5] => range(-1000.0, 1000.0),
    }
}

settings! {
    #[section("clouds")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneClouds")]
    pub struct Clouds {
        pub enabled: bool = false => any,
        pub coverage: f32 = 0.45 => range(0.0, 1.0),
        pub base: f32 = 1800.0 => range(-10_000.0, 30_000.0),
        pub thickness: f32 = 1400.0 => range(50.0, 10_000.0),
        pub density: f32 = 1.0 => range(0.01, 4.0),
        pub wind: [f32; 2] = [8.0, 3.0] => range(-100.0, 100.0),
        pub scale: f32 = 3000.0 => range(200.0, 50_000.0),
        pub seed: u32 = 1 => any,
        pub shadows: f32 = 0.75 => range(0.0, 1.0),
        pub resolution: u32 = 2048 => range(256, 4096),
        pub slices: u32 = 64 => range(1, 64),
        pub in_view: bool = true => any,
    }
}

settings! {
    #[section("sky_light")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSkyLight")]
    pub struct SkyLighting {
        pub preset: LightPreset = LightPreset::GoldenHour => any,
        pub zenith: Option<[f32; 3]> = None => at_least(0.0),
        pub horizon: Option<[f32; 3]> = None => at_least(0.0),
        pub ground: Option<[f32; 3]> = None => at_least(0.0),
        pub ambient: Option<f32> = None => at_least(0.0),
        pub fog_start: Option<f32> = None => at_least(0.0),
        pub fog_density: Option<f32> = None => at_least(0.0),
        pub fog_falloff: Option<f32> = None => at_least(0.0),
        pub fog_max: Option<f32> = None => range(0.0, 1.0),
        pub aerial_start: Option<f32> = None => at_least(0.0),
        pub aerial_density: Option<f32> = None => at_least(0.0),
        pub aerial_falloff: Option<f32> = None => at_least(0.0),
        pub aerial_color: Option<[f32; 3]> = None => at_least(0.0),
        pub aerial_glow: Option<f32> = None => at_least(0.0),
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SkyLightingValues {
    pub zenith: Vec3,
    pub horizon: Vec3,
    pub ground: Vec3,
    pub ambient: f32,
    pub fog_start: f32,
    pub fog_density: f32,
    pub fog_falloff: f32,
    pub fog_max: f32,
    pub aerial_start: f32,
    pub aerial_density: f32,
    pub aerial_falloff: f32,
    pub aerial_color: Vec3,
    pub aerial_glow: f32,
}

pub(crate) fn sky_light_for(preset: LightPreset) -> SkyLightingValues {
    match preset {
        LightPreset::Day => SkyLightingValues {
            zenith: Vec3::new(0.1, 0.3, 0.9),
            horizon: Vec3::new(0.55, 0.75, 1.0),
            ground: Vec3::new(0.24, 0.21, 0.18),
            ambient: 0.6,
            fog_start: 80.0,
            fog_density: 0.0011,
            fog_falloff: 0.02,
            fog_max: 0.4,
            aerial_start: 1000.0,
            aerial_density: 1.0e-4,
            aerial_falloff: 5.5e-4,
            aerial_color: Vec3::new(0.56, 0.72, 1.0),
            aerial_glow: 0.25,
        },
        LightPreset::GoldenHour => SkyLightingValues {
            zenith: Vec3::new(0.05, 0.16, 0.62),
            horizon: Vec3::new(1.15, 0.56, 0.32),
            ground: Vec3::new(0.2, 0.13, 0.12),
            ambient: 0.62,
            fog_start: 80.0,
            fog_density: 0.0014,
            fog_falloff: 0.022,
            fog_max: 0.45,
            aerial_start: 1000.0,
            aerial_density: 1.1e-4,
            aerial_falloff: 6.0e-4,
            aerial_color: Vec3::new(0.36, 0.48, 0.92),
            aerial_glow: 0.7,
        },
        LightPreset::Dusk => SkyLightingValues {
            zenith: Vec3::new(0.03, 0.04, 0.32),
            horizon: Vec3::new(1.0, 0.36, 0.42),
            ground: Vec3::new(0.08, 0.05, 0.09),
            ambient: 0.65,
            fog_start: 60.0,
            fog_density: 0.0016,
            fog_falloff: 0.02,
            fog_max: 0.45,
            aerial_start: 800.0,
            aerial_density: 1.4e-4,
            aerial_falloff: 6.0e-4,
            aerial_color: Vec3::new(0.3, 0.3, 0.66),
            aerial_glow: 0.9,
        },
        LightPreset::Night => SkyLightingValues {
            zenith: Vec3::new(0.008, 0.012, 0.045),
            horizon: Vec3::new(0.07, 0.09, 0.22),
            ground: Vec3::new(0.02, 0.02, 0.04),
            ambient: 0.9,
            fog_start: 40.0,
            fog_density: 0.0022,
            fog_falloff: 0.02,
            fog_max: 0.7,
            aerial_start: 600.0,
            aerial_density: 1.6e-4,
            aerial_falloff: 6.0e-4,
            aerial_color: Vec3::new(0.05, 0.07, 0.16),
            aerial_glow: 0.15,
        },
    }
}

fn vec3_or(authored: Option<[f32; 3]>, preset: Vec3) -> Vec3 {
    authored.map_or(preset, Vec3::from_array)
}

impl SkyLighting {
    pub fn values(&self) -> SkyLightingValues {
        let preset = sky_light_for(self.preset);
        SkyLightingValues {
            zenith: vec3_or(self.zenith, preset.zenith),
            horizon: vec3_or(self.horizon, preset.horizon),
            ground: vec3_or(self.ground, preset.ground),
            ambient: self.ambient.unwrap_or(preset.ambient),
            fog_start: self.fog_start.unwrap_or(preset.fog_start),
            fog_density: self.fog_density.unwrap_or(preset.fog_density),
            fog_falloff: self.fog_falloff.unwrap_or(preset.fog_falloff),
            fog_max: self.fog_max.unwrap_or(preset.fog_max),
            aerial_start: self.aerial_start.unwrap_or(preset.aerial_start),
            aerial_density: self.aerial_density.unwrap_or(preset.aerial_density),
            aerial_falloff: self.aerial_falloff.unwrap_or(preset.aerial_falloff),
            aerial_color: vec3_or(self.aerial_color, preset.aerial_color),
            aerial_glow: self.aerial_glow.unwrap_or(preset.aerial_glow),
        }
    }
}

pub(crate) fn haze_runs(haze: &Haze, sky: &Sky, sky_light: &SkyLightingValues) -> bool {
    let fog = !sky.enabled && sky_light.fog_density > 0.0 && sky_light.fog_max > 0.0;
    haze.enabled && haze.reach > 0.0 && (haze.strength > 0.0 || fog)
}
