use engine_core::settings;

settings! {
    #[section("sky")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSky")]
    pub struct Sky {
        pub enabled: bool = true => any,
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
        pub ambient: f32 = 1.0 => at_least(0.0),
    }
}

settings! {
    #[section("haze")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneHaze")]
    pub struct Haze {
        pub enabled: bool = false => any,
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
        pub resolution: u32 = 2048 => multiple(64, 256, 4096),
        pub slices: u32 = 64 => range(1, 64),
        pub in_view: bool = true => any,
    }
}

pub(crate) fn haze_runs(haze: &Haze) -> bool {
    haze.enabled && haze.reach > 0.0 && haze.strength > 0.0
}
