use engine_core::settings;

settings! {
    #[section("sky")]
    #[switch(on)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSky")]
    pub struct Sky {
        pub planet_radius: f32 = 6360.0 => range(100.0, 100_000.0),
        pub atmosphere_height: f32 = 100.0 => range(1.0, 1000.0),
        pub rayleigh: [f32; 3] = [5.802e-3, 13.558e-3, 33.1e-3] => range(0.0, 1.0),
        pub rayleigh_height: f32 = 8.0 => range(0.1, 100.0),
        pub mie: f32 = 3.996e-3 => range(0.0, 1.0),
        pub mie_absorption: f32 = 0.444e-3 => range(0.0, 1.0),
        pub mie_height: f32 = 1.2 => range(0.1, 100.0),
        pub mie_g: f32 = 0.8 => range(0.0, 0.99),
        pub ozone: [f32; 3] = [0.65e-3, 1.881e-3, 0.085e-3] => range(0.0, 1.0),
        pub ground_albedo: [f32; 3] = [0.3, 0.3, 0.3] => range(0.0, 1.0),
        pub sun_intensity: f32 = 3.0 => range(0.0, 1000.0),
        pub sun_size: f32 = 0.27 => range(0.01, 10.0),
        pub altitude: f32 = 0.0 => range(0.0, 100_000.0),
        pub ambient: f32 = 1.0 => at_least(0.0),
        pub light_taps: [u32; 2] = [16, 6] => range(1, 64),
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
        pub star_density: f32 = 1.0 => range(0.0, 2.0),
        pub moon_size: f32 = 1.0 => range(0.25, 4.0),
    }
}

settings! {
    #[section("clouds")]
    #[switch(off)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneClouds")]
    pub struct Clouds {
        pub coverage: f32 = 0.45 => range(0.0, 1.0),
        pub base: f32 = 1800.0 => range(-10_000.0, 30_000.0),
        pub thickness: f32 = 1400.0 => range(50.0, 10_000.0),
        pub density: f32 = 1.0 => range(0.01, 4.0),
        pub wind: [f32; 2] = [8.0, 3.0] => range(-100.0, 100.0),
        pub scale: f32 = 3000.0 => range(200.0, 50_000.0),
        pub seed: u32 = 1 => any,
        pub resolution: u32 = 2048 => multiple(64, 256, 4096),
        pub slices: u32 = 64 => range(1, 64),
        pub in_view: bool = true => any,
        pub fewest_steps: u32 = 32 => range(1, 512),
        pub most_steps: u32 = 96 => range(1, 512),
        pub step_limit: u32 = 128 => range(1, 1024),
        pub step: f32 = 100.0 => range(1.0, 10_000.0),
        pub nearest_step: f32 = 25.0 => range(0.1, 10_000.0),
        pub step_share: f32 = 0.1 => range(0.001, 1.0),
        pub empty_stride: f32 = 2.5 => range(1.0, 16.0),
        pub step_blur: f32 = 0.25 => range(0.0, 8.0),
        pub sun_step: f32 = 60.0 => range(1.0, 10_000.0),
        pub sun_step_growth: f32 = 1.8 => range(1.01, 8.0),
        pub weather: u32 = 43 => range(1, 1000),
        pub groups: u32 = 10 => range(1, 1000),
        pub banks: u32 = 3 => range(1, 1000),
        pub bank_cover: [f32; 2] = [0.85, 1.7] => range(0.0, 8.0),
        pub bank_noise: [f32; 2] = [0.61, 0.72] => range(0.0, 1.0),
        pub group_cover: [f32; 2] = [0.85, 1.15] => range(0.0, 8.0),
        pub group_noise: [f32; 2] = [0.6, 0.74] => range(0.0, 1.0),
        pub cover_noise: [f32; 2] = [0.95, 0.4] => range(0.0, 1.0),
        pub base_rise: f32 = 0.1 => range(0.001, 1.0),
        pub top: [f32; 2] = [0.35, 0.75] => range(0.0, 0.99),
        pub warp: f32 = 1.5 => range(0.0, 16.0),
        pub detail: f32 = 0.13 => range(0.001, 8.0),
        pub erosion: f32 = 0.4 => range(0.0, 1.0),
        pub wisp_height: f32 = 0.33 => range(0.001, 1.0),
        pub forward: f32 = 0.8 => range(-0.99, 0.99),
        pub backward: f32 = -0.3 => range(-0.99, 0.99),
        pub backward_share: f32 = 0.3 => range(0.0, 1.0),
        pub powder: f32 = 0.35 => range(0.0, 1.0),
        pub powder_depth: f32 = 120.0 => range(0.0, 10_000.0),
        pub sky_light: [f32; 2] = [0.3, 1.2] => range(0.0, 8.0),
        pub sky_gray: f32 = 0.6 => range(0.0, 1.0),
        pub far_shade: f32 = 0.75 => range(0.0, 100.0),
        pub fade: f32 = 30_000.0 => range(1.0, 10_000_000.0),
        pub high: f32 = 0.35 => range(0.0, 1.0),
        pub high_base: f32 = 9000.0 => range(-10_000.0, 30_000.0),
        pub high_opacity: f32 = 1.5 => range(0.0, 8.0),
        pub high_sheets: u32 = 3 => range(1, 1000),
        pub high_noise: [f32; 2] = [0.8, 0.6] => range(0.0, 1.0),
        pub high_soft: f32 = 0.06 => range(0.001, 1.0),
        pub high_fibre: f32 = 0.8 => range(0.001, 100.0),
        pub high_stretch: f32 = 8.0 => range(1.0, 100.0),
        pub high_turn: f32 = 19.0 => range(-360.0, 360.0),
        pub high_wisps: [f32; 2] = [0.3, 0.62] => range(0.0, 1.0),
    }
}
