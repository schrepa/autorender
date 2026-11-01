use engine_core::{settings, stats};

settings! {
    #[section("temporal_antialiasing")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneTemporalAntialiasing")]
    pub struct TemporalAntialiasing {
        pub enabled: bool = false => any,
        pub nearest_motion: bool = true => any,
        pub rounded_clamp: bool = false => any,
        pub particle_response: f32 = 2.0 => range(0.0, 4.0),
    }
}

stats! {
    #[section("temporal_antialiasing_stats")]
    pub struct TemporalAntialiasingStats {
        pub active: bool,
    }
}
