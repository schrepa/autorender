use engine_core::{settings, stats};

settings! {
    #[section("bloom")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneBloom")]
    pub struct Bloom {
        pub intensity: f32 = 0.6 => range(0.0, 4.0),
        pub threshold: f32 = 2.0 => at_least(0.0),
    }
}

stats! {
    #[section("bloom_stats")]
    pub struct BloomStats {
        pub active: bool,
    }
}

impl Bloom {
    pub fn enabled(&self) -> bool {
        self.intensity > 0.0
    }
}
