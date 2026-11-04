use engine_core::{settings, stats};

settings! {
    #[section("ambient_occlusion")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneAmbientOcclusion")]
    pub struct AmbientOcclusion {
        pub enabled: bool = true => any,
        pub intensity: f32 = 0.8 => range(0.0, 1.0),
        pub radius: f32 = 2.5 => above(0.0),
    }
}

impl AmbientOcclusion {
    pub fn occludes(&self) -> bool {
        self.enabled && self.intensity > 0.0
    }
}

stats! {
    #[section("ambient_occlusion_stats")]
    pub struct AmbientOcclusionStats {
        pub active: bool,
    }
}
