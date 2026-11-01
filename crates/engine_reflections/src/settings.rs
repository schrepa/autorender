use engine_core::{settings, stats};

settings! {
    #[section("reflections")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneReflections")]
    pub struct Reflections {
        pub enabled: bool = false => any,
        pub intensity: f32 = 1.0 => range(0.0, 1.0),
        pub max_roughness: f32 = 0.75 => range(0.05, 1.0),
        pub distance: f32 = 120.0 => range(0.001, 100_000.0),
    }
}

stats! {
    #[section("reflections_stats")]
    pub struct ReflectionsStats {
        pub active: bool,
    }
}

impl Reflections {
    pub fn reflects(&self) -> bool {
        self.enabled && self.intensity > 0.0
    }
}
