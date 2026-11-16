use engine_core::settings;

settings! {
    #[section("surface_light")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneSurfaceLight")]
    pub struct SurfaceLight {
        pub rim: f32 = 0.2 => at_least(0.0),
        pub specular: f32 = 1.0 => at_least(0.0),
        pub emission: f32 = 0.5 => at_least(0.0),
        pub specular_variance: f32 = 0.15 => at_least(0.0),
        pub specular_limit: f32 = 0.2 => at_least(0.0),
        pub crossing: f32 = 1.0 => range(0.0, 2.0),
        pub crossing_tint: f32 = 0.5 => range(0.0, 2.0),
        pub forward: f32 = 1.0 => at_least(0.0),
        pub forward_bend: f32 = 0.3 => range(0.0, 0.95),
        pub forward_focus: f32 = 4.0 => above(0.0),
    }
}
