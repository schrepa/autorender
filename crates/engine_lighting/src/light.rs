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
    }
}
