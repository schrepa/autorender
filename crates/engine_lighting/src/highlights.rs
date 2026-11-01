use engine_core::settings;

settings! {
    #[section("highlights")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneHighlights")]
    pub struct Highlights {
        pub enabled: bool = true => any,
        pub capped: bool = true => any,
        pub sun_size: f32 = 0.27 => range(0.0, 10.0),
        pub cap: f32 = 4.0 => at_least(0.0),
        pub light_radius: f32 = 0.1 => range(0.0, 100.0),
    }
}
