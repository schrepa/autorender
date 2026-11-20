use engine::prelude::settings;

settings! {
    #[section("volume")]
    #[serde(deny_unknown_fields)]
    pub struct Volume {
        pub master: f32 = 1.0 => range(0.0, 1.0),
        pub effects: f32 = 1.0 => range(0.0, 1.0),
        pub interface: f32 = 1.0 => range(0.0, 1.0),
        pub footsteps: f32 = 1.0 => range(0.0, 1.0),
    }
}
