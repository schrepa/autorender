use engine_core::settings;

settings! {
    #[section("reflections")]
    #[switch(off)]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "SceneReflections")]
    pub struct Reflections {
        pub intensity: f32 = 1.0 => range(0.0, 1.0),
        pub max_roughness: f32 = 0.75 => range(0.05, 1.0),
        pub roughness_fade: f32 = 0.2 => above_to(0.0, 1.0),
        pub distance: f32 = 120.0 => range(0.001, 100_000.0),
        pub taps: u32 = 8 => range(1, 8),
        pub behind_steps: u32 = 16 => range(0, 96),
        pub far_cell: u32 = 8 => range(1, 16),
        pub far_refine: f32 = 1.0 => range(0.0, 16.0),
        pub far_edge: f32 = 4.0 => range(0.0, 64.0),
        pub far_contrast: f32 = 0.25 => range(0.0, 1.0),
    }
}
