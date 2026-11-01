use engine_core::{settings, stats};

const GRAVITY: f32 = 9.81;

settings! {
    #[section("physics")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "ScenePhysics")]
    pub struct Physics {
        pub enabled: bool = true => any,
        pub gravity: [f32; 3] = [0.0, -GRAVITY, 0.0] => range(-100.0, 100.0),
        pub water_density: f32 = 1000.0 => range(0.0, 20_000.0),
        pub air_density: f32 = 1.225 => range(0.0, 100.0),
        pub iterations: u32 = 4 => range(1, 16),
    }
}

stats! {
    #[section("physics_stats")]
    #[derive(Copy, Eq)]
    pub struct PhysicsStats {
        pub bodies: u32,
        pub awake: u32,
        pub held: u32,
        pub wet: u32,
        pub solids: u32,
        pub contacts: u32,
        pub water_samples: u32,
        pub invalid: u32,
    }
}
