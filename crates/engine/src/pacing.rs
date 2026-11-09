use bevy_ecs::world::World;
use engine_core::{Engine, settings};

settings! {
    #[section("pacing")]
    #[derive(Copy)]
    #[serde(deny_unknown_fields)]
    #[schemars(rename = "ScenePacing")]
    pub struct Pacing {
        pub margin_ms: f64 = 4.0 => at_least(0.0),
        pub smoothing: f64 = 0.1 => above_to(0.0, 1.0),
        pub interval_limit_ms: f64 = 100.0 => above(0.0),
    }
}

pub(crate) fn register(engine: &mut Engine) {
    engine.settings::<Pacing>();
}

pub(crate) fn of(world: &World) -> Pacing {
    world.get_resource::<Pacing>().copied().unwrap_or_default()
}
