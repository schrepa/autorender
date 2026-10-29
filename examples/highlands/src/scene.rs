use engine::prelude::*;
use engine::{Error, ErrorCode};

use crate::field::{self, hamlet_radius};
use crate::land::{self, RockMaps};
use crate::views::{self, set_view};

const SCENE_LOOK: &str = "scene";

const HIGHLANDS_LOOK: &str = "golden";

type Look = (&'static str, fn() -> Atmosphere, f32);

const LOOKS: [Look; 5] = [
    ("golden", Atmosphere::golden_hour, 0.35),
    ("dusk", Atmosphere::dusk, 0.8),
    ("night", Atmosphere::night, 1.0),
    ("day", Atmosphere::day, 0.0),
    ("sky", physical_look, 0.5),
];

const EFFECTS: [&str; 7] = [
    "shadows",
    "occlusion",
    "bloom",
    "lights",
    "reflections",
    "particles",
    "probes",
];

pub(crate) fn create(bodies: u32, rocks: &RockMaps) -> engine::Result<Engine> {
    let mut engine = Engine::default();
    let collision = bodies > 0;
    if collision {
        engine
            .app
            .add_bricks(engine::bricks::standard())
            .insert_resource(collision_budgets());
    } else {
        engine
            .app
            .add_bricks(engine::bricks::standard().without(["collision", "physics"]));
    }
    furnish(&mut engine.app, bodies, rocks, true)?;
    Ok(engine)
}

pub(crate) fn furnish(
    app: &mut App,
    bodies: u32,
    rocks: &RockMaps,
    flying: bool,
) -> engine::Result<()> {
    let (atmosphere, lamps) = look(SCENE_LOOK)?;
    app.insert_resource(atmosphere);
    apply_sky(app.world_mut(), SCENE_LOOK);
    let world = app.world_mut();
    let plateau = land::populate(world, bodies, hamlet_radius(), rocks)?;
    field::field(world, plateau)?;
    environment(world);
    world.insert_resource(reflection_settings());
    world.insert_resource(LocalLightSettings {
        distance: 220.0,
        intensity: lamps,
        ..LocalLightSettings::default()
    });
    if bodies > 0 {
        refresh_collision_world(world)?;
    }
    set_view(world, "overview")?;
    if flying {
        app.add_plugins(FlyCameraPlugin);
    }
    views::add_systems(app);
    land::add_systems(app);
    app.add_systems(FixedSimulation, field::animate);
    Ok(())
}

fn environment(world: &mut World) {
    world.insert_resource(TemporalAntialiasing {
        enabled: true,
        ..TemporalAntialiasing::default()
    });
    world.insert_resource(Wind {
        velocity: (Vec2::new(5.5, 2.0).normalize() * 2.5).to_array(),
        gusts: 0.55,
    });
    world.insert_resource(Clouds {
        coverage: 0.38,
        base: 1500.0,
        thickness: 1500.0,
        wind: [11.0, 4.0],
        shadows: 0.6,
        ..Clouds::default()
    });
    world.insert_resource(AutoExposure::default());
    world.insert_resource(Water {
        color: [0.003, 0.02, 0.024],
        clarity: 4.5,
        sediment: 0.01,
        sediment_color: [0.04, 0.16, 0.15],
        simulation_cell: 1.0,
        ..Water::default()
    });
}

fn reflection_settings() -> Reflections {
    Reflections {
        distance: 4000.0,
        ..Reflections::default()
    }
}

fn collision_budgets() -> engine::collision::CollisionConfig {
    engine::collision::CollisionConfig {
        max_colliders: Some(10_000),
        max_candidate_pairs: Some(100_000),
        max_sensor_pairs: Some(16_384),
        max_query_candidates: Some(10_000),
        max_query_hits: Some(1024),
    }
}

fn look_names() -> impl Iterator<Item = &'static str> {
    std::iter::once(SCENE_LOOK).chain(LOOKS.iter().map(|(name, ..)| *name))
}

pub(crate) fn looks() -> String {
    look_names().collect::<Vec<_>>().join("|")
}

fn exclusions() -> impl Iterator<Item = String> {
    EFFECTS.iter().map(|effect| format!("no-{effect}"))
}

pub(crate) fn effects() -> String {
    format!("all|none|{}", exclusions().collect::<Vec<_>>().join(","))
}

pub(crate) fn set_look(world: &mut World, name: &str) -> engine::Result<()> {
    let (atmosphere, lamps) = look(name)?;
    world.insert_resource(atmosphere);
    apply_sky(world, name);
    if let Some(mut local) = world.get_resource_mut::<LocalLightSettings>() {
        local.intensity = lamps;
    }
    Ok(())
}

pub(crate) fn set_effects(world: &mut World, name: &str, look_name: &str) -> engine::Result<()> {
    let mut on = [true; EFFECTS.len()];
    match name {
        "all" => {}
        "none" => on = [false; EFFECTS.len()],
        _ => {
            for token in name.split(',') {
                let excluded = token
                    .strip_prefix("no-")
                    .and_then(|effect| EFFECTS.iter().position(|name| *name == effect));
                match excluded {
                    Some(index) => on[index] = false,
                    None => {
                        let tokens = ["all", "none"].map(String::from).into_iter();
                        return Err(Error::new(ErrorCode::InvalidValue, "$.effects")
                            .got(token)
                            .variants(tokens.chain(exclusions())));
                    }
                }
            }
        }
    }
    let [
        shadows,
        occlusion,
        bloom,
        lights,
        reflections,
        particles,
        probes,
    ] = on;
    let (_, lamps) = look(look_name)?;
    world.resource_mut::<Shadows>().enabled = shadows;
    world.resource_mut::<AmbientOcclusion>().intensity = if occlusion {
        AmbientOcclusion::default().intensity
    } else {
        0.0
    };
    world.resource_mut::<Bloom>().intensity = if bloom {
        Bloom::default().intensity
    } else {
        0.0
    };
    if reflections {
        world.insert_resource(reflection_settings());
    } else {
        world.remove_resource::<Reflections>();
    }
    {
        let mut local = world.resource_mut::<LocalLightSettings>();
        local.enabled = lights;
        local.intensity = lamps;
    }
    world.resource_mut::<ParticleSettings>().enabled = particles;
    if probes {
        world.insert_resource(ReflectionProbeSettings::default());
    } else {
        world.remove_resource::<ReflectionProbeSettings>();
    }
    Ok(())
}

fn physical_look() -> Atmosphere {
    Atmosphere {
        ambient: 1.0,
        rim: 0.2,
        emission: 0.5,
        exposure: 1.0,
        saturation: 1.05,
        contrast: 1.1,
        vibrance: 0.15,
        shadow_tint: Vec3::new(0.95, 0.98, 1.06),
        highlight_tint: Vec3::new(1.04, 1.0, 0.96),
        ..Atmosphere::day()
    }
}

fn apply_sky(world: &mut World, name: &str) {
    if resolve(name) == "sky" {
        world.insert_resource(Sky::default());
        world.insert_resource(TimeOfDay {
            hours: 16.5,
            latitude: 42.0,
            north: 30.0,
            ..TimeOfDay::default()
        });
    } else {
        world.remove_resource::<Sky>();
        world.remove_resource::<TimeOfDay>();
    }
}

fn look(asked: &str) -> engine::Result<(Atmosphere, f32)> {
    let name = resolve(asked);
    LOOKS
        .iter()
        .find(|(look, ..)| *look == name)
        .map(|&(_, atmosphere, lamps)| (atmosphere(), lamps))
        .ok_or_else(|| {
            Error::new(ErrorCode::InvalidValue, "$.look")
                .got(asked)
                .variants(look_names())
        })
}

fn resolve(name: &str) -> &str {
    if name == SCENE_LOOK {
        HIGHLANDS_LOOK
    } else {
        name
    }
}
