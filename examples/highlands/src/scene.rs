use engine::prelude::*;
use engine::{Error, ErrorCode};

use crate::land::{self, RockMaps};
use crate::views;

const SCENE: &str = include_str!("../highlands.scene.json");

const EFFECTS: [&str; 7] = [
    "shadows",
    "occlusion",
    "bloom",
    "lights",
    "reflections",
    "particles",
    "probes",
];

#[derive(Resource, Clone, Copy)]
struct SceneReflections(Option<Reflections>);

pub(crate) fn create(bodies: u32, rocks: &RockMaps) -> engine::Result<(Engine, SceneInstance)> {
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
    let instance = furnish(&mut engine.app, bodies, rocks, true)?;
    Ok((engine, instance))
}

pub(crate) fn furnish(
    app: &mut App,
    bodies: u32,
    rocks: &RockMaps,
    flying: bool,
) -> engine::Result<SceneInstance> {
    let document: SceneDocument = engine::scene::parse_json(SCENE)?;
    let world = app.world_mut();
    land::provide(world, rocks)?;
    let instance = document.instantiate(world)?;
    land::populate(world, &document, bodies)?;
    let reflections = world.get_resource::<Reflections>().copied();
    world.insert_resource(SceneReflections(reflections));
    if bodies > 0 {
        refresh_collision_world(world)?;
    }
    views::start(world);
    if flying {
        app.add_plugins(FlyCameraPlugin);
    }
    views::add_systems(app);
    land::add_systems(app);
    Ok(instance)
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

fn exclusions() -> impl Iterator<Item = String> {
    EFFECTS.iter().map(|effect| format!("no-{effect}"))
}

pub(crate) fn effects() -> String {
    format!("all|none|{}", exclusions().collect::<Vec<_>>().join(","))
}

pub(crate) fn set_effects(world: &mut World, name: &str) -> engine::Result<()> {
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
    let scene = world
        .get_resource::<SceneReflections>()
        .and_then(|scene| scene.0);
    if let Some(settings) = scene.filter(|_| reflections) {
        world.insert_resource(settings);
    } else {
        world.remove_resource::<Reflections>();
    }
    world.resource_mut::<LocalLightSettings>().enabled = lights;
    world.resource_mut::<ParticleSettings>().enabled = particles;
    if probes {
        world.insert_resource(ReflectionProbeSettings::default());
    } else {
        world.remove_resource::<ReflectionProbeSettings>();
    }
    Ok(())
}
