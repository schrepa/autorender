use std::sync::OnceLock;

use engine::prelude::*;
use engine::{Error, ErrorCode, Expected};
use serde_json::{Value, json};

use crate::land::{self, RockMaps};
use crate::options::Options;

pub(crate) const SCENE: &str = include_str!("../highlands.scene.json");

const SPEED: f32 = 160.0;

const EFFECTS: [&str; 7] = [
    "shadows",
    "occlusion",
    "bloom",
    "lights",
    "reflections",
    "particles",
    "probes",
];

pub(crate) struct Furnished {
    pub(crate) instance: SceneInstance,
    pub(crate) start: Camera,
}

pub(crate) fn set(world: &mut World, key: &str, value: Value) -> engine::Result<()> {
    engine::host::apply_setting(world, key, &value)
}

pub(crate) fn furnish(
    world: &mut World,
    options: &Options,
    rocks: &RockMaps,
    flying: bool,
) -> engine::Result<Furnished> {
    let document: SceneDocument = engine::scene::parse_json(SCENE)?;
    land::provide(world, rocks)?;
    let instance = document.instantiate(world)?;
    land::populate(world, &document, options.bodies)?;
    if options.bodies > 0 {
        set(world, "collision", collision_budgets())?;
    } else {
        set(world, "collision.enabled", json!(false))?;
        set(world, "physics.enabled", json!(false))?;
    }
    if flying {
        set(world, "fly_camera.speed", json!(SPEED))?;
    } else {
        set(world, "fly_camera.enabled", json!(false))?;
    }
    let start = *world.resource::<Camera>();
    Ok(Furnished { instance, start })
}

fn collision_budgets() -> Value {
    json!({
        "max_colliders": 10_000,
        "max_candidate_pairs": 100_000,
        "max_sensor_pairs": 16_384,
        "max_query_candidates": 10_000,
        "max_query_hits": 1024,
    })
}

fn scene_reflections() -> Option<&'static Value> {
    static SECTION: OnceLock<Option<Value>> = OnceLock::new();
    SECTION
        .get_or_init(|| {
            engine::bricks::parse_json::<Value>(SCENE)
                .ok()
                .and_then(|mut scene| Some(scene.get_mut("environment")?.get_mut("reflections")?.take()))
        })
        .as_ref()
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
    set(world, "shadows.enabled", json!(shadows))?;
    if occlusion {
        set(world, "ambient_occlusion", json!({}))?;
    } else {
        set(world, "ambient_occlusion.intensity", json!(0.0))?;
    }
    if bloom {
        set(world, "bloom", json!({}))?;
    } else {
        set(world, "bloom.intensity", json!(0.0))?;
    }
    match scene_reflections().filter(|_| reflections) {
        Some(section) => set(world, "reflections", section.clone())?,
        None => set(world, "reflections.enabled", json!(false))?,
    }
    set(world, "local_lights.enabled", json!(lights))?;
    set(world, "particles.enabled", json!(particles))?;
    if probes {
        set(world, "reflection_probes", json!({}))
    } else {
        set(world, "reflection_probes.enabled", json!(false))
    }
}

pub(crate) fn stand(world: &mut World, place: &str) -> engine::Result<()> {
    let malformed = |expected| {
        Error::new(ErrorCode::InvalidValue, "$.stand")
            .expected(expected)
            .got(place)
    };
    let numbers: Vec<f32> = place
        .split(',')
        .map(|number| number.trim().parse::<f32>())
        .collect::<Result<_, _>>()
        .map_err(|_| malformed(Expected::Number))?;
    let [x, z, yaw, pitch] = numbers[..] else {
        return Err(malformed(Expected::Array));
    };
    if numbers.iter().any(|number| !number.is_finite()) {
        return Err(malformed(Expected::Finite));
    }
    if pitch.abs() >= 90.0 {
        return Err(Error::new(ErrorCode::OutOfRange, "$.stand")
            .range(-90.0, 90.0)
            .got(pitch));
    }
    set(world, "highlands.stand", json!([x, z, yaw, pitch]))
}
